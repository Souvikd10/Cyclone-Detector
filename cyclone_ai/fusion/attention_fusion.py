"""
Attention-Gated Fusion — Stage 3.

Combines per-source feature maps using cross-attention, weighted by
learned reliability scores. Produces a single fused feature map for
the shared encoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for spatial feature maps.

    Adds learnable or sinusoidal position information so the attention
    mechanism is spatially aware.
    """

    def __init__(self, channels: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.channels = channels

        pe = torch.zeros(channels, max_h, max_w)
        half_c = channels // 2

        # Height encoding
        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, half_c, 2).float() * -(math.log(10000.0) / half_c))
        pe[0:half_c:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        pe[1:half_c:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)

        # Width encoding
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()
        pe[half_c::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        pe[half_c + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x [B, C, H, W]."""
        _, _, H, W = x.shape
        return x + self.pe[:, :, :H, :W]


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention module for fusing source feature maps.

    Each source feature map attends to all other sources, weighted by
    reliability scores. The output is a single fused feature map.

    Args:
        embed_dim: Embedding dimension (must match source encoder output channels).
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        source_features: dict[str, torch.Tensor],
        reliability_weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse source feature maps using cross-attention.

        Args:
            source_features: Dict mapping source_name → [B, C, H', W'].
            reliability_weights: Dict mapping source_name → [B, 1].

        Returns:
            Fused feature map [B, C, H', W'].
        """
        source_names = list(source_features.keys())
        B, C, H, W = next(iter(source_features.values())).shape

        # Reshape to sequence: [B, H*W, C] for each source
        source_seqs = {}
        for name in source_names:
            feat = source_features[name]  # [B, C, H, W]
            seq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]
            # Scale by reliability weight
            weight = reliability_weights[name]  # [B, 1]
            seq = seq * weight.unsqueeze(-1)  # [B, HW, C]
            source_seqs[name] = seq

        # Concatenate all sources: [B, N*HW, C]
        all_sources = torch.cat(list(source_seqs.values()), dim=1)

        # Use mean of all sources as query (the "fused" representation seeks info)
        query = torch.stack(list(source_seqs.values()), dim=1).mean(dim=1)  # [B, HW, C]

        # Multi-head attention
        Q = self.q_proj(query)   # [B, HW, C]
        K = self.k_proj(all_sources)  # [B, N*HW, C]
        V = self.v_proj(all_sources)  # [B, N*HW, C]

        # Reshape for multi-head
        head_dim = C // self.num_heads
        Q = Q.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)    # [B, heads, HW, d]
        K = K.view(B, -1, self.num_heads, head_dim).transpose(1, 2)        # [B, heads, N*HW, d]
        V = V.view(B, -1, self.num_heads, head_dim).transpose(1, 2)        # [B, heads, N*HW, d]

        # Scaled dot-product attention
        scale = math.sqrt(head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, HW, N*HW]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # [B, heads, HW, d]
        out = out.transpose(1, 2).reshape(B, H * W, C)  # [B, HW, C]
        out = self.out_proj(out)
        out = self.norm(out + query)  # Residual + LayerNorm

        # Reshape back to spatial
        fused = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return fused


class AttentionGatedFusion(nn.Module):
    """
    Complete attention-gated fusion module (Stage 3).

    Combines per-source encoders, reliability estimation, and cross-attention
    fusion into a single module.

    Input:  [B, C_sources, H, W] multi-channel input (all sources stacked)
    Output: [B, output_dim, H', W'] fused feature map

    Args:
        source_names: List of data source names.
        source_channels: Output channels from source encoders.
        output_dim: Output feature map channels.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        use_positional_encoding: Whether to add 2D positional encoding.
    """

    def __init__(
        self,
        source_names: list[str],
        source_channels: int = 64,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.source_names = source_names
        self.source_channels = source_channels
        self.output_dim = output_dim

        # Per-source encoders
        from cyclone_ai.fusion.source_encoder import MultiSourceEncoder
        self.source_encoders = MultiSourceEncoder(
            source_names=source_names,
            output_channels=source_channels,
        )

        # Reliability network
        from cyclone_ai.fusion.reliability import ReliabilityNetwork
        self.reliability = ReliabilityNetwork(
            feature_dim=source_channels,
            num_sources=len(source_names),
        )

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            embed_dim=source_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Optional positional encoding
        self.use_pos_enc = use_positional_encoding
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding2D(source_channels)

        # Project fused features to output dimension
        self.projection = nn.Sequential(
            nn.Conv2d(source_channels, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full fusion pipeline.

        Args:
            x: [B, C, H, W] multi-channel input where C = len(source_names).
            metadata: [B, num_sources, metadata_dim] per-source metadata (optional).

        Returns:
            Fused feature map [B, output_dim, H/16, W/16].
        """
        # Step 1: Encode each source independently
        source_features = self.source_encoders(x)  # dict: name → [B, 64, H/16, W/16]

        # Step 2: Add positional encoding
        if self.use_pos_enc:
            source_features = {
                name: self.pos_enc(feat)
                for name, feat in source_features.items()
            }

        # Step 3: Estimate reliability weights
        reliability_weights = self.reliability(source_features, metadata)

        # Step 4: Cross-attention fusion
        fused = self.cross_attention(source_features, reliability_weights)

        # Step 5: Project to output dimension
        output = self.projection(fused)

        return output

    def get_reliability_scores(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Get reliability scores for inspection/logging."""
        with torch.no_grad():
            source_features = self.source_encoders(x)
            return self.reliability.get_reliability_summary(source_features, metadata)
