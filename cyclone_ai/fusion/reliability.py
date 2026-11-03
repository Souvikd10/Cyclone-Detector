"""
Learned Reliability Weights — Stage 3.

A small network that estimates per-source reliability scores based on
the source features and metadata (time-of-day, data quality flags, etc.).
Sources with missing data or known degradation are automatically downweighted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityNetwork(nn.Module):
    """
    Learns per-source reliability weights from source features + metadata.

    For each source, the network takes:
    - Global average pooled features from the source encoder
    - Metadata vector (time-of-day sin/cos, quality flag, coverage percentage)

    And outputs a scalar reliability weight. All weights are softmax-normalized
    to sum to 1, ensuring the fusion is a proper weighted combination.

    Args:
        feature_dim: Dimension of global-average-pooled source features.
        metadata_dim: Dimension of per-source metadata vector.
        hidden_dim: Hidden layer size.
        num_sources: Number of data sources.
        temperature: Softmax temperature (lower = sharper weighting).
    """

    def __init__(
        self,
        feature_dim: int = 64,
        metadata_dim: int = 4,
        hidden_dim: int = 128,
        num_sources: int = 7,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.temperature = temperature

        # Per-source reliability estimator
        input_dim = feature_dim + metadata_dim
        self.reliability_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # scalar reliability score
        )

    def forward(
        self,
        source_features: dict[str, torch.Tensor],
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute reliability weights for each source.

        Args:
            source_features: Dict mapping source_name → [B, C, H', W'] feature maps.
            metadata: [B, num_sources, metadata_dim] per-source metadata.
                      If None, uses zeros (no metadata influence).

        Returns:
            Dict mapping source_name → [B, 1] reliability weight (softmax normalized).
        """
        source_names = list(source_features.keys())
        B = next(iter(source_features.values())).shape[0]
        device = next(iter(source_features.values())).device

        # Default metadata if not provided
        if metadata is None:
            metadata_dim = 4  # default
            metadata = torch.zeros(B, len(source_names), metadata_dim, device=device)

        raw_scores = []
        for i, name in enumerate(source_names):
            feat = source_features[name]  # [B, C, H', W']

            # Global average pool → [B, C]
            gap = feat.mean(dim=(-2, -1))  # [B, C]

            # Concatenate with metadata
            meta = metadata[:, i, :] if i < metadata.shape[1] else torch.zeros(
                B, metadata.shape[2], device=device
            )
            combined = torch.cat([gap, meta], dim=-1)  # [B, C + metadata_dim]

            # Compute reliability score
            score = self.reliability_net(combined)  # [B, 1]
            raw_scores.append(score)

        # Stack and normalize with softmax
        scores = torch.cat(raw_scores, dim=-1)  # [B, num_sources]
        weights = F.softmax(scores / self.temperature, dim=-1)  # [B, num_sources]

        # Return as dict
        result = {}
        for i, name in enumerate(source_names):
            result[name] = weights[:, i : i + 1]  # [B, 1]

        return result

    def get_reliability_summary(
        self,
        source_features: dict[str, torch.Tensor],
        metadata: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Get human-readable reliability scores (for logging/XAI).

        Returns:
            Dict mapping source_name → mean reliability weight across batch.
        """
        with torch.no_grad():
            weights = self.forward(source_features, metadata)
            return {name: float(w.mean()) for name, w in weights.items()}
