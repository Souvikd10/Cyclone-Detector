"""
Temporal Evolution Block — Stage 7.

Dense sequence modeling of structural and intensity changes over time.
Supports ConvLSTM (default) and Transformer encoder architectures.

Takes a sequence of global feature vectors from the shared encoder
and produces a temporal-context vector capturing evolution patterns.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.

    Replaces fully connected gates with convolutional operations
    to preserve spatial structure in hidden states. For 1D feature
    vectors, uses kernel_size=1 (equivalent to standard LSTM).
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Gates: input, forget, cell, output (4× hidden_dim)
        self.gates = nn.Conv1d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True,
        )

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim, L] input features.
            h_prev: [B, hidden_dim, L] previous hidden state.
            c_prev: [B, hidden_dim, L] previous cell state.

        Returns:
            (h_next, c_next) each [B, hidden_dim, L].
        """
        combined = torch.cat([x, h_prev], dim=1)  # [B, input+hidden, L]
        gates = self.gates(combined)  # [B, 4*hidden, L]

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM for temporal sequence modeling.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden state dimension.
        num_layers: Number of LSTM layers.
        kernel_size: Convolution kernel size (1 for vector inputs).
        dropout: Dropout between layers.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        kernel_size: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, input_dim] temporal sequence of feature vectors.

        Returns:
            Tuple of:
                output: [B, T, hidden_dim] hidden states for all timesteps
                last_hidden: [B, hidden_dim] last hidden state
        """
        B, T, D = x.shape

        # Reshape for Conv1d: [B, D, 1] (treating feature dim as channels)
        outputs = []

        for layer_idx, cell in enumerate(self.cells):
            h = torch.zeros(B, self.hidden_dim, 1, device=x.device)
            c = torch.zeros(B, self.hidden_dim, 1, device=x.device)
            layer_outputs = []

            for t in range(T):
                if layer_idx == 0:
                    inp = x[:, t, :].unsqueeze(-1)  # [B, D, 1]
                else:
                    inp = outputs[t]  # from previous layer

                h, c = cell(inp, h, c)
                layer_outputs.append(h)

            outputs = layer_outputs

            if layer_idx < self.num_layers - 1:
                outputs = [self.dropout(o) for o in outputs]

        # Stack outputs: [B, T, hidden_dim]
        stacked = torch.cat(outputs, dim=-1).squeeze(1) if self.hidden_dim == 1 else \
            torch.stack([o.squeeze(-1) for o in outputs], dim=1)  # [B, T, hidden_dim]

        last_hidden = outputs[-1].squeeze(-1)  # [B, hidden_dim]

        return stacked, last_hidden


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal sequence modeling.

    Uses sinusoidal time-aware positional encoding and learned
    embeddings for each time step.

    Args:
        input_dim: Input feature dimension.
        model_dim: Transformer model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 768,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.model_dim = model_dim

        # Project input to model dimension
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Positional encoding
        self.pos_encoding = self._create_sinusoidal_pe(max_seq_len, model_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(model_dim)

    @staticmethod
    def _create_sinusoidal_pe(max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, input_dim] temporal sequence.

        Returns:
            Tuple of:
                output: [B, T, model_dim] encoded sequence
                pooled: [B, model_dim] mean-pooled output
        """
        B, T, _ = x.shape

        # Project and add positional encoding
        x = self.input_proj(x)  # [B, T, model_dim]
        x = x + self.pos_encoding[:, :T, :]

        # Transformer
        output = self.transformer(x)  # [B, T, model_dim]
        output = self.norm(output)

        # Mean pooling over time
        pooled = output.mean(dim=1)  # [B, model_dim]

        return output, pooled


class TemporalBlock(nn.Module):
    """
    Temporal Evolution Block (Stage 7).

    Wraps ConvLSTM or Transformer encoder based on configuration.
    Produces a temporal-context vector for downstream heads.

    For single-frame inference (no history), falls back to linear
    projection of the input features.

    Args:
        input_dim: Dimension of per-frame features from encoder.
        output_dim: Output temporal feature dimension.
        model_type: "convlstm" or "transformer".
        sequence_length: Expected sequence length.
        **kwargs: Architecture-specific hyperparameters.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        model_type: str = "convlstm",
        sequence_length: int = 8,
        # ConvLSTM params
        convlstm_hidden: int = 256,
        convlstm_layers: int = 2,
        convlstm_kernel: int = 1,
        # Transformer params
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = model_type
        self.sequence_length = sequence_length

        if model_type == "convlstm":
            self.temporal_model = ConvLSTM(
                input_dim=input_dim,
                hidden_dim=convlstm_hidden,
                num_layers=convlstm_layers,
                kernel_size=convlstm_kernel,
                dropout=dropout,
            )
            temporal_out_dim = convlstm_hidden
        elif model_type == "transformer":
            self.temporal_model = TemporalTransformer(
                input_dim=input_dim,
                model_dim=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
            )
            temporal_out_dim = transformer_dim
        else:
            raise ValueError(f"Unknown temporal model type: {model_type}")

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(temporal_out_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # Fallback for single-frame (no temporal context)
        self.single_frame_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        logger.info(f"TemporalBlock: type={model_type}, input={input_dim}, output={output_dim}")

    def forward(
        self,
        features: torch.Tensor,
        is_sequence: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, input_dim] sequence of per-frame features,
                      or [B, input_dim] single frame.
            is_sequence: Whether input is a temporal sequence.

        Returns:
            [B, output_dim] temporal context vector.
        """
        if not is_sequence or features.dim() == 2:
            # Single frame fallback
            if features.dim() == 3:
                features = features[:, -1, :]  # take last frame
            return self.single_frame_proj(features)

        # Temporal modeling
        _, pooled = self.temporal_model(features)  # [B, temporal_out_dim]
        output = self.output_proj(pooled)  # [B, output_dim]

        return output
