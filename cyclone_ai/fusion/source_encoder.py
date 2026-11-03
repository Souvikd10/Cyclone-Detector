"""
Per-Source Encoder — Stage 3.

Lightweight CNN encoders that independently encode each data source
(IR, WV, VIS, SST, etc.) into feature maps before fusion.
"""

import torch
import torch.nn as nn


class SourceEncoderBlock(nn.Module):
    """Single convolutional block with GELU activation and BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SingleSourceEncoder(nn.Module):
    """
    Lightweight 4-layer CNN for encoding a single data source.

    Input:  [B, 1, H, W] — single-channel image
    Output: [B, output_channels, H/16, W/16] — feature map

    Each layer: Conv2d → BatchNorm → GELU → MaxPool2d(2)
    Four layers gives a 16× spatial downsampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        output_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        ch_in = in_channels
        for i in range(num_layers):
            ch_out = base_channels * (2 ** min(i, 2))  # 16, 32, 64, 64
            if i == num_layers - 1:
                ch_out = output_channels
            layers.append(SourceEncoderBlock(ch_in, ch_out, kernel_size))
            ch_in = ch_out

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] single-channel input.

        Returns:
            [B, output_channels, H/16, W/16] feature map.
        """
        return self.encoder(x)


class MultiSourceEncoder(nn.Module):
    """
    Collection of per-source encoders for all data channels.

    Creates a separate SingleSourceEncoder for each data source and
    applies them in parallel.

    Args:
        source_names: List of source/channel names.
        base_channels: Base channel count for each encoder.
        output_channels: Output channels per source.
    """

    def __init__(
        self,
        source_names: list[str],
        base_channels: int = 16,
        output_channels: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.source_names = source_names

        self.encoders = nn.ModuleDict({
            name: SingleSourceEncoder(
                in_channels=1,
                base_channels=base_channels,
                output_channels=output_channels,
                num_layers=num_layers,
            )
            for name in source_names
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encode each source channel independently.

        Args:
            x: [B, C, H, W] multi-channel input where C = len(source_names).

        Returns:
            Dict mapping source_name → [B, output_channels, H', W'] feature map.
        """
        features = {}
        for i, name in enumerate(self.source_names):
            # Extract single channel: [B, 1, H, W]
            source_input = x[:, i : i + 1, :, :]
            features[name] = self.encoders[name](source_input)

        return features

    @property
    def num_sources(self) -> int:
        return len(self.source_names)
