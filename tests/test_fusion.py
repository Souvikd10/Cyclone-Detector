"""
Tests for the fusion module (Stage 3).
"""

import torch
import pytest


class TestSourceEncoder:
    """Test per-source CNN encoders."""

    def test_single_source_encoder_shape(self):
        from cyclone_ai.fusion.source_encoder import SingleSourceEncoder

        encoder = SingleSourceEncoder(in_channels=1, output_channels=64, num_layers=4)
        x = torch.randn(2, 1, 128, 128)
        out = encoder(x)

        # 4 layers of MaxPool2d(2) → 128 / 16 = 8
        assert out.shape == (2, 64, 8, 8)

    def test_multi_source_encoder(self):
        from cyclone_ai.fusion.source_encoder import MultiSourceEncoder

        names = ["IR", "WV", "VIS"]
        encoder = MultiSourceEncoder(source_names=names, output_channels=64)
        x = torch.randn(2, 3, 64, 64)
        features = encoder(x)

        assert len(features) == 3
        for name in names:
            assert name in features
            assert features[name].shape == (2, 64, 4, 4)


class TestReliabilityNetwork:
    """Test learned reliability weights."""

    def test_weights_sum_to_one(self):
        from cyclone_ai.fusion.reliability import ReliabilityNetwork

        net = ReliabilityNetwork(feature_dim=64, num_sources=3)
        features = {
            "IR": torch.randn(4, 64, 8, 8),
            "WV": torch.randn(4, 64, 8, 8),
            "VIS": torch.randn(4, 64, 8, 8),
        }

        weights = net(features)
        total = sum(w.squeeze() for w in weights.values())

        # Should sum to 1.0 for each sample in batch
        assert torch.allclose(total, torch.ones(4), atol=1e-5)

    def test_reliability_summary(self):
        from cyclone_ai.fusion.reliability import ReliabilityNetwork

        net = ReliabilityNetwork(feature_dim=64, num_sources=2)
        features = {
            "A": torch.randn(2, 64, 4, 4),
            "B": torch.randn(2, 64, 4, 4),
        }

        summary = net.get_reliability_summary(features)
        assert "A" in summary
        assert "B" in summary
        assert abs(summary["A"] + summary["B"] - 1.0) < 0.1


class TestAttentionFusion:
    """Test the complete attention-gated fusion module."""

    def test_fusion_forward_pass(self):
        from cyclone_ai.fusion.attention_fusion import AttentionGatedFusion

        source_names = ["IR_TIR1", "IR_TIR2", "WV"]
        fusion = AttentionGatedFusion(
            source_names=source_names,
            source_channels=64,
            output_dim=256,
            num_heads=4,
        )

        x = torch.randn(2, 3, 64, 64)
        out = fusion(x)

        # Input 64×64, source encoders downsample 16×, so spatial = 4×4
        assert out.shape == (2, 256, 4, 4)

    def test_fusion_with_metadata(self):
        from cyclone_ai.fusion.attention_fusion import AttentionGatedFusion

        source_names = ["A", "B"]
        fusion = AttentionGatedFusion(
            source_names=source_names,
            source_channels=32,
            output_dim=128,
        )

        x = torch.randn(2, 2, 32, 32)
        metadata = torch.randn(2, 2, 4)  # [B, num_sources, metadata_dim]
        out = fusion(x, metadata=metadata)

        assert out.shape[0] == 2
        assert out.shape[1] == 128

    def test_reliability_scores_accessible(self):
        from cyclone_ai.fusion.attention_fusion import AttentionGatedFusion

        names = ["IR", "WV", "VIS"]
        fusion = AttentionGatedFusion(source_names=names, source_channels=32, output_dim=64)

        x = torch.randn(2, 3, 32, 32)
        scores = fusion.get_reliability_scores(x)

        assert len(scores) == 3
        for name in names:
            assert name in scores
            assert 0.0 <= scores[name] <= 1.0
