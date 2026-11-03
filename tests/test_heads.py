"""
Tests for prediction heads (Stages 5–8).
"""

import torch
import pytest


class TestDetectionHead:
    """Test detection head (Stage 5)."""

    def test_forward_shapes(self):
        from cyclone_ai.heads.detection import DetectionHead

        head = DetectionHead(feature_dim=768)
        spatial = torch.randn(2, 768, 4, 4)
        out = head(spatial, target_mask_size=(128, 128))

        assert out["presence_logit"].shape == (2, 1)
        assert out["confidence"].shape == (2, 1)
        assert out["center"].shape == (2, 2)
        assert out["bbox"].shape == (2, 4)
        assert out["mask_logit"].shape == (2, 1, 128, 128)
        assert out["mask"].shape == (2, 1, 128, 128)

    def test_confidence_range(self):
        from cyclone_ai.heads.detection import DetectionHead

        head = DetectionHead(feature_dim=256)
        spatial = torch.randn(4, 256, 8, 8)
        out = head(spatial)

        conf = out["confidence"]
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_loss_computation(self):
        from cyclone_ai.heads.detection import DetectionHead

        head = DetectionHead(feature_dim=128)
        spatial = torch.randn(2, 128, 4, 4)
        preds = head(spatial, target_mask_size=(64, 64))

        targets = {
            "present": torch.tensor([[1.0], [0.0]]),
            "center": torch.randn(2, 2),
            "bbox": torch.randn(2, 4),
            "mask": torch.rand(2, 1, 64, 64),
        }

        losses = head.compute_loss(preds, targets)
        assert "total" in losses
        assert losses["total"].requires_grad

    def test_fpn_decoder_upsamples(self):
        from cyclone_ai.heads.detection import FPNDecoder

        decoder = FPNDecoder(in_channels=256, mid_channels=128)
        x = torch.randn(1, 256, 4, 4)
        out = decoder(x, target_size=(512, 512))
        assert out.shape == (1, 1, 512, 512)


class TestStructureHead:
    """Test structure head (Stage 6)."""

    def test_forward_shapes(self):
        from cyclone_ai.heads.structure import StructureHead

        head = StructureHead(feature_dim=768)
        spatial = torch.randn(2, 768, 4, 4)
        out = head(spatial)

        assert out["eye_logit"].shape == (2, 1)
        assert out["eye_confidence"].shape == (2, 1)
        assert out["eye_center"].shape == (2, 2)
        assert out["eye_diameter_km"].shape == (2, 1)
        assert out["symmetry"].shape == (2, 1)
        assert out["circularity"].shape == (2, 1)
        assert out["attention_map"].shape[0] == 2

    def test_scores_bounded(self):
        from cyclone_ai.heads.structure import StructureHead

        head = StructureHead(feature_dim=256)
        spatial = torch.randn(4, 256, 8, 8)
        out = head(spatial)

        # eye_confidence, symmetry, circularity should all be in [0, 1]
        assert (out["eye_confidence"] >= 0).all() and (out["eye_confidence"] <= 1).all()
        assert (out["symmetry"] >= 0).all() and (out["symmetry"] <= 1).all()
        assert (out["circularity"] >= 0).all() and (out["circularity"] <= 1).all()

        # diameter must be non-negative
        assert (out["eye_diameter_km"] >= 0).all()

    def test_focal_loss(self):
        from cyclone_ai.heads.structure import StructureHead

        head = StructureHead(feature_dim=128)
        spatial = torch.randn(4, 128, 4, 4)
        preds = head(spatial)

        targets = {
            "eye_present": torch.tensor([[1], [0], [1], [0]], dtype=torch.float32),
            "eye_center": torch.rand(4, 2),
            "eye_diameter_km": torch.tensor([[40], [0], [30], [0]], dtype=torch.float32),
            "symmetry": torch.rand(4, 1),
            "circularity": torch.rand(4, 1),
        }

        losses = head.compute_loss(preds, targets)
        assert losses["eye_presence"].item() > 0
        assert losses["total"].requires_grad


class TestTemporalBlock:
    """Test temporal block (Stage 7)."""

    def test_convlstm_sequence(self):
        from cyclone_ai.heads.temporal import TemporalBlock

        block = TemporalBlock(input_dim=768, output_dim=512, model_type="convlstm")
        seq = torch.randn(2, 8, 768)  # 8 timesteps
        out = block(seq, is_sequence=True)

        assert out.shape == (2, 512)

    def test_transformer_sequence(self):
        from cyclone_ai.heads.temporal import TemporalBlock

        block = TemporalBlock(
            input_dim=256, output_dim=128,
            model_type="transformer",
            transformer_dim=128, transformer_heads=4, transformer_layers=2,
        )
        seq = torch.randn(2, 8, 256)
        out = block(seq, is_sequence=True)

        assert out.shape == (2, 128)

    def test_single_frame_fallback(self):
        from cyclone_ai.heads.temporal import TemporalBlock

        block = TemporalBlock(input_dim=768, output_dim=512)
        single = torch.randn(2, 768)
        out = block(single, is_sequence=False)

        assert out.shape == (2, 512)

    def test_convlstm_cell(self):
        from cyclone_ai.heads.temporal import ConvLSTMCell

        cell = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=1)
        x = torch.randn(2, 64, 1)
        h = torch.zeros(2, 32, 1)
        c = torch.zeros(2, 32, 1)
        h_next, c_next = cell(x, h, c)

        assert h_next.shape == (2, 32, 1)
        assert c_next.shape == (2, 32, 1)


class TestClassificationHead:
    """Test classification head (Stage 8)."""

    def test_forward_shapes(self):
        from cyclone_ai.heads.classification import ClassificationHead

        head = ClassificationHead(global_dim=768, temporal_dim=512)
        global_feat = torch.randn(2, 768)
        temporal_feat = torch.randn(2, 512)
        out = head(global_feat, temporal_feat)

        assert out["category_logits"].shape == (2, 7)   # 7 IMD categories
        assert out["stage_logits"].shape == (2, 5)       # 5 stages
        assert out["pattern_logits"].shape == (2, 7)     # 7 patterns

    def test_probabilities_sum_to_one(self):
        from cyclone_ai.heads.classification import ClassificationHead

        head = ClassificationHead(global_dim=256, temporal_dim=128)
        out = head(torch.randn(4, 256), torch.randn(4, 128))

        cat_sum = out["category_probs"].sum(dim=-1)
        assert torch.allclose(cat_sum, torch.ones(4), atol=1e-5)

    def test_contract_output(self):
        from cyclone_ai.heads.classification import ClassificationHead

        head = ClassificationHead(global_dim=128, temporal_dim=64)
        out = head(torch.randn(1, 128), torch.randn(1, 64))
        contract = head.to_contract(out)

        assert contract["category"] in head.categories
        assert contract["stage"] in head.stages
        assert contract["pattern"] in head.patterns
