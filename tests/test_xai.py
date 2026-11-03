"""
Tests for XAI layer (Stage 13) and OOD estimation (Stage 12).
"""

import numpy as np
import pytest


class TestGradCAM:
    """Test Grad-CAM heatmap generation."""

    def test_synthetic_heatmap_shape(self):
        from cyclone_ai.xai.gradcam import GradCAMGenerator
        import torch

        gen = GradCAMGenerator()  # no model → synthetic mode
        heatmap = gen.generate(torch.randn(1, 7, 512, 512))

        assert heatmap.shape == (512, 512)
        assert heatmap.dtype == np.float32
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1.0

    def test_overlay(self):
        from cyclone_ai.xai.gradcam import GradCAMGenerator

        gen = GradCAMGenerator()
        heatmap = np.random.rand(64, 64).astype(np.float32)
        base = np.random.rand(64, 64).astype(np.float32)

        overlay = gen.overlay_on_image(heatmap, base, alpha=0.5)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8


class TestExplanationGenerator:
    """Test explanation tag generation."""

    def test_eye_tags(self):
        from cyclone_ai.xai.explanations import ExplanationGenerator

        gen = ExplanationGenerator()
        tags = gen.generate(
            structure={
                "eye_present": True, "eye_confidence": 0.9,
                "eye_diameter_km": 40, "symmetry": 0.85, "circularity": 0.8,
            },
        )

        assert "well-defined eye" in tags
        assert "strong eyewall convection" in tags

    def test_no_eye_tags(self):
        from cyclone_ai.xai.explanations import ExplanationGenerator

        gen = ExplanationGenerator()
        tags = gen.generate(
            structure={
                "eye_present": False, "eye_confidence": 0.1,
                "symmetry": 0.3, "circularity": 0.2,
            },
        )

        assert "well-defined eye" not in tags
        assert "asymmetric structure" in tags

    def test_ri_tags(self):
        from cyclone_ai.xai.explanations import ExplanationGenerator

        gen = ExplanationGenerator()
        tags = gen.generate(
            structure={"symmetry": 0.5, "circularity": 0.5},
            ri={"probability": 0.8, "confidence": 0.9},
        )

        assert "rapid intensification likely" in tags

    def test_environmental_tags(self):
        from cyclone_ai.xai.explanations import ExplanationGenerator

        gen = ExplanationGenerator()
        tags = gen.generate(
            structure={"symmetry": 0.5, "circularity": 0.5},
            env_scalars={"WIND_SHEAR_mean": 3.0, "SST_mean": 31.0},
        )

        assert "low wind shear environment" in tags
        assert "very warm SST" in tags

    def test_always_produces_tags(self):
        from cyclone_ai.xai.explanations import ExplanationGenerator

        gen = ExplanationGenerator()
        tags = gen.generate(structure={"symmetry": 0.5, "circularity": 0.5})

        assert len(tags) > 0

    def test_no_duplicates(self):
        from cyclone_ai.xai.explanations import ExplanationGenerator

        gen = ExplanationGenerator()
        tags = gen.generate(
            structure={"eye_present": True, "eye_confidence": 0.9, "symmetry": 0.9, "circularity": 0.9},
            classification={"category": "SuCS", "stage": "mature", "pattern": "CDO"},
            ri={"probability": 0.8},
            env_scalars={"WIND_SHEAR_mean": 3.0, "SST_mean": 31.0},
        )

        assert len(tags) == len(set(tags))


class TestOODEstimation:
    """Test OOD detection (Stage 12)."""

    def test_mahalanobis_fitted(self):
        from cyclone_ai.uncertainty.ood_estimation import OODDetector

        detector = OODDetector()
        training_features = np.random.randn(100, 64).astype(np.float32)
        detector.fit(training_features)

        assert detector.is_fitted

        # In-distribution sample should have low distance
        in_dist = training_features[0]
        dist = detector.mahalanobis_distance(in_dist)
        assert dist < 10  # should be reasonable

        # Out-of-distribution sample should have high distance
        ood_sample = np.ones(64, dtype=np.float32) * 10
        ood_dist = detector.mahalanobis_distance(ood_sample)
        assert ood_dist > dist

    def test_unfitted_returns_default(self):
        from cyclone_ai.uncertainty.ood_estimation import OODDetector

        detector = OODDetector()
        dist = detector.mahalanobis_distance(np.random.randn(64).astype(np.float32))
        assert dist == 1.0  # default value

    def test_uncertainty_estimator(self):
        from cyclone_ai.uncertainty.ood_estimation import UncertaintyEstimator

        estimator = UncertaintyEstimator(novelty_threshold=3.0)

        result = estimator.estimate(
            encoder_features=np.random.randn(64).astype(np.float32),
            ri_probability=0.1,
            forecast={"track": [], "intensity": []},
            corrections_applied=[],
        )

        assert "novelty_score" in result
        assert "human_review_flag" in result
        assert isinstance(result["human_review_flag"], bool)

    def test_human_review_ambiguous_ri(self):
        from cyclone_ai.uncertainty.ood_estimation import UncertaintyEstimator

        estimator = UncertaintyEstimator(ri_ambiguity_range=(0.3, 0.7))

        result = estimator.estimate(
            encoder_features=np.random.randn(64).astype(np.float32),
            ri_probability=0.5,  # Ambiguous!
            forecast={"track": [], "intensity": []},
            corrections_applied=[],
        )

        assert result["human_review_flag"] is True

    def test_save_load(self, tmp_path):
        from cyclone_ai.uncertainty.ood_estimation import OODDetector

        detector = OODDetector()
        detector.fit(np.random.randn(50, 32).astype(np.float32))

        save_path = str(tmp_path / "ood.npz")
        detector.save(save_path)

        detector2 = OODDetector()
        detector2.load(save_path)
        assert detector2.is_fitted
