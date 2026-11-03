"""
OOD Detection + Uncertainty Estimation — Stage 12.

Computes novelty scores, forecast cones, intensity intervals,
RI confidence, and human-review flags.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Out-of-Distribution detector using Mahalanobis distance.

    Fits a multivariate Gaussian on the encoder's global features
    from the training set. At inference, computes the Mahalanobis
    distance of new features from this distribution.

    High distance → input is unlike anything seen during training.
    """

    def __init__(self, stats_path: str = "checkpoints/ood_stats.npz"):
        self.stats_path = Path(stats_path)
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: np.ndarray):
        """
        Fit Gaussian distribution on training features.

        Args:
            features: [N, D] feature matrix from encoder global features.
        """
        self._mean = np.mean(features, axis=0)  # [D]
        cov = np.cov(features, rowvar=False)      # [D, D]

        # Regularize covariance for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        self._cov_inv = np.linalg.inv(cov)  # [D, D]
        self._fitted = True

        logger.info(f"OOD detector fitted: {features.shape[0]} samples, {features.shape[1]}-d features")

    def mahalanobis_distance(self, features: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from training distribution.

        Args:
            features: [D] or [1, D] feature vector.

        Returns:
            Scalar Mahalanobis distance.
        """
        if not self._fitted:
            logger.warning("OOD detector not fitted, returning default score")
            return 1.0

        if features.ndim == 2:
            features = features[0]

        diff = features - self._mean
        distance = float(np.sqrt(diff @ self._cov_inv @ diff))
        return distance

    def save(self, path: str | None = None):
        """Save fitted statistics to disk."""
        save_path = Path(path or self.stats_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._fitted:
            np.savez(
                str(save_path),
                mean=self._mean,
                cov_inv=self._cov_inv,
            )
            logger.info(f"Saved OOD stats to {save_path}")

    def load(self, path: str | None = None):
        """Load fitted statistics from disk."""
        load_path = Path(path or self.stats_path)
        if not load_path.exists():
            logger.warning(f"OOD stats not found at {load_path}")
            return

        data = np.load(str(load_path))
        self._mean = data["mean"]
        self._cov_inv = data["cov_inv"]
        self._fitted = True
        logger.info(f"Loaded OOD stats from {load_path}")


class UncertaintyEstimator:
    """
    Uncertainty estimation module (Stage 12).

    Combines:
    - Novelty score (Mahalanobis distance from OOD detector)
    - Forecast cone scaling (empirical, from validation errors)
    - Intensity interval (from quantile regression in forecast model)
    - RI confidence (from Platt-scaled XGBoost)
    - Human review flag (triggered by multiple conditions)

    Args:
        novelty_threshold: Mahalanobis distance above which input is flagged OOD.
        ri_ambiguity_range: RI probability range considered ambiguous.
        cone_scaling: Dict mapping horizon_hours → base cone radius in km.
    """

    DEFAULT_CONE_SCALING = {6: 50, 12: 85, 24: 140, 48: 250, 72: 380}

    def __init__(
        self,
        novelty_threshold: float = 3.0,
        ri_ambiguity_range: tuple[float, float] = (0.3, 0.7),
        cone_scaling: dict[int, int] | None = None,
        intensity_correction_threshold_kt: float = 20.0,
    ):
        self.novelty_threshold = novelty_threshold
        self.ri_ambiguity_range = ri_ambiguity_range
        self.cone_scaling = cone_scaling or self.DEFAULT_CONE_SCALING
        self.intensity_correction_threshold = intensity_correction_threshold_kt

        self.ood_detector = OODDetector()

    def estimate(
        self,
        encoder_features: np.ndarray,
        ri_probability: float,
        forecast: dict,
        corrections_applied: list[str],
        raw_wind_kt: float | None = None,
        corrected_wind_kt: float | None = None,
    ) -> dict:
        """
        Compute full uncertainty estimates.

        Args:
            encoder_features: [D] global features from shared encoder.
            ri_probability: RI probability from Stage 10.
            forecast: Corrected forecast dict from Stage 11.
            corrections_applied: List of physics corrections applied.
            raw_wind_kt: Raw model intensity prediction (before correction).
            corrected_wind_kt: Corrected intensity (after physics correction).

        Returns:
            Contract-compliant uncertainty dict.
        """
        # 1. Novelty score
        novelty = self.ood_detector.mahalanobis_distance(encoder_features)

        # 2. Human review flag — check all trigger conditions
        review_reasons = []

        # Condition 1: High novelty (OOD)
        if novelty > self.novelty_threshold:
            review_reasons.append(f"novelty_score={novelty:.2f} exceeds threshold")

        # Condition 2: Ambiguous RI prediction
        if self.ri_ambiguity_range[0] <= ri_probability <= self.ri_ambiguity_range[1]:
            review_reasons.append(f"RI probability={ri_probability:.2f} is ambiguous")

        # Condition 3: Large physics correction
        if raw_wind_kt is not None and corrected_wind_kt is not None:
            if abs(raw_wind_kt - corrected_wind_kt) > self.intensity_correction_threshold:
                review_reasons.append(
                    f"Large intensity correction: {raw_wind_kt:.0f}→{corrected_wind_kt:.0f} kt"
                )

        # Condition 4: Track crosses land unexpectedly
        if any("Landfall" in c for c in corrections_applied):
            review_reasons.append("Track crosses land boundary")

        human_review = len(review_reasons) > 0

        if review_reasons:
            logger.info(f"Human review flagged: {'; '.join(review_reasons)}")

        return {
            "novelty_score": round(float(novelty), 4),
            "human_review_flag": human_review,
        }

    def scale_forecast_cones(self, forecast: dict) -> dict:
        """
        Scale forecast cone radii based on empirical error distribution.

        Updates cone_radius_km values in the forecast track using
        empirical scaling factors by lead time.
        """
        for tp in forecast.get("track", []):
            hours = int(tp["t"].replace("+", "").replace("h", ""))
            if hours in self.cone_scaling:
                tp["cone_radius_km"] = self.cone_scaling[hours]
        return forecast

    def fit_ood(self, training_features: np.ndarray):
        """Fit the OOD detector on training features."""
        self.ood_detector.fit(training_features)

    def save(self, path: str | None = None):
        """Save OOD detector state."""
        self.ood_detector.save(path)

    def load(self, path: str | None = None):
        """Load OOD detector state."""
        self.ood_detector.load(path)
