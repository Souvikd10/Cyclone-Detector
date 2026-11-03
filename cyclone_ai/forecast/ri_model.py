"""
Rapid Intensification (RI) Model — Stage 10.

XGBoost classifier on tabular environmental/structural features
predicting whether wind increases ≥30kt in the next 24 hours.

Separate from the neural network gradient graph — trained independently.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RIFeatureExtractor:
    """
    Extract tabular features for the RI model from pipeline outputs.

    Feature groups:
    - Current state: wind, pressure, lat, lon
    - Environmental: SST, wind shear, moisture (means + stds)
    - Structural: symmetry, circularity, eye presence, eye diameter
    - Temporal: intensity change over 6h, 12h, 24h windows
    """

    FEATURE_NAMES = [
        # Current state (4)
        "wind_kt", "pressure_hpa", "latitude", "longitude",
        # Environmental (6)
        "sst_mean", "sst_std", "wind_shear_mean",
        "moisture_mean", "sst_minus_threshold",
        "shear_intensity_ratio",
        # Structural (4)
        "symmetry", "circularity", "eye_present", "eye_diameter_km",
        # Temporal (4)
        "delta_wind_6h", "delta_wind_12h", "delta_wind_24h",
        "intensification_rate",
    ]

    SST_THRESHOLD = 26.5  # °C — minimum for sustained cyclone

    def extract(
        self,
        wind_kt: float,
        pressure_hpa: float,
        lat: float,
        lon: float,
        env_scalars: dict[str, float],
        structure: dict,
        temporal_history: list[float] | None = None,
    ) -> np.ndarray:
        """
        Extract feature vector for a single sample.

        Args:
            wind_kt: Current max sustained wind (knots).
            pressure_hpa: Current min central pressure (hPa).
            lat: Current latitude.
            lon: Current longitude.
            env_scalars: Environmental scalar features (from ERA5Client).
            structure: Structure head output dict.
            temporal_history: List of past wind values [t-24h, t-18h, ..., t-6h, t]
                              for computing Δwind. If None, set to 0.

        Returns:
            1D numpy array of shape [num_features].
        """
        features = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        # Current state
        features[0] = wind_kt
        features[1] = pressure_hpa
        features[2] = lat
        features[3] = lon

        # Environmental
        features[4] = env_scalars.get("SST_mean", 28.0)
        features[5] = env_scalars.get("SST_std", 0.5)
        features[6] = env_scalars.get("WIND_SHEAR_mean", 10.0)
        features[7] = env_scalars.get("MOISTURE_mean", 65.0)
        features[8] = features[4] - self.SST_THRESHOLD  # SST anomaly
        features[9] = features[6] / max(wind_kt, 1.0)  # shear-to-intensity ratio

        # Structural
        features[10] = structure.get("symmetry", 0.5)
        features[11] = structure.get("circularity", 0.5)
        features[12] = float(structure.get("eye_present", False))
        features[13] = structure.get("eye_diameter_km", 0.0)

        # Temporal
        if temporal_history and len(temporal_history) >= 2:
            features[14] = wind_kt - temporal_history[-2] if len(temporal_history) >= 2 else 0.0  # 6h
            features[15] = wind_kt - temporal_history[-3] if len(temporal_history) >= 3 else 0.0  # 12h
            features[16] = wind_kt - temporal_history[-5] if len(temporal_history) >= 5 else 0.0  # 24h
            features[17] = features[14] / 6.0  # kt/hour intensification rate
        else:
            features[14:18] = 0.0

        return features

    def extract_batch(self, samples: list[dict]) -> np.ndarray:
        """Extract features for a batch of samples."""
        return np.stack([
            self.extract(**s) for s in samples
        ])


class RIModel:
    """
    Rapid Intensification classifier (Stage 10).

    Uses XGBoost for binary classification: RI (≥30kt in 24h) vs. non-RI.
    Probabilities are Platt-scaled for calibration.

    Usage:
        model = RIModel()
        model.train(features, labels)
        prob, conf = model.predict(features)
    """

    def __init__(
        self,
        ri_threshold_kt: float = 30.0,
        xgb_params: dict | None = None,
        model_path: str = "checkpoints/ri_model.json",
    ):
        self.ri_threshold = ri_threshold_kt
        self.model_path = Path(model_path)
        self.feature_extractor = RIFeatureExtractor()

        self.xgb_params = xgb_params or {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 5,
            "use_label_encoder": False,
            "verbosity": 0,
        }

        self._model = None
        self._calibrator = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
    ) -> dict:
        """
        Train the XGBoost RI classifier.

        Args:
            features: [N, num_features] training features.
            labels: [N] binary labels (1 = RI, 0 = non-RI).
            val_features: Optional validation features.
            val_labels: Optional validation labels.

        Returns:
            Dict with training metrics.
        """
        try:
            import xgboost as xgb
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
        except ImportError as e:
            logger.error(f"Missing dependency for RI training: {e}")
            raise

        logger.info(f"Training RI model: {len(features)} samples, {labels.sum():.0f} positive")

        # Train XGBoost
        n_estimators = self.xgb_params.pop("n_estimators", 200)
        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            **self.xgb_params,
        )

        eval_set = [(features, labels)]
        if val_features is not None:
            eval_set.append((val_features, val_labels))

        self._model.fit(
            features, labels,
            eval_set=eval_set,
            verbose=False,
        )
        self.xgb_params["n_estimators"] = n_estimators  # restore

        # Platt scaling for calibrated probabilities
        self._calibrator = CalibratedClassifierCV(
            self._model, cv=3, method="sigmoid",
        )
        self._calibrator.fit(features, labels)

        # Compute metrics
        metrics = {}
        if val_features is not None and val_labels is not None:
            val_probs = self._calibrator.predict_proba(val_features)[:, 1]
            metrics["val_auc"] = float(roc_auc_score(val_labels, val_probs))
            metrics["val_ap"] = float(average_precision_score(val_labels, val_probs))
            logger.info(f"RI model val AUC: {metrics['val_auc']:.4f}, AP: {metrics['val_ap']:.4f}")

        return metrics

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Predict RI probability for a single sample.

        Args:
            features: [num_features] or [1, num_features] feature vector.

        Returns:
            Tuple of (probability, confidence).
            Confidence is derived from how far the probability is from 0.5.
        """
        if not self.is_trained:
            logger.warning("RI model not trained, returning default prediction")
            return 0.1, 0.5

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._calibrator is not None:
            prob = float(self._calibrator.predict_proba(features)[0, 1])
        else:
            prob = float(self._model.predict_proba(features)[0, 1])

        # Confidence: higher when probability is far from 0.5 (clear decision)
        confidence = 2.0 * abs(prob - 0.5)  # 0.0 at prob=0.5, 1.0 at prob=0 or 1
        confidence = min(confidence + 0.5, 1.0)  # floor at 0.5

        return prob, confidence

    def save(self, path: str | None = None):
        """Save model to disk."""
        save_path = Path(path or self.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            self._model.save_model(str(save_path))
            logger.info(f"Saved RI model to {save_path}")

            # Save calibrator with pickle
            import pickle
            cal_path = save_path.with_suffix(".calibrator.pkl")
            with open(cal_path, "wb") as f:
                pickle.dump(self._calibrator, f)

    def load(self, path: str | None = None):
        """Load model from disk."""
        load_path = Path(path or self.model_path)

        if not load_path.exists():
            logger.warning(f"RI model not found at {load_path}")
            return

        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(load_path))

            # Load calibrator
            import pickle
            cal_path = load_path.with_suffix(".calibrator.pkl")
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    self._calibrator = pickle.load(f)

            logger.info(f"Loaded RI model from {load_path}")
        except Exception as e:
            logger.warning(f"Failed to load RI model: {e}")

    def to_contract(self, features: np.ndarray) -> dict:
        """Predict and format as contract-compliant RI dict."""
        prob, conf = self.predict(features)
        return {
            "probability": round(prob, 4),
            "confidence": round(conf, 4),
        }
