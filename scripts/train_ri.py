"""
RI Model Training Script — Trains XGBoost rapid intensification classifier.

Usage:
    python scripts/train_ri.py --num-samples 500
    python scripts/train_ri.py --output checkpoints/ri_model.json
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_ri")


def generate_synthetic_ri_data(num_samples: int = 500, ri_fraction: float = 0.05):
    """
    Generate synthetic tabular data for RI training.

    RI occurs ~5% of the time, triggered by:
    - Warm SST (>28°C)
    - Low wind shear (<10 m/s)
    - Moderate current intensity (30-90 kt)
    - Developing eye
    """
    np.random.seed(42)

    from cyclone_ai.forecast.ri_model import RIFeatureExtractor
    extractor = RIFeatureExtractor()

    features_list = []
    labels = []

    for i in range(num_samples):
        wind = np.random.uniform(20, 120)
        sst = np.random.uniform(25, 32)
        shear = np.random.uniform(2, 25)

        # RI is more likely with warm SST, low shear, moderate wind
        ri_prob = 0.0
        if sst > 28 and shear < 10 and 30 < wind < 90:
            ri_prob = 0.4
        if sst > 29 and shear < 7 and 40 < wind < 80:
            ri_prob = 0.7

        is_ri = np.random.random() < ri_prob

        env_scalars = {
            "SST_mean": sst,
            "SST_std": np.random.uniform(0.1, 1.0),
            "WIND_SHEAR_mean": shear,
            "MOISTURE_mean": np.random.uniform(50, 85),
        }

        eye_present = wind >= 64 or (wind >= 50 and np.random.random() > 0.5)
        structure = {
            "symmetry": np.random.uniform(0.3, 0.9),
            "circularity": np.random.uniform(0.2, 0.85),
            "eye_present": eye_present,
            "eye_diameter_km": np.random.uniform(20, 60) if eye_present else 0.0,
        }

        feats = extractor.extract(
            wind_kt=wind,
            pressure_hpa=1013 - wind * 0.6,
            lat=np.random.uniform(8, 22),
            lon=np.random.uniform(70, 95),
            env_scalars=env_scalars,
            structure=structure,
        )

        features_list.append(feats)
        labels.append(int(is_ri))

    features = np.stack(features_list)
    labels = np.array(labels)

    logger.info(f"Generated {num_samples} samples: {labels.sum()} RI ({labels.mean() * 100:.1f}%)")
    return features, labels


def train_ri(args):
    """Train the XGBoost RI model."""
    from cyclone_ai.forecast.ri_model import RIModel

    # Generate data
    features, labels = generate_synthetic_ri_data(args.num_samples)

    # Split
    split_idx = int(0.8 * len(features))
    train_X, val_X = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    # Train
    model = RIModel(model_path=args.output)
    metrics = model.train(train_X, train_y, val_X, val_y)

    # Test prediction
    test_prob, test_conf = model.predict(val_X[0])
    logger.info(f"Sample prediction: prob={test_prob:.4f}, conf={test_conf:.4f}")

    # Save
    model.save()
    logger.info(f"Model saved to {args.output}")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("cyclone-ri-training")
        with mlflow.start_run(run_name="ri-xgboost"):
            mlflow.log_params({
                "num_samples": args.num_samples,
                "train_size": len(train_X),
                "val_size": len(val_X),
                "ri_fraction_train": float(train_y.mean()),
            })
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RI XGBoost model")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--output", default="checkpoints/ri_model.json")
    args = parser.parse_args()
    train_ri(args)
