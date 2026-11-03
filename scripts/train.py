"""
Multi-Task Training Script — Trains all neural network heads jointly.

Stages trained: Detection (5), Structure (6), Temporal (7),
Classification (8), Forecast (9).

Usage:
    python scripts/train.py --config configs/model.yaml --epochs 100
    python scripts/train.py --device cuda --batch-size 4
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train")


class SyntheticTrainingDataset(Dataset):
    """
    Synthetic dataset for training all heads.

    Generates random tensors with plausible shapes and labels for
    multi-task training. Replace with real data for production training.
    """

    def __init__(self, num_samples: int = 200, num_channels: int = 7, image_size: int = 128):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.image_size = image_size

        # Pre-generate for consistency
        np.random.seed(42)
        self.wind_speeds = np.random.uniform(15, 140, num_samples).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        wind = self.wind_speeds[idx]
        intensity_factor = wind / 80.0
        has_eye = wind >= 64

        # Generate synthetic input (mimic preprocessed satellite data)
        image = np.random.randn(self.num_channels, self.image_size, self.image_size).astype(np.float32)

        # Add vortex-like signal
        cy, cx = self.image_size // 2, self.image_size // 2
        y, x = np.ogrid[-cy:self.image_size - cy, -cx:self.image_size - cx]
        r = np.sqrt(x * x + y * y).astype(np.float32)
        vortex = np.exp(-r ** 2 / (2 * (20 * intensity_factor) ** 2))

        for c in range(min(4, self.num_channels)):
            image[c] += vortex * intensity_factor * 3

        # Labels
        # Detection
        present = 1.0 if wind >= 25 else 0.0
        center = np.array([0.5 + np.random.uniform(-0.05, 0.05),
                          0.5 + np.random.uniform(-0.05, 0.05)], dtype=np.float32)
        bbox = np.array([0.3, 0.3, 0.7, 0.7], dtype=np.float32)
        mask = (vortex > 0.3).astype(np.float32)[np.newaxis]  # [1, H, W]
        # Resize mask to match expected size
        from scipy.ndimage import zoom
        if mask.shape[1] != 512:
            mask = zoom(mask, (1, 512 / mask.shape[1], 512 / mask.shape[2]), order=0)

        # Structure
        eye_present = 1.0 if has_eye else 0.0
        eye_center = center.copy()
        eye_diameter = np.float32(40.0 if has_eye else 0.0)
        symmetry = np.float32(0.8 if has_eye else np.random.uniform(0.3, 0.6))
        circularity = np.float32(0.75 if has_eye else np.random.uniform(0.2, 0.5))

        # Classification
        imd_thresholds = [(120, 6), (90, 5), (64, 4), (48, 3), (34, 2), (28, 1), (17, 0)]
        category_idx = 0
        for thresh, idx_val in imd_thresholds:
            if wind >= thresh:
                category_idx = idx_val
                break

        stage_idx = 2 if wind >= 64 else (1 if wind >= 34 else 0)
        pattern_idx = 4 if has_eye else (0 if wind >= 48 else 1)

        return {
            "image": torch.from_numpy(image),
            "detection": {
                "present": torch.tensor([present]),
                "center": torch.from_numpy(center),
                "bbox": torch.from_numpy(bbox),
                "mask": torch.from_numpy(mask),
            },
            "structure": {
                "eye_present": torch.tensor([eye_present]),
                "eye_center": torch.from_numpy(eye_center),
                "eye_diameter_km": torch.tensor([eye_diameter]),
                "symmetry": torch.tensor([symmetry]),
                "circularity": torch.tensor([circularity]),
            },
            "classification": {
                "category": torch.tensor(category_idx, dtype=torch.long),
                "stage": torch.tensor(stage_idx, dtype=torch.long),
                "pattern": torch.tensor(pattern_idx, dtype=torch.long),
            },
        }


def collate_fn(batch):
    """Custom collate for nested dict structure."""
    images = torch.stack([b["image"] for b in batch])

    detection = {
        "present": torch.stack([b["detection"]["present"] for b in batch]),
        "center": torch.stack([b["detection"]["center"] for b in batch]),
        "bbox": torch.stack([b["detection"]["bbox"] for b in batch]),
        "mask": torch.stack([b["detection"]["mask"] for b in batch]),
    }

    structure = {
        "eye_present": torch.stack([b["structure"]["eye_present"] for b in batch]),
        "eye_center": torch.stack([b["structure"]["eye_center"] for b in batch]),
        "eye_diameter_km": torch.stack([b["structure"]["eye_diameter_km"] for b in batch]),
        "symmetry": torch.stack([b["structure"]["symmetry"] for b in batch]),
        "circularity": torch.stack([b["structure"]["circularity"] for b in batch]),
    }

    classification = {
        "category": torch.stack([b["classification"]["category"] for b in batch]),
        "stage": torch.stack([b["classification"]["stage"] for b in batch]),
        "pattern": torch.stack([b["classification"]["pattern"] for b in batch]),
    }

    return {"image": images, "detection": detection, "structure": structure, "classification": classification}


class MultiTaskModel(nn.Module):
    """Wraps all trainable components for multi-task training."""

    def __init__(self, config: dict):
        super().__init__()

        encoder_config = config.get("encoder", {})
        fusion_config = config.get("fusion", {})

        channel_names = ["IR_TIR1", "IR_TIR2", "WV", "VIS", "SST", "WIND_SHEAR", "MOISTURE"]
        feature_dim = encoder_config.get("feature_dim", 768)

        from cyclone_ai.fusion.attention_fusion import AttentionGatedFusion
        from cyclone_ai.encoder.shared_encoder import SharedEncoder
        from cyclone_ai.heads.detection import DetectionHead
        from cyclone_ai.heads.structure import StructureHead
        from cyclone_ai.heads.temporal import TemporalBlock
        from cyclone_ai.heads.classification import ClassificationHead

        self.fusion = AttentionGatedFusion(
            source_names=channel_names,
            source_channels=config.get("source_encoder", {}).get("output_channels", 64),
            output_dim=fusion_config.get("embed_dim", 256),
        )
        self.encoder = SharedEncoder(
            backbone_name=encoder_config.get("backbone", "convnext_tiny"),
            pretrained=encoder_config.get("pretrained", True),
            input_channels=fusion_config.get("embed_dim", 256),
            feature_dim=feature_dim,
        )
        self.detection_head = DetectionHead(feature_dim=feature_dim)
        self.structure_head = StructureHead(feature_dim=feature_dim)
        self.temporal_block = TemporalBlock(input_dim=feature_dim, output_dim=512)
        self.classification_head = ClassificationHead(global_dim=feature_dim, temporal_dim=512)

    def forward(self, x):
        fused = self.fusion(x)
        features = self.encoder(fused)

        spatial = features["spatial"]
        global_feat = features["global"]

        det = self.detection_head(spatial)
        struct = self.structure_head(spatial)
        temporal = self.temporal_block(global_feat, is_sequence=False)
        classif = self.classification_head(global_feat, temporal)

        return det, struct, classif


def train(args):
    """Main training loop."""
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    training_config = config.get("training", {})
    batch_size = args.batch_size or training_config.get("batch_size", 8)
    epochs = args.epochs or training_config.get("epochs", 50)
    lr = args.lr or training_config.get("learning_rate", 1e-4)

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training on {device}, batch_size={batch_size}, epochs={epochs}")

    # Dataset
    dataset = SyntheticTrainingDataset(num_samples=args.num_samples, image_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    model = MultiTaskModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=training_config.get("weight_decay", 1e-5),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights
    loss_weights = {"detection": 1.0, "structure": 0.5, "classification": 1.0}

    # Mixed precision
    use_amp = training_config.get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri(training_config.get("mlflow_tracking_uri", "./mlruns"))
        mlflow.set_experiment("cyclone-training")
        mlflow.start_run(run_name=f"multitask-{time.strftime('%Y%m%d-%H%M%S')}")
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "device": str(device)})
        use_mlflow = True
    except ImportError:
        use_mlflow = False

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_losses = {"detection": 0, "structure": 0, "classification": 0, "total": 0}
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    det_pred, struct_pred, class_pred = model(images)
                    losses = _compute_all_losses(
                        det_pred, struct_pred, class_pred,
                        batch, device, model, loss_weights,
                    )
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                det_pred, struct_pred, class_pred = model(images)
                losses = _compute_all_losses(
                    det_pred, struct_pred, class_pred,
                    batch, device, model, loss_weights,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            num_batches += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {epoch_losses['total']:.4f} | "
            f"Det: {epoch_losses['detection']:.4f} | "
            f"Struct: {epoch_losses['structure']:.4f} | "
            f"Class: {epoch_losses['classification']:.4f}"
        )

        if use_mlflow:
            for k, v in epoch_losses.items():
                mlflow.log_metric(f"loss_{k}", v, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # Save best
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            save_dir = Path("checkpoints")
            save_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info(f"  → Saved best model (loss={best_loss:.4f})")

    # Final save
    torch.save(model.state_dict(), Path("checkpoints") / "final_model.pt")
    logger.info(f"Training complete! Best loss: {best_loss:.4f}")

    if use_mlflow:
        mlflow.end_run()


def _compute_all_losses(det_pred, struct_pred, class_pred, batch, device, model, weights):
    """Compute weighted multi-task loss."""
    # Move targets to device
    det_targets = {k: v.to(device) for k, v in batch["detection"].items()}
    struct_targets = {k: v.to(device) for k, v in batch["structure"].items()}
    class_targets = {k: v.to(device) for k, v in batch["classification"].items()}

    det_losses = model.detection_head.compute_loss(det_pred, det_targets)
    struct_losses = model.structure_head.compute_loss(struct_pred, struct_targets)
    class_losses = model.classification_head.compute_loss(class_pred, class_targets)

    total = (
        weights["detection"] * det_losses["total"]
        + weights["structure"] * struct_losses["total"]
        + weights["classification"] * class_losses["total"]
    )

    return {
        "detection": det_losses["total"],
        "structure": struct_losses["total"],
        "classification": class_losses["total"],
        "total": total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cyclone multi-task model")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=200, help="Synthetic training samples")
    args = parser.parse_args()
    train(args)
