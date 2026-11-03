"""
PyTorch Dataset & DataLoader — Stage 2.

Provides CycloneDataset (single-frame) and CycloneSequenceDataset (temporal)
for training the multi-head model and forecast models.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CycloneDataset(Dataset):
    """
    PyTorch Dataset for single-frame cyclone data.

    Each sample is a preprocessed, storm-centered crop with associated metadata.

    Args:
        crops: List of crop dicts from storm_cropper.create_storm_crops().
        transform: Optional callable for data augmentation.
    """

    def __init__(self, crops: list[dict], transform=None):
        self.crops = crops
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        crop = self.crops[idx]

        # Convert to tensor [C, H, W]
        image = torch.from_numpy(crop["data"]).float()

        # Metadata dict
        metadata = {
            "timestamp": crop.get("timestamp", ""),
            "center_lat": crop.get("center_lat", 0.0),
            "center_lon": crop.get("center_lon", 0.0),
            "wind_kt": crop.get("wind_kt", 0.0),
            "pressure_hpa": crop.get("pressure_hpa", 0.0),
            "category": crop.get("category", ""),
            "channel_names": crop.get("channel_names", []),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, metadata


class CycloneSequenceDataset(Dataset):
    """
    PyTorch Dataset for temporal sequences of cyclone data.

    Groups consecutive crops into fixed-length sequences for the
    temporal block (Stage 7) and forecast model (Stage 9).

    Args:
        crops: List of crop dicts, sorted chronologically within each storm.
        seq_len: Number of frames per sequence.
        stride: Step between sequence start positions.
    """

    def __init__(self, crops: list[dict], seq_len: int = 8, stride: int = 1):
        self.crops = crops
        self.seq_len = seq_len
        self.stride = stride

        # Build valid sequence start indices
        self.sequences = self._build_sequences()
        logger.info(
            f"CycloneSequenceDataset: {len(self.sequences)} sequences "
            f"(seq_len={seq_len}, stride={stride})"
        )

    def _build_sequences(self) -> list[list[int]]:
        """Identify valid contiguous sequences."""
        sequences = []
        n = len(self.crops)

        for start in range(0, n - self.seq_len + 1, self.stride):
            seq_indices = list(range(start, start + self.seq_len))
            sequences.append(seq_indices)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[dict]]:
        indices = self.sequences[idx]

        frames = []
        metadata_list = []

        for i in indices:
            crop = self.crops[i]
            frame = torch.from_numpy(crop["data"]).float()
            frames.append(frame)

            metadata_list.append({
                "timestamp": crop.get("timestamp", ""),
                "center_lat": crop.get("center_lat", 0.0),
                "center_lon": crop.get("center_lon", 0.0),
                "wind_kt": crop.get("wind_kt", 0.0),
                "pressure_hpa": crop.get("pressure_hpa", 0.0),
                "category": crop.get("category", ""),
            })

        # Stack frames [T, C, H, W]
        sequence = torch.stack(frames, dim=0)
        return sequence, metadata_list


def create_dataloader(
    crops: list[dict],
    batch_size: int = 8,
    seq_len: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Factory function for creating DataLoaders.

    Args:
        crops: List of crop dicts.
        batch_size: Batch size.
        seq_len: If provided, creates a CycloneSequenceDataset.
                 If None, creates a CycloneDataset (single-frame).
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for CUDA transfers.

    Returns:
        PyTorch DataLoader.
    """
    if seq_len is not None:
        dataset = CycloneSequenceDataset(crops, seq_len=seq_len)
    else:
        dataset = CycloneDataset(crops)

    # Custom collate for metadata dicts
    if seq_len is not None:
        collate_fn = _collate_sequences
    else:
        collate_fn = _collate_frames

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info(
        f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}, "
        f"type={'sequence' if seq_len else 'frame'}"
    )
    return loader


def _collate_frames(batch):
    """Custom collate for single-frame samples."""
    images = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return images, metadata


def _collate_sequences(batch):
    """Custom collate for sequence samples."""
    sequences = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return sequences, metadata


def split_crops(
    crops: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split crops into train/val/test sets.

    Uses random shuffling with a fixed seed for reproducibility.

    Returns:
        Tuple of (train_crops, val_crops, test_crops).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(crops))

    n_train = int(len(crops) * train_frac)
    n_val = int(len(crops) * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_crops = [crops[i] for i in train_idx]
    val_crops = [crops[i] for i in val_idx]
    test_crops = [crops[i] for i in test_idx]

    logger.info(
        f"Split: train={len(train_crops)}, val={len(val_crops)}, test={len(test_crops)}"
    )
    return train_crops, val_crops, test_crops
