"""
MinIO Storage Helper — Object storage for inference artifacts.

Handles uploading and serving mask images, Grad-CAM heatmaps,
and other binary artifacts via MinIO S3-compatible storage.
"""

import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MinIOStorage:
    """
    MinIO object storage client for cyclone prediction artifacts.

    Stores:
    - Segmentation masks (PNG)
    - Grad-CAM heatmaps (PNG)
    - Any other inference artifacts

    Falls back to local filesystem if MinIO is not available.

    Usage:
        storage = MinIOStorage()
        url = storage.upload_image(array, "inference/storm123/mask.png")
        # returns "minio://cyclone-artifacts/inference/storm123/mask.png"
    """

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket: str = "cyclone-artifacts",
        secure: bool = False,
        fallback_dir: str = "outputs/artifacts",
    ):
        self.endpoint = endpoint
        self.bucket = bucket
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._use_fallback = False

        try:
            from minio import Minio

            self._client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )

            # Ensure bucket exists
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info(f"Created MinIO bucket: {bucket}")

            logger.info(f"MinIO connected: {endpoint}/{bucket}")

        except Exception as e:
            logger.warning(
                f"MinIO not available ({e}). "
                f"Falling back to local storage at {self.fallback_dir}"
            )
            self._use_fallback = True

    def upload_image(
        self,
        image: np.ndarray,
        object_name: str,
        colormap: Optional[str] = None,
    ) -> str:
        """
        Upload a numpy array as a PNG image.

        Args:
            image: 2D array (grayscale) or 3D array (H, W, C) in RGB.
            object_name: S3 object key (path within bucket).
            colormap: Optional matplotlib colormap name to apply to 2D arrays.

        Returns:
            URI string: "minio://bucket/object_name" or "file:///local/path".
        """
        from PIL import Image

        # Apply colormap if needed
        if image.ndim == 2 and colormap:
            image = self._apply_colormap(image, colormap)
        elif image.ndim == 2:
            # Normalize to 0-255
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)

        pil_image = Image.fromarray(image.astype(np.uint8))

        if self._use_fallback:
            return self._save_local(pil_image, object_name)
        else:
            return self._upload_minio(pil_image, object_name)

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to storage."""
        if self._use_fallback:
            local_path = self.fallback_dir / object_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return f"file:///{local_path.as_posix()}"
        else:
            stream = io.BytesIO(data)
            self._client.put_object(
                self.bucket, object_name, stream,
                length=len(data), content_type=content_type,
            )
            return f"minio://{self.bucket}/{object_name}"

    def _upload_minio(self, pil_image, object_name: str) -> str:
        """Upload PIL Image to MinIO."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        self._client.put_object(
            self.bucket, object_name, buffer,
            length=buffer.getbuffer().nbytes,
            content_type="image/png",
        )
        uri = f"minio://{self.bucket}/{object_name}"
        logger.debug(f"Uploaded to MinIO: {uri}")
        return uri

    def _save_local(self, pil_image, object_name: str) -> str:
        """Save PIL Image to local filesystem."""
        local_path = self.fallback_dir / object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(str(local_path), format="PNG")
        uri = f"file:///{local_path.as_posix()}"
        logger.debug(f"Saved locally: {uri}")
        return uri

    @staticmethod
    def _apply_colormap(image: np.ndarray, colormap: str) -> np.ndarray:
        """Apply a matplotlib colormap to a 2D array."""
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
            # Normalize to [0, 1]
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(image)
            # Apply colormap → RGBA → RGB
            colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
            return colored
        except ImportError:
            logger.warning("matplotlib not available for colormap, using grayscale")
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            return np.zeros_like(image, dtype=np.uint8)
