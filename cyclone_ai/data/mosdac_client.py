"""
MOSDAC Client — INSAT-3D/3DR/3DS satellite data acquisition.

Stage 1: Data Acquisition (satellite imagery component).

When data_mode is "synthetic", generates realistic INSAT-like satellite imagery
with Gaussian-blob cyclone signatures. When "real", downloads from MOSDAC API.
The interface is identical in both modes so downstream stages are unaffected.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SatelliteProduct:
    """Metadata for a single satellite product/channel."""

    channel_name: str
    wavelength_um: float
    data: Optional[np.ndarray] = None
    filepath: Optional[str] = None
    timestamp: Optional[str] = None
    lat_range: tuple = (0.0, 30.0)
    lon_range: tuple = (60.0, 100.0)
    resolution_deg: float = 0.05


@dataclass
class SatelliteScene:
    """A complete multi-channel satellite scene for a given timestamp."""

    storm_id: str
    timestamp: str
    products: dict = field(default_factory=dict)  # channel_name → SatelliteProduct
    center_lat: float = 0.0
    center_lon: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Stack all channels into a single [C, H, W] array."""
        arrays = []
        for name in sorted(self.products.keys()):
            if self.products[name].data is not None:
                arrays.append(self.products[name].data)
        if not arrays:
            raise ValueError("No product data loaded in scene")
        return np.stack(arrays, axis=0).astype(np.float32)


class SyntheticINSATGenerator:
    """
    Generates synthetic INSAT-like satellite imagery with realistic cyclone signatures.

    The generator creates multi-channel images (IR, WV, VIS) with:
    - Background radiance gradient (latitude-dependent)
    - Gaussian-blob cyclone vortex with spiral arm structure
    - Optional eye feature (clear center in mature storms)
    - Channel-appropriate value ranges and noise
    """

    # Realistic value ranges for each channel (in brightness temperature / reflectance)
    CHANNEL_PARAMS = {
        "IR_TIR1": {
            "wavelength_um": 10.8,
            "bg_mean": 280.0,       # K — warm ocean background
            "cyclone_delta": -50.0,  # K — cold cloud tops
            "eye_delta": 30.0,       # K — warm eye (partial clearing)
            "noise_std": 2.0,
        },
        "IR_TIR2": {
            "wavelength_um": 12.0,
            "bg_mean": 275.0,
            "cyclone_delta": -45.0,
            "eye_delta": 25.0,
            "noise_std": 2.0,
        },
        "WV": {
            "wavelength_um": 6.8,
            "bg_mean": 240.0,       # K — dry upper tropo
            "cyclone_delta": -30.0,  # K — moist upper tropo over cyclone
            "eye_delta": 15.0,
            "noise_std": 3.0,
        },
        "VIS": {
            "wavelength_um": 0.65,
            "bg_mean": 0.05,        # reflectance — dark ocean
            "cyclone_delta": 0.7,   # reflectance — bright cloud tops
            "eye_delta": -0.4,      # reflectance — darker eye
            "noise_std": 0.02,
        },
    }

    def __init__(self, image_size: int = 512, seed: Optional[int] = None):
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def generate_scene(
        self,
        storm_id: str,
        timestamp: str,
        center_lat: float,
        center_lon: float,
        intensity_factor: float = 1.0,
        has_eye: bool = False,
        eye_radius_frac: float = 0.03,
        cyclone_radius_frac: float = 0.25,
    ) -> SatelliteScene:
        """
        Generate a full multi-channel synthetic satellite scene.

        Args:
            storm_id: Storm identifier.
            timestamp: ISO8601 timestamp.
            center_lat: Cyclone center latitude.
            center_lon: Cyclone center longitude.
            intensity_factor: Scale factor for cyclone signature (0–2). Higher = stronger.
            has_eye: Whether to generate an eye feature.
            eye_radius_frac: Eye radius as fraction of image size.
            cyclone_radius_frac: Cyclone outer radius as fraction of image size.

        Returns:
            SatelliteScene with all channels populated.
        """
        scene = SatelliteScene(
            storm_id=storm_id,
            timestamp=timestamp,
            center_lat=center_lat,
            center_lon=center_lon,
        )

        # Generate coordinate grids
        y = np.linspace(-1, 1, self.image_size)
        x = np.linspace(-1, 1, self.image_size)
        xx, yy = np.meshgrid(x, y)

        # Offset center slightly from image center for realism
        cx_offset = self.rng.uniform(-0.05, 0.05)
        cy_offset = self.rng.uniform(-0.05, 0.05)
        r = np.sqrt((xx - cx_offset) ** 2 + (yy - cy_offset) ** 2)

        # Cyclone radial profile (Gaussian envelope)
        sigma = cyclone_radius_frac * 2
        cyclone_mask = np.exp(-(r ** 2) / (2 * sigma ** 2))

        # Add spiral arm structure
        theta = np.arctan2(yy - cy_offset, xx - cx_offset)
        spiral = 0.3 * np.sin(2 * theta + 5 * r) * cyclone_mask
        cyclone_mask = np.clip(cyclone_mask + spiral * intensity_factor, 0, 1)

        # Eye mask (if present)
        eye_mask = np.zeros_like(r)
        if has_eye:
            eye_sigma = eye_radius_frac * 2
            eye_mask = np.exp(-(r ** 2) / (2 * eye_sigma ** 2))

        # Generate each channel
        for ch_name, params in self.CHANNEL_PARAMS.items():
            img = self._generate_channel(
                ch_name, params, cyclone_mask, eye_mask,
                intensity_factor, has_eye,
            )
            scene.products[ch_name] = SatelliteProduct(
                channel_name=ch_name,
                wavelength_um=params["wavelength_um"],
                data=img,
                timestamp=timestamp,
            )

        return scene

    def _generate_channel(
        self,
        channel_name: str,
        params: dict,
        cyclone_mask: np.ndarray,
        eye_mask: np.ndarray,
        intensity_factor: float,
        has_eye: bool,
    ) -> np.ndarray:
        """Generate a single channel image."""
        H = W = self.image_size

        # Background with latitude gradient
        bg_gradient = np.linspace(0.95, 1.05, H).reshape(-1, 1) * np.ones((1, W))
        background = params["bg_mean"] * bg_gradient

        # Apply cyclone signature
        img = background + params["cyclone_delta"] * cyclone_mask * intensity_factor

        # Apply eye if present
        if has_eye:
            img += params["eye_delta"] * eye_mask * intensity_factor

        # Add noise
        noise = self.rng.normal(0, params["noise_std"], (H, W))
        img += noise

        return img.astype(np.float32)


class MOSDACClient:
    """
    Client for MOSDAC satellite data acquisition.

    Supports two modes:
    - "synthetic": Generates realistic synthetic INSAT-like data (default)
    - "real": Downloads actual INSAT-3D/3DR/3DS data from MOSDAC API (requires credentials)

    Usage:
        client = MOSDACClient(config_path="configs/data_sources.yaml")
        scene = client.get_scene("AMPHAN_2020", "2020-05-17T06:00:00Z",
                                  center_lat=16.5, center_lon=87.0)
        numpy_data = scene.to_numpy()  # shape: [C, H, W]
    """

    def __init__(self, config_path: str = "configs/data_sources.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.data_mode = config.get("data_mode", "synthetic")
        self.mosdac_config = config.get("mosdac", {})
        self.synthetic_config = config.get("synthetic", {})
        self.download_dir = Path(self.mosdac_config.get("download_dir", "data/raw/mosdac"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.image_size = self.synthetic_config.get("image_size", 512)
        self._generator = SyntheticINSATGenerator(image_size=self.image_size)

        if self.data_mode == "synthetic":
            logger.info("MOSDACClient initialized in SYNTHETIC mode")
        else:
            logger.info("MOSDACClient initialized in REAL mode (MOSDAC API)")
            self._validate_credentials()

    def _validate_credentials(self):
        """Check that MOSDAC credentials are configured in environment or config."""
        username = os.environ.get("MOSDAC_USERNAME") or self.mosdac_config.get("username", "")
        password = os.environ.get("MOSDAC_PASSWORD") or self.mosdac_config.get("password", "")
        if not username or not password:
            raise ValueError(
                "MOSDAC credentials required for real data mode. "
                "Set MOSDAC_USERNAME / MOSDAC_PASSWORD environment variables, "
                "or configure in configs/data_sources.yaml, or switch data_mode to 'synthetic'."
            )

    def get_scene(
        self,
        storm_id: str,
        timestamp: str,
        center_lat: float = 15.0,
        center_lon: float = 85.0,
        intensity_factor: float = 1.0,
        has_eye: bool = False,
    ) -> SatelliteScene:
        """
        Fetch a satellite scene for the given storm and timestamp.

        Args:
            storm_id: Storm identifier (e.g., "2020121N10087").
            timestamp: ISO8601 timestamp.
            center_lat: Cyclone center latitude (used for cropping in real mode,
                        center placement in synthetic mode).
            center_lon: Cyclone center longitude.
            intensity_factor: For synthetic mode — controls cyclone signature strength.
            has_eye: For synthetic mode — whether to include eye feature.

        Returns:
            SatelliteScene with all available channels.
        """
        if self.data_mode == "synthetic":
            return self._get_synthetic_scene(
                storm_id, timestamp, center_lat, center_lon,
                intensity_factor, has_eye,
            )
        else:
            return self._get_real_scene(storm_id, timestamp, center_lat, center_lon)

    def _get_synthetic_scene(
        self,
        storm_id: str,
        timestamp: str,
        center_lat: float,
        center_lon: float,
        intensity_factor: float,
        has_eye: bool,
    ) -> SatelliteScene:
        """Generate synthetic satellite scene."""
        scene = self._generator.generate_scene(
            storm_id=storm_id,
            timestamp=timestamp,
            center_lat=center_lat,
            center_lon=center_lon,
            intensity_factor=intensity_factor,
            has_eye=has_eye,
        )
        logger.info(
            f"Generated synthetic scene for {storm_id} at {timestamp} "
            f"(center: {center_lat:.1f}°N, {center_lon:.1f}°E)"
        )
        return scene

    def _get_real_scene(
        self,
        storm_id: str,
        timestamp: str,
        center_lat: float,
        center_lon: float,
    ) -> SatelliteScene:
        """
        Download real INSAT-3D/3DR data from MOSDAC.

        Flow:
        1. Authenticate with MOSDAC portal via session login
        2. Search catalog for each product (TIR1, TIR2, WV, VIS) near timestamp
        3. Download HDF5 file
        4. Parse with h5py → extract 2D brightness temp / reflectance array
        5. Crop to storm-centered region
        Falls back to synthetic if any step fails.
        """
        import requests
        import h5py
        import io
        from datetime import datetime, timedelta

        scene = SatelliteScene(
            storm_id=storm_id,
            timestamp=timestamp,
            center_lat=center_lat,
            center_lon=center_lon,
        )

        username = os.environ.get("MOSDAC_USERNAME") or self.mosdac_config.get("username", "")
        password = os.environ.get("MOSDAC_PASSWORD") or self.mosdac_config.get("password", "")
        api_endpoint = self.mosdac_config.get("api_endpoint", "https://mosdac.gov.in/apidata")
        products = self.mosdac_config.get("products", {})

        # Parse target timestamp
        ts_clean = timestamp.replace("Z", "+00:00")
        try:
            target_dt = datetime.fromisoformat(ts_clean)
        except ValueError:
            target_dt = datetime.strptime(timestamp.split("+")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")

        # Time window: ±3 hours around target
        start_dt = target_dt - timedelta(hours=3)
        end_dt = target_dt + timedelta(hours=3)
        date_str = target_dt.strftime("%Y-%m-%d")
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Channel name mapping from config keys to our internal names
        config_to_channel = {
            "ir_tir1": "IR_TIR1",
            "ir_tir2": "IR_TIR2",
            "wv": "WV",
            "vis": "VIS",
        }

        # HDF5 dataset keys commonly found in INSAT-3D L1B files
        hdf5_dataset_keys = [
            "IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_VIS",
            "Brightness_Temperature", "Albedo", "Radiance",
            "IMG_MIR", "Count",
        ]

        session = requests.Session()
        logged_in = False

        # Step 1: Authenticate
        try:
            login_url = "https://mosdac.gov.in/accounts/login/"
            login_page = session.get(login_url, timeout=15)

            # Extract CSRF token if present
            csrf_token = ""
            if "csrfmiddlewaretoken" in login_page.text:
                import re
                match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', login_page.text)
                if match:
                    csrf_token = match.group(1)

            login_data = {
                "username": username,
                "password": password,
                "csrfmiddlewaretoken": csrf_token,
            }
            headers = {"Referer": login_url}
            resp = session.post(login_url, data=login_data, headers=headers, timeout=15)
            logged_in = resp.status_code == 200 and "logout" in resp.text.lower()

            if logged_in:
                logger.info("MOSDAC login successful")
            else:
                logger.warning("MOSDAC login may have failed — will attempt download anyway")
        except Exception as e:
            logger.warning(f"MOSDAC login error: {e} — trying unauthenticated access")

        # Step 2: For each product, search + download
        channels_found = 0
        for config_key, product_info in products.items():
            channel_name = config_to_channel.get(config_key)
            if not channel_name:
                continue

            dataset_id = product_info.get("dataset_id", "")
            try:
                # Search API for files matching dataset + date
                search_url = f"{api_endpoint}/search"
                params = {
                    "datasetId": dataset_id,
                    "startDate": start_str,
                    "endDate": end_str,
                    "geoRange": f"{center_lat-5},{center_lon-5},{center_lat+5},{center_lon+5}",
                }
                resp = session.get(search_url, params=params, timeout=30)

                file_url = None
                if resp.status_code == 200:
                    try:
                        results = resp.json()
                        if isinstance(results, list) and len(results) > 0:
                            file_url = results[0].get("fileUrl") or results[0].get("url")
                        elif isinstance(results, dict):
                            files = results.get("files", results.get("data", []))
                            if files:
                                file_url = files[0].get("fileUrl") or files[0].get("url")
                    except Exception:
                        pass

                # Fallback: try direct catalog URL pattern
                if not file_url:
                    # MOSDAC typical URL pattern for INSAT-3D L1B
                    date_path = target_dt.strftime("%d%b%Y").upper()
                    file_url = (
                        f"https://mosdac.gov.in/data/archive/"
                        f"{dataset_id}/{target_dt.year}/{date_path}/"
                    )

                if file_url:
                    # Download the HDF5 file
                    dl_resp = session.get(file_url, timeout=60, stream=True)
                    if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                        # Save locally
                        save_dir = self.download_dir / dataset_id / date_str
                        save_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"{dataset_id}_{target_dt.strftime('%Y%m%d_%H%M')}.h5"
                        save_path = save_dir / fname
                        save_path.write_bytes(dl_resp.content)

                        # Parse HDF5
                        data_array = self._parse_hdf5(save_path, channel_name, hdf5_dataset_keys)
                        if data_array is not None:
                            scene.products[channel_name] = SatelliteProduct(
                                channel_name=channel_name,
                                wavelength_um=self._get_wavelength(channel_name),
                                data=data_array,
                                filepath=str(save_path),
                                timestamp=timestamp,
                            )
                            channels_found += 1
                            logger.info(f"Downloaded {channel_name} from MOSDAC ({save_path.name})")
                            continue

                logger.warning(f"Could not download {channel_name} ({dataset_id}) — using synthetic")

            except Exception as e:
                logger.warning(f"Error downloading {channel_name}: {e} — using synthetic")

            # Fallback: generate synthetic for this channel
            if channel_name not in scene.products:
                synth = self._generator.generate_scene(
                    storm_id, timestamp, center_lat, center_lon,
                    intensity_factor=1.0, has_eye=False,
                )
                if channel_name in synth.products:
                    scene.products[channel_name] = synth.products[channel_name]
                    scene.products[channel_name].filepath = "synthetic"

        if channels_found > 0:
            logger.info(f"MOSDAC scene: {channels_found}/4 real channels loaded")
        else:
            logger.warning("No real MOSDAC data obtained — full synthetic fallback")
            return self._get_synthetic_scene(
                storm_id, timestamp, center_lat, center_lon, 1.0, False,
            )

        return scene

    def _parse_hdf5(self, filepath, channel_name: str, dataset_keys: list) -> "np.ndarray | None":
        """Parse an INSAT-3D HDF5 file and extract the 2D data array."""
        import h5py

        try:
            with h5py.File(filepath, "r") as f:
                # Try known dataset keys
                for key in dataset_keys:
                    if key in f:
                        raw = f[key][:]
                        if raw.ndim == 2:
                            data = raw.astype(np.float32)
                        elif raw.ndim == 3:
                            data = raw[0].astype(np.float32)  # take first band
                        else:
                            continue

                        # Resize to target image_size
                        if data.shape[0] != self.image_size or data.shape[1] != self.image_size:
                            from scipy.ndimage import zoom
                            zh = self.image_size / data.shape[0]
                            zw = self.image_size / data.shape[1]
                            data = zoom(data, (zh, zw), order=1).astype(np.float32)

                        # Replace fill values with NaN
                        data[data < -9990] = np.nan
                        data[data > 1e10] = np.nan

                        logger.debug(f"Parsed {channel_name} from {filepath.name}: {data.shape}")
                        return data

                # If no known key, try first 2D dataset
                def _find_2d(group):
                    for k in group:
                        item = group[k]
                        if hasattr(item, "shape") and len(item.shape) >= 2:
                            return item[:]
                        elif hasattr(item, "keys"):
                            result = _find_2d(item)
                            if result is not None:
                                return result
                    return None

                raw = _find_2d(f)
                if raw is not None:
                    data = raw.astype(np.float32)
                    if data.ndim == 3:
                        data = data[0]
                    from scipy.ndimage import zoom
                    zh = self.image_size / data.shape[0]
                    zw = self.image_size / data.shape[1]
                    data = zoom(data, (zh, zw), order=1).astype(np.float32)
                    data[data < -9990] = np.nan
                    data[data > 1e10] = np.nan
                    return data

        except Exception as e:
            logger.warning(f"Failed to parse HDF5 {filepath}: {e}")

        return None

    @staticmethod
    def _get_wavelength(channel_name: str) -> float:
        """Return wavelength in μm for a channel name."""
        wavelengths = {"IR_TIR1": 10.8, "IR_TIR2": 12.0, "WV": 6.8, "VIS": 0.65}
        return wavelengths.get(channel_name, 0.0)

    def get_scene_sequence(
        self,
        storm_id: str,
        timestamps: list[str],
        centers: list[tuple[float, float]],
        intensity_factors: Optional[list[float]] = None,
        eye_flags: Optional[list[bool]] = None,
    ) -> list[SatelliteScene]:
        """
        Fetch a temporal sequence of satellite scenes.

        Args:
            storm_id: Storm identifier.
            timestamps: List of ISO8601 timestamps.
            centers: List of (lat, lon) tuples for each timestamp.
            intensity_factors: Optional intensity scaling per timestamp.
            eye_flags: Optional eye presence per timestamp.

        Returns:
            List of SatelliteScene objects in temporal order.
        """
        n = len(timestamps)
        if intensity_factors is None:
            intensity_factors = [1.0] * n
        if eye_flags is None:
            eye_flags = [False] * n

        scenes = []
        for i, (ts, (lat, lon)) in enumerate(zip(timestamps, centers)):
            scene = self.get_scene(
                storm_id=storm_id,
                timestamp=ts,
                center_lat=lat,
                center_lon=lon,
                intensity_factor=intensity_factors[i],
                has_eye=eye_flags[i],
            )
            scenes.append(scene)

        logger.info(f"Generated sequence of {n} scenes for {storm_id}")
        return scenes
