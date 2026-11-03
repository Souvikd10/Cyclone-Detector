"""
Pipeline Orchestrator — Wires stages 1–13 together.

The CyclonePipeline class manages model loading, device placement,
and sequential execution of all pipeline stages.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class CyclonePipeline:
    """
    End-to-end cyclone prediction pipeline.

    Orchestrates all 13 stages from raw data to structured prediction JSON.

    Usage:
        pipeline = CyclonePipeline(config_dir="configs")
        result = pipeline.run(storm_id="2020121N10087", timestamp="2020-05-17T06:00:00Z")
    """

    def __init__(
        self,
        config_dir: str = "configs",
        device: Optional[str] = None,
    ):
        self.config_dir = Path(config_dir)

        # Load configs
        self.data_config = self._load_yaml("data_sources.yaml")
        self.model_config = self._load_yaml("model.yaml")
        self.inference_config = self._load_yaml("inference.yaml")

        # Determine device
        if device is None:
            device = self.inference_config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Pipeline device: {self.device}")

        # Initialize all components
        self._init_data_clients()
        self._init_preprocessing()
        self._init_models()
        self._init_post_processing()
        self._init_storage()

    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML config file."""
        path = self.config_dir / filename
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Config file not found: {path}")
        return {}

    # ══════════════════════════════════════════════════════════════════
    # Initialization
    # ══════════════════════════════════════════════════════════════════

    def _init_data_clients(self):
        """Initialize data acquisition clients (Stage 1)."""
        from cyclone_ai.data.mosdac_client import MOSDACClient
        from cyclone_ai.data.ibtracs_client import IBTrACSClient
        from cyclone_ai.data.era5_client import ERA5Client

        config_path = str(self.config_dir / "data_sources.yaml")
        self.mosdac = MOSDACClient(config_path=config_path)
        self.ibtracs = IBTrACSClient(config_path=config_path)
        self.era5 = ERA5Client(config_path=config_path)

    def _init_preprocessing(self):
        """Initialize preprocessing modules (Stage 2)."""
        from cyclone_ai.data.preprocessing import Normalizer, MissingDataHandler
        from cyclone_ai.data.storm_cropper import StormCropper

        self.normalizer = Normalizer()
        self.missing_handler = MissingDataHandler()
        self.cropper = StormCropper(crop_size=512, field_of_view_deg=10.0)

        self.channel_names = self.data_config.get("synthetic", {}).get(
            "channels",
            ["IR_TIR1", "IR_TIR2", "WV", "VIS", "SST", "WIND_SHEAR", "MOISTURE"],
        )

    def _init_models(self):
        """Initialize all neural network models (Stages 3–10)."""
        encoder_config = self.model_config.get("encoder", {})
        fusion_config = self.model_config.get("fusion", {})
        temporal_config = self.model_config.get("temporal", {})
        forecast_config = self.model_config.get("forecast", {})

        # Stage 3: Fusion
        from cyclone_ai.fusion.attention_fusion import AttentionGatedFusion
        self.fusion = AttentionGatedFusion(
            source_names=self.channel_names,
            source_channels=self.model_config.get("source_encoder", {}).get("output_channels", 64),
            output_dim=fusion_config.get("embed_dim", 256),
            num_heads=fusion_config.get("num_heads", 4),
        ).to(self.device)

        # Stage 4: Encoder
        from cyclone_ai.encoder.shared_encoder import SharedEncoder
        self.encoder = SharedEncoder(
            backbone_name=encoder_config.get("backbone", "convnext_tiny"),
            pretrained=encoder_config.get("pretrained", True),
            input_channels=fusion_config.get("embed_dim", 256),
            feature_dim=encoder_config.get("feature_dim", 768),
        ).to(self.device)

        feature_dim = encoder_config.get("feature_dim", 768)

        # Stage 5: Detection
        from cyclone_ai.heads.detection import DetectionHead
        self.detection_head = DetectionHead(feature_dim=feature_dim).to(self.device)

        # Stage 6: Structure
        from cyclone_ai.heads.structure import StructureHead
        self.structure_head = StructureHead(feature_dim=feature_dim).to(self.device)

        # Stage 7: Temporal
        from cyclone_ai.heads.temporal import TemporalBlock
        temporal_output_dim = 512
        self.temporal_block = TemporalBlock(
            input_dim=feature_dim,
            output_dim=temporal_output_dim,
            model_type=temporal_config.get("model_type", "convlstm"),
            convlstm_hidden=temporal_config.get("convlstm_hidden", 256),
            convlstm_layers=temporal_config.get("convlstm_layers", 2),
        ).to(self.device)

        # Stage 8: Classification
        from cyclone_ai.heads.classification import ClassificationHead
        self.classification_head = ClassificationHead(
            global_dim=feature_dim,
            temporal_dim=temporal_output_dim,
        ).to(self.device)

        # Stage 9: Forecast
        from cyclone_ai.forecast.track_intensity import ForecastModel
        self.forecast_model = ForecastModel(
            temporal_dim=temporal_output_dim,
            env_dim=forecast_config.get("env_scalar_dim", 8),
            horizons=forecast_config.get("horizons_h", [6, 12, 24, 48, 72]),
        ).to(self.device)

        # Stage 10: RI
        from cyclone_ai.forecast.ri_model import RIModel
        self.ri_model = RIModel()

        # Load trained weights if checkpoint exists
        self._load_trained_weights()

        logger.info("All models initialized (stages 3–10)")

    def _load_trained_weights(self, weights_path: Optional[str] = None):
        """Load trained neural network checkpoint if available."""
        if weights_path is None:
            for candidate in [
                Path("checkpoints/best_model.pt"),
                self.config_dir.parent / "checkpoints" / "best_model.pt",
            ]:
                if candidate.exists():
                    weights_path = str(candidate)
                    break

        if not weights_path or not Path(weights_path).exists():
            logger.info("No trained checkpoint found — running with base initialized weights")
            return

        try:
            ckpt = torch.load(weights_path, map_location=self.device)
            if isinstance(ckpt, dict):
                module_map = {
                    "fusion": self.fusion,
                    "encoder": self.encoder,
                    "detection_head": self.detection_head,
                    "structure_head": self.structure_head,
                    "temporal_block": self.temporal_block,
                    "classification_head": self.classification_head,
                }
                for prefix, mod in module_map.items():
                    sub_dict = {
                        k[len(prefix) + 1:]: v
                        for k, v in ckpt.items()
                        if k.startswith(f"{prefix}.")
                    }
                    if sub_dict:
                        try:
                            # Filter out shape mismatches between backbones
                            model_state = mod.state_dict()
                            filtered_sub_dict = {
                                k: v for k, v in sub_dict.items()
                                if k in model_state and model_state[k].shape == v.shape
                            }
                            mod.load_state_dict(filtered_sub_dict, strict=False)
                            logger.info(f"Loaded trained weights for {prefix} ({len(filtered_sub_dict)}/{len(sub_dict)} layers)")
                        except Exception as mod_err:
                            logger.debug(f"Could not load all layers for {prefix}: {mod_err}")
                logger.info(f"Loaded trained checkpoint: {weights_path}")
        except Exception as e:
            logger.warning(f"Failed loading checkpoint {weights_path}: {e}")

    def _init_post_processing(self):
        """Initialize post-processing modules (Stages 11–13)."""
        physics_config = self.model_config.get("physics", {})

        # Stage 11: Physics correction
        from cyclone_ai.forecast.physics_correction import PhysicsCorrector
        self.physics_corrector = PhysicsCorrector(
            max_track_speed_kt=physics_config.get("max_track_speed_kt", 30),
            max_intensity_change_kt=physics_config.get("max_intensity_change_kt", 40),
            sst_threshold_c=physics_config.get("sst_threshold_c", 26.5),
            landfall_decay_rate=physics_config.get("landfall_decay_rate", 0.9),
        )

        # Stage 12: Uncertainty
        from cyclone_ai.uncertainty.ood_estimation import UncertaintyEstimator
        uncertainty_config = self.inference_config.get("uncertainty", {})
        self.uncertainty_estimator = UncertaintyEstimator(
            novelty_threshold=uncertainty_config.get("novelty_threshold", 3.0),
            ri_ambiguity_range=tuple(uncertainty_config.get("ri_ambiguity_range", [0.3, 0.7])),
        )

        # Stage 13: XAI
        from cyclone_ai.xai.gradcam import GradCAMGenerator
        from cyclone_ai.xai.explanations import ExplanationGenerator
        self.gradcam = GradCAMGenerator()  # model wiring done after init
        self.explanation_gen = ExplanationGenerator()

    def _init_storage(self):
        """Initialize artifact storage."""
        from cyclone_ai.storage import MinIOStorage

        minio_config = self.inference_config.get("minio", {})
        self.storage = MinIOStorage(
            endpoint=minio_config.get("endpoint", "localhost:9000"),
            access_key=minio_config.get("access_key", "minioadmin"),
            secret_key=minio_config.get("secret_key", "minioadmin"),
            bucket=minio_config.get("bucket", "cyclone-artifacts"),
            secure=minio_config.get("secure", False),
        )

    # ══════════════════════════════════════════════════════════════════
    # Main Pipeline
    # ══════════════════════════════════════════════════════════════════

    def run(self, storm_id: str, timestamp: str) -> dict:
        """
        Run the complete 13-stage pipeline.

        Args:
            storm_id: Storm identifier (IBTrACS SID or custom).
            timestamp: ISO8601 timestamp.

        Returns:
            Prediction dict conforming to the output contract schema.
        """
        logger.info(f"Pipeline run: storm={storm_id}, time={timestamp}")

        # ── Stage 1: Data Acquisition ──
        track = self.ibtracs.get_storm(storm_id)
        track_point = self._find_closest_point(track, timestamp)

        scene = self.mosdac.get_scene(
            storm_id=storm_id,
            timestamp=timestamp,
            center_lat=track_point.lat,
            center_lon=track_point.lon,
            intensity_factor=track_point.wind_kt / 80.0,
            has_eye=track_point.wind_kt >= 64,
        )

        env = self.era5.get_snapshot(
            storm_id=storm_id,
            timestamp=timestamp,
            center_lat=track_point.lat,
            center_lon=track_point.lon,
        )

        # ── Stage 2: Preprocessing ──
        sat_data = scene.to_numpy()
        env_data = env.to_numpy()

        if sat_data.shape[1:] != env_data.shape[1:]:
            from cyclone_ai.data.preprocessing import GeoAligner
            aligner = GeoAligner()
            env_data = aligner.regrid(
                env_data,
                source_lat_range=(0, 30), source_lon_range=(60, 100),
                target_lat_range=(0, 30), target_lon_range=(60, 100),
                target_size=sat_data.shape[1:],
            )

        combined = np.concatenate([sat_data, env_data], axis=0)
        cleaned, quality = self.missing_handler.handle(combined, self.channel_names)
        crop = self.cropper.crop(cleaned, center_lat=track_point.lat, center_lon=track_point.lon)
        normalized = self.normalizer.fit_transform(crop, self.channel_names)

        # Crop metadata for coordinate conversion
        crop_metadata = {
            "center_lat": track_point.lat,
            "center_lon": track_point.lon,
            "patch_lat_range": (track_point.lat - 5.0, track_point.lat + 5.0),
            "patch_lon_range": (track_point.lon - 5.0, track_point.lon + 5.0),
        }

        # ── Stages 3–4: Fusion + Encoder ──
        input_tensor = torch.from_numpy(normalized).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            self.fusion.eval()
            self.encoder.eval()
            fused = self.fusion(input_tensor)
            features = self.encoder(fused)

        spatial_features = features["spatial"]   # [1, 768, H', W']
        global_features = features["global"]      # [1, 768]

        # ── Stage 5: Detection ──
        with torch.no_grad():
            self.detection_head.eval()
            detection_preds = self.detection_head(spatial_features)

        # Upload mask
        mask_np = detection_preds["mask"][0, 0].cpu().numpy()
        ts_safe = timestamp.replace(":", "-")
        mask_url = self.storage.upload_image(
            mask_np, f"inference/{storm_id}/{ts_safe}/mask.png"
        )
        detection_contract = self.detection_head.to_contract(detection_preds, crop_metadata, mask_url)

        # ── Stage 6: Structure ──
        with torch.no_grad():
            self.structure_head.eval()
            structure_preds = self.structure_head(spatial_features)
        structure_contract = self.structure_head.to_contract(structure_preds, crop_metadata)

        # ── Stage 7: Temporal (single-frame fallback) ──
        with torch.no_grad():
            self.temporal_block.eval()
            temporal_features = self.temporal_block(global_features, is_sequence=False)

        # ── Stage 8: Classification ──
        with torch.no_grad():
            self.classification_head.eval()
            class_preds = self.classification_head(global_features, temporal_features)
        classification_contract = self.classification_head.to_contract(class_preds)

        # ── Stage 9: Forecast ──
        env_scalars = env.get_scalar_features(track_point.lat, track_point.lon)
        env_scalar_tensor = self._build_env_tensor(env_scalars)
        current_pos = torch.tensor(
            [[track_point.lat, track_point.lon]], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            self.forecast_model.eval()
            forecast_preds = self.forecast_model(temporal_features, env_scalar_tensor, current_pos)
        forecast_contract = self.forecast_model.to_contract(forecast_preds)

        # ── Stage 10: RI ──
        ri_features = self.ri_model.feature_extractor.extract(
            wind_kt=track_point.wind_kt,
            pressure_hpa=track_point.pressure_hpa,
            lat=track_point.lat,
            lon=track_point.lon,
            env_scalars=env_scalars,
            structure=structure_contract,
        )
        ri_contract = self.ri_model.to_contract(ri_features)

        # ── Stage 11: Physics Correction ──
        sst_value = env_scalars.get("SST_mean", 28.0)
        corrected_forecast, corrected_ri, corrections = self.physics_corrector.correct(
            forecast=forecast_contract,
            ri=ri_contract,
            current_wind_kt=track_point.wind_kt,
            current_lat=track_point.lat,
            current_lon=track_point.lon,
            sst=sst_value,
        )

        # ── Stage 12: Uncertainty ──
        encoder_features_np = global_features[0].cpu().numpy()
        uncertainty_contract = self.uncertainty_estimator.estimate(
            encoder_features=encoder_features_np,
            ri_probability=corrected_ri["probability"],
            forecast=corrected_forecast,
            corrections_applied=corrections,
        )

        # Scale forecast cones
        corrected_forecast = self.uncertainty_estimator.scale_forecast_cones(corrected_forecast)

        # ── Stage 13: XAI ──
        gradcam_heatmap = self.gradcam.generate(input_tensor)
        gradcam_url = self.storage.upload_image(
            gradcam_heatmap,
            f"inference/{storm_id}/{ts_safe}/gradcam.png",
            colormap="jet",
        )

        explanation_tags = self.explanation_gen.generate(
            structure=structure_contract,
            classification=classification_contract,
            ri=corrected_ri,
            uncertainty=uncertainty_contract,
            env_scalars=env_scalars,
            forecast=corrected_forecast,
        )

        # ── Build final output ──
        result = {
            "storm_id": storm_id,
            "timestamp": timestamp,
            "detection": detection_contract,
            "structure": structure_contract,
            "current_state": classification_contract,
            "forecast": corrected_forecast,
            "ri": corrected_ri,
            "uncertainty": uncertainty_contract,
            "xai": {
                "gradcam_url": gradcam_url,
                "explanation_tags": explanation_tags,
            },
        }

        return result

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    def _find_closest_point(self, track, timestamp: str):
        """Find the track point closest to the given timestamp."""
        from datetime import datetime

        target = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        best_point = track.points[0]
        best_diff = float("inf")

        for point in track.points:
            pt_time = datetime.fromisoformat(point.timestamp.replace("Z", "+00:00"))
            diff = abs((pt_time - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_point = point

        return best_point

    def _build_env_tensor(self, env_scalars: dict) -> torch.Tensor:
        """Build environmental scalar tensor for forecast model."""
        # Pad to expected dimension (8)
        keys = ["SST_mean", "SST_std", "WIND_SHEAR_mean", "WIND_SHEAR_std",
                "MOISTURE_mean", "MOISTURE_std"]
        values = [env_scalars.get(k, 0.0) for k in keys]
        # Pad to 8
        while len(values) < 8:
            values.append(0.0)
        return torch.tensor([values[:8]], dtype=torch.float32, device=self.device)
