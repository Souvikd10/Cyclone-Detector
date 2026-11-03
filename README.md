# Cyclone Prediction & Decision-Support System

Comprehensive AI/ML engine, backend infrastructure, and decision-support system for tropical cyclone detection, tracking, intensity forecasting, and impact assessment.

---

## 🏗️ System Architecture & Services

```
┌───────────────────────────────────────────────────────────────┐
│                        Docker Compose                         │
│  ┌─────────┐  ┌───────┐  ┌───────┐  ┌──────────────────────┐ │
│  │ FastAPI │  │ MinIO │  │MLflow │  │ Cyclone AI Engine    │ │
│  │  :8000  │  │ :9000 │  │ :5001 │  │ (Stages 1–13)        │ │
│  └────┬────┘  └───────┘  └───────┘  └──────────────────────┘ │
│       │                                                       │
│  ┌────┴──────────┐  ┌──────────┐                              │
│  │ PostgreSQL +  │  │ Prefect  │                              │
│  │   PostGIS     │  │  :4200   │                              │
│  │    :5432      │  └──────────┘                              │
│  └───────────────┘                                            │
└───────────────────────────────────────────────────────────────┘
```

| Service | URL | Purpose |
|---|---|---|
| **API** | `http://localhost:8000` | FastAPI Backend |
| **API Docs** | `http://localhost:8000/docs` | Swagger UI Documentation |
| **MinIO** | `http://localhost:9001` | Object Storage Console |
| **MLflow** | `http://localhost:5001` | Experiment & Model Tracking |
| **Prefect** | `http://localhost:4200` | Pipeline Orchestration |
| **PostgreSQL** | `localhost:5432` | PostGIS-enabled Spatio-temporal DB |

---

## 🌪️ AI/ML Pipeline Stages (Member A)

```
Raw Data → Preprocessing → Fusion → Shared Encoder → Multi-Head Predictions → Contract JSON
```

| Stage | Module | Description |
|---|---|---|
| 1 | `cyclone_ai/data/mosdac_client.py` | INSAT-3D/3DR satellite data acquisition (synthetic + real) |
| 1 | `cyclone_ai/data/ibtracs_client.py` | IBTrACS best-track historical data |
| 1 | `cyclone_ai/data/era5_client.py` | SST, wind shear, moisture environmental fields |
| 2 | `cyclone_ai/data/preprocessing.py` | Timestamp sync, geo-alignment, normalization |
| 2 | `cyclone_ai/data/storm_cropper.py` | Storm-centered 512×512 patch extraction |
| 2 | `cyclone_ai/data/dataset.py` | PyTorch Dataset & DataLoader factories |
| 3 | `cyclone_ai/fusion/source_encoder.py` | Per-source lightweight CNN encoders |
| 3 | `cyclone_ai/fusion/reliability.py` | Learned per-source reliability weights |
| 3 | `cyclone_ai/fusion/attention_fusion.py` | Cross-attention gated fusion |
| 4 | `cyclone_ai/encoder/shared_encoder.py` | ConvNeXt-Tiny backbone (via `timm`) |
| 5 | `cyclone_ai/heads/detection.py` | Cyclone detection, bbox, center, FPN mask |
| 6 | `cyclone_ai/heads/structure.py` | Spatial attention pool, eye detection, symmetry, circularity |
| 7 | `cyclone_ai/heads/temporal.py` | ConvLSTM / Transformer temporal sequence modeling |
| 8 | `cyclone_ai/heads/classification.py` | IMD category, lifecycle stage, structural pattern |
| 9 | `cyclone_ai/forecast/track_intensity.py` | Transformer decoder track & intensity forecasting (+6h to +72h) |
| 10 | `cyclone_ai/forecast/ri_model.py` | Tabular XGBoost rapid intensification classifier |
| 11 | `cyclone_ai/forecast/physics_correction.py` | Track speed, SST floor, Kaplan-DeMaria landfall decay |
| 12 | `cyclone_ai/uncertainty/ood_estimation.py` | Mahalanobis OOD novelty score & human review flag |
| 13 | `cyclone_ai/xai/gradcam.py`, `explanations.py` | Grad-CAM heatmaps & 15 meteorological explanation tags |

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run Inference via Python
```python
from cyclone_ai.inference import run_inference

result = run_inference(
    storm_id="2020121N10087",
    timestamp="2020-05-17T06:00:00Z",
    config_dir="configs",
)
print(result["current_state"]["category"])  # e.g., "VSCS"
```

### 4. Generate Sample Outputs (for Members B & C)
```bash
python scripts/generate_sample_json.py
```

### 5. Start Backend Services (Docker)
```bash
docker compose up --build
```

---

## 📋 Inference Output Contract JSON

```json
{
  "storm_id": "string",
  "timestamp": "ISO8601",
  "detection": {
    "present": true,
    "confidence": 0.0,
    "center": {"lat": 0, "lon": 0},
    "bbox": [0,0,0,0],
    "mask_url": "minio://path"
  },
  "structure": {
    "eye_present": true,
    "eye_confidence": 0.0,
    "eye_center": {"lat":0,"lon":0},
    "eye_diameter_km": 0,
    "symmetry": 0.0,
    "circularity": 0.0
  },
  "current_state": {
    "category": "string",
    "stage": "string",
    "pattern": "string"
  },
  "forecast": {
    "track": [{"t": "+6h", "lat": 0, "lon": 0, "cone_radius_km": 0}],
    "intensity": [{"t": "+6h", "wind_kt": 0, "interval": [0, 0]}]
  },
  "ri": {"probability": 0.0, "confidence": 0.0},
  "uncertainty": {"novelty_score": 0.0, "human_review_flag": false},
  "xai": {
    "gradcam_url": "minio://path",
    "explanation_tags": ["strong eyewall", "symmetric convection"]
  }
}
```
