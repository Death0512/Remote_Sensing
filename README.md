# Remote Sensing Territorial Surveillance System

Production-grade territorial surveillance and early warning remote sensing system protecting Vietnam's sovereignty across maritime, airspace, and land domains within a 3,000 km geodesic radius centered on Vietnam (16.0°N, 108.0°E).

## Key Capabilities

- **Geodesic Surveillance Zones (AEQD + WGS84)**:
  - **Zone 1 (0 – 500 km)**: Critical / Sovereignty & EEZ
  - **Zone 2 (500 – 1,500 km)**: High / Approach Buffer
  - **Zone 3 (1,500 – 3,000 km)**: Strategic / Early Warning
- **Multi-Sensor Detection Pipelines**:
  - **SAR Vessel Detection**: 2D Ordered-Statistic CFAR (OS-CFAR), land-mask pre-filtering, and PCA spatial covariance heading extraction.
  - **Optical Aircraft Detection**: YOLOv8-OBB inference for parked and taxiing aircraft with oriented bounding boxes.
  - **Temporal Parallax In-Flight Detector**: Sentinel-2 L1C TOA band-stagger RGB temporal parallax validation for aircraft velocities (300–1000 km/h).
  - **Dual-Stream Change Detection**: Bi-temporal difference proposers with cosine similarity suppression for infrastructure and artificial island changes.
- **Fusion & Intelligence**:
  - **SAR-AIS Correlator**: Spatiotemporal gate matching; unverified targets flagged as `DARK_VESSEL`.
  - **Multi-Look Tracker**: Temporal trajectory association with ground velocity and heading derivation.
  - **Normalized Threat Scoring**: Multi-factor scoring engine normalized to $[0, 100]$.
- **Web GIS Interactive Command Dashboard**:
  - MapLibre GL JS vector frontend with true metric AEQD rings.
  - Multi-source basemap switcher: High-Resolution Satellite (ESRI World Imagery HD), Topography/Elevation (OpenTopoMap), Ocean Bathymetry (ArcGIS Oceans), Nautical Seamarks (OpenSeaMap), and Tactical Dark.
  - Real-time NASA GIBS daily true-color satellite imagery with interactive date timeline slider.
  - Live Map-Tile Scanner (`POST /api/scan-map-aoi`): Direct satellite tile extraction and automated target analysis with real 256x256 image chip generation.
  - Operator audit actions: Human-in-the-loop threat confirmation and dismissal with SQLite persistence and append-only audit trail.

## Directory Structure

```text
.
├── configs/               # System configurations (zones, pipelines, models, data sources)
├── data/
│   ├── aoi/               # Standard GeoJSON perimeters (zones, center, chokepoints)
│   ├── outputs/           # Real-time alert chips, SQLite DB, and audit logs
│   ├── samples/           # Sample raster imagery for validation
│   └── weights/           # Pretrained neural network weights
├── src/
│   ├── api/               # FastAPI backend endpoints
│   ├── cli/               # Unified CLI entrypoint (`scan`, `serve`, `ingest`, `status`)
│   ├── core/              # Contracts, geodesy, tiling, coordinate transforms, CFAR
│   ├── detectors/         # SAR, Optical OBB, Parallax, and Change detectors
│   ├── engine/            # Fusion, tracker, anomaly scorer, alert generator
│   └── ingestion/         # Planetary Computer, CDSE, GFW, and weights clients
├── tests/                 # Unit and integration test suites
├── web/                   # MapLibre GL JS vector dashboard (HTML, CSS, JS)
├── PLAN.md                # Comprehensive architectural blueprint
├── RULES.md               # Engineering constraints & verification invariants
├── TODO.md                # Implementation milestones and status
├── pyproject.toml         # PEP 621 package metadata
└── pytest.ini             # Test runner configuration
```

## Quick Start

### 1. Environment Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Launch Interactive Web GIS Dashboard

```bash
.venv/bin/python -m src.cli.main serve --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

### 3. Run Pipeline Tests

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

### 4. CLI Scan on Local Imagery

```bash
.venv/bin/python -m src.cli.main scan --image data/samples/demo_sar.npy --sensor S1_GRD --output-dir data/outputs
```
