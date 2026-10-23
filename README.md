# Remote Sensing Territorial Surveillance System

Production-grade territorial surveillance and early warning remote sensing system protecting Vietnam's sovereignty across maritime, airspace, and land domains within a 3,000 km geodesic radius centered on Vietnam (16.0°N, 108.0°E).

## Key Capabilities (V3 Observation-Driven Architecture)

- **Geodesic Surveillance Zones (AEQD + WGS84)**:
  - **Zone 1 (0 – 500 km)**: Critical / Sovereignty & EEZ
  - **Zone 2 (500 – 1,500 km)**: High / Approach Buffer
  - **Zone 3 (1,500 – 3,000 km)**: Strategic / Early Warning
- **Multi-Sensor Detection Pipelines**:
  - **SAR Vessel Detection**: True sliding-window CA/OS-CFAR parameterized in ground meters with GSHHG coastline land-mask prefiltering.
  - **Optical Aircraft Detection**: YOLOv8-OBB and truthfully labeled heuristic baselines with cloud masking and quality tiers.
  - **Temporal Parallax In-Flight Detector**: Sentinel-2 L1C TOA band-stagger RGB temporal parallax validation for aircraft velocities.
  - **Calibrated Change Detection**: Multi-stage bitemporal co-registration with sub-pixel residual gating and empirical thresholding.
- **Intelligence & Kinematic Tracking**:
  - **Fail-Closed AIS Association**: Multi-hypothesis 6-state correlator requiring verified receiver coverage before dark-vessel alerts.
  - **Class-Specific Kalman Kinematics**: 2D Kalman filters with class-specific process noise and Mahalanobis innovation gating.
  - **Explainable Threat Hypotheses**: Multi-source atomic evidence accumulator with Platt-calibrated confidence vectors.
- **Enterprise Alert Packaging & Ledger**:
  - **Radiometric Alert Packages**: Self-contained bundles with raw 32-bit float GeoTIFFs, analysis tensors, and GeoJSON detections.
  - **Tamper-Evident Review Ledger**: SQLite review database with atomic transactions, SHA-256 hash-chaining, and append-only triggers.
- **Production API & Web GIS Command Dashboard**:
  - **Strict Security & RBAC**: HMAC-SHA256 tokens, 5 discrete roles, explicit CORS whitelist, and sanitized error envelopes.
  - **XSS-Immune Vector Dashboard**: MapLibre GL JS frontend with delegated event listeners and true metric AEQD rings.
  - **Spatiotemporal Leakage-Free Benchmarks**: Dual-axis validation enforcing spatial disjointness and temporal guard intervals.

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
│   ├── alerting/          # Radiometric alert packager and tamper-evident review ledger
│   ├── api/               # FastAPI backend with JWT auth and RBAC roles
│   ├── benchmark/         # Spatiotemporal leakage-free dataset generator and metrics
│   ├── cli/               # Unified CLI entrypoint (scan, serve, ingest, status)
│   ├── detection/         # SAR, Optical, Change, and Experimental Parallax detectors
│   ├── domain/            # Canonical frozen contracts, geometries, and confidence vectors
│   ├── ingestion/         # STAC and OData catalog clients with SHA-256 verification
│   ├── intelligence/      # Fail-closed AIS correlator, hypothesis engine, and GFW audit
│   ├── preprocessing/     # Calibration, GSHHG land masks, SCL cloud masks, georeferencing
│   ├── provenance/        # Dynamic runtime provenance, git commit, and model hashes
│   └── tracking/          # 2D Kalman kinematics and gated Hungarian association
├── tests/                 # Canonical V3 unit, contract, and benchmark test suites
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
uv run python -m src.cli.main serve --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

### 3. Run Pipeline Tests

```bash
uv run pytest
```

### 4. CLI Scan on Local Imagery

```bash
uv run python -m src.cli.main scan --image data/samples/demo_sar.npy --sensor S1_SAR --output-dir data/outputs
```
