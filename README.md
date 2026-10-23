# Remote Sensing Territorial Surveillance System

Production-grade territorial surveillance and early warning remote sensing system protecting Vietnam's sovereignty across maritime, airspace, and land domains within a 3,000 km geodesic radius centered on Vietnam (16.0°N, 108.0°E).

## Key Capabilities

- **Canonical Domain Contracts**:
  - Frozen dataclass contracts across domain entities: `Observation`, `Measurement`, `Candidate`, `Detection`, `Evidence`, `Association`, `Track`, `Hypothesis`, `Decision`, `Alert`, `Geometries`, and `ConfidenceVector`.
  - Platt probability calibration mapping detector heuristic scores to calibrated posterior probabilities.
- **Dynamic Runtime Cryptographic Provenance**:
  - Dynamic runtime provenance tracking git commit, byte checksums, environment state, and execution timestamps.
  - Fail-closed cryptographic model registry validating weight file integrity prior to inference execution with runtime SHA-256 hash chaining.
- **Pre-Filtering & Metric Preprocessing**:
  - GSHHG coastline land-mask prefiltering applied prior to detector execution.
  - Metric-GSD 2D Ordered-Statistic CFAR (OS-CFAR) on calibrated SAR imagery.
  - SCL-based cloud and shadow gating with kinematic evidence thresholds for optical acquisitions.
  - Sub-pixel bitemporal co-registration with residual displacement validation.
- **Fail-Closed AIS Intelligence Ladder**:
  - Fail-closed AIS coverage ladder preventing dark-vessel alerts without active verified receiver coverage.
  - Dual-window spatial and temporal correlation with speed and course feasibility validation.
- **Kinematic Multi-Look Tracking**:
  - Class-specific 2D Kalman filter state estimation with specialized process noise covariance matrices.
  - Gated Hungarian association driven by metric Mahalanobis distance gating.
- **Tamper-Evident Review Ledger**:
  - Multi-tier radiometric alert package generation with contextual chips.
  - Tamper-evident, hash-chained SQLite review ledger with append-only database triggers preventing modification or deletion.
- **Production API & Hardened Web GIS**:
  - Production JWT authentication with 5 role-based access control tiers (`viewer`, `analyst`, `operator`, `supervisor`, `admin`).
  - Strict CORS origin gating, rate limiting, and sanitized error envelopes.
  - XSS-immune MapLibre GL JS vector dashboard with geodesic AEQD surveillance rings and real-time NASA GIBS true-color basemaps.
- **Leakage-Free Spatiotemporal Benchmarking**:
  - Spatiotemporal partition leakage validation in benchmarking.
  - Metric evaluation calculating strict IoU matching, monotonic AP@0.5, and false alarm rate per square kilometer (FAR/km²).

## Directory Structure

```text
.
├── configs/
├── data/
│   ├── aoi/
│   ├── outputs/
│   ├── samples/
│   └── weights/
├── src/
│   ├── alerting/
│   │   ├── package/
│   │   └── review/
│   ├── api/
│   ├── benchmark/
│   ├── cli/
│   ├── detection/
│   │   ├── change/
│   │   ├── experimental/
│   │   ├── optical/
│   │   └── sar/
│   ├── domain/
│   ├── ingestion/
│   ├── intelligence/
│   │   ├── ais/
│   │   ├── calibration/
│   │   ├── gfw/
│   │   └── hypothesis/
│   ├── preprocessing/
│   ├── provenance/
│   └── tracking/
├── tests/
├── web/
├── pyproject.toml
└── pytest.ini
```

### Module Overview

- `src/domain/`: Canonical frozen contracts (`Observation`, `Measurement`, `Candidate`, `Detection`, `Evidence`, `Association`, `Track`, `Hypothesis`, `Decision`, `Alert`, `Geometries`, `ConfidenceVector`).
- `src/provenance/`: Dynamic runtime provenance, git commit, byte checksums, and cryptographic model verification.
- `src/preprocessing/`: Radiometric calibration, SCL cloud masking, GSHHG land masking, georeferencing pipeline, and bitemporal co-registration.
- `src/detection/`: Multi-sensor detection engines (`sar/`, `optical/`, `change/`, and `experimental/` parallax).
- `src/intelligence/`: Intelligence fusion (`ais/` fail-closed correlator, `calibration/` Platt scaling, `gfw/` retrospective audit, `hypothesis/` explainable engine, and `accumulator.py`).
- `src/tracking/`: 2D Kalman filter with class-specific motion models and gated Hungarian association via Mahalanobis distance.
- `src/alerting/`: Radiometric multi-tier package builder (`package/`) and SQLite append-only hash-chained audit ledger (`review/`).
- `src/benchmark/`: Spatiotemporal leakage-free partitioning, IoU matching, AP@0.5, and false alarm density per km².
- `src/api/`: Production JWT/RBAC auth with 5 roles, CORS origin gating, and sanitized error envelopes.
- `src/cli/`: Unified CLI entrypoint (`scan`, `serve`, `ingest`, `generate-aoi`, `status`).
- `src/ingestion/`: Real CDSE and Planetary Computer catalog and windowed downloader with SHA-256 validation and mock isolation.

## Quick Start

### 1. Environment Setup

```bash
uv sync
```

### 2. Run Test Suite

```bash
uv run pytest
```

### 3. Launch Interactive Web GIS Dashboard

```bash
uv run python -m src.cli.main serve --host 0.0.0.0 --port 8000
```

Access the dashboard at `http://localhost:8000`.

### 4. CLI Surveillance Scan

```bash
uv run python -m src.cli.main scan --image data/samples/demo_sar.npy --sensor S1_GRD --detector auto --output-dir data/outputs --verified-ais-coverage
```

### 5. Generate Geodesic AOI Boundaries

```bash
uv run python -m src.cli.main generate-aoi --output-dir data/aoi
```
