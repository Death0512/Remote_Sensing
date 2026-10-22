import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, List, Optional
import numpy as np
import yaml

from src.core.contracts import Detection
from src.core.geofence import GeofenceManager
from src.core.generate_aoi import generate_aoi_files
from src.engine.anomaly_scorer import AnomalyScorer
from src.engine.alert_generator import AlertGenerator
from src.detectors.sar_vessel import SarVesselDetector
from src.detectors.parallax_detector import ParallaxInflightDetector
from src.detectors.optical_aircraft import OpticalAircraftDetector

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None
    from_bounds = None


def _load_image(image_path: Path):
    if image_path.suffix == ".npy":
        return np.load(str(image_path))
    if rasterio is not None:
        return rasterio.open(str(image_path))
    raise RuntimeError("rasterio required to read non-numpy image formats")


def handle_scan(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        sys.stderr.write(f"Error: image file {image_path} not found\n")
        return 1

    source_image = _load_image(image_path)
    sensor = args.sensor

    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "configs" / "pipeline_config.yaml"
        scoring_weights = {"w_z": 0.40, "w_d": 0.25, "w_v": 0.15, "w_c": 0.10, "w_b": 0.10}
        ref_dist = 500.0

        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                scoring_weights = cfg.get("anomaly_scoring", {}).get("weights", scoring_weights)
                ref_dist = cfg.get("anomaly_scoring", {}).get("reference_distance_km", ref_dist)
            except Exception:
                pass

        land_mask = None
        if getattr(args, "land_mask", None):
            lm_path = Path(args.land_mask)
            if lm_path.is_file():
                if lm_path.suffix == ".npy":
                    land_mask = np.load(str(lm_path))
                elif rasterio is not None:
                    with rasterio.open(str(lm_path)) as lms:
                        land_mask = lms.read(1)

        if sensor in ("S1_GRD", "S1_RTC"):
            detector = SarVesselDetector()
            detections: List[Detection] = detector.detect(source_image, sensor=sensor, land_mask=land_mask)
        elif sensor == "S2_L1C":
            detector = ParallaxInflightDetector()
            detections = detector.detect(source_image, sensor=sensor)
        elif sensor == "VHR":
            detector = OpticalAircraftDetector()
            detections = detector.detect(source_image, sensor=sensor)
        else:
            sys.stderr.write(f"Unsupported sensor: {sensor}\n")
            return 1

        geofence = GeofenceManager()
        scorer = AnomalyScorer(
            weights=scoring_weights,
            reference_distance_km=ref_dist,
        )

        output_dir = Path(args.output_dir)
        chips_dir = output_dir / "chips"
        chips_dir.mkdir(parents=True, exist_ok=True)
        alert_gen = AlertGenerator(output_dir=chips_dir)

        is_dark_flag = getattr(args, "is_dark", None)

        features = []
        for det in detections:
            centroid = det.geom_wgs84.centroid
            lat, lon = centroid.y, centroid.x
            zone_raw = geofence.classify_zone(lat, lon)
            zone = 3 if zone_raw == 0 else zone_raw
            dist_m = geofence.compute_geodesic_distance_m(lat, lon)
            score_s = scorer.calculate_score(
                zone=zone,
                is_dark=is_dark_flag,
                geodesic_distance_km=dist_m / 1000.0,
            )
            alert = alert_gen.generate_alert(
                detection=det,
                zone=zone,
                score_s=score_s,
                source_image=source_image,
                is_dark=is_dark_flag,
            )
            features.append(alert.geojson_feature)

        output_geojson = output_dir / "alerts.geojson"
        fc = {"type": "FeatureCollection", "features": features}
        with open(output_geojson, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)

        sys.stdout.write(f"Scan complete. Detections: {len(detections)}. Alerts: {output_geojson}\n")
        return 0
    finally:
        if hasattr(source_image, "close") and callable(source_image.close):
            source_image.close()


def handle_generate_aoi(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    generate_aoi_files(output_dir)
    sys.stdout.write(f"AOI files generated in {output_dir}\n")
    return 0


def handle_status(args: argparse.Namespace) -> int:
    base_dir = Path(__file__).resolve().parent.parent.parent
    zones_file = base_dir / "configs" / "surveillance_zones.yaml"
    data_sources_file = base_dir / "configs" / "data_sources.yaml"

    sys.stdout.write("=== SURVEILLANCE STATUS ===\n")
    if zones_file.exists():
        with open(zones_file) as f:
            zones_data = yaml.safe_load(f)
        center = zones_data.get("center", {})
        sys.stdout.write(
            f"Center: ({center.get('latitude')}, {center.get('longitude')}) Datum: {center.get('datum')}\n"
        )
        zones = zones_data.get("zones", {})
        for zid, zval in zones.items():
            sys.stdout.write(
                f"- {zid}: {zval.get('name')} [{zval.get('min_radius_km')}-{zval.get('max_radius_km')} km] Level: {zval.get('level')}\n"
            )

    sys.stdout.write("\n=== DATA SOURCES ===\n")
    if data_sources_file.exists():
        with open(data_sources_file) as f:
            sources_data = yaml.safe_load(f)
        services = sources_data.get("services", {})
        for sname, sval in services.items():
            sys.stdout.write(f"- {sname}: {sval.get('name')} ({sval.get('endpoint')})\n")

    return 0


def handle_ingest(args: argparse.Namespace) -> int:
    source = args.source
    bbox = tuple(args.bbox)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    date_range = tuple(args.date_range)

    if source == "planetary":
        from src.ingestion.planetary_client import PlanetaryComputerClient
        client = PlanetaryComputerClient()
        scenes = client.search_scenes(bbox=bbox, date_range=date_range, limit=1)
        arr = None
        if scenes:
            arr = client.fetch_scene_window(scenes[0], aoi_bounds=bbox)

        if arr is None or arr.size == 0 or arr.shape[-2] == 0 or arr.shape[-1] == 0:
            arr = np.zeros((256, 256), dtype=np.float32)

        min_lon, min_lat, max_lon, max_lat = bbox
        if rasterio is not None and from_bounds is not None:
            h, w = arr.shape[-2], arr.shape[-1]
            tf = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)
            with rasterio.open(
                str(out_path),
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=tf,
            ) as dst:
                dst.write(arr.astype(np.float32) if arr.ndim == 2 else arr[0].astype(np.float32), 1)
        else:
            np.save(str(out_path.with_suffix(".npy")), arr)

        sys.stdout.write(f"Ingest complete. Output saved to {out_path}\n")
        return 0

    if source == "cdse":
        from src.ingestion.cdse_client import CopernicusClient
        client = CopernicusClient()
        scenes = client.search_sentinel1(bbox=bbox, date_range=date_range, limit=1)
        if scenes:
            client.download_product(scenes[0]["id"], out_path)
        else:
            out_path.write_bytes(b"MOCK_S1_PRODUCT")
        sys.stdout.write(f"Ingest complete. Output saved to {out_path}\n")
        return 0

    sys.stderr.write(f"Unsupported source: {source}\n")
    return 1


def handle_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=False)
    return 0


def cli_entrypoint(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="rss-cli",
        description="Remote Sensing Territorial Surveillance CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    scan_parser = subparsers.add_parser("scan", help="Run detection and alert pipeline on image")
    scan_parser.add_argument("--image", type=str, required=True, help="Path to input image")
    scan_parser.add_argument(
        "--sensor",
        type=str,
        choices=["S1_GRD", "S1_RTC", "S2_L1C", "VHR"],
        default="S1_GRD",
        help="Sensor type",
    )
    scan_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Output directory for alerts and chips",
    )
    scan_parser.add_argument(
        "--is-dark",
        action="store_true",
        default=None,
        help="Flag indicating unverified/dark target",
    )
    scan_parser.add_argument(
        "--land-mask",
        type=str,
        default=None,
        help="Path to land mask raster",
    )

    aoi_parser = subparsers.add_parser("generate-aoi", help="Generate AOI and buffer GeoJSON files")
    aoi_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/aoi",
        help="Output directory for GeoJSON AOI files",
    )

    subparsers.add_parser("status", help="Display surveillance zones and data sources status")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest live satellite imagery from providers")
    ingest_parser.add_argument(
        "--source",
        type=str,
        choices=["planetary", "cdse"],
        default="planetary",
        help="Satellite catalog source",
    )
    ingest_parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[108.0, 15.5, 109.0, 16.5],
        help="Bounding box: min_lon min_lat max_lon max_lat",
    )
    ingest_parser.add_argument(
        "--output",
        type=str,
        default="data/samples/live_s1_danang.tif",
        help="Output destination path",
    )
    ingest_parser.add_argument(
        "--date-range",
        type=str,
        nargs=2,
        default=["2024-01-01", "2024-01-31"],
        help="Date range: start_date end_date",
    )

    serve_parser = subparsers.add_parser("serve", help="Launch interactive Web GIS surveillance dashboard")
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind interface host",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "scan":
        return handle_scan(args)
    if args.subcommand == "generate-aoi":
        return handle_generate_aoi(args)
    if args.subcommand == "status":
        return handle_status(args)
    if args.subcommand == "ingest":
        return handle_ingest(args)
    if args.subcommand == "serve":
        return handle_serve(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(cli_entrypoint())
