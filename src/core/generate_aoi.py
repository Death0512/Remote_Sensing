import json
from pathlib import Path
from shapely.geometry import Point, mapping
from src.core.geofence import GeofenceManager


def generate_aoi_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = GeofenceManager()

    center_feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(Point(108.0, 16.0)),
                "properties": {"name": "Vietnam Reference Center", "latitude": 16.0, "longitude": 108.0},
            }
        ],
    }
    with open(output_dir / "center_point.geojson", "w") as f:
        json.dump(center_feature, f, indent=2)

    zones = [
        ("zone1_critical_500km.geojson", 500_000.0, "Zone 1 Critical", 1),
        ("zone2_buffer_1500km.geojson", 1_500_000.0, "Zone 2 Buffer", 2),
        ("zone3_early_warning_3000km.geojson", 3_000_000.0, "Zone 3 Early Warning", 3),
    ]

    for filename, radius_m, name, zone_id in zones:
        poly = manager.get_zone_buffer_polygon(radius_m)
        feature = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "name": name,
                        "zone_id": zone_id,
                        "radius_km": radius_m / 1000.0,
                    },
                }
            ],
        }
        with open(output_dir / filename, "w") as f:
            json.dump(feature, f, indent=2)

    chokepoints = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(Point(101.3, 2.5)),
                "properties": {"name": "Malacca Strait", "type": "strategic_chokepoint"},
            },
            {
                "type": "Feature",
                "geometry": mapping(Point(105.8, -5.9)),
                "properties": {"name": "Sunda Strait", "type": "strategic_chokepoint"},
            },
            {
                "type": "Feature",
                "geometry": mapping(Point(115.7, -8.5)),
                "properties": {"name": "Lombok Strait", "type": "strategic_chokepoint"},
            },
            {
                "type": "Feature",
                "geometry": mapping(Point(121.0, 20.5)),
                "properties": {"name": "Bashi Channel", "type": "strategic_chokepoint"},
            },
        ],
    }
    with open(output_dir / "strategic_chokepoints.geojson", "w") as f:
        json.dump(chokepoints, f, indent=2)


if __name__ == "__main__":
    generate_aoi_files(Path(__file__).resolve().parent.parent.parent / "data" / "aoi")
