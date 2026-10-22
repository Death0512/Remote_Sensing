from src.core.tiling import generate_tiling_windows, point_deduplicate


def test_tiling_windows_coverage():
    windows = list(generate_tiling_windows(image_width=1200, image_height=1200, tile_size=800, overlap_px=128))
    assert len(windows) == 4
    for w in windows:
        assert w.width <= 800
        assert w.height <= 800


def test_point_deduplicate():
    points = [
        (100.0, 100.0, 0.95),
        (102.0, 101.0, 0.80),
        (300.0, 300.0, 0.90),
    ]
    deduped = point_deduplicate(points, distance_threshold_px=10.0)
    assert len(deduped) == 2
    assert deduped[0] == (100.0, 100.0, 0.95)
    assert deduped[1] == (300.0, 300.0, 0.90)
