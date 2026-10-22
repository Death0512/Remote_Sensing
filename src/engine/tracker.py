from datetime import datetime, timezone
from typing import List, Optional, Tuple
import pyproj
from shapely.geometry import Point

from src.core.contracts import Detection, Track


class TemporalTracker:
    def __init__(
        self,
        max_distance_meters: float = 50000.0,
        max_dt_seconds: float = 7200.0,
    ) -> None:
        self.max_distance_meters = max_distance_meters
        self.max_dt_seconds = max_dt_seconds
        self.geod = pyproj.Geod(ellps="WGS84")

    def _extract_coords(self, detection: Detection) -> Tuple[float, float]:
        centroid: Point = detection.geom_wgs84.centroid
        return centroid.x, centroid.y

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def associate(
        self,
        detections_t1: List[Detection],
        detections_t2: List[Detection],
    ) -> List[Track]:
        candidate_pairs = []
        for d1 in detections_t1:
            time1 = self._ensure_utc(d1.acquisition_time)
            for d2 in detections_t2:
                time2 = self._ensure_utc(d2.acquisition_time)
                dt = (time2 - time1).total_seconds()
                if dt == 0:
                    continue
                if dt < 0:
                    earlier, later = d2, d1
                    dt_pos = -dt
                else:
                    earlier, later = d1, d2
                    dt_pos = dt

                if dt_pos > self.max_dt_seconds:
                    continue

                lon_e, lat_e = self._extract_coords(earlier)
                lon_l, lat_l = self._extract_coords(later)
                az12, _, dist = self.geod.inv(lon_e, lat_e, lon_l, lat_l)
                dist = float(dist)

                if dist <= self.max_distance_meters:
                    candidate_pairs.append((dist, dt_pos, az12, earlier, later))

        candidate_pairs.sort(key=lambda x: x[0])

        assigned_earlier = set()
        assigned_later = set()
        tracks: List[Track] = []

        for dist, dt_pos, az12, earlier, later in candidate_pairs:
            if id(earlier) in assigned_earlier or id(later) in assigned_later:
                continue
            assigned_earlier.add(id(earlier))
            assigned_later.add(id(later))
            velocity_mps = dist / dt_pos
            heading_deg = float(az12 % 360)
            track_id = f"TRK-{earlier.det_id}-{later.det_id}"
            tracks.append(
                Track(
                    track_id=track_id,
                    detections=[earlier, later],
                    velocity_mps=velocity_mps,
                    heading_deg=heading_deg,
                    time_delta_seconds=dt_pos,
                )
            )

        all_assigned = assigned_earlier | assigned_later
        for d in detections_t1:
            if id(d) not in all_assigned:
                all_assigned.add(id(d))
                tracks.append(
                    Track(
                        track_id=f"TRK-{d.det_id}",
                        detections=[d],
                        velocity_mps=None,
                        heading_deg=d.heading_deg,
                        time_delta_seconds=0.0,
                    )
                )

        for d in detections_t2:
            if id(d) not in all_assigned:
                all_assigned.add(id(d))
                tracks.append(
                    Track(
                        track_id=f"TRK-{d.det_id}",
                        detections=[d],
                        velocity_mps=None,
                        heading_deg=d.heading_deg,
                        time_delta_seconds=0.0,
                    )
                )

        return tracks

    def track(
        self,
        detections_t1: List[Detection],
        detections_t2: Optional[List[Detection]] = None,
    ) -> List[Track]:
        if detections_t2 is not None:
            return self.associate(detections_t1, detections_t2)
        if not detections_t1:
            return []
        grouped = {}
        for d in detections_t1:
            t = self._ensure_utc(d.acquisition_time)
            grouped.setdefault(t, []).append(d)
        sorted_times = sorted(grouped.keys())
        if len(sorted_times) == 1:
            return [
                Track(
                    track_id=f"TRK-{d.det_id}",
                    detections=[d],
                    velocity_mps=None,
                    heading_deg=d.heading_deg,
                    time_delta_seconds=0.0,
                )
                for d in grouped[sorted_times[0]]
            ]
        all_tracks = []
        for i in range(len(sorted_times) - 1):
            t1 = sorted_times[i]
            t2 = sorted_times[i + 1]
            sub_tracks = self.associate(grouped[t1], grouped[t2])
            all_tracks.extend(sub_tracks)
        return all_tracks
