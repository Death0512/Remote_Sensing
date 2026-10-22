from datetime import datetime
from typing import Sequence, Union
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from src.domain.enums import Capability


class LeakageError(Exception):
    pass


class SplitValidator:
    capability: Capability = Capability.IMPLEMENTED

    def _normalize_geometry(
        self,
        geom_or_bbox: Union[BaseGeometry, tuple[float, float, float, float], list[float]],
    ) -> BaseGeometry:
        if isinstance(geom_or_bbox, (tuple, list)):
            if len(geom_or_bbox) != 4:
                raise ValueError("Bounding box must be (minx, miny, maxx, maxy)")
            return box(
                geom_or_bbox[0], geom_or_bbox[1], geom_or_bbox[2], geom_or_bbox[3]
            )
        return geom_or_bbox

    def _normalize_interval(
        self, interval: tuple[datetime, datetime]
    ) -> tuple[datetime, datetime]:
        start, end = interval
        if start > end:
            return end, start
        return start, end

    def validate_splits(
        self,
        train_aois: Sequence[
            Union[BaseGeometry, tuple[float, float, float, float], list[float]]
        ],
        test_aois: Sequence[
            Union[BaseGeometry, tuple[float, float, float, float], list[float]]
        ],
        train_intervals: Sequence[tuple[datetime, datetime]],
        test_intervals: Sequence[tuple[datetime, datetime]],
    ) -> bool:
        norm_train_aois = [self._normalize_geometry(a) for a in train_aois]
        norm_test_aois = [self._normalize_geometry(a) for a in test_aois]

        for i, train_geom in enumerate(norm_train_aois):
            for j, test_geom in enumerate(norm_test_aois):
                if train_geom.intersects(test_geom):
                    inter = train_geom.intersection(test_geom)
                    if inter.area > 0.0:
                        raise LeakageError(
                            f"Spatial leakage detected between train AOI {i} and test AOI {j}: intersection area = {inter.area}"
                        )

        norm_train_intervals = [
            self._normalize_interval(t) for t in train_intervals
        ]
        norm_test_intervals = [
            self._normalize_interval(t) for t in test_intervals
        ]

        for i, (train_start, train_end) in enumerate(norm_train_intervals):
            for j, (test_start, test_end) in enumerate(norm_test_intervals):
                latest_start = max(train_start, test_start)
                earliest_end = min(train_end, test_end)
                if latest_start < earliest_end:
                    raise LeakageError(
                        f"Temporal leakage detected between train interval {i} [{train_start.isoformat()} - {train_end.isoformat()}] and test interval {j} [{test_start.isoformat()} - {test_end.isoformat()}]"
                    )

        return True
