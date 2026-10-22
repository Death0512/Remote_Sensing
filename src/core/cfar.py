from typing import Optional
import numpy as np


class OSCFARDetector:
    def __init__(
        self,
        guard_m: float = 50.0,
        train_m: float = 200.0,
        pixel_spacing_m: float = 10.0,
        pfa: float = 1e-5,
        k_order_ratio: float = 0.75,
    ) -> None:
        self.guard_px = max(1, int(round(guard_m / pixel_spacing_m)))
        self.train_px = max(self.guard_px + 1, int(round(train_m / pixel_spacing_m)))
        self.pfa = pfa
        self.k_order_ratio = k_order_ratio

        offsets = []
        for dr in range(-self.train_px, self.train_px + 1):
            for dc in range(-self.train_px, self.train_px + 1):
                if not (abs(dr) <= self.guard_px and abs(dc) <= self.guard_px):
                    offsets.append((dr, dc))
        self.offsets = np.array(offsets, dtype=np.int32)
        self.offsets_r = self.offsets[:, 0]
        self.offsets_c = self.offsets[:, 1]

    def detect(self, intensity: np.ndarray, land_mask: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = intensity.shape
        detections = np.zeros((h, w), dtype=bool)

        padded_intensity = np.pad(intensity, self.train_px, mode="reflect")
        padded_mask = None
        if land_mask is not None:
            padded_mask = np.pad(land_mask, self.train_px, mode="constant", constant_values=1)

        t_px = self.train_px

        if h * w > 10000:
            valid_intensity = intensity[intensity > -1000]
            if len(valid_intensity) > 0:
                cand_thresh = float(np.percentile(valid_intensity, 99.0))
            else:
                cand_thresh = float(np.percentile(intensity, 99.0))
            candidates = np.argwhere(intensity >= cand_thresh)
        else:
            candidates = np.argwhere(np.ones((h, w), dtype=bool))

        for r, c in candidates:
            if land_mask is not None and land_mask[r, c]:
                continue

            pr = r + t_px
            pc = c + t_px

            sample_r = pr + self.offsets_r
            sample_c = pc + self.offsets_c

            if padded_mask is not None:
                mask_vals = padded_mask[sample_r, sample_c]
                valid_indices = (mask_vals == 0)
                if not np.any(valid_indices):
                    continue
                train_cells = padded_intensity[sample_r[valid_indices], sample_c[valid_indices]]
            else:
                train_cells = padded_intensity[sample_r, sample_c]

            n = len(train_cells)
            if n == 0:
                continue

            k_idx = int(round(self.k_order_ratio * (n - 1)))
            representative_val = float(np.partition(train_cells, k_idx)[k_idx])

            alpha = n * (self.pfa ** (-1.0 / n) - 1.0)
            threshold = representative_val * alpha

            if intensity[r, c] > threshold:
                detections[r, c] = True

        return detections
