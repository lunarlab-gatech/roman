from __future__ import annotations

import colorsys
from dataclasses import dataclass
import numpy as np
from typeguard import typechecked

def hsvF_to_rgb255(hsvF: np.ndarray[float]) -> np.ndarray[np.uint8]:
    hsvF_clamped = np.clip(hsvF, 0, 1)
    r, g, b = colorsys.hsv_to_rgb(hsvF_clamped[0], hsvF_clamped[1], hsvF_clamped[2])
    return np.array([int(round(r * 255)), int(round(g * 255)), int(round(b * 255))], dtype=np.uint8)

def rgb255_to_hsvF(rgb255: np.ndarray[np.uint8]) -> np.ndarray[float]:
    rgbF = np.clip(rgb255.astype(np.float32) / 255.0, 0.0, 1.0)
    h, s, v = colorsys.rgb_to_hsv(rgbF[0], rgbF[1], rgbF[2])
    return np.array([h, s, v], dtype=float)

def rgb255_to_bgr255(rgb: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return rgb[::-1]

@dataclass
@typechecked
class HSVSpace:
    h_range: tuple[float, float]
    s_range: tuple[float, float]
    v_range: tuple[float, float]

    def split(self, n: int, axis: int) -> list[HSVSpace]:
        """Split this HSV space into `n` parts along the given axis (0=H, 1=S, 2=V)."""
        if axis == 0:
            r0, r1 = self.h_range
        elif axis == 1:
            r0, r1 = self.s_range
        else:
            r0, r1 = self.v_range

        step = (r1 - r0) / n
        subspaces = []
        for i in range(n):
            split_range = (r0 + i * step, r0 + (i + 1) * step)
            if axis == 0:
                subspaces.append(HSVSpace(split_range, self.s_range, self.v_range))
            elif axis == 1:
                subspaces.append(HSVSpace(self.h_range, split_range, self.v_range))
            else:
                subspaces.append(HSVSpace(self.h_range, self.s_range, split_range))
        return subspaces