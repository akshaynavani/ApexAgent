"""LiDAR-style ray sensors that measure distance to the track borders."""

from __future__ import annotations

import numpy as np

from apexagent.sim.geometry import ray_segment_distances


class RaySensor:
    """Fan of rays around the car's heading.

    Only border segments near the car are tested, which keeps ray casting cheap on long tracks.
    """

    def __init__(self, num_rays: int = 11, fov_deg: float = 180.0, max_range: float = 60.0) -> None:
        self.num_rays = int(num_rays)
        self.max_range = float(max_range)
        half = np.deg2rad(fov_deg) / 2.0
        self.angles = np.linspace(-half, half, self.num_rays) if self.num_rays > 1 else np.zeros(1)

    def directions(self, heading: float) -> np.ndarray:
        a = heading + self.angles
        return np.stack([np.cos(a), np.sin(a)], axis=1)

    def read(self, origin: np.ndarray, heading: float, track) -> np.ndarray:
        """Raw distances (metres) for each ray, clipped to ``max_range``."""
        a, b = track.border_a, track.border_b
        # Cull segments whose nearer endpoint is out of range (+ segment length slack).
        da = np.linalg.norm(a - origin, axis=1)
        db = np.linalg.norm(b - origin, axis=1)
        near = np.minimum(da, db) <= self.max_range + 5.0
        if not np.any(near):
            return np.full(self.num_rays, self.max_range)
        return ray_segment_distances(origin, self.directions(heading), a[near], b[near], self.max_range)

    def read_normalized(self, origin: np.ndarray, heading: float, track) -> np.ndarray:
        return self.read(origin, heading, track) / self.max_range

    def endpoints(self, origin: np.ndarray, heading: float, distances: np.ndarray) -> np.ndarray:
        """World-space hit points, for visualization."""
        return origin[None, :] + self.directions(heading) * distances[:, None]
