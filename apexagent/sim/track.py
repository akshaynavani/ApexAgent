"""Closed race tracks defined by a spline centerline and a constant width."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexagent.sim.geometry import catmull_rom_closed, polyline_normals, resample_closed

# Control points (metres) for the built-in layouts. All are traversed counter-clockwise.
BUILTIN_TRACKS: dict[str, list[tuple[float, float]]] = {
    "oval": [(0, -40), (60, -40), (90, 0), (60, 40), (0, 40), (-60, 40), (-90, 0), (-60, -40)],
    "kidney": [
        (0, -50), (70, -45), (100, 0), (75, 45), (35, 30), (0, 45),
        (-40, 55), (-85, 30), (-95, -15), (-55, -50),
    ],
    "apex_gp": [
        (0, -70), (60, -75), (110, -50), (120, -5), (90, 20), (60, 5),
        (35, 30), (60, 60), (30, 85), (-25, 80), (-55, 50), (-40, 20),
        (-80, 10), (-115, -25), (-95, -60), (-45, -70),
    ],
}


@dataclass
class TrackProjection:
    """Where a point sits relative to the centerline."""

    index: int  # segment index along the centerline
    s: float  # arc-length position of the projected point in [0, length)
    lateral: float  # signed offset from centerline (positive = left)
    heading: float  # centerline tangent direction (radians)
    point: np.ndarray  # projected point on the centerline


class Track:
    """A closed track with constant width.

    The centerline is a uniformly resampled polyline. Borders are offset along normals.
    Checkpoints are evenly spaced gates across the track used for progress bonuses.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        width: float = 12.0,
        spacing: float = 2.0,
        num_checkpoints: int = 20,
        name: str = "custom",
    ) -> None:
        self.name = name
        self.width = float(width)
        self.half_width = self.width / 2.0
        dense = catmull_rom_closed(np.asarray(control_points, dtype=np.float64), samples_per_segment=30)
        self.center = resample_closed(dense, spacing)
        self.n = len(self.center)

        nxt = np.roll(self.center, -1, axis=0)
        self.seg_vec = nxt - self.center
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)
        self.seg_dir = self.seg_vec / (self.seg_len[:, None] + 1e-12)
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())

        self.tangents, self.normals = polyline_normals(self.center)
        self.left = self.center + self.normals * self.half_width
        self.right = self.center - self.normals * self.half_width
        # Border segments as (a, b) arrays for ray casting.
        self.border_a = np.vstack([self.left, self.right])
        self.border_b = np.vstack([np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)])

        self.num_checkpoints = int(num_checkpoints)
        self.checkpoint_s = np.linspace(0.0, self.length, self.num_checkpoints, endpoint=False)
        self.checkpoint_idx = np.searchsorted(self.cum_s, self.checkpoint_s, side="right") - 1

        self.curvature = self._compute_curvature()

    # ------------------------------------------------------------------ geometry
    def _compute_curvature(self) -> np.ndarray:
        heading = np.arctan2(self.seg_dir[:, 1], self.seg_dir[:, 0])
        dh = np.diff(np.concatenate([heading, heading[:1]]))
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        return dh / (self.seg_len + 1e-12)

    def project(self, pos: np.ndarray, hint: int | None = None, window: int = 25) -> TrackProjection:
        """Project ``pos`` onto the centerline.

        With ``hint`` (previous segment index) only a local window is searched, which keeps
        the step cost O(window) instead of O(n) and avoids snapping across nearby straights.
        """
        if hint is None:
            idx = np.arange(self.n)
        else:
            idx = (np.arange(hint - window, hint + window + 1)) % self.n
        a = self.center[idx]
        d = self.seg_dir[idx]
        L = self.seg_len[idx]
        rel = pos[None, :] - a
        t = np.clip(np.einsum("ij,ij->i", rel, d), 0.0, L)
        proj = a + d * t[:, None]
        dist2 = np.sum((pos[None, :] - proj) ** 2, axis=1)
        k = int(np.argmin(dist2))
        i = int(idx[k])
        p = proj[k]
        dir_i = self.seg_dir[i]
        offset = pos - p
        lateral = float(dir_i[0] * offset[1] - dir_i[1] * offset[0])
        s = float((self.cum_s[i] + t[k]) % self.length)
        heading = float(np.arctan2(dir_i[1], dir_i[0]))
        return TrackProjection(index=i, s=s, lateral=lateral, heading=heading, point=p)

    def is_on_track(self, proj: TrackProjection, margin: float = 0.0) -> bool:
        return abs(proj.lateral) <= self.half_width - margin

    def pose_at(self, s: float) -> tuple[np.ndarray, float]:
        """Centerline position and heading at arc length ``s``."""
        s = s % self.length
        i = int(np.searchsorted(self.cum_s, s, side="right") - 1)
        t = s - self.cum_s[i]
        pos = self.center[i] + self.seg_dir[i] * t
        heading = float(np.arctan2(self.seg_dir[i, 1], self.seg_dir[i, 0]))
        return pos, heading

    def progress_delta(self, s_prev: float, s_new: float) -> float:
        """Signed arc-length moved, unwrapping across the start/finish line."""
        ds = s_new - s_prev
        half = self.length / 2.0
        if ds > half:
            ds -= self.length
        elif ds < -half:
            ds += self.length
        return ds

    def curvature_ahead(self, index: int, distances: list[float]) -> np.ndarray:
        """Centerline curvature sampled at given distances ahead of segment ``index``."""
        out = []
        for dist in distances:
            j = (index + int(round(dist / max(self.length / self.n, 1e-6)))) % self.n
            out.append(self.curvature[j])
        return np.asarray(out, dtype=np.float64)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.vstack([self.left, self.right])
        return pts.min(axis=0), pts.max(axis=0)


def random_control_points(rng: np.random.Generator, n_points: int = 12, radius: float = 80.0) -> np.ndarray:
    """Random non-self-intersecting closed layout: noisy radii around sorted angles."""
    angles = np.sort(rng.uniform(0.0, 2 * np.pi, size=n_points))
    # Enforce a minimum angular gap so the spline does not fold back on itself.
    min_gap = 2 * np.pi / (n_points * 2.5)
    for i in range(1, n_points):
        angles[i] = max(angles[i], angles[i - 1] + min_gap)
    angles = angles[angles < 2 * np.pi - min_gap]
    radii = radius * rng.uniform(0.55, 1.15, size=len(angles))
    # Light smoothing of radii avoids hairpins that are narrower than the track.
    radii = 0.5 * radii + 0.25 * (np.roll(radii, 1) + np.roll(radii, -1))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def make_track(
    name: str = "oval",
    width: float = 12.0,
    num_checkpoints: int = 20,
    rng: np.random.Generator | None = None,
) -> Track:
    """Build a built-in track by name, or ``"random"`` for a procedurally generated one."""
    if name == "random":
        rng = rng if rng is not None else np.random.default_rng()
        pts = random_control_points(rng)
    elif name in BUILTIN_TRACKS:
        pts = np.asarray(BUILTIN_TRACKS[name], dtype=np.float64)
    else:
        raise KeyError(f"unknown track '{name}', choose from {sorted(BUILTIN_TRACKS)} or 'random'")
    return Track(pts, width=width, num_checkpoints=num_checkpoints, name=name)
