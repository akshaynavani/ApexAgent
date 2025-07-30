"""Small vectorised 2D geometry helpers used by the track and sensors."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (radians) into [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return vec @ rot.T


def catmull_rom_closed(points: np.ndarray, samples_per_segment: int = 20) -> np.ndarray:
    """Sample a closed centripetal-free (uniform) Catmull-Rom spline through ``points``.

    Returns an (N, 2) array of points, without repeating the first point at the end.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        raise ValueError("need at least 3 control points for a closed spline")
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(seg)
    return np.concatenate(out, axis=0)


def resample_closed(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline to roughly uniform arc-length spacing."""
    closed = np.vstack([polyline, polyline[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(round(total / spacing)), 8)
    s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def polyline_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents and left-hand normals of a closed polyline (central differences)."""
    tang = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return tang, normals


def ray_segment_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Distance along each ray to the nearest segment hit (clipped to ``max_range``).

    origin: (2,), directions: (R, 2) unit vectors, seg_a/seg_b: (S, 2) segment endpoints.
    Solves origin + t*d = a + u*(b-a) for every (ray, segment) pair at once.
    """
    d = directions[:, None, :]  # (R,1,2)
    e = (seg_b - seg_a)[None, :, :]  # (1,S,2)
    w = (seg_a - origin)[None, :, :]  # (1,S,2)
    denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]  # (R,S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / denom
        u = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / denom
    valid = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(valid, t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """True if segment p1-p2 properly intersects segment q1-q2."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    return (d1 * d2 < 0.0) and (d3 * d4 < 0.0)
