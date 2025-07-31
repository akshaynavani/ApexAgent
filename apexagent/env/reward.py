"""Reward engineering for lap completion and track navigation.

The reward is a weighted sum of interpretable terms. Each term is reported separately in the
env ``info`` dict, so training curves show *which* incentive the agent is actually chasing.

Design rationale
- ``progress`` (dense): metres advanced along the centerline. This is the backbone signal; it is
  measured along the track rather than as raw speed so cutting across grass or spinning in place
  earns nothing.
- ``checkpoint`` / ``lap`` (sparse): milestone bonuses that make lap completion the explicit goal.
  Only *new* checkpoints count, so oscillating across a gate cannot be farmed.
- ``speed``: speed projected onto the track direction, rewards committing to fast driving.
- ``centerline``: quadratic penalty on lateral offset; keeps a safety margin early in training.
  Kept small so the agent can still learn to use the full track width (racing line).
- ``smoothness``: penalises steering jerk; reduces the bang-bang steering typical of PPO.
- ``wrong_way``: penalises driving backwards along the track.
- ``crash``: terminal penalty for leaving the track.
- ``time``: constant per-step cost that encourages finishing laps quickly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class RewardConfig:
    progress: float = 0.1  # per metre of centerline progress
    checkpoint: float = 2.0  # per newly reached checkpoint
    lap: float = 25.0  # per completed lap
    speed: float = 0.02  # per step at max speed, aligned with track
    centerline: float = 0.05  # * (lateral / half_width)^2
    smoothness: float = 0.02  # * |delta steer command|
    wrong_way: float = 0.1  # per metre driven backwards
    crash: float = 10.0  # one-off on leaving the track
    time: float = 0.01  # per step

    @classmethod
    def from_dict(cls, d: dict | None) -> "RewardConfig":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown reward weights: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepContext:
    """Everything the shaper needs from one env step."""

    progress_delta: float  # metres along centerline (signed)
    new_checkpoints: int
    new_laps: int
    speed: float
    max_speed: float
    heading_error: float  # car heading minus track heading (rad)
    lateral: float
    half_width: float
    steer_cmd_delta: float
    crashed: bool


class RewardShaper:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()

    def compute(self, ctx: StepContext) -> tuple[float, dict[str, float]]:
        c = self.cfg
        forward = max(ctx.progress_delta, 0.0)
        backward = max(-ctx.progress_delta, 0.0)
        aligned_speed = ctx.speed * np.cos(ctx.heading_error) / max(ctx.max_speed, 1e-6)
        terms = {
            "progress": c.progress * forward,
            "checkpoint": c.checkpoint * ctx.new_checkpoints,
            "lap": c.lap * ctx.new_laps,
            "speed": c.speed * max(aligned_speed, 0.0),
            "centerline": -c.centerline * (ctx.lateral / max(ctx.half_width, 1e-6)) ** 2,
            "smoothness": -c.smoothness * abs(ctx.steer_cmd_delta),
            "wrong_way": -c.wrong_way * backward,
            "crash": -c.crash if ctx.crashed else 0.0,
            "time": -c.time,
        }
        return float(sum(terms.values())), {k: float(v) for k, v in terms.items()}
