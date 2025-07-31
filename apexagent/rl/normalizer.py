"""Running statistics for observation and reward normalisation."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running mean / variance."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ObsNormalizer:
    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, enabled: bool = True) -> None:
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.enabled = enabled

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if not self.enabled:
            return obs.astype(np.float32)
        if update:
            self.rms.update(obs.reshape(-1, *self.rms.mean.shape))
        out = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "clip": self.clip, "rms": self.rms.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.enabled = bool(d["enabled"])
        self.clip = float(d["clip"])
        self.rms.load_state_dict(d["rms"])


class RewardScaler:
    """Scales rewards by the std of the discounted return (as in the PPO implementation details).

    Means are not subtracted, so the sign of the reward is preserved.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, enabled: bool = True) -> None:
        self.rms = RunningMeanStd(())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.enabled = enabled

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return rewards.astype(np.float32)
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        self.returns[dones] = 0.0
        scaled = rewards / np.sqrt(self.rms.var + 1e-8)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)
