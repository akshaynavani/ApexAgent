"""Minimal synchronous vectorised environment with auto-reset.

Gymnasium ships vector envs too, but their auto-reset semantics changed across versions.
This tiny wrapper pins down the behaviour PPO relies on: when an episode ends, the returned
observation is already the first observation of the next episode, and the true final
observation is available in ``infos[i]["final_obs"]`` for value bootstrapping on truncation.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np


class SyncVecEnv:
    def __init__(self, env_fns: list[Callable[[], gym.Env]], seed: int | None = None) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._seed = seed

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        obs, infos = [], []
        for i, env in enumerate(self.envs):
            seed = None if self._seed is None else self._seed + i
            o, info = env.reset(seed=seed)
            obs.append(o)
            infos.append(info)
        return np.stack(obs), infos

    def step(self, actions: np.ndarray):
        obs, rewards, terms, truncs, infos = [], [], [], [], []
        for env, act in zip(self.envs, actions):
            o, r, term, trunc, info = env.step(act)
            if term or trunc:
                info = dict(info)
                info["final_obs"] = o
                info["episode"] = {
                    "return": info.get("episode_return", 0.0),
                    "length": getattr(env, "steps", 0),
                    "laps": info.get("laps", 0),
                    "lap_times": info.get("lap_times", []),
                    "crashed": info.get("crashed", False),
                    "finished": info.get("finished", False),
                    "track": info.get("track", ""),
                }
                o, _ = env.reset()
            obs.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
