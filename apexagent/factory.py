"""Builds environments from a run config dict."""

from __future__ import annotations

from apexagent.env.race_env import EnvConfig, RaceEnv
from apexagent.env.reward import RewardConfig


def make_env(cfg: dict, render_mode: str | None = None, env_overrides: dict | None = None) -> RaceEnv:
    env_dict = dict(cfg.get("env", {}))
    env_dict.update(env_overrides or {})
    return RaceEnv(
        config=EnvConfig.from_dict(env_dict),
        reward_config=RewardConfig.from_dict(cfg.get("reward")),
        render_mode=render_mode,
    )


def env_fn(cfg: dict, render_mode: str | None = None, env_overrides: dict | None = None):
    return lambda: make_env(cfg, render_mode, env_overrides)
