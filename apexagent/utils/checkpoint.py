"""Saving / loading agent checkpoints together with the config needed to rebuild the env."""

from __future__ import annotations

from pathlib import Path

import torch

from apexagent.rl.ppo import PPO, PPOConfig


def save_checkpoint(agent: PPO, cfg: dict, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent.state_dict(), "config": cfg, "extra": extra or {}}
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[PPO, dict]:
    """Rebuild a PPO agent from a checkpoint. Returns (agent, full run config)."""
    # weights_only=False: the payload contains numpy arrays (normaliser stats) and the config dict.
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["agent"]
    ppo_cfg = PPOConfig.from_dict(state["ppo_config"])
    agent = PPO(state["obs_dim"], state["act_dim"], ppo_cfg, torch.device(device))
    agent.load_state_dict(state)
    return agent, payload["config"]
