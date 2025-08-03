"""Proximal Policy Optimization (clipped surrogate) for continuous control."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch
import torch.nn as nn

from apexagent.rl.buffer import RolloutBuffer
from apexagent.rl.networks import ActorCritic
from apexagent.rl.normalizer import ObsNormalizer, RewardScaler
from apexagent.rl.vec_env import SyncVecEnv


@dataclass
class PPOConfig:
    total_timesteps: int = 3_000_000
    num_envs: int = 8
    rollout_steps: int = 256
    num_epochs: int = 10
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    init_log_std: float = -0.5
    normalize_obs: bool = True
    normalize_reward: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "PPOConfig":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown ppo options: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def num_updates(self) -> int:
        return max(self.total_timesteps // self.batch_size, 1)


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig, device: torch.device, seed: int = 0) -> None:
        self.cfg = config
        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.model = ActorCritic(obs_dim, act_dim, config.hidden_sizes, config.init_log_std).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.obs_norm = ObsNormalizer((obs_dim,), enabled=config.normalize_obs)
        self.reward_scaler = RewardScaler(config.num_envs, config.gamma, enabled=config.normalize_reward)
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self.update_idx = 0
        self.episode_stats: deque[dict] = deque(maxlen=100)
        self.reward_term_sums: dict[str, float] = {}
        self._next_obs: np.ndarray | None = None
        self._next_done: np.ndarray | None = None

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True, update_norm: bool = False) -> np.ndarray:
        """Action(s) for raw (unnormalised) observation(s). Used for evaluation and viewing."""
        single = obs.ndim == 1
        o = self.obs_norm(obs[None] if single else obs, update=update_norm)
        action, _, _ = self.model.act(torch.as_tensor(o, device=self.device), deterministic=deterministic)
        a = np.clip(action.cpu().numpy(), -1.0, 1.0)
        return a[0] if single else a

    @torch.no_grad()
    def _value(self, norm_obs: np.ndarray) -> np.ndarray:
        return self.model.value(torch.as_tensor(norm_obs, device=self.device)).cpu().numpy()

    # ------------------------------------------------------------------ rollout
    def collect_rollout(self, envs: SyncVecEnv, buffer: RolloutBuffer) -> None:
        if self._next_obs is None:
            raw, _ = envs.reset()
            self._next_obs = self.obs_norm(raw)
            self._next_done = np.zeros(envs.num_envs, dtype=bool)
        self.reward_term_sums = {}

        while not buffer.full:
            obs_t = torch.as_tensor(self._next_obs, device=self.device)
            with torch.no_grad():
                action, logprob, value = self.model.act(obs_t)
            action_np = action.cpu().numpy()
            raw_next, reward, term, trunc, infos = envs.step(action_np)
            self.global_step += envs.num_envs

            done = term | trunc
            scaled = self.reward_scaler(reward.astype(np.float64), done)
            # Bootstrap through time-limit truncations: the episode did not really end there.
            for i in np.nonzero(trunc & ~term)[0]:
                final = self.obs_norm(infos[i]["final_obs"][None], update=False)
                scaled[i] += self.cfg.gamma * float(self._value(final)[0])

            buffer.add(self._next_obs, action_np, logprob.cpu().numpy(), scaled, self._next_done, value.cpu().numpy())

            for info in infos:
                for k, v in info.get("reward_terms", {}).items():
                    self.reward_term_sums[k] = self.reward_term_sums.get(k, 0.0) + v
                if "episode" in info:
                    self.episode_stats.append(info["episode"])

            self._next_obs = self.obs_norm(raw_next)
            self._next_done = done

        next_value = self._value(self._next_obs)
        buffer.finish(next_value, self._next_done, self.cfg.gamma, self.cfg.gae_lambda)

    # ------------------------------------------------------------------ update
    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        if cfg.anneal_lr:
            frac = 1.0 - self.update_idx / cfg.num_updates
            for g in self.optimizer.param_groups:
                g["lr"] = max(frac, 0.0) * cfg.learning_rate

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        stop = False
        for _epoch in range(cfg.num_epochs):
            for mb in buffer.minibatches(cfg.num_minibatches, self.device, self.rng):
                new_logp, entropy, new_v = self.model.evaluate(mb["obs"], mb["actions"])
                log_ratio = new_logp - mb["logprobs"]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()  # low-variance k3 estimator
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                adv = mb["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()

                if cfg.clip_value:
                    v_clipped = mb["values"] + torch.clamp(new_v - mb["values"], -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max((new_v - mb["returns"]) ** 2, (v_clipped - mb["returns"]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - mb["returns"]) ** 2).mean()

                ent = entropy.mean()
                loss = policy_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append(approx_kl.item())
                stats["clip_frac"].append(clip_frac.item())

                if cfg.target_kl is not None and approx_kl.item() > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        self.update_idx += 1
        y_pred, y_true = buffer.values.reshape(-1), buffer.returns.reshape(-1)
        var_y = np.var(y_true)
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["explained_variance"] = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
        out["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        out["action_std"] = float(self.model.log_std.exp().mean().item())
        out["early_stopped"] = float(stop)
        return out

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
            "global_step": self.global_step,
            "update_idx": self.update_idx,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "ppo_config": self.cfg.to_dict(),
        }

    def load_state_dict(self, d: dict, load_optimizer: bool = True) -> None:
        self.model.load_state_dict(d["model"])
        if load_optimizer and "optimizer" in d:
            self.optimizer.load_state_dict(d["optimizer"])
        self.obs_norm.load_state_dict(d["obs_norm"])
        self.global_step = int(d.get("global_step", 0))
        self.update_idx = int(d.get("update_idx", 0))
