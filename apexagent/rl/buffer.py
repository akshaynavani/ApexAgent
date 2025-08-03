"""On-policy rollout storage with Generalized Advantage Estimation."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    next_done: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) over a (T, N) rollout.

    ``dones[t]`` marks that the observation at step t is the *first* of a new episode
    (i.e. the previous step ended an episode), matching the CleanRL convention.
    Returns (advantages, returns), both (T, N).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            not_done = 1.0 - next_done.astype(np.float32)
            nv = next_value
        else:
            not_done = 1.0 - dones[t + 1].astype(np.float32)
            nv = values[t + 1]
        delta = rewards[t] + gamma * nv * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        adv[t] = last
    return adv, adv + values


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int) -> None:
        self.T, self.N = num_steps, num_envs
        self.obs = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, act_dim), dtype=np.float32)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=bool)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value) -> None:
        t = self.ptr
        self.obs[t], self.actions[t], self.logprobs[t] = obs, action, logprob
        self.rewards[t], self.dones[t], self.values[t] = reward, done, value
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.T

    def finish(self, next_value: np.ndarray, next_done: np.ndarray, gamma: float, lam: float) -> None:
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, next_value, next_done, gamma, lam
        )
        self.ptr = 0

    def minibatches(self, num_minibatches: int, device: torch.device, rng: np.random.Generator) -> Iterator[dict]:
        B = self.T * self.N
        flat = {
            "obs": self.obs.reshape(B, -1),
            "actions": self.actions.reshape(B, -1),
            "logprobs": self.logprobs.reshape(B),
            "advantages": self.advantages.reshape(B),
            "returns": self.returns.reshape(B),
            "values": self.values.reshape(B),
        }
        idx = rng.permutation(B)
        mb_size = B // num_minibatches
        for start in range(0, mb_size * num_minibatches, mb_size):
            mb = idx[start : start + mb_size]
            yield {k: torch.as_tensor(v[mb], device=device) for k, v in flat.items()}
