"""Actor-critic MLP with a diagonal Gaussian policy."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim: int, hidden: list[int], out_dim: int, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (no shared trunk; more stable for continuous control)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] = (256, 256), init_log_std: float = -0.5):
        super().__init__()
        hidden = list(hidden_sizes)
        self.actor_mean = mlp(obs_dim, hidden, act_dim, out_std=0.01)
        self.critic = mlp(obs_dim, hidden, 1, out_std=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action. Returns (action, log_prob, value)."""
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Log-prob, entropy and value for given state-action pairs (used in the PPO update)."""
        dist = self.distribution(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.value(obs)
