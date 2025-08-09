import numpy as np
import torch

from apexagent.config import load_config
from apexagent.factory import env_fn
from apexagent.rl import PPO, PPOConfig, RolloutBuffer, RunningMeanStd, SyncVecEnv, compute_gae


def test_gae_matches_discounted_return_when_lambda_one():
    T, N, gamma = 5, 1, 0.9
    rewards = np.ones((T, N), dtype=np.float32)
    values = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=bool)
    adv, ret = compute_gae(rewards, values, dones, np.zeros(N), np.ones(N, dtype=bool), gamma, 1.0)
    expected = [sum(gamma**k for k in range(T - t)) for t in range(T)]
    assert np.allclose(ret[:, 0], expected)


def test_gae_cuts_at_episode_boundary():
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.array([[False], [False], [True]])  # step 2 starts a new episode
    adv, _ = compute_gae(rewards, values, dones, np.array([100.0]), np.array([False]), 0.99, 0.95)
    # Step 1's advantage must not include anything from step 2 onwards.
    assert adv[1, 0] == np.float32(1.0)


def test_running_mean_std():
    rms = RunningMeanStd((2,))
    data = np.random.default_rng(0).normal(3.0, 2.0, size=(5000, 2))
    for chunk in np.array_split(data, 10):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0), atol=1e-2)
    assert np.allclose(rms.var, data.var(0), atol=5e-2)


def test_ppo_short_training_run_updates_policy():
    cfg = load_config(overrides=["env.tracks=[oval]", "env.max_steps=64"])
    ppo_cfg = PPOConfig(num_envs=2, rollout_steps=32, num_epochs=2, num_minibatches=2, total_timesteps=128)
    envs = SyncVecEnv([env_fn(cfg) for _ in range(2)], seed=0)
    obs_dim, act_dim = envs.observation_space.shape[0], envs.action_space.shape[0]
    agent = PPO(obs_dim, act_dim, ppo_cfg, torch.device("cpu"), seed=0)
    buffer = RolloutBuffer(32, 2, obs_dim, act_dim)
    before = [p.detach().clone() for p in agent.model.parameters()]
    agent.collect_rollout(envs, buffer)
    stats = agent.update(buffer)
    after = list(agent.model.parameters())
    assert agent.global_step == 64
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    action = agent.predict(envs.reset()[0][0])
    assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
    envs.close()
