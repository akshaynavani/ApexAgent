"""Train a PPO driver.

    python -m apexagent.train                                  # default config
    python -m apexagent.train --config configs/gp.yaml
    python -m apexagent.train --set ppo.total_timesteps=500000 env.tracks=[oval]
    python -m apexagent.train --resume runs/<run>/last.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from apexagent.config import load_config, save_config
from apexagent.evaluate import evaluate
from apexagent.factory import env_fn
from apexagent.rl.buffer import RolloutBuffer
from apexagent.rl.normalizer import RewardScaler
from apexagent.rl.ppo import PPO, PPOConfig
from apexagent.rl.vec_env import SyncVecEnv
from apexagent.utils.checkpoint import load_checkpoint, save_checkpoint
from apexagent.utils.logger import Logger
from apexagent.utils.seeding import resolve_device, seed_everything


def episode_summary(agent: PPO) -> dict[str, float]:
    eps = list(agent.episode_stats)
    if not eps:
        return {}
    lap_times = [t for e in eps for t in e["lap_times"]]
    out = {
        "ep_return": float(np.mean([e["return"] for e in eps])),
        "ep_length": float(np.mean([e["length"] for e in eps])),
        "ep_laps": float(np.mean([e["laps"] for e in eps])),
        "lap_rate": float(np.mean([e["laps"] >= 1 for e in eps])),
        "finish_rate": float(np.mean([e["finished"] for e in eps])),
        "crash_rate": float(np.mean([e["crashed"] for e in eps])),
    }
    if lap_times:
        out["best_lap"] = float(min(lap_times))
        out["mean_lap"] = float(np.mean(lap_times))
    return out


def eval_score(summary: dict[str, float]) -> float:
    """Model selection: finishing first, then laps, then lap time (lower is better)."""
    score = 1000.0 * summary.get("finish_rate", 0.0) + 100.0 * summary.get("laps", 0.0)
    if "mean_lap" in summary:
        score -= summary["mean_lap"]
    return score


def train(cfg: dict, resume: str | None = None) -> Path:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(cfg.get("device", "auto"))

    run_name = cfg.get("run_name") or f"{'-'.join(cfg['env']['tracks'])}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.get("log_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    ppo_cfg = PPOConfig.from_dict(cfg["ppo"])
    envs = SyncVecEnv([env_fn(cfg) for _ in range(ppo_cfg.num_envs)], seed=seed)
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]

    if resume:
        agent, _ = load_checkpoint(resume, device)
        agent.cfg = ppo_cfg
        agent.reward_scaler = RewardScaler(ppo_cfg.num_envs, ppo_cfg.gamma, enabled=ppo_cfg.normalize_reward)
        print(f"resumed from {resume} at step {agent.global_step:,}")
    else:
        agent = PPO(obs_dim, act_dim, ppo_cfg, device, seed=seed)
    buffer = RolloutBuffer(ppo_cfg.rollout_steps, ppo_cfg.num_envs, obs_dim, act_dim)

    logger = Logger(run_dir)
    logger.text("config", f"```yaml\n{yaml.safe_dump(cfg, sort_keys=False)}\n```")
    tcfg = cfg.get("train", {})
    best_score = -float("inf")
    print(f"training on {device} | {ppo_cfg.num_updates} updates of {ppo_cfg.batch_size} steps -> {run_dir}")

    try:
        while agent.update_idx < ppo_cfg.num_updates:
            agent.collect_rollout(envs, buffer)
            stats = agent.update(buffer)
            u, step = agent.update_idx, agent.global_step

            summary = episode_summary(agent)
            logger.scalars("train", stats, step)
            logger.scalars("episode", summary, step)
            n_steps = ppo_cfg.batch_size
            logger.scalars("reward_terms", {k: v / n_steps for k, v in agent.reward_term_sums.items()}, step)
            logger.console(u, ppo_cfg.num_updates, step, {**summary, **stats})

            if tcfg.get("eval_every") and u % tcfg["eval_every"] == 0:
                res = evaluate(agent, cfg, episodes=tcfg.get("eval_episodes", 3), seed=seed + 10_000)
                ev = res.summary()
                logger.scalars("eval", ev, step)
                score = eval_score(ev)
                if score > best_score:
                    best_score = score
                    save_checkpoint(agent, cfg, run_dir / "best.pt", extra={"eval": ev})
                    print(f"  new best eval: {json.dumps({k: round(v, 3) for k, v in ev.items()})}")

            if tcfg.get("checkpoint_every") and u % tcfg["checkpoint_every"] == 0:
                save_checkpoint(agent, cfg, run_dir / "last.pt")

            if tcfg.get("render_every") and u % tcfg["render_every"] == 0:
                # Live peek at the current policy; blocks training while the episode plays.
                evaluate(agent, cfg, episodes=1, tracks=[cfg["env"]["tracks"][0]], render_mode="human")
    except KeyboardInterrupt:
        print("interrupted, saving checkpoint")
    finally:
        save_checkpoint(agent, cfg, run_dir / "last.pt")
        envs.close()
        logger.close()
    print(f"done. checkpoints in {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the ApexAgent PPO race driver")
    p.add_argument("--config", default=None, help="YAML file merged over configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--resume", default=None, help="checkpoint to continue training from")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
