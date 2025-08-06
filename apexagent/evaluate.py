"""Deterministic evaluation of a trained agent: lap completion rate and lap times.

    python -m apexagent.evaluate runs/<run>/best.pt --episodes 10 --track kidney
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from apexagent.factory import make_env
from apexagent.rl.ppo import PPO


@dataclass
class EvalResult:
    episodes: int
    mean_return: float
    mean_laps: float
    finish_rate: float  # fraction of episodes that completed laps_to_finish
    crash_rate: float
    best_lap: float | None
    mean_lap: float | None
    per_track: dict = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        out = {
            "return": self.mean_return,
            "laps": self.mean_laps,
            "finish_rate": self.finish_rate,
            "crash_rate": self.crash_rate,
        }
        if self.best_lap is not None:
            out["best_lap"] = self.best_lap
            out["mean_lap"] = self.mean_lap
        return out


def evaluate(
    agent: PPO,
    cfg: dict,
    episodes: int = 5,
    tracks: list[str] | None = None,
    seed: int = 10_000,
    render_mode: str | None = None,
    deterministic: bool = True,
) -> EvalResult:
    """Run ``episodes`` per track (or across the config's tracks) without updating normaliser stats."""
    tracks = tracks or list(cfg["env"]["tracks"])
    env = make_env(cfg, render_mode=render_mode, env_overrides={"random_start": False})
    returns, laps, finished, crashed, lap_times = [], [], [], [], []
    per_track: dict[str, dict] = {}
    try:
        for t_idx, track in enumerate(tracks):
            t_laps, t_times = [], []
            for ep in range(episodes):
                obs, info = env.reset(seed=seed + 1000 * t_idx + ep, options={"track": track})
                done = False
                while not done:
                    action = agent.predict(obs, deterministic=deterministic)
                    obs, _, term, trunc, info = env.step(action)
                    done = term or trunc
                returns.append(info["episode_return"])
                laps.append(info["laps"])
                finished.append(info.get("finished", False))
                crashed.append(info.get("crashed", False))
                lap_times.extend(info["lap_times"])
                t_laps.append(info["laps"])
                t_times.extend(info["lap_times"])
            per_track[track] = {
                "mean_laps": float(np.mean(t_laps)),
                "best_lap": float(min(t_times)) if t_times else None,
            }
    finally:
        env.close()

    return EvalResult(
        episodes=len(returns),
        mean_return=float(np.mean(returns)),
        mean_laps=float(np.mean(laps)),
        finish_rate=float(np.mean(finished)),
        crash_rate=float(np.mean(crashed)),
        best_lap=float(min(lap_times)) if lap_times else None,
        mean_lap=float(np.mean(lap_times)) if lap_times else None,
        per_track=per_track,
    )


def main(argv: list[str] | None = None) -> None:
    from apexagent.utils.checkpoint import load_checkpoint

    p = argparse.ArgumentParser(description="Evaluate a trained ApexAgent checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--track", action="append", help="track(s) to evaluate on; defaults to training tracks")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--render", action="store_true", help="watch the evaluation live (pygame)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    args = p.parse_args(argv)

    agent, cfg = load_checkpoint(args.checkpoint)
    res = evaluate(
        agent,
        cfg,
        episodes=args.episodes,
        tracks=args.track,
        render_mode="human" if args.render else None,
        deterministic=not args.stochastic,
    )
    if args.json:
        print(json.dumps(asdict(res), indent=2))
        return
    print(f"episodes      {res.episodes}")
    print(f"mean return   {res.mean_return:.2f}")
    print(f"mean laps     {res.mean_laps:.2f}")
    print(f"finish rate   {res.finish_rate:.0%}")
    print(f"crash rate    {res.crash_rate:.0%}")
    if res.best_lap is not None:
        print(f"best lap      {res.best_lap:.2f}s   mean lap {res.mean_lap:.2f}s")
    for track, stats in res.per_track.items():
        best = f"{stats['best_lap']:.2f}s" if stats["best_lap"] is not None else "-"
        print(f"  {track:<10} laps {stats['mean_laps']:.2f}  best {best}")


if __name__ == "__main__":
    main()
