"""Watch a trained agent drive in real time, or drive yourself to sanity-check the env.

    python -m apexagent.watch runs/<run>/best.pt                 # agent drives
    python -m apexagent.watch runs/<run>/best.pt --track apex_gp
    python -m apexagent.watch --manual --track kidney            # arrow keys / WASD
    python -m apexagent.watch runs/<run>/best.pt --record out    # save frames as PNGs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apexagent.config import load_config
from apexagent.factory import make_env


def keyboard_action(pg) -> np.ndarray:
    keys = pg.key.get_pressed()
    steer = float(keys[pg.K_LEFT] or keys[pg.K_a]) - float(keys[pg.K_RIGHT] or keys[pg.K_d])
    throttle = float(keys[pg.K_UP] or keys[pg.K_w]) - float(keys[pg.K_DOWN] or keys[pg.K_s])
    return np.array([steer, throttle], dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Real-time visualisation of ApexAgent")
    p.add_argument("checkpoint", nargs="?", help="trained checkpoint (.pt); omit with --manual")
    p.add_argument("--manual", action="store_true", help="drive with the keyboard instead of the agent")
    p.add_argument("--track", default=None, help="track name (oval, kidney, apex_gp, random)")
    p.add_argument("--episodes", type=int, default=0, help="number of episodes (0 = loop forever)")
    p.add_argument("--stochastic", action="store_true", help="sample from the policy instead of its mean")
    p.add_argument("--random-start", action="store_true", help="start at a random point on the track")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--record", default=None, help="directory to dump PNG frames into")
    args = p.parse_args(argv)

    if not args.manual and not args.checkpoint:
        p.error("pass a checkpoint, or --manual to drive yourself")

    agent = None
    if args.checkpoint:
        from apexagent.utils.checkpoint import load_checkpoint

        agent, cfg = load_checkpoint(args.checkpoint)
    else:
        cfg = load_config()

    overrides = {"random_start": args.random_start}
    if args.manual:
        overrides.update({"off_track_terminates": False, "stall_steps": 10**9, "max_steps": 10**9})
    env = make_env(cfg, render_mode="human", env_overrides=overrides)
    env.metadata["render_fps"] = args.fps

    record_dir = Path(args.record) if args.record else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
    frame = 0

    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            options = {"track": args.track} if args.track else None
            obs, info = env.reset(options=options)
            env.render()
            env._renderer.fps = args.fps
            done = False
            while not done:
                if args.manual:
                    action = keyboard_action(env._renderer.pg)
                else:
                    action = agent.predict(obs, deterministic=not args.stochastic)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                if record_dir:
                    env._renderer.pg.image.save(env._renderer.screen, str(record_dir / f"frame_{frame:06d}.png"))
                    frame += 1
            episode += 1
            laps = ", ".join(f"{t:.2f}s" for t in info["lap_times"]) or "none"
            reason = "finished" if info.get("finished") else "crashed" if info.get("crashed") else "timeout/stall"
            print(f"episode {episode}: {reason} | return {info['episode_return']:.1f} | laps: {laps}")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
