import numpy as np
import pytest

from apexagent.env import EnvConfig, RaceEnv, RewardConfig, RewardShaper, StepContext


def make(**kw):
    cfg = EnvConfig(tracks=["oval"], random_start=False, **kw)
    return RaceEnv(cfg)


def ctx(**kw):
    base = dict(
        progress_delta=0.0, new_checkpoints=0, new_laps=0, speed=0.0, max_speed=38.0,
        heading_error=0.0, lateral=0.0, half_width=6.0, steer_cmd_delta=0.0, crashed=False,
    )
    base.update(kw)
    return StepContext(**base)


def test_spaces_and_reset():
    env = make()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (env.cfg.num_rays + 6 + len(env.cfg.lookahead),)
    assert info["laps"] == 0


def test_step_api_and_reward_terms():
    env = make()
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(np.array([0.0, 1.0]))
    assert obs.dtype == np.float32
    assert isinstance(r, float)
    assert set(info["reward_terms"]) == set(RewardConfig().to_dict())
    assert r == pytest.approx(sum(info["reward_terms"].values()))


def test_full_left_lock_crashes():
    env = make()
    env.reset(seed=0)
    done, info = False, {}
    for _ in range(500):
        _, _, term, trunc, info = env.step(np.array([-1.0, 1.0]))
        if term or trunc:
            done = True
            break
    assert done and info["crashed"]


def test_stalling_truncates():
    env = make(stall_steps=30)
    env.reset(seed=0)
    for i in range(30):
        _, _, term, trunc, info = env.step(np.array([0.0, 0.0]))
    assert trunc and not term and info["stalled"]


def test_centerline_follower_completes_a_lap():
    """A simple pure-pursuit controller proves laps are achievable and counted."""
    env = make(laps_to_finish=1, max_steps=5000)
    env.reset(seed=0)
    track = env.track
    for _ in range(5000):
        target, _ = track.pose_at(env.proj.s + 12.0)
        to_t = target - env.car.pos
        err = np.arctan2(to_t[1], to_t[0]) - env.car.heading
        err = (err + np.pi) % (2 * np.pi) - np.pi
        steer = np.clip(2.5 * err, -1, 1)
        throttle = 0.6 if env.car.speed < 15 else 0.0
        _, _, term, trunc, info = env.step(np.array([steer, throttle]))
        if term or trunc:
            break
    assert info["finished"] and not info["crashed"]
    assert len(info["lap_times"]) == 1 and info["lap_times"][0] > 0


def test_reward_progress_and_wrong_way():
    shaper = RewardShaper()
    fwd, t_fwd = shaper.compute(ctx(progress_delta=2.0))
    back, t_back = shaper.compute(ctx(progress_delta=-2.0))
    assert t_fwd["progress"] > 0 and t_fwd["wrong_way"] == 0
    assert t_back["progress"] == 0 and t_back["wrong_way"] < 0
    assert fwd > back


def test_reward_crash_and_lap_bonus():
    shaper = RewardShaper(RewardConfig(crash=7.0, lap=11.0))
    _, t = shaper.compute(ctx(crashed=True, new_laps=1))
    assert t["crash"] == -7.0 and t["lap"] == 11.0


def test_reward_centerline_penalty_grows_with_offset():
    shaper = RewardShaper()
    _, a = shaper.compute(ctx(lateral=1.0))
    _, b = shaper.compute(ctx(lateral=5.0))
    assert b["centerline"] < a["centerline"] <= 0


def test_unknown_reward_key_rejected():
    with pytest.raises(KeyError):
        RewardConfig.from_dict({"not_a_term": 1.0})
