"""Gymnasium environment: drive a car around a closed track as fast as possible."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from apexagent.env.reward import RewardConfig, RewardShaper, StepContext
from apexagent.sim.car import Car, CarParams
from apexagent.sim.geometry import wrap_angle
from apexagent.sim.sensors import RaySensor
from apexagent.sim.track import Track, make_track


@dataclass
class EnvConfig:
    tracks: list[str] = field(default_factory=lambda: ["oval"])  # sampled per episode; "random" allowed
    track_width: float = 12.0
    num_checkpoints: int = 20
    dt: float = 0.05  # physics step (s)
    action_repeat: int = 2  # physics steps per agent action
    max_steps: int = 2000  # agent steps before truncation
    laps_to_finish: int = 2  # episode terminates successfully after this many laps
    num_rays: int = 11
    ray_fov_deg: float = 180.0
    ray_range: float = 60.0
    lookahead: list[float] = field(default_factory=lambda: [10.0, 25.0, 45.0])  # m, curvature preview
    random_start: bool = True  # start at a random point on the centerline
    start_speed: float = 0.0
    stall_steps: int = 80  # truncate if no new checkpoint for this many steps (x checkpoint)
    off_track_terminates: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown env options: {sorted(unknown)}")
        d = dict(d)
        if isinstance(d.get("tracks"), str):
            d["tracks"] = [d["tracks"]]
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


class RaceEnv(gym.Env):
    """Single-car racing environment.

    Observation (float32):
        [ray distances (normalised) x num_rays,
         speed / max_speed, wheel angle / max_steer, yaw rate (scaled),
         sin(heading error), cos(heading error), lateral offset / half width,
         curvature at each lookahead distance (scaled)]

    Action (float32, [-1, 1]^2): [steering, throttle(+)/brake(-)].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        car_params: CarParams | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.shaper = RewardShaper(reward_config)
        self.car = Car(params=car_params or CarParams())
        self.sensor = RaySensor(self.cfg.num_rays, self.cfg.ray_fov_deg, self.cfg.ray_range)
        self.render_mode = render_mode
        self._renderer = None
        self._track_cache: dict[str, Track] = {}

        obs_dim = self.cfg.num_rays + 6 + len(self.cfg.lookahead)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.track: Track | None = None
        self._reset_state()

    # ------------------------------------------------------------------ helpers
    def _reset_state(self) -> None:
        self.steps = 0
        self.proj = None
        self.prev_s = 0.0
        self.total_progress = 0.0  # signed metres along centerline since start
        self.best_checkpoint = 0  # highest checkpoint count reached
        self.laps = 0
        self.lap_start_step = 0
        self.lap_times: list[float] = []
        self.last_checkpoint_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_rays = np.full(self.cfg.num_rays, self.cfg.ray_range)
        self.last_reward_terms: dict[str, float] = {}
        self.episode_return = 0.0

    def _get_track(self, name: str) -> Track:
        if name == "random":
            return make_track("random", self.cfg.track_width, self.cfg.num_checkpoints, rng=self.np_random)
        if name not in self._track_cache:
            self._track_cache[name] = make_track(name, self.cfg.track_width, self.cfg.num_checkpoints)
        return self._track_cache[name]

    @property
    def checkpoint_spacing(self) -> float:
        return self.track.length / self.track.num_checkpoints

    def _observation(self) -> np.ndarray:
        car, track, proj = self.car, self.track, self.proj
        self.last_rays = self.sensor.read(car.pos, car.heading, track)
        heading_err = wrap_angle(car.heading - proj.heading)
        curv = track.curvature_ahead(proj.index, self.cfg.lookahead) * 20.0
        obs = np.concatenate(
            [
                self.last_rays / self.sensor.max_range,
                [
                    car.speed / car.params.max_speed,
                    car.steer / car.params.max_steer,
                    car.yaw_rate / 2.0,
                    np.sin(heading_err),
                    np.cos(heading_err),
                    proj.lateral / track.half_width,
                ],
                np.clip(curv, -3.0, 3.0),
            ]
        )
        return obs.astype(np.float32)

    def _info(self) -> dict:
        return {
            "track": self.track.name,
            "laps": self.laps,
            "lap_times": list(self.lap_times),
            "progress": self.total_progress / self.track.length,  # in laps
            "speed": self.car.speed,
            "lateral": self.proj.lateral,
            "reward_terms": dict(self.last_reward_terms),
            "episode_return": self.episode_return,
        }

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self._reset_state()

        track_name = options.get("track") or self.np_random.choice(self.cfg.tracks)
        self.track = self._get_track(str(track_name))

        if "start_s" in options:
            start_s = float(options["start_s"])
        elif self.cfg.random_start:
            start_s = float(self.np_random.uniform(0.0, self.track.length))
        else:
            start_s = 0.0
        pos, heading = self.track.pose_at(start_s)
        self.car.reset(pos, heading, speed=self.cfg.start_speed)

        self.proj = self.track.project(self.car.pos)
        self.prev_s = self.proj.s
        if self._renderer is not None:
            self._renderer.reset_trail()
        return self._observation(), self._info()

    def step(self, action):
        assert self.track is not None, "call reset() before step()"
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        steer_cmd, throttle_cmd = float(action[0]), float(action[1])

        crashed = False
        progress_delta = 0.0
        for _ in range(self.cfg.action_repeat):
            self.car.step(steer_cmd, throttle_cmd, self.cfg.dt)
            self.proj = self.track.project(self.car.pos, hint=self.proj.index)
            ds = self.track.progress_delta(self.prev_s, self.proj.s)
            self.prev_s = self.proj.s
            progress_delta += ds
            if not self.track.is_on_track(self.proj, margin=self.car.params.width / 2.0):
                crashed = True
                break
        self.steps += 1
        self.total_progress += progress_delta

        # Milestones only count the first time they are reached (no farming by reversing).
        reached = int(np.floor(self.total_progress / self.checkpoint_spacing))
        new_checkpoints = max(reached - self.best_checkpoint, 0)
        if new_checkpoints:
            self.best_checkpoint = reached
            self.last_checkpoint_step = self.steps
        laps_now = self.best_checkpoint // self.track.num_checkpoints
        new_laps = max(laps_now - self.laps, 0)
        if new_laps:
            self.laps = laps_now
            self.lap_times.append((self.steps - self.lap_start_step) * self.cfg.dt * self.cfg.action_repeat)
            self.lap_start_step = self.steps

        heading_err = float(wrap_angle(self.car.heading - self.proj.heading))
        ctx = StepContext(
            progress_delta=progress_delta,
            new_checkpoints=new_checkpoints,
            new_laps=new_laps,
            speed=self.car.speed,
            max_speed=self.car.params.max_speed,
            heading_error=heading_err,
            lateral=self.proj.lateral,
            half_width=self.track.half_width,
            steer_cmd_delta=steer_cmd - float(self.prev_action[0]),
            crashed=crashed and self.cfg.off_track_terminates,
        )
        reward, terms = self.shaper.compute(ctx)
        self.last_reward_terms = terms
        self.episode_return += reward
        self.prev_action = action

        terminated = (crashed and self.cfg.off_track_terminates) or self.laps >= self.cfg.laps_to_finish
        stalled = (self.steps - self.last_checkpoint_step) >= self.cfg.stall_steps
        truncated = (self.steps >= self.cfg.max_steps or stalled) and not terminated

        obs = self._observation()
        info = self._info()
        info["crashed"] = crashed
        info["finished"] = self.laps >= self.cfg.laps_to_finish
        info["stalled"] = stalled
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from apexagent.viz.renderer import Renderer  # lazy: pygame is optional

            self._renderer = Renderer(headless=self.render_mode == "rgb_array")
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
