"""Kinematic bicycle-model car with simple longitudinal dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apexagent.sim.geometry import wrap_angle


@dataclass
class CarParams:
    wheelbase: float = 2.6  # m
    length: float = 4.4  # m (for rendering / collision footprint)
    width: float = 1.9  # m
    max_steer: float = np.deg2rad(32.0)  # rad, front wheel angle
    steer_rate: float = np.deg2rad(240.0)  # rad/s, actuator slew rate
    max_speed: float = 38.0  # m/s (~137 km/h)
    max_reverse: float = 0.0  # no reversing: agent must recover by steering
    engine_accel: float = 9.0  # m/s^2 at full throttle
    brake_decel: float = 18.0  # m/s^2 at full brake
    drag: float = 0.0045  # quadratic aero drag coefficient
    rolling: float = 0.6  # m/s^2 rolling resistance
    max_lateral_accel: float = 14.0  # m/s^2; above this the car understeers


@dataclass
class Car:
    params: CarParams = field(default_factory=CarParams)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    speed: float = 0.0
    steer: float = 0.0  # current front wheel angle (rad)
    yaw_rate: float = 0.0

    def reset(self, pos: np.ndarray, heading: float, speed: float = 0.0) -> None:
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.heading = float(heading)
        self.speed = float(speed)
        self.steer = 0.0
        self.yaw_rate = 0.0

    def step(self, steer_cmd: float, throttle_cmd: float, dt: float) -> None:
        """Advance the car by ``dt`` seconds.

        steer_cmd in [-1, 1] maps to target wheel angle; throttle_cmd in [-1, 1] where
        positive accelerates and negative brakes.
        """
        p = self.params
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))
        throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))

        # Steering actuator with a slew-rate limit.
        target = steer_cmd * p.max_steer
        max_delta = p.steer_rate * dt
        self.steer += float(np.clip(target - self.steer, -max_delta, max_delta))

        # Longitudinal dynamics.
        if throttle_cmd >= 0.0:
            # Engine power falls off with speed so top speed is approached smoothly.
            accel = throttle_cmd * p.engine_accel * (1.0 - 0.6 * self.speed / p.max_speed)
        else:
            accel = throttle_cmd * p.brake_decel if self.speed > 0.0 else 0.0
        accel -= p.drag * self.speed * abs(self.speed)
        accel -= p.rolling * np.sign(self.speed)
        new_speed = self.speed + accel * dt
        # Rolling resistance/braking must not flip the direction of travel.
        if self.speed > 0.0 and new_speed < 0.0:
            new_speed = 0.0
        self.speed = float(np.clip(new_speed, -p.max_reverse, p.max_speed))

        # Lateral: kinematic yaw rate, capped by tyre grip (simple understeer model).
        yaw_rate = self.speed * np.tan(self.steer) / p.wheelbase
        max_yaw = p.max_lateral_accel / max(abs(self.speed), 1e-3)
        self.yaw_rate = float(np.clip(yaw_rate, -max_yaw, max_yaw))

        self.heading = float(wrap_angle(self.heading + self.yaw_rate * dt))
        self.pos = self.pos + self.speed * dt * np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def lateral_accel(self) -> float:
        return self.speed * self.yaw_rate

    def corners(self) -> np.ndarray:
        """Footprint corners (4, 2) in world coordinates."""
        p = self.params
        hl, hw = p.length / 2.0, p.width / 2.0
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        c, s = np.cos(self.heading), np.sin(self.heading)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.pos
