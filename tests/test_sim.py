import numpy as np
import pytest

from apexagent.sim.car import Car
from apexagent.sim.geometry import ray_segment_distances, segments_intersect, wrap_angle
from apexagent.sim.sensors import RaySensor
from apexagent.sim.track import BUILTIN_TRACKS, make_track


def test_wrap_angle_range():
    angles = np.linspace(-10, 10, 101)
    wrapped = wrap_angle(angles)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)
    assert np.allclose(np.sin(wrapped), np.sin(angles))


def test_ray_hits_wall_at_expected_distance():
    origin = np.zeros(2)
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    a = np.array([[5.0, -1.0]])
    b = np.array([[5.0, 1.0]])
    d = ray_segment_distances(origin, dirs, a, b, max_range=50.0)
    assert d[0] == pytest.approx(5.0)
    assert d[1] == pytest.approx(50.0)  # facing away: no hit


def test_segments_intersect():
    assert segments_intersect(np.array([0, 0]), np.array([2, 2]), np.array([0, 2]), np.array([2, 0]))
    assert not segments_intersect(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1]))


@pytest.mark.parametrize("name", list(BUILTIN_TRACKS))
def test_track_projection_on_centerline(name):
    track = make_track(name)
    for s in np.linspace(0, track.length, 17, endpoint=False):
        pos, heading = track.pose_at(s)
        proj = track.project(pos)
        assert abs(proj.lateral) < 1e-6
        assert abs(track.progress_delta(s, proj.s)) < 1e-3
        assert track.is_on_track(proj)


def test_lateral_sign_left_is_positive():
    track = make_track("oval")
    pos, heading = track.pose_at(10.0)
    left = pos + 3.0 * np.array([-np.sin(heading), np.cos(heading)])
    assert track.project(left).lateral == pytest.approx(3.0, abs=0.2)


def test_progress_delta_wraps_finish_line():
    track = make_track("oval")
    assert track.progress_delta(track.length - 1.0, 1.0) == pytest.approx(2.0)
    assert track.progress_delta(1.0, track.length - 1.0) == pytest.approx(-2.0)


def test_random_track_is_valid():
    track = make_track("random", rng=np.random.default_rng(3))
    assert track.length > 100
    assert np.all(np.isfinite(track.left))


def test_car_accelerates_and_turns_left():
    car = Car()
    car.reset(np.zeros(2), 0.0)
    for _ in range(40):
        car.step(0.0, 1.0, 0.05)
    assert car.speed > 5.0
    assert car.pos[0] > 0 and abs(car.pos[1]) < 1e-6
    for _ in range(20):
        car.step(1.0, 0.3, 0.05)
    assert car.heading > 0.1  # positive steer turns counter-clockwise


def test_car_braking_never_reverses():
    car = Car()
    car.reset(np.zeros(2), 0.0, speed=3.0)
    for _ in range(100):
        car.step(0.0, -1.0, 0.05)
    assert car.speed == 0.0


def test_speed_capped():
    car = Car()
    car.reset(np.zeros(2), 0.0)
    for _ in range(2000):
        car.step(0.0, 1.0, 0.05)
    assert car.speed <= car.params.max_speed + 1e-9


def test_sensor_symmetric_on_centerline():
    track = make_track("oval")
    pos, heading = track.pose_at(0.0)  # middle of a straight
    sensor = RaySensor(num_rays=5, fov_deg=180, max_range=60)
    d = sensor.read(pos, heading, track)
    assert d[0] == pytest.approx(track.half_width, rel=0.05)
    assert d[-1] == pytest.approx(track.half_width, rel=0.05)
