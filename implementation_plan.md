# ApexAgent — Implementation Plan

**Goal:** train an agent that autonomously drives a 2D race car around closed tracks and completes
laps, using Proximal Policy Optimization (PPO). The reward is engineered around lap completion and
track navigation, and the agent's behaviour can be watched in real time.

**Stack:** Python 3.10+, NumPy, PyTorch (PPO written from scratch), Gymnasium (env API),
pygame (real-time visualization), TensorBoard (training curves), PyYAML (configs), pytest.

## Key design decisions

- **Custom environment instead of CarRacing-v2.** A lightweight kinematic bicycle-model car on a
  procedurally generated / spline-defined track. Observations are low-dimensional (LiDAR-style
  ray distances + speed + heading error), which keeps PPO training CPU-friendly (minutes, not hours).
- **Continuous actions** `[steer, throttle/brake]` in `[-1, 1]`, Gaussian policy with tanh-free
  clipping and a state-independent learnable log-std.
- **Own PPO implementation** (clipped surrogate, GAE(λ), value clipping, entropy bonus, advantage
  normalisation, observation running-mean normalisation, linear LR annealing) rather than a
  library, so the algorithm is transparent and tweakable.
- **Reward engineering as a separate, composable module** with weights in YAML, so different
  shaping strategies can be compared without touching the env.
- **Visualization decoupled from simulation:** the env produces state; a pygame renderer draws it.
  Headless training runs without pygame installed / without a display.

## Phase 1 — Simulation core: track, car physics, sensors
Deliverables
- Track model: centerline from control points (Catmull-Rom spline), width, left/right borders,
  arc-length parameterisation, projection of a point onto the centerline (progress + lateral offset),
  checkpoint gates. A few built-in tracks + a random track generator.
- Car model: kinematic bicycle model with throttle/brake, drag, steering limits, max speed.
- Ray-cast sensors against track borders (segment intersection, vectorised in NumPy).

Files: `pyproject.toml`, `requirements.txt`, `apexagent/__init__.py`, `apexagent/sim/track.py`,
`apexagent/sim/car.py`, `apexagent/sim/sensors.py`, `apexagent/sim/geometry.py`

## Phase 2 — Gymnasium environment + reward engineering
Deliverables
- `RaceEnv(gymnasium.Env)`: observation/action spaces, reset (optionally random start & random
  track), step, termination on crash / lap target, truncation on timeout or no-progress stall.
- `RewardShaper`: progress along centerline, checkpoint bonus, lap bonus, speed bonus,
  centerline-offset penalty, heading alignment, steering-smoothness penalty, crash penalty,
  per-step time penalty. Weights loaded from config; per-term breakdown in `info`.
- Lap timing & lap counting.

Files: `apexagent/env/race_env.py`, `apexagent/env/reward.py`, `apexagent/env/__init__.py`,
`configs/default.yaml`

## Phase 3 — PPO from scratch
Deliverables
- Actor-critic MLP (separate heads, orthogonal init), Gaussian policy.
- Rollout buffer with GAE(λ), handling terminations vs truncations (bootstrap on truncation).
- Synchronous vectorised env wrapper, running mean/std observation normaliser.
- PPO update: clipped objective, value clipping, entropy bonus, minibatch epochs, grad clipping,
  approx-KL early stopping.

Files: `apexagent/rl/networks.py`, `apexagent/rl/buffer.py`, `apexagent/rl/normalizer.py`,
`apexagent/rl/vec_env.py`, `apexagent/rl/ppo.py`

## Phase 4 — Training pipeline & evaluation
Deliverables
- YAML config loader with CLI overrides, seeding.
- `train.py` CLI: training loop, TensorBoard logging (returns, lap counts, reward terms, losses,
  KL), periodic checkpointing (model + normaliser stats), best-model tracking.
- `evaluate.py` CLI: deterministic rollouts, lap completion rate, best/mean lap time.

Files: `apexagent/config.py`, `apexagent/train.py`, `apexagent/evaluate.py`,
`apexagent/utils/logger.py`, `apexagent/utils/checkpoint.py`, `apexagent/utils/seeding.py`

## Phase 5 — Real-time visualization & tests
Deliverables
- pygame renderer: track, checkpoints, car sprite, sensor rays, trail coloured by speed, HUD
  (speed, lap, lap time, reward breakdown, action bars).
- `watch.py`: load a checkpoint and watch the agent drive live; manual-drive mode with keyboard
  for testing the env and reward; optional live viewing during training (`--render-every`).
- Unit tests for geometry, track projection, car physics, env API, reward terms, GAE, PPO update.

Files: `apexagent/viz/renderer.py`, `apexagent/watch.py`, `tests/*.py`
