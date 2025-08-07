"""Real-time pygame visualisation of the race environment.

Draws the track, checkpoint gates, sensor rays, a speed-coloured trail and a HUD with lap timing,
the current action and the per-term reward breakdown, so you can *see* what the reward shaping
is doing to the agent's behaviour.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

WINDOW = (1200, 760)
PANEL_W = 300
MARGIN = 30

BG = (24, 28, 34)
GRASS = (36, 64, 44)
ASPHALT = (62, 66, 72)
BORDER = (230, 230, 230)
KERB = (200, 60, 60)
CHECKPOINT = (240, 200, 60)
FINISH = (255, 255, 255)
RAY = (90, 200, 255)
RAY_HIT = (255, 90, 90)
CAR = (255, 140, 0)
TEXT = (230, 232, 236)
MUTED = (140, 146, 156)
POS = (90, 210, 120)
NEG = (230, 90, 90)


def speed_color(frac: float) -> tuple[int, int, int]:
    """Blue (slow) -> green -> red (fast)."""
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac < 0.5:
        t = frac / 0.5
        return (int(40 * (1 - t)), int(120 + 100 * t), int(255 * (1 - t)))
    t = (frac - 0.5) / 0.5
    return (int(255 * t), int(220 * (1 - t)), 40)


class Renderer:
    def __init__(self, headless: bool = False, fps: int = 20, trail_len: int = 600) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # imported lazily so training does not require pygame

        self.pg = pygame
        pygame.init()
        self.headless = headless
        self.fps = fps
        if headless:
            self.screen = pygame.Surface(WINDOW)
        else:
            self.screen = pygame.display.set_mode(WINDOW)
            pygame.display.set_caption("ApexAgent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 15)
        self.big = pygame.font.SysFont("menlo,consolas,monospace", 26, bold=True)
        self.trail: deque[tuple[np.ndarray, float]] = deque(maxlen=trail_len)
        self._track_id = None
        self._track_surface = None
        self.paused = False
        self.show_rays = True
        self.speed_mult = 1

    # ------------------------------------------------------------------ transforms
    def _fit(self, track) -> None:
        lo, hi = track.bounds()
        avail = np.array([WINDOW[0] - PANEL_W - 2 * MARGIN, WINDOW[1] - 2 * MARGIN], dtype=float)
        span = np.maximum(hi - lo, 1e-6)
        self.scale = float(min(avail / span))
        self.offset = lo
        self.origin = np.array([MARGIN, MARGIN]) + (avail - span * self.scale) / 2.0

    def to_screen(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        xy = (pts - self.offset) * self.scale + self.origin
        xy[:, 1] = WINDOW[1] - xy[:, 1]  # flip y so +y is up
        return xy

    # ------------------------------------------------------------------ static layer
    def _build_track_surface(self, track) -> None:
        pg = self.pg
        self._fit(track)
        surf = pg.Surface(WINDOW)
        surf.fill(BG)
        pg.draw.rect(surf, GRASS, (0, 0, WINDOW[0] - PANEL_W, WINDOW[1]))
        left, right = self.to_screen(track.left), self.to_screen(track.right)
        n = len(left)
        for i in range(n):
            j = (i + 1) % n
            quad = [left[i], left[j], right[j], right[i]]
            pg.draw.polygon(surf, ASPHALT, quad)
        pg.draw.lines(surf, BORDER, True, left.tolist(), 2)
        pg.draw.lines(surf, BORDER, True, right.tolist(), 2)
        # Kerbs on high-curvature sections, on the inside of the corner.
        for i in range(n):
            if abs(track.curvature[i]) > 0.03:
                side = left if track.curvature[i] > 0 else right
                pg.draw.line(surf, KERB, side[i], side[(i + 1) % n], 4)
        self._track_surface = surf
        self._track_id = id(track)

    # ------------------------------------------------------------------ drawing
    def reset_trail(self) -> None:
        self.trail.clear()

    def _handle_events(self) -> None:
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                raise SystemExit(0)
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.close()
                    raise SystemExit(0)
                if event.key == pg.K_SPACE:
                    self.paused = not self.paused
                if event.key == pg.K_r:
                    self.show_rays = not self.show_rays
                if event.key == pg.K_f:
                    self.speed_mult = 1 if self.speed_mult > 1 else 4

    def draw(self, env):
        pg = self.pg
        track, car = env.track, env.car
        if self._track_id != id(track):
            self._build_track_surface(track)
            self.reset_trail()

        if not self.headless:
            self._handle_events()
            while self.paused:
                self._handle_events()
                self.clock.tick(20)

        self.trail.append((car.pos.copy(), car.speed / car.params.max_speed))
        self.screen.blit(self._track_surface, (0, 0))

        # Speed-coloured trail.
        if len(self.trail) > 1:
            pts = self.to_screen(np.array([p for p, _ in self.trail]))
            for i in range(1, len(pts)):
                pg.draw.line(self.screen, speed_color(self.trail[i][1]), pts[i - 1], pts[i], 3)

        # Checkpoint gates. The env counts them from the episode's start position, so the gates
        # (and the start/finish line) move with random starts.
        spacing = env.checkpoint_spacing
        next_cp = env.best_checkpoint + 1
        for k in range(track.num_checkpoints):
            s_k = (env.start_s + k * spacing) % track.length
            idx = int(np.searchsorted(track.cum_s, s_k, side="right") - 1)
            l, r = self.to_screen(track.left[idx])[0], self.to_screen(track.right[idx])[0]
            if k == next_cp % track.num_checkpoints:
                pg.draw.line(self.screen, (120, 255, 160), l, r, 3)
            elif k == 0:
                pg.draw.line(self.screen, FINISH, l, r, 3)
            else:
                pg.draw.line(self.screen, CHECKPOINT, l, r, 1)

        # Sensor rays.
        if self.show_rays:
            origin = self.to_screen(car.pos)[0]
            hits = self.to_screen(env.sensor.endpoints(car.pos, car.heading, env.last_rays))
            for hit, dist in zip(hits, env.last_rays):
                close = dist < env.sensor.max_range * 0.999
                pg.draw.line(self.screen, RAY, origin, hit, 1)
                if close:
                    pg.draw.circle(self.screen, RAY_HIT, hit.astype(int), 3)

        # Car body, drawn at least a few pixels big so it is visible on large tracks.
        corners = car.corners()
        scr = self.to_screen(corners)
        center = scr.mean(axis=0)
        min_px = 10.0
        size = np.linalg.norm(scr[0] - scr[2])
        if size < min_px:
            scr = center + (scr - center) * (min_px / max(size, 1e-6))
        pg.draw.polygon(self.screen, CAR, scr.tolist())
        nose = self.to_screen(car.pos + 0.6 * car.params.length * np.array([np.cos(car.heading), np.sin(car.heading)]))[0]
        pg.draw.line(self.screen, (255, 255, 255), center, nose, 2)

        self._draw_hud(env)

        if self.headless:
            return np.transpose(pg.surfarray.array3d(self.screen), (1, 0, 2))
        pg.display.flip()
        self.clock.tick(self.fps * self.speed_mult)
        return None

    def _text(self, txt: str, x: int, y: int, color=TEXT, font=None) -> int:
        surf = (font or self.font).render(txt, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 4

    def _bar(self, label: str, value: float, x: int, y: int, w: int = 180) -> int:
        pg = self.pg
        self._text(f"{label:<9}", x, y, MUTED)
        bx, bw, bh = x + 90, w, 12
        pg.draw.rect(self.screen, (60, 64, 72), (bx, y + 3, bw, bh))
        mid = bx + bw // 2
        v = float(np.clip(value, -1.0, 1.0))
        px = int(v * bw / 2)
        rect = (mid, y + 3, px, bh) if px >= 0 else (mid + px, y + 3, -px, bh)
        pg.draw.rect(self.screen, POS if v >= 0 else NEG, rect)
        pg.draw.line(self.screen, TEXT, (mid, y + 1), (mid, y + bh + 4), 1)
        return y + 22

    def _draw_hud(self, env) -> None:
        pg = self.pg
        x0 = WINDOW[0] - PANEL_W
        pg.draw.rect(self.screen, (18, 20, 24), (x0, 0, PANEL_W, WINDOW[1]))
        x, y = x0 + 16, 16
        car = env.car
        y = self._text("APEXAGENT", x, y, CAR, self.big)
        y = self._text(f"track    {env.track.name}", x, y, MUTED)
        y += 6
        y = self._text(f"{car.speed * 3.6:6.1f} km/h", x, y, TEXT, self.big)
        y = self._text(f"lap      {env.laps + 1} / {env.cfg.laps_to_finish}", x, y)
        lap_t = (env.steps - env.lap_start_step) * env.cfg.dt * env.cfg.action_repeat
        y = self._text(f"lap time {lap_t:6.2f}s", x, y)
        best = f"{min(env.lap_times):6.2f}s" if env.lap_times else "   --  "
        y = self._text(f"best     {best}", x, y)
        y = self._text(f"progress {env.total_progress / env.track.length:6.2f} laps", x, y)
        y = self._text(f"offset   {env.proj.lateral:+6.2f} m", x, y)
        y = self._text(f"return   {env.episode_return:8.2f}", x, y)
        y += 8
        y = self._text("ACTION", x, y, MUTED)
        y = self._bar("steer", float(env.prev_action[0]), x, y)
        y = self._bar("throttle", float(env.prev_action[1]), x, y)
        y += 8
        y = self._text("REWARD TERMS (this step)", x, y, MUTED)
        for k, v in env.last_reward_terms.items():
            color = POS if v > 0 else NEG if v < 0 else MUTED
            y = self._text(f"{k:<11}{v:+8.3f}", x, y, color)
        y = WINDOW[1] - 70
        for line in ("SPACE pause   R rays", "F fast-forward   ESC quit"):
            y = self._text(line, x, y, MUTED)

    def close(self) -> None:
        try:
            if not self.headless:
                self.pg.display.quit()
            self.pg.quit()
        except Exception:
            pass
