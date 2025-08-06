"""TensorBoard + console logging."""

from __future__ import annotations

import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.start = time.time()

    def scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(tag, text, step)

    def console(self, update: int, total: int, step: int, summary: dict[str, float]) -> None:
        elapsed = time.time() - self.start
        sps = step / max(elapsed, 1e-6)
        parts = [f"upd {update:>4}/{total}", f"step {step:>9,}", f"{sps:6.0f} sps"]
        for k in ("ep_return", "ep_laps", "lap_rate", "best_lap", "crash_rate", "approx_kl", "action_std"):
            if k in summary:
                parts.append(f"{k} {summary[k]:.3f}")
        print(" | ".join(parts), flush=True)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
