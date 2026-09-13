"""Minimal elapsed/ETA progress reporter for training and self-play loops."""

import time


class Progress:
    """平滑的单行动态刷新进度条 (带实时吞吐、耗时与 ETA)."""

    def __init__(self, total: int, label: str, every_s: float = 0.15) -> None:
        self.total = max(total, 1)
        self.label = label
        self.every_s = every_s
        self.t0 = time.time()
        self._last = -1e9

    def update(self, done: int, extra: str = "") -> None:
        now = time.time()
        if now - self._last < self.every_s and done < self.total:
            return
        self._last = now
        elapsed = now - self.t0
        done = min(done, self.total)
        pct = (done / self.total) * 100.0
        rate = done / max(elapsed, 1e-6)
        eta = (self.total - done) / max(rate, 1e-9)

        done_disp = int(done) if isinstance(done, float) else done
        line = f"[{self.label}] {pct:5.1f}% ({done_disp}/{self.total}) elapsed:{elapsed:4.0f}s | ETA:{eta:4.0f}s"
        if extra:
            line += f" | {extra}"

        # \r 覆盖上一行，右侧填充空格保证无残影
        print(f"\r{line}  ".ljust(110), end="", flush=True)

    def done(self, final_msg: str = "") -> None:
        self.update(self.total, extra=final_msg)
        print()
