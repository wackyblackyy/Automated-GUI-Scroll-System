"""Screen capture and mouse input. Imports mss/pyautogui lazily so the rest
of the package (and the tests) work on machines without a display."""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


class ScreenController:
    """Captures one monitor; all coordinates are relative to that monitor."""

    def __init__(self, monitor: int = 1, dry_run: bool = False):
        import mss
        import pyautogui

        pyautogui.FAILSAFE = True    # slam the mouse into a screen corner to abort
        pyautogui.PAUSE = 0.01        # snappy: minimal gap between mouse actions
        self._pg = pyautogui
        self._sct = mss.mss()
        self.mon = self._sct.monitors[monitor]
        self.dry_run = dry_run

    def grab(self) -> np.ndarray:
        shot = np.asarray(self._sct.grab(self.mon))      # BGRA
        return np.ascontiguousarray(shot[:, :, :3])

    def _abs(self, x: int, y: int) -> tuple[int, int]:
        return self.mon["left"] + x, self.mon["top"] + y

    def click(self, x: int, y: int, label: str = "") -> None:
        ax, ay = self._abs(x, y)
        if self.dry_run:
            log.info("[dry-run] would click %s at (%d,%d)", label, ax, ay)
            return
        log.debug("click %s at (%d,%d)", label, ax, ay)
        self._pg.click(ax, ay)

    def drag(self, x: int, y1: int, y2: int, duration: float) -> None:
        ax, ay1 = self._abs(x, y1)
        _, ay2 = self._abs(x, y2)
        self._pg.moveTo(ax, ay1)
        self._pg.mouseDown()
        self._pg.moveTo(ax, ay2, duration=duration)
        time.sleep(0.05)             # brief hold so the list stops where the drag ends
        self._pg.mouseUp()

    def wheel(self, x: int, y: int, clicks: int) -> None:
        ax, ay = self._abs(x, y)
        self._pg.moveTo(ax, ay)
        self._pg.scroll(clicks)


def list_monitors() -> list[dict]:
    import mss
    return mss.mss().monitors
