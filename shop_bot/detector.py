"""Reads a Secret Shop screenshot: finds the HUD, the target items and their Buy buttons."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import vision
from .vision import Match

log = logging.getLogger(__name__)

# Relative to the item icon (in reference pixels): the row's Buy button lies
# to the right of the icon, roughly level with it.
ROW_ABOVE, ROW_BELOW = 70, 130


@dataclass
class Calibration:
    scale: float
    hud: Match

    def to_screen(self, ref_xy, anchor_ref) -> tuple[int, int]:
        """Map a reference-layout point onto the current frame."""
        x = self.hud.x + (ref_xy[0] - anchor_ref[0]) * self.scale
        y = self.hud.y + (ref_xy[1] - anchor_ref[1]) * self.scale
        return int(round(x)), int(round(y))


@dataclass
class Found:
    name: str
    icon: Match
    buy: Match | None   # None = row visible but no active Buy button (sold out / cut off)


class ShopDetector:
    def __init__(self, cfg: dict, base_dir: str | Path = "."):
        self.cfg = cfg
        self.base = base = Path(base_dir)
        t = cfg["templates"]
        self.hud_tmpl = vision.load_image(base / t["hud"])
        self.buy_tmpl = vision.load_image(base / t["buy_button"])
        self.refresh_tmpl = vision.load_image(base / t["refresh_button"])
        self.confirm_buy_tmpl = _optional(base / t.get("confirm_buy", ""))
        # the refresh popup's "Yes" usually looks like the purchase one; reuse it if not captured
        self.confirm_refresh_tmpl = _optional(base / t.get("confirm_refresh", ""))
        # both popups share the game's dialog style; try one's button for the other until learned
        if self.confirm_refresh_tmpl is None:
            self.confirm_refresh_tmpl = self.confirm_buy_tmpl
        if self.confirm_buy_tmpl is None:
            self.confirm_buy_tmpl = self.confirm_refresh_tmpl
        self.targets = [
            (tg["name"], vision.load_image(base / tg["template"]), float(tg.get("threshold", 0.8)))
            for tg in cfg["targets"]
        ]
        c = cfg.get("calibration_scales", {})
        self.cal_scales = vision.scale_range(c.get("min", 0.35), c.get("max", 2.0), c.get("step", 0.05))
        self.calibration: Calibration | None = None

    # ------------------------------------------------------------------ HUD
    def calibrate(self, frame: np.ndarray) -> Calibration | None:
        """Locate the Secret Shop HUD and infer the UI scale. Coarse then fine search."""
        # coarse pass on a half-size frame (4x fewer pixels), fine pass near the winner
        small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        coarse = vision.find_best(small, self.hud_tmpl, [s * 0.5 for s in self.cal_scales])
        if coarse is None:
            return None
        s0 = coarse.scale * 2
        pad = 12 * s0 + 40
        near = (coarse.x * 2 - pad, coarse.y * 2 - pad,
                (coarse.x + coarse.w) * 2 + pad, (coarse.y + coarse.h) * 2 + pad)
        fine_scales = [s for s in vision.scale_range(s0 - 0.05, s0 + 0.05, 0.01) if s > 0]
        best = vision.find_best(frame, self.hud_tmpl, fine_scales, near)
        if best is None:
            return None
        if best.score < self.cfg["thresholds"]["hud"]:
            log.debug("HUD not found (best %.2f at scale %.2f)", best.score, best.scale)
            return None
        self.calibration = Calibration(best.scale, best)
        log.info("Secret Shop HUD found at (%d,%d), UI scale %.2f, score %.2f",
                 best.x, best.y, best.scale, best.score)
        return self.calibration

    def hud_visible(self, frame: np.ndarray) -> bool:
        """Cheap check at the known scale (falls back to a full calibration)."""
        if self.calibration is None:
            return self.calibrate(frame) is not None
        s, h = self.calibration.scale, self.calibration.hud
        pad = 40
        m = vision.find_best(frame, self.hud_tmpl, [s],
                             (h.x - pad, h.y - pad, h.x + h.w + pad, h.y + h.h + pad))
        if m and m.score >= self.cfg["thresholds"]["hud"]:
            self.calibration.hud = m      # window may have moved
            return True
        return self.calibrate(frame) is not None

    def list_region(self) -> tuple[int, int, int, int]:
        assert self.calibration, "calibrate() first"
        return self._region("list_region")

    def _region(self, name: str) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.cfg["layout"][name]
        a = self.cfg["layout"]["hud_anchor"]
        return (*self.calibration.to_screen((x1, y1), a), *self.calibration.to_screen((x2, y2), a))

    def _scales(self) -> list[float]:
        # calibration is accurate to 0.01, well inside what the templates tolerate
        if self.calibration:
            return [self.calibration.scale]
        return [0.98, 1.0, 1.02]

    # ---------------------------------------------------------------- items
    def find_targets(self, frame: np.ndarray) -> list[Found]:
        """Target items fully inside the list, each paired with its row's Buy button."""
        region = self.list_region() if self.calibration else None
        icons = region
        if region:      # item icons only ever appear in one column of the list
            cx1, cx2 = self.cfg["layout"]["icon_column_x"]
            a = self.cfg["layout"]["hud_anchor"]
            icons = (self.calibration.to_screen((cx1, 0), a)[0], region[1],
                     self.calibration.to_screen((cx2, 0), a)[0], region[3])
        scales = self._scales()
        s = scales[0]
        found: list[Found] = []
        for name, tmpl, thr in self.targets:
            for icon in vision.find_all(frame, tmpl, thr, scales, icons):
                buy_region = (icon.x + icon.w, icon.y - ROW_ABOVE * s,
                              region[2] if region else frame.shape[1], icon.y + ROW_BELOW * s)
                buy = vision.find_best(frame, self.buy_tmpl, scales, buy_region)
                if buy and buy.score < self.cfg["thresholds"]["buttons"]:
                    buy = None
                found.append(Found(name, icon, buy))
        found.sort(key=lambda f: f.icon.y)
        return found

    # -------------------------------------------------------------- buttons
    def find_button(self, frame: np.ndarray, which: str) -> Match | None:
        tmpl = {
            "refresh": self.refresh_tmpl,
            "confirm_buy": self.confirm_buy_tmpl,
            "confirm_refresh": self.confirm_refresh_tmpl,
        }[which]
        if tmpl is None:
            return None
        region = self._region("refresh_region") if which == "refresh" and self.calibration else None
        m = vision.find_best(frame, tmpl, self._scales(), region)
        return m if m and m.score >= self.cfg["thresholds"]["buttons"] else None

    def has_template(self, which: str) -> bool:
        return {"confirm_buy": self.confirm_buy_tmpl,
                "confirm_refresh": self.confirm_refresh_tmpl}[which] is not None

    # ------------------------------------------------ popups without a template
    def find_popup_button(self, before: np.ndarray, after: np.ndarray) -> Match | None:
        """Find the confirm button of a popup that appeared between two frames.

        The game's confirm buttons are bright blue ("Confirm") or green like
        the shop's Buy button. Such blobs that were not already on screen
        before the click belong to the popup; the right-most sizeable one is
        the Confirm button (Cancel sits on the left and is brown).
        """
        if before.shape != after.shape:
            return None
        s = self.calibration.scale if self.calibration else 1.0
        new = _button_mask(after) & ~cv2.dilate(_button_mask(before), np.ones((15, 15), np.uint8))
        new = cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        new = cv2.morphologyEx(new, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(new)
        buttons = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # a button: wider than tall, at least ~120x40 reference px
            if w >= 120 * s and 35 * s <= h <= 160 * s and w > 1.5 * h and area > 0.5 * w * h:
                buttons.append(Match(int(x), int(y), int(w), int(h), area / (w * h), s))
        if not buttons:
            return None
        return max(buttons, key=lambda m: (m.x + m.w, m.w * m.h))

    def learn_template(self, which: str, frame: np.ndarray, m: Match) -> Path:
        """Save a popup button found by find_popup_button so later runs template-match it."""
        crop = frame[m.y:m.y + m.h, m.x:m.x + m.w]
        s = self.calibration.scale if self.calibration else 1.0
        if abs(s - 1.0) > 0.01:
            crop = cv2.resize(crop, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        path = self.base / self.cfg["templates"][which]
        cv2.imwrite(str(path), crop)
        if which == "confirm_buy":
            self.confirm_buy_tmpl = crop
            if self.confirm_refresh_tmpl is None:
                self.confirm_refresh_tmpl = crop
        else:
            self.confirm_refresh_tmpl = crop
        return path


def _button_mask(frame: np.ndarray) -> np.ndarray:
    """Pixels coloured like the game's green (Buy) or bright blue (Confirm) buttons.

    The popup banner behind Confirm is the same hue but more saturated
    (S ~200 vs ~170), hence the saturation ceiling for blue.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (50, 90, 45), (80, 255, 255))
    blue = cv2.inRange(hsv, (98, 120, 55), (116, 190, 255))
    return green | blue


def _optional(path: Path) -> np.ndarray | None:
    if path.suffix == "" or not path.is_file():
        return None
    return vision.load_image(path)
