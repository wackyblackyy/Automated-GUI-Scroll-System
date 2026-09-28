"""The scan -> (buy) -> scroll -> (refresh) loop."""
from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import vision
from .detector import Found, ShopDetector
from .vision import Match

log = logging.getLogger(__name__)


class StopBot(Exception):
    pass


class ShopBot:
    def __init__(self, cfg: dict, detector: ShopDetector, screen):
        self.cfg = cfg
        self.det = detector
        self.screen = screen
        self.act = cfg["actions"]
        self.scroll_cfg = cfg["scroll"]
        self.totals: Counter[str] = Counter()
        self.bought: Counter[str] = Counter()
        self.refreshes = 0
        self.scroll_up = True     # scan direction for the next page (flips each page)
        self.at_end = False       # False until the list has been seen to stop at an end
        self.csv_path = Path(cfg["log_csv"]) if cfg.get("log_csv") else None

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        if not self.act.get("auto_learn_popups", True):
            if self.act["buy"] and not self.det.has_template("confirm_buy"):
                raise StopBot("buy is enabled but templates/confirm_buy.png is missing (see README)")
            if self.act["refresh"] and not self.det.has_template("confirm_refresh"):
                raise StopBot("refresh is enabled but no confirm template exists (see README)")

        self._require_hud()
        try:
            while True:
                self.scan_page()
                if not self.act["refresh"]:
                    break
                if self.refreshes >= self.act["max_refreshes"]:
                    log.info("Reached max_refreshes (%d)", self.act["max_refreshes"])
                    break
                self.refresh()
        finally:
            self._summary()

    def scan_page(self) -> dict[str, int]:
        """Scan the whole list, scrolling from one end to the other.

        The list is scanned in whichever direction it is already facing: down
        on one page, back up on the next, so no time goes into returning to
        the top. If the list turns out to be at the end we were going to
        scroll towards (e.g. the game reset it to the top after a refresh),
        the bot flips direction after one scroll and remembers that for
        later pages. Returns {item: 1} for each target seen on this page.
        """
        seen: dict[str, Match] = {}
        up = self.scroll_up
        at_end = self.at_end          # do we know the page starts at one end of the list?
        moved = flipped_early = False
        prev = None
        for _ in range(self.scroll_cfg["max_scrolls"] + 1):
            frame = self._require_hud()
            region = self.det.list_region()
            if self._scan_frame(frame, region, seen):
                frame = self.screen.grab()    # a purchase changed the rows; don't mistake it for scrolling
            if len(seen) == len(self.det.targets):
                self.at_end = False   # stopped mid-list; next page must verify both ends
                break
            crop = _crop(frame, region)
            if prev is not None:
                shift = self._list_shift(prev, crop, up)
                still = shift is not None and shift < self._min_shift(crop)
                moved |= not still
                log.debug("flick %s moved the list %s px", "up" if up else "down",
                          "far" if shift is None else int(shift))
                # an end is reached when the list didn't move, or moved less than the flick
                if still or (shift is not None and shift < 0.85 * self._flick_px(crop)):
                    if at_end and moved:
                        self.at_end = True
                        break         # travelled end to end: whole list covered
                    flipped_early |= at_end and not moved
                    at_end, moved, up = True, False, not up
            prev = crop
            self._scroll(region, up=up)

        else:
            log.warning("Scroll limit reached without finding the list ends; refreshing anyway")
            self.at_end = False
        # next page: keep going the way the list resets, otherwise come back the other way
        self.scroll_up = up if flipped_early else not up
        if not seen:
            log.info("Page %d: no target items", self.refreshes)
        return {k: 1 for k in seen}

    def _flick_px(self, crop: np.ndarray) -> float:
        if self.scroll_cfg["method"] != "drag":
            return 0.0
        return (self.scroll_cfg["drag_from"] - self.scroll_cfg["drag_to"]) * crop.shape[0]

    def _min_shift(self, crop: np.ndarray) -> float:
        return max(6.0, 0.02 * crop.shape[0])

    def _list_shift(self, prev: np.ndarray, cur: np.ndarray, up: bool) -> float | None:
        """How far the list content moved in the scroll direction, in pixels.

        Tracks a band of rows (icons, names, prices) between two frames, so the
        game's animated background doesn't count as movement. Returns None
        when the band left the view, i.e. the list moved further than can be
        measured (a full flick).
        """
        h = prev.shape[0]
        bh = h // 4
        g_prev = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        g_cur = cv2.GaussianBlur(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        # follow the band at the edge the content moves away from, so it stays in view
        y0 = 0 if up else h - bh
        band = g_prev[y0:y0 + bh]
        # same place? (checked first: similar-looking rows must not fake a move)
        if cv2.matchTemplate(g_cur[y0:y0 + bh], band, cv2.TM_CCOEFF_NORMED)[0, 0] > 0.9:
            return 0.0
        res = cv2.matchTemplate(g_cur, band, cv2.TM_CCOEFF_NORMED)[:, 0]
        y = int(np.argmax(res))
        if res[y] < 0.8:
            return None
        return float(y - y0 if up else y0 - y)

    def _scan_frame(self, frame: np.ndarray, region, seen: dict[str, Match]) -> bool:
        """Handle new target items in this frame. Returns True if anything was clicked."""
        clicked = False
        for f in self.det.find_targets(frame):
            if f.name in seen:
                continue                # already handled on this page
            if f.buy is None:
                if not self._near_bottom(f.icon, region):
                    log.info("%s visible but already bought / sold out", f.name)
                    seen[f.name] = f.icon
                continue                # else: row cut off at the bottom; caught after scrolling
            seen[f.name] = f.icon
            self._on_found(f)
            clicked |= bool(self.act["buy"])
        return clicked

    # -------------------------------------------------------------- actions
    def _on_found(self, f: Found) -> None:
        log.info(">>> FOUND %s (score %.2f) on page %d", f.name, f.icon.score, self.refreshes)
        self.totals[f.name] += 1
        bought = False
        if self.act["buy"]:
            bought = self.buy(f)
        self._log_csv(f.name, bought)

    def buy(self, f: Found) -> bool:
        # the template is the "1/1" pill; "Buy" sits to its right
        ox, oy = self.cfg["layout"]["buy_click_offset"]
        s = self.det.calibration.scale
        before = self.screen.grab()
        self.screen.click(f.buy.x + int(ox * s), f.buy.y + int(oy * s), f"Buy {f.name}")
        if self.screen.dry_run:
            return False
        if not self._confirm("confirm_buy", before):
            log.warning("Purchase popup for %s was not confirmed (not enough gold?)", f.name)
            return False
        self.bought[f.name] += 1
        log.info("Bought %s", f.name)
        return True

    def refresh(self) -> None:
        frame = self._require_hud()
        btn = self.det.find_button(frame, "refresh")
        if btn is None:
            # the window may have moved or been resized: recalibrate and look everywhere
            log.info("Refresh button not where expected - recalibrating")
            self.det.calibrate(frame)
            btn = self.det.find_button(frame, "refresh", anywhere=True)
        if btn is None:
            raise StopBot("Refresh button not found")
        self.screen.click(*btn.center, "Refresh")
        if not self.screen.dry_run and not self._confirm("confirm_refresh", frame):
            raise StopBot("Refresh popup was not confirmed (out of skystones?)")
        self.refreshes += 1
        self._wait_for_new_list(_crop(frame, self.det.list_region()))
        log.info("Refreshed (%d/%d)", self.refreshes, self.act["max_refreshes"])

    def _scroll(self, region, up: bool = False) -> None:
        """Move the list down (drag finger upward) or, with up=True, back toward the top."""
        x1, y1, x2, y2 = region
        cx = (x1 + x2) // 2
        h = y2 - y1
        if self.scroll_cfg["method"] == "wheel":
            clicks = self.scroll_cfg["wheel_clicks"]
            self.screen.wheel(cx, y1 + h // 2, -clicks if up else clicks)
        else:
            a = int(y1 + h * self.scroll_cfg["drag_from"])
            b = int(y1 + h * self.scroll_cfg["drag_to"])
            if up:
                a, b = b, a
            self.screen.drag(cx, a, b, self.scroll_cfg["drag_duration"])
        time.sleep(self.scroll_cfg["settle"])

    # -------------------------------------------------------------- helpers
    def _require_hud(self) -> np.ndarray:
        for _ in range(5):
            frame = self.screen.grab()
            if self.det.hud_visible(frame):
                return frame
            time.sleep(1.0)
        raise StopBot("Secret Shop screen not visible - open the Secret Shop and try again")

    def _wait_for_new_list(self, old: np.ndarray, timeout: float = 2.5) -> None:
        """Return as soon as the refreshed list is on screen (instead of a fixed sleep)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.screen.grab()
            if self.det.hud_visible(frame):
                new = _crop(frame, self.det.list_region())
                if vision.frame_diff(new, old) > 8:
                    time.sleep(self.scroll_cfg["settle"])   # let the slide-in animation finish
                    return
            time.sleep(0.05)

    def _confirm(self, which: str, before: np.ndarray) -> bool:
        """Wait for the "are you sure?" popup and press its Yes/Buy button.

        Uses the saved template when there is one. Otherwise (first run) it
        looks for a green button that wasn't on screen before the click, and
        once pressing it has closed the popup, saves it as the template.
        """
        deadline = time.time() + self.act["popup_timeout"]
        last: Match | None = None
        while time.time() < deadline:
            time.sleep(0.1)
            frame = self.screen.grab()
            m = self.det.find_button(frame, which)
            if m:
                self.screen.click(*m.center, which)
                time.sleep(self.act["click_delay"])
                return True
            if not self.act.get("auto_learn_popups", True):
                continue
            cand = self.det.find_popup_button(before, frame)
            # wait for the popup's open animation to finish: same box twice in a row
            if cand and last and _same_box(cand, last):
                return self._click_and_learn(which, frame, cand)
            last = cand
        return False

    def _click_and_learn(self, which: str, frame: np.ndarray, m: Match) -> bool:
        log.info("No %s template yet - pressing detected popup button at %s", which, m.center)
        self.screen.click(*m.center, which)
        time.sleep(self.act["click_delay"])
        box = (m.x, m.y, m.x + m.w, m.y + m.h)
        gone = vision.frame_diff(_crop(self.screen.grab(), box), _crop(frame, box)) > 12
        if not gone:
            log.warning("Popup still open after pressing the detected button; not saving a template")
            return False
        log.info("Learned %s template -> %s", which, self.det.learn_template(which, frame, m))
        return True

    def _near_bottom(self, icon: Match, region) -> bool:
        return icon.y + icon.h > region[3] - 60 * self.det.calibration.scale

    def _log_csv(self, name: str, bought: bool) -> None:
        if not self.csv_path:
            return
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "refresh", "item", "bought"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), self.refreshes, name, bought])

    def _summary(self) -> None:
        log.info("---- summary: %d refreshes ----", self.refreshes)
        for name, _, _ in self.det.targets:
            log.info("  %-20s found %d, bought %d", name, self.totals[name], self.bought[name])


def _same_box(a: Match, b: Match, tol: int = 4) -> bool:
    return all(abs(p - q) <= tol for p, q in ((a.x, b.x), (a.y, b.y), (a.w, b.w), (a.h, b.h)))


def _crop(frame: np.ndarray, region) -> np.ndarray:
    x1, y1, x2, y2 = region
    return frame[max(0, y1):y2, max(0, x1):x2]
