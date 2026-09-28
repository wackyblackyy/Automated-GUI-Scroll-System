"""The scan -> (buy) -> scroll -> (refresh) loop."""
from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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
            self.scroll_to_top()        # the list may have been left scrolled down
            while True:
                if self.refreshes and not self.scroll_cfg["top_after_refresh"]:
                    self.scroll_to_top()
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
        """Scan the whole list top to bottom. Returns {item: count} seen on this page."""
        seen_this_page: dict[str, Match] = {}
        prev_list = None
        for step in range(self.scroll_cfg["max_scrolls"] + 1):
            frame = self._require_hud()
            region = self.det.list_region()
            for f in self.det.find_targets(frame):
                if f.name in seen_this_page:
                    continue            # already handled this item (seen before a scroll)
                if f.buy is None and not self._near_bottom(f.icon, region):
                    log.info("%s visible but no active Buy button (sold out?)", f.name)
                    seen_this_page[f.name] = f.icon
                    continue
                if f.buy is None:
                    continue            # row cut off at the bottom; catch it after scrolling
                seen_this_page[f.name] = f.icon
                self._on_found(f)

            if len(seen_this_page) == len(self.det.targets):
                break                   # everything we want is already found on this page

            crop = _crop(frame, region)
            if prev_list is not None and vision.frame_diff(crop, prev_list) < self.scroll_cfg["end_diff"]:
                log.debug("list stopped moving -> bottom reached after %d scrolls", step)
                break
            prev_list = crop
            self._scroll(region)

        if not seen_this_page:
            log.info("Page %d: no target items", self.refreshes)
        return {k: 1 for k in seen_this_page}

    def scroll_to_top(self) -> None:
        prev = None
        for _ in range(self.scroll_cfg["max_scrolls"]):
            frame = self._require_hud()
            region = self.det.list_region()
            crop = _crop(frame, region)
            if prev is not None and vision.frame_diff(crop, prev) < self.scroll_cfg["end_diff"]:
                return
            prev = crop
            self._scroll(region, up=True)

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
        time.sleep(0.5)
        self.bought[f.name] += 1
        log.info("Bought %s", f.name)
        return True

    def refresh(self) -> None:
        frame = self._require_hud()
        btn = self.det.find_button(frame, "refresh")
        if btn is None:
            raise StopBot("Refresh button not found")
        self.screen.click(*btn.center, "Refresh")
        if not self.screen.dry_run and not self._confirm("confirm_refresh", frame):
            raise StopBot("Refresh popup was not confirmed (out of skystones?)")
        self.refreshes += 1
        time.sleep(self.act["click_delay"] + 0.7)   # let the new list animate in
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

    def _confirm(self, which: str, before: np.ndarray) -> bool:
        """Wait for the "are you sure?" popup and press its Yes/Buy button.

        Uses the saved template when there is one. Otherwise (first run) it
        looks for a green button that wasn't on screen before the click, and
        once pressing it has closed the popup, saves it as the template.
        """
        deadline = time.time() + self.act["popup_timeout"]
        last: Match | None = None
        while time.time() < deadline:
            time.sleep(0.25)
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
