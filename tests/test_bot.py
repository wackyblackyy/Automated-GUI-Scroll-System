"""Runs the scan loop against a simulated, scrollable shop screen."""
import cv2
import numpy as np
import pytest
import yaml

from shop_bot import vision
from shop_bot.bot import ShopBot
from shop_bot.detector import ShopDetector
from test_detector import FIX, ROOT, composite_shop

LIST_Y1, LIST_Y2 = 90, 1040


class FakeScreen:
    """Static HUD; the list area is a window onto a tall strip that drags scroll."""

    def __init__(self):
        self.base = vision.load_image(FIX / "shop_hud.png")
        plain = self.base[LIST_Y1:LIST_Y2]
        with_targets = composite_shop()[LIST_Y1:LIST_Y2]
        self.strip = np.vstack([plain, with_targets])   # targets only appear after scrolling
        self.offset = 0
        self.clicks = []
        self.dry_run = True

    def grab(self):
        f = self.base.copy()
        f[LIST_Y1:LIST_Y2] = self.strip[self.offset:self.offset + (LIST_Y2 - LIST_Y1)]
        return f

    def click(self, x, y, label=""):
        self.clicks.append((label, x, y))

    def drag(self, x, y1, y2, duration):
        bottom = len(self.strip) - (LIST_Y2 - LIST_Y1)
        self.offset = max(0, min(self.offset + (y1 - y2), bottom))

    def wheel(self, x, y, clicks):
        raise AssertionError("config uses drag")


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["log_csv"] = ""
    cfg["actions"]["buy"] = True
    return ShopBot(cfg, ShopDetector(cfg, ROOT), FakeScreen())


def test_scrolls_until_both_found_and_clicks_buy_once_each(bot):
    seen = bot.scan_page()
    assert set(seen) == {"Mystic Medals", "Covenant Bookmarks"}
    assert bot.screen.offset > 0
    buys = [c for c in bot.screen.clicks if c[0].startswith("Buy")]
    assert sorted(c[0] for c in buys) == ["Buy Covenant Bookmarks", "Buy Mystic Medals"]


def test_stops_at_bottom_when_nothing_found(bot):
    s = bot.screen
    s.strip = np.vstack([s.base[LIST_Y1:LIST_Y2]] * 2)
    assert bot.scan_page() == {}
    assert s.offset == len(s.strip) - (LIST_Y2 - LIST_Y1)
    assert s.clicks == []


def test_first_page_mid_list_covers_both_ends(bot):
    s = bot.screen
    bot.act["buy"] = False
    s.strip = np.vstack([composite_shop()[LIST_Y1:LIST_Y2], s.base[LIST_Y1:LIST_Y2], s.base[LIST_Y1:LIST_Y2]])
    s.offset = 950                              # user left the list in the middle
    assert set(bot.scan_page()) == {"Mystic Medals", "Covenant Bookmarks"}


def test_alternates_direction_when_list_keeps_position(bot):
    s = bot.screen
    bot.act["buy"] = False
    bottom = len(s.strip) - (LIST_Y2 - LIST_Y1)
    s.strip = np.vstack([s.base[LIST_Y1:LIST_Y2]] * 2)       # nothing to find: full sweep
    bot.scan_page()                                          # page 1 from the top: ends at the bottom
    assert s.offset == bottom
    drags = []
    real = s.drag
    s.drag = lambda x, y1, y2, d: (drags.append(y2 > y1), real(x, y1, y2, d))
    bot.scan_page()                                          # page 2 (no reset): straight back up
    assert s.offset == 0 and all(drags)                      # only upward drags
    assert len(drags) == 2                                   # full flick + short one; no wasted check flick


def test_learns_that_refresh_resets_list_to_top(bot):
    s = bot.screen
    bot.act["buy"] = False
    s.strip = np.vstack([s.base[LIST_Y1:LIST_Y2]] * 2)
    bot.scan_page()                  # ends at bottom
    s.offset = 0                     # game resets to top on refresh
    bot.scan_page()                  # tries up once, flips, sweeps down
    assert s.offset > 0
    s.offset = 0
    drags = []
    real = s.drag
    s.drag = lambda x, y1, y2, d: (drags.append(y2 > y1), real(x, y1, y2, d))
    bot.scan_page()
    assert drags and not any(drags)  # now goes straight down, no wasted upward flick


def test_refresh_popup_falls_back_to_buy_popup_template(tmp_path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["templates"]["confirm_buy"] = "templates/buy_button.png"      # any existing image
    cfg["templates"]["confirm_refresh"] = "templates/does_not_exist.png"
    assert ShopDetector(cfg, ROOT).has_template("confirm_refresh")
    cfg["templates"]["confirm_buy"] = "templates/does_not_exist.png"
    cfg["templates"]["confirm_refresh"] = "templates/confirm_refresh.png"
    assert ShopDetector(cfg, ROOT).has_template("confirm_buy")


def popup_over(frame):
    """Dimmed shop with a centred popup: grey Cancel on the left, green button on the right."""
    shot = vision.load_image(FIX / "shop_hud.png")
    out = (frame * 0.45).astype(np.uint8)
    out[330:760, 560:1300] = (40, 35, 30)
    out[640:720, 610:880] = (90, 90, 90)                  # Cancel
    out[630:720, 900:1260] = cv2.resize(shot[912:1000, 95:530], (360, 90))  # green Refresh-style button
    return out


def test_finds_new_popup_button_but_not_existing_buttons(bot):
    before = bot.screen.grab()
    bot.det.calibrate(before)
    m = bot.det.find_popup_button(before, popup_over(before))
    assert m is not None
    cx, cy = m.center
    assert 1000 <= cx <= 1160 and 650 <= cy <= 710      # on the green button, not Cancel
    assert m.w > 250
    assert bot.det.find_popup_button(before, before) is None


def test_confirm_learns_template_on_first_popup(bot, tmp_path, monkeypatch):
    s = bot.screen
    bot.det.calibrate(s.grab())
    bot.det.base = tmp_path
    (tmp_path / "templates").mkdir()
    bot.det.confirm_buy_tmpl = bot.det.confirm_refresh_tmpl = None
    before = s.grab()
    popup = popup_over(before)
    frames = iter([popup, popup, before])          # popup animates in, then closes on click
    s.grab = lambda: next(frames)
    assert bot._confirm("confirm_buy", before)
    label, x, y = s.clicks[-1]
    assert label == "confirm_buy" and 900 <= x <= 1260 and 630 <= y <= 720
    assert (tmp_path / "templates" / "confirm_buy.png").is_file()
    assert bot.det.has_template("confirm_refresh")          # reused for the refresh popup


class ModalFakeScreen(FakeScreen):
    """Buy/Refresh open a confirm popup that only closes when its green button is pressed."""

    def __init__(self, pages):
        super().__init__()
        self.pages = pages          # list of strips; Refresh loads the next one
        self.page = 0
        self.strip = pages[0]
        self.dry_run = False
        self.pending = None

    def grab(self):
        f = super().grab()
        return popup_over(f) if self.pending else f

    def click(self, x, y, label=""):
        super().click(x, y, label)
        if self.pending:
            if 900 <= x <= 1260 and 630 <= y <= 720:
                if self.pending == "refresh":
                    self.page += 1
                    self.strip, self.offset = self.pages[self.page % len(self.pages)], 0
                self.pending = None
        elif label.startswith("Buy"):
            self.pending = "buy"
        elif label == "Refresh":
            self.pending = "refresh"


def test_full_automatic_run(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["log_csv"] = str(tmp_path / "log.csv")
    cfg["actions"]["max_refreshes"] = 3
    for k in ("confirm_buy", "confirm_refresh"):
        cfg["templates"][k] = str(tmp_path / f"{k}.png")      # start with no popup templates

    base = vision.load_image(FIX / "shop_hud.png")
    plain = np.vstack([base[LIST_Y1:LIST_Y2]] * 2)
    targets = np.vstack([base[LIST_Y1:LIST_Y2], composite_shop()[LIST_Y1:LIST_Y2]])
    screen = ModalFakeScreen([plain, targets])
    bot = ShopBot(cfg, ShopDetector(cfg, ROOT), screen)
    bot.run()

    assert bot.refreshes == 3
    # pages 1 and 3 hold both items -> 4 purchases, all confirmed
    assert bot.bought == {"Mystic Medals": 2, "Covenant Bookmarks": 2}
    assert screen.pending is None
    assert (tmp_path / "confirm_buy.png").is_file()
    assert len((tmp_path / "log.csv").read_text().splitlines()) == 5
