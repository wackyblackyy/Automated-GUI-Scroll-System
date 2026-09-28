"""Offline tests against real screenshots (no screen or mouse needed)."""
from pathlib import Path

import cv2
import pytest
import yaml

from shop_bot import vision
from shop_bot.detector import ShopDetector

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture
def det():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return ShopDetector(cfg, ROOT)


def composite_shop():
    """The HUD screenshot with the two target rows pasted over existing rows."""
    shop = vision.load_image(FIX / "shop_hud.png")
    medals = vision.load_image(FIX / "row_mystic_medals.png")[40:210, 190:1235]
    books = vision.load_image(FIX / "row_covenant_bookmarks.png")[30:200, 190:1235]
    # icon of the row crops sits at x=212 / 210; shop icons sit at x~800
    shop[265:435, 778:1823] = medals      # over "Friendship Points"
    shop[685:855, 780:1825] = books       # over "Obsidian Dragon"
    return shop


def test_hud_detected_no_false_positives(det):
    frame = vision.load_image(FIX / "shop_hud.png")
    cal = det.calibrate(frame)
    assert cal is not None and abs(cal.scale - 1.0) < 0.02
    assert det.find_targets(frame) == []


@pytest.mark.parametrize("scale", [0.6, 0.75, 1.0, 1.4])
def test_finds_both_items_at_any_resolution(det, scale):
    frame = cv2.resize(composite_shop(), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cal = det.calibrate(frame)
    assert cal is not None and abs(cal.scale - scale) < 0.03

    found = {f.name: f for f in det.find_targets(frame)}
    assert set(found) == {"Mystic Medals", "Covenant Bookmarks"}
    for m in found.values():   # Buy pill must be on the same row, to the right
        assert m.buy is not None, m.name
        assert m.buy.x > m.icon.x + m.icon.w
        assert abs(m.buy.center[1] - m.icon.center[1]) < 80 * scale


def test_buy_click_lands_on_buy_text(det):
    frame = composite_shop()
    det.calibrate(frame)
    ox, oy = det.cfg["layout"]["buy_click_offset"]
    for f in det.find_targets(frame):
        x, y = f.buy.x + ox, f.buy.y + oy
        # "Buy" label spans roughly x 1690-1760 in the reference screenshot
        assert 1690 <= x <= 1760, (f.name, x)
        assert f.buy.y < y < f.buy.y + f.buy.h


def test_list_region_excludes_hero_panel(det):
    det.calibrate(vision.load_image(FIX / "shop_hud.png"))
    x1, y1, x2, y2 = det.list_region()
    assert x1 > 550 and x2 <= 1852 and y1 > 60


def test_confirm_button_found_on_real_refresh_popup(det):
    frame = vision.load_image(FIX / "refresh_popup.png")      # 1530x861, dimmed behind the popup
    assert det.calibrate(frame) is not None
    m = det.find_button(frame, "confirm_refresh")
    assert m is not None
    x, y = m.center
    assert 790 <= x <= 1005 and 525 <= y <= 585                # Confirm, not Cancel (x 525-750)


def test_popup_button_finder_picks_blue_confirm_not_cancel(det):
    after = vision.load_image(FIX / "refresh_popup.png")
    before = cv2.resize(vision.load_image(FIX / "shop_hud.png"), (after.shape[1], after.shape[0]))
    det.calibrate(after)
    m = det.find_popup_button(before, after)
    assert m is not None
    x, y = m.center
    assert 790 <= x <= 1005 and 525 <= y <= 585


def test_hud_is_the_title_not_shop_level_text(det):
    """Regression: the old template matched "Secret Shop Level 13" (bottom-left),
    shifting the whole layout off-screen so Refresh was never found."""
    frame = vision.load_image(FIX / "shop_hud_1919.png")      # 1919x1079 live screenshot
    cal = det.calibrate(frame)
    assert cal is not None and cal.hud.y < 100 and abs(cal.scale - 1.036) < 0.03
    btn = det.find_button(frame, "refresh")
    assert btn is not None
    x, y = btn.center
    assert 280 <= x <= 540 and 950 <= y <= 1030               # the green Refresh, bottom-left
    x1, y1, x2, y2 = det.list_region()
    assert 600 < x1 < 700 and x2 > 1800 and y2 > 1000          # item list on the right
    assert det.find_targets(frame) == []                       # none of the targets on this page
