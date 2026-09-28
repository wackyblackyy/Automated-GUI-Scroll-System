"""CLI: python -m shop_bot {run,scan-image,capture,monitors}"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

from . import vision
from .detector import ShopDetector

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cmd_run(args, cfg):
    from .bot import ShopBot, StopBot
    from .controller import ScreenController

    if args.scan_only:
        cfg["actions"]["buy"] = cfg["actions"]["refresh"] = False
    if args.dry_run:
        cfg["actions"]["refresh"] = False     # nothing changes without clicks: one page is enough
    if args.refreshes is not None:
        cfg["actions"]["max_refreshes"] = args.refreshes
    det = ShopDetector(cfg, ROOT)
    screen = ScreenController(cfg.get("monitor", 1), dry_run=args.dry_run)
    print(f"Starting in {args.delay}s - switch to the game. Move the mouse to a screen corner to abort.")
    time.sleep(args.delay)
    try:
        ShopBot(cfg, det, screen).run()
    except StopBot as e:
        logging.error("Stopped: %s", e)
        return 1
    except KeyboardInterrupt:
        logging.info("Interrupted")
    except Exception as e:
        if type(e).__name__ == "FailSafeException":
            logging.info("Aborted: mouse moved to a screen corner")
            return 0
        raise
    return 0


def cmd_scan_image(args, cfg):
    """Offline check: detect HUD + target items in a screenshot file."""
    det = ShopDetector(cfg, ROOT)
    frame = vision.load_image(args.image)
    cal = det.calibrate(frame)
    if cal is None:
        print("HUD (Secret Shop title) not found - scanning the whole image at scale 1.0")
    found = det.find_targets(frame)
    boxes = []
    for f in found:
        state = "Buy button found" if f.buy else "no Buy button"
        print(f"FOUND {f.name:20s} score={f.icon.score:.2f} at {f.icon.center} ({state})")
        boxes.append((f.name, f.icon))
        if f.buy:
            boxes.append(("Buy", f.buy))
    if not found:
        print("No target items on this screen.")
    if args.out:
        cv2.imwrite(args.out, vision.annotate(frame, boxes))
        print(f"annotated image -> {args.out}")
    return 0 if found else 2


def cmd_capture(args, cfg):
    """Screenshot the monitor; with --crop NAME, drag a box to save templates/NAME.png."""
    from .controller import ScreenController
    print(f"Capturing in {args.delay}s - switch to the game...")
    time.sleep(args.delay)
    frame = ScreenController(cfg.get("monitor", 1)).grab()
    cv2.imwrite(args.out, frame)
    print(f"saved {args.out} ({frame.shape[1]}x{frame.shape[0]})")
    if not args.crop:
        return 0
    # templates are matched at the scale the HUD is found at, so store them at reference scale
    det = ShopDetector(cfg, ROOT)
    cal = det.calibrate(frame)
    if cal is None:
        print("Secret Shop HUD not visible in the capture - open the shop (with the popup) and retry")
        return 1
    print("Drag a tight box around the button, then press ENTER (c = cancel)")
    x, y, w, h = cv2.selectROI("select template", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("cancelled")
        return 1
    crop = frame[y:y + h, x:x + w]
    if abs(cal.scale - 1.0) > 0.01:
        crop = cv2.resize(crop, None, fx=1 / cal.scale, fy=1 / cal.scale, interpolation=cv2.INTER_AREA)
    out = ROOT / "templates" / f"{args.crop}.png"
    cv2.imwrite(str(out), crop)
    print(f"template saved -> {out}")
    return 0


def cmd_monitors(args, cfg):
    from .controller import list_monitors
    for i, m in enumerate(list_monitors()):
        print(i, m, "(all monitors)" if i == 0 else "")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="shop_bot", description=__doc__)
    p.add_argument("-c", "--config", default=str(ROOT / "config.yaml"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="scan, buy, scroll and refresh the live Secret Shop")
    r.add_argument("-n", "--refreshes", type=int, metavar="N",
                   help="stop after N refreshes (default: actions.max_refreshes in config)")
    r.add_argument("--scan-only", action="store_true", help="just report items: no buying, no refreshing")
    r.add_argument("--dry-run", action="store_true", help="log clicks instead of performing them")
    r.add_argument("--delay", type=float, default=3.0, help="seconds before starting")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("scan-image", help="detect items in a screenshot file")
    s.add_argument("image")
    s.add_argument("-o", "--out", help="write an annotated copy here")
    s.set_defaults(fn=cmd_scan_image)

    c = sub.add_parser("capture", help="save a screenshot (for making templates)")
    c.add_argument("-o", "--out", default="capture.png")
    c.add_argument("--crop", metavar="NAME", help="select a region and save it as templates/NAME.png")
    c.add_argument("--delay", type=float, default=3.0)
    c.set_defaults(fn=cmd_capture)

    m = sub.add_parser("monitors", help="list monitors")
    m.set_defaults(fn=cmd_monitors)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    return args.fn(args, load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
