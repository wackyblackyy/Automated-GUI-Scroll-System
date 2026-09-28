# Automated GUI Scroll System – Secret Shop scanner

Hands-free loop for the Epic Seven **Secret Shop**:

1. Finds the Secret Shop screen (the "Secret Shop" title in the top-left) and works out your window's size and scale.
2. Scans the item list on the right for **Covenant Bookmarks** and **Mystic Medals**.
3. Clicks **Buy** on each one it finds and presses **Yes/Buy** on the confirmation popup.
4. Flicks the list to the other end and scans on the way. It scans down on one page and back up on the next, so it never spends time scrolling back to the top. If the game resets the list to the top after a refresh, the bot notices and always scans downward.
5. Presses **Refresh** (bottom-left), confirms the popup, and starts over.
6. Stops after `max_refreshes` (default 50 refreshes = 150 skystones), then prints a summary and appends each item it finds to `shop_log.csv`.

It works from screenshots and mouse input, so it runs on any PC client or emulator window. Nothing is read from the game's memory.

## Setup

```bash
pip install -r requirements.txt
```

## Run

1. Open the game and go to the **Secret Shop**. The window must be fully visible and not covered by other windows.
2. Start the bot, then click back into the game within 3 seconds:

```bash
python -m shop_bot run              # fully automatic: scan, buy, scroll, refresh, repeat
python -m shop_bot run -n 20        # same, but stop after 20 refreshes
python -m shop_bot run --scan-only  # only report what's there, no buying/refreshing
python -m shop_bot run --dry-run    # log where it *would* click
```

**To stop it:** move the mouse into any screen corner (PyAutoGUI fail-safe), or press Ctrl+C in the terminal.

### The confirmation popups

The blue **Confirm** button from the "Use Skystone to refresh?" popup is saved as `templates/confirm_refresh.png`, and the bot also tries it on the purchase popup. If the purchase popup's button looks different, the bot looks for the new bright-blue or green button that appeared after the click. It picks the right-most one (Cancel is brown), clicks it, checks the popup closed, and saves it as `templates/confirm_buy.png` for next time.

If it ever picks the wrong button, delete the saved PNG and make one yourself:

```bash
python -m shop_bot capture --crop confirm_buy      # open the Buy popup first, then drag a box around its Yes/Buy button
python -m shop_bot capture --crop confirm_refresh  # same for the Refresh popup
```

## Checking detection offline

```bash
python -m shop_bot scan-image my_screenshot.png -o annotated.png
python -m pytest            # runs against the screenshots in tests/fixtures
```

## Tuning (`config.yaml`)

| Setting | What it does |
|---|---|
| `targets` | Items to hunt for. To add another item, crop its icon into `templates/` and add an entry here. |
| `actions.max_refreshes` | Skystone budget (3 per refresh). |
| `scroll.method` | Use `drag` (default) for emulator/touch-style lists, or `wheel` if your client scrolls with the mouse wheel. |
| `scroll.drag_duration`, `scroll.settle` | Increase these if the list flings past items or the screenshot is taken mid-scroll. |
| `actions.click_delay` | Increase this on a slow PC/emulator. |
| `monitor` | Which monitor the game is on (`python -m shop_bot monitors`). |

Layout numbers are in the pixel coordinates of the 1852×1040 reference screenshot. They are rescaled automatically, so a different window size doesn't need any changes.

## How it works

- `shop_bot/vision.py`: multi-scale OpenCV template matching.
- `shop_bot/detector.py`: finds the HUD and UI scale, target icons (only in the icon column), each row's "1/1" Buy pill, the Refresh button, and popup buttons.
- `shop_bot/bot.py`: the scan → buy → scroll → refresh loop, plus popup learning.
- `shop_bot/controller.py`: screen capture (`mss`) and mouse input (`pyautogui`).

The bot finds a row's Buy button by matching its **"1/1"** half, so the click glow over "Buy" doesn't break detection. A row with no "1/1" (already bought) is skipped.

Each scan takes about 40 ms per frame after the first calibration. Each flick is a 0.15 s drag, followed by a 0.3 s pause so the list can settle. The bot measures how far the list moved, so hitting the end of the list is detected without an extra check flick. After a refresh it continues as soon as the new list appears, instead of waiting a fixed time.

## Notes

- Scaling on Windows: if clicks land in the wrong place, set the game's display to 100% scaling, or run Python with "High DPI scaling override: Application".
- If the game shows an extra "purchase complete" popup that needs a tap, tell me and I'll add a step to close it.
- Automating a game can break its terms of service. You use this at your own risk.
