"""Template matching helpers (pure OpenCV/numpy, no screen access)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float = 1.0

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def _resize(tmpl: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return tmpl
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(tmpl, None, fx=scale, fy=scale, interpolation=interp)


def _nms(matches: list[Match], overlap: float = 0.3) -> list[Match]:
    kept: list[Match] = []
    for m in sorted(matches, key=lambda m: m.score, reverse=True):
        if all(_iou(m, k) <= overlap for k in kept):
            kept.append(m)
    return kept


def _iou(a: Match, b: Match) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def find_all(frame: np.ndarray, tmpl: np.ndarray, threshold: float,
             scales: list[float] | tuple[float, ...] = (1.0,),
             region: tuple[int, int, int, int] | None = None) -> list[Match]:
    """Every occurrence of `tmpl` in `frame` scoring >= threshold.

    `region` (x1, y1, x2, y2) restricts the search; returned coordinates are
    always in full-frame pixels.
    """
    ox, oy = 0, 0
    if region is not None:
        x1, y1, x2, y2 = _clip(region, frame.shape)
        frame = frame[y1:y2, x1:x2]
        ox, oy = x1, y1

    found: list[Match] = []
    for s in scales:
        t = _resize(tmpl, s)
        th, tw = t.shape[:2]
        if th > frame.shape[0] or tw > frame.shape[1] or th < 8 or tw < 8:
            continue
        res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for x, y in zip(xs, ys):
            found.append(Match(int(x) + ox, int(y) + oy, tw, th, float(res[y, x]), s))
    return _nms(found)


def find_best(frame: np.ndarray, tmpl: np.ndarray,
              scales: list[float] | tuple[float, ...] = (1.0,),
              region: tuple[int, int, int, int] | None = None) -> Match | None:
    """Single best match across scales (no threshold applied)."""
    ox, oy = 0, 0
    if region is not None:
        x1, y1, x2, y2 = _clip(region, frame.shape)
        frame = frame[y1:y2, x1:x2]
        ox, oy = x1, y1

    best: Match | None = None
    for s in scales:
        t = _resize(tmpl, s)
        th, tw = t.shape[:2]
        if th > frame.shape[0] or tw > frame.shape[1] or th < 8 or tw < 8:
            continue
        res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(res)
        if best is None or score > best.score:
            best = Match(x + ox, y + oy, tw, th, float(score), s)
    return best


def scale_range(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference; used to tell whether a scroll moved."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(cv2.absdiff(a, b)))


def _clip(region, shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in region)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def annotate(frame: np.ndarray, matches: list[tuple[str, Match]]) -> np.ndarray:
    out = frame.copy()
    for label, m in matches:
        cv2.rectangle(out, (m.x, m.y), (m.x + m.w, m.y + m.h), (0, 255, 255), 3)
        cv2.putText(out, f"{label} {m.score:.2f}", (m.x, max(20, m.y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out
