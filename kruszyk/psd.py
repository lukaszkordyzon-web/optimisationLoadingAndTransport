"""Particle size distribution (krzywa uziarnienia) of a pile and its effect on loading and crushing.

A curve is a list of [sieve size in mm, % passing]. From it we read P50/P80 (size below which 50/80 %
of the material lies) and the oversize share (bigger than the crusher's maximum feed size). These are
turned into the fragmentation index 0..1 used by the physics: 0 = fine, well blasted material
(fast digging, full crusher throughput), 1 = blocky material (slow digging, hammering, lower throughput).
"""
from __future__ import annotations

import math

SIEVES_MM = [10, 25, 50, 100, 200, 300, 500, 800, 1000, 1200]
P80_FINE_MM = 150.0    # P80 at or below this = excellent fragmentation (index 0)
P80_COARSE_MM = 900.0  # P80 at or above this = very poor fragmentation (index 1)


def rosin_rammler(p80_mm: float, n: float = 1.2, sizes=SIEVES_MM) -> list[list[float]]:
    """Typical blasted-rock curve defined by just two numbers: P80 and uniformity n
    (n ≈ 0.8 wide range of sizes … 2 very uniform)."""
    k = math.log(5)  # 80 % passing at x = P80
    return [[float(x), round(100 * (1 - math.exp(-k * (x / p80_mm) ** n)), 1)] for x in sizes]


PRESETS = {
    "drobne (dobry odstrzał)": rosin_rammler(150, 1.3),
    "średnie": rosin_rammler(300, 1.2),
    "średnio-grube (P50 240, P80 450 mm)": rosin_rammler(450, 1.34),
    "grube": rosin_rammler(550, 1.1),
    "bardzo grube (dużo brył)": rosin_rammler(850, 1.0),
}
DEFAULT_PRESET = "średnie"


def clean(curve) -> list[list[float]]:
    """Sorted, valid points; % passing forced to be non-decreasing and within 0..100."""
    pts = sorted((float(x), float(p)) for x, p in (curve or []) if x is not None and p is not None and float(x) > 0)
    out, last = [], 0.0
    for x, p in pts:
        last = min(100.0, max(last, p))
        out.append([x, last])
    return out


def size_at(curve, passing_pct: float) -> float | None:
    """Sieve size (mm) at which `passing_pct` % passes – log-interpolated (P50, P80…)."""
    c = clean(curve)
    if not c:
        return None
    if passing_pct <= c[0][1]:
        return c[0][0]
    for (x0, p0), (x1, p1) in zip(c, c[1:]):
        if p0 <= passing_pct <= p1 and p1 > p0:
            f = (passing_pct - p0) / (p1 - p0)
            return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))
    # beyond the last sieve: extrapolate with the last slope (still a sensible "very coarse" answer)
    if len(c) < 2:
        return c[-1][0] * 1.5
    (x0, p0), (x1, p1) = c[-2], c[-1]
    if p1 <= p0:
        return c[-1][0] * 1.5
    f = (passing_pct - p0) / (p1 - p0)
    return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))


def passing_at(curve, size_mm: float) -> float:
    c = clean(curve)
    if not c:
        return 100.0
    if size_mm <= c[0][0]:
        return c[0][1]
    for (x0, p0), (x1, p1) in zip(c, c[1:]):
        if x0 <= size_mm <= x1:
            f = (math.log(size_mm) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return p0 + f * (p1 - p0)
    return c[-1][1]


def fragmentation_index(curve, max_feed_mm: float = 800.0) -> float:
    """0 = fine (P80 ≤ 150 mm), 1 = blocky (P80 ≥ 900 mm); oversize above the crusher feed size adds to it."""
    p80 = size_at(curve, 80)
    if p80 is None:
        return 0.3
    base = (math.log(max(p80, 1.0)) - math.log(P80_FINE_MM)) / (math.log(P80_COARSE_MM) - math.log(P80_FINE_MM))
    oversize = 100 - passing_at(curve, max_feed_mm)
    return round(min(1.0, max(0.0, base + 0.5 * oversize / 100)), 3)


def summary(curve, max_feed_mm: float = 800.0, crusher_penalty: float = 0.4) -> dict:
    """Key numbers of a curve and what they mean for the equipment (same formulas as physics.py)."""
    frag = fragmentation_index(curve, max_feed_mm)
    p50, p80 = size_at(curve, 50), size_at(curve, 80)
    return {
        "P50 [mm]": round(p50) if p50 else None,
        "P80 [mm]": round(p80) if p80 else None,
        "nadziarno [%]": round(100 - passing_at(curve, max_feed_mm), 1),
        "wskaźnik rozdrobnienia": frag,
        "załadunek dłuższy o [%]": round(50 * frag, 1),          # dig_factor = 1 + 0.5 * frag
        "kruszarka wolniej o [%]": round(100 * crusher_penalty * frag, 1),
    }


def preset_of(curve) -> str:
    """Name of the preset equal to the curve, or 'własna'."""
    c = clean(curve)
    for name, pts in PRESETS.items():
        if c == clean(pts):
            return name
    return "własna"
