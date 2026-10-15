"""The month work plan a person sets (Kruszyk 2): on every shift, where every excavator loads (an extraction area) and
how many tonnes, and which excavator every truck drives to – or that it stands.

The program works out the rest:
- the piles: what lies in an area at the start is its pile A/1; every next blast of the area is a new pile A/2,
  A/3 … taken in order as the excavators of the area load out the tonnes planned (the rest of a pile is loaded
  together with the next one);
- drilling and firing of every blast, backwards from the shift its pile is needed (schedule.plan_blasts);
- what does not add up: a blast that cannot be ready in time, an area without reserves, an excavator without
  trucks, a truck sent to an excavator that stands, more tonnes than the excavator can load;
- the frozen schedule both months follow (PeriodPlan.frozen): piles, blasts, tonnes, trucks.
"""
from __future__ import annotations

import copy
import math

from .schedule import area_blast_t, plan_blasts

LATE = "late"
EXHAUSTED = "exhausted"
OVER = "over"
NO_TRUCKS = "no_trucks"
IDLE_TRUCK = "idle_truck"


def cell(work: dict, ex: str, i: int) -> tuple[str, float]:
    """(area, tonnes) of an excavator on a shift; ("", 0) = it stands."""
    cells = work["ex"].get(ex) or []
    c = cells[i] if i < len(cells) else {}
    a, t = str(c.get("a") or ""), float(c.get("t") or 0.0)
    return (a, t) if a and t > 0 else ("", 0.0)


def fit(work: dict, n: int) -> dict:
    """The plan cut or extended (last shift repeated) to n shifts."""
    out = {"start": dict(work.get("start") or {}), "ex": {}, "tr": {}}
    for ex, cells in (work.get("ex") or {}).items():
        cells = list(cells) or [{"a": "", "t": 0.0}]
        out["ex"][ex] = [dict(c) for c in (cells + [cells[-1]] * n)[:n]]
    for tid, cells in (work.get("tr") or {}).items():
        cells = list(cells) or [""]
        out["tr"][tid] = (cells + [cells[-1]] * n)[:n]
    return out


def default(excavators: list, trucks: list[str], areas: list[dict], rates: dict[str, float], n: int,
            start: dict[str, float], near: dict[str, str] | None = None, cap_t: float = math.inf) -> dict:
    """A first plan to start from: every excavator in the area nearest to it (or the first it may work in), at about
    90 % of what it loads today; the trucks dealt out evenly; what lies in the areas as given."""
    ids = [a["id"] for a in areas]
    work = {"start": {a: float(start.get(a, 0.0)) for a in ids}, "ex": {}, "tr": {}}
    for j, e in enumerate(excavators):
        allowed = [a for a in ids if e.may_work(a)] or ids
        a = (near or {}).get(e.id) if (near or {}).get(e.id) in allowed else (allowed[j % len(allowed)] if allowed else "")
        t = round(0.9 * rates.get(e.id, 0.0), -2) or 1000.0
        work["ex"][e.id] = [{"a": a, "t": t} for _ in range(n)]
    exs = [e.id for e in excavators]
    for j, tid in enumerate(trucks):
        work["tr"][tid] = [exs[j % len(exs)] if exs else ""] * n
    return work


def evaluate(work: dict, areas: list[dict], rigs: list[dict], n: int, cap_t: float = math.inf, fire: int = 1,
             rates: dict[str, float] | None = None) -> dict:
    """Everything that follows from the plan: the pile of every excavator on every shift, the blasts (drilling and
    firing), what lies on every pile, the tonnes the plan gives per shift, and the issues."""
    by_id = {a["id"]: a for a in areas}
    grid = {ex: [""] * n for ex in work["ex"]}
    left: dict[str, list[float]] = {}
    release: dict[str, int] = {}  # earliest shift the drilling of a pile may start (the area's own rule)
    issues: list[dict] = []
    for a_id, a in by_id.items():
        s = float(work["start"].get(a_id, 0.0))
        b = area_blast_t(a, cap_t)
        reserve = float(a.get("reserves_t") or 0.0)
        most = min(60, math.ceil(reserve / b)) if reserve > 0 else 0  # blasts the area still has
        cum, done = [], 0.0
        warned = False
        for i in range(n):
            on = [ex for ex in work["ex"] if cell(work, ex, i)[0] == a_id]
            d = sum(cell(work, ex, i)[1] for ex in on)
            cum.append(done)
            if not on:
                continue
            x = done + d / 2  # the pile they load at mid-shift
            j = 1 if x < s else 2 + int((x - s) // b)
            if j - 1 > most:
                j = 1 + most
                if not warned:
                    warned = True
                    issues.append({"kind": EXHAUSTED, "ex": on[0], "shift": i, "pile": f"{a_id}/{j}", "area": a_id})
            for ex in on:
                grid[ex][i] = f"{a_id}/{j}"
            done += d
        used = {int(p.rsplit("/", 1)[1]) for cells in grid.values() for p in cells if p.startswith(f"{a_id}/")}
        below = a.get("drill_below_t")
        if below is not None and below == below:  # drilling starts when this little is left to dig in the area
            for j in sorted(used - {1}):
                end_prev = s + (j - 2) * b  # what lies in the area before this blast
                release[f"{a_id}/{j}"] = next((i for i, c in enumerate(cum) if end_prev - c <= float(below)), n)
        for j in sorted(used | ({1} if s > 0 else set())):
            lo = 0.0 if j == 1 else s + (j - 2) * b
            hi = s if j == 1 else lo + b
            left[f"{a_id}/{j}"] = [max(0.0, min(hi - lo, hi - c)) for c in cum]
    start_piles = {f"{a}/1" for a in by_id}
    blasts = plan_blasts(grid, areas, rigs, n, cap_t, fire, skip=start_piles, release=release)
    for p, bl in blasts.items():  # nothing lies there before the blast
        if p in left:
            left[p] = [v if i >= bl["avail"] else 0.0 for i, v in enumerate(left[p])]
    planned = []
    for i in range(n):
        got = 0.0
        for ex in work["ex"]:
            p = grid[ex][i]
            if p and not (p in blasts and blasts[p]["avail"] > i):
                got += cell(work, ex, i)[1]
        planned.append(got)
    seen = set()
    for ex, cells in grid.items():
        for i, p in enumerate(cells):
            if p in blasts and blasts[p]["avail"] > i and (ex, p) not in seen:
                seen.add((ex, p))
                issues.append({"kind": LATE, "ex": ex, "shift": i, "pile": p, "avail": blasts[p]["avail"]})
    for ex in work["ex"]:
        r = (rates or {}).get(ex, 0.0)
        over = [i for i in range(n) if r and cell(work, ex, i)[1] > 1.1 * r]
        if over:
            issues.append({"kind": OVER, "ex": ex, "shift": over[0], "shifts": len(over),
                           "t": cell(work, ex, over[0])[1], "rate": r})
        bare = [i for i in range(n) if cell(work, ex, i)[1] > 0
                and not any(cells[i] == ex for cells in work["tr"].values() if i < len(cells))]
        if bare:
            issues.append({"kind": NO_TRUCKS, "ex": ex, "shift": bare[0], "shifts": len(bare)})
    for tid, cells in work["tr"].items():
        idle = [i for i, x in enumerate(cells[:n]) if x and cell(work, x, i)[1] <= 0]
        if idle:
            issues.append({"kind": IDLE_TRUCK, "truck": tid, "ex": cells[idle[0]], "shift": idle[0], "shifts": len(idle)})
    return {"grid": grid, "blasts": blasts, "left": left, "planned": planned, "issues": issues}


def frozen(work: dict, ev: dict, n: int) -> dict:
    """The plan for PeriodPlan.frozen: per shift the excavators' piles, the blasts that become ready, the tonnes of
    every excavator and where every truck drives."""
    grid, blasts = ev["grid"], ev["blasts"]
    return {"ex": [{ex: cells[i] for ex, cells in grid.items() if cells[i]} for i in range(n)],
            "blasts": [sorted(p for p, b in blasts.items() if b["avail"] == i) for i in range(n)],
            "quota": [{ex: cell(work, ex, i)[1] for ex in work["ex"]} for i in range(n)],
            "trucks": [{tid: (cells[i] if i < len(cells) else "") for tid, cells in work["tr"].items()} for i in range(n)]}


def scenario_raw(raw: dict, work: dict, grid: dict[str, list[str]], cap_t: float = math.inf) -> dict:
    """The mine to simulate the plan on: the piles A/1 with what lies in the areas at the start, every excavator
    standing at the pile of its first area in the plan (the map's own piles and digging lines are not used)."""
    from .site import area_start_piles, loading_point, site_to_scenario

    new = copy.deepcopy(raw)
    new.setdefault("blasting", {})["enabled"] = True  # the pit closes for the blasts of the plan (only on those shifts)
    site = new["site"]
    for a in site.get("areas", []):
        a["start_t"] = float(work["start"].get(a["id"], 0.0))
    piles = area_start_piles(site, cap_t)
    if not piles:
        raise ValueError("Na mapie nie ma obszarów wydobycia")
    site["piles"], site["planned_piles"] = piles, []
    poly = {p["id"]: p["polygon"] for p in piles}
    for e in site.get("excavators", []):
        first = next((p for p in grid.get(e["id"], []) if p), "")
        pile = f"{first.rsplit('/', 1)[0]}/1" if first else ""
        if pile not in poly:
            allowed = [p for p in poly if not e.get("areas") or p.split("/")[0] in e["areas"]] or list(poly)
            pile = allowed[0]
        e["pile"] = pile
        e["lat"], e["lon"] = loading_point(site, poly[pile])
        for k in ("dig_path", "then_pile"):
            e.pop(k, None)
    return site_to_scenario(site, new)


def nearest_areas(site: dict) -> tuple[dict[str, float], dict[str, str]]:
    """From a map that still has piles: what lies in every area (each pile to the nearest area) and the area nearest
    to every excavator – the start of a first plan."""
    from .site import centroid, haversine_m

    areas = [a for a in site.get("areas", []) if len(a.get("polygon") or []) >= 3 and a.get("active", True)]
    if not areas:
        return {}, {}

    def near(pt):
        return min(areas, key=lambda a: haversine_m(pt, centroid(a["polygon"])))["id"]

    start: dict[str, float] = {}
    for p in site.get("piles", []):
        if len(p.get("polygon") or []) >= 3:
            a = near(centroid(p["polygon"]))
            start[a] = start.get(a, 0.0) + float(p.get("tonnage_t") or 0.0)
    exs = {e["id"]: near((e["lat"], e["lon"])) for e in site.get("excavators", []) if e.get("lat") is not None}
    return start, exs
