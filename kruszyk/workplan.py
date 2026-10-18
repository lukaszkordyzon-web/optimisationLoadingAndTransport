"""The month work plan a person sets (Kruszyk 2), on every shift:
- every drill rig: in which extraction area it drills and how many tonnes,
- every excavator: in which area it loads and how many tonnes,
- every truck: to which excavator it drives – or that it stands.

The program works out the rest:
- the drilling: every order (tonnes in an area) is drilled for as many shifts as it takes, orders in a queue;
- the blasts: every order is one blast, fired when it is drilled out; the material is there after the firing – a
  blast can be moved later by the planner;
- the piles: what lies in an area at the start is its pile A/1, every blast is the next pile A/2, A/3 …, loaded in
  order;
- an excavator loads only what is blasted: the rest of its tonnes is a shortage;
- what a rig can drill on a shift (metres per shift × tonnes per metre of the area, less the drive from another
  area along the roads);
- what does not add up, and the frozen schedule both months follow (PeriodPlan.frozen).
"""
from __future__ import annotations

import copy
import math

ENDLESS_T = 1e9  # an area without planned drilling: the material never runs out
SHORT = "short"
EXHAUSTED = "exhausted"
OVER = "over"
UNFINISHED = "unfinished"
TOO_BIG = "too_big"
NO_TRUCKS = "no_trucks"
IDLE_TRUCK = "idle_truck"


def cell(work: dict, ex: str, i: int, part: str = "ex") -> tuple[str, float]:
    """(area, tonnes) of an excavator (or, with part="dr", a drill rig) on a shift; ("", 0) = it stands."""
    cells = (work.get(part) or {}).get(ex) or []
    c = cells[i] if i < len(cells) else {}
    a, t = str(c.get("a") or ""), float(c.get("t") or 0.0)
    return (a, t) if a and t > 0 else ("", 0.0)


def fit(work: dict, n: int) -> dict:
    """The plan cut or extended (last shift repeated) to n shifts."""
    out = {"start": dict(work.get("start") or {}), "ex": {}, "tr": {}, "dr": {}, "fire": dict(work.get("fire") or {}),
           "mode": work.get("mode", "limit"), "drill": work.get("drill", True)}
    for k in ("machines", "roads"):
        if work.get(k):
            out[k] = work[k]
    for part in ("ex", "dr"):
        for k, cells in (work.get(part) or {}).items():
            cells = list(cells) or [{"a": "", "t": 0.0}]
            out[part][k] = [dict(c) for c in (cells + [cells[-1]] * n)[:n]]
    for tid, cells in (work.get("tr") or {}).items():
        cells = list(cells) or [""]
        out["tr"][tid] = (cells + [cells[-1]] * n)[:n]
    return out


def rig_rate(rig: dict, area: dict, t_per_m: float = 50.0) -> float:
    """Tonnes a rig drills on a whole shift in an area."""
    return float(rig.get("m_per_shift") or 250.0) * float(area.get("t_per_m") or t_per_m)


def default(excavators: list, trucks: list[str], areas: list[dict], rates: dict[str, float], n: int,
            start: dict[str, float], near: dict[str, str] | None = None, cap_t: float = math.inf,
            rigs: list[dict] | None = None, fire: int = 1) -> dict:
    """A first plan to start from: every excavator in the area nearest to it (or the first it may work in), at about
    90 % of what it loads today; the trucks dealt out evenly; the rigs drilling where the material runs short."""
    ids = [a["id"] for a in areas]
    work = {"start": {a: float(start.get(a, 0.0)) for a in ids}, "ex": {}, "tr": {}, "dr": {}, "fire": {}}
    for j, e in enumerate(excavators):
        allowed = [a for a in ids if e.may_work(a)] or ids
        a = (near or {}).get(e.id) if (near or {}).get(e.id) in allowed else (allowed[j % len(allowed)] if allowed else "")
        t = round(0.9 * rates.get(e.id, 0.0), -2) or 1000.0
        work["ex"][e.id] = [{"a": a, "t": t} for _ in range(n)]
    exs = [e.id for e in excavators]
    for j, tid in enumerate(trucks):
        work["tr"][tid] = [exs[j % len(exs)] if exs else ""] * n
    work["dr"] = propose_drilling(work, areas, rigs or [], n, cap_t, fire)
    return work


def proposal_blast_t(area: dict, cap_t: float = math.inf) -> float:
    """Size of the blasts a first plan orders: the area's former blast size, else 10 000 t; within the explosive
    limit."""
    return min(float(area.get("blast_max_t") or 10000.0), cap_t)


def propose_drilling(work: dict, areas: list[dict], rigs: list[dict], n: int, cap_t: float = math.inf,
                     fire: int = 1, shift_h: float = 8.0) -> dict:
    """Drilling that keeps the excavators supplied: when a rig is free, it gets an order of one whole blast in the
    area that would run short soonest; it stays free when nothing is needed."""
    by_id = {a["id"]: a for a in areas}
    need = {a: [0.0] * n for a in by_id}  # tonnes the excavators want, cumulative up to the shift (inclusive)
    for a in by_id:
        acc = 0.0
        for i in range(n):
            acc += sum(cell(work, ex, i)[1] for ex in work["ex"] if cell(work, ex, i)[0] == a)
            need[a][i] = acc
    have = {a: float(work["start"].get(a, 0.0)) for a in by_id}  # lying + ordered
    left = {a: float(by_id[a].get("reserves_t") or 0.0) for a in by_id}
    out = {r["id"]: [{"a": "", "t": 0.0} for _ in range(n)] for r in rigs}
    busy = {r["id"]: 0.0 for r in rigs}  # shifts of drilling still ahead of the rig
    ahead = fire + 2
    for i in range(n):
        for r in rigs:
            if busy[r["id"]] <= 1e-6:
                best, gap = "", 0.0
                for a, ar in by_id.items():
                    g = need[a][min(n - 1, i + ahead + 2)] + proposal_blast_t(ar, cap_t) - have[a]
                    if left[a] > 0 and need[a][-1] > 0 and g > gap:
                        best, gap = a, g
                if best:
                    t = min(proposal_blast_t(by_id[best], cap_t), left[best])
                    out[r["id"]][i] = {"a": best, "t": float(round(t))}
                    have[best] += t
                    left[best] -= t
                    busy[r["id"]] += t / rig_rate(r, by_id[best])
            busy[r["id"]] = max(0.0, busy[r["id"]] - 1.0)
    return out


def evaluate(work: dict, areas: list[dict], rigs: list[dict], n: int, cap_t: float = math.inf, fire: int = 1,
             rates: dict[str, float] | None = None, travel_h: dict | None = None, shift_h: float = 8.0) -> dict:
    """Everything that follows from the plan: what the rigs really drill, the blasts, what lies in every area, the
    pile every excavator loads and how much of its tonnes it gets, the tonnes per shift, and the issues."""
    by_id = {a["id"]: a for a in areas}
    if not work.get("drill", True):  # drilling not planned: the material is there all the time
        work = dict(work, start={a: ENDLESS_T for a in by_id}, dr={}, fire={})
        rigs = []
    issues: list[dict] = []
    # ---- drilling: every order (tonnes in an area) is drilled for as many shifts as it takes, orders one after
    # another; a drive to another area eats into the shift; "postój" stops the rig for that shift
    drilled = {a: [0.0] * n for a in by_id}
    drill_by: dict[str, list[list[str]]] = {a: [[] for _ in range(n)] for a in by_id}
    queued = {a: [0.0] * n for a in by_id}  # still to drill in the area at the end of every shift
    rig_rows: dict[str, list[dict]] = {}
    done_orders: list[tuple] = []  # (shift drilled out, rig, area, tonnes, first shift drilled)
    rig_cap: dict[str, list[float]] = {}
    for r in rigs:
        rid = r["id"]
        cells = (work.get("dr") or {}).get(rid) or []
        queue: list[list] = []
        prev, rows, caps, seq = "", [], [], 0
        for i in range(n):
            a, t = cell(work, rid, i, "dr")
            if a in by_id and t > 0:
                queue.append([a, t, t, None, seq])
                seq += 1
                if cap_t < math.inf and t > cap_t + 1:
                    issues.append({"kind": TOO_BIG, "rig": rid, "shift": i, "t": t, "cap": cap_t, "area": a})
            stop = bool(i < len(cells) and cells[i].get("stop"))
            row = {"order": (a, t) if a in by_id and t > 0 else None, "stop": stop, "drill": [], "drive_h": 0.0}
            hours = 0.0 if stop else shift_h
            caps.append(rig_rate(r, by_id[queue[0][0]]) if queue and not stop else 0.0)
            while queue and hours > 1e-6:
                qa = queue[0][0]
                if prev and prev != qa:
                    drive = (travel_h or {}).get((prev, qa), 0.0)
                    row["drive_h"] += drive
                    hours -= drive
                prev = qa
                if hours <= 1e-6:
                    break
                per_h = rig_rate(r, by_id[qa]) / shift_h
                d = min(queue[0][1], per_h * hours)
                if queue[0][3] is None:
                    queue[0][3] = i
                drilled[qa][i] += d
                drill_by[qa][i].append(rid)
                row["drill"].append((qa, d))
                row.setdefault("orders", []).append((queue[0][4], qa, d))
                queue[0][1] -= d
                hours -= d / per_h
                if queue[0][1] <= 1.0:  # the order is drilled out: it is one blast
                    qa_, _, t0, first, sq = queue.pop(0)
                    done_orders.append((i, rid, qa_, t0, first, sq))
            row["waiting"] = sum(q[1] for q in queue)
            for q in queue:
                queued[q[0]][i] += q[1]
            rows.append(row)
        rig_rows[rid], rig_cap[rid] = rows, caps
        if queue:
            issues.append({"kind": UNFINISHED, "rig": rid, "shift": n - 1, "t": sum(q[1] for q in queue),
                           "area": ", ".join(dict.fromkeys(q[0] for q in queue))})
    # ---- blasts: every drill order is one blast, fired when it is drilled out (the planner may move it later)
    blasts: dict[str, dict] = {}
    k = {a: 2 for a in by_id}
    total = {a: 0.0 for a in by_id}
    warned = set()
    pile_of: dict[tuple, str] = {}  # (rig, order number) -> the blast it becomes
    for i, rid, a_id, t, first, sq in sorted(done_orders):
        a = by_id[a_id]
        reserve = float(a.get("reserves_t") or 0.0)
        if reserve > 0 and total[a_id] + t > reserve + 1:
            if a_id not in warned:
                warned.add(a_id)
                issues.append({"kind": EXHAUSTED, "area": a_id, "shift": i})
            t = max(0.0, reserve - total[a_id])
            if t <= 1:
                continue
        total[a_id] += t
        pid = f"{a_id}/{k[a_id]}"
        k[a_id] += 1
        pile_of[(rid, sq)] = pid
        delay = int((work.get("fire") or {}).get(pid, 0))
        avail = i + 1 + fire + delay
        blasts[pid] = {"area": a_id, "t": round(t), "m": round(t / float(a.get("t_per_m") or 50.0)), "done": i,
                       "fire": list(range(i + 1 + delay, avail)), "avail": avail, "auto": i + 1 + fire, "delay": delay,
                       "rig": rid, "drill": [(rid, x) for x in range(first if first is not None else i, i + 1)]}
    for rid, rows_ in rig_rows.items():  # what the rig drills, by the blast it will be
        for row in rows_:
            row["piles"] = [(pile_of.get((rid, sq), f"{a}/?"), d) for sq, a, d in row.get("orders", [])]
    # ---- excavators: they load only what lies blasted, piles in order
    grid = {ex: [""] * n for ex in work["ex"]}
    quota = {ex: [0.0] * n for ex in work["ex"]}
    area_left: dict[str, list[float]] = {}
    short: dict[str, list[float]] = {}
    for a_id in by_id:
        s = float(work["start"].get(a_id, 0.0))
        piles = [(f"{a_id}/1", s, -1)] + sorted(((p, float(b["t"]), b["avail"]) for p, b in blasts.items()
                                                 if b["area"] == a_id), key=lambda x: x[2])
        bounds, c = [], 0.0
        for p, t, av in piles:
            c += t
            bounds.append((p, c, av))
        taken, lying, miss = 0.0, [], [0.0] * n
        for i in range(n):
            there = sum(t for _, t, av in piles if av <= i)
            lying.append(max(0.0, there - taken))
            on = [ex for ex in work["ex"] if cell(work, ex, i)[0] == a_id]
            want = sum(cell(work, ex, i)[1] for ex in on)
            if not on:
                continue
            got = min(want, max(0.0, there - taken))
            miss[i] = want - got
            x = taken + got / 2 if got > 0 else max(0.0, taken - 1.0)
            pile = next((p for p, cb, av in bounds if x < cb and av <= i), None)
            if pile is None:  # nothing blasted yet: it waits where the material will be
                pile = next((p for p, cb, av in bounds if x < cb), bounds[-1][0])
            for ex in on:
                grid[ex][i] = pile
                quota[ex][i] = cell(work, ex, i)[1] * (got / want if want else 0.0)
            taken += got
        area_left[a_id] = lying
        short[a_id] = miss
        if any(m > 1 for m in miss):
            i = next(j for j, m in enumerate(miss) if m > 1)
            exs = sorted({ex for j, m in enumerate(miss) if m > 1 for ex in work["ex"] if cell(work, ex, j)[0] == a_id})
            issues.append({"kind": SHORT, "area": a_id, "shift": i, "t": sum(miss), "ex": ", ".join(exs),
                           "shifts": sum(1 for m in miss if m > 1)})
    planned = [sum(quota[ex][i] for ex in quota) for i in range(n)]
    # ---- excavators and trucks
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
    return {"grid": grid, "quota": quota, "blasts": blasts, "area_left": area_left, "short": short,
            "drilled": drilled, "rig_cap": rig_cap, "rig_rows": rig_rows, "planned": planned, "issues": issues}


def frozen(work: dict, ev: dict, n: int) -> dict:
    """The plan for PeriodPlan.frozen: per shift the excavators' piles, the blasts that become ready (with their
    tonnes), the tonnes every excavator gets and where every truck drives. With the plan's tonnes as a minimum
    (`mode` "min") an excavator with tonnes loads on as long as it has blasted material, trucks and time."""
    grid, blasts = ev["grid"], ev["blasts"]
    if work.get("mode") == "min":
        quota = [{ex: cell(work, ex, i)[1] for ex in work["ex"]} for i in range(n)]
    else:
        quota = [{ex: ev["quota"][ex][i] for ex in work["ex"]} for i in range(n)]
    return {"open": work.get("mode") == "min","ex": [{ex: cells[i] for ex, cells in grid.items() if cells[i]} for i in range(n)],
            "blasts": [sorted((p for p, b in blasts.items() if b["avail"] == i),
                              key=lambda p: (p.rsplit("/", 1)[0], int(p.rsplit("/", 1)[1]))) for i in range(n)],
            "blast_t": {p: float(b["t"]) for p, b in blasts.items()},
            "quota": quota,
            "trucks": [{tid: (cells[i] if i < len(cells) else "") for tid, cells in work["tr"].items()} for i in range(n)]}


def area_travel_h(site: dict, rigs: list[dict]) -> dict[tuple[str, str], float]:
    """Hours a rig drives between every two areas along the roads (the slowest rig's speed)."""
    from .site import build_route, loading_point

    areas = [a for a in site.get("areas", []) if len(a.get("polygon") or []) >= 3 and a.get("active", True)]
    kmh = min([float(r.get("speed_kmh") or 2.0) for r in rigs] or [2.0])
    out = {}
    for a in areas:
        for b in areas:
            if a["id"] == b["id"]:
                continue
            try:
                r = build_route(copy.deepcopy(site), loading_point(site, a["polygon"]), "rig",
                                end=loading_point(site, b["polygon"]), end_name="obszar")
                km = sum(s["length_m"] for s in r["segments"]) / 1000
            except ValueError:
                continue
            out[(a["id"], b["id"])] = km / max(kmh, 0.1)
    return out


def scenario_raw(raw: dict, work: dict, grid: dict[str, list[str]], cap_t: float = math.inf) -> dict:
    """The mine to simulate the plan on: the piles A/1 with what lies in the areas at the start, every excavator
    standing at the pile of its first area in the plan (the map's own piles and digging lines are not used)."""
    from .site import area_start_piles, loading_point, site_to_scenario

    new = copy.deepcopy(raw)
    new.setdefault("blasting", {})["enabled"] = True  # the pit closes for the blasts of the plan (only on those shifts)
    site = new["site"]
    for a in site.get("areas", []):
        a["start_t"] = ENDLESS_T if not work.get("drill", True) else float(work["start"].get(a["id"], 0.0))
    piles = area_start_piles(site, cap_t)
    if not piles:
        raise ValueError("There are no extraction areas on the map")
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
