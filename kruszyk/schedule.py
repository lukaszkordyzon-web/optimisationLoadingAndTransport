"""The month schedule a person plans: on every shift, where every excavator loads (the grid).

The planner decides only that. The program works out the rest:
- every new pile of an extraction area ("A/1", "A/2" …) is a blast: its holes must be drilled (metres = tonnes / t per
  metre, shifts = metres / the rig's output) and fired before the pile can be loaded – scheduled backwards from the
  shift the planner first sends an excavator there, on the free drill rigs, strips of an area in their order;
- what does not fit (no free rig early enough) becomes a conflict with the earliest shift the pile can be there;
- an excavator sent to a pile that is not there yet, or already dug out (estimated from today's loading rates);
- the frozen schedule (PeriodPlan.frozen) both months follow – today's driving and the driving signals – so the two
  differ only in how the trucks drive.
"""
from __future__ import annotations

import math

LATE = "late"
EMPTY = "empty"
NOT_YET = "not_yet"


def area_blast_t(area: dict, cap_t: float = math.inf) -> float:
    """Size of one blast in an area: its own size, never more than the explosive limit allows."""
    return min(float(area.get("blast_max_t") or 20000.0), cap_t)


def area_of(pid: str, areas: dict) -> tuple[str, int] | None:
    """('A', 2) for 'A/2' when A is an extraction area."""
    if "/" not in str(pid):
        return None
    a, k = str(pid).rsplit("/", 1)
    return (a, int(k)) if a in areas and k.isdigit() and int(k) >= 1 else None


def plan_blasts(grid: dict[str, list[str]], areas: list[dict], rigs: list[dict], n: int, cap_t: float = math.inf,
                fire_shifts: int = 1, t_per_m: float = 50.0) -> dict[str, dict]:
    """Drilling and firing of every new pile the grid uses. Returns {pile: {area, t, m, need, avail, fire, drill,
    late}}: `avail` = first shift the pile can be loaded (the planner's first use, or later when it cannot be
    ready), `fire` = the shifts of the blast (evacuation, firing, airing) just before it, `drill` = [(rig, shift)]."""
    by_id = {a["id"]: a for a in areas}
    need: dict[str, int] = {}
    for cells in grid.values():
        for i, p in enumerate(cells[:n]):
            if area_of(p, by_id):
                need[p] = min(need.get(p, n), i)
    # the strips of an area are fired in their order: an earlier strip is needed no later than a later one
    for a in by_id:
        ks = sorted(area_of(p, by_id)[1] for p in need if p.startswith(f"{a}/"))
        if not ks:
            continue
        for k in range(ks[-1], 1, -1):
            later, earlier = f"{a}/{k}", f"{a}/{k - 1}"
            need[earlier] = min(need.get(earlier, n), need.get(later, n))
    rig_m = {r["id"]: float(r.get("m_per_shift") or 250.0) for r in rigs if r.get("in_work", True)}
    busy: dict[str, set] = {r: set() for r in rig_m}
    out: dict[str, dict] = {}
    for pid in sorted(need, key=lambda p: (need[p], p)):
        a_id, k = area_of(pid, by_id)
        a = by_id[a_id]
        t = area_blast_t(a, cap_t)
        m = t / float(a.get("t_per_m") or t_per_m)
        ready = k == 1 and a.get("first_drilled")  # the first strip may be drilled already
        first, prev = need[pid], out.get(f"{a_id}/{k - 1}")
        if prev:  # never before the previous strip of the same area
            first = max(first, prev["avail"])
        best = None  # (start, rig, shifts): the latest window that still ends before the blast
        if not ready and rig_m:
            for rid, mps in rig_m.items():
                d = max(1, math.ceil(m / mps))
                end = first - fire_shifts
                for s in range(end - d, -1, -1):
                    if all(x not in busy[rid] for x in range(s, s + d)):
                        if best is None or s > best[0]:
                            best = (s, rid, d)
                        break
        late = False
        if ready or not rig_m:  # drilled already (or drilling not modelled): only the firing is left
            avail, drill = max(first, fire_shifts), []
        elif best:
            s, rid, d = best
            avail, drill = first, [(rid, x) for x in range(s, s + d)]
        else:  # no rig free early enough: the earliest it can be done
            late = True
            cands = []
            for rid, mps in rig_m.items():
                d = max(1, math.ceil(m / mps))
                s = 0
                while any(x in busy[rid] for x in range(s, s + d)):
                    s += 1
                cands.append((s + d + fire_shifts, s, rid, d))
            avail, s, rid, d = min(cands)
            drill = [(rid, x) for x in range(s, s + d)]
        for rid, x in drill:
            busy[rid].add(x)
        out[pid] = {"area": a_id, "t": round(t), "m": round(m), "need": need[pid], "avail": avail,
                    "fire": [x for x in range(avail - fire_shifts, avail) if x >= 0], "drill": drill,
                    "late": late or avail > need[pid]}
    return out


def check(grid: dict[str, list[str]], blasts: dict[str, dict], start_t: dict[str, float], rates: dict[str, float],
          n: int) -> list[dict]:
    """What does not add up: an excavator on a pile that is not there yet, or already dug out (estimated with
    today's loading per shift). One issue per excavator and pile: {ex, shift, pile, kind, avail}."""
    left = dict(start_t)
    issues, seen = [], set()
    for i in range(n):
        for p, b in blasts.items():
            if b["avail"] == i:
                left[p] = left.get(p, 0.0) + b["t"]
        on: dict[str, list[str]] = {}
        for ex, cells in grid.items():
            p = cells[i] if i < len(cells) else ""
            if not p:
                continue
            if p in blasts and blasts[p]["avail"] > i:
                if (ex, p, NOT_YET) not in seen:
                    seen.add((ex, p, NOT_YET))
                    issues.append({"ex": ex, "shift": i, "pile": p, "kind": NOT_YET, "avail": blasts[p]["avail"]})
                continue
            if left.get(p, 0.0) <= 1.0:
                if (ex, p, EMPTY) not in seen:
                    seen.add((ex, p, EMPTY))
                    issues.append({"ex": ex, "shift": i, "pile": p, "kind": EMPTY, "avail": None})
                continue
            on.setdefault(p, []).append(ex)
        for p, exs in on.items():
            left[p] = max(0.0, left[p] - sum(rates.get(x, 0.0) for x in exs))
    return issues


def depletion(grid: dict[str, list[str]], blasts: dict[str, dict], start_t: dict[str, float],
              rates: dict[str, float], n: int) -> dict[str, list[float]]:
    """Tonnes lying on every pile at the start of every shift (estimate with today's loading per shift)."""
    left = dict(start_t)
    out = {p: [] for p in list(start_t) + list(blasts)}
    for i in range(n):
        for p, b in blasts.items():
            if b["avail"] == i:
                left[p] = left.get(p, 0.0) + b["t"]
        for p in out:
            out[p].append(round(left.get(p, 0.0)))
        on: dict[str, float] = {}
        for ex, cells in grid.items():
            p = cells[i] if i < len(cells) else ""
            if p and left.get(p, 0.0) > 1.0 and not (p in blasts and blasts[p]["avail"] > i):
                on[p] = on.get(p, 0.0) + rates.get(ex, 0.0)
        for p, r in on.items():
            left[p] = max(0.0, left[p] - r)
    return out


def propose(excavators: list, start_t: dict[str, float], areas: list[dict], rates: dict[str, float], n: int,
            cap_t: float = math.inf) -> dict[str, list[str]]:
    """A first schedule to start from: every excavator stays on its pile while it lasts, then takes the next blast
    of an area it may work in (areas taken in turns). The planner corrects it."""
    left = dict(start_t)
    nxt = {a["id"]: 1 for a in areas}
    turn = {e.id: j for j, e in enumerate(excavators)}
    cur = {e.id: e.blast for e in excavators}
    grid = {e.id: [] for e in excavators}
    for i in range(n):
        for e in excavators:
            rate = rates.get(e.id, 0.0)
            shared = sum(rates.get(x.id, 0.0) for x in excavators if cur[x.id] == cur[e.id])
            if rate > 0 and left.get(cur[e.id], 0.0) < 0.5 * max(shared, rate):
                allowed = [a for a in areas if e.may_work(a["id"]) and float(a.get("reserves_t") or 0)
                           > (nxt[a["id"]] - 1) * area_blast_t(a, cap_t)]
                if allowed:
                    a = allowed[turn[e.id] % len(allowed)]
                    turn[e.id] += 1
                    pid = f"{a['id']}/{nxt[a['id']]}"
                    nxt[a["id"]] += 1
                    left[pid] = area_blast_t(a, cap_t)
                    cur[e.id] = pid
            grid[e.id].append(cur[e.id])
        on: dict[str, float] = {}
        for e in excavators:
            on[cur[e.id]] = on.get(cur[e.id], 0.0) + rates.get(e.id, 0.0)
        for p, r in on.items():
            left[p] = left.get(p, 0.0) - r
    return grid


def frozen(grid: dict[str, list[str]], blasts: dict[str, dict], n: int) -> dict:
    """The schedule for PeriodPlan.frozen: per shift the excavators' piles and the blasts that become ready."""
    return {"ex": [{ex: cells[i] for ex, cells in grid.items() if i < len(cells) and cells[i]} for i in range(n)],
            "blasts": [sorted(p for p, b in blasts.items() if b["avail"] == i) for i in range(n)]}
