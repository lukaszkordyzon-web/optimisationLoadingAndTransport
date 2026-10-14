"""A week / month of work: shifts one after another, with the mine state carried over.

Every shift is the usual single-shift simulation. Between shifts the period keeps:
  * the material left in every pile (it runs out – and comes back with the next blast),
  * blasts from the schedule (+ tonnes to a pile on given working days; the blasting closure
    only happens on that day's first shift),
  * the crusher stockpile level,
  * how far each excavator has advanced along its digging line.
Weather, breakdowns and drivers are drawn per shift (different seed every shift).
"""
from __future__ import annotations

import copy
import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Scenario, _routes
from .sim import simulate
from .site import BLAST_TIMES, _crusher_routes, area_strips, build_route, centroid, loading_point, poly_distance_m

WEEKDAYS = ["pon", "wt", "śr", "czw", "pt", "sob", "nd"]


@dataclass
class PileBlasts:
    """Blasting into one pile: every `every_days` working days, starting on working day `first_day` (1 = first)."""
    pile: str
    tonnage_t: float
    every_days: int = 0  # 0 = no blasts in the period (only what is in the pile now)
    first_day: int = 1
    share: float | None = None  # desired share of the crusher feed from this pile [%] (None = no target)


@dataclass
class PeriodPlan:
    start: dt.date
    days: int = 30                    # calendar days
    shifts_per_day: int = 2
    saturdays: bool = False
    sundays: bool = False
    holidays: list = field(default_factory=list)  # dates without work
    blasts: list = field(default_factory=list)    # [PileBlasts]
    target_t: float = 0.0             # production target for the period (0 = none)
    move_excavators: bool = True      # an excavator without material drives to the nearest free blasted pile
    move_setup_min: float = 15.0      # + this on top of the drive (preparing the new face)
    # drilling: blasts in new places must be drilled first (0 rigs = drilling not modelled)
    drills: int = 0
    drill_speed_kmh: float = 2.0      # moving between blasts
    drill_m_per_shift: float = 250.0  # drilling output of one rig per shift [m]
    # particular rigs with their own output, speed and breakdowns: [{id, m_per_shift, speed_kmh}] (overrides drills)
    rigs: list = field(default_factory=list)
    clear_pct: float = 10.0           # the next strip of an area can be drilled when ≤ this % of the previous blast is left
    t_per_m: float = 50.0             # default tonnes per metre drilled (when a blast / area has none)
    lead_m: float = 30.0              # a lower bench is blasted only this far behind the upper bench face

    def working_days(self) -> list[dt.date]:
        out = []
        for i in range(self.days):
            d = self.start + dt.timedelta(days=i)
            if d in self.holidays or (d.weekday() == 5 and not self.saturdays) or (d.weekday() == 6 and not self.sundays):
                continue
            out.append(d)
        return out

    def shifts(self) -> list[tuple[dt.date, int, int]]:
        """[(date, working-day number from 1, shift number from 1)]."""
        return [(d, i + 1, s + 1) for i, d in enumerate(self.working_days()) for s in range(self.shifts_per_day)]


def _shift_start(first: str, shift_no: int, hours: float) -> str:
    h, m = (int(x) for x in first.split(":"))
    total = (h * 60 + m + int((shift_no - 1) * hours * 60)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


MIN_PILE_T = 50.0  # less than this in a pile = nothing to load


def _blends_ok(sc: Scenario, ex_pile: dict, ex_id: str, pile: str) -> bool:
    """Trucks loading at two excavators need a crusher that takes both materials – also after a move."""
    for f in sc.trucks:
        if f.count and f.blend and ex_id in f.blend:
            piles = [pile if x == ex_id else ex_pile[x] for x in f.blend]
            if not any(all(c.takes(p) for p in piles) for c in sc.crushers):
                return False
    return True


def run_period(sc: Scenario, plan: PeriodPlan, seed: int = 0, progress=None) -> pd.DataFrame:
    """One row per shift. `progress(i, n)` is called after every shift (for a progress bar)."""
    from .config import Blast
    from .site import haversine_m

    shifts = plan.shifts()
    left = {b.id: float(b.tonnage_t) for b in sc.blasts}  # material in the piles now
    planned = {p["id"]: p for p in sc.planned_blasts if p["id"] not in left}
    for pid in planned:
        left[pid] = 0.0  # not blasted yet
    # where trucks load at every pile: the route of the excavator that works it / of the planned blast
    pile_route = {p["id"]: p["route"] for p in planned.values()}
    pile_croutes = {p["id"]: dict(p.get("crusher_routes") or {}) for p in planned.values()}
    for e in sc.excavators:
        pile_route.setdefault(e.blast, e.route)
        pile_croutes.setdefault(e.blast, dict(e.crusher_routes))
    ex_croutes = {e.id: dict(e.crusher_routes) for e in sc.excavators}
    taken = {p: any(c.takes(p) for c in sc.crushers) for p in left}  # some crusher accepts this material
    ex_pile = {e.id: e.blast for e in sc.excavators}
    ex_route = {e.id: e.route for e in sc.excavators}
    stock = float(sc.stockpile.initial_t)
    dig_step = {e.id: 0 for e in sc.excavators}
    blasts = {b.pile: b for b in plan.blasts}
    first_start = sc.shift_plan.start_time
    move_kmh = sc.traffic.excavator_move_kmh
    # feed mix targets: the period plan overrides the shares of the existing piles, planned blasts bring their own
    share = {b.id: b.target_share for b in sc.blasts}
    share |= {b.pile: b.share for b in plan.blasts if b.pile in share}
    share |= {pid: p.get("target_share") for pid, p in planned.items()}
    # extraction areas: blasts are fired automatically, strip after strip, when the area runs short of material
    areas = []
    for a in sc.areas:
        reserve = float(a.get("reserves_t") or 0)
        size = max(float(a.get("blast_max_t") or 0), 1.0)
        n = min(60, max(1, math.ceil(reserve / size))) if reserve > 0 else 0
        strips = area_strips(sc.site or {}, a["polygon"], n) if sc.site and n else []
        areas.append({"cfg": a, "reserve": reserve, "strips": strips, "next": 0, "last_day": -10_000})
    blast_log: list[dict] = []
    warnings: list[str] = []
    waits: dict[str, str] = {}  # blast -> why it is held back by the bench order (latest reason)
    # drilling jobs: blast id -> metres to drill / drilled; rigs move between them
    rig_cfg = [dict(r) for r in plan.rigs] or [{"id": f"W-{i + 1}", "m_per_shift": plan.drill_m_per_shift,
                                                 "speed_kmh": plan.drill_speed_kmh} for i in range(plan.drills)]
    drilling = bool(rig_cfg)
    jobs: dict[str, dict] = {}
    rigs = []
    for c in rig_cfg:
        key = c["id"] if c["id"] in sc.reliability.machines else "drill"  # own breakdown data, else the type's
        mtbf, mttr = sc.reliability.rates(key, "drill", sc.shift_hours)
        rigs.append({"id": c["id"], "m": float(c.get("m_per_shift") or plan.drill_m_per_shift),
                     "kmh": float(c.get("speed_kmh") or plan.drill_speed_kmh), "mtbf": mtbf, "mttr": mttr,
                     "pos": None, "job": None, "travel_h": 0.0, "down_h": 0.0, "lost_h": 0.0})
    frng = np.random.default_rng(seed + 424_243)  # drill rig breakdowns
    carry: dict = {}  # repairs going on from one shift into the next: machine / truck id -> seconds
    prev_end: dt.datetime | None = None
    drill_log: list[dict] = []
    fired: set[str] = set()

    def tpm(src: dict) -> float:
        return float(src.get("t_per_m") or plan.t_per_m or 50.0)

    for pid, p in planned.items():
        m = float(p["tonnage_t"]) / tpm(p)
        jobs[pid] = {"m": m, "done": m if not drilling or p.get("drilled") else 0.0, "poly": p.get("polygon"),
                     "prio": (0, int(p.get("blast_day") or 1))}

    def area_job(ar) -> str | None:
        """The next strip of an area becomes a drilling job once the previous blast has been taken out."""
        a = ar["cfg"]
        if ar["reserve"] <= 1 or ar["next"] >= len(ar["strips"]):
            return None
        pid = f"{a['id']}/{ar['next'] + 1}"
        if pid in jobs:
            return pid
        if ar["next"] > 0:
            prev = f"{a['id']}/{ar['next']}"
            if prev not in planned or left.get(prev, 0.0) > plan.clear_pct / 100 * float(planned[prev]["tonnage_t"]):
                return None  # previous blast not taken out yet – no access for the rigs
        m = min(float(a["blast_max_t"]), ar["reserve"]) / tpm(a)
        first_ready = ar["next"] == 0 and a.get("first_drilled", True)
        jobs[pid] = {"m": m, "done": m if not drilling or first_ready else 0.0, "poly": ar["strips"][ar["next"]],
                     "prio": (1, ar["next"])}
        return pid

    def taken_out(pid: str) -> bool:
        return pid in planned and left.get(pid, 0.0) <= plan.clear_pct / 100 * float(planned[pid]["tonnage_t"])

    def front_ok(ar) -> str | None:
        """Benches: the next strip of a lower area may be blasted only when every part of a HIGHER area closer
        than the minimum lead is already blasted and taken out. Returns the blocking area, or None."""
        level = float(ar["cfg"].get("level_m") or 0.0)
        strip = ar["strips"][ar["next"]]
        for up in areas:
            if up is ar or float(up["cfg"].get("level_m") or 0.0) <= level + 0.5:
                continue
            uid = up["cfg"]["id"]
            for k, poly in enumerate(up["strips"]):
                done = k < up["next"] and taken_out(f"{uid}/{k + 1}")
                if not done and poly_distance_m(strip, poly) < plan.lead_m:
                    return f"{uid}/{k + 1}"
        return None

    def ready(pid: str) -> bool:
        return pid in jobs and jobs[pid]["done"] >= jobs[pid]["m"] - 1e-6

    def drill_shift(day, shift_no):
        """One shift of drilling: free rigs take the most urgent job, drive there and drill."""
        if not drilling:
            return
        for r in rigs:
            if r["job"] is None:
                todo = sorted((j["prio"], pid) for pid, j in jobs.items()
                              if not ready(pid) and pid not in fired and not any(x["job"] == pid for x in rigs))
                if todo:
                    pid = todo[0][1]
                    r["job"] = pid
                    poly = jobs[pid]["poly"]
                    if r["pos"] is not None and poly:
                        r["travel_h"] = haversine_m(r["pos"], centroid(poly)) / 1000 / max(r["kmh"], 0.1)
        for r in rigs:
            # breakdowns: a repair from before eats into this shift; a new one may stop the rig for the rest of it
            avail = sc.shift_hours - min(sc.shift_hours, r["down_h"])
            r["lost_h"] = sc.shift_hours - avail
            r["down_h"] = max(0.0, r["down_h"] - sc.shift_hours)
            if avail > 0 and r["mtbf"] < math.inf and frng.random() < 1 - math.exp(-avail / r["mtbf"]):
                at = frng.uniform(0, avail)
                repair = float(frng.lognormal(math.log(r["mttr"]) - 0.5, 1.0))
                lost = min(avail - at, repair)
                r["down_h"] += repair - lost
                r["lost_h"] += lost
                avail -= lost
            if r["job"] is None:
                continue
            j = jobs[r["job"]]
            move = min(avail, r["travel_h"])
            r["travel_h"] -= move
            j["done"] = min(j["m"], j["done"] + r["m"] * (avail - move) / sc.shift_hours)
            if ready(r["job"]):
                drill_log.append({"strzal": r["job"], "wiertnica": r["id"], "data": day, "zmiana": shift_no})
                if j["poly"]:
                    r["pos"] = centroid(j["poly"])
                r["job"] = None

    rows = []
    for i, (day, wd, shift_no) in enumerate(shifts):
        blasted = {}
        def self_blast(ar, a):
            """Fire the next strip of an extraction area (a new pile with its own route)."""
            pid = f"{a['id']}/{ar['next'] + 1}"
            poly = ar["strips"][ar["next"]]
            ar["next"] += 1
            t = min(float(a["blast_max_t"]), ar["reserve"])
            load_at = loading_point(sc.site, poly)
            try:
                sc.routes[f"P-{pid}"] = _routes({"r": build_route(sc.site, load_at, f"P-{pid}")})["r"]
            except ValueError:
                warnings.append(f"Strzał {pid}: brak dojazdu drogą – doprowadź drogę do obszaru {a['id']}")
                return
            raw_routes: dict = {}
            croutes = _crusher_routes(sc.site, load_at, f"P-{pid}", raw_routes)
            for rid in croutes.values():
                sc.routes[rid] = _routes({"r": raw_routes[rid]})["r"]
            ar["reserve"] -= t
            ar["last_day"] = wd
            planned[pid] = {"id": pid, "tonnage_t": t, "material": a.get("material", ""), "auto": True,
                            "fragmentation": float(a.get("fragmentation", 0.3)), "psd": a.get("psd"),
                            "loose_density_t_m3": a.get("loose_density_t_m3"), "polygon": poly,
                            **{k: a[k] for k in BLAST_TIMES if a.get(k)}}
            left[pid], pile_route[pid], pile_croutes[pid] = t, f"P-{pid}", croutes
            taken[pid] = any(c.takes(pid) for c in sc.crushers)
            share[pid] = a.get("target_share")
            blasted[pid] = t
            blast_log.append({"strzal": pid, "obszar": a["id"], "data": day, "t": t, "polygon": poly, "auto": True})

        if shift_no == 1:  # blasts are fired on the first shift of a working day
            for pid, b in blasts.items():
                if b.every_days > 0 and wd >= b.first_day and (wd - b.first_day) % b.every_days == 0 and pid in left:
                    left[pid] += float(b.tonnage_t)
                    blasted[pid] = float(b.tonnage_t)
            for pid, p in list(planned.items()):
                if p.get("auto") or pid in fired or int(p.get("blast_day") or 1) > wd:
                    continue
                if not ready(pid):
                    if int(p.get("blast_day") or 1) == wd:
                        warnings.append(f"Strzał {pid}: w dniu {wd} otwór niegotowy (wiercenie) – strzał przesunięty")
                    continue
                fired.add(pid)
                left[pid] += float(p["tonnage_t"])
                blasted[pid] = float(p["tonnage_t"])
                late = wd - int(p.get("blast_day") or 1)
                blast_log.append({"strzal": pid, "obszar": "", "data": day, "t": float(p["tonnage_t"]),
                                  "polygon": p.get("polygon"), "auto": False, "opoznienie_dni": late})
            # excavators without material and no free pile to go to = blasts needed now
            idle = [x for x in ex_pile if left[ex_pile[x]] < MIN_PILE_T]
            busy_piles = {ex_pile[x] for x in ex_pile if left[ex_pile[x]] >= MIN_PILE_T}
            free_piles = [p for p in left if left[p] >= MIN_PILE_T and p not in busy_piles and taken.get(p)]
            need = max(0, len(idle) - len(free_piles)) if plan.move_excavators else 0
            for ar in areas:
                a = ar["cfg"]
                open_t = sum(t for pid, t in left.items() if pid.startswith(f"{a['id']}/"))
                if ar["reserve"] <= 1 or wd - ar["last_day"] < int(a.get("min_gap_days") or 1):
                    continue
                n_blasts = max(1 if open_t < float(a.get("reblast_below_t") or 0) else 0, need)
                n_blasts = min(n_blasts, int(a.get("max_per_day") or 1), len(ar["strips"]) - ar["next"])
                for _ in range(n_blasts):
                    pid = area_job(ar)
                    if pid is None or not ready(pid):
                        break  # the strip is not drilled yet (or the previous blast is not taken out)
                    blocker = front_ok(ar)
                    if not blocker:
                        waits.pop(pid, None)
                    if blocker:
                        waits[pid] = (f"{day.strftime('%d.%m')}: strzał {pid} czeka na wyprzedzenie frontu – najpierw "
                                      f"{blocker} na wyższym poziomie (min. {plan.lead_m:.0f} m)")
                        break
                    need = max(0, need - 1)
                    fired.add(pid)
                    self_blast(ar, a)

        # an excavator without material drives to the nearest blasted pile nobody else works
        moves, delay = [], {}
        if plan.move_excavators:
            for e in sc.excavators:
                if left[ex_pile[e.id]] >= MIN_PILE_T:
                    continue
                busy = {ex_pile[x] for x in ex_pile if x != e.id and left[ex_pile[x]] >= MIN_PILE_T}
                free = [p for p in left if left[p] >= MIN_PILE_T and p not in busy and p in pile_route and taken[p]
                        and _blends_ok(sc, ex_pile, e.id, p)]
                if not free:
                    continue
                here = sc.routes[ex_route[e.id]].path
                def dist(p, here=here):
                    there = sc.routes[pile_route[p]].path
                    return haversine_m(here[0], there[0]) if here and there else 0.0
                target = min(free, key=dist)
                d = dist(target)
                delay[e.id] = round(d / (move_kmh / 3.6) / 60 + plan.move_setup_min, 1)
                moves.append(f"{e.id} → {target} ({round(d):,} m".replace(",", " ") + f"; {delay[e.id]:.0f} min)")
                ex_pile[e.id], ex_route[e.id] = target, pile_route[target]
                ex_croutes[e.id] = pile_croutes.get(target, {})
        v = copy.deepcopy(sc)
        for b in v.blasts:
            b.tonnage_t = max(left[b.id], 0.0)
        for pid, p in planned.items():
            v.blasts.append(Blast(id=pid, tonnage_t=max(left[pid], 0.0), fragmentation=float(p.get("fragmentation", 0.3)),
                                  material=p.get("material", ""), loose_density_t_m3=p.get("loose_density_t_m3"),
                                  psd=p.get("psd")))
        # only piles that can be loaded this shift take part in the feed mix (shares are renormalised)
        worked = {ex_pile[x] for x in ex_pile if left[ex_pile[x]] >= MIN_PILE_T}
        for b in v.blasts:
            b.target_share = share.get(b.id) if b.id in worked and share.get(b.id) else None
        v.stockpile.initial_t = stock
        v.blasting.enabled = bool(sc.blasting.enabled and blasted)
        timed = next((planned[p] for p in blasted if p in planned and planned[p].get("fire_from")), None)
        if timed:  # the blast brings its own crew times and place
            v.blasting.enabled = True
            for k_ in BLAST_TIMES:
                if timed.get(k_):
                    setattr(v.blasting, k_, timed[k_])
            if timed.get("polygon") and v.site is not None:
                c_ = centroid(timed["polygon"])
                v.site["blast"] = {"lat": c_[0], "lon": c_[1]}
        v.shift_plan.start_time = _shift_start(first_start, shift_no, sc.shift_hours)
        # repairs go on between shifts (nights, weekends) – what is left at this shift's start
        hh, mm = (int(x) for x in v.shift_plan.start_time.split(":"))
        start_dt = dt.datetime.combine(day, dt.time(hh, mm))
        if shift_no > 1 and start_dt.time() < dt.datetime.strptime(first_start, "%H:%M").time():
            start_dt += dt.timedelta(days=1)  # a night shift starts after midnight
        gap = (start_dt - prev_end).total_seconds() if prev_end else 0.0
        v.carry_down = {k_: s_ - gap for k_, s_ in carry.items() if s_ - gap > 0}
        gap_h = gap / 3600
        for r in rigs:
            r["down_h"] = max(0.0, r["down_h"] - gap_h)
        prev_end = start_dt + dt.timedelta(hours=sc.shift_hours)
        for e in v.excavators:
            moved = ex_pile[e.id] != e.blast
            e.blast, e.route, e.crusher_routes = ex_pile[e.id], ex_route[e.id], ex_croutes[e.id]
            e.start_delay_min = delay.get(e.id, 0.0)
            if moved:  # on another pile: no digging line there
                e.dig_routes, e.dig_path = [], None
            elif e.dig_routes:  # continue along the face where the excavator stopped
                k = min(dig_step[e.id], len(e.dig_routes) - 1)
                e.route, e.dig_routes = e.dig_routes[k], e.dig_routes[k:]
        res = simulate(v, seed=seed + 7919 * i)
        k = res.kpi
        carry = dict(k.get("carry_down") or {})
        loaded = res.loads.groupby("blast")["tonnes"].sum() if len(res.loads) else pd.Series(dtype=float)
        for pid in left:
            left[pid] = max(0.0, left[pid] - float(loaded.get(pid, 0.0)))
        stock = float(k.get("stockpile_end_t", stock) or 0.0)
        for ar in areas:
            area_job(ar)
        drill_shift(day, shift_no)
        for e in v.excavators:
            if e.dig_routes:
                dig_step[e.id] += int(res.excavator_steps.get(e.id, 0))
        idle_empty = []
        for e in k["excavators"]:
            if v.blast(ex_pile[e["excavator"]]).tonnage_t <= 1:
                idle_empty.append(f"{e['excavator']} (od rana)")
            elif e.get("blast_done_at_h") is not None and e["blast_done_at_h"] < sc.shift_hours - 0.5:
                idle_empty.append(f"{e['excavator']} (od {e['blast_done_at_h']:.1f} h)")
        rows.append({
            "data": day, "dzien": WEEKDAYS[day.weekday()], "dzien_roboczy": wd, "zmiana": shift_no,
            "start": v.shift_plan.start_time, "deszcz": bool(k["rain"]),
            "wywiezione_t": k["hauled_t"], "koszt_zl": k["cost_total"], "koszt_na_t": k["cost_per_t"],
            "paliwo_l": k["fuel_l"], "kolejki_h": round(k["truck_queue_excavator_h"] + k["truck_queue_crusher_h"], 2),
            "koparki_czekaly_h": round(sum(e["waiting_for_trucks_h"] for e in k["excavators"]), 2),
            "kruszarka_bez_nadawy_h": k["crusher_starved_h"],
            "awarie_h": round(k["truck_breakdown_h"] + k["excavator_down_h"]["awaria"] + k["crusher_breakdown_h"]
                              + sum(r["lost_h"] for r in rigs), 2),
            "awarie_wozidel_h": k["truck_breakdown_h"], "awarie_koparek_h": k["excavator_down_h"]["awaria"],
            "awarie_kruszarek_h": k["crusher_breakdown_h"],
            "awarie_wiertnic_h": round(sum(r["lost_h"] for r in rigs), 2),
            **{f"awaria_{c['crusher']}_h": c["down_h"] for c in k.get("crushers", [])},
            "strzal": ", ".join(f"{p} +{t:,.0f} t".replace(",", " ") for p, t in blasted.items()),
            "przejazdy_koparek": ", ".join(moves),
            "wiercenie": ", ".join(f"{r['id']}: {r['job']} {jobs[r['job']]['done']:.0f}/{jobs[r['job']]['m']:.0f} m"
                                   for r in rigs if r["job"]),
            "koparki_na": ", ".join(f"{x}:{ex_pile[x]}" for x in ex_pile),
            "skonczyl_sie_urobek": ", ".join(idle_empty),
            "sklad_cel": ", ".join(f"{b} {t:.0f}%" for b, t in (k.get("feed_target") or {}).items() if t > 0),
            "sklad": ", ".join(f"{b} {t:.0f}%" for b, t in (k.get("feed_share") or {}).items() if t > 0),
            "sklad_odchylka_pp": k.get("feed_dev_pp", 0.0),
            **{f"zostalo_{p}_t": round(left[p]) for p in left},
            **{f"z_{p}_t": round(float(loaded.get(p, 0.0))) for p in left},
            "halda_t": round(stock),
            **{f"kr_{c['crusher']}_t": round(c["crushed_t"]) for c in k.get("crushers", [])},
        })
        if progress:
            progress(i + 1, len(shifts))
    df = pd.DataFrame(rows)
    zero = [c for c in df.columns if (c.startswith("z_") or c.startswith("zostalo_")) and c.endswith("_t")]
    df[zero] = df[zero].fillna(0)  # piles blasted during the period have no values before their blast
    df.attrs["blasts"] = blast_log
    df.attrs["drilling"] = drill_log
    df.attrs["warnings"] = sorted(set(warnings))
    df.attrs["waits"] = [waits[k] for k in sorted(waits)]
    return df


def _shares(df: pd.DataFrame) -> dict:
    """Share of every pile in the material loaded over the period [%]."""
    cols = [c for c in df.columns if c.startswith("z_") and c.endswith("_t") and df[c].sum() > 0]
    total = float(sum(df[c].sum() for c in cols))
    return {c[2:-2]: round(100 * float(df[c].sum()) / total, 1) for c in cols} if total else {}


def summary(df: pd.DataFrame, plan: PeriodPlan, crushers: list | None = None) -> dict:
    if df.empty:
        return {}
    t = float(df["wywiezione_t"].sum())
    cost = float(df["koszt_zl"].sum())
    return {
        "zmian": len(df), "dni_roboczych": int(df["dzien_roboczy"].max()),
        "wywiezione_t": round(t), "koszt_zl": round(cost), "koszt_na_t": round(cost / t, 2) if t else None,
        "srednio_na_zmiane_t": round(t / len(df)),
        "cel_t": plan.target_t, "realizacja_celu_pct": round(100 * t / plan.target_t, 1) if plan.target_t else None,
        "zmian_bez_urobku": int((df["skonczyl_sie_urobek"] != "").sum()),
        "udzialy_pct": _shares(df),
        "kruszarki": [{"kruszarka": c.id, "produkt": c.product or "–",
                       "skruszone_t": round(float(df[f"kr_{c.id}_t"].sum())) if f"kr_{c.id}_t" in df else 0,
                       "plan_t": round(c.target_t * len(df)) if c.target_t else None,
                       "wykonanie_pct": round(100 * float(df[f"kr_{c.id}_t"].sum()) / (c.target_t * len(df)), 1)
                       if c.target_t and f"kr_{c.id}_t" in df else None} for c in crushers] if crushers else [],
        "srednia_odchylka_skladu_pp": round(float((df["sklad_odchylka_pp"] * df["wywiezione_t"]).sum() / t), 1) if t else 0.0,
        "kruszarka_bez_nadawy_h": round(float(df["kruszarka_bez_nadawy_h"].sum()), 1),
    }


# ---------------------------------------------------------------- main goal of the period: choose the plan
GOALS = {
    "cheapest": "💰 Najtańsza tona (najniższy koszt zł/t)",
    "max": "🏭 Największa produkcja (najwięcej ton)",
    "deadline": "🎯 X ton do dnia – najtaniej",
}


def plan_metrics(df: pd.DataFrame, target_t: float = 0.0, deadline: dt.date | None = None) -> dict:
    """What a calculated period gives: tonnes, cost, and when the cumulative target is reached."""
    t = float(df["wywiezione_t"].sum())
    cost = float(df["koszt_zl"].sum())
    out = {"wywiezione_t": round(t), "koszt_zl": round(cost), "koszt_na_t": round(cost / t, 2) if t else None,
           "do_terminu_t": None, "cel_osiagniety": None}
    if target_t:
        cum = df.groupby("data")["wywiezione_t"].sum().cumsum()
        hit = cum[cum >= target_t]
        out["cel_osiagniety"] = hit.index[0] if len(hit) else None
        if deadline:
            out["do_terminu_t"] = round(float(cum[cum.index <= deadline].iloc[-1])) if (cum.index <= deadline).any() else 0
    return out


def score(m: dict, goal: str, target_t: float = 0.0, deadline: dt.date | None = None) -> tuple:
    """Sort key: smaller = better."""
    cpt = m["koszt_na_t"] if m["koszt_na_t"] is not None else 1e9
    if goal == "cheapest":
        return (cpt,)
    if goal == "max":
        return (-m["wywiezione_t"], cpt)
    ok = m["cel_osiagniety"] is not None and (deadline is None or m["cel_osiagniety"] <= deadline)
    # plans that make it in time first (cheapest of them), then the ones closest to the target
    return (0, cpt) if ok else (1, -(m["do_terminu_t"] or m["wywiezione_t"]))


def optimize_period(sc: Scenario, plan: PeriodPlan, goal: str, trucks: list[int], shifts: list[int],
                    saturdays: list[bool], dispatches: list[str], target_t: float = 0.0,
                    deadline: dt.date | None = None, seed: int = 0, progress=None) -> pd.DataFrame:
    """Calculate the period for every combination of the levers (same weather and breakdowns for all –
    common random numbers) and rank them by the main goal."""
    import itertools

    combos = list(itertools.product(trucks, shifts, saturdays, dispatches))
    rows = []
    for i, (n, sh, sat, disp) in enumerate(combos):
        v = sc.with_truck_count(int(n))
        v.dispatch = disp
        p = copy.deepcopy(plan)
        p.shifts_per_day, p.saturdays = int(sh), bool(sat)
        df = run_period(v, p, seed=seed)
        m = plan_metrics(df, target_t, deadline)
        rows.append({"wozidla": int(n), "zmian_na_dobe": int(sh), "soboty": bool(sat), "przydzial": disp,
                     "zmian": len(df), **m, "_df": df, "_score": score(m, goal, target_t, deadline)})
        if progress:
            progress(i + 1, len(combos))
    rows.sort(key=lambda r: r["_score"])
    out = pd.DataFrame(rows)
    out.insert(0, "miejsce", range(1, len(out) + 1))
    return out
