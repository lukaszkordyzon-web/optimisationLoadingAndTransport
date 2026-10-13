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
from dataclasses import dataclass, field

import pandas as pd

from .config import Scenario
from .sim import simulate

WEEKDAYS = ["pon", "wt", "śr", "czw", "pt", "sob", "nd"]


@dataclass
class PileBlasts:
    """Blasting into one pile: every `every_days` working days, starting on working day `first_day` (1 = first)."""
    pile: str
    tonnage_t: float
    every_days: int = 0  # 0 = no blasts in the period (only what is in the pile now)
    first_day: int = 1


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


def run_period(sc: Scenario, plan: PeriodPlan, seed: int = 0, progress=None) -> pd.DataFrame:
    """One row per shift. `progress(i, n)` is called after every shift (for a progress bar)."""
    shifts = plan.shifts()
    left = {b.id: float(b.tonnage_t) for b in sc.blasts}  # material in the piles now
    stock = float(sc.stockpile.initial_t)
    dig_step = {e.id: 0 for e in sc.excavators}
    blasts = {b.pile: b for b in plan.blasts}
    first_start = sc.shift_plan.start_time
    rows = []
    for i, (day, wd, shift_no) in enumerate(shifts):
        blasted = {}
        if shift_no == 1:  # blasts are fired on the first shift of a working day
            for pid, b in blasts.items():
                if b.every_days > 0 and wd >= b.first_day and (wd - b.first_day) % b.every_days == 0 and pid in left:
                    left[pid] += float(b.tonnage_t)
                    blasted[pid] = float(b.tonnage_t)
        v = copy.deepcopy(sc)
        for b in v.blasts:
            b.tonnage_t = max(left[b.id], 0.0)
        v.stockpile.initial_t = stock
        v.blasting.enabled = bool(sc.blasting.enabled and blasted)
        v.shift_plan.start_time = _shift_start(first_start, shift_no, sc.shift_hours)
        for e in v.excavators:  # continue along the face where the excavator stopped
            k = min(dig_step[e.id], max(len(e.dig_routes) - 1, 0))
            if e.dig_routes:
                e.route, e.dig_routes = e.dig_routes[k], e.dig_routes[k:]
        res = simulate(v, seed=seed + 7919 * i)
        k = res.kpi
        loaded = res.loads.groupby("blast")["tonnes"].sum() if len(res.loads) else pd.Series(dtype=float)
        for b in sc.blasts:
            left[b.id] = max(0.0, left[b.id] - float(loaded.get(b.id, 0.0)))
        stock = float(k.get("stockpile_end_t", stock) or 0.0)
        for e in v.excavators:
            dig_step[e.id] += int(res.excavator_steps.get(e.id, 0))
        pile_of = {e.id: e.blast for e in sc.excavators}
        idle_empty = []
        for e in k["excavators"]:
            if v.blast(pile_of[e["excavator"]]).tonnage_t <= 1:
                idle_empty.append(f"{e['excavator']} (od rana)")
            elif e.get("blast_done_at_h") is not None and e["blast_done_at_h"] < sc.shift_hours - 0.5:
                idle_empty.append(f"{e['excavator']} (od {e['blast_done_at_h']:.1f} h)")
        rows.append({
            "data": day, "dzien": WEEKDAYS[day.weekday()], "dzien_roboczy": wd, "zmiana": shift_no,
            "start": v.shift_plan.start_time, "deszcz": bool(k["rain"]),
            "wywiezione_t": k["hauled_t"], "koszt_zl": k["cost_total"], "koszt_na_t": k["cost_per_t"],
            "paliwo_l": k["fuel_l"], "kolejki_h": round(k["truck_queue_excavator_h"] + k["truck_queue_crusher_h"], 2),
            "koparki_czekaly_h": round(sum(e["waiting_for_trucks_h"] for e in k["excavators"]), 2),
            "kruszarka_bez_nadawy_h": k["crusher_starved_h"], "awarie_h": k["truck_breakdown_h"],
            "strzal": ", ".join(f"{p} +{t:,.0f} t".replace(",", " ") for p, t in blasted.items()),
            "skonczyl_sie_urobek": ", ".join(idle_empty),
            **{f"zostalo_{b.id}_t": round(left[b.id]) for b in sc.blasts},
            **{f"z_{b.id}_t": round(float(loaded.get(b.id, 0.0))) for b in sc.blasts},
            "halda_t": round(stock),
        })
        if progress:
            progress(i + 1, len(shifts))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, plan: PeriodPlan) -> dict:
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
        "kruszarka_bez_nadawy_h": round(float(df["kruszarka_bez_nadawy_h"].sum()), 1),
    }
