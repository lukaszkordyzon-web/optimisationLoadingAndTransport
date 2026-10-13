"""Calibration: tune the synthetic model to what a real shift measured.

Measured on site (only the tonnes are required):
  tonnes_t       – crusher weigh scale / dispatch: tonnes hauled in the shift
  trips          – number of truck loads in the shift            -> real average load per trip
  trucks         – trucks that really worked that shift
  load_min       – average loading time of one truck (stopwatch)  -> load_factor
  travel_min     – average driving per trip, loaded + empty        -> travel_factor
  fuel_l         – fuel of the haul fleet in the shift (refuelling) -> fuel_factor
  kwh            – crusher energy meter in the shift                -> kwh_factor
What remains of the gap in tonnes after the measured times is the time the model knows no cause for (waiting for
orders, cleaning, talks, unrecorded stops): it is fitted as minutes lost per trip.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np

from .config import Calibration, Scenario
from .sim import simulate

SEEDS = (11, 12, 13)  # every fitted value is an average over these shifts (rain, breakdowns, drivers differ)


def model_shift(sc: Scenario, seeds=SEEDS) -> dict:
    """Average of a few simulated shifts: what the calibration compares with the measurements."""
    rows = []
    for s in seeds:
        r = simulate(sc, seed=s)
        k = r.kpi
        st = r.states.groupby("state")["duration_s"].sum() / 60 if len(r.states) else {}
        cycles = max(k["cycles"], 1)
        rows.append({"tonnes_t": k["hauled_t"], "trips": k["cycles"],
                     "load_min": float(st.get("zaladunek", 0.0)) / cycles,
                     "travel_min": float(st.get("jazda_pusty", 0.0) + st.get("jazda_pelny", 0.0)
                                         + st.get("przejazd_miedzy_usypami", 0.0)) / cycles,
                     "fuel_l": k["fuel_l"], "kwh": k["crusher_kwh"],
                     "queue_crusher_h": float(st.get("kolejka_kruszarka", 0.0)) / 60,
                     "queue_excavator_h": float(st.get("kolejka_koparka", 0.0)) / 60})
        last = k
    out = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    busy = max(last["crushers"], key=lambda c: c["utilization"])
    out["limit"] = ""
    if busy["utilization"] >= 0.9:
        out["limit"] = (f"kruszarka {busy['crusher']} pracuje na {100 * busy['utilization']:.0f} % "
                        f"({busy['capacity_tph']:g} t/h)")
    elif max(e["utilization"] for e in last["excavators"]) >= 0.85:
        out["limit"] = "koparki ładują bez przerwy"
    return out


def _with(sc: Scenario, **cal) -> Scenario:
    out = copy.deepcopy(sc)
    for k, v in cal.items():
        setattr(out.calibration, k, v)
    return out


def calibrate(sc: Scenario, measured: dict, progress=None) -> tuple[Scenario, dict, dict]:
    """Returns (calibrated scenario, model before, model after). The scenario runs as today (natural dispatch)."""
    base = copy.deepcopy(sc)
    base.dispatch = "natural"
    base.calibration = Calibration()
    # calibrated on a shift without breakdowns: they are drawn separately (reliability), not hidden in the delays
    base.reliability.enabled = False
    m = {k: float(v) for k, v in measured.items() if v}
    if m.get("trucks"):
        base = base.with_truck_count(int(m["trucks"]))
    before = model_shift(base)
    step = [0]

    def tick(text: str) -> None:
        step[0] += 1
        if progress:
            progress(min(step[0] / 14, 1.0), text)

    tick("dzisiejszy model")
    # 1. real average load per trip (trucks are rarely filled to the nominal payload)
    if m.get("tonnes_t") and m.get("trips") and before["trips"]:
        ratio = (m["tonnes_t"] / m["trips"]) / (before["tonnes_t"] / before["trips"])
        for f in base.trucks:
            f.payload_t = round(f.payload_t * ratio, 1)
    # 2. measured times -> factors
    cur = model_shift(base) if m.get("trips") else before
    cal = {}
    if m.get("load_min") and cur["load_min"]:
        cal["load_factor"] = round(m["load_min"] / cur["load_min"], 3)
    if m.get("travel_min") and cur["travel_min"]:
        cal["travel_factor"] = round(m["travel_min"] / cur["travel_min"], 3)
    tick("czasy z pomiarów")
    # 3. the rest of the gap in tonnes: unknown losses per trip (or, if the model is too slow, faster pace)
    target = m.get("tonnes_t")
    if target:
        at0 = model_shift(_with(base, **cal))
        if at0["tonnes_t"] > target:
            lo, hi = 0.0, 60.0
            best = (abs(at0["tonnes_t"] - target), 0.0)
            for _ in range(10):  # tonnes jump by whole trips, so keep the closest point tried, not the last one
                mid = (lo + hi) / 2
                got = model_shift(_with(base, **cal, delay_min_per_trip=mid))["tonnes_t"]
                best = min(best, (abs(got - target), mid))
                lo, hi = (mid, hi) if got > target else (lo, mid)
                tick(f"przestoje {mid:.1f} min/kurs")
            cal["delay_min_per_trip"] = round(best[1], 2)
        elif at0["tonnes_t"] < 0.98 * target and not (m.get("load_min") and m.get("travel_min")):
            lo, hi = 0.4, 1.0  # the model is slower than reality: real driving / loading is quicker
            best = (abs(at0["tonnes_t"] - target), 1.0)
            for _ in range(8):
                mid = (lo + hi) / 2
                kw = {"travel_factor": cal.get("travel_factor", mid), "load_factor": cal.get("load_factor", mid)}
                got = model_shift(_with(base, **kw))["tonnes_t"]
                best = min(best, (abs(got - target), mid))
                lo, hi = (lo, mid) if got < target else (mid, hi)  # shorter times -> more tonnes
                tick(f"tempo ×{mid:.2f}")
            mid = round(best[1], 3)
            cal.setdefault("travel_factor", mid)
            cal.setdefault("load_factor", mid)
    # 4. fuel and energy: straight ratios on the fitted model
    fitted = _with(base, **cal)
    after = model_shift(fitted)
    if m.get("fuel_l") and after["fuel_l"]:
        cal["fuel_factor"] = round(m["fuel_l"] / after["fuel_l"], 3)
    if m.get("kwh") and after["kwh"]:
        cal["kwh_factor"] = round(m["kwh"] / after["kwh"], 3)
    fitted = _with(base, **cal)
    fitted.calibration.measured = dict(measured)
    fitted.calibration.date = dt.date.today().isoformat()
    after = model_shift(fitted)
    tick("gotowe")
    if target and after["tonnes_t"] < 0.97 * target:  # the model can't get there: say what holds it back
        after["note"] = (f"Model nie dochodzi do {target:,.0f} t – ogranicza go: {after['limit'] or 'flota'}. Sprawdź "
                         "wpisaną wydajność (kruszarki, łyżki koparek, liczba wozideł) – rzeczywista musi być wyższa."
                         ).replace(",", " ")
    fitted.dispatch = sc.dispatch
    fitted.reliability = copy.deepcopy(sc.reliability)
    return fitted, before, after


def apply_to_raw(raw: dict, fitted: Scenario) -> None:
    """Write the calibration (and the real loads / fleet) back into the mine description."""
    from dataclasses import asdict

    raw["calibration"] = asdict(fitted.calibration)
    by_model = {f.model: f for f in fitted.trucks}
    for t in raw.get("trucks", []):
        f = by_model.get(t.get("model"))
        if f is not None:
            t["payload_t"], t["count"] = f.payload_t, f.count
