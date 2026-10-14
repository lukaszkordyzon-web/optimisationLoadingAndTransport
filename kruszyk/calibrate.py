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

SEEDS = (11, 12, 13, 14, 15, 16)  # one shift is noisy: tiny changes reshuffle the queues
MAX_DELAY_MIN = 240.0  # losses per trip the fit may assume  # every fitted value is an average over these shifts (rain, breakdowns, drivers differ)


def model_rows(sc: Scenario, seeds=SEEDS) -> tuple[list[dict], dict]:
    """The measured quantities of simulated shifts, one row per shift (+ KPIs of the last one)."""
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
    return rows, last


def model_shift(sc: Scenario, seeds=SEEDS) -> dict:
    """Average of a few simulated shifts: what the calibration compares with the measurements."""
    rows, last = model_rows(sc, seeds)
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


def calibrate(sc: Scenario, measured: dict, progress=None, breakdowns: bool = False,
              seeds=SEEDS) -> tuple[Scenario, dict, dict]:
    """Returns (calibrated scenario, model before, model after). The scenario runs as today (natural dispatch).
    One typical shift: calibrated without breakdowns (they are drawn separately – reliability – not hidden in the
    delays). The mean of many shifts (`breakdowns=True`) already contains the breakdowns of those shifts."""
    base = copy.deepcopy(sc)
    base.dispatch = "natural"
    base.calibration = Calibration()
    base.reliability.enabled = bool(breakdowns and sc.reliability.enabled)
    m = {k: float(v) for k, v in measured.items() if v}
    if m.get("trucks"):
        base = base.with_truck_count(int(round(m["trucks"])))
    before = model_shift(base, seeds)
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
    cur = model_shift(base, seeds) if m.get("trips") else before
    cal = {}
    if m.get("load_min") and cur["load_min"]:
        cal["load_factor"] = round(m["load_min"] / cur["load_min"], 3)
    if m.get("travel_min") and cur["travel_min"]:
        cal["travel_factor"] = round(m["travel_min"] / cur["travel_min"], 3)
    tick("czasy z pomiarów")
    # 3. the rest of the gap in tonnes: unknown losses per trip (or, if the model is too slow, faster pace)
    target = m.get("tonnes_t")
    if target:
        at0 = model_shift(_with(base, **cal), seeds)
        if at0["tonnes_t"] > target:
            lo, hi = 0.0, MAX_DELAY_MIN
            best = (abs(at0["tonnes_t"] - target), 0.0)
            for _ in range(12):  # tonnes jump by whole trips, so keep the closest point tried, not the last one
                mid = round((lo + hi) / 2, 2)  # the value stored is exactly the value tested
                got = model_shift(_with(base, **cal, delay_min_per_trip=mid), seeds)["tonnes_t"]
                best = min(best, (abs(got - target), mid))
                lo, hi = (mid, hi) if got > target else (lo, mid)
                tick(f"przestoje {mid:.1f} min/kurs")
            cal["delay_min_per_trip"] = best[1]
        elif at0["tonnes_t"] < 0.98 * target and not (m.get("load_min") and m.get("travel_min")):
            lo, hi = 0.4, 1.0  # the model is slower than reality: real driving / loading is quicker
            best = (abs(at0["tonnes_t"] - target), 1.0)
            for _ in range(8):
                mid = round((lo + hi) / 2, 3)
                kw = {"travel_factor": cal.get("travel_factor", mid), "load_factor": cal.get("load_factor", mid)}
                got = model_shift(_with(base, **kw), seeds)["tonnes_t"]
                best = min(best, (abs(got - target), mid))
                lo, hi = (lo, mid) if got < target else (mid, hi)  # shorter times -> more tonnes
                tick(f"tempo ×{mid:.2f}")
            mid = best[1]
            cal.setdefault("travel_factor", mid)
            cal.setdefault("load_factor", mid)
    # 4. fuel and energy: straight ratios on the fitted model
    fitted = _with(base, **cal)
    after = model_shift(fitted, seeds)
    if m.get("fuel_l") and after["fuel_l"]:
        cal["fuel_factor"] = round(m["fuel_l"] / after["fuel_l"], 3)
    if m.get("kwh") and after["kwh"]:
        cal["kwh_factor"] = round(m["kwh"] / after["kwh"], 3)
    fitted = _with(base, **cal)
    fitted.calibration.measured = dict(measured)
    fitted.calibration.date = dt.date.today().isoformat()
    after = model_shift(fitted, seeds)
    tick("gotowe")
    if target and after["tonnes_t"] > 1.03 * target:  # even huge losses don't bring the model down to the measurement
        after["note"] = (f"Model nie schodzi do {target:,.0f} t nawet przy {cal.get('delay_min_per_trip', 0):.0f} min "
                         "przestoju na kurs – dane się nie zgadzają. Sprawdź liczbę wozideł w pracy i liczbę kursów: "
                         "przy tej flocie i tych drogach produkcja byłaby dużo wyższa.").replace(",", " ")
    elif target and after["tonnes_t"] < 0.97 * target:  # the model can't get there: say what holds it back
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


# ------------------------------------------------------------------ several shifts: means and spread
FIELDS = ("tonnes_t", "trips", "trucks", "load_min", "travel_min", "fuel_l", "kwh")
SEEDS_MANY = tuple(range(21, 37))   # the mean of the measured shifts is fitted on 16 simulated shifts
SEEDS_SPREAD = tuple(range(41, 71))  # the spread is compared on 30


def robust_sd(v) -> float:
    """Spread of the typical shifts: interquartile range / 1.349 (= sd for a normal spread). One shift with a long
    crusher breakdown would double the plain standard deviation of 20 shifts; this one it hardly moves."""
    v = np.asarray(v, dtype=float)
    return float((np.quantile(v, 0.75) - np.quantile(v, 0.25)) / 1.349) if len(v) >= 4 else 0.0


def shift_stats(df) -> dict:
    """Mean, standard deviation and number of measured shifts of every quantity (empty cells left out)."""
    out = {}
    for k in FIELDS:
        if k not in df:
            continue
        v = np.asarray([float(x) for x in df[k] if x is not None and x == x and float(x) > 0], dtype=float)
        if len(v):
            out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "rsd": robust_sd(v),
                      "n": int(len(v)),
                      "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9))}
    return out


def spread_check(sc: Scenario, seeds=SEEDS_SPREAD) -> dict:
    """The model's shifts as the measured ones are summarised: mean, sd, P10, P90 of every quantity."""
    rows, _ = model_rows(sc, seeds)
    out = {}
    for k in rows[0]:
        v = np.asarray([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "rsd": robust_sd(v),
                  "p10": float(np.quantile(v, 0.1)),
                  "p90": float(np.quantile(v, 0.9)), "values": v.tolist()}
    return out


def calibrate_shifts(sc: Scenario, df, progress=None) -> tuple[Scenario, dict, dict, dict]:
    """Calibration on a table of real shifts (a week, a month): the model is fitted to their MEAN – with the
    breakdowns, which are part of those shifts – and its shift-to-shift spread of tonnes to their standard
    deviation (a pace factor per shift for what the model has no cause for). Returns (calibrated scenario,
    measured statistics, model check before, model check after)."""
    stats = shift_stats(df)
    if "tonnes_t" not in stats:
        raise ValueError("Wpisz co najmniej tony z kilku zmian")
    means = {k: v["mean"] for k, v in stats.items()}

    def tick(f, text):
        if progress:
            progress(f, text)
    before = spread_check(sc)
    fitted, _, _ = calibrate(sc, means, progress=lambda f, t: tick(0.7 * f, t), breakdowns=True, seeds=SEEDS_MANY)
    target = stats["tonnes_t"]["rsd"]  # spread of the typical shifts (robust to one long breakdown)
    probe = copy.deepcopy(fitted)
    probe.dispatch = "natural"

    def model_sd(x: float) -> float:
        probe.calibration.shift_sd = x
        return robust_sd([r["tonnes_t"] for r in model_rows(probe, SEEDS_SPREAD)[0]])
    best = 0.0
    if stats["tonnes_t"]["n"] >= 8 and target > model_sd(0.0):  # the mine varies more than breakdowns and rain explain
        lo, hi, best_gap = 0.0, 0.5, abs(model_sd(0.0) - target)
        for i in range(7):
            mid = round((lo + hi) / 2, 3)
            got = model_sd(mid)
            if abs(got - target) < best_gap:
                best, best_gap = mid, abs(got - target)
            lo, hi = (mid, hi) if got < target else (lo, mid)
            tick(0.7 + 0.04 * (i + 1), f"rozrzut między zmianami ×{mid:.2f}")
    fitted.calibration.shift_sd = best
    fitted.calibration.measured = {k: round(v, 3) for k, v in means.items()}
    fitted.calibration.measured_sd = {k: round(v["rsd"], 3) for k, v in stats.items()}
    fitted.calibration.n_shifts = int(stats["tonnes_t"]["n"])
    check = copy.deepcopy(fitted)
    check.dispatch = "natural"
    after = spread_check(check)
    tick(1.0, "gotowe")
    return fitted, stats, before, after
