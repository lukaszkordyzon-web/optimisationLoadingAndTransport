"""Test mine for validating the models: one plant, one layout, parameters changed one at a time.

Layout (from the crusher down):  crusher -- 400 m flat -- ramp 1 (600 m) -- level 1: 250 m to blast B1
                                                                        -- ramp 2 (500 m) -- level 2: 250 m to blast B2
Equipment chosen the way a planning engineer would (see `human_sizing`): a 400 t/h primary crusher (blasted limestone,
P80 ~ 450 mm), one 70 t excavator with a 4.5 m³ rock bucket on each level, 36 t rigid dump trucks – as many as keep
the crusher fed at ~90 % without long queues.
"""
from __future__ import annotations

import copy
import math

from kruszyk.config import scenario_from_dict
from kruszyk.physics import crusher_rate_tph, load_time_s, route_time_s

BASE = {
    "trucks": 3, "payload_t": 36.0, "crusher_tph": 400.0, "hopper_t": 100.0, "bucket_m3": 4.5, "swing_s": 27.0,
    "grade_pct": 8.0, "flat_m": 400.0, "ramp1_m": 600.0, "ramp2_m": 500.0, "bench_m": 250.0,
    "frag": 0.3, "b1_t": 15_000.0, "b2_t": 15_000.0, "rain": 0.2, "breakdowns": True, "delay_min": 0.0,
    "shift_h": 8.0, "excavators": 2, "share_b1": 0.0,
    "single_lane": False,  # ramps one lane wide, with passing bays between equal stretches
    "bays": 1,             # passing bays on each single-lane ramp
}

TRUCK = {"model": "CAT 770G (sztywnoramowe 36 t)", "payload_t": 36.0, "empty_mass_t": 33.6, "power_kw": 382,
         "max_speed_kmh": 60, "drive_efficiency": 0.75, "fuel_lph_loaded": 46, "fuel_lph_empty": 25, "fuel_lph_idle": 8,
         "tank_l": 568.0}


def _seg(name, length, grade, limit):
    return {"name": name, "length_m": float(length), "grade_pct": float(grade), "speed_limit_kmh": float(limit)}


def make_raw(**changes) -> dict:
    p = {**BASE, **changes}
    g = p["grade_pct"]

    def ramp(name, length):
        if not p["single_lane"]:
            return [_seg(name, length, g, 25)]
        # two single-lane stretches with a passing bay between them (uphill = loaded trucks have priority)
        n = int(p["bays"]) + 1
        return [{**_seg(name, length / n, g, 25), "section": f"{name}#{i}", "section_dir": 1} for i in range(n)]

    lvl1 = [_seg("front", 30, 0, 15), _seg("poziom 1", p["bench_m"], 0, 30), *ramp("rampa 1", p["ramp1_m"]),
            _seg("droga do zakładu", p["flat_m"], 0, 40), _seg("plac_kruszarki", 30, 0, 15)]
    lvl2 = [_seg("front", 30, 0, 15), _seg("poziom 2", p["bench_m"], 0, 30), *ramp("rampa 2", p["ramp2_m"]),
            _seg("poziom 1 (przejazd)", 150, 0, 30), *lvl1[2:]]
    scale = p["payload_t"] / TRUCK["payload_t"]
    truck = {**TRUCK, "payload_t": p["payload_t"], "empty_mass_t": round(TRUCK["empty_mass_t"] * scale, 1),
             "power_kw": round(TRUCK["power_kw"] * scale), "count": int(p["trucks"])}
    excavators = [{"id": "K-1", "bucket_m3": p["bucket_m3"], "swing_cycle_s": p["swing_s"], "fill_factor": 0.85,
                   "blast": "B1", "route": "T-1"},
                  {"id": "K-2", "bucket_m3": p["bucket_m3"], "swing_cycle_s": p["swing_s"], "fill_factor": 0.85,
                   "blast": "B2", "route": "T-2"}][: int(p["excavators"])]
    return {
        "name": "Kopalnia testowa (walidacja)", "shift_hours": p["shift_h"], "seed": 7, "dispatch": "natural",
        "loose_density_t_m3": 1.7, "rolling_resistance_pct": 3.0,
        "blasts": [{"id": "B1", "tonnage_t": p["b1_t"], "fragmentation": p["frag"], "material": "wapień",
                    **({"target_share": p["share_b1"]} if p["share_b1"] else {})},
                   {"id": "B2", "tonnage_t": p["b2_t"], "fragmentation": p["frag"], "material": "wapień",
                    **({"target_share": 100 - p["share_b1"]} if p["share_b1"] else {})}],
        "excavators": excavators, "trucks": [truck],
        "routes": {"T-1": {"segments": lvl1}, "T-2": {"segments": lvl2}},
        "crusher": {"id": "C-1", "capacity_tph": p["crusher_tph"], "hopper_t": p["hopper_t"], "dump_time_s": 60.0,
                    "max_feed_mm": 800.0, "kwh_per_t": 2.0, "fragmentation_penalty": 0.4, "product": "grys 8–16"},
        "variability": {"rain_probability": p["rain"]},
        "shift_plan": {"start_time": "06:00", "startup_min": 15.0, "end_early_min": 10.0,
                       "breaks": [{"start": "10:00", "min": 20.0}]},
        "costs": {"truck_per_h": 260.0, "excavator_per_h": 520.0, "crusher_per_h": 1100.0, "fuel_per_l": 6.3,
                  "energy_per_kwh": 0.85, "drill_blast_per_t": 3.6, "drill_blast_fine_extra_per_t": 2.0,
                  "excavator_idle_share": 0.35},
        "reliability": {"enabled": bool(p["breakdowns"]), "ref_shifts_per_day": 2.0, "ref_days_per_month": 22.0},
        "calibration": {"delay_min_per_trip": p["delay_min"]},
    }


def make(**changes):
    return scenario_from_dict(make_raw(**changes))


def human_sizing(**changes) -> dict:
    """What a planning engineer computes by hand: cycle per level, t/h per truck, trucks to keep the crusher fed."""
    sc = make(**changes)
    f = sc.trucks[0]
    out = {"crusher_eff_tph": crusher_rate_tph(sc.crusher, sc.blasts[0].fragmentation)}
    for ex in sc.excavators:
        route = sc.routes[ex.route]
        load = load_time_s(ex, f, sc.blast(ex.blast), sc.loose_density_t_m3) + 45  # + spotting
        cycle = load + route_time_s(route, f, True, 3.0) + route_time_s(route, f, False, 3.0) + 60
        out[ex.id] = {"load_min": load / 60, "cycle_min": cycle / 60, "truck_tph": f.payload_t * 3600 / cycle,
                      "excavator_tph": f.payload_t * 3600 / load}
    per_truck = sum(out[e.id]["truck_tph"] for e in sc.excavators) / len(sc.excavators)
    out["truck_tph_avg"] = per_truck
    # trucks for 90 % of the crusher, ~85 % truck availability in a real shift (breaks, start-up, queues)
    out["trucks_needed"] = math.ceil(0.9 * out["crusher_eff_tph"] / (0.85 * per_truck))
    return out


def variant(**changes):
    return copy.deepcopy(make(**changes))
