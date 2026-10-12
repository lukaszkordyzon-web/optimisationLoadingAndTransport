"""Analytical (no simulation) calculator: match factor and fleet sizing."""
from __future__ import annotations

import math

from .config import Scenario, TruckFleet
from .physics import crusher_rate_tph, expected_cycle


def representative_truck(sc: Scenario) -> TruckFleet:
    return max(sc.trucks, key=lambda f: f.count)


def match_factor_report(sc: Scenario) -> dict:
    """Per-excavator cycle breakdown, trucks needed and overall match factor.

    MF = trucks * loading_time / (excavators * truck_cycle). MF<1: excavators wait,
    MF>1: trucks queue. With several excavators we compare the fleet with the sum of
    per-excavator demand.
    """
    truck = representative_truck(sc)
    rows = []
    trucks_needed_total = 0.0
    excavator_tph_total = 0.0
    for ex in sc.excavators:
        c = expected_cycle(sc, ex, truck)
        service = c["spot_s"] + c["load_s"]
        needed = c["cycle_s"] / service
        trucks_needed_total += needed
        tph = truck.payload_t * 3600 / service
        excavator_tph_total += tph
        rows.append({
            "excavator": ex.id,
            "route": ex.route,
            "blast": ex.blast,
            **{k: round(v, 1) for k, v in c.items()},
            "trucks_needed": round(needed, 2),
            "excavator_max_tph": round(tph, 1),
        })
    n = sc.truck_count
    frag = sum(sc.blast(e.blast).fragmentation for e in sc.excavators) / len(sc.excavators)
    crusher_tph = crusher_rate_tph(sc.crusher, frag)
    mf = n / trucks_needed_total
    haul_tph = min(excavator_tph_total * min(mf, 1.0), crusher_tph)
    return {
        "truck_model": truck.model,
        "excavators": rows,
        "trucks": n,
        "trucks_needed": round(trucks_needed_total, 2),
        "match_factor": round(mf, 3),
        "excavator_capacity_tph": round(excavator_tph_total, 1),
        "crusher_capacity_tph": round(crusher_tph, 1),
        "expected_tph": round(haul_tph, 1),
        "bottleneck": _bottleneck(mf, excavator_tph_total, crusher_tph),
        # MF ~0.95 is usually cost-optimal; if the crusher can't take what the excavators dig,
        # more trucks only lengthen the queue at the hopper
        "recommended_trucks": math.ceil(
            trucks_needed_total * min(1.0, crusher_tph / excavator_tph_total) - 0.25),
    }


def _bottleneck(mf: float, ex_tph: float, crusher_tph: float) -> str:
    if crusher_tph < ex_tph * min(mf, 1.0):
        return "kruszarka"
    if mf < 0.95:
        return "wozidła (koparki czekają)"
    if mf > 1.05:
        return "koparki (wozidła się kolejkują)"
    return "zbalansowany"
