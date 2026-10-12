"""Deterministic (expected) times: truck travel on grades, loading, crushing.

Used both by the simulation (as the mean around which noise is drawn) and by the
analytical calculator / dispatcher (as the estimate).
"""
from __future__ import annotations

import math

from .config import Blast, Crusher, Excavator, Route, Scenario, Segment, TruckFleet

G = 9.81


def segment_speed_kmh(seg: Segment, truck: TruckFleet, loaded: bool, rr_pct: float,
                      forward: bool | None = None, load_t: float | None = None) -> float:
    """Expected speed on a segment.

    Uphill: power-limited speed v = P*eta / (m*g*(grade+rr)).
    Downhill: limited by retarder/safety, the steeper the slower.
    `forward` = driving in the segment's direction (default: loaded trucks drive forward, i.e. towards
    the crusher); `load_t` = actual load (default: full when loaded, else empty).
    """
    forward = loaded if forward is None else forward
    load = (truck.payload_t if loaded else 0.0) if load_t is None else load_t
    loaded = load > 0
    mass_kg = (truck.empty_mass_t + load) * 1000
    grade = seg.grade_pct if forward else -seg.grade_pct
    resistance = (grade + rr_pct) / 100.0
    cap = min(seg.speed_limit_kmh, truck.max_speed_kmh)
    if resistance > 0:
        v_ms = truck.power_kw * 1000 * truck.drive_efficiency / (mass_kg * G * resistance)
        v = v_ms * 3.6
    else:
        v = cap
    if grade < -2:  # descending: retarder limit, heavier = slower
        down_cap = truck.max_speed_kmh * max(0.35, 1 - 0.045 * (abs(grade) - 2) * (1.3 if loaded else 1.0))
        v = min(v, down_cap)
    return max(5.0, min(v, cap))


def segment_time_s(seg: Segment, truck: TruckFleet, loaded: bool, rr_pct: float,
                   forward: bool | None = None, load_t: float | None = None) -> float:
    return seg.length_m / (segment_speed_kmh(seg, truck, loaded, rr_pct, forward, load_t) / 3.6)


def route_time_s(route: Route, truck: TruckFleet, loaded: bool, rr_pct: float) -> float:
    return sum(segment_time_s(s, truck, loaded, rr_pct) for s in route.segments)


def passes_per_truck(ex: Excavator, truck: TruckFleet, loose_density: float) -> int:
    bucket_t = ex.bucket_m3 * ex.fill_factor * loose_density
    return max(1, math.ceil(truck.payload_t / bucket_t - 0.15))  # tolerate ~15 % underload of last pass


def dig_factor(blast: Blast) -> float:
    """Poor fragmentation -> longer swing cycles (harder digging, sorting blocks)."""
    return 1.0 + 0.5 * blast.fragmentation


def load_time_s(ex: Excavator, truck: TruckFleet, blast: Blast, loose_density: float) -> float:
    density = blast.loose_density_t_m3 or loose_density
    return passes_per_truck(ex, truck, density) * ex.swing_cycle_s * dig_factor(blast)


def crusher_rate_tph(crusher: Crusher, fragmentation: float) -> float:
    return crusher.capacity_tph * (1 - crusher.fragmentation_penalty * fragmentation)


def expected_cycle(sc: Scenario, ex: Excavator, truck: TruckFleet) -> dict[str, float]:
    """Expected truck cycle components for one excavator, without queueing."""
    route = sc.routes[ex.route]
    blast = sc.blast(ex.blast)
    parts = {
        "spot_s": ex.spot_time_s,
        "load_s": load_time_s(ex, truck, blast, sc.loose_density_t_m3),
        "haul_s": route_time_s(route, truck, True, sc.rolling_resistance_pct),
        "dump_s": sc.crusher.dump_time_s,
        "return_s": route_time_s(route, truck, False, sc.rolling_resistance_pct),
    }
    parts["cycle_s"] = sum(parts.values())
    return parts


def crusher_kwh_per_t(crusher: Crusher, fragmentation: float) -> float:
    return crusher.kwh_per_t * (1 + crusher.kwh_fragmentation_extra * fragmentation)


def drill_blast_cost_per_t(costs, fragmentation: float) -> float:
    """Finer fragmentation = higher powder factor = more expensive blasting."""
    return costs.drill_blast_per_t + costs.drill_blast_fine_extra_per_t * (1 - fragmentation)
