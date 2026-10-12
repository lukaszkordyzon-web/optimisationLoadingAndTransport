"""Scenario configuration: everything that describes the mine is set here (YAML)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Segment:
    name: str
    length_m: float
    grade_pct: float  # positive = uphill in the loaded direction (front -> crusher)
    speed_limit_kmh: float = 40.0


@dataclass
class Route:
    id: str
    segments: list[Segment]

    @property
    def length_m(self) -> float:
        return sum(s.length_m for s in self.segments)


@dataclass
class Blast:
    id: str
    tonnage_t: float
    # 0 = excellent fragmentation, 1 = poor (big blocks): slows digging and crushing
    fragmentation: float = 0.3


@dataclass
class Excavator:
    id: str
    bucket_m3: float
    swing_cycle_s: float
    blast: str
    route: str
    fill_factor: float = 0.9
    spot_time_s: float = 40.0


@dataclass
class TruckFleet:
    model: str
    count: int
    payload_t: float
    empty_mass_t: float
    power_kw: float
    max_speed_kmh: float = 45.0
    drive_efficiency: float = 0.75


@dataclass
class Crusher:
    capacity_tph: float
    hopper_t: float
    dump_time_s: float = 60.0
    # how much poor fragmentation reduces throughput (0.4 -> -40 % at fragmentation=1)
    fragmentation_penalty: float = 0.4


@dataclass
class Variability:
    driver_sd: float = 0.07       # persistent per-driver speed factor
    trip_sd: float = 0.08         # per-trip noise on travel time
    load_sd: float = 0.10         # per-load noise on loading time
    rain_probability: float = 0.2  # chance the shift is rainy (slower haul roads)
    rain_slowdown: float = 0.12
    truck_mtbf_h: float = 40.0    # mean time between truck breakdowns
    truck_mttr_min: float = 35.0  # mean repair time


@dataclass
class Costs:
    truck_per_h: float = 450.0      # PLN/h incl. operator and fuel
    excavator_per_h: float = 900.0
    crusher_per_h: float = 1500.0


@dataclass
class Scenario:
    name: str
    shift_hours: float
    blasts: list[Blast]
    excavators: list[Excavator]
    trucks: list[TruckFleet]
    routes: dict[str, Route]
    crusher: Crusher
    loose_density_t_m3: float = 1.75
    rolling_resistance_pct: float = 3.0
    dispatch: str = "dynamic"  # "dynamic" (system assigns) or "fixed" (truck bound to excavator)
    seed: int = 42
    variability: Variability = field(default_factory=Variability)
    costs: Costs = field(default_factory=Costs)

    @property
    def truck_count(self) -> int:
        return sum(f.count for f in self.trucks)

    def blast(self, blast_id: str) -> Blast:
        return next(b for b in self.blasts if b.id == blast_id)

    def validate(self) -> None:
        blast_ids = {b.id for b in self.blasts}
        for ex in self.excavators:
            if ex.blast not in blast_ids:
                raise ValueError(f"Koparka {ex.id}: nieznany odstrzał '{ex.blast}'")
            if ex.route not in self.routes:
                raise ValueError(f"Koparka {ex.id}: nieznana trasa '{ex.route}'")
        if not self.excavators:
            raise ValueError("Scenariusz musi mieć co najmniej jedną koparkę")
        if self.truck_count < 1:
            raise ValueError("Scenariusz musi mieć co najmniej jedno wozidło")
        if self.dispatch not in ("dynamic", "fixed"):
            raise ValueError("dispatch musi być 'dynamic' albo 'fixed'")

    def with_truck_count(self, n: int) -> "Scenario":
        """Copy of the scenario with the fleet scaled to n trucks (first fleet type absorbs the change)."""
        import copy

        sc = copy.deepcopy(self)
        others = sum(f.count for f in sc.trucks[1:])
        sc.trucks[0].count = max(0, n - others)
        if sc.truck_count != n:  # n smaller than the other fleets: trim from the end
            for f in reversed(sc.trucks):
                excess = sc.truck_count - n
                if excess <= 0:
                    break
                f.count -= min(f.count, excess)
        return sc


def _build(cls, data: dict):
    return cls(**data)


def load_scenario(path: str | Path) -> Scenario:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return scenario_from_dict(raw)


def scenario_from_dict(raw: dict) -> Scenario:
    routes = {
        rid: Route(rid, [_build(Segment, s) for s in r["segments"]])
        for rid, r in raw["routes"].items()
    }
    sc = Scenario(
        name=raw.get("name", "scenariusz"),
        shift_hours=raw["shift_hours"],
        blasts=[_build(Blast, b) for b in raw["blasts"]],
        excavators=[_build(Excavator, e) for e in raw["excavators"]],
        trucks=[_build(TruckFleet, t) for t in raw["trucks"]],
        routes=routes,
        crusher=_build(Crusher, raw["crusher"]),
        loose_density_t_m3=raw.get("loose_density_t_m3", 1.75),
        rolling_resistance_pct=raw.get("rolling_resistance_pct", 3.0),
        dispatch=raw.get("dispatch", "dynamic"),
        seed=raw.get("seed", 42),
        variability=_build(Variability, raw.get("variability", {})),
        costs=_build(Costs, raw.get("costs", {})),
    )
    sc.validate()
    return sc
