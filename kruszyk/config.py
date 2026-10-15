"""Scenario configuration: everything that describes the mine is set here (YAML)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Segment:
    name: str
    length_m: float
    grade_pct: float  # positive = uphill in the loaded direction (front -> crusher)
    speed_limit_kmh: float = 40.0
    section: str | None = None  # single-lane stretch between passing bays (one direction at a time)
    section_dir: int = 0        # +1 / -1: direction of this route (forward) relative to the road drawing
    curves: list = field(default_factory=list)  # bends on this stretch: [[radius m, turn deg], …] (from the drawing)


@dataclass
class Route:
    id: str
    segments: list[Segment]
    path: list | None = None  # [[lat, lon], ...] front -> crusher, when drawn on the map

    @property
    def length_m(self) -> float:
        return sum(s.length_m for s in self.segments)


@dataclass
class Blast:
    id: str
    tonnage_t: float
    # 0 = excellent fragmentation, 1 = poor (big blocks): slows digging and crushing
    fragmentation: float = 0.3
    material: str = ""
    loose_density_t_m3: float | None = None  # overrides the scenario density for this pile
    psd: list | None = None  # grading curve [[mm, % passing], ...]; when given, it sets `fragmentation`
    target_share: float | None = None  # desired share of the crusher feed from this pile [%] (flexible trucks)


@dataclass
class Excavator:
    id: str
    bucket_m3: float
    swing_cycle_s: float
    blast: str
    route: str
    fill_factor: float = 0.9
    spot_time_s: float = 40.0  # used when `spotting` is empty (one fixed spotting time)
    # how trucks spot at this excavator: mode (see SPOT_MODES) -> % of trucks, e.g. {tyl_1: 60, bok_rondo: 40}
    spotting: dict = field(default_factory=dict)
    # advance along the face: routes from evenly spaced points of the digging line (start -> end);
    # the excavator moves to the next point every `relocate_every_t` tonnes, standing `relocate_min` meanwhile
    dig_routes: list = field(default_factory=list)
    dig_path: list | None = None
    relocate_every_t: float = 600.0
    relocate_min: float = 4.0
    crusher_routes: dict = field(default_factory=dict)  # other crushers: crusher id -> route id from here
    start_delay_min: float = 0.0  # not loading at the start of the shift (driving to a new pile)
    # where it may work in the period: ids of extraction areas / planned blasts / piles (empty = anywhere)
    areas: list = field(default_factory=list)
    # rule of the shift: when the pile is (almost) taken out, drive to another one and load there
    then_blast: str = ""            # the next pile
    then_route: str = ""            # route from its loading point to the main crusher
    then_crusher_routes: dict = field(default_factory=dict)
    then_move_min: float = 0.0      # driving time between the piles
    then_below_t: float = 0.0       # move when less than this is left (the rest stays for the others)
    quota_t: float | None = None    # tonnes to load on this shift (the month plan); None = as much as there is

    def may_work(self, pile_id: str) -> bool:
        return not self.areas or pile_id in self.areas or pile_id.split("/")[0] in self.areas


@dataclass
class TruckFleet:
    model: str
    count: int
    payload_t: float
    empty_mass_t: float
    power_kw: float
    max_speed_kmh: float = 45.0
    drive_efficiency: float = 0.75
    fuel_lph_loaded: float = 60.0  # fuel burn while driving loaded
    fuel_lph_empty: float = 32.0
    fuel_lph_idle: float = 10.0    # queueing / spotting / loading / dumping with engine running
    tank_l: float = 700.0          # fuel tank; trucks start the shift full (fuelled before the shift)
    # which excavators these trucks may serve (empty = all)
    excavators: list = field(default_factory=list)
    # two-pile trips: {excavator: share of the payload}, loaded in this order, e.g. {K-1: 0.5, K-2: 0.5}
    blend: dict = field(default_factory=dict)
    id: str = ""  # one particular truck (count 1), e.g. "WZ-03"; empty = a group of identical trucks
    # the assignment to `excavators` is binding for every dispatch; False = only today's practice, the system /
    # the learned policy may send the truck elsewhere when that is cheaper
    assign_locked: bool = True


def truck_units(trucks: list) -> list[tuple[str, "TruckFleet"]]:
    """Every truck with its id: named trucks keep their id, groups get WZ-01, WZ-02 … (skipping names in use)."""
    used = {f.id for f in trucks if f.id}
    out, n = [], 0
    for f in trucks:
        for j in range(f.count):
            if f.id:
                tid = f.id if j == 0 else f"{f.id}-{j + 1}"
            else:
                n += 1
                while f"WZ-{n:02d}" in used:
                    n += 1
                tid = f"WZ-{n:02d}"
            out.append((tid, f))
    return out


def expand_trucks(raw_trucks: list[dict]) -> list[dict]:
    """The fleet as one entry per truck (count 1, own id) – so every truck can have its own parameters."""
    fleets = [TruckFleet(**{k: v for k, v in t.items() if k in TruckFleet.__dataclass_fields__}) for t in raw_trucks]
    out, k = [], 0
    for t, f in zip(raw_trucks, fleets):
        for _ in range(int(t.get("count", 0))):
            tid = truck_units(fleets)[k][0]
            out.append({**t, "id": tid, "count": 1})
            k += 1
        if not int(t.get("count", 0)):  # a type with no trucks in work stays as one idle truck
            out.append({**t, "id": t.get("id") or f"{t.get('model', 'wozidło')} (postój)", "count": 0})
    return out


# What a crusher can produce and how much energy it typically takes per tonne (crushing + screening).
# Finer product = more crushing stages = more kWh/t. Indicative values – overwrite them with your meters.
PRODUCTS = {
    "kamień łamany / narzut 63–250": 0.6,
    "kruszywo 0–63 (podbudowa)": 0.9,
    "tłuczeń 31,5–63": 1.0,
    "mieszanka 0–31,5": 1.2,
    "kliniec 16–31,5": 1.4,
    "grys 8–16": 1.8,
    "grys 2–8": 2.2,
    "piasek łamany 0–2": 2.8,
}


@dataclass
class Crusher:
    capacity_tph: float
    hopper_t: float
    dump_time_s: float = 60.0
    # how much poor fragmentation reduces throughput (0.4 -> -40 % at fragmentation=1)
    fragmentation_penalty: float = 0.4
    max_feed_mm: float = 800.0      # bigger blocks = oversize (hammering, slower feeding)
    kwh_per_t: float = 1.1          # specific energy at perfect fragmentation
    kwh_fragmentation_extra: float = 0.6  # +60 % energy at fragmentation=1
    id: str = "C-1"
    accepts: list = field(default_factory=list)  # piles (blast ids) it takes material from; empty = all
    product: str = ""          # what it produces, e.g. "kruszywo 0–31,5"
    target_t: float = 0.0      # production plan per shift [t] (0 = none)

    def takes(self, blast_id: str) -> bool:
        # blasts generated in an extraction area are called "<area>/<n>" – accepting the area accepts them
        return not self.accepts or blast_id in self.accepts or blast_id.split("/")[0] in self.accepts


@dataclass
class Stockpile:
    """Buffer stockpile next to the crusher, fed back to the hopper by a wheel loader."""
    enabled: bool = False
    divert_wait_s: float = 180.0  # dump on the stockpile if the expected wait at the hopper is longer
    dump_time_s: float = 40.0
    loader_tph: float = 450.0
    feed_below_t: float = 60.0    # loader tops up the hopper when its level drops below this
    initial_t: float = 0.0
    loader_per_h: float = 350.0   # PLN/h
    rehandle_cost_per_t: float = 1.5


@dataclass
class Pacing:
    """Speed advice: a loaded truck that would queue at the crusher is told to drive slower."""
    enabled: bool = False
    max_slowdown: float = 0.25    # at most +25 % haul time


@dataclass
class Variability:
    driver_sd: float = 0.07       # persistent per-driver speed factor
    trip_sd: float = 0.08         # per-trip noise on travel time
    load_sd: float = 0.10         # per-load noise on loading time
    rain_probability: float = 0.2  # chance the shift is rainy (slower haul roads)
    rain_slowdown: float = 0.12
    truck_mtbf_h: float = 40.0    # mean time between truck breakdowns
    truck_mttr_min: float = 35.0  # mean repair time
    # practice on site: some drivers go faster than the posted limit
    speeding_share: float = 0.3   # share of drivers who exceed limits
    speeding_pct: float = 10.0    # by how much (%)


@dataclass
class ShiftPlan:
    """A real shift is not 8 h of hauling: start-up, breaks, finishing early."""
    start_time: str = "06:00"
    startup_min: float = 15.0      # taking over machines, daily checks, warm-up
    end_early_min: float = 10.0    # parking, handover before the end of the shift
    breaks: list = field(default_factory=lambda: [{"start": "10:00", "min": 20}])


@dataclass
class Blasting:
    """Blasting in the pit: crew drives to the blast point, the pit is closed for trucks while firing."""
    enabled: bool = False
    crew_arrive: str = "09:00"     # crew at the blast point
    fire_from: str = "11:00"       # firing window – trucks and excavators out of the pit
    fire_to: str = "11:20"
    evacuation_min: float = 10.0   # trucks must leave the pit this long before firing
    crew_leave: str = "11:40"
    retreat_m: float = 200.0       # without a parking marker the excavator backs off this far along its route
    place: str = ""                # where today's blast is: id of a pile / area / planned blast on the map
    # explosives: the permit limits one blast (in Poland typically 3 000 kg); powder factor of the rock
    max_explosive_kg: float = 0.0  # 0 = no limit
    powder_factor_kg_t: float = 0.145  # limestone ≈ 0.35–0.40 kg/m³ in situ ≈ 0.13–0.15 kg/t

    def max_blast_t(self) -> float:
        """The biggest blast the explosive limit allows [t] (inf = no limit)."""
        if self.max_explosive_kg > 0 and self.powder_factor_kg_t > 0:
            return self.max_explosive_kg / self.powder_factor_kg_t
        return float("inf")


@dataclass
class Inspection:
    """Shift foreman's (sztygar) round through the pit in a light vehicle."""
    enabled: bool = False
    start: str = "09:30"
    stops: list = field(default_factory=list)  # excavator ids, visited in order
    stop_min: float = 10.0


@dataclass
class Refueling:
    truck_reserve_pct: float = 15.0   # a truck goes to the fuel station below this level
    truck_refuel_min: float = 12.0
    tanker_times: list = field(default_factory=list)  # fuel tanker rounds to the excavators, e.g. ["13:00"]
    excavator_refuel_min: float = 20.0
    at_parking: bool = False       # excavators drive to their parking place to be refuelled


@dataclass
class Traffic:
    yield_s: float = 30.0          # time a truck loses giving way to a light vehicle coming up
    excavator_move_kmh: float = 2.5  # tracked excavator driving to its parking place and back
    light_speed_kmh: float = 30.0  # light vehicles (crew, foreman, tanker) – never above the road limit


@dataclass
class Advisor:
    """Real-time recommendations: speed for each truck (arrive when the excavator is free, not earlier),
    and for the excavator operator (fill the bucket before the truck spots)."""
    enabled: bool = False
    max_slowdown: float = 0.3      # at most this much slower than normal (never faster than the limit)
    arrival_margin_s: float = 0.0  # aim to arrive this long BEFORE the excavator is free (travel times vary)
    crusher_pacing: float = 1.0    # how strongly to slow loaded trucks that would queue at the crusher (0 = not)


@dataclass
class Costs:
    truck_per_h: float = 300.0      # PLN/h incl. operator, without fuel
    excavator_per_h: float = 900.0
    crusher_per_h: float = 1200.0   # without energy
    fuel_per_l: float = 6.5
    energy_per_kwh: float = 0.9
    # drilling & blasting: better fragmentation (lower value) needs more explosive
    drill_blast_per_t: float = 3.0
    drill_blast_fine_extra_per_t: float = 2.5  # extra cost at fragmentation=0
    excavator_idle_share: float = 0.35  # a parked excavator still costs this share of its hourly rate (operator)
    # part of the truck rate that goes with engine hours (service, tyres, wear); a parked truck saves it
    truck_variable_share: float = 0.4
    blend_penalty_per_t_pp: float = 0.05  # zł per tonne for each percentage point the feed misses its target mix


@dataclass
class ExcavatorPolicy:
    """Excavator management learned by the agent (used with the RL dispatch). Defaults = today:
    all excavators work and every truck is filled to the payload."""
    min_trucks_per_excavator: float = 0.0  # fewer flexible trucks per excavator than this -> park the weakest one
    topup_min_fill: float = 1.0            # send a truck without its last bucket if already this full…
    topup_min_queue: float = 1.5           # …and at least this many trucks are waiting
    review_min: float = 30.0               # how often the number of working excavators is reviewed
    # trucks: park (engine off) instead of queueing at the crusher. Off when < 0; otherwise a truck parks when
    # it would queue at the crusher longer than this many minutes after its next cycle
    truck_park_queue_min: float = -1.0
    truck_min_working_share: float = 0.5   # never park more than the rest of this share of the fleet
    truck_park_check_min: float = 3.0      # a parked truck looks at the queue again every few minutes


# Ways of spotting a truck at the excavator. in_s = manoeuvre to the loading position, out_s = leaving it,
# extra_m = extra metres driven on the floor (turning loop), double = loading from both sides (the next truck
# spots on the other side while this one is loaded), out_blocks = the next truck can't spot until this one left.
SPOT_MODES = {
    "tyl_1": {"short": "tyłem, 1 strona", "label": "tyłem, z jednej strony", "in_s": 45.0, "out_s": 10.0, "extra_m": 0.0,
              "double": False, "out_blocks": True},
    "tyl_2": {"short": "tyłem, 2 strony", "label": "tyłem, z dwóch stron (dwustronnie)", "in_s": 45.0, "out_s": 10.0, "extra_m": 0.0,
              "double": True, "out_blocks": False},
    "bok_rondo": {"short": "bokiem + rondko", "label": "bokiem, wyjazd prosto z objazdem (rondko)", "in_s": 15.0, "out_s": 10.0, "extra_m": 80.0,
                  "double": False, "out_blocks": False},
    "bok_cofanie": {"short": "bokiem, wyjazd tyłem", "label": "bokiem, wyjazd cofaniem", "in_s": 15.0, "out_s": 40.0, "extra_m": 0.0,
                    "double": False, "out_blocks": True},
}


@dataclass
class Spotting:
    """Manoeuvres at the loading face. `times` overrides SPOT_MODES per mode, e.g. {tyl_1: {in_s: 60}}."""
    floor_speed_kmh: float = 10.0  # driving on the blast floor / loading area
    times: dict = field(default_factory=dict)

    def mode(self, key: str) -> dict:
        return {**SPOT_MODES[key], **(self.times.get(key) or {})}


DISPATCH_MODES = ("natural", "fixed", "dynamic", "rl", "signals")


# breakdowns when nothing is known about a machine: failures per month and their total hours (assumptions)
RELIABILITY_DEFAULTS = {"truck": (4.0, 6.0), "excavator": (3.0, 8.0), "crusher": (4.0, 12.0), "drill": (3.0, 8.0)}
RELIABILITY_KINDS = {"truck": "wozidło (każde)", "excavator": "koparka", "crusher": "kruszarka", "drill": "wiertnica (każda)"}
REPAIR_SIGMA = 1.0  # spread of repair times (lognormal): most are short, a few last longer than a shift


@dataclass
class Reliability:
    """Breakdowns per machine: failures per month and their total hours – averages of the last 3–6 months,
    only stops longer than ~15 min, planned maintenance excluded. The month is the reference period below
    (shifts per day × working days): a machine that works more fails more often.
    Keys: machine id (excavators, crushers, trucks – e.g. "WZ-03"), "truck:<model>" (every truck of that type
    without its own entry), "drill" (per rig)."""
    enabled: bool = True
    ref_shifts_per_day: float = 2.0
    ref_days_per_month: float = 22.0
    machines: dict = field(default_factory=dict)  # key -> {"count": failures / month, "hours": total h / month}

    def values(self, key: str, kind: str) -> tuple[float, float, bool]:
        """(failures / month, hours / month, is it an assumption)"""
        m = self.machines.get(key)
        if m and m.get("count") is not None and m.get("hours") is not None:
            return float(m["count"]), float(m["hours"]), False
        count, hours = RELIABILITY_DEFAULTS[kind]
        return count, hours, True

    def rates(self, key: str, kind: str, shift_hours: float) -> tuple[float, float]:
        """(mean operating hours between failures, mean repair hours); (inf, 0) = never fails."""
        count, hours, _ = self.values(key, kind)
        if not self.enabled or count <= 0 or hours <= 0:
            return math.inf, 0.0
        return self.ref_shifts_per_day * shift_hours * self.ref_days_per_month / count, hours / count


@dataclass
class Calibration:
    """Model tuned to a real shift (see calibrate.py). Factors = real / model; 1.0 = assumptions only."""
    travel_factor: float = 1.0      # driving times
    load_factor: float = 1.0        # loading times
    delay_min_per_trip: float = 0.0  # losses the model doesn't know (waiting for orders, cleaning, talks…) per trip
    fuel_factor: float = 1.0
    kwh_factor: float = 1.0
    measured: dict = field(default_factory=dict)  # what was measured (one shift, or the mean of several)
    date: str = ""                  # when it was calibrated
    # several shifts: how much they differ (standard deviations) and how many there were
    measured_sd: dict = field(default_factory=dict)
    n_shifts: int = 0
    # shift-to-shift variation the model has no cause for (crew, material, organisation of the day): every shift
    # gets one random pace factor for driving and loading with this spread (0 = none)
    shift_sd: float = 0.0


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
    # how trucks pick an excavator after dumping:
    #   natural – today's practice: back to "own" excavator, sometimes elsewhere, no information
    #   fixed   – always the own excavator
    #   dynamic – system heuristic: where the truck will be loaded soonest
    #   rl      – policy learned by reinforcement learning (rl_weights)
    dispatch: str = "dynamic"
    natural_switch_prob: float = 0.0  # drivers keep to the excavator the foreman gave them (raise if they switch on their own)
    rl_weights: list | None = None
    seed: int = 42
    variability: Variability = field(default_factory=Variability)
    costs: Costs = field(default_factory=Costs)
    stockpile: Stockpile = field(default_factory=Stockpile)
    shift_plan: ShiftPlan = field(default_factory=ShiftPlan)
    blasting: Blasting = field(default_factory=Blasting)
    inspection: Inspection = field(default_factory=Inspection)
    refueling: Refueling = field(default_factory=Refueling)
    traffic: Traffic = field(default_factory=Traffic)
    advisor: Advisor = field(default_factory=Advisor)
    excavator_policy: ExcavatorPolicy = field(default_factory=ExcavatorPolicy)
    pacing: Pacing = field(default_factory=Pacing)
    calibration: Calibration = field(default_factory=Calibration)
    reliability: Reliability = field(default_factory=Reliability)
    # repairs still going on at the start of the shift: machine id / truck id -> seconds (set by the period run)
    carry_down: dict = field(default_factory=dict)
    spotting: Spotting = field(default_factory=Spotting)
    site: dict | None = None  # map geometry (see site.py)
    links: dict = field(default_factory=dict)  # "K-1>K-2" -> Route between two excavators
    # future blasts drawn on the map (used by the week / month run): [{id, tonnage_t, material, psd,
    # fragmentation, loose_density_t_m3, blast_day, route}] – `route` = from its loading point to the crusher
    planned_blasts: list = field(default_factory=list)
    crushers_extra: list = field(default_factory=list)  # second, third … crusher [Crusher]
    # extraction areas: the week / month run fires blasts in them automatically (see period.py)
    areas: list = field(default_factory=list)
    # goal of the shift: {"kind": "feed" | "product", "t": goal in its unit, "item": product, "feed_t": feed it
    # needs through the plant, "stop": haulage stops once it is reached}; empty = as much as possible
    shift_goal: dict = field(default_factory=dict)

    @property
    def goal_t(self) -> float:
        return float((self.shift_goal or {}).get("feed_t") or 0.0)

    @property
    def crushers(self) -> list:
        return [self.crusher] + list(self.crushers_extra)

    @property
    def truck_count(self) -> int:
        return sum(f.count for f in self.trucks)

    def excavator(self, ex_id: str) -> "Excavator":
        return next(e for e in self.excavators if e.id == ex_id)

    def blast(self, blast_id: str) -> Blast:
        return next(b for b in self.blasts if b.id == blast_id)

    def validate(self) -> None:
        blast_ids = {b.id for b in self.blasts}
        for ex in self.excavators:
            if ex.blast not in blast_ids:
                raise ValueError(f"Koparka {ex.id}: nieznany odstrzał '{ex.blast}'")
            if ex.route not in self.routes:
                raise ValueError(f"Koparka {ex.id}: nieznana trasa '{ex.route}'")
            missing = [r for r in ex.dig_routes if r not in self.routes]
            missing += [p["route"] for p in self.planned_blasts if p.get("route") not in self.routes]
            if missing:
                raise ValueError(f"Koparka {ex.id}: brak tras postępu {', '.join(missing)} – zbuduj trasy z mapy")
            bad = [m for m in ex.spotting if m not in SPOT_MODES]
            if bad:
                raise ValueError(f"Koparka {ex.id}: nieznany sposób podstawiania {', '.join(bad)}")
            if ex.spotting and sum(float(v or 0) for v in ex.spotting.values()) <= 0:
                raise ValueError(f"Koparka {ex.id}: udziały sposobów podstawiania muszą być > 0")
        if not self.excavators:
            raise ValueError("Scenariusz musi mieć co najmniej jedną koparkę")
        if self.truck_count < 1:
            raise ValueError("Scenariusz musi mieć co najmniej jedno wozidło")
        if self.dispatch not in DISPATCH_MODES:
            raise ValueError(f"dispatch musi być jednym z: {', '.join(DISPATCH_MODES)}")
        if self.dispatch == "rl" and not self.rl_weights:
            raise ValueError("Przydział 'rl' wymaga wytrenowanej polityki (rl_weights)")
        ex_ids = {e.id for e in self.excavators}
        for f in self.trucks:
            bad = [x for x in list(f.excavators) + list(f.blend) if x not in ex_ids]
            if bad:
                raise ValueError(f"Wozidła {f.model}: nieznane koparki {', '.join(bad)}")
            if f.blend:
                if sum(f.blend.values()) <= 0:
                    raise ValueError(f"Wozidła {f.model}: udziały mieszania muszą być > 0")
                stops = list(f.blend)
                for a, b in zip(stops, stops[1:]):
                    if f"{a}>{b}" not in self.links:
                        raise ValueError(f"Wozidła {f.model}: brak połączenia {a} → {b} "
                                         "(zbuduj trasy z mapy, żeby wyznaczyć przejazd między koparkami)")
        ids = [c.id for c in self.crushers]
        if len(set(ids)) != len(ids):
            raise ValueError("Kruszarki muszą mieć różne oznaczenia")
        for e in self.excavators:
            if not isinstance(e.crusher_routes, dict):
                raise ValueError(f"Koparka {e.id}: trasy do kruszarek uszkodzone – zbuduj trasy z mapy ponownie")
            bad = [r for r in e.crusher_routes.values() if r not in self.routes]
            if bad:
                raise ValueError(f"Koparka {e.id}: nieznane trasy do kruszarek {', '.join(map(str, bad))}")
        for c in self.crushers_extra:
            missing = [e.id for e in self.excavators if c.id not in e.crusher_routes and c.takes(e.blast)]
            if missing:
                raise ValueError(f"Kruszarka {c.id}: brak tras od koparek {', '.join(missing)} – zbuduj trasy z mapy")
        for e in self.excavators:
            if not any(c.takes(e.blast) for c in self.crushers):
                raise ValueError(f"Żadna kruszarka nie przyjmuje urobku z {e.blast} (koparka {e.id}) – "
                                 "ustaw, z których usypów przyjmują kruszarki")
        for f in self.trucks:
            if f.count and f.blend and not any(all(c.takes(self.excavator(x).blast) for x in f.blend) for c in self.crushers):
                raise ValueError(f"Wozidła {f.model}: żadna kruszarka nie przyjmuje mieszanki z {', '.join(f.blend)}")
        for r in self.routes.values():
            if not r.segments:
                raise ValueError(f"Trasa {r.id} nie ma odcinków")

    def with_truck_count(self, n: int) -> "Scenario":
        """Copy of the scenario with the fleet scaled to n trucks (first fleet type absorbs the change)."""
        import copy

        sc = copy.deepcopy(self)
        if sc.trucks and all(f.id and f.count <= 1 for f in sc.trucks):  # one entry per truck
            working = [f for f in sc.trucks if f.count]
            while len(working) > n:  # take the last ones out of work
                working.pop().count = 0
            spare = [f for f in sc.trucks if not f.count]
            while len(working) < n and spare:  # idle trucks come back first
                spare[0].count = 1
                working.append(spare.pop(0))
            while len(working) < n:  # then copies of the last truck
                used = {f.id for f in sc.trucks}
                k = len(sc.trucks) + 1
                while f"WZ-{k:02d}" in used:
                    k += 1
                new = copy.deepcopy(working[-1] if working else sc.trucks[-1])
                new.id, new.count = f"WZ-{k:02d}", 1
                sc.trucks.append(new)
                working.append(new)
            return sc
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
    return cls(**(data or {}))


def load_scenario(path: str | Path) -> Scenario:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return scenario_from_dict(raw)


def _routes(raw_routes: dict | None) -> dict[str, Route]:
    return {rid: Route(rid, [_build(Segment, s) for s in r["segments"]], r.get("path"))
            for rid, r in (raw_routes or {}).items()}


def scenario_from_dict(raw: dict) -> Scenario:
    routes = _routes(raw["routes"])
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
        stockpile=_build(Stockpile, raw.get("stockpile", {})),
        pacing=_build(Pacing, raw.get("pacing", {})),
        spotting=_build(Spotting, raw.get("spotting", {})),
        shift_plan=_build(ShiftPlan, raw.get("shift_plan", {})),
        blasting=_build(Blasting, raw.get("blasting", {})),
        inspection=_build(Inspection, raw.get("inspection", {})),
        refueling=_build(Refueling, raw.get("refueling", {})),
        traffic=_build(Traffic, raw.get("traffic", {})),
        advisor=_build(Advisor, raw.get("advisor", {})),
        calibration=_build(Calibration, raw.get("calibration", {})),
        reliability=_build(Reliability, raw.get("reliability", {})),
        excavator_policy=_build(ExcavatorPolicy, raw.get("excavator_policy", {})),
        site=raw.get("site"),
        links=_routes(raw.get("links")),
        natural_switch_prob=raw.get("natural_switch_prob", 0.0),
        rl_weights=raw.get("rl_weights"),
        planned_blasts=[dict(p) for p in raw.get("planned_blasts") or []],
        crushers_extra=[_build(Crusher, c) for c in raw.get("crushers_extra") or []],
        areas=[dict(a) for a in raw.get("areas") or []],
        shift_goal=dict(raw.get("shift_goal") or {}),
    )
    from .psd import fragmentation_index

    for b in sc.blasts:  # a grading curve, when given, decides how blocky the material is
        if b.psd:
            b.fragmentation = fragmentation_index(b.psd, sc.crusher.max_feed_mm)
    for p in sc.planned_blasts + sc.areas:
        if p.get("psd"):
            p["fragmentation"] = fragmentation_index(p["psd"], sc.crusher.max_feed_mm)
    sc.validate()
    return sc


def scenario_to_dict(sc: Scenario) -> dict:
    """Inverse of scenario_from_dict (for saving edited scenarios to YAML)."""
    from dataclasses import asdict

    d = {k: v for k, v in asdict(sc).items() if k not in ("routes", "site", "links", "rl_weights")}

    def routes(rs: dict) -> dict:
        out = {}
        for rid, r in rs.items():
            out[rid] = {"segments": [{k: v for k, v in asdict(s).items()
                                      if not (k in ("section", "section_dir") and not v)} for s in r.segments]}
            if r.path:
                out[rid]["path"] = r.path
        return out

    d["routes"] = routes(sc.routes)
    if sc.links:
        d["links"] = routes(sc.links)
    if sc.rl_weights:
        d["rl_weights"] = [float(w) for w in sc.rl_weights]
    for f in d["trucks"]:
        if not f["excavators"]:
            f.pop("excavators")
        if not f["blend"]:
            f.pop("blend")
    d["blasts"] = [{k: v for k, v in b.items() if v not in (None, "")} for b in d["blasts"]]
    if sc.site:
        d["site"] = sc.site
    return d


class _NoAliasDumper(yaml.SafeDumper):
    """Writes repeated lists (e.g. the same grading curve) in full instead of &id/*id references."""

    def ignore_aliases(self, data):
        return True


def yaml_dump(data) -> str:
    return yaml.dump(data, Dumper=_NoAliasDumper, allow_unicode=True, sort_keys=False)


def dump_scenario(sc: Scenario) -> str:
    return yaml_dump(scenario_to_dict(sc))


def clock_to_s(hhmm: str, start: str = "06:00") -> float:
    """'HH:MM' -> seconds since the shift start (times after midnight wrap around)."""
    def minutes(t: str) -> int:
        h, m = str(t).strip().split(":")
        return int(h) * 60 + int(m)
    d = minutes(hhmm) - minutes(start)
    return (d % (24 * 60)) * 60.0


def s_to_clock(sec: float, start: str = "06:00") -> str:
    h, m = str(start).split(":")
    total = (int(h) * 60 + int(m) + int(sec // 60)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"
