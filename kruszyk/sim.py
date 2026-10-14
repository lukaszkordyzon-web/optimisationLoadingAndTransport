"""Discrete-event simulation of the load-haul-crush chain (digital twin).

Produces synthetic data (segment travel times, truck states, loads, hopper level)
in the same shape the phone app will later deliver from real GPS geofences.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

from .config import REPAIR_SIGMA, Excavator, Scenario, TruckFleet, clock_to_s, truck_units
from .site import blast_point, haversine_m
from .physics import (crusher_kwh_per_t, crusher_rate_tph, drill_blast_cost_per_t, load_time_s, passes_per_truck,
                      route_time_s, segment_time_s)

CRUSHER_TICK_S = 10.0
LOADER_TICK_S = 30.0
DRIVE_STATES = {"jazda_pusty", "przejazd_miedzy_usypami", "jazda_pelny"}
IDLE_STATES = {"kolejka_koparka", "podstawienie", "zaladunek", "kolejka_kruszarka", "wyladunek",
               "wyladunek_halda", "postoj_koparki", "wyjazd_z_frontu", "mijanka", "inne_przestoje"}
CONNECTORS = {"front", "plac_kruszarki", "dojazd"}  # short links to the network, not real roads
PREFILL_NATURAL_P = 0.3  # without information the operator sometimes guesses and fills the bucket in time
NATURAL_QUEUE_SWITCH_S = 180.0  # a driver goes to another crusher when the queue at the nearest is this much longer


@dataclass
class TruckState:
    id: str
    fleet: TruckFleet
    driver_factor: float
    home_excavator: str  # used by "fixed" dispatch
    trip: int = 0
    speeder: bool = False  # drives above the limit (site practice)
    at: str = ""  # crusher where the truck is (dumped last); trucks start at the main crusher
    fuel_left_l: float = 0.0
    down_end: float = 0.0  # end of the current / last repair


@dataclass
class CrusherUnit:
    """One crusher: its hopper, dump point and counters."""
    cfg: object
    hopper: simpy.Container
    dump_point: simpy.Resource
    frag: float = 0.0          # fragmentation of the material in the hopper
    queued_dump_t: float = 0.0  # tonnes waiting at the dump point
    inbound_t: float = 0.0     # tonnes on loaded trucks driving here
    crushed_t: float = 0.0
    busy_s: float = 0.0
    starved_s: float = 0.0
    kwh: float = 0.0
    down_until: float = 0.0  # broken down until
    down_s: float = 0.0


@dataclass
class ExcavatorState:
    cfg: Excavator
    resource: simpy.Resource
    remaining_t: float
    committed_t: float = 0.0
    # trucks sent here but not yet loaded: truck_id -> (expected arrival, expected service)
    pending: dict = field(default_factory=dict)
    busy_until: float = 0.0  # expected end of the service currently in progress
    depleted_at: float | None = None
    loading_s: float = 0.0
    spotting_s: float = 0.0
    last_end: float = 0.0       # when the previous loading finished (excavator free since then)
    prefills: int = 0           # loads started with a bucket filled in advance
    topup_skips: int = 0        # trucks sent without the last bucket
    parked: bool = False        # parked by the excavator policy (not dispatched to)
    parked_since: float | None = None
    parked_s: float = 0.0
    down: list = field(default_factory=list)  # (from, to, reason): breaks, blasting, refuelling
    bays: simpy.Resource | None = None  # loading positions: 2 when trucks are loaded from both sides
    spot_modes: list = field(default_factory=list)  # [(share, mode dict)]
    route_id: str = ""          # current route to the crusher (changes as the excavator advances along the face)
    initial_t: float = 0.0
    loaded_t: float = 0.0
    down_end: float = 0.0  # end of the current / last breakdown
    next_reloc_t: float = math.inf
    dig_step: int = 0
    relocations: int = 0


def _shift_tracks(frames: list, relocs: list, routes: dict, ex) -> list:
    """Moves to the parking place were planned from the starting point of the face; after the excavator
    has advanced, they start and end at its current place instead."""
    if not relocs:
        return frames
    base = tuple(routes[ex.cfg.route].path[0])
    out = []
    for t, lat, lon in frames:
        cur = base
        for tr, new, _ in relocs:
            if tr <= t:
                cur = new
        if cur != base and haversine_m((lat, lon), base) < 1.0 and not any(abs(t - tr) < 0.2 for tr, *_ in relocs):
            lat, lon = cur
        out.append([t, lat, lon])
    return out


class Simulation:
    def __init__(self, sc: Scenario, seed: int | None = None):
        sc.validate()
        self.sc = sc
        self.rng = np.random.default_rng(sc.seed if seed is None else seed)
        # breakdowns draw from their own stream: the rest of the shift stays comparable with / without them
        self.frng = np.random.default_rng((sc.seed if seed is None else seed) + 99_991)
        self.env = simpy.Environment()
        self.shift_s = sc.shift_hours * 3600
        v = sc.variability
        self.rain = bool(self.rng.random() < v.rain_probability)
        self.weather_factor = 1 + (v.rain_slowdown if self.rain else 0.0)

        self.excavators = {
            e.id: ExcavatorState(e, simpy.PriorityResource(self.env, 1), sc.blast(e.blast).tonnage_t, route_id=e.route)
            for e in sc.excavators
        }
        for ex in self.excavators.values():
            mix = {k: float(v or 0) for k, v in ex.cfg.spotting.items() if float(v or 0) > 0}
            if mix:
                total = sum(mix.values())
                ex.spot_modes = [(v / total, sc.spotting.mode(k)) for k, v in mix.items()]
            else:  # one fixed spotting time (scenarios without spotting modes)
                ex.spot_modes = [(1.0, {"in_s": ex.cfg.spot_time_s, "out_s": 0.0, "extra_m": 0.0,
                                        "double": False, "out_blocks": False})]
            double = any(m["double"] for _, m in ex.spot_modes)
            ex.bays = simpy.Resource(self.env, 2 if double else 1)
            if ex.cfg.start_delay_min > 0:  # drives to a new pile first
                ex.down.append((0.0, ex.cfg.start_delay_min * 60, "przejazd"))
            if sc.carry_down.get(ex.cfg.id):  # repair from the previous shift still going on
                ex.down_end = float(sc.carry_down[ex.cfg.id])
                ex.down.append((0.0, ex.down_end, "awaria"))
        # several excavators on the same blast share its tonnage
        by_blast: dict[str, list[ExcavatorState]] = {}
        for ex in self.excavators.values():
            by_blast.setdefault(ex.cfg.blast, []).append(ex)
        for exs in by_blast.values():
            for ex in exs:
                ex.remaining_t /= len(exs)
                ex.initial_t = ex.remaining_t
                if len(ex.cfg.dig_routes) > 1 and ex.cfg.relocate_every_t > 0:
                    ex.next_reloc_t = ex.cfg.relocate_every_t

        self.trucks: list[TruckState] = []
        # driver/truck style is persistent across shifts (drawn from the scenario seed, not the shift seed)
        crew_rng = np.random.default_rng(sc.seed)
        per_fleet: dict[int, int] = {}
        for i, (tid, fleet) in enumerate(truck_units(sc.trucks)):
            allowed = self.allowed_ids(fleet)
            j = per_fleet[id(fleet)] = per_fleet.get(id(fleet), -1) + 1
            self.trucks.append(TruckState(
                id=tid,
                fleet=fleet,
                driver_factor=float(np.clip(crew_rng.normal(1.0, v.driver_sd), 0.8, 1.3)),
                home_excavator=allowed[(i + j) % len(allowed)],
                speeder=bool(crew_rng.random() < v.speeding_share),
                fuel_left_l=fleet.tank_l,
            ))

        self.units = {c.id: CrusherUnit(c, simpy.Container(self.env, capacity=c.hopper_t, init=0),
                                        simpy.Resource(self.env, 1)) for c in sc.crushers}
        self.main = self.units[sc.crusher.id]
        self.hopper, self.dump_point = self.main.hopper, self.main.dump_point
        for t in self.trucks:
            t.at = sc.crusher.id
        self.crushed_t = 0.0
        self.crusher_busy_s = 0.0
        self.crusher_starved_s = 0.0
        self.crusher_kwh = 0.0
        self.stockpile_t = sc.stockpile.initial_t
        self.stockpile_in_t = 0.0
        self.rehandled_t = 0.0
        self.fuel_l = 0.0
        self.dumped_t = 0.0
        self.decisions = 0
        # feed mix: what each pile should give (normalised targets) and what it has given so far
        targets = {b.id: float(b.target_share) for b in sc.blasts if b.target_share}
        tot = sum(targets.values())
        self.targets = {b.id: targets.get(b.id, 0.0) / tot for b in sc.blasts} if tot > 0 else {}
        self.loaded_by_blast = {b.id: 0.0 for b in sc.blasts}
        # excavators that rigidly assigned trucks depend on – never parked
        self.pinned = set()
        for f in sc.trucks:
            if f.count and f.blend:
                self.pinned |= set(f.blend)
            elif f.count and len(f.excavators) == 1:
                self.pinned.add(f.excavators[0])
        self.flex_trucks = sum(f.count for f in sc.trucks if not f.blend and len(f.excavators) != 1)
        self.yields, self.yield_s = 0, 0.0
        self.relocated_at: dict[str, list] = {}  # excavator -> [(time, new position, previous position)]
        self.sections: dict[str, dict] = {}  # single-lane stretches between passing bays
        self.bay_stops = 0
        self.truck_refuels = 0
        self.over_limit_segments, self.drive_segments = 0, 0

        # ----- the shift as it really is: start-up, breaks, end, blasting, light vehicles, refuelling
        plan = sc.shift_plan
        self.t_start = plan.startup_min * 60
        self.work_end = self.shift_s - plan.end_early_min * 60
        self.breaks = [(clock_to_s(b["start"], plan.start_time), clock_to_s(b["start"], plan.start_time) + float(b["min"]) * 60)
                       for b in plan.breaks or [] if b.get("start")]
        bl = sc.blasting
        self.closure = None
        if bl.enabled:
            a = clock_to_s(bl.fire_from, plan.start_time) - bl.evacuation_min * 60
            self.closure = (max(0.0, a), clock_to_s(bl.fire_to, plan.start_time))
        self.excavator_moves = self._excavator_moves()  # ex id -> (Route, travel time s)
        self.excavator_tracks: dict[str, list] = {}
        for ex in self.excavators.values():
            ex.last_end = self.t_start
            ex.down += [(a, b, "przerwa") for a, b in self.breaks]
            if self.closure:
                move_t = self.excavator_moves.get(ex.cfg.id, (None, 0.0))[1]
                # the excavator leaves before the evacuation and comes back after the opening
                ex.down.append((max(0.0, self.closure[0] - move_t), self.closure[1] + move_t, "strzelanie"))
                self._track_move(ex.cfg.id, self.closure[0] - move_t, self.closure[1] + move_t)
        # trucks stop going down when the first excavator starts driving away to its parking place
        self.truck_closure = None
        if self.closure:
            starts = [w[0] for ex in self.excavators.values() for w in ex.down if w[2] == "strzelanie"]
            self.truck_closure = (min(starts + [self.closure[0]]), self.closure[1])
        self.light_vehicles: list[dict] = []   # for the animation
        self.light_passages: dict[str, list] = {}  # road name -> [(t0, t1, going_up)]
        self._schedule_light_vehicles()

        self.segments: list[dict] = []
        self.states: list[dict] = []
        self.loads: list[dict] = []
        self.hopper_log: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _log_state(self, truck: TruckState, state: str, start: float, ex: str | None = None,
                   end: float | None = None):
        end = self.env.now if end is None else end
        if end > start:
            self.states.append({"truck": truck.id, "state": state, "start_s": start,
                                "end_s": end, "duration_s": end - start,
                                "excavator": ex, "trip": truck.trip, "crusher": truck.at})
            if state in IDLE_STATES:
                used = truck.fleet.fuel_lph_idle * (end - start) / 3600 * self.sc.calibration.fuel_factor
                self.fuel_l += used
                truck.fuel_left_l -= used

    def _noise(self, sd: float) -> float:
        return float(self.rng.lognormal(-sd * sd / 2, sd))

    @staticmethod
    def _blocking_s(m: dict) -> float:
        """Excavator time a truck's manoeuvres take (it can't load anyone meanwhile)."""
        if m["double"]:
            return 0.0  # spots on the free side while the previous truck is being loaded
        return m["in_s"] + (m["out_s"] if m["out_blocks"] else 0.0)

    def _expected_service(self, ex: ExcavatorState, fleet: TruckFleet) -> float:
        blast = self.sc.blast(ex.cfg.blast)
        spot = sum(p * self._blocking_s(m) for p, m in ex.spot_modes)
        return spot + load_time_s(ex.cfg, fleet, blast, self.sc.loose_density_t_m3)

    def _route_of(self, ex: ExcavatorState):
        return self.sc.routes[ex.route_id or ex.cfg.route]

    def _route_to(self, ex: ExcavatorState, crusher_id: str | None):
        """Route between the excavator and a crusher (the main one follows the excavator along the face)."""
        if not crusher_id or crusher_id == self.sc.crusher.id:
            return self._route_of(ex)
        return self.sc.routes[ex.cfg.crusher_routes[crusher_id]]

    def _pick_crusher(self, truck: TruckState, plan, last: ExcavatorState, tonnes: float) -> CrusherUnit:
        """Where to dump: a crusher taking all the material on board. Today's practice: the nearest one, unless
        the driver sees a clearly longer queue there than at another line (no knowledge of trucks on the way);
        with the system: the one where the truck is unloaded soonest (drive + expected wait, trucks on the way)."""
        ok = [u for u in self.units.values() if all(u.cfg.takes(e.cfg.blast) for e, _ in plan)]
        if len(ok) <= 1:
            return ok[0] if ok else self.main
        fleet, rr = truck.fleet, self.sc.rolling_resistance_pct
        if self.sc.dispatch == "natural":
            near = min(ok, key=lambda u: self._route_to(last, u.cfg.id).length_m)
            seen = {u.cfg.id: self.expected_dump_wait_s(tonnes, unit=u) for u in ok}  # what the queues look like now
            other = min(ok, key=lambda u: seen[u.cfg.id])
            return other if seen[near.cfg.id] - seen[other.cfg.id] > NATURAL_QUEUE_SWITCH_S else near

        def finish(u):
            haul = route_time_s(self._route_to(last, u.cfg.id), fleet, True, rr)
            return haul + self.expected_dump_wait_s(tonnes, extra_t=u.inbound_t, ahead_s=haul, unit=u)
        return min(ok, key=finish)

    def _relocate(self, ex: ExcavatorState):
        """The excavator moves to the next point of its digging line: the loader stands meanwhile
        (trucks wait), then trucks use the route from the new position."""
        env, cfg = self.env, ex.cfg
        n = len(cfg.dig_routes) - 1
        step = min(n, ex.dig_step + 1)
        if step <= ex.dig_step:
            return
        with ex.resource.request(priority=-1) as req:  # before the next truck in the queue
            yield req
            a = env.now
            old = self.sc.routes[cfg.dig_routes[ex.dig_step]].path
            new = self.sc.routes[cfg.dig_routes[step]].path
            yield env.timeout(cfg.relocate_min * 60)
            ex.down.append((a, env.now, "przestawienie"))
            ex.dig_step, ex.route_id = step, cfg.dig_routes[step]
            ex.relocations += 1
            ex.last_end = env.now
            if old and new:
                fr = self.excavator_tracks.setdefault(cfg.id, [[0.0, *old[0]]])
                fr.append([round(a, 1), *old[0]])
                fr.append([round(env.now, 1), *new[0]])
                self.relocated_at.setdefault(cfg.id, []).append((env.now, tuple(new[0]), tuple(old[0])))

    def _queue_len(self, ex: ExcavatorState) -> int:
        return len(ex.bays.queue) + len(ex.resource.queue)

    def _active(self, ex: ExcavatorState) -> bool:
        return ex.remaining_t - ex.committed_t > 1e-6

    def _crusher_rate(self, unit: CrusherUnit | None = None) -> float:
        u = unit or self.main
        return crusher_rate_tph(u.cfg, u.frag)

    def expected_dump_wait_s(self, tonnes: float, extra_t: float = 0.0, ahead_s: float = 0.0,
                             unit: CrusherUnit | None = None) -> float:
        """Expected wait at the hopper for `tonnes`, if `extra_t` more arrives first and the
        crusher keeps draining for `ahead_s` seconds before we get there."""
        u = unit or self.main
        rate = self._crusher_rate(u)
        backlog = u.hopper.level + u.queued_dump_t + extra_t - rate * ahead_s / 3600
        missing = max(0.0, backlog + tonnes - u.cfg.hopper_t)
        hopper_wait = missing / rate * 3600
        queue_wait = (len(u.dump_point.queue) + len(u.dump_point.users)) * u.cfg.dump_time_s
        down = max(0.0, u.down_until - self.env.now - ahead_s)  # a broken crusher: the operator knows (radio)
        return max(hopper_wait, queue_wait if extra_t == 0 else 0.0) + (down if missing > 0 else 0.0)

    # ------------------------------------------------------------ light vehicles
    def _light_trip(self, name: str, route, forward: bool, t0: float, frames: list, record: bool = True) -> float:
        """A light vehicle drives a route starting at t0; records its road passages. Returns arrival time."""
        from .anim import _at, _cum

        segs = list(range(len(route.segments)))
        if not forward:
            segs.reverse()
        path = route.path
        cum = _cum(path) if path else None
        seg_len = [x.length_m for x in route.segments]
        scale = (cum[-1] / sum(seg_len)) if path and sum(seg_len) else 1.0
        bounds = np.concatenate([[0.0], np.cumsum(seg_len)]) * scale
        t = t0
        for k in segs:
            seg = route.segments[k]
            v = max(5.0, min(self.sc.traffic.light_speed_kmh, seg.speed_limit_kmh))
            dt = seg.length_m / (v / 3.6)
            grade = seg.grade_pct if forward else -seg.grade_pct
            if record and seg.name not in CONNECTORS:
                self.light_passages.setdefault(seg.name, []).append((t, t + dt, grade > 0.5, name))
            if record and path:
                d0, d1 = (bounds[k], bounds[k + 1]) if forward else (bounds[k + 1], bounds[k])
                for d in (d0, d1):
                    lat, lon = _at(path, cum, d)
                    frames.append([round(t + (dt if d == d1 else 0.0), 1), round(lat, 7), round(lon, 7)])
            t += dt
        return t

    def _visit_round(self, name: str, kind: str, start: float, stops: list[str], stop_s: float,
                     refuel: bool = False) -> None:
        """crusher -> stop 1 -> stop 2 … -> crusher. A tanker round also stops the excavators for refuelling."""
        sc = self.sc
        stops = [x for x in stops if x in self.excavators]
        if not stops:
            return
        frames: list = []
        t, prev = start, None
        for x in stops:
            ex = self.excavators[x]
            if prev is None:
                t = self._light_trip(name, self._route_of(ex), False, t, frames)
            else:
                link = sc.links.get(f"{prev}>{x}")
                if link is not None:
                    t = self._light_trip(name, link, True, t, frames)
                else:  # no link drawn: back to the crusher and down again
                    t = self._light_trip(name, self._route_of(self.excavators[prev]), True, t, frames)
                    t = self._light_trip(name, self._route_of(ex), False, t, frames)
            if refuel:
                self._refuel_window(ex, t, stop_s)
            if frames:
                frames.append([round(t + stop_s, 1), frames[-1][1], frames[-1][2]])
            t += stop_s
            prev = x
        t = self._light_trip(name, self._route_of(self.excavators[prev]), True, t, frames)
        self.light_vehicles.append({"name": name, "kind": kind, "frames": frames})

    def _schedule_light_vehicles(self) -> None:
        sc = self.sc
        start = sc.shift_plan.start_time
        bl = sc.blasting
        site = sc.site or {}
        if bl.place and site.get("blast_place") != bl.place:
            site = dict(site, blast_place=bl.place)
        if bl.enabled and blast_point(site) and site.get("roads") and site.get("crusher"):
            from .config import _routes
            from .site import crew_route

            try:
                r = crew_route(site)  # dedicated crew route if drawn, else the shortest road path
                if r is None:
                    raise ValueError("brak trasy")
                route = _routes({"strzelanie": r})["strzelanie"]
                frames: list = []
                down = self._light_trip("ekipa strzałowa", route, False, 0.0, [], record=False)  # duration only
                t_in = max(0.0, clock_to_s(bl.crew_arrive, start) - down)
                self._light_trip("ekipa strzałowa", route, False, t_in, frames)
                t_out = clock_to_s(bl.crew_leave, start)
                if frames:
                    frames.append([round(t_out, 1), frames[-1][1], frames[-1][2]])
                self._light_trip("ekipa strzałowa", route, True, t_out, frames)
                self.light_vehicles.append({"name": "ekipa strzałowa", "kind": "strzelanie", "frames": frames})
            except ValueError:
                pass  # blast point not connected to the roads: the closure still applies
        ins = sc.inspection
        if ins.enabled and ins.stops:
            if not self._dedicated_round("sztygar", "objazd", "sztygar", clock_to_s(ins.start, start), list(ins.stops),
                                         ins.stop_min * 60):
                self._visit_round("sztygar", "objazd", clock_to_s(ins.start, start), list(ins.stops), ins.stop_min * 60)
        for i, tt in enumerate(sc.refueling.tanker_times or []):
            ids = [e.id for e in sc.excavators]
            if not self._dedicated_round(f"cysterna {i + 1}", "cysterna", "cysterna", clock_to_s(tt, start), ids,
                                         sc.refueling.excavator_refuel_min * 60, refuel=True):
                self._visit_round(f"cysterna {i + 1}", "cysterna", clock_to_s(tt, start), ids,
                                  sc.refueling.excavator_refuel_min * 60, refuel=True)

    def _dedicated_round(self, name: str, kind: str, role: str, start: float, stops: list[str], stop_s: float,
                         refuel: bool = False) -> bool:
        """Round along a route drawn for this role, stopping at the points nearest the excavators.
        Returns False when no dedicated route is drawn (the caller falls back to the roads)."""
        from .config import _routes
        from .site import round_route

        site = self.sc.site or {}
        stops = [x for x in stops if x in self.excavators]
        pos = {e["id"]: (e["lat"], e["lon"]) for e in site.get("excavators", [])}
        r = round_route(site, role, [pos[x] for x in stops if x in pos]) if site else None
        if r is None:
            return False
        route = _routes({role: r})[role]
        stop_after = r.get("stop_after", [])
        frames: list = []
        t = start
        for k, seg in enumerate(route.segments):
            t = self._light_segment(name, route, k, t, frames)
            for j, sk in enumerate(stop_after):
                if sk == k:
                    if refuel and j < len(stops):
                        self._refuel_window(self.excavators[stops[j]], t, stop_s)
                    if frames:
                        frames.append([round(t + stop_s, 1), frames[-1][1], frames[-1][2]])
                    t += stop_s
        path = route.path
        if path and haversine_m(path[0], path[-1]) > 50:  # not a loop: drive back the same way
            t = self._light_trip(name, route, False, t, frames)
        self.light_vehicles.append({"name": name, "kind": kind, "frames": frames})
        return True

    def _light_segment(self, name: str, route, k: int, t: float, frames: list) -> float:
        """One segment k of a route driven forward by a light vehicle."""
        from .anim import _at, _cum

        seg = route.segments[k]
        v = max(5.0, min(self.sc.traffic.light_speed_kmh, seg.speed_limit_kmh))
        dt = seg.length_m / (v / 3.6)
        if seg.name not in CONNECTORS:
            self.light_passages.setdefault(seg.name, []).append((t, t + dt, seg.grade_pct > 0.5, name))
        if route.path:
            cum = _cum(route.path)
            seg_len = [x.length_m for x in route.segments]
            scale = cum[-1] / sum(seg_len) if sum(seg_len) else 1.0
            d0, d1 = sum(seg_len[:k]) * scale, sum(seg_len[:k + 1]) * scale
            for d, tt in ((d0, t), (d1, t + dt)):
                lat, lon = _at(route.path, cum, d)
                frames.append([round(tt, 1), round(lat, 7), round(lon, 7)])
        return t + dt

    # ------------------------------------------------------------ excavator moves
    def _excavator_moves(self) -> dict:
        """Where each excavator goes for blasting/refuelling and how long it drives there."""
        from .config import _routes
        from .site import excavator_move

        site = self.sc.site or {}
        out = {}
        if not site.get("excavators"):
            return out
        for e in site["excavators"]:
            if e["id"] not in self.excavators:
                continue
            ex = self.excavators[e["id"]]
            r_cr = self.sc.routes.get(ex.cfg.route)
            m = excavator_move(site, e, r_cr.path if r_cr else None, self.sc.blasting.retreat_m)
            if m is None:
                continue
            route = _routes({"m": m})["m"]
            t = route.length_m / (self.sc.traffic.excavator_move_kmh / 3.6)
            out[e["id"]] = (route, t, bool(m.get("parking")))
        return out

    def _refuel_window(self, ex: ExcavatorState, t: float, stop_s: float) -> None:
        move = self.excavator_moves.get(ex.cfg.id)
        if self.sc.refueling.at_parking and move and move[2]:
            a, b = t - move[1], t + stop_s + move[1]
            self._track_move(ex.cfg.id, a, b)
        else:
            a, b = t, t + stop_s
        ex.down.append((max(0.0, a), b, "tankowanie"))

    def _track_move(self, ex_id: str, a: float, b: float) -> None:
        """Animation keyframes: drive to the place, stay, drive back."""
        from .anim import _at, _cum

        move = self.excavator_moves.get(ex_id)
        if not move or not move[0].path:
            return
        route, t_move = move[0], move[1]
        path = route.path
        cum = _cum(path)
        fr = self.excavator_tracks.setdefault(ex_id, [[0.0, *path[0]]])
        n = 8
        for i in range(n + 1):  # out
            lat, lon = _at(path, cum, cum[-1] * i / n)
            fr.append([round(a + t_move * i / n, 1), lat, lon])
        for i in range(n + 1):  # back
            lat, lon = _at(path, cum, cum[-1] * (1 - i / n))
            fr.append([round(b - t_move + t_move * i / n, 1), lat, lon])

    def _yield_delay(self, road: str, t0: float, t1: float, truck_going_up: bool) -> tuple[int, float]:
        """Giving way: a vehicle going DOWN gives way to one coming UP. So a truck driving down loses time
        for every light vehicle coming up this road meanwhile; a truck going up keeps going."""
        if truck_going_up or road in CONNECTORS:
            return 0, 0.0
        n = sum(1 for a, b, up, _ in self.light_passages.get(road, []) if up and a < t1 and b > t0)
        return n, n * self.sc.traffic.yield_s

    # ------------------------------------------------------------ shift windows
    def _window(self, t: float, windows) -> tuple[float, float] | None:
        for a, b, *_ in windows:
            if a <= t < b:
                return a, b
        return None

    def _excavator_wait(self, ex: ExcavatorState):
        """Excavator unavailable (break, blasting, refuelling): wait until it is back."""
        w = self._window(self.env.now, ex.down)
        while w is not None:
            yield self.env.timeout(w[1] - self.env.now)
            w = self._window(self.env.now, ex.down)

    # --------------------------------------------------------------- dispatcher
    def allowed_ids(self, fleet: TruckFleet) -> list[str]:
        ids = list(fleet.blend) or list(fleet.excavators)
        return ids or [e.id for e in self.sc.excavators]

    def predicted_finish(self, ex: ExcavatorState, arrival: float, service: float) -> float:
        """When would a truck arriving at `arrival` be loaded, given the trucks already sent there."""
        t = max(self.env.now, ex.busy_until)
        for arr, srv in sorted(ex.pending.values()):
            t = max(t, arr) + srv
        return max(t, arrival) + service

    def features(self, truck: TruckState, ex: ExcavatorState) -> list[float]:
        """What the dispatcher knows about sending `truck` to `ex` now (RL state for one action)."""
        sc = self.sc
        route = self._route_to(ex, truck.at)
        travel = route_time_s(route, truck.fleet, False, sc.rolling_resistance_pct)
        haul = route_time_s(route, truck.fleet, True, sc.rolling_resistance_pct)
        service = self._expected_service(ex, truck.fleet)
        wait = self.predicted_finish(ex, self.env.now + travel, service) - (self.env.now + travel) - service
        initial = max(sc.blast(ex.cfg.blast).tonnage_t, 1.0)
        deficit = 0.0
        if self.targets:  # how far this pile is behind its share of the feed, in truckloads
            total = sum(self.loaded_by_blast.values())
            deficit = (self.targets.get(ex.cfg.blast, 0.0) * total - self.loaded_by_blast[ex.cfg.blast]) / truck.fleet.payload_t
            deficit = float(np.clip(deficit, -10, 10))
        return [max(wait, 0.0) / 60, travel / 60, haul / 60, service / 60,
                (ex.remaining_t - ex.committed_t) / initial, float(len(ex.pending)),
                float(ex.cfg.id == truck.home_excavator), deficit]

    def dispatch(self, truck: TruckState) -> ExcavatorState | None:
        allowed = set(self.allowed_ids(truck.fleet))
        candidates = [ex for ex in self.excavators.values() if ex.cfg.id in allowed and self._active(ex)]
        if not candidates:
            return None
        # parked excavators are skipped – unless they are the only ones this truck may serve
        active = [ex for ex in candidates if not ex.parked] or candidates
        self.decisions += 1
        mode = self.sc.dispatch
        home = self.excavators.get(truck.home_excavator)
        if mode in ("fixed", "natural") and home in active:
            # natural: drivers mostly go back to "their" excavator, sometimes elsewhere, without information
            if mode == "fixed" or self.rng.random() >= self.sc.natural_switch_prob:
                return home
        if mode == "natural":
            return active[int(self.rng.integers(len(active)))]
        if mode == "rl":
            w = np.asarray(self.sc.rl_weights, dtype=float)
            feats = {ex.cfg.id: self.features(truck, ex) for ex in active}
            n = len(next(iter(feats.values())))
            w = np.pad(w, (0, max(0, n - len(w))))[:n]  # policies saved before new features keep working
            return min(active, key=lambda ex: float(np.dot(w, feats[ex.cfg.id])))
        best, best_finish = None, math.inf
        for ex in active:
            travel = route_time_s(self._route_to(ex, truck.at), truck.fleet, False,
                                  self.sc.rolling_resistance_pct)
            finish = self.predicted_finish(ex, self.env.now + travel, self._expected_service(ex, truck.fleet))
            if finish < best_finish:
                best, best_finish = ex, finish
        return best

    # ---------------------------------------------------------------- single-lane roads (passing bays)
    def _section(self, key: str) -> dict:
        if key not in self.sections:
            self.sections[key] = {"inside": {1: 0, -1: 0}, "waiting": {1: 0, -1: 0}, "approach": {},
                                  "up": 0, "changed": self.env.event()}
        return self.sections[key]

    def _notify(self, s: dict) -> None:
        ev, s["changed"] = s["changed"], self.env.event()
        ev.succeed()

    @staticmethod
    def _dirs(seg, forward: bool) -> tuple[int, int]:
        """(direction along the road drawing, +1 uphill / -1 downhill / 0 flat) of a truck on this segment."""
        d = (seg.section_dir or 1) * (1 if forward else -1)
        grade = seg.grade_pct if forward else -seg.grade_pct
        return d, (1 if grade > 0.5 else -1 if grade < -0.5 else 0)

    def _approach(self, truck: TruckState, seg, forward: bool, eta: float) -> None:
        d, slope = self._dirs(seg, forward)
        s = self._section(seg.section)
        if slope > 0:
            s["up"] = d
        s["approach"][truck.id] = (d, eta)

    def _enter_section(self, truck: TruckState, seg, forward: bool, route, k: int, load_t: float):
        """Uphill trucks have priority: a truck going down does not enter while one is coming up inside
        the stretch, waiting at its lower bay, or about to reach it before we could get through."""
        env = self.env
        d, slope = self._dirs(seg, forward)
        s = self._section(seg.section)
        if slope > 0:
            s["up"] = d
        s["approach"].pop(truck.id, None)
        through = segment_time_s(seg, truck.fleet, load_t > 0, self.sc.rolling_resistance_pct, forward, load_t)
        waited = False
        while True:
            blocked = s["inside"][-d] > 0
            if not blocked and slope <= 0 and s["up"] == -d:
                blocked = (s["waiting"][-d] > 0 or
                           any(ad == -d and eta <= env.now + through for ad, eta in s["approach"].values()))
            if not blocked:
                break
            if not waited:
                s["waiting"][d] += 1
                waited = True
                self.bay_stops += 1
            yield env.any_of([s["changed"], env.timeout(10.0)])
        if waited:
            s["waiting"][d] -= 1
        s["inside"][d] += 1
        self._notify(s)

    def _leave_section(self, seg, forward: bool) -> None:
        d, _ = self._dirs(seg, forward)
        s = self._section(seg.section)
        s["inside"][d] -= 1
        self._notify(s)

    # ---------------------------------------------------------------- processes
    def _drive(self, truck: TruckState, route, forward: bool, load_t: float, leg: str, ex_id: str,
               pace: float = 1.0):
        """Drive a whole route (forward = towards its end). leg: pusty | miedzy | pelny."""
        idx = list(range(len(route.segments)) if forward else reversed(range(len(route.segments))))
        f = truck.fleet
        share = load_t / f.payload_t if f.payload_t else 0.0
        fuel_rate = (f.fuel_lph_empty + (f.fuel_lph_loaded - f.fuel_lph_empty) * share) * self.sc.calibration.fuel_factor
        start_state = self.env.now
        v = self.sc.variability
        for pos, k in enumerate(idx):
            seg = route.segments[k]
            if seg.section:  # single-lane stretch: wait at the passing bay until the way is free
                t_wait = self.env.now
                yield from self._enter_section(truck, seg, forward, route, k, load_t)
                if self.env.now > t_wait:
                    self._log_state(truck, {"pusty": "jazda_pusty", "miedzy": "przejazd_miedzy_usypami",
                                            "pelny": "jazda_pelny"}[leg], start_state, ex_id, end=t_wait)
                    self._log_state(truck, "mijanka", t_wait, ex_id)
                    start_state = self.env.now
            mean = segment_time_s(seg, f, load_t > 0, self.sc.rolling_resistance_pct, forward, load_t)
            # limits are binding; drivers who speed (site practice) go up to speeding_pct over the limit
            over = 1 + (v.speeding_pct / 100 if truck.speeder else 0.0)
            if truck.speeder:
                fast = dataclasses.replace(seg, speed_limit_kmh=seg.speed_limit_kmh * over)
                mean_drv = segment_time_s(fast, f, load_t > 0, self.sc.rolling_resistance_pct, forward, load_t)
            else:
                mean_drv = mean
            natural = mean_drv * truck.driver_factor * self.weather_factor * self._noise(v.trip_sd)
            natural = max(natural, seg.length_m / (seg.speed_limit_kmh * over / 3.6))
            natural *= self.sc.calibration.travel_factor  # measured on site: real roads, real drivers
            dur = natural * pace
            start = self.env.now
            grade = seg.grade_pct if forward else -seg.grade_pct
            n_y, delay = self._yield_delay(seg.name, start, start + dur, grade > 0.5)
            if n_y:
                self.yields += n_y
                self.yield_s += delay
            nxt = route.segments[idx[pos + 1]] if pos + 1 < len(idx) else None
            if nxt is not None and nxt.section:  # let the traffic on the next single-lane stretch know we're coming
                self._approach(truck, nxt, forward, start + dur + delay)
            yield self.env.timeout(dur + delay)
            if seg.section:
                self._leave_section(seg, forward)
            # slower driving burns roughly the same fuel per km (same climbing energy)
            used = fuel_rate * natural / 3600
            self.fuel_l += used
            truck.fuel_left_l -= used
            v_kmh = seg.length_m / dur * 3.6
            self.drive_segments += 1
            if seg.name not in CONNECTORS and v_kmh > seg.speed_limit_kmh + 0.5:
                self.over_limit_segments += 1
            self.segments.append({
                "truck": truck.id, "truck_model": f.model, "trip": truck.trip,
                "excavator": ex_id, "route": route.id, "segment": seg.name, "seg_idx": k,
                "forward": int(forward), "leg": leg,
                "loaded": int(load_t > 0), "payload_t": load_t, "length_m": seg.length_m,
                "grade_pct": seg.grade_pct if forward else -seg.grade_pct,
                "speed_limit_kmh": seg.speed_limit_kmh, "rain": int(self.rain),
                "hour_of_shift": start / 3600, "pace": pace, "start_s": start, "duration_s": dur + delay,
                "physics_s": mean, "v_kmh": round(v_kmh, 1), "yield_s": delay,
            })
        state = {"pusty": "jazda_pusty", "miedzy": "przejazd_miedzy_usypami", "pelny": "jazda_pelny"}[leg]
        self._log_state(truck, state, start_state, ex_id)

    def _pick_spot_mode(self, ex: ExcavatorState) -> dict:
        if len(ex.spot_modes) == 1:
            return ex.spot_modes[0][1]
        r, acc = self.rng.random(), 0.0
        for p, m in ex.spot_modes:
            acc += p
            if r < acc:
                return m
        return ex.spot_modes[-1][1]

    def _load_at(self, truck: TruckState, ex: ExcavatorState, amount: float, committed: float):
        """Queue, spot and load `amount` t at `ex`, then leave the loading position. Returns tonnes loaded.
        One-sided spotting: the truck needs the excavator for its whole manoeuvre. Two-sided: it spots on
        the free side while the previous truck is still being loaded, and only the loading itself waits."""
        env, sc, fleet = self.env, self.sc, truck.fleet
        mode = self._pick_spot_mode(ex)
        t0 = env.now
        with ex.bays.request() as bay:
            yield bay
            self._log_state(truck, "kolejka_koparka", t0, ex.cfg.id)
            ex.pending.pop(truck.id, None)
            ex.busy_until = max(env.now, ex.busy_until) + self._expected_service(ex, fleet)
            tonnes = min(amount, ex.remaining_t)
            pol = sc.excavator_policy
            if sc.dispatch == "rl" and pol.topup_min_fill < 0.999 and tonnes >= amount - 1e-6:
                # learned: with trucks waiting, send this one without its last bucket
                blast0 = sc.blast(ex.cfg.blast)
                n_pass = passes_per_truck(ex.cfg, fleet, blast0.loose_density_t_m3 or sc.loose_density_t_m3)
                if n_pass > 1 and (n_pass - 1) / n_pass >= pol.topup_min_fill and self._queue_len(ex) >= pol.topup_min_queue:
                    tonnes = amount * (n_pass - 1) / n_pass
                    ex.topup_skips += 1
            ex.remaining_t -= tonnes
            ex.committed_t -= committed
            if tonnes <= 1e-6:
                return 0.0
            if mode["double"]:
                t0 = env.now
                yield env.timeout(mode["in_s"] * self._noise(0.2))
                self._log_state(truck, "podstawienie", t0, ex.cfg.id)
            t0 = env.now
            with ex.resource.request() as req:
                yield req
                self._log_state(truck, "kolejka_koparka", t0, ex.cfg.id)
                t0 = env.now
                yield from self._excavator_wait(ex)
                if env.now > t0:
                    self._log_state(truck, "postoj_koparki", t0, ex.cfg.id)
                idle_before = env.now - ex.last_end  # how long the excavator had nothing to load

                if not mode["double"]:
                    t0 = env.now
                    yield env.timeout(mode["in_s"] * self._noise(0.2))
                    ex.spotting_s += env.now - t0
                    self._log_state(truck, "podstawienie", t0, ex.cfg.id)
                    t0 = env.now
                    yield from self._excavator_wait(ex)  # the window may have started while the truck was spotting
                    if env.now > t0:
                        self._log_state(truck, "postoj_koparki", t0, ex.cfg.id)

                blast = sc.blast(ex.cfg.blast)
                full_load = load_time_s(ex.cfg, fleet, blast, sc.loose_density_t_m3)
                one_pass = ex.cfg.swing_cycle_s * (1.0 + 0.5 * blast.fragmentation)
                load = full_load * (tonnes / fleet.payload_t) * self._noise(sc.variability.load_sd) * sc.calibration.load_factor
                # a bucket filled while the truck was still coming saves one pass – if the operator knew it was coming
                p_prefill = 1.0 if sc.advisor.enabled else PREFILL_NATURAL_P
                if idle_before >= one_pass and load > 1.5 * one_pass and self.rng.random() < p_prefill:
                    load -= one_pass
                    ex.prefills += 1
                t0 = env.now
                yield env.timeout(load)
                ex.loading_s += env.now - t0
                ex.last_end = env.now
                self._log_state(truck, "zaladunek", t0, ex.cfg.id)
                self.loaded_by_blast[blast.id] += tonnes
                self.loads.append({"truck": truck.id, "excavator": ex.cfg.id, "blast": blast.id,
                                   "material": blast.material, "tonnes": tonnes, "fragmentation": blast.fragmentation,
                                   "loaded_at_s": env.now, "trip": truck.trip})
                if ex.remaining_t <= 1e-6 and ex.depleted_at is None:
                    ex.depleted_at = env.now
                ex.loaded_t += tonnes
                if ex.loaded_t >= ex.next_reloc_t and ex.remaining_t > 1e-6:
                    ex.next_reloc_t += ex.cfg.relocate_every_t
                    env.process(self._relocate(ex))
                if mode["out_blocks"] and mode["out_s"] > 0:  # nobody can spot until this truck has left
                    t0 = env.now
                    yield env.timeout(mode["out_s"] * self._noise(0.2))
                    ex.spotting_s += env.now - t0
                    self._log_state(truck, "wyjazd_z_frontu", t0, ex.cfg.id)
            if mode["double"] and mode["out_s"] > 0:  # leaves its own side, the excavator already loads the other
                t0 = env.now
                yield env.timeout(mode["out_s"] * self._noise(0.2))
                self._log_state(truck, "wyjazd_z_frontu", t0, ex.cfg.id)
        out = (0.0 if (mode["out_blocks"] or mode["double"]) else mode["out_s"] * self._noise(0.2))
        out += mode["extra_m"] / (sc.spotting.floor_speed_kmh / 3.6)
        if out > 0:  # turning loop / leaving without blocking the next truck
            t0 = env.now
            yield env.timeout(out)
            self._log_state(truck, "wyjazd_z_frontu", t0, ex.cfg.id)
        return tonnes

    def _plan(self, truck: TruckState) -> list[tuple[ExcavatorState, float]] | None:
        """Loading stops for the next trip: [(excavator, tonnes)]."""
        fleet = truck.fleet
        if fleet.blend:
            stops = [(self.excavators[x], s) for x, s in fleet.blend.items() if self._active(self.excavators[x])]
            if stops:
                total = sum(s for _, s in stops)
                return [(ex, fleet.payload_t * s / total) for ex, s in stops]
            return None
        ex = self.dispatch(truck)
        return [(ex, fleet.payload_t)] if ex else None

    def _est_cycle(self, truck: TruckState) -> tuple[float, float]:
        """Expected (time until loaded, full cycle) for the truck's usual excavator – for shift planning."""
        sc = self.sc
        ex = self.excavators.get(truck.home_excavator) or next(iter(self.excavators.values()))
        r = self._route_to(ex, truck.at)
        to_loaded = route_time_s(r, truck.fleet, False, sc.rolling_resistance_pct) + self._expected_service(ex, truck.fleet)
        if truck.fleet.blend:  # two piles: one more drive and one more loading in the pit
            to_loaded *= 1.8
        return to_loaded, to_loaded + route_time_s(r, truck.fleet, True, sc.rolling_resistance_pct) + sc.crusher.dump_time_s

    def _shift_gate(self, truck: TruckState):
        """Before a new trip: wait out a break or the blasting closure; stop if the trip won't fit the shift."""
        env = self.env
        while True:
            now = env.now
            w = self._window(now, self.breaks)
            if w:
                yield env.timeout(w[1] - now)
                self._log_state(truck, "przerwa", now)
                continue
            to_loaded, cycle = self._est_cycle(truck)
            # safety margin: queues at the excavator can make the stay in the pit longer than expected
            tc = self.truck_closure
            if tc and now < tc[1] and now + 1.3 * to_loaded + 120 > tc[0]:
                yield env.timeout(tc[1] - now)  # nobody goes down before the blast is over
                self._log_state(truck, "postoj_strzelanie", now)
                continue
            if now + cycle > self.work_end:
                self._log_state(truck, "koniec_zmiany", now, end=self.shift_s)
                return False
            return True

    def truck_process(self, truck: TruckState):
        env, v, sc = self.env, self.sc.variability, self.sc
        while True:
            if not (yield from self._shift_gate(truck)):
                return
            plan = self._plan(truck)
            if plan is None:
                # logged up front: events exactly at shift end are not processed by env.run(until=...)
                self._log_state(truck, "postoj_brak_urobku", env.now, end=self.shift_s)
                return
            truck.trip += 1
            fleet = truck.fleet
            for ex, amount in plan:
                ex.committed_t += amount
            first = plan[0][0]
            travel = route_time_s(self._route_to(first, truck.at), fleet, False, sc.rolling_resistance_pct)
            service = self._expected_service(first, fleet)
            pace_empty = 1.0
            if sc.advisor.enabled:
                # recommended speed: arrive when the excavator is free – not earlier (queue), not later
                wait = self.predicted_finish(first, env.now + travel, service) - service - (env.now + travel)
                wait -= sc.advisor.arrival_margin_s  # learned safety margin: better a bit early than late
                pace_empty = 1 + min(sc.advisor.max_slowdown, max(0.0, wait) / travel)
            first.pending[truck.id] = (env.now + travel * pace_empty, service)
            cycle_start = env.now

            yield from self._drive(truck, self._route_to(first, truck.at), False, 0.0, "pusty", first.cfg.id, pace_empty)
            tonnes, prev = 0.0, None
            for ex, amount in plan:
                if prev is not None:
                    link = sc.links[f"{prev.cfg.id}>{ex.cfg.id}"]
                    yield from self._drive(truck, link, True, tonnes, "miedzy", ex.cfg.id)
                tonnes += yield from self._load_at(truck, ex, amount, amount)
                prev = ex
            last = prev
            if tonnes <= 1e-6:
                continue  # material ran out while driving there
            blast = sc.blast(last.cfg.blast)
            frag = float(np.mean([sc.blast(e.cfg.blast).fragmentation for e, _ in plan]))

            # which crusher takes this load
            unit = self._pick_crusher(truck, plan, last, tonnes)
            # speed advice: if we would only queue at the crusher, drive slower instead
            route = self._route_to(last, unit.cfg.id)
            pace = 1.0
            if (sc.pacing.enabled and sc.dispatch != "natural") or sc.advisor.enabled:
                haul = route_time_s(route, fleet, True, sc.rolling_resistance_pct)
                wait = self.expected_dump_wait_s(tonnes, extra_t=unit.inbound_t, ahead_s=haul, unit=unit)
                if sc.advisor.enabled:
                    pace = 1 + min(sc.advisor.max_slowdown, sc.advisor.crusher_pacing * wait / haul)
                else:
                    pace = 1 + min(sc.pacing.max_slowdown, wait / haul)
            unit.inbound_t += tonnes
            yield from self._drive(truck, route, True, tonnes, "pelny", last.cfg.id, pace)
            unit.inbound_t -= tonnes
            truck.at = unit.cfg.id

            sp = sc.stockpile
            t0 = env.now
            if sp.enabled and unit is self.main and self.expected_dump_wait_s(tonnes) > sp.divert_wait_s:
                yield env.timeout(sp.dump_time_s * self._noise(0.15))
                self.stockpile_t += tonnes
                self.stockpile_in_t += tonnes
                self._log_state(truck, "wyladunek_halda", t0)
                self._mark_dumped(truck, env.now, "halda")
            else:
                unit.queued_dump_t += tonnes
                with unit.dump_point.request() as req:
                    yield req
                    yield unit.hopper.put(tonnes)  # waits while the hopper is full
                    unit.queued_dump_t -= tonnes
                    self._log_state(truck, "kolejka_kruszarka", t0)
                    self._mix_into_hopper(tonnes, frag, unit)
                    t0 = env.now
                    yield env.timeout(unit.cfg.dump_time_s * self._noise(0.15))
                    self._log_state(truck, "wyladunek", t0)
                self._mark_dumped(truck, env.now, "kosz", unit.cfg.id)
            self.dumped_t += tonnes
            if sc.calibration.delay_min_per_trip > 0:  # losses measured on site that the model has no cause for
                t0 = env.now
                yield env.timeout(sc.calibration.delay_min_per_trip * 60 * self._noise(0.5))
                self._log_state(truck, "inne_przestoje", t0)

            # fuel: trucks start full; below the reserve they go to the fuel station next to the crusher
            if truck.fuel_left_l < fleet.tank_l * sc.refueling.truck_reserve_pct / 100:
                t0 = env.now
                yield env.timeout(sc.refueling.truck_refuel_min * 60 * self._noise(0.15))
                self._log_state(truck, "tankowanie", t0)
                truck.fuel_left_l = fleet.tank_l
                self.truck_refuels += 1

            # random breakdown, probability proportional to the time worked (cycle length)
            cycle = env.now - cycle_start
            rel_key = truck.id if truck.id in sc.reliability.machines else f"truck:{fleet.model}"
            mtbf, mttr = sc.reliability.rates(rel_key, "truck", sc.shift_hours)
            if mtbf < math.inf and self.frng.random() < 1 - math.exp(-cycle / (mtbf * 3600)):
                t0 = env.now
                truck.down_end = t0 + self._repair_s(mttr)
                if truck.down_end >= self.shift_s:  # the rest of the repair goes on to the next shift
                    self._log_state(truck, "awaria", t0, end=self.shift_s)
                    return
                yield env.timeout(truck.down_end - t0)
                self._log_state(truck, "awaria", t0)

    def _mix_into_hopper(self, tonnes: float, frag: float, unit: CrusherUnit | None = None):
        u = unit or self.main
        level = u.hopper.level
        u.frag = ((level - tonnes) * u.frag + tonnes * frag) / max(level, 1e-9)

    def _mark_dumped(self, truck: TruckState, t: float, where: str, crusher: str | None = None):
        for load in reversed(self.loads):
            if load["truck"] == truck.id and load["trip"] == truck.trip:
                load["dumped_at_s"] = t
                load["dumped_to"] = where
                load["crusher"] = crusher or self.sc.crusher.id
            elif load["truck"] == truck.id:
                return

    def _repair_s(self, mttr_h: float) -> float:
        """Repair time with the given mean: lognormal – mostly short, sometimes longer than a shift."""
        return float(self.frng.lognormal(math.log(mttr_h * 3600) - REPAIR_SIGMA ** 2 / 2, REPAIR_SIGMA))

    def breakdown_process(self, kind: str, obj):
        """Random breakdowns of an excavator or a crusher while the shift runs."""
        key = obj.cfg.id
        mtbf, mttr = self.sc.reliability.rates(key, kind, self.sc.shift_hours)
        if mtbf == math.inf:
            return
        env = self.env
        while True:
            yield env.timeout(self.frng.exponential(mtbf * 3600))
            if env.now >= self.shift_s:
                return
            end = env.now + self._repair_s(mttr)
            if kind == "excavator":
                obj.down.append((env.now, end, "awaria"))
                obj.down_end = end
            else:
                obj.down_until = end
            yield env.timeout(end - env.now)

    def crusher_process(self, u: CrusherUnit):
        env = self.env
        while True:
            if env.now < u.down_until:  # broken down: the hopper fills up, trucks queue or go elsewhere
                u.down_s += CRUSHER_TICK_S
                yield env.timeout(CRUSHER_TICK_S)
                continue
            rate = self._crusher_rate(u)
            capacity = rate * CRUSHER_TICK_S / 3600
            amount = min(u.hopper.level, capacity)
            if amount > 1e-9:
                yield u.hopper.get(amount)
                kwh = amount * crusher_kwh_per_t(u.cfg, u.frag) * self.sc.calibration.kwh_factor
                u.crushed_t += amount
                u.kwh += kwh
                u.busy_s += CRUSHER_TICK_S * amount / capacity
                u.starved_s += CRUSHER_TICK_S * (1 - amount / capacity)
                self.crushed_t += amount
                self.crusher_kwh += kwh
                self.crusher_busy_s += CRUSHER_TICK_S * amount / capacity
                self.crusher_starved_s += CRUSHER_TICK_S * (1 - amount / capacity)
            else:
                u.starved_s += CRUSHER_TICK_S
                self.crusher_starved_s += CRUSHER_TICK_S
            if u is self.main and int(env.now) % 60 == 0:
                self.hopper_log.append({"t_s": env.now, "level_t": self.hopper.level,
                                        "stockpile_t": self.stockpile_t, "crushed_t": self.crushed_t,
                                        "dumped_t": self.dumped_t, "fuel_l": self.fuel_l,
                                        "kwh": self.crusher_kwh, "stockpile_in_t": self.stockpile_in_t,
                                        "rehandled_t": self.rehandled_t})
            yield env.timeout(CRUSHER_TICK_S)

    def loader_process(self):
        """Wheel loader feeding the hopper from the buffer stockpile when the hopper runs low."""
        sp = self.sc.stockpile
        frag = float(np.mean([b.fragmentation for b in self.sc.blasts]))
        while True:
            free = self.hopper.capacity - self.hopper.level - self.main.queued_dump_t
            if self.stockpile_t > 1e-6 and self.hopper.level < sp.feed_below_t and free > 1:
                amount = min(self.stockpile_t, sp.loader_tph * LOADER_TICK_S / 3600, free)
                yield self.hopper.put(amount)
                self._mix_into_hopper(amount, frag)
                self.stockpile_t -= amount
                self.rehandled_t += amount
            yield self.env.timeout(LOADER_TICK_S)

    # ---------------------------------------------------------------------- run
    def excavator_review(self):
        """Learned rule: with too few flexible trucks per excavator, park the excavators with the longest
        truck cycle (they add cost but little tonnage). Reviewed every few minutes; parked ones can return."""
        pol, sc = self.sc.excavator_policy, self.sc
        fleet = next((f for f in sc.trucks if f.count and not f.blend and len(f.excavators) != 1), None)
        if fleet is None:
            return
        yield self.env.timeout(self.t_start)
        while True:
            cand = [ex for ex in self.excavators.values() if ex.remaining_t > 1e-6]
            k = max(1, math.ceil(self.flex_trucks / pol.min_trucks_per_excavator))
            pinned = [ex for ex in cand if ex.cfg.id in self.pinned]
            free = sorted((ex for ex in cand if ex.cfg.id not in self.pinned),
                          key=lambda ex: route_time_s(self._route_of(ex), fleet, False, sc.rolling_resistance_pct)
                          + route_time_s(self._route_of(ex), fleet, True, sc.rolling_resistance_pct)
                          + self._expected_service(ex, fleet))
            keep = {ex.cfg.id for ex in pinned} | {ex.cfg.id for ex in free[:max(0, k - len(pinned))]}
            now = self.env.now
            for ex in self.excavators.values():
                park = ex.cfg.id not in keep
                if park and not ex.parked:
                    ex.parked, ex.parked_since = True, now
                elif not park and ex.parked:
                    ex.parked_s += now - ex.parked_since
                    ex.parked, ex.parked_since = False, None
            yield self.env.timeout(pol.review_min * 60)

    def run(self) -> "SimResult":
        for u in self.units.values():
            u.down_until = float(self.sc.carry_down.get(u.cfg.id, 0.0))
            self.env.process(self.crusher_process(u))
            self.env.process(self.breakdown_process("crusher", u))
        for ex in self.excavators.values():
            self.env.process(self.breakdown_process("excavator", ex))
        if self.sc.dispatch == "rl" and self.sc.excavator_policy.min_trucks_per_excavator > 0.05:
            self.env.process(self.excavator_review())
        if self.sc.stockpile.enabled:
            self.env.process(self.loader_process())
        for i, truck in enumerate(self.trucks):
            self.env.process(self._delayed_start(truck, self.t_start + i * 30.0))  # start-up, staggered departures
        self.env.run(until=self.shift_s)
        return SimResult(self)

    def _delayed_start(self, truck: TruckState, delay: float):
        carry = float(self.sc.carry_down.get(truck.id, 0.0))
        if carry > 0:  # still in repair from the previous shift
            truck.down_end = carry
            self._log_state(truck, "awaria", 0.0, end=min(carry, self.shift_s))
            if carry >= self.shift_s:
                return
            delay = max(delay, carry)
        yield self.env.timeout(delay)
        yield from self.truck_process(truck)


class SimResult:
    def __init__(self, sim: Simulation):
        self.scenario = sim.sc
        self.rain = sim.rain
        self.segments = pd.DataFrame(sim.segments)
        self.states = pd.DataFrame(sim.states, columns=["truck", "state", "start_s", "end_s",
                                                        "duration_s", "excavator", "trip", "crusher"])
        self.loads = pd.DataFrame(sim.loads, columns=None if sim.loads else [
            "truck", "excavator", "blast", "material", "tonnes", "fragmentation", "loaded_at_s", "trip", "dumped_at_s"])
        self.hopper = pd.DataFrame(sim.hopper_log)
        self.truck_ids = [t.id for t in sim.trucks]
        self.light_vehicles = sim.light_vehicles
        self.excavator_down = {e.cfg.id: list(e.down) for e in sim.excavators.values()}
        self.excavator_steps = {k: ex.dig_step for k, ex in sim.excavators.items()}  # advance along the face
        self.excavator_tracks = {k: _shift_tracks(sorted(v), sim.relocated_at.get(k, []), sim.sc.routes, sim.excavators[k])
                                 for k, v in sim.excavator_tracks.items()}
        self.closure = sim.closure
        self.breaks = sim.breaks
        self.t_start, self.work_end = sim.t_start, sim.work_end
        self.kpi = self._kpis(sim)
        self.timeline = self._timeline(sim)

    def _timeline(self, sim: Simulation) -> pd.DataFrame:
        """Cumulative production and cost minute by minute (what a live dashboard would show)."""
        sc, c = sim.sc, sim.sc.costs
        h = self.hopper.copy()
        if h.empty:
            return h
        if h["t_s"].iloc[-1] < sim.shift_s:  # close the series at the end of the shift
            last = h.iloc[-1].to_dict() | {"t_s": float(sim.shift_s), "fuel_l": sim.fuel_l, "kwh": sim.crusher_kwh,
                                            "stockpile_in_t": sim.stockpile_in_t}
            h = pd.concat([h, pd.DataFrame([last])], ignore_index=True)
        # same formulas as the final KPIs (_kpis), applied to what has happened up to each moment
        fixed_per_h = (sc.truck_count * c.truck_per_h + len(sc.excavators) * c.excavator_per_h
                       + len(sc.crushers) * c.crusher_per_h
                       + (sc.stockpile.loader_per_h if sc.stockpile.enabled else 0.0))
        parked_s = sum(ex.parked_s for ex in sim.excavators.values())
        loads = self.loads
        done = (loads[loads["dumped_at_s"].notna()].sort_values("dumped_at_s") if "dumped_at_s" in loads
                else loads.iloc[0:0])
        t = h["t_s"].to_numpy()
        blast_ids = [b.id for b in sc.blasts]
        cum = {b: np.zeros(len(t)) for b in blast_ids}
        frag_t = np.zeros(len(t))
        if len(done):
            idx = np.searchsorted(done["dumped_at_s"].to_numpy(), t, side="right")
            for b in blast_ids:
                w = np.concatenate([[0.0], np.cumsum(np.where(done["blast"].to_numpy() == b, done["tonnes"].to_numpy(), 0.0))])
                cum[b] = w[idx]
            fw = np.concatenate([[0.0], np.cumsum(done["tonnes"].to_numpy() * done["fragmentation"].to_numpy())])
            frag_t = fw[idx]
            # tonnes dumped so far, from the same loads as the final KPIs (every crusher, every pile)
            h["dumped_t"] = np.concatenate([[0.0], np.cumsum(done["tonnes"].to_numpy())])[idx]
        dumped = sum(cum.values())
        with np.errstate(invalid="ignore", divide="ignore"):
            frag = np.where(dumped > 0, frag_t / np.where(dumped > 0, dumped, 1), 0.0)
            dev = np.zeros(len(t))
            for b, target in sim.targets.items():
                share = np.where(dumped > 0, 100 * cum[b] / np.where(dumped > 0, dumped, 1), 0.0)
                dev += 0.5 * np.abs(share - 100 * target)
        drill = dumped * (c.drill_blast_per_t + c.drill_blast_fine_extra_per_t * (1 - frag))
        parked_share = np.clip(t / sim.shift_s, 0, 1) * parked_s / 3600 * (1 - c.excavator_idle_share)
        h["cost"] = (fixed_per_h * t / 3600 - c.excavator_per_h * parked_share
                     + h["fuel_l"] * c.fuel_per_l + h["kwh"] * c.energy_per_kwh + drill
                     + dumped * c.blend_penalty_per_t_pp * dev
                     + (h["stockpile_in_t"] * sc.stockpile.rehandle_cost_per_t if sc.stockpile.enabled else 0.0))
        h["cost_per_t"] = h["cost"] / h["dumped_t"].where(h["dumped_t"] > 0)
        h["fuel_l_per_t"] = h["fuel_l"] / h["dumped_t"].where(h["dumped_t"] > 0)
        return h

    def _kpis(self, sim: Simulation) -> dict:
        sc, shift_s = sim.sc, sim.shift_s
        loads = self.loads
        done = loads[loads["dumped_at_s"].notna()] if "dumped_at_s" in loads else loads.iloc[0:0]
        dumped = float(done["tonnes"].sum())
        st = self.states.groupby("state")["duration_s"].sum() / 3600 if len(self.states) else pd.Series(dtype=float)
        n_cycles = len(done[["truck", "trip"]].drop_duplicates()) if len(done) else 0

        for ex in sim.excavators.values():  # close parking that lasted until the end of the shift
            if ex.parked and ex.parked_since is not None:
                ex.parked_s += shift_s - ex.parked_since
                ex.parked_since = shift_s
        parked_h = sum(ex.parked_s for ex in sim.excavators.values()) / 3600
        # feed mix achieved vs target (on dumped tonnes)
        feed = done.groupby("blast")["tonnes"].sum() if len(done) else pd.Series(dtype=float)
        feed_share = {b.id: round(100 * float(feed.get(b.id, 0.0)) / dumped, 1) if dumped else 0.0 for b in sc.blasts}
        feed_target = {k: round(100 * v, 1) for k, v in sim.targets.items()}
        feed_dev_pp = round(0.5 * sum(abs(feed_share[k] - feed_target[k]) for k in feed_target), 1) if feed_target else 0.0

        ex_rows = []
        for ex in sim.excavators.values():
            active_s = (ex.depleted_at if ex.depleted_at is not None else shift_s) - ex.parked_s
            work = ex.loading_s + ex.spotting_s
            ex_rows.append({
                "excavator": ex.cfg.id,
                "loaded_t": round(float(loads.loc[loads["excavator"] == ex.cfg.id, "tonnes"].sum()) if len(loads) else 0.0, 1),
                "remaining_t": round(ex.remaining_t, 1),
                "utilization": round(work / active_s, 3) if active_s else 0.0,
                "waiting_for_trucks_h": round((active_s - work) / 3600, 2),
                "blast_done_at_h": round(ex.depleted_at / 3600, 2) if ex.depleted_at else None,
                "parked_h": round(ex.parked_s / 3600, 2),
                "topup_skips": ex.topup_skips,
            })

        h, c = sc.shift_hours, sc.costs
        frag_loaded = float((done["tonnes"] * done["fragmentation"]).sum() / dumped) if dumped else 0.0
        cost = {
            "wozidla": sc.truck_count * c.truck_per_h * h,
            "paliwo": sim.fuel_l * c.fuel_per_l,
            # a parked excavator still costs its operator, but no fuel / wear
            "koparki": c.excavator_per_h * (len(sc.excavators) * h - parked_h * (1 - c.excavator_idle_share)),
            "kruszarka": c.crusher_per_h * h * len(sc.crushers),
            "energia": sim.crusher_kwh * c.energy_per_kwh,
            # everything put on the stockpile will be rehandled at some point – charge it now
            "halda_ladowarka": (sc.stockpile.loader_per_h * h + sc.stockpile.rehandle_cost_per_t
                                * (sim.stockpile_in_t + min(sim.rehandled_t, sc.stockpile.initial_t)))
            if sc.stockpile.enabled else 0.0,
            "wiercenie_strzelanie": dumped * drill_blast_cost_per_t(c, frag_loaded),
            "odchylka_skladu": dumped * c.blend_penalty_per_t_pp * feed_dev_pp,
        }
        total = sum(cost.values())
        return {
            "trucks": sc.truck_count,
            "dispatch": sc.dispatch,
            "rain": sim.rain,
            "hauled_t": round(dumped, 1),
            "crushed_t": round(sim.crushed_t, 1),
            "throughput_tph": round(dumped / h, 1),
            "crushed_tph": round(sim.crushed_t / h, 1),
            "cycles": n_cycles,
            "decisions": sim.decisions,
            "yields": sim.yields,
            "yield_h": round(sim.yield_s / 3600, 2),
            "bay_stops": sim.bay_stops,
            "relocations": sum(e.relocations for e in sim.excavators.values()),
            "bay_wait_h": round(float(st.get("mijanka", 0.0)), 2),
            "other_delay_h": round(float(st.get("inne_przestoje", 0.0)), 2),
            "spot_exit_h": round(float(st.get("podstawienie", 0.0)) + float(st.get("wyjazd_z_frontu", 0.0)), 2),
            "over_limit_pct": round(100 * sim.over_limit_segments / max(sim.drive_segments, 1), 1),
            "speeders": sum(t.speeder for t in sim.trucks),
            "prefills": sum(e.prefills for e in sim.excavators.values()),
            "truck_refuels": sim.truck_refuels,
            "truck_break_h": round(float(st.get("przerwa", 0.0)), 2),
            "truck_blasting_h": round(float(st.get("postoj_strzelanie", 0.0)), 2),
            "truck_excavator_down_h": round(float(st.get("postoj_koparki", 0.0)), 2),
            "excavator_down_h": {r: round(sum(max(0.0, min(b, shift_s) - a) for e in sim.excavators.values()
                                              for a, b, rr in e.down if rr == r) / 3600, 2)
                                 for r in ("przerwa", "strzelanie", "tankowanie", "przestawienie", "przejazd", "awaria")},
            "truck_queue_excavator_h": round(float(st.get("kolejka_koparka", 0.0)), 2),
            "truck_queue_crusher_h": round(float(st.get("kolejka_kruszarka", 0.0)), 2),
            "truck_breakdown_h": round(float(st.get("awaria", 0.0)), 2),
            "truck_idle_no_material_h": round(float(st.get("postoj_brak_urobku", 0.0)), 2),
            "avg_queue_per_cycle_min": round(60 * float(st.get("kolejka_koparka", 0.0) + st.get("kolejka_kruszarka", 0.0)) / max(n_cycles, 1), 2),
            "crusher_utilization": round(sim.crusher_busy_s / shift_s / len(sim.units), 3),
            "crushers": [{"crusher": u.cfg.id, "crushed_t": round(u.crushed_t, 1),
                          "utilization": round(u.busy_s / shift_s, 3), "starved_h": round(u.starved_s / 3600, 2),
                          "accepts": list(u.cfg.accepts), "product": u.cfg.product, "target_t": u.cfg.target_t,
                          "plan_pct": round(100 * u.crushed_t / u.cfg.target_t, 1) if u.cfg.target_t else None,
                          "capacity_tph": u.cfg.capacity_tph, "kwh": round(u.kwh),
                          "kwh_per_t": round(u.kwh / u.crushed_t, 2) if u.crushed_t else None,
                          "down_h": round(u.down_s / 3600, 2)}
                         for u in sim.units.values()],
            "crusher_breakdown_h": round(sum(u.down_s for u in sim.units.values()) / 3600, 2),
            # repairs that go on into the next shift: id -> seconds left at the end of this one
            "carry_down": {k_: round(t_ - shift_s, 1) for k_, t_ in
                           [(u.cfg.id, u.down_until) for u in sim.units.values()]
                           + [(e.cfg.id, e.down_end) for e in sim.excavators.values()]
                           + [(t.id, t.down_end) for t in sim.trucks] if t_ > shift_s},
            "crusher_starved_h": round(sim.crusher_starved_s / 3600, 2),
            "stockpile_in_t": round(sim.stockpile_in_t, 1),
            "stockpile_end_t": round(sim.stockpile_t, 1),
            "fuel_l": round(sim.fuel_l, 0),
            "fuel_l_per_t": round(sim.fuel_l / dumped, 3) if dumped else None,
            "crusher_kwh_per_t": round(sim.crusher_kwh / sim.crushed_t, 3) if sim.crushed_t else None,
            "crusher_kwh": round(sim.crusher_kwh, 1),
            "cost_total": round(total, 0),
            "cost_per_t": round(total / dumped, 2) if dumped else None,
            "cost_breakdown": {k: round(v, 0) for k, v in cost.items()},
            "parked_excavator_h": round(parked_h, 2),
            "topup_skips": sum(e.topup_skips for e in sim.excavators.values()),
            "feed_share": feed_share,
            "feed_target": feed_target,
            "feed_dev_pp": feed_dev_pp,
            "excavators": ex_rows,
        }


def simulate(sc: Scenario, seed: int | None = None) -> SimResult:
    return Simulation(sc, seed).run()
