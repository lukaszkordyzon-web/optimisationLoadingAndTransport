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

from .config import Excavator, Scenario, TruckFleet, clock_to_s
from .physics import (crusher_kwh_per_t, crusher_rate_tph, drill_blast_cost_per_t, load_time_s,
                      route_time_s, segment_time_s)

CRUSHER_TICK_S = 10.0
LOADER_TICK_S = 30.0
DRIVE_STATES = {"jazda_pusty", "przejazd_miedzy_usypami", "jazda_pelny"}
IDLE_STATES = {"kolejka_koparka", "podstawienie", "zaladunek", "kolejka_kruszarka", "wyladunek",
               "wyladunek_halda", "postoj_koparki"}
CONNECTORS = {"front", "plac_kruszarki", "dojazd"}  # short links to the network, not real roads
PREFILL_NATURAL_P = 0.3  # without information the operator sometimes guesses and fills the bucket in time


@dataclass
class TruckState:
    id: str
    fleet: TruckFleet
    driver_factor: float
    home_excavator: str  # used by "fixed" dispatch
    trip: int = 0
    speeder: bool = False  # drives above the limit (site practice)
    fuel_left_l: float = 0.0


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
    down: list = field(default_factory=list)  # (from, to, reason): breaks, blasting, refuelling


class Simulation:
    def __init__(self, sc: Scenario, seed: int | None = None):
        sc.validate()
        self.sc = sc
        self.rng = np.random.default_rng(sc.seed if seed is None else seed)
        self.env = simpy.Environment()
        self.shift_s = sc.shift_hours * 3600
        v = sc.variability
        self.rain = bool(self.rng.random() < v.rain_probability)
        self.weather_factor = 1 + (v.rain_slowdown if self.rain else 0.0)

        self.excavators = {
            e.id: ExcavatorState(e, simpy.Resource(self.env, 1), sc.blast(e.blast).tonnage_t)
            for e in sc.excavators
        }
        # several excavators on the same blast share its tonnage
        by_blast: dict[str, list[ExcavatorState]] = {}
        for ex in self.excavators.values():
            by_blast.setdefault(ex.cfg.blast, []).append(ex)
        for exs in by_blast.values():
            for ex in exs:
                ex.remaining_t /= len(exs)

        self.trucks: list[TruckState] = []
        # driver/truck style is persistent across shifts (drawn from the scenario seed, not the shift seed)
        crew_rng = np.random.default_rng(sc.seed)
        i = 0
        for fleet in sc.trucks:
            allowed = self.allowed_ids(fleet)
            for j in range(fleet.count):
                self.trucks.append(TruckState(
                    id=f"WZ-{i + 1:02d}",
                    fleet=fleet,
                    driver_factor=float(np.clip(crew_rng.normal(1.0, v.driver_sd), 0.8, 1.3)),
                    home_excavator=allowed[(i + j) % len(allowed)],
                    speeder=bool(crew_rng.random() < v.speeding_share),
                    fuel_left_l=fleet.tank_l,
                ))
                i += 1

        self.hopper = simpy.Container(self.env, capacity=sc.crusher.hopper_t, init=0)
        self.dump_point = simpy.Resource(self.env, 1)
        self.hopper_frag = 0.0
        self.crushed_t = 0.0
        self.crusher_busy_s = 0.0
        self.crusher_starved_s = 0.0
        self.crusher_kwh = 0.0
        self.queued_dump_t = 0.0   # tonnes waiting at the dump point
        self.inbound_t = 0.0       # tonnes on loaded trucks driving to the crusher
        self.stockpile_t = sc.stockpile.initial_t
        self.stockpile_in_t = 0.0
        self.rehandled_t = 0.0
        self.fuel_l = 0.0
        self.dumped_t = 0.0
        self.decisions = 0
        self.yields, self.yield_s = 0, 0.0
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
        for ex in self.excavators.values():
            ex.last_end = self.t_start
            ex.down += [(a, b, "przerwa") for a, b in self.breaks]
            if self.closure:
                ex.down.append((*self.closure, "strzelanie"))
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
                                "excavator": ex, "trip": truck.trip})
            if state in IDLE_STATES:
                used = truck.fleet.fuel_lph_idle * (end - start) / 3600
                self.fuel_l += used
                truck.fuel_left_l -= used

    def _noise(self, sd: float) -> float:
        return float(self.rng.lognormal(-sd * sd / 2, sd))

    def _expected_service(self, ex: ExcavatorState, fleet: TruckFleet) -> float:
        blast = self.sc.blast(ex.cfg.blast)
        return ex.cfg.spot_time_s + load_time_s(ex.cfg, fleet, blast, self.sc.loose_density_t_m3)

    def _active(self, ex: ExcavatorState) -> bool:
        return ex.remaining_t - ex.committed_t > 1e-6

    def _crusher_rate(self) -> float:
        return crusher_rate_tph(self.sc.crusher, self.hopper_frag)

    def expected_dump_wait_s(self, tonnes: float, extra_t: float = 0.0, ahead_s: float = 0.0) -> float:
        """Expected wait at the hopper for `tonnes`, if `extra_t` more arrives first and the
        crusher keeps draining for `ahead_s` seconds before we get there."""
        rate = self._crusher_rate()
        backlog = self.hopper.level + self.queued_dump_t + extra_t - rate * ahead_s / 3600
        missing = max(0.0, backlog + tonnes - self.sc.crusher.hopper_t)
        hopper_wait = missing / rate * 3600
        queue_wait = (len(self.dump_point.queue) + len(self.dump_point.users)) * self.sc.crusher.dump_time_s
        return max(hopper_wait, queue_wait if extra_t == 0 else 0.0)

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
                t = self._light_trip(name, sc.routes[ex.cfg.route], False, t, frames)
            else:
                link = sc.links.get(f"{prev}>{x}")
                if link is not None:
                    t = self._light_trip(name, link, True, t, frames)
                else:  # no link drawn: back to the crusher and down again
                    t = self._light_trip(name, sc.routes[self.excavators[prev].cfg.route], True, t, frames)
                    t = self._light_trip(name, sc.routes[ex.cfg.route], False, t, frames)
            if refuel:
                ex.down.append((t, t + stop_s, "tankowanie"))
            if frames:
                frames.append([round(t + stop_s, 1), frames[-1][1], frames[-1][2]])
            t += stop_s
            prev = x
        t = self._light_trip(name, sc.routes[self.excavators[prev].cfg.route], True, t, frames)
        self.light_vehicles.append({"name": name, "kind": kind, "frames": frames})

    def _schedule_light_vehicles(self) -> None:
        sc = self.sc
        start = sc.shift_plan.start_time
        bl = sc.blasting
        site = sc.site or {}
        if bl.enabled and site.get("blast") and site.get("roads") and site.get("crusher"):
            from .config import _routes
            from .site import build_route

            try:
                r = build_route(site, (site["blast"]["lat"], site["blast"]["lon"]), "strzelanie")
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
            self._visit_round("sztygar", "objazd", clock_to_s(ins.start, start), list(ins.stops), ins.stop_min * 60)
        for i, tt in enumerate(sc.refueling.tanker_times or []):
            self._visit_round(f"cysterna {i + 1}", "cysterna", clock_to_s(tt, start), [e.id for e in sc.excavators],
                              sc.refueling.excavator_refuel_min * 60, refuel=True)

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
        route = sc.routes[ex.cfg.route]
        travel = route_time_s(route, truck.fleet, False, sc.rolling_resistance_pct)
        haul = route_time_s(route, truck.fleet, True, sc.rolling_resistance_pct)
        service = self._expected_service(ex, truck.fleet)
        wait = self.predicted_finish(ex, self.env.now + travel, service) - (self.env.now + travel) - service
        initial = max(sc.blast(ex.cfg.blast).tonnage_t, 1.0)
        return [max(wait, 0.0) / 60, travel / 60, haul / 60, service / 60,
                (ex.remaining_t - ex.committed_t) / initial, float(len(ex.pending)),
                float(ex.cfg.id == truck.home_excavator)]

    def dispatch(self, truck: TruckState) -> ExcavatorState | None:
        allowed = set(self.allowed_ids(truck.fleet))
        active = [ex for ex in self.excavators.values() if ex.cfg.id in allowed and self._active(ex)]
        if not active:
            return None
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
            return min(active, key=lambda ex: float(np.dot(w, self.features(truck, ex))))
        best, best_finish = None, math.inf
        for ex in active:
            travel = route_time_s(self.sc.routes[ex.cfg.route], truck.fleet, False,
                                  self.sc.rolling_resistance_pct)
            finish = self.predicted_finish(ex, self.env.now + travel, self._expected_service(ex, truck.fleet))
            if finish < best_finish:
                best, best_finish = ex, finish
        return best

    # ---------------------------------------------------------------- processes
    def _drive(self, truck: TruckState, route, forward: bool, load_t: float, leg: str, ex_id: str,
               pace: float = 1.0):
        """Drive a whole route (forward = towards its end). leg: pusty | miedzy | pelny."""
        idx = range(len(route.segments)) if forward else reversed(range(len(route.segments)))
        f = truck.fleet
        share = load_t / f.payload_t if f.payload_t else 0.0
        fuel_rate = f.fuel_lph_empty + (f.fuel_lph_loaded - f.fuel_lph_empty) * share
        start_state = self.env.now
        v = self.sc.variability
        for k in idx:
            seg = route.segments[k]
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
            dur = natural * pace
            start = self.env.now
            grade = seg.grade_pct if forward else -seg.grade_pct
            n_y, delay = self._yield_delay(seg.name, start, start + dur, grade > 0.5)
            if n_y:
                self.yields += n_y
                self.yield_s += delay
            yield self.env.timeout(dur + delay)
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

    def _load_at(self, truck: TruckState, ex: ExcavatorState, amount: float, committed: float):
        """Queue, spot and load `amount` t at `ex`. Returns tonnes actually loaded."""
        env, sc, fleet = self.env, self.sc, truck.fleet
        t0 = env.now
        with ex.resource.request() as req:
            yield req
            self._log_state(truck, "kolejka_koparka", t0, ex.cfg.id)
            ex.pending.pop(truck.id, None)
            ex.busy_until = env.now + self._expected_service(ex, fleet)
            tonnes = min(amount, ex.remaining_t)
            ex.remaining_t -= tonnes
            ex.committed_t -= committed
            if tonnes <= 1e-6:
                return 0.0
            t0 = env.now
            yield from self._excavator_wait(ex)
            if env.now > t0:
                self._log_state(truck, "postoj_koparki", t0, ex.cfg.id)
            idle_before = env.now - ex.last_end  # how long the excavator had nothing to load

            t0 = env.now
            yield env.timeout(ex.cfg.spot_time_s * self._noise(0.2))
            ex.spotting_s += env.now - t0
            self._log_state(truck, "podstawienie", t0, ex.cfg.id)

            blast = sc.blast(ex.cfg.blast)
            full_load = load_time_s(ex.cfg, fleet, blast, sc.loose_density_t_m3)
            one_pass = ex.cfg.swing_cycle_s * (1.0 + 0.5 * blast.fragmentation)
            load = full_load * (tonnes / fleet.payload_t) * self._noise(sc.variability.load_sd)
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
            self.loads.append({"truck": truck.id, "excavator": ex.cfg.id, "blast": blast.id,
                               "material": blast.material, "tonnes": tonnes, "fragmentation": blast.fragmentation,
                               "loaded_at_s": env.now, "trip": truck.trip})
            if ex.remaining_t <= 1e-6 and ex.depleted_at is None:
                ex.depleted_at = env.now
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
        r = sc.routes[ex.cfg.route]
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
            if self.closure and now < self.closure[1] and now + 1.3 * to_loaded + 120 > self.closure[0]:
                yield env.timeout(self.closure[1] - now)  # nobody goes down before the blast is over
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
            travel = route_time_s(sc.routes[first.cfg.route], fleet, False, sc.rolling_resistance_pct)
            service = self._expected_service(first, fleet)
            pace_empty = 1.0
            if sc.advisor.enabled:
                # recommended speed: arrive when the excavator is free – not earlier (queue), not later
                wait = self.predicted_finish(first, env.now + travel, service) - service - (env.now + travel)
                pace_empty = 1 + min(sc.advisor.max_slowdown, max(0.0, wait) / travel)
            first.pending[truck.id] = (env.now + travel * pace_empty, service)
            cycle_start = env.now

            yield from self._drive(truck, sc.routes[first.cfg.route], False, 0.0, "pusty", first.cfg.id, pace_empty)
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

            # speed advice: if we would only queue at the crusher, drive slower instead
            route = sc.routes[last.cfg.route]
            pace = 1.0
            if (sc.pacing.enabled and sc.dispatch != "natural") or sc.advisor.enabled:
                haul = route_time_s(route, fleet, True, sc.rolling_resistance_pct)
                wait = self.expected_dump_wait_s(tonnes, extra_t=self.inbound_t, ahead_s=haul)
                pace = 1 + min(sc.pacing.max_slowdown, wait / haul)
            self.inbound_t += tonnes
            yield from self._drive(truck, route, True, tonnes, "pelny", last.cfg.id, pace)
            self.inbound_t -= tonnes

            sp = sc.stockpile
            t0 = env.now
            if sp.enabled and self.expected_dump_wait_s(tonnes) > sp.divert_wait_s:
                yield env.timeout(sp.dump_time_s * self._noise(0.15))
                self.stockpile_t += tonnes
                self.stockpile_in_t += tonnes
                self._log_state(truck, "wyladunek_halda", t0)
                self._mark_dumped(truck, env.now, "halda")
            else:
                self.queued_dump_t += tonnes
                with self.dump_point.request() as req:
                    yield req
                    yield self.hopper.put(tonnes)  # waits while the hopper is full
                    self.queued_dump_t -= tonnes
                    self._log_state(truck, "kolejka_kruszarka", t0)
                    self._mix_into_hopper(tonnes, frag)
                    t0 = env.now
                    yield env.timeout(sc.crusher.dump_time_s * self._noise(0.15))
                    self._log_state(truck, "wyladunek", t0)
                self._mark_dumped(truck, env.now, "kosz")
            self.dumped_t += tonnes

            # fuel: trucks start full; below the reserve they go to the fuel station next to the crusher
            if truck.fuel_left_l < fleet.tank_l * sc.refueling.truck_reserve_pct / 100:
                t0 = env.now
                yield env.timeout(sc.refueling.truck_refuel_min * 60 * self._noise(0.15))
                self._log_state(truck, "tankowanie", t0)
                truck.fuel_left_l = fleet.tank_l
                self.truck_refuels += 1

            # random breakdown, probability proportional to cycle length
            cycle = env.now - cycle_start
            if self.rng.random() < 1 - math.exp(-cycle / (v.truck_mtbf_h * 3600)):
                t0 = env.now
                yield env.timeout(self.rng.exponential(v.truck_mttr_min * 60))
                self._log_state(truck, "awaria", t0)

    def _mix_into_hopper(self, tonnes: float, frag: float):
        level = self.hopper.level
        self.hopper_frag = ((level - tonnes) * self.hopper_frag + tonnes * frag) / max(level, 1e-9)

    def _mark_dumped(self, truck: TruckState, t: float, where: str):
        for load in reversed(self.loads):
            if load["truck"] == truck.id and load["trip"] == truck.trip:
                load["dumped_at_s"] = t
                load["dumped_to"] = where
            elif load["truck"] == truck.id:
                return

    def crusher_process(self):
        env = self.env
        while True:
            rate = self._crusher_rate()
            capacity = rate * CRUSHER_TICK_S / 3600
            amount = min(self.hopper.level, capacity)
            if amount > 1e-9:
                yield self.hopper.get(amount)
                self.crushed_t += amount
                self.crusher_kwh += amount * crusher_kwh_per_t(self.sc.crusher, self.hopper_frag)
                self.crusher_busy_s += CRUSHER_TICK_S * amount / capacity
                self.crusher_starved_s += CRUSHER_TICK_S * (1 - amount / capacity)
            else:
                self.crusher_starved_s += CRUSHER_TICK_S
            if int(env.now) % 60 == 0:
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
            free = self.hopper.capacity - self.hopper.level - self.queued_dump_t
            if self.stockpile_t > 1e-6 and self.hopper.level < sp.feed_below_t and free > 1:
                amount = min(self.stockpile_t, sp.loader_tph * LOADER_TICK_S / 3600, free)
                yield self.hopper.put(amount)
                self._mix_into_hopper(amount, frag)
                self.stockpile_t -= amount
                self.rehandled_t += amount
            yield self.env.timeout(LOADER_TICK_S)

    # ---------------------------------------------------------------------- run
    def run(self) -> "SimResult":
        self.env.process(self.crusher_process())
        if self.sc.stockpile.enabled:
            self.env.process(self.loader_process())
        for i, truck in enumerate(self.trucks):
            self.env.process(self._delayed_start(truck, self.t_start + i * 30.0))  # start-up, staggered departures
        self.env.run(until=self.shift_s)
        return SimResult(self)

    def _delayed_start(self, truck: TruckState, delay: float):
        yield self.env.timeout(delay)
        yield from self.truck_process(truck)


class SimResult:
    def __init__(self, sim: Simulation):
        self.scenario = sim.sc
        self.rain = sim.rain
        self.segments = pd.DataFrame(sim.segments)
        self.states = pd.DataFrame(sim.states, columns=["truck", "state", "start_s", "end_s",
                                                        "duration_s", "excavator", "trip"])
        self.loads = pd.DataFrame(sim.loads)
        self.hopper = pd.DataFrame(sim.hopper_log)
        self.truck_ids = [t.id for t in sim.trucks]
        self.light_vehicles = sim.light_vehicles
        self.excavator_down = {e.cfg.id: list(e.down) for e in sim.excavators.values()}
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
        fixed_per_h = (sc.truck_count * c.truck_per_h + len(sc.excavators) * c.excavator_per_h + c.crusher_per_h
                       + (sc.stockpile.loader_per_h if sc.stockpile.enabled else 0.0))
        frag = float(np.mean([b.fragmentation for b in sc.blasts]))
        h["cost"] = (fixed_per_h * h["t_s"] / 3600 + h["fuel_l"] * c.fuel_per_l + h["kwh"] * c.energy_per_kwh
                     + h["dumped_t"] * drill_blast_cost_per_t(c, frag)
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

        ex_rows = []
        for ex in sim.excavators.values():
            active_s = ex.depleted_at if ex.depleted_at is not None else shift_s
            work = ex.loading_s + ex.spotting_s
            ex_rows.append({
                "excavator": ex.cfg.id,
                "loaded_t": round(float(loads.loc[loads["excavator"] == ex.cfg.id, "tonnes"].sum()) if len(loads) else 0.0, 1),
                "remaining_t": round(ex.remaining_t, 1),
                "utilization": round(work / active_s, 3) if active_s else 0.0,
                "waiting_for_trucks_h": round((active_s - work) / 3600, 2),
                "blast_done_at_h": round(ex.depleted_at / 3600, 2) if ex.depleted_at else None,
            })

        h, c = sc.shift_hours, sc.costs
        frag_loaded = float((done["tonnes"] * done["fragmentation"]).sum() / dumped) if dumped else 0.0
        cost = {
            "wozidla": sc.truck_count * c.truck_per_h * h,
            "paliwo": sim.fuel_l * c.fuel_per_l,
            "koparki": len(sc.excavators) * c.excavator_per_h * h,
            "kruszarka": c.crusher_per_h * h,
            "energia": sim.crusher_kwh * c.energy_per_kwh,
            # everything put on the stockpile will be rehandled at some point – charge it now
            "halda_ladowarka": (sc.stockpile.loader_per_h * h + sc.stockpile.rehandle_cost_per_t
                                * (sim.stockpile_in_t + min(sim.rehandled_t, sc.stockpile.initial_t)))
            if sc.stockpile.enabled else 0.0,
            "wiercenie_strzelanie": dumped * drill_blast_cost_per_t(c, frag_loaded),
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
            "over_limit_pct": round(100 * sim.over_limit_segments / max(sim.drive_segments, 1), 1),
            "speeders": sum(t.speeder for t in sim.trucks),
            "prefills": sum(e.prefills for e in sim.excavators.values()),
            "truck_refuels": sim.truck_refuels,
            "truck_break_h": round(float(st.get("przerwa", 0.0)), 2),
            "truck_blasting_h": round(float(st.get("postoj_strzelanie", 0.0)), 2),
            "truck_excavator_down_h": round(float(st.get("postoj_koparki", 0.0)), 2),
            "excavator_down_h": {r: round(sum(max(0.0, min(b, shift_s) - a) for e in sim.excavators.values()
                                              for a, b, rr in e.down if rr == r) / 3600, 2)
                                 for r in ("przerwa", "strzelanie", "tankowanie")},
            "truck_queue_excavator_h": round(float(st.get("kolejka_koparka", 0.0)), 2),
            "truck_queue_crusher_h": round(float(st.get("kolejka_kruszarka", 0.0)), 2),
            "truck_breakdown_h": round(float(st.get("awaria", 0.0)), 2),
            "truck_idle_no_material_h": round(float(st.get("postoj_brak_urobku", 0.0)), 2),
            "avg_queue_per_cycle_min": round(60 * float(st.get("kolejka_koparka", 0.0) + st.get("kolejka_kruszarka", 0.0)) / max(n_cycles, 1), 2),
            "crusher_utilization": round(sim.crusher_busy_s / shift_s, 3),
            "crusher_starved_h": round(sim.crusher_starved_s / 3600, 2),
            "stockpile_in_t": round(sim.stockpile_in_t, 1),
            "stockpile_end_t": round(sim.stockpile_t, 1),
            "fuel_l": round(sim.fuel_l, 0),
            "fuel_l_per_t": round(sim.fuel_l / dumped, 3) if dumped else None,
            "crusher_kwh_per_t": round(sim.crusher_kwh / sim.crushed_t, 3) if sim.crushed_t else None,
            "cost_total": round(total, 0),
            "cost_per_t": round(total / dumped, 2) if dumped else None,
            "cost_breakdown": {k: round(v, 0) for k, v in cost.items()},
            "excavators": ex_rows,
        }


def simulate(sc: Scenario, seed: int | None = None) -> SimResult:
    return Simulation(sc, seed).run()
