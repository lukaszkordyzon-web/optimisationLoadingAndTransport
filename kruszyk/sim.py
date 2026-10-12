"""Discrete-event simulation of the load-haul-crush chain (digital twin).

Produces synthetic data (segment travel times, truck states, loads, hopper level)
in the same shape the phone app will later deliver from real GPS geofences.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

from .config import Excavator, Scenario, TruckFleet
from .physics import crusher_rate_tph, load_time_s, route_time_s, segment_time_s

CRUSHER_TICK_S = 10.0


@dataclass
class TruckState:
    id: str
    fleet: TruckFleet
    driver_factor: float
    home_excavator: str  # used by "fixed" dispatch
    trip: int = 0


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


class Simulation:
    def __init__(self, sc: Scenario, seed: int | None = None):
        sc.validate()
        max_payload = max(f.payload_t for f in sc.trucks)
        if sc.crusher.hopper_t < max_payload:
            raise ValueError("Kosz kruszarki mniejszy niż ładowność wozidła")
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
        ex_ids = list(self.excavators)
        # driver/truck style is persistent across shifts (drawn from the scenario seed, not the shift seed)
        crew_rng = np.random.default_rng(sc.seed)
        i = 0
        for fleet in sc.trucks:
            for _ in range(fleet.count):
                self.trucks.append(TruckState(
                    id=f"WZ-{i + 1:02d}",
                    fleet=fleet,
                    driver_factor=float(np.clip(crew_rng.normal(1.0, v.driver_sd), 0.8, 1.3)),
                    home_excavator=ex_ids[i % len(ex_ids)],
                ))
                i += 1

        self.hopper = simpy.Container(self.env, capacity=sc.crusher.hopper_t, init=0)
        self.dump_point = simpy.Resource(self.env, 1)
        self.hopper_frag = 0.0
        self.crushed_t = 0.0
        self.crusher_busy_s = 0.0
        self.crusher_starved_s = 0.0

        self.segments: list[dict] = []
        self.states: list[dict] = []
        self.loads: list[dict] = []
        self.hopper_log: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _log_state(self, truck: TruckState, state: str, start: float, ex: str | None = None):
        if self.env.now > start:
            self.states.append({"truck": truck.id, "state": state, "start_s": start,
                                "end_s": self.env.now, "duration_s": self.env.now - start,
                                "excavator": ex, "trip": truck.trip})

    def _noise(self, sd: float) -> float:
        return float(self.rng.lognormal(-sd * sd / 2, sd))

    def _expected_service(self, ex: ExcavatorState, fleet: TruckFleet) -> float:
        blast = self.sc.blast(ex.cfg.blast)
        return ex.cfg.spot_time_s + load_time_s(ex.cfg, fleet, blast, self.sc.loose_density_t_m3)

    def _active(self, ex: ExcavatorState) -> bool:
        return ex.remaining_t - ex.committed_t > 1e-6

    # --------------------------------------------------------------- dispatcher
    def predicted_finish(self, ex: ExcavatorState, arrival: float, service: float) -> float:
        """When would a truck arriving at `arrival` be loaded, given the trucks already sent there."""
        t = max(self.env.now, ex.busy_until)
        for arr, srv in sorted(ex.pending.values()):
            t = max(t, arr) + srv
        return max(t, arrival) + service

    def dispatch(self, truck: TruckState) -> ExcavatorState | None:
        active = [ex for ex in self.excavators.values() if self._active(ex)]
        if not active:
            return None
        if self.sc.dispatch == "fixed":
            home = self.excavators[truck.home_excavator]
            if self._active(home):
                return home
        best, best_finish = None, math.inf
        for ex in active:
            travel = route_time_s(self.sc.routes[ex.cfg.route], truck.fleet, False,
                                  self.sc.rolling_resistance_pct)
            finish = self.predicted_finish(ex, self.env.now + travel, self._expected_service(ex, truck.fleet))
            if finish < best_finish:
                best, best_finish = ex, finish
        return best

    # ---------------------------------------------------------------- processes
    def _drive(self, truck: TruckState, ex: ExcavatorState, loaded: bool):
        route = self.sc.routes[ex.cfg.route]
        segs = route.segments if loaded else list(reversed(route.segments))
        payload = truck.fleet.payload_t if loaded else 0.0
        start_state = self.env.now
        for seg in segs:
            mean = segment_time_s(seg, truck.fleet, loaded, self.sc.rolling_resistance_pct)
            dur = mean * truck.driver_factor * self.weather_factor * self._noise(self.sc.variability.trip_sd)
            start = self.env.now
            yield self.env.timeout(dur)
            self.segments.append({
                "truck": truck.id, "truck_model": truck.fleet.model, "trip": truck.trip,
                "excavator": ex.cfg.id, "route": route.id, "segment": seg.name,
                "loaded": int(loaded), "payload_t": payload, "length_m": seg.length_m,
                "grade_pct": seg.grade_pct if loaded else -seg.grade_pct,
                "speed_limit_kmh": seg.speed_limit_kmh, "rain": int(self.rain),
                "hour_of_shift": start / 3600, "start_s": start, "duration_s": dur,
                "physics_s": mean,
            })
        self._log_state(truck, "jazda_pelny" if loaded else "jazda_pusty", start_state, ex.cfg.id)

    def truck_process(self, truck: TruckState):
        env, v = self.env, self.sc.variability
        while True:
            ex = self.dispatch(truck)
            if ex is None:
                # logged up front: events exactly at shift end are not processed by env.run(until=...)
                self.states.append({"truck": truck.id, "state": "postoj_brak_urobku", "start_s": env.now,
                                    "end_s": self.shift_s, "duration_s": self.shift_s - env.now,
                                    "excavator": None, "trip": truck.trip})
                return
            truck.trip += 1
            fleet = truck.fleet
            ex.committed_t += fleet.payload_t
            travel = route_time_s(self.sc.routes[ex.cfg.route], fleet, False, self.sc.rolling_resistance_pct)
            ex.pending[truck.id] = (env.now + travel, self._expected_service(ex, fleet))
            cycle_start = env.now

            yield from self._drive(truck, ex, loaded=False)

            t0 = env.now
            with ex.resource.request() as req:
                yield req
                self._log_state(truck, "kolejka_koparka", t0, ex.cfg.id)
                ex.pending.pop(truck.id, None)
                ex.busy_until = env.now + self._expected_service(ex, fleet)
                tonnes = min(fleet.payload_t, ex.remaining_t)
                ex.remaining_t -= tonnes
                ex.committed_t -= fleet.payload_t

                t0 = env.now
                yield env.timeout(ex.cfg.spot_time_s * self._noise(0.2))
                ex.spotting_s += env.now - t0
                self._log_state(truck, "podstawienie", t0, ex.cfg.id)

                blast = self.sc.blast(ex.cfg.blast)
                full_load = load_time_s(ex.cfg, fleet, blast, self.sc.loose_density_t_m3)
                t0 = env.now
                yield env.timeout(full_load * (tonnes / fleet.payload_t) * self._noise(v.load_sd))
                ex.loading_s += env.now - t0
                self._log_state(truck, "zaladunek", t0, ex.cfg.id)
                self.loads.append({"truck": truck.id, "excavator": ex.cfg.id, "blast": blast.id,
                                   "tonnes": tonnes, "loaded_at_s": env.now, "trip": truck.trip})
                if ex.remaining_t <= 1e-6 and ex.depleted_at is None:
                    ex.depleted_at = env.now

            yield from self._drive(truck, ex, loaded=True)

            t0 = env.now
            with self.dump_point.request() as req:
                yield req
                yield self.hopper.put(tonnes)  # waits while the hopper is full
                self._log_state(truck, "kolejka_kruszarka", t0)
                level = self.hopper.level
                self.hopper_frag = ((level - tonnes) * self.hopper_frag + tonnes * blast.fragmentation) / max(level, 1e-9)
                t0 = env.now
                yield env.timeout(self.sc.crusher.dump_time_s * self._noise(0.15))
                self._log_state(truck, "wyladunek", t0)
            self._mark_dumped(truck, env.now)

            # random breakdown, probability proportional to cycle length
            cycle = env.now - cycle_start
            if self.rng.random() < 1 - math.exp(-cycle / (v.truck_mtbf_h * 3600)):
                t0 = env.now
                yield env.timeout(self.rng.exponential(v.truck_mttr_min * 60))
                self._log_state(truck, "awaria", t0)

    def _mark_dumped(self, truck: TruckState, t: float):
        for load in reversed(self.loads):
            if load["truck"] == truck.id and load["trip"] == truck.trip:
                load["dumped_at_s"] = t
                return

    def crusher_process(self):
        env = self.env
        while True:
            rate = crusher_rate_tph(self.sc.crusher, self.hopper_frag)
            capacity = rate * CRUSHER_TICK_S / 3600
            amount = min(self.hopper.level, capacity)
            if amount > 1e-9:
                yield self.hopper.get(amount)
                self.crushed_t += amount
                self.crusher_busy_s += CRUSHER_TICK_S * amount / capacity
                self.crusher_starved_s += CRUSHER_TICK_S * (1 - amount / capacity)
            else:
                self.crusher_starved_s += CRUSHER_TICK_S
            if int(env.now) % 60 == 0:
                self.hopper_log.append({"t_s": env.now, "level_t": self.hopper.level})
            yield env.timeout(CRUSHER_TICK_S)

    # ---------------------------------------------------------------------- run
    def run(self) -> "SimResult":
        self.env.process(self.crusher_process())
        for i, truck in enumerate(self.trucks):
            self.env.process(self._delayed_start(truck, i * 30.0))  # stagger departures
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
        self.kpi = self._kpis(sim)

    def _kpis(self, sim: Simulation) -> dict:
        sc, shift_s = sim.sc, sim.shift_s
        loads = self.loads
        dumped = loads.loc[loads["dumped_at_s"].notna(), "tonnes"].sum() if "dumped_at_s" in loads else 0.0
        st = self.states.groupby("state")["duration_s"].sum() / 3600 if len(self.states) else pd.Series(dtype=float)
        n_cycles = int(loads["dumped_at_s"].notna().sum()) if "dumped_at_s" in loads else 0

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
        hours = sc.shift_hours
        cost = (sc.truck_count * sc.costs.truck_per_h + len(sc.excavators) * sc.costs.excavator_per_h
                + sc.costs.crusher_per_h) * hours
        return {
            "trucks": sc.truck_count,
            "dispatch": sc.dispatch,
            "rain": sim.rain,
            "hauled_t": round(float(dumped), 1),
            "crushed_t": round(sim.crushed_t, 1),
            "throughput_tph": round(float(dumped) / hours, 1),
            "cycles": n_cycles,
            "truck_queue_excavator_h": round(float(st.get("kolejka_koparka", 0.0)), 2),
            "truck_queue_crusher_h": round(float(st.get("kolejka_kruszarka", 0.0)), 2),
            "truck_breakdown_h": round(float(st.get("awaria", 0.0)), 2),
            "truck_idle_no_material_h": round(float(st.get("postoj_brak_urobku", 0.0)), 2),
            "avg_queue_per_cycle_min": round(60 * float(st.get("kolejka_koparka", 0.0) + st.get("kolejka_kruszarka", 0.0)) / max(n_cycles, 1), 2),
            "crusher_utilization": round(sim.crusher_busy_s / shift_s, 3),
            "crusher_starved_h": round(sim.crusher_starved_s / 3600, 2),
            "cost_per_t": round(cost / dumped, 2) if dumped else None,
            "excavators": ex_rows,
        }


def simulate(sc: Scenario, seed: int | None = None) -> SimResult:
    return Simulation(sc, seed).run()
