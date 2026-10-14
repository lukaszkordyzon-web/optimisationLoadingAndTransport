"""Production goal for a period, planned backwards from the crushers.

Management sets how many tonnes of which products the period must deliver (e.g. 200 000 t: 75 % chippings,
15 % 16–31.5, 10 % mixes). Every crusher line turns one tonne of feed into a known split of products (its yields).
A linear programme then picks, at the lowest cost, how many tonnes go from which pile / extraction area / planned
blast to which crusher – within crusher capacity and the material available. From those flows follow the feed
targets of the crushers, the feed shares of the piles, how much has to be blasted and drilled in every area, and how
many trucks the haulage needs. The period simulation then checks the plan shift by shift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import PRODUCTS, Scenario

# goal items may name a group of products ("grysy" = all chippings)
GROUPS = {
    "grysy (2–16)": ["grys 2–8", "grys 8–16"],
    "mieszanki / podbudowa": ["mieszanka 0–31,5", "kruszywo 0–63 (podbudowa)"],
}
GOAL_ITEMS = list(GROUPS) + list(PRODUCTS)

# typical split of one tonne of feed into products [%] – the rest is undersize / fines not sold
YIELD_PRESETS = {
    "linia grysowa (3 stopnie)": {"grys 2–8": 25.0, "grys 8–16": 30.0, "kliniec 16–31,5": 20.0,
                                  "mieszanka 0–31,5": 10.0, "piasek łamany 0–2": 10.0},
    "linia klińca / tłucznia (2 stopnie)": {"tłuczeń 31,5–63": 30.0, "kliniec 16–31,5": 30.0, "grys 8–16": 15.0,
                                            "mieszanka 0–31,5": 20.0},
    "kruszarka wstępna (podbudowa, narzut)": {"kamień łamany / narzut 63–250": 20.0,
                                              "kruszywo 0–63 (podbudowa)": 75.0},
}
SHORTAGE_COST = 500.0      # zł per tonne of a product the plan can't deliver (makes the LP always solvable)
CRUSHER_UTIL = 0.85        # share of shift time a crusher really crushes (stops, waiting, maintenance)
FLEET_UTIL = 0.80          # share of shift time a truck really hauls
LOADING_UTIL = 0.60        # share of shift time an excavator really loads (waiting for trucks, moves, spotting)
TRUCK_KMH = 20.0           # average speed loaded / empty
TRUCK_FIXED_H = 0.12       # spotting, loading, dumping per cycle
DETOUR = 1.35              # road distance / straight line


def default_yields(product: str) -> dict:
    """Yields of a crusher that so far had only its main product set."""
    for preset in YIELD_PRESETS.values():
        if product and product in preset and max(preset, key=preset.get) == product:
            return dict(preset)
    if product in YIELD_PRESETS["linia grysowa (3 stopnie)"]:
        return dict(YIELD_PRESETS["linia grysowa (3 stopnie)"])
    return {product: 90.0} if product else dict(YIELD_PRESETS["linia grysowa (3 stopnie)"])


def members(item: str) -> list[str]:
    return GROUPS.get(item, [item])


@dataclass
class Source:
    id: str
    kind: str            # "usyp" (lies, blasted) / "obszar" (extraction area) / "strzał" (planned blast)
    supply_t: float
    blast_t: float       # of which still to be blasted (drilling + blasting cost)
    t_per_m: float = 50.0
    blast_max_t: float = 0.0
    pos: tuple | None = None
    route_m: float = 0.0


@dataclass
class GoalPlan:
    total_t: float
    mix: dict                       # goal item -> % of total
    yields: dict                    # crusher id -> {product: %}
    shifts: int                     # shifts in the period
    feasible: bool = True
    products: list = field(default_factory=list)
    crushers: list = field(default_factory=list)
    flows: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    drilling: dict = field(default_factory=dict)
    haulage: dict = field(default_factory=dict)
    cost_zl: float = 0.0
    feed_t: float = 0.0
    limits: list = field(default_factory=list)

    @property
    def cost_per_t(self) -> float | None:
        made = sum(p["plan_t"] for p in self.products)
        return round(self.cost_zl / made, 2) if made else None


def sources_of(sc: Scenario, pile_reblast_t: dict | None = None, t_per_m: float = 50.0) -> list[Source]:
    """Everything the period can feed: piles lying now (+ re-blasts planned on them), planned blasts, areas."""
    from .site import centroid

    site = sc.site or {}
    poly = {p["id"]: p.get("polygon") for p in site.get("piles", [])}
    route_m = {}
    for e in sc.excavators:
        r = sc.routes.get(e.route)
        if r is not None:
            route_m.setdefault(e.blast, sum(s.length_m for s in r.segments))
    out = []
    for b in sc.blasts:
        extra = float((pile_reblast_t or {}).get(b.id, 0.0))
        p = poly.get(b.id)
        out.append(Source(b.id, "usyp", float(b.tonnage_t) + extra, extra, t_per_m,
                          pos=centroid(p) if p else None, route_m=route_m.get(b.id, 0.0)))
    for p in sc.planned_blasts:
        if p["id"] in {s.id for s in out}:
            continue
        t = float(p.get("tonnage_t") or 0)
        out.append(Source(p["id"], "strzał", t, 0.0 if p.get("drilled") else t, float(p.get("t_per_m") or t_per_m),
                          pos=centroid(p["polygon"]) if p.get("polygon") else None))
    for a in sc.areas:
        t = float(a.get("reserves_t") or 0)
        out.append(Source(a["id"], "obszar", t, t, float(a.get("t_per_m") or t_per_m),
                          blast_max_t=min(float(a.get("blast_max_t") or 20000), sc.blasting.max_blast_t()),
                          pos=centroid(a["polygon"]) if a.get("polygon") else None))
    return [s for s in out if s.supply_t > 0]


def _crusher_pos(sc: Scenario) -> dict:
    site = sc.site or {}
    pos = {}
    if site.get("crusher"):
        pos[sc.crusher.id] = (site["crusher"]["lat"], site["crusher"]["lon"])
    for c in site.get("crushers", []):
        if c.get("lat") is not None:
            pos[c["id"]] = (c["lat"], c["lon"])
    return pos


def haul_m(sc: Scenario, s: Source, crusher_id: str, cpos: dict) -> float:
    from .site import haversine_m

    if s.pos and crusher_id in cpos:
        return DETOUR * haversine_m(s.pos, cpos[crusher_id])
    return s.route_m or 1000.0


def plan_backwards(sc: Scenario, total_t: float, mix: dict, yields: dict, shifts: int,
                   pile_reblast_t: dict | None = None, t_per_m: float = 50.0, drills: int = 0,
                   drill_m_per_shift: float = 250.0, today: dict | None = None) -> GoalPlan:
    """The cheapest flows source -> crusher that deliver `total_t` of products split by `mix` (in %).
    `today` = KPIs of a simulated shift as it runs today (hauled_t): calibrates the haulage estimate to the real
    roads – ramps, single-lane sections, queues."""
    from scipy.optimize import linprog

    gp = GoalPlan(total_t=float(total_t), mix=dict(mix), yields=yields, shifts=int(shifts))
    hours = shifts * sc.shift_hours
    costs = sc.costs
    fleet = next((f for f in sc.trucks if f.count), sc.trucks[0])
    truck_h_cost = costs.truck_per_h + costs.fuel_per_l * 0.5 * (fleet.fuel_lph_loaded + fleet.fuel_lph_empty)
    cpos = _crusher_pos(sc)
    srcs = sources_of(sc, pile_reblast_t, t_per_m)
    crushers = list(sc.crushers)
    demand = {item: float(total_t) * float(pct) / 100 for item, pct in mix.items() if pct and float(pct) > 0}
    items = list(demand)

    def cycle_h(d: float) -> float:
        return 2 * d / 1000 / TRUCK_KMH + TRUCK_FIXED_H

    calib = 1.0
    if today and today.get("hauled_t"):
        lengths = [sc.routes[e.route].length_m for e in sc.excavators if e.route in sc.routes]
        model = sum(cycle_h(d) for d in lengths) / len(lengths) / fleet.payload_t if lengths else 0.0
        real = sc.truck_count * sc.shift_hours / float(today["hauled_t"])
        if model > 0:
            calib = max(0.5, min(4.0, real / model))
    gp.haulage["kalibracja"] = round(calib, 2)
    pairs = []  # (source index, crusher index, cost per t, truck hours per t, haul m)
    for i, s in enumerate(srcs):
        for j, c in enumerate(crushers):
            if not c.takes(s.id):
                continue
            d = haul_m(sc, s, c.id, cpos)
            truck_h_t = calib * cycle_h(d) / fleet.payload_t
            crush = costs.crusher_per_h / (c.capacity_tph * CRUSHER_UTIL) + c.kwh_per_t * costs.energy_per_kwh
            blast = costs.drill_blast_per_t * (s.blast_t / s.supply_t if s.supply_t else 0.0)
            pairs.append((i, j, truck_h_t * truck_h_cost + crush + blast, truck_h_t, d))
    nx, nu = len(pairs), len(items)
    if not nx:
        gp.feasible = False
        gp.limits.append("Żadna kruszarka nie przyjmuje urobku z żadnego usypu ani obszaru.")
        return gp
    cvec = np.array([p[2] for p in pairs] + [SHORTAGE_COST] * nu)
    a_ub, b_ub = [], []
    for k, item in enumerate(items):  # yield of every flow into the goal item >= demand (minus shortage)
        row = np.zeros(nx + nu)
        for n, (_, j, *_r) in enumerate(pairs):
            row[n] = -sum(float(yields.get(crushers[j].id, {}).get(p, 0.0)) for p in members(item)) / 100
        row[nx + k] = -1.0
        a_ub.append(row)
        b_ub.append(-demand[item])
    cap = {}
    for j, c in enumerate(crushers):
        cap[j] = c.capacity_tph * hours * CRUSHER_UTIL
        row = np.zeros(nx + nu)
        for n, p in enumerate(pairs):
            row[n] = 1.0 if p[1] == j else 0.0
        a_ub.append(row)
        b_ub.append(cap[j])
    for i, s in enumerate(srcs):
        row = np.zeros(nx + nu)
        for n, p in enumerate(pairs):
            row[n] = 1.0 if p[0] == i else 0.0
        a_ub.append(row)
        b_ub.append(s.supply_t)
    # the excavators load everything (they move between piles, so their output counts together)
    load_cap = sum(e.bucket_m3 * e.fill_factor * sc.loose_density_t_m3 * 3600 / e.swing_cycle_s
                   for e in sc.excavators) * hours * LOADING_UTIL
    a_ub.append(np.r_[np.ones(nx), np.zeros(nu)])
    b_ub.append(load_cap)
    res = linprog(cvec, A_ub=np.array(a_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * (nx + nu), method="highs")
    if res.status != 0:
        gp.feasible = False
        gp.limits.append(f"Nie udało się policzyć planu ({res.message}).")
        return gp
    x, u = res.x[:nx], res.x[nx:]

    feed_c = {j: 0.0 for j in range(len(crushers))}
    from_s = {i: 0.0 for i in range(len(srcs))}
    truck_h = 0.0
    for n, (i, j, cost, th, d) in enumerate(pairs):
        if x[n] < 1:
            continue
        feed_c[j] += x[n]
        from_s[i] += x[n]
        truck_h += x[n] * th
        gp.flows.append({"skad": srcs[i].id, "rodzaj": srcs[i].kind, "kruszarka": crushers[j].id,
                         "t": round(x[n]), "odleglosc_m": round(d), "koszt_na_t": round(cost, 2)})
    gp.feed_t = round(sum(feed_c.values()))
    gp.cost_zl = round(float(np.dot(cvec[:nx], x)))

    made = {}
    for item in items:
        made[item] = sum(feed_c[j] * sum(float(yields.get(c.id, {}).get(p, 0.0)) for p in members(item)) / 100
                         for j, c in enumerate(crushers))
    full_c = [c.id for j, c in enumerate(crushers) if feed_c[j] >= 0.99 * cap[j]]
    empty_s = [s.id for i, s in enumerate(srcs) if from_s[i] >= 0.99 * s.supply_t]
    for k, item in enumerate(items):
        short = float(u[k])
        gp.products.append({"produkt": item, "cel_t": round(demand[item]), "plan_t": round(made[item]),
                            "brak_t": round(short) if short > 1 else 0})
        if short > 1:
            makers = [c.id for c in crushers if any(yields.get(c.id, {}).get(p) for p in members(item))]
            if not makers:
                why = "żadna kruszarka go nie produkuje – ustaw uzyski kruszarek"
            elif sum(feed_c.values()) >= 0.99 * load_cap:
                why = (f"koparki nie załadują więcej (ok. {load_cap / shifts:,.0f} t na zmianę razem) – więcej zmian "
                       "albo koparka").replace(",", " ")
            elif all(m in full_c for m in makers):
                why = f"kruszarki {', '.join(makers)} pracują na pełnej wydajności – więcej zmian albo godzin"
            elif empty_s:
                why = f"brakuje urobku ({', '.join(empty_s)} wybrane do końca) – większe zasoby / nowy obszar"
            else:
                why = "ograniczenie łączne kruszarek i urobku"
            gp.limits.append(f"**{item}**: brakuje {short:,.0f} t – {why}.".replace(",", " "))
    gp.feasible = not gp.limits

    for j, c in enumerate(crushers):
        gp.crushers.append({"kruszarka": c.id, "nadawa_t": round(feed_c[j]),
                            "na_zmiane_t": round(feed_c[j] / shifts) if shifts else 0,
                            "wykorzystanie_pct": float(round(100 * feed_c[j] / (c.capacity_tph * hours), 1)) if hours else 0.0,
                            "pelna": c.id in full_c})
    drill_m = 0.0
    for i, s in enumerate(srcs):
        if from_s[i] < 1 and s.kind == "usyp":
            gp.sources.append({"skad": s.id, "rodzaj": s.kind, "dostepne_t": round(s.supply_t), "plan_t": 0,
                               "udzial_pct": 0.0, "do_odstrzalu_t": 0, "strzalow": 0, "wiercenie_m": 0})
            continue
        if from_s[i] < 1:
            continue
        to_blast = min(from_s[i], s.blast_t) if s.kind == "usyp" else (from_s[i] if s.blast_t else 0.0)
        n_blasts = math.ceil(to_blast / s.blast_max_t) if s.blast_max_t and to_blast else (1 if to_blast else 0)
        m = to_blast / s.t_per_m if to_blast else 0.0
        drill_m += m
        gp.sources.append({"skad": s.id, "rodzaj": s.kind, "dostepne_t": round(s.supply_t), "plan_t": round(from_s[i]),
                           "udzial_pct": float(round(100 * from_s[i] / gp.feed_t, 1)) if gp.feed_t else 0.0,
                           "do_odstrzalu_t": round(to_blast), "strzalow": n_blasts, "wiercenie_m": round(m)})
    rig_shifts = drill_m / drill_m_per_shift if drill_m_per_shift else 0.0
    gp.drilling = {"metry": round(drill_m), "zmian_wiertnicy": float(round(rig_shifts, 1)),
                   "wiertnic_potrzeba": math.ceil(rig_shifts / shifts) if shifts and rig_shifts else 0,
                   "wiertnic_jest": drills}
    if drills and rig_shifts > drills * shifts:
        gp.limits.append(f"Wiercenie: potrzeba {rig_shifts:.0f} zmian wiertnicy, a {drills} wiertnic(e) mają "
                         f"{drills * shifts} – dołóż wiertnicę albo zmiany.")
    trucks = math.ceil(truck_h / (hours * FLEET_UTIL)) if hours else 0
    gp.haulage |= {"motogodziny": round(truck_h), "wozidel_potrzeba": trucks, "wozidel_jest": int(sc.truck_count),
                   "zaladunek_na_zmiane_t": round(load_cap / shifts) if shifts else 0}
    return gp


def apply_plan(sc: Scenario, gp: GoalPlan) -> None:
    """Put the backwards plan into the scenario: crusher feed targets and accepted piles, feed shares."""
    feed = {c["kruszarka"]: c for c in gp.crushers}
    multi = len(sc.crushers) > 1
    flows_to = {}
    for f in gp.flows:
        flows_to.setdefault(f["kruszarka"], set()).add(f["skad"])
    used = {f["skad"] for f in gp.flows}
    all_ids = [b.id for b in sc.blasts] + [p["id"] for p in sc.planned_blasts + sc.areas]
    for c in sc.crushers:
        c.target_t = float(feed.get(c.id, {}).get("na_zmiane_t", 0.0))
        if multi:  # a crusher takes what the plan sends to it (+ piles without any flow, so they stay reachable)
            keep = {x for x in all_ids if x not in used and c.takes(x)}
            c.accepts = sorted(flows_to.get(c.id, set()) | keep) or list(c.accepts)
    share = {s["skad"]: s["udzial_pct"] for s in gp.sources if s["udzial_pct"] > 0}
    for b in sc.blasts:
        b.target_share = share.get(b.id)
    for p in sc.planned_blasts + sc.areas:
        p["target_share"] = share.get(p["id"])


def products_made(crushed: dict, yields: dict, items: list[str]) -> dict:
    """Tonnes of every goal item out of the tonnes crushed per crusher."""
    return {item: sum(t * sum(float(yields.get(cid, {}).get(p, 0.0)) for p in members(item)) / 100
                      for cid, t in crushed.items()) for item in items}
