"""What the phones would show at a given moment: positions, operator ETA list, driver instructions.

Built from simulation output now; later the same functions run on live telemetry.
"""
from __future__ import annotations

import pandas as pd

from .config import Scenario
from .physics import route_time_s
from .site import point_along, polyline_length_m

STATE_LABELS = {
    "jazda_pusty": "jedzie pusty do koparki",
    "kolejka_koparka": "czeka pod koparką",
    "podstawienie": "podstawia się",
    "zaladunek": "ładowany",
    "jazda_pelny": "jedzie pełny do kruszarki",
    "kolejka_kruszarka": "czeka pod kruszarką",
    "wyladunek": "wyładunek do kosza",
    "wyladunek_halda": "wyładunek na hałdę",
    "awaria": "awaria",
    "postoj_brak_urobku": "postój – brak urobku",
}


def route_profile(sc: Scenario, route_id: str) -> pd.DataFrame:
    """Points of the route as (distance to crusher, elevation relative to crusher)."""
    segs = sc.routes[route_id].segments
    total = sum(s.length_m for s in segs)
    climb = sum(s.length_m * s.grade_pct / 100 for s in segs)
    rows, dist, elev = [], total, -climb
    rows.append({"route": route_id, "segment": segs[0].name, "dist_m": dist, "elev_m": elev})
    for s in segs:
        dist -= s.length_m
        elev += s.length_m * s.grade_pct / 100
        rows.append({"route": route_id, "segment": s.name, "dist_m": dist, "elev_m": elev})
    return pd.DataFrame(rows)


def _point_on_route(sc: Scenario, route_id: str, dist_from_front: float) -> tuple[float, float]:
    prof = route_profile(sc, route_id)
    total = prof["dist_m"].iloc[0]
    target = total - dist_from_front  # distance to crusher
    # profile is ordered from front (max dist) to crusher (0)
    for a, b in zip(prof.itertuples(), prof.iloc[1:].itertuples()):
        if b.dist_m <= target <= a.dist_m:
            f = 0 if a.dist_m == b.dist_m else (a.dist_m - target) / (a.dist_m - b.dist_m)
            return target, a.elev_m + f * (b.elev_m - a.elev_m)
    return (0.0, 0.0) if target <= 0 else (total, prof["elev_m"].iloc[0])


def snapshot(res, t: float) -> pd.DataFrame:
    """State of every truck at time t (seconds from shift start)."""
    sc: Scenario = res.scenario
    st = res.states[(res.states["start_s"] <= t) & (res.states["end_s"] > t)]
    seg = res.segments
    if len(seg):  # only the trips in progress matter
        keys = set(zip(st["truck"], st["trip"]))
        seg = seg[[k in keys for k in zip(seg["truck"], seg["trip"])]]
    ex_route = {e.id: e.route for e in sc.excavators}
    rows = []
    for truck in res.truck_ids:
        cur = st[(st["truck"] == truck) & (st["start_s"] <= t) & (st["end_s"] > t)]
        if cur.empty:
            continue  # before first departure or in a state still open at shift end
        s = cur.iloc[0]
        row = {"truck": truck, "state": s["state"], "status": STATE_LABELS.get(s["state"], s["state"]),
               "excavator": s["excavator"], "trip": int(s["trip"]), "since_min": (t - s["start_s"]) / 60,
               "segment": None, "eta_s": None}
        ex = s["excavator"]
        if ex is None or (isinstance(ex, float) and pd.isna(ex)):
            # crusher-side states: find the excavator of this trip from the loaded haul
            trip_seg = seg[(seg["truck"] == truck) & (seg["trip"] == s["trip"])]
            ex = trip_seg["excavator"].iloc[0] if len(trip_seg) else None
            row["excavator"] = ex
        route = ex_route.get(ex) if ex else None
        row["route"] = route
        total = sc.routes[route].length_m if route else 0.0

        if s["state"] in ("jazda_pusty", "jazda_pelny"):
            loaded = int(s["state"] == "jazda_pelny")
            leg = seg[(seg["truck"] == truck) & (seg["trip"] == s["trip"]) & (seg["loaded"] == loaded)]
            leg = leg.sort_values("start_s")
            on = leg[(leg["start_s"] <= t) & (leg["start_s"] + leg["duration_s"] > t)]
            if len(on):
                k = on.iloc[0]
                progress = (t - k["start_s"]) / k["duration_s"]
                row["segment"] = k["segment"]
                done = leg[leg["start_s"] < k["start_s"]]["length_m"].sum() + progress * k["length_m"]
                dist_from_front = total - done if not loaded else done
                # ETA uses the expected (physics) pace – what the app knows, not the future
                later = leg[leg["start_s"] > k["start_s"]]["physics_s"].sum()
                row["eta_s"] = (1 - progress) * k["physics_s"] + later
            else:
                dist_from_front = 0.0 if not loaded else total
        elif s["state"] in ("kolejka_koparka", "podstawienie", "zaladunek"):
            dist_from_front = 0.0
        else:
            dist_from_front = total
        if route:
            row["dist_m"], row["elev_m"] = _point_on_route(sc, route, dist_from_front)
            path = sc.routes[route].path
            if path:
                geo_len = polyline_length_m(path)
                row["lat"], row["lon"] = point_along(path, dist_from_front * geo_len / total if total else 0.0)
        else:
            row["dist_m"], row["elev_m"] = 0.0, 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def operator_view(snap: pd.DataFrame, excavator: str) -> dict:
    """What the excavator operator sees: who is being loaded, who waits, who is coming and when."""
    if snap.empty:
        return {"loading": None, "waiting": [], "incoming": []}
    mine = snap[snap["excavator"] == excavator]
    loading = mine[mine["state"].isin(["zaladunek", "podstawienie"])]
    waiting = mine[mine["state"] == "kolejka_koparka"]
    incoming = mine[mine["state"] == "jazda_pusty"].sort_values("eta_s")
    return {
        "loading": loading["truck"].iloc[0] if len(loading) else None,
        "waiting": list(waiting["truck"]),
        "incoming": [(r.truck, r.eta_s, r.segment) for r in incoming.itertuples()],
    }


def driver_instruction(row) -> str:
    s = row["state"]
    if s == "jazda_pusty":
        return f"Jedź do {row['excavator']} – dojazd ok. {row['eta_s'] / 60:.1f} min" if row["eta_s"] else f"Jedź do {row['excavator']}"
    if s == "jazda_pelny":
        return "Jedź do kruszarki"
    if s == "kolejka_koparka":
        return f"Czekaj pod {row['excavator']}"
    if s in ("podstawienie", "zaladunek"):
        return "Załadunek"
    if s == "kolejka_kruszarka":
        return "Czekaj – kosz pełny"
    if s in ("wyladunek", "wyladunek_halda"):
        return "Wyładunek"
    if s == "awaria":
        return "Zgłoszona awaria"
    return "Koniec pracy – brak urobku"


def expected_cycle_min(sc: Scenario, excavator: str) -> float:
    ex = next(e for e in sc.excavators if e.id == excavator)
    truck = max(sc.trucks, key=lambda f: f.count)
    r = sc.routes[ex.route]
    return (route_time_s(r, truck, True, sc.rolling_resistance_pct)
            + route_time_s(r, truck, False, sc.rolling_resistance_pct)) / 60
