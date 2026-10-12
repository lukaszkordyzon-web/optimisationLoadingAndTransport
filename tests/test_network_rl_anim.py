import copy
import json
from pathlib import Path

import pytest

from kruszyk.anim import STATE_CODES, animation_payload
from kruszyk.config import load_scenario
from kruszyk.rl import FEATURE_NAMES, train
from kruszyk.sim import simulate
from kruszyk.site import RoadGraph, analyze_site, haversine_m

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


@pytest.fixture
def demo():
    return load_scenario(DEMO)


def test_branch_from_middle_of_ramp_is_a_junction(demo):
    site = demo.site
    g = RoadGraph(site["roads"])
    info = analyze_site(site)
    assert len(info["junctions"]) == 1 and not info["warnings"]
    j = info["junctions"][0]
    # the junction lies on the ramp, between its two drawn vertices
    ramp = next(r for r in site["roads"] if r["kind"] == "rampa")["coords"]
    assert haversine_m(ramp[0], g.nodes[j]) + haversine_m(g.nodes[j], ramp[1]) == pytest.approx(
        haversine_m(ramp[0], ramp[1]), rel=0.01)
    # elevation of the junction = top of ramp - 9 % of the distance down the ramp
    top = info["elev"][g.attach(tuple(ramp[0]))[0]]
    assert info["elev"][j] == pytest.approx(top - 0.09 * haversine_m(ramp[0], g.nodes[j]), abs=1.0)
    lv = info["levels"]
    assert lv["K-2"]["elev_m"] == lv["K-3"]["elev_m"] < lv["K-1"]["elev_m"] < 0  # bottom level below the branch


def test_route_grades_follow_travel_direction(demo):
    k1 = [s.grade_pct for s in demo.routes["T-K-1"].segments if s.name == "rampa główna"]
    down = [s.grade_pct for s in demo.links["K-1>K-2"].segments if s.name == "rampa główna"]
    assert k1 == [9.0] and down == [-9.0]
    flat = [s.grade_pct for s in demo.routes["T-K-2"].segments if s.name == "poziom dolny"]
    assert flat == [0.0]


def test_allowed_excavators_are_respected(demo):
    sc = copy.deepcopy(demo)
    sc.trucks = sc.trucks[:1]
    sc.trucks[0].excavators = ["K-3"]
    for policy in ("natural", "dynamic"):
        sc.dispatch = policy
        res = simulate(sc)
        assert set(res.loads["excavator"]) == {"K-3"}


def test_two_pile_trips_load_both_in_order(demo):
    res = simulate(demo)
    blend_trucks = {t for t in res.truck_ids if t in ("WZ-08", "WZ-09")}
    loads = res.loads[res.loads["truck"].isin(blend_trucks) & res.loads["dumped_at_s"].notna()]  # finished trips
    per_trip = loads.groupby(["truck", "trip"])["excavator"].apply(list)
    assert (per_trip.str[0] == "K-1").all() and (per_trip.str[-1] == "K-2").all()
    assert (loads.groupby(["truck", "trip"])["tonnes"].sum() <= 40 + 1e-6).all()
    assert "przejazd_miedzy_usypami" in set(res.states["state"])


def test_rl_policy_runs_and_learns(demo):
    sc = copy.deepcopy(demo)
    r = train(sc, iterations=3, population=6, episodes=1, shift_hours=2, eval_episodes=1)
    assert list(r.history["iteracja"]) == [0, 1, 2, 3]
    assert len(r.weights) == len(FEATURE_NAMES)
    # the untrained start (all weights 0 -> always the first excavator) is beaten after learning
    assert r.history["koszt_polityka"].iloc[-1] < r.history["koszt_polityka"].iloc[0]
    assert len(r.comparison) == 4
    sc.dispatch, sc.rl_weights = "rl", r.weights
    assert simulate(sc).kpi["hauled_t"] > 0


def test_animation_payload(demo):
    res = simulate(demo)
    data = json.loads(animation_payload(res, simulate(demo, seed=1), "test"))
    assert len(data["trucks"]) == demo.truck_count
    lat0, lon0 = demo.site["center"]
    for frames in data["trucks"].values():
        times = [f[0] for f in frames]
        assert times == sorted(times)
        assert all(abs(f[1] - lat0) < 0.05 and abs(f[2] - lon0) < 0.05 for f in frames)
        assert all(0 <= f[3] < len(STATE_CODES) for f in frames)
    assert len(data["series"]["t"]) > 100 and data["baseline"]["cpt"]
