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
    assert len(r.comparison) == 5  # natural, fixed, heuristic, RL dispatch only, full RL
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


def test_agent_learns_advice_settings_too(demo):
    from kruszyk.rl import decode_advisor, explain_advisor

    # starting point of the learned advice = the hand-made rules
    assert decode_advisor([0, 0, 0]) == pytest.approx({"arrival_margin_s": 0.0, "max_slowdown": 0.3, "crusher_pacing": 1.0})
    r = train(copy.deepcopy(demo), iterations=2, population=5, episodes=1, shift_hours=2, eval_episodes=1,
              advisor=True, learn_advisor=True)
    adv = r.advisor
    assert -120 <= adv["arrival_margin_s"] <= 120 and 0 < adv["max_slowdown"] < 0.6 and 0 < adv["crusher_pacing"] < 2
    assert list(r.comparison["polityka"])[-2:] == ["polityka RL – sam przydział", "polityka RL – pełna"]
    assert len(explain_advisor(adv)) == 3


def test_excavator_management_parks_and_skips(demo):
    sc = copy.deepcopy(demo)
    sc.dispatch, sc.rl_weights = "rl", [1, 1, 0.5, 1, 0, 0.5, 0, 0]
    sc.excavator_policy.min_trucks_per_excavator = 5.0   # 7 free trucks -> only 2 excavators may work
    res = simulate(sc)
    assert res.kpi["parked_excavator_h"] > 0
    parked = [e for e in res.kpi["excavators"] if e["parked_h"] > 0]
    assert parked and all(e["excavator"] not in ("K-1", "K-2") for e in parked)  # K-1/K-2 serve the pinned blend trucks
    base = simulate(demo).kpi["cost_breakdown"]["koparki"]
    assert res.kpi["cost_breakdown"]["koparki"] < base   # a parked excavator costs less
    sc.excavator_policy.min_trucks_per_excavator = 0.0
    sc.excavator_policy.topup_min_fill, sc.excavator_policy.topup_min_queue = 0.5, 0
    res = simulate(sc)
    assert res.kpi["topup_skips"] > 0
    loads = res.loads.groupby(["truck", "trip"])["tonnes"].sum()
    assert (loads < 55 - 1).any()


def test_feed_targets_measured_and_steered(demo):
    sc = copy.deepcopy(demo)
    for b, t in zip(sc.blasts, [70, 15, 15]):
        b.target_share = t
    k = simulate(sc).kpi
    assert k["feed_target"] == {"U-01": 70.0, "U-02": 15.0, "U-03": 15.0}
    assert k["feed_dev_pp"] > 0 and k["cost_breakdown"]["odchylka_skladu"] > 0
    # a policy that follows the deficit gets closer to the target mix than one that ignores it
    sc.dispatch = "rl"
    sc.rl_weights = [0, 0, 0, 0, 0, 0, 0, -5.0]
    follow = simulate(sc).kpi["feed_dev_pp"]
    sc.rl_weights = [0, 0, 0, 0, 0, 0, 0, 5.0]
    against = simulate(sc).kpi["feed_dev_pp"]
    assert follow < against


def test_old_policies_without_new_features_still_run(demo):
    sc = copy.deepcopy(demo)
    sc.dispatch, sc.rl_weights = "rl", [1.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.0]  # saved before the feed-mix feature
    assert simulate(sc).kpi["hauled_t"] > 0
