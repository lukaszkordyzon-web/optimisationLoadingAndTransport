"""Rules of the shift: goal, excavator moving to the next pile, binding / free truck assignment."""
import copy
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from kruszyk.config import scenario_from_dict
from kruszyk.sim import simulate
from kruszyk.site import site_to_scenario

ROOT = Path(__file__).parent.parent
APP = str(ROOT / "app.py")
DEMO = ROOT / "scenarios" / "kruszyk_demo_zakrzowek.yaml"


def _raw():
    return yaml.safe_load(DEMO.read_text(encoding="utf-8"))


def test_goal_reached_then_trucks_park():
    raw = _raw()
    raw["shift_goal"] = {"kind": "feed", "t": 1500, "feed_t": 1500, "stop": True}
    k = simulate(scenario_from_dict(raw), seed=1).kpi
    assert k["goal_reached_h"] is not None and 1500 <= k["hauled_t"] < 1800
    assert k["parked_truck_h"] > 1  # engines off after the goal


def test_missed_goal_costs_more_for_the_policy():
    raw = _raw()
    raw["shift_goal"] = {"kind": "feed", "t": 50_000, "feed_t": 50_000, "stop": True}
    k = simulate(scenario_from_dict(raw), seed=1).kpi
    assert k["goal_reached_h"] is None and k["goal_pct"] < 20
    assert k["objective"] > 2 * k["cost_per_t"]


def test_excavator_drives_to_the_next_pile_during_the_shift():
    raw = _raw()
    site = copy.deepcopy(raw["site"])
    for e in site["excavators"]:
        if e["id"] == "K-2":
            e["then_pile"], e["then_below_t"] = "U-02", 0.0
    sc = scenario_from_dict(site_to_scenario(site, raw))
    for b in sc.blasts:
        if b.id == "U-01":
            b.tonnage_t = 1200.0  # runs out early in the shift
    k = simulate(sc, seed=1).kpi
    assert [m["to"] for m in k["excavator_moves"]] == ["U-02"]
    assert k["feed_share"]["U-02"] > 30


def test_free_assignment_only_binds_today_practice():
    raw = _raw()
    for t in raw["trucks"]:
        t["excavators"], t["assign_locked"] = ["K-1"], False
    sc = scenario_from_dict(raw)
    nat = copy.deepcopy(sc)
    nat.dispatch = "natural"
    assert not simulate(nat, seed=2).kpi["off_rule_trips"]
    dyn = copy.deepcopy(sc)
    dyn.dispatch = "dynamic"
    assert simulate(dyn, seed=2).kpi["off_rule_trips"]  # the system also uses K-2
    for t in raw["trucks"]:
        t["assign_locked"] = True
    dyn = scenario_from_dict(raw)
    dyn.dispatch = "dynamic"
    assert not simulate(dyn, seed=2).kpi["off_rule_trips"]


def test_rules_screen_saves_goal_assignment_and_move():
    at = AppTest.from_file(APP, default_timeout=300).run()
    ver = at.session_state["ver"]
    at.radio(key=f"rules_goal_kind_{ver}").set_value("Tony przez zakład").run()
    at.number_input(key=f"rules_goal_t_{ver}").set_value(1800.0).run()
    at.selectbox(key=f"rules_then_K-2_{ver}").set_value("U-02").run()
    at.multiselect(key=f"rules_tr_K-1_{ver}").set_value(["WZ-01", "WZ-02"]).run()
    at.toggle(key=f"rules_lock_K-1_{ver}").set_value(True).run()
    at.button(key="rules_apply").click().run()
    assert not at.exception, at.exception
    raw = at.session_state["raw"]
    assert raw["shift_goal"]["feed_t"] == 1800.0
    assert next(e for e in raw["excavators"] if e["id"] == "K-2")["then_blast"] == "U-02"
    t1 = next(t for t in raw["trucks"] if t["id"] == "WZ-01")
    assert t1["excavators"] == ["K-1"] and t1["assign_locked"] is True
    assert not next(t for t in raw["trucks"] if t["id"] == "WZ-03")["excavators"]
