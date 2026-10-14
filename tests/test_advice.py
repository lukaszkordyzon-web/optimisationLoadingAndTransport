"""Recommendations from the learned policy: checked on the same shifts as today's practice."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "walidacja"))
from kopalnia_testowa import make  # noqa: E402

from kruszyk.advice import recommend  # noqa: E402


def _parking_policy(sc):
    pol = copy.deepcopy(sc)
    pol.dispatch, pol.rl_weights = "rl", [1.0] + [0.0] * 8
    pol.excavator_policy.truck_park_queue_min, pol.excavator_policy.truck_min_working_share = 5.0, 0.4
    return pol


def test_too_many_trucks_recommends_a_smaller_fleet_without_losing_production():
    sc = make(trucks=6)  # the crusher is full with 3
    out = recommend(sc, _parking_policy(sc), seed=3)
    fleet = [r for r in out["organisational"] if r["kind"] == "fleet"]
    assert fleet and fleet[0]["d_cost_pct"] < -5 and fleet[0]["d_t_pct"] >= -1.5
    assert any("Parkowanie" in x["title"] for x in out["system"])
    assert out["full"]["n"] == 10


def test_a_balanced_mine_gets_no_fleet_cut():
    sc = make(trucks=3)  # the engineer's fleet: nothing to cut
    out = recommend(sc, _parking_policy(sc), seed=3)
    assert not [r for r in out["organisational"] if r["kind"] == "fleet"]
