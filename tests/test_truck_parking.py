"""Trucks parked by the learned policy instead of queueing at a full crusher."""
import copy
from pathlib import Path

from kruszyk.config import load_scenario
from kruszyk.rl import decode_excavators, explain_excavators
from kruszyk.sim import simulate

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml"


def _rl(sc, margin, share=0.5):
    s = copy.deepcopy(sc)
    s.dispatch, s.rl_weights = "rl", [1.0] + [0.0] * 7
    s.excavator_policy.truck_park_queue_min, s.excavator_policy.truck_min_working_share = margin, share
    return s


def test_parking_cuts_the_crusher_queue_and_fuel():
    sc = load_scenario(DEMO)  # one crusher, six trucks: the crusher is the limit
    off, on = simulate(_rl(sc, -1.0), seed=3).kpi, simulate(_rl(sc, 5.0), seed=3).kpi
    assert off["parked_truck_h"] == 0
    assert on["parked_truck_h"] > 1
    assert on["truck_queue_crusher_h"] < off["truck_queue_crusher_h"]
    assert on["fuel_l"] < off["fuel_l"]
    assert on["hauled_t"] > 0.85 * off["hauled_t"]


def test_parked_truck_saves_its_engine_hour_cost():
    sc = load_scenario(DEMO)
    k = simulate(_rl(sc, 5.0), seed=3).kpi
    c = sc.costs
    full = sc.truck_count * c.truck_per_h * sc.shift_hours
    assert abs(k["cost_breakdown"]["wozidla"] - (full - k["parked_truck_h"] * c.truck_per_h * c.truck_variable_share)) < 2 + full * 0.001


def test_at_least_the_working_share_keeps_driving():
    sc = load_scenario(DEMO)
    r = simulate(_rl(sc, 0.0, share=0.5), seed=4)
    park = r.states[r.states["state"] == "parking"]
    for t in range(0, int(sc.shift_hours * 3600), 60):
        assert ((park["start_s"] <= t) & (park["end_s"] > t)).sum() <= sc.truck_count - 3


def test_untrained_policy_does_not_park_and_is_explained():
    p = decode_excavators([0.0] * 5)
    assert p["truck_park_queue_min"] < 0
    assert any("nie są odstawiane" in s for s in explain_excavators(p))
    p = decode_excavators([0.0, 0.0, 0.0, 1.0, 0.0])
    assert 0 < p["truck_park_queue_min"] < 30
    assert any("parkuje" in s for s in explain_excavators(p))
    assert decode_excavators([0.0, 0.0, 0.0])["truck_park_queue_min"] < 0  # policies saved before keep working
