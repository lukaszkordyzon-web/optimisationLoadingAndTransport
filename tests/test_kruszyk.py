from pathlib import Path

import pytest

from kruszyk.analytics import match_factor_report
from kruszyk.config import load_scenario
from kruszyk.ml import generate_dataset, train_travel_model
from kruszyk.optimize import recommend, sweep_trucks
from kruszyk.physics import segment_speed_kmh
from kruszyk.sim import simulate

SCENARIO = Path(__file__).parent.parent / "scenarios" / "kruszyk_bazowy.yaml"


@pytest.fixture
def sc():
    return load_scenario(SCENARIO)


def test_loaded_truck_is_slower_uphill(sc):
    ramp = sc.routes["T-dolny"].segments[1]
    truck = sc.trucks[0]
    assert segment_speed_kmh(ramp, truck, loaded=True, rr_pct=3) < segment_speed_kmh(ramp, truck, False, 3)


def test_with_truck_count(sc):
    assert sc.with_truck_count(3).truck_count == 3
    assert sc.truck_count == 8  # original untouched


def test_match_factor_scales_with_fleet(sc):
    assert match_factor_report(sc.with_truck_count(4))["match_factor"] < match_factor_report(sc)["match_factor"]


def test_simulation_mass_balance(sc):
    res = simulate(sc)
    k = res.kpi
    assert k["hauled_t"] > 0
    # cannot crush more than was dumped, nor load more than the blasts hold
    assert k["crushed_t"] <= k["hauled_t"] + 1e-6
    assert res.loads["tonnes"].sum() <= sum(b.tonnage_t for b in sc.blasts) + 1e-6
    # truck states never overlap and fit in the shift
    for _, g in res.states.sort_values("start_s").groupby("truck"):
        assert (g["start_s"].iloc[1:].to_numpy() >= g["end_s"].iloc[:-1].to_numpy() - 1e-6).all()
    assert res.states["end_s"].max() <= sc.shift_hours * 3600 + 1e-6


def test_small_blast_gets_finished(sc):
    for b in sc.blasts:
        b.tonnage_t = 1000
    k = simulate(sc).kpi
    assert all(e["remaining_t"] == 0 and e["blast_done_at_h"] for e in k["excavators"])
    assert k["truck_idle_no_material_h"] > 0


def test_more_trucks_less_excavator_idle(sc):
    few = simulate(sc.with_truck_count(3)).kpi
    many = simulate(sc.with_truck_count(7)).kpi
    assert few["throughput_tph"] < many["throughput_tph"]
    assert few["avg_queue_per_cycle_min"] < many["avg_queue_per_cycle_min"]


def test_sweep_recommendation(sc):
    summary = sweep_trucks(sc, 3, 9, replications=2)
    best = recommend(summary)
    assert 3 <= best["trucks"] <= 9


def test_travel_model_beats_physics_baseline(sc):
    data = generate_dataset(sc, shifts=12)
    _, m = train_travel_model(data)
    assert m["mae_model_s"] < m["mae_physics_baseline_s"]


def test_live_cost_ends_at_the_final_cost_per_tonne():
    """The live counter in the animation must end exactly at the shift result (same cost items)."""
    import glob

    from kruszyk.config import load_scenario
    from kruszyk.sim import simulate

    for f in sorted(glob.glob(str(Path(__file__).parent.parent / "scenarios" / "*.yaml"))):
        r = simulate(load_scenario(f), seed=5)
        assert r.timeline["t_s"].iloc[-1] == r.scenario.shift_hours * 3600
        assert r.timeline["cost_per_t"].iloc[-1] == pytest.approx(r.kpi["cost_per_t"], abs=0.006), f
