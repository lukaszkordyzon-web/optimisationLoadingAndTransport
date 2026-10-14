"""Regressions found by the validation on the test mine (tools/walidacja)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "walidacja"))
from kopalnia_testowa import BASE, make  # noqa: E402

from kruszyk.sim import Simulation, simulate  # noqa: E402


def test_a_group_of_trucks_is_dealt_out_among_the_excavators():
    sc = make(trucks=4)  # one row: 4 identical trucks
    homes = [t.home_excavator for t in Simulation(sc, 1).trucks]
    assert sorted(homes) == ["K-1", "K-1", "K-2", "K-2"]
    sc.dispatch = "natural"
    share = simulate(sc, seed=1).kpi["feed_share"]["B1"]
    assert 35 < share < 65  # today's practice works both levels


def test_nothing_is_crushed_that_was_not_hauled():
    p = {k: BASE[k] * 3 for k in ("flat_m", "ramp1_m", "ramp2_m", "bench_m")}
    for seed in range(6):
        sc = make(**p)
        sc.dispatch = "natural"
        k = simulate(sc, seed=seed).kpi
        assert k["crushed_t"] <= k["hauled_t"] + 1e-6


def test_goal_plan_full_cost_is_close_to_the_simulation():
    from kruszyk.goal import plan_backwards

    sc = make(b1_t=300_000, b2_t=300_000)
    gp = plan_backwards(sc, total_t=60_000, mix={"grysy (2–16)": 60.0}, shifts=40,
                        yields={"C-1": {"grys 2–8": 25.0, "grys 8–16": 30.0}})
    per_feed = gp.full_cost_zl / gp.feed_t
    assert gp.feasible and 14 < per_feed < 30  # the simulation gives 16–21 zł/t for this mine
    assert gp.full_cost_zl > 1.5 * gp.cost_zl
