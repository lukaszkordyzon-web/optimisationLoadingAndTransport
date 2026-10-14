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


def _natural(**kw):
    sc = make(**kw)
    sc.dispatch = "natural"
    return sc


def test_today_practice_has_no_system_knowledge():
    """'As today' and the calibration: no advice, no parking, no skipped buckets – whatever is stored in the mine."""
    plain = simulate(_natural(trucks=5), seed=4).kpi
    sc = _natural(trucks=5)
    sc.advisor.enabled = True  # e.g. stored after learning in the advanced mode
    sc.excavator_policy.truck_park_queue_min = 0.0
    sc.excavator_policy.topup_min_fill = 0.8
    sc.excavator_policy.min_trucks_per_excavator = 3.0
    r = simulate(sc, seed=4)
    assert r.kpi["hauled_t"] == plain["hauled_t"] and r.kpi["cost_per_t"] == plain["cost_per_t"]
    assert r.kpi["parked_truck_h"] == 0 and r.kpi["topup_skips"] == 0 and r.kpi["parked_excavator_h"] == 0
    assert (r.segments["pace"] == 1.0).all()  # nobody slows down on advice


def test_drivers_mostly_go_back_to_their_own_excavator():
    sc = _natural(trucks=4, breakdowns=False)
    sim = Simulation(sc, 2)
    home = {t.id: t.home_excavator for t in sim.trucks}
    loads = sim.run().loads
    share = (loads["excavator"] == loads["truck"].map(home)).mean()
    # natural_switch_prob 25 %, half of the switches land on the own excavator again -> ~87 %
    assert 0.78 < share < 0.95


def test_a_broken_excavator_is_known_from_the_radio():
    sc = _natural(trucks=4, breakdowns=False)
    sc.carry_down = {"K-1": 3 * 3600.0}  # K-1 in repair for the first 3 hours
    r = simulate(sc, seed=2)
    early = r.loads[r.loads["loaded_at_s"] < 3 * 3600]
    assert len(early) and (early["excavator"] == "K-2").all()
    waits = r.states[(r.states["state"] == "postoj_koparki") & (r.states["excavator"] == "K-1")]["duration_s"].sum()
    assert waits < 1800  # nobody stands for hours at the broken excavator


def test_ramp_traffic_is_known_to_the_system():
    """Plans of the trucks on single-lane ramps: an empty truck sent down meets the loaded one coming up."""
    sc = make(single_lane=True, trucks=6, crusher_tph=800, bucket_m3=3.0, swing_s=30)
    sc.dispatch, sc.rl_weights = "rl", [1.0] + [0.0] * 7 + [1.0]  # the learned policy reads the ramp traffic
    sim = Simulation(sc, 3)
    seen = []
    orig = sim._ramp_wait_s

    def spy(plan, me=None):
        out = orig(plan, me)
        seen.append(out[0])
        return out
    sim._ramp_wait_s = spy
    sim.env.process(sim.crusher_process(sim.main))
    for i, t in enumerate(sim.trucks):
        sim.env.process(sim._delayed_start(t, sim.t_start + i * 30.0))
    sim.env.run(until=3 * 3600)
    assert seen and max(seen) > 0  # the dispatch saw a meeting on the ramp


def test_a_stop_at_a_passing_bay_costs_pulling_away():
    sc = make(single_lane=True, trucks=6, crusher_tph=800)
    sc.dispatch = "natural"
    r = simulate(sc, seed=5)
    bays = r.states[r.states["state"] == "mijanka"]["duration_s"]
    assert len(bays) and bays.min() >= 25 / 3.6 / (2 * 0.5) - 1e-6  # at least the time to pull away (empty)


def test_loaded_truck_waits_only_for_one_already_on_the_stretch():
    """Uphill (loaded) has priority; it stops at a bay only when an empty truck has already passed it."""
    sc = make(single_lane=True, trucks=6, crusher_tph=800, bucket_m3=3.0, swing_s=30)
    sc.dispatch = "natural"
    st = simulate(sc, seed=5).states.sort_values(["truck", "start_s"])
    st["prev"] = st.groupby("truck")["state"].shift(1)
    bays = st[st["state"] == "mijanka"].groupby("prev")["duration_s"].agg(["count", "mean"])
    assert bays.loc["jazda_pusty", "count"] > 4 * bays.loc["jazda_pelny", "count"]
    assert bays.loc["jazda_pusty", "mean"] > bays.loc["jazda_pelny", "mean"]
