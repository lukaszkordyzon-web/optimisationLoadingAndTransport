import copy
from pathlib import Path

import pytest

from kruszyk.config import clock_to_s, load_scenario, s_to_clock
from kruszyk.sim import Simulation, simulate

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


@pytest.fixture
def demo():
    return load_scenario(DEMO)


def _starts_in(df, a, b):
    """Loadings that START inside the window (one already under way is finished first)."""
    return df[(df["start_s"] >= a) & (df["start_s"] < b)]


def test_clock():
    assert clock_to_s("09:30") == 3.5 * 3600
    assert clock_to_s("02:00", "22:00") == 4 * 3600
    assert s_to_clock(5 * 3600 + 15 * 60) == "11:15"


def test_nobody_loads_during_blasting_or_breaks(demo):
    res = simulate(demo)
    loading = res.states[res.states["state"] == "zaladunek"]
    a, b = res.closure
    assert a == clock_to_s("11:00") - 600 and b == clock_to_s("11:20")
    assert _starts_in(loading, a, b).empty
    for a, b in res.breaks:
        assert _starts_in(loading, a + 1, b - 1).empty
    assert "postoj_strzelanie" in set(res.states["state"])
    assert res.kpi["excavator_down_h"]["strzelanie"] > 0


def test_shift_starts_after_startup(demo):
    res = simulate(demo)
    first = res.states[res.states["state"] != "postoj_brak_urobku"]["start_s"].min()
    assert first >= demo.shift_plan.startup_min * 60


def test_light_vehicles_drive_and_trucks_give_way(demo):
    res = simulate(demo)
    names = {v["name"] for v in res.light_vehicles}
    assert names == {"ekipa strzałowa", "sztygar", "cysterna 1"}
    assert res.kpi["yields"] >= 1
    # tanker stops each excavator for refuelling, nobody is loaded there meanwhile
    for ex, windows in res.excavator_down.items():
        for a, b, why in windows:
            if why == "tankowanie":
                busy = res.states[(res.states["excavator"] == ex) & (res.states["state"] == "zaladunek")]
                assert _starts_in(busy, a + 1, b - 1).empty


def test_yield_rule_down_gives_way_to_up(demo):
    sim = Simulation(copy.deepcopy(demo))
    sim.light_passages = {"rampa główna": [(100.0, 160.0, True, "x"), (300.0, 360.0, False, "y")]}
    assert sim._yield_delay("rampa główna", 120, 200, truck_going_up=False)[0] == 1   # truck down, car up -> wait
    assert sim._yield_delay("rampa główna", 120, 200, truck_going_up=True)[0] == 0    # truck up keeps going
    assert sim._yield_delay("rampa główna", 310, 350, truck_going_up=False)[0] == 0   # car down gives way
    assert sim._yield_delay("front", 120, 200, truck_going_up=False)[0] == 0


def test_speed_limits_are_binding_unless_drivers_speed(demo):
    sc = copy.deepcopy(demo)
    sc.variability.speeding_share = 0.0
    res = simulate(sc)
    roads = res.segments[~res.segments["segment"].isin(["front", "plac_kruszarki", "dojazd"])]
    assert res.kpi["over_limit_pct"] == 0
    assert (roads["v_kmh"] <= roads["speed_limit_kmh"] + 0.5).all()
    sc.variability.speeding_share, sc.variability.speeding_pct = 1.0, 20
    res = simulate(sc)
    roads = res.segments[~res.segments["segment"].isin(["front", "plac_kruszarki", "dojazd"])]
    assert res.kpi["over_limit_pct"] > 0
    assert (roads["v_kmh"] <= roads["speed_limit_kmh"] * 1.2 + 0.5).all()


def test_small_tank_forces_refuelling(demo):
    sc = copy.deepcopy(demo)
    for f in sc.trucks:
        f.tank_l = 120
    res = simulate(sc)
    assert res.kpi["truck_refuels"] > 0 and "tankowanie" in set(res.states["state"])


def test_advisor_prefills_buckets_and_slows_trucks_instead_of_queueing(demo):
    base = simulate(demo).kpi
    sc = copy.deepcopy(demo)
    sc.advisor.enabled = True
    adv = simulate(sc)
    assert adv.kpi["prefills"] > base["prefills"]
    assert (adv.segments.loc[adv.segments["leg"] == "pusty", "pace"] > 1.0).any()
