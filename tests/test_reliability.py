"""Breakdowns: failures per month and their total hours, per machine."""
import datetime as dt
import math
from pathlib import Path

import numpy as np
import yaml

from kruszyk.config import Reliability, scenario_from_dict
from kruszyk.period import PeriodPlan, run_period
from kruszyk.sim import simulate

MINE = Path(__file__).parent.parent / "scenarios" / "kruszyk_kopalnia.yaml"


def mine(**machines):
    raw = yaml.safe_load(MINE.read_text(encoding="utf-8"))
    raw["reliability"] = {"machines": machines}
    sc = scenario_from_dict(raw)
    sc.dispatch = "natural"
    return sc


def test_rates_follow_the_month_and_the_shifts_worked():
    r = Reliability(ref_shifts_per_day=2, ref_days_per_month=22, machines={"C-2": {"count": 4, "hours": 12}})
    mtbf, mttr = r.rates("C-2", "crusher", 8.0)
    assert mttr == 3.0 and mtbf == 2 * 8 * 22 / 4
    assert r.values("K-1", "excavator")[2]  # nothing given: an assumption
    assert Reliability(enabled=False).rates("C-2", "crusher", 8.0) == (math.inf, 0.0)
    one_shift = Reliability(ref_shifts_per_day=1, machines=r.machines).rates("C-2", "crusher", 8.0)[0]
    assert one_shift == mtbf / 2  # the same failures in half the hours: a weaker machine


def test_a_breakdown_of_the_bottleneck_crusher_costs_tonnes():
    reliable = mine(**{"C-2": {"count": 0, "hours": 0}})
    weak = mine(**{"C-2": {"count": 60, "hours": 120}})  # ~ every 6 h, 2 h each
    t_ok = np.mean([simulate(reliable, seed=s).kpi["hauled_t"] for s in range(4)])
    runs = [simulate(weak, seed=s).kpi for s in range(4)]
    assert np.mean([k["hauled_t"] for k in runs]) < 0.9 * t_ok
    assert sum(k["crusher_breakdown_h"] for k in runs) > 0


def test_excavator_breakdown_stops_loading_there():
    sc = mine(**{"K-3": {"count": 300, "hours": 3000}})  # practically always broken
    k = simulate(sc, seed=1).kpi
    k3 = next(e for e in k["excavators"] if e["excavator"] == "K-3")
    assert k["excavator_down_h"]["awaria"] > 4 and k3["loaded_t"] < 1500


def test_long_repair_goes_on_into_the_next_shifts():
    sc = mine(**{"C-2": {"count": 20, "hours": 1200}})  # repairs of ~60 h on average
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=5, shifts_per_day=2)
    df = run_period(sc, plan, seed=2)
    down = df["awaria_C-2_h"].tolist()
    # somewhere a whole shift is lost and the next one starts still broken
    assert any(a >= sc.shift_hours - 0.1 and b > 0 for a, b in zip(down, down[1:]))


def test_no_breakdowns_when_switched_off():
    sc = mine()
    sc.reliability.enabled = False
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=3, shifts_per_day=2, drills=1)
    df = run_period(sc, plan, seed=4)
    assert df["awarie_h"].sum() == 0
