"""The month schedule planned by a person: drilling and blasting worked out backwards, conflicts, the frozen run."""
import copy
import datetime as dt
from pathlib import Path

from kruszyk.config import load_scenario
from kruszyk.period import PeriodPlan, run_period, schedule_of
from kruszyk.rl import today
from kruszyk.schedule import EMPTY, NOT_YET, check, frozen, plan_blasts, propose

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml"
AREAS = [{"id": "A", "blast_max_t": 12000, "t_per_m": 30, "reserves_t": 100000},
         {"id": "B", "blast_max_t": 8000, "t_per_m": 33, "reserves_t": 100000}]
RIG = [{"id": "W-1", "m_per_shift": 250}]


def test_blast_is_drilled_and_fired_just_before_the_pile_is_needed():
    grid = {"K-1": ["U-01"] * 6 + ["A/1"] * 4}
    b = plan_blasts(grid, AREAS, RIG, 10, fire_shifts=1)["A/1"]
    assert b["m"] == 400 and b["avail"] == 6 and b["fire"] == [5]  # 400 mb / 250 = 2 shifts, then the blast
    assert [x for _, x in b["drill"]] == [3, 4] and not b["late"]


def test_one_rig_cannot_drill_two_blasts_at_once_and_a_late_blast_is_a_conflict():
    grid = {"K-1": ["U-01", "U-01", "A/1", "A/1"], "K-2": ["U-02", "U-02", "B/1", "B/1"]}
    bl = plan_blasts(grid, AREAS, RIG, 4, fire_shifts=1)
    shifts = [x for b in bl.values() for _, x in b["drill"]]
    assert len(shifts) == len(set(shifts))  # never two jobs on the rig in one shift
    late = [p for p, b in bl.items() if b["late"]]
    assert late  # 3 shifts of drilling + 2 blasts do not fit in 2 shifts
    issues = check(grid, bl, {"U-01": 10000, "U-02": 10000}, {"K-1": 1000, "K-2": 1000}, 4)
    assert any(i["kind"] == NOT_YET and i["pile"] in late for i in issues)


def test_strips_of_an_area_are_fired_in_order():
    grid = {"K-1": ["U-01"] * 8 + ["A/2"] * 2}
    bl = plan_blasts(grid, AREAS, RIG, 10)
    assert "A/1" in bl and bl["A/1"]["avail"] <= bl["A/2"]["avail"]


def test_dug_out_pile_is_reported():
    grid = {"K-1": ["U-01"] * 5}
    issues = check(grid, {}, {"U-01": 2500}, {"K-1": 1000}, 5)
    assert [(i["kind"], i["shift"]) for i in issues] == [(EMPTY, 3)]


def test_proposed_schedule_frozen_month_follows_it_exactly():
    sc = load_scenario(DEMO)
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=4, shifts_per_day=2)
    n = len(plan.shifts())
    start = {b.id: b.tonnage_t for b in sc.blasts}
    rates = {e.id: 1300.0 for e in sc.excavators}
    grid = propose(sc.excavators, start, sc.areas, rates, n, sc.blasting.max_blast_t())
    assert all(len(c) == n for c in grid.values())
    bl = plan_blasts(grid, sc.areas, [{"id": "W-1", "m_per_shift": 280}], n, sc.blasting.max_blast_t())
    p = copy.deepcopy(plan)
    p.frozen = frozen(grid, bl, n)
    a = run_period(today(sc), p, seed=1)
    followed = schedule_of(a)["ex"]
    for i in range(n):  # where the plan's pile is there, the excavator is on it
        for ex, pile in p.frozen["ex"][i].items():
            if not (pile in bl and bl[pile]["avail"] > i):
                assert followed[i][ex] == pile, (i, ex, pile, followed[i])
    b = run_period(today(sc), p, seed=1)  # the frozen plan: nothing is decided on the way – runs repeat exactly
    assert list(a["wywiezione_t"]) == list(b["wywiezione_t"])
