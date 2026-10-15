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


def test_tonnes_the_schedule_gives():
    from kruszyk.schedule import hauled_plan

    grid = {"K-1": ["U-01"] * 4}
    assert hauled_plan(grid, {}, {"U-01": 2500}, {"K-1": 1000}, 4) == [1000, 1000, 500, 0]  # no more than lies there


def test_work_plan_numbers_piles_of_an_area_and_blasts_them_in_time():
    from kruszyk import workplan

    work = {"start": {"A": 3000.0}, "ex": {"K-1": [{"a": "A", "t": 1000.0}] * 10}, "tr": {"WZ-01": ["K-1"] * 10}}
    ev = workplan.evaluate(work, AREAS, RIG, 10, fire=1)
    assert ev["grid"]["K-1"][:3] == ["A/1"] * 3 and ev["grid"]["K-1"][3] == "A/2"  # 3000 t lie there: 3 shifts
    assert ev["blasts"]["A/2"]["avail"] == 3 and not ev["blasts"]["A/2"]["late"]  # fired just in time
    assert ev["planned"] == [1000.0] * 10 and not ev["issues"]
    fr = workplan.frozen(work, ev, 10)
    assert fr["quota"][0] == {"K-1": 1000.0} and fr["trucks"][0] == {"WZ-01": "K-1"} and fr["blasts"][3] == ["A/2"]


def test_work_plan_issues():
    from kruszyk import workplan

    work = {"start": {"A": 0.0}, "ex": {"K-1": [{"a": "A", "t": 3000.0}] * 2, "K-2": [{"a": "", "t": 0.0}] * 2},
            "tr": {"WZ-01": ["K-2"] * 2}}
    ev = workplan.evaluate(work, AREAS, RIG, 2, fire=1, rates={"K-1": 1500.0})
    kinds = {i["kind"] for i in ev["issues"]}
    assert {workplan.LATE, workplan.OVER, workplan.NO_TRUCKS, workplan.IDLE_TRUCK} <= kinds


def test_month_follows_the_work_plan():
    from kruszyk import workplan
    import yaml

    raw = yaml.safe_load(DEMO.read_text(encoding="utf-8"))
    sc = load_scenario(DEMO)
    n, cap = 4, sc.blasting.max_blast_t()
    work = {"start": {"O2": 3000.0, "O3": 7000.0},
            "ex": {"K-1": [{"a": "O2", "t": 1000.0}] * n, "K-2": [{"a": "O3", "t": 1500.0}] * 2 + [{"a": "", "t": 0}] * 2},
            "tr": {f.id: (["K-1"] * n if j < 3 else ["K-2"] * 2 + [""] * 2) for j, f in enumerate(sc.trucks)}}
    ev = workplan.evaluate(work, sc.areas, raw["drills"], n, cap)
    from kruszyk.config import scenario_from_dict

    sim = scenario_from_dict(workplan.scenario_raw(raw, work, ev["grid"], cap))
    p = PeriodPlan(start=dt.date(2026, 11, 2), days=2, shifts_per_day=2)
    p.frozen = workplan.frozen(work, ev, n)
    df = run_period(today(sim), p, seed=1)
    assert all(df["wywiezione_t"] <= 2500 + 40)  # never more than the plan's tonnes (+ the last truckload)
    assert df["z_O3/1_t"].iloc[2:].sum() == 0  # K-2 stands on the 2nd day
    assert df["z_O2/1_t"].sum() + df.get("z_O2/2_t", 0 * df["z_O2/1_t"]).sum() > 0
