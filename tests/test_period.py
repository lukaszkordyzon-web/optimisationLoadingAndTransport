"""Week / month: shifts in a row with the mine state carried over."""
import datetime as dt
from pathlib import Path

from kruszyk.config import load_scenario
from kruszyk.period import PeriodPlan, PileBlasts, run_period, summary

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


def test_calendar_working_days_and_shifts():
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=7, shifts_per_day=2, holidays=[dt.date(2026, 11, 4)])
    assert [d.day for d in plan.working_days()] == [2, 3, 5, 6]  # Mon-Fri without the holiday
    assert len(plan.shifts()) == 8
    assert len(PeriodPlan(start=dt.date(2026, 11, 2), days=7, saturdays=True).working_days()) == 6


def test_piles_run_out_and_blasts_refill_them():
    sc = load_scenario(DEMO)
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=7, shifts_per_day=2,
                      blasts=[PileBlasts("U-02", 10_000, every_days=2, first_day=2)], target_t=60_000)
    df = run_period(sc, plan, seed=3)
    assert len(df) == 10
    start = {b.id: b.tonnage_t for b in sc.blasts}
    # material is carried over: what is left = start + blasts - hauled from that pile
    for p in start:
        added = 10_000 * df["strzal"].str.contains(p).sum() if p == "U-02" else 0
        assert abs(df[f"zostalo_{p}_t"].iloc[-1] - (start[p] + added - df[f"z_{p}_t"].sum())) <= 2
    assert df["strzal"].str.contains("U-02").sum() == 2  # 5 working days: blasts on day 2 and 4
    assert (df["wywiezione_t"] >= 0).all()
    assert (df["skonczyl_sie_urobek"] != "").any()  # U-01 is never re-blasted, it runs out
    s = summary(df, plan)
    assert s["wywiezione_t"] == round(df["wywiezione_t"].sum()) and s["realizacja_celu_pct"] > 0


def test_shift_without_material_does_not_crash():
    sc = load_scenario(DEMO)
    sc.areas, sc.planned_blasts = [], []
    for b in sc.blasts:
        b.tonnage_t = 0.0
    df = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=1, shifts_per_day=1), seed=1)
    assert df["wywiezione_t"].iloc[0] == 0


def test_planned_blasts_appear_and_idle_excavators_move_there():
    sc = load_scenario(DEMO)
    sc.areas = []  # only the blasts planned on the map
    assert {p["id"] for p in sc.planned_blasts} == {"S-01", "S-02"}
    df = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=7, shifts_per_day=2), seed=1)
    assert df["strzal"].str.contains("S-01").any() and df["strzal"].str.contains("S-02").any()
    moves = df.loc[df["przejazdy_koparek"] != "", "przejazdy_koparek"]
    assert len(moves) and moves.str.contains("→ S-0").all()
    assert df["z_S-01_t"].sum() > 0  # material from the new pile reached the crusher
    still = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=7, shifts_per_day=2, move_excavators=False), seed=1)
    assert (still["przejazdy_koparek"] == "").all() and still["z_S-01_t"].sum() == 0
    assert df["wywiezione_t"].sum() > still["wywiezione_t"].sum()


def test_feed_shares_only_from_piles_that_can_be_loaded():
    sc = load_scenario(DEMO)
    for p in sc.planned_blasts:
        p["target_share"] = 25.0
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=7, shifts_per_day=2,
                      blasts=[PileBlasts("U-01", 9000, every_days=3, first_day=2, share=50.0)])
    df = run_period(sc, plan, seed=1)
    first = df["sklad_cel"].iloc[0]
    assert "U-01 46%" in first and "S-01" not in first  # S-01 not blasted yet -> not in the target mix
    later = df.loc[df["koparki_na"].str.contains("S-01"), "sklad_cel"]
    assert later.str.contains("S-01").any()  # once blasted and worked, it gets its share
    s = summary(df, plan)
    assert abs(sum(s["udzialy_pct"].values()) - 100) < 0.5


def test_goal_ranks_plans():
    from kruszyk.period import optimize_period

    sc = load_scenario(DEMO)
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=4, blasts=[PileBlasts(b.id, b.tonnage_t, 2, 2) for b in sc.blasts])
    cheap = optimize_period(sc, plan, "cheapest", [5, 9], [1, 2], [False], ["dynamic"], seed=1)
    assert len(cheap) == 4 and list(cheap["koszt_na_t"]) == sorted(cheap["koszt_na_t"])
    most = optimize_period(sc, plan, "max", [5, 9], [1, 2], [False], ["dynamic"], seed=1)
    assert most["wywiezione_t"].iloc[0] == most["wywiezione_t"].max()
    dl = optimize_period(sc, plan, "deadline", [5, 9], [1, 2], [False], ["dynamic"], target_t=20_000,
                         deadline=dt.date(2026, 11, 5), seed=1)
    best = dl.iloc[0]
    assert best["cel_osiagniety"] is not None and best["cel_osiagniety"] <= dt.date(2026, 11, 5)
    ok = dl[dl["cel_osiagniety"].notna()]
    assert best["koszt_na_t"] == ok["koszt_na_t"].min()  # the cheapest of the plans that make it in time


def test_extraction_area_is_blasted_automatically_strip_by_strip():
    from kruszyk.site import area_strips, polygon_area_m2

    sc = load_scenario(DEMO)
    area = sc.areas[0]
    strips = area_strips(sc.site, area["polygon"], 4)
    assert len(strips) == 4
    assert abs(sum(polygon_area_m2(s) for s in strips) - polygon_area_m2(area["polygon"])) < 0.02 * polygon_area_m2(area["polygon"])
    for p in sc.planned_blasts:  # only the area this time
        p["blast_day"] = 999
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=14, shifts_per_day=2)
    df = run_period(sc, plan, seed=1)
    auto = [b for b in df.attrs["blasts"] if b["auto"]]
    assert len(auto) >= 2 and [b["strzal"] for b in auto][:2] == ["O1/1", "O1/2"]
    assert all(b["t"] <= area["blast_max_t"] for b in auto)
    days = [b["data"] for b in auto]
    wd = {d: i for i, d in enumerate(plan.working_days(), start=1)}
    assert all(wd[b] - wd[a] >= area["min_gap_days"] for a, b in zip(days, days[1:]))  # gap respected
    assert sum(b["t"] for b in auto) <= area["reserves_t"]
    assert df[[c for c in df.columns if c.startswith("z_O1/")]].sum().sum() > 0  # material reached the crushers
    # the blast closes the pit at the area's own hours: fired on the first shift of that day
    first = df[(df["data"] == days[0]) & (df["zmiana"] == 1)]
    assert first["strzal"].str.contains("O1/1").all()


def test_drilling_limits_blasts():
    """Blasts in new places wait for drilling; the next strip of an area only after the previous blast is taken out."""
    sc = load_scenario(DEMO)
    sc.planned_blasts = []
    free = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=14, drills=0), seed=1)
    slow = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=14, drills=1, drill_m_per_shift=60), seed=1)
    n_free = len([b for b in free.attrs["blasts"] if b["auto"]])
    n_slow = len([b for b in slow.attrs["blasts"] if b["auto"]])
    assert n_slow < n_free  # slow drilling -> fewer blasts
    drilled = {d["strzal"]: d for d in slow.attrs["drilling"]}
    for b in slow.attrs["blasts"]:
        if b["strzal"] != "O1/1":  # the first strip is drilled already
            assert b["strzal"] in drilled and drilled[b["strzal"]]["data"] <= b["data"]
    # metres: 20 000 t / 50 t/mb = 400 m -> at 60 m per shift a strip takes ~7 shifts
    assert (slow["wiercenie"] != "").sum() >= 6


def test_lower_bench_waits_for_the_upper_bench():
    sc = load_scenario(DEMO)
    sc.planned_blasts = []
    up = sc.areas[0]
    up["level_m"] = -50.0
    low = dict(up, id="O2", level_m=-65.0, reserves_t=40000,
               polygon=[[c[0] - 0.0002, c[1]] for c in up["polygon"]])  # right below the upper area
    far = dict(up, id="O3", level_m=-65.0, reserves_t=40000,
               polygon=[[c[0] - 0.003, c[1]] for c in up["polygon"]])  # 300 m away: no constraint
    sc.areas += [low, far]
    df = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=7, lead_m=30), seed=1)
    fired = [b["strzal"] for b in df.attrs["blasts"]]
    assert "O3/1" in fired and "O2/1" not in fired
    assert any("O2/1" in w for w in df.attrs["waits"])
    free = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=7, lead_m=0), seed=1)
    assert "O2/1" in [b["strzal"] for b in free.attrs["blasts"]] or len(free.attrs["waits"]) < len(df.attrs["waits"])
