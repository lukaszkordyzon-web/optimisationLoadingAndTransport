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
    for b in sc.blasts:
        b.tonnage_t = 0.0
    df = run_period(sc, PeriodPlan(start=dt.date(2026, 11, 2), days=1, shifts_per_day=1), seed=1)
    assert df["wywiezione_t"].iloc[0] == 0


def test_planned_blasts_appear_and_idle_excavators_move_there():
    sc = load_scenario(DEMO)
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
