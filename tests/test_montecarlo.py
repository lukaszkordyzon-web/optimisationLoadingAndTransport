"""Chance that a period plan is met (Monte Carlo over random shifts)."""
import datetime as dt
from pathlib import Path

import yaml

from kruszyk.config import scenario_from_dict
from kruszyk.montecarlo import chance, run_many
from kruszyk.period import PeriodPlan

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


def setup():
    sc = scenario_from_dict(yaml.safe_load(DEMO.read_text(encoding="utf-8")))
    sc.dispatch = "natural"
    return sc, PeriodPlan(start=dt.date(2026, 11, 2), days=3, shifts_per_day=1)


def test_runs_differ_and_chance_follows_the_target():
    sc, plan = setup()
    runs = run_many(sc, plan, n=6, seed=3, workers=1)
    total = [float(df["wywiezione_t"].sum()) for df in runs]
    assert len(set(total)) > 1  # different weather / breakdowns / drivers in every run
    easy = chance(runs, plan, target_t=min(total) * 0.9)
    hard = chance(runs, plan, target_t=max(total) * 1.1)
    mid = chance(runs, plan, target_t=sorted(total)[3])
    assert easy["p"] == 1.0 and hard["p"] == 0.0 and 0 < mid["p"] < 1
    assert easy["t_p10"] <= easy["t_p50"] <= easy["t_p90"]
    assert list(easy["band"]["p50"]) == sorted(easy["band"]["p50"])  # cumulative


def test_parallel_runs_equal_sequential_ones():
    sc, plan = setup()
    seq = run_many(sc, plan, n=3, seed=5, workers=1)
    par = run_many(sc, plan, n=3, seed=5, workers=3)
    assert [float(d["wywiezione_t"].sum()) for d in seq] == [float(d["wywiezione_t"].sum()) for d in par]


def test_model_uncertainty_widens_the_spread():
    sc, plan = setup()
    sc.reliability.enabled = False  # only the calibration varies between the two sets of runs
    sc.calibration.delay_min_per_trip = 5.0
    plain = chance(run_many(sc, plan, n=6, seed=7, workers=1), plan)
    unsure = chance(run_many(sc, plan, n=6, seed=7, model_sd=0.15, workers=1), plan)
    assert unsure["t_p90"] - unsure["t_p10"] > plain["t_p90"] - plain["t_p10"]
