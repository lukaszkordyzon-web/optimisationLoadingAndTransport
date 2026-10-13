"""Several crushers: each takes material only from chosen piles; trucks pick a crusher that accepts the load."""
import datetime as dt
from pathlib import Path

import pytest
import yaml

from kruszyk.config import scenario_from_dict
from kruszyk.period import PeriodPlan, run_period, summary
from kruszyk.sim import simulate
from kruszyk.site import site_to_scenario

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


def two_crushers(main_accepts, extra_accepts, target=4000.0):
    raw = yaml.safe_load(DEMO.read_text())
    s = raw["site"]
    s["crushers"] = [{"id": "C-2", "lat": 49.9946, "lon": 20.0240, "capacity_tph": 900, "hopper_t": 120,
                      "accepts": extra_accepts, "product": "tłuczeń 31,5–63", "target_t": target}]
    s["crusher"]["accepts"] = main_accepts
    s["crusher"]["product"] = "kruszywo 0–31,5"
    return scenario_from_dict(site_to_scenario(s, raw))


def test_material_goes_only_to_crushers_that_accept_it():
    sc = two_crushers(["U-01", "U-02", "S-02"], ["U-03", "S-01"])
    assert all("C-2" in e.crusher_routes for e in sc.excavators)
    for dispatch in ("natural", "dynamic"):
        sc.dispatch = dispatch
        r = simulate(sc, seed=1)
        done = r.loads.dropna(subset=["dumped_at_s"])
        pairs = set(zip(done["blast"], done["crusher"]))
        assert pairs <= {("U-01", "C-1"), ("U-02", "C-1"), ("U-03", "C-2")}
        c2 = next(c for c in r.kpi["crushers"] if c["crusher"] == "C-2")
        assert c2["crushed_t"] > 0 and c2["product"] == "tłuczeń 31,5–63" and c2["plan_pct"] is not None
        assert r.kpi["cost_breakdown"]["kruszarka"] == pytest.approx(2 * sc.costs.crusher_per_h * sc.shift_hours)


def test_crusher_taking_everything_shares_the_load():
    sc = two_crushers([], [])
    sc.dispatch = "dynamic"
    r = simulate(sc, seed=2)
    by = {c["crusher"]: c["crushed_t"] for c in r.kpi["crushers"]}
    assert by["C-1"] > 0 and by["C-2"] > 0


def test_no_crusher_for_a_pile_is_an_error():
    with pytest.raises(ValueError, match="Żadna kruszarka nie przyjmuje"):
        two_crushers(["U-01"], ["U-02"])


def test_month_with_two_crushers_reports_plan():
    sc = two_crushers(["U-01", "U-02", "S-02"], ["U-03", "S-01"])
    plan = PeriodPlan(start=dt.date(2026, 11, 2), days=3)
    df = run_period(sc, plan, seed=1)
    assert "kr_C-2_t" in df and df["kr_C-2_t"].sum() > 0
    rows = {r["kruszarka"]: r for r in summary(df, plan, sc.crushers)["kruszarki"]}
    assert rows["C-2"]["plan_t"] == 4000 * len(df) and rows["C-2"]["wykonanie_pct"] > 0


def test_product_sets_energy_per_tonne():
    from kruszyk.config import PRODUCTS

    raw = yaml.safe_load(DEMO.read_text())
    s = raw["site"]
    s["crushers"] = [{"id": "C-2", "lat": 49.9946, "lon": 20.0240, "capacity_tph": 900, "hopper_t": 120,
                      "accepts": ["U-03"], "product": "piasek łamany 0–2", "kwh_per_t": PRODUCTS["piasek łamany 0–2"]}]
    s["crusher"]["accepts"] = ["U-01", "U-02", "S-01", "S-02", "O1"]
    sc = scenario_from_dict(site_to_scenario(s, raw))
    assert sc.crushers_extra[0].kwh_per_t == PRODUCTS["piasek łamany 0–2"] > sc.crusher.kwh_per_t
    by = {c["crusher"]: c for c in simulate(sc, seed=1).kpi["crushers"]}
    assert by["C-2"]["kwh_per_t"] > 1.5 * by["C-1"]["kwh_per_t"]  # fine product = much more energy per tonne
