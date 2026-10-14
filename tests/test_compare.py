"""Scenario comparison on many shifts and the fleet table."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "walidacja"))
from kopalnia_testowa import make  # noqa: E402

from kruszyk.compare import compare, fleet_sweep, sign_threshold  # noqa: E402

APP = str(Path(__file__).parent.parent / "app.py")


def test_sign_threshold():
    assert [sign_threshold(n) for n in (10, 20, 30)] == [9, 15, 20]


def test_same_scenario_shows_no_difference():
    r = compare(make(), make(), n=10, seed=3)
    assert r["cost_verdict"] == r["t_verdict"] == "bez pewnej różnicy"
    assert r["cost_better"] == r["cost_worse"] == 0 and abs(r["d_cost_pct"]) < 1e-9


def test_extra_truck_at_a_full_crusher_costs_more():
    r = compare(make(trucks=3), make(trucks=5), n=20, seed=3)
    assert r["cost_verdict"] == "gorzej" and r["d_cost_pct"] > 3
    assert abs(r["d_t_pct"]) < 3  # (almost) the same tonnes: the crusher is full


def test_bigger_crusher_hauls_more():
    r = compare(make(trucks=4), make(trucks=4, crusher_tph=600), n=10, seed=3)
    assert r["t_verdict"] == "lepiej" and r["d_t_pct"] > 15


def test_fleet_table_finds_what_an_engineer_finds():
    tab, rec = fleet_sweep(make(), [1, 2, 3, 4, 5, 6], n=10, seed=3)
    assert list(tab["wozidla"]) == [1, 2, 3, 4, 5, 6]
    assert rec["najtansza"] in (2, 3) and rec["pelna_produkcja"] == 3 and rec["plateau"]
    assert tab["kolejka_kruszarka_h"].is_monotonic_increasing


def test_variant_screen_compares_and_sizes_the_fleet():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="work_kind").set_value("Planowanie produkcji").run()
    next(b for b in at.button if "Utwórz wariant" in b.label).click().run()
    at.selectbox(key="vcmp_n").set_value(10).run()
    at.button(key="vcmp_go").click().run()
    assert not at.exception, at.exception
    verdicts = [m.value for m in at.markdown if "koszt na tonę:" in m.value]
    assert verdicts and "bez pewnej różnicy" in verdicts[0]  # an unchanged copy = no difference at all
    assert "wariant gorszy" not in verdicts[0] and "wariant lepszy" not in verdicts[0]
    shows = at.selectbox(key="vshow_which")
    shows.set_value(shows.options[1]).run()  # the shift where the variant gains most: on the map
    assert not at.exception, at.exception
    at.slider(key="vfleet_range").set_value((3, 6)).run()
    at.button(key="vfleet_go").click().run()
    assert not at.exception, at.exception
    assert any("Najtańsza tona" in s.value for s in at.success)
