"""Machine catalogue: a model brings its parameters."""
import copy
from pathlib import Path

from streamlit.testing.v1 import AppTest

from kruszyk import catalog
from kruszyk.config import load_scenario
from kruszyk.sim import simulate

APP = str(Path(__file__).parent.parent / "app.py")
DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml"


def test_every_model_has_its_parameters():
    for kind in catalog.KINDS:
        for name in catalog.names(kind):
            p = catalog.params(kind, name)
            assert set(p) == set(catalog.FIELDS[kind]), (kind, name)
            assert all(v > 0 for v in p.values()), (kind, name)


def test_trucks_added_from_the_catalogue():
    at = AppTest.from_file(APP, default_timeout=300).run()
    n = len(at.session_state["raw"]["trucks"])
    at.selectbox(key="fleet_cat_model").set_value("CAT 773G (sztywnoramowe 55 t)").run()
    at.number_input(key="fleet_cat_n").set_value(2).run()
    at.button(key="fleet_cat_add").click().run()
    assert not at.exception, at.exception
    trucks = at.session_state["raw"]["trucks"]
    assert len(trucks) == n + 2
    assert trucks[-1]["payload_t"] == 55.0 and trucks[-1]["model"].startswith("CAT 773G")
    assert len({t["id"] for t in trucks}) == len(trucks)


def test_mobile_crusher_with_a_small_hopper_takes_whole_trucks():
    sc = load_scenario(DEMO)
    v = copy.deepcopy(sc)
    p = catalog.params("crusher", "Kleemann MC 110 Z EVO – szczękowa mobilna")
    for k, val in p.items():
        setattr(v.crushers[0], k, val)
    assert v.crushers[0].hopper_t < max(f.payload_t for f in v.trucks)
    k = simulate(v, seed=2).kpi
    assert k["hauled_t"] > 1000 and k["crushed_t"] > 1000
