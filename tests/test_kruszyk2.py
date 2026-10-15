"""Kruszyk 2: mine → schedule → the same month driven today and with the driving signals."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "kruszyk2.py")


def test_whole_flow_on_the_start_mine():
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["k2_quick"] = True  # a short learning – the flow, not the result, is tested here
    at.session_state["k2_plan"] = {"start": "2026-11-02", "days": 3, "shifts": 2, "sat": False, "hol": "", "fire": 1,
                                   "real_t": 0.0}
    at.run()
    assert not at.exception, at.exception
    at.radio(key="k2_mine_tab").set_value("Maszyny").run()
    assert not at.exception, at.exception
    at.radio(key="k2_nav").set_value("② Harmonogram").run()
    assert not at.exception, at.exception
    work = at.session_state["work"]
    assert set(work["ex"]) == {"K-1", "K-2"} and all(len(c) == 6 for c in work["ex"].values())
    assert set(work["dr"]) == {"W-1", "W-2"}
    assert len(work["tr"]) == 6 and all(x in ("K-1", "K-2") for c in work["tr"].values() for x in c)
    at.button(key="k2_approve").click().run()
    assert at.session_state["k2_nav"] == "③ Symulacja"
    at.button(key="k2_month_go").click().run()
    assert not at.exception, at.exception
    m = at.session_state["month"]
    assert len(m["today"]) == len(m["signals"]) == 6
    assert list(m["today"]["koparki_na"]) == list(m["signals"]["koparki_na"])  # the same frozen schedule
    at.radio(key="k2_nav").set_value("④ Wynik").run()
    assert not at.exception, at.exception
    assert any("Koszt tony" in x.label for x in at.metric)


def test_schedule_editing_sets_area_tonnes_and_trucks():
    at = AppTest.from_file(APP, default_timeout=600)
    at.session_state["k2_plan"] = {"start": "2026-11-02", "days": 3, "shifts": 2, "sat": False, "hol": "", "fire": 1,
                                   "real_t": 0.0}
    at.run()
    at.radio(key="k2_nav").set_value("② Harmonogram").run()
    at.selectbox(key="k2_e_ex").set_value("K-2").run()
    at.selectbox(key="k2_e_from").set_value(2).run()
    at.selectbox(key="k2_e_area").set_value("O1").run()
    at.number_input(key="k2_e_t").set_value(1200.0).run()
    at.button(key="k2_e_set").click().run()
    assert not at.exception, at.exception
    assert at.session_state["work"]["ex"]["K-2"][2:] == [{"a": "O1", "t": 1200.0}] * 4
    at.multiselect(key="k2_t_ids").set_value(["WZ-01", "WZ-02"]).run()
    at.selectbox(key="k2_t_ex").set_value("nie jeździ").run()
    at.button(key="k2_t_set").click().run()
    assert not at.exception, at.exception
    assert at.session_state["work"]["tr"]["WZ-01"] == [""] * 6
    at.selectbox(key="k2_d_ex").set_value("W-2").run()
    at.selectbox(key="k2_d_area").set_value("O1").run()
    at.number_input(key="k2_d_t").set_value(5000.0).run()
    at.button(key="k2_d_set").click().run()
    assert not at.exception, at.exception
    assert at.session_state["work"]["dr"]["W-2"] == [{"a": "O1", "t": 5000.0}] * 6
