"""Smoke test of the Streamlit panel: every scenario preset loads and renders without exceptions."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Wywiezione" in m.label for m in at.metric)


def test_every_preset_loads():
    at = AppTest.from_file(APP, default_timeout=120).run()
    for name in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].select(name).run()
        at.sidebar.button[0].click().run()
        assert not at.exception, (name, at.exception)


def test_switching_draw_mode_keeps_map():
    at = AppTest.from_file(APP, default_timeout=120).run()
    key_before = at.session_state["map_key"]
    for mode in ["Usyp (materiał)", "Koparka", "Kruszarka", "Droga / rampa"]:
        at.radio(key="draw_mode").set_value(mode).run()
        assert not at.exception, (mode, at.exception)
        assert at.session_state["map_key"] == key_before  # map not rebuilt


def test_demo_map_tabs_render():
    at = AppTest.from_file(APP, default_timeout=180).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    at.radio(key="anim_policy").set_value("dynamic").run()
    assert not at.exception, at.exception
