"""Smoke test of the Streamlit panel: every scenario preset loads and renders without exceptions."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "app.py")


def advanced_app(timeout: int = 120) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout).run()
    at.toggle(key="advanced").set_value(True).run()
    return at


def test_app_renders_without_errors():
    at = advanced_app()
    assert not at.exception, at.exception
    assert any("Wywiezione" in m.label for m in at.metric)


def test_every_preset_loads():
    at = advanced_app()
    for name in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].select(name).run()
        at.sidebar.button[0].click().run()
        assert not at.exception, (name, at.exception)


def test_switching_draw_mode_keeps_map():
    at = advanced_app()
    key_before = at.session_state["map_key"]
    for group, mode in [("⛏️ Urobek i koparki", "Usyp (materiał)"), ("⛏️ Urobek i koparki", "Koparka"),
                        ("🛣️ Drogi", "Kruszarka"), ("🛣️ Drogi", "Droga / rampa"), ("📅 Plan strzałów", None),
                        ("⛏️ Urobek i koparki", "Kierunek urabiania koparki")]:
        at.radio(key="draw_group").set_value(group).run()
        if mode:
            gi = ["🛣️ Drogi", "⛏️ Urobek i koparki"].index(group)
            at.radio(key=f"draw_mode_{gi}").set_value(mode).run()
        assert not at.exception, (mode, at.exception)
        assert at.session_state["map_key"] == key_before  # map not rebuilt


def test_demo_map_tabs_render():
    at = advanced_app(180)
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    anim = next(r for r in at.radio if r.label == "Przydział wozideł w animacji")
    assert anim.value == "natural"  # follows the scenario's dispatch
    anim.set_value("dynamic").run()
    assert not at.exception, at.exception


def test_applied_rl_policy_is_shown_in_animation():
    import yaml

    at = advanced_app(180)
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    # what "Zastosuj politykę RL" does: weights + dispatch rl into the scenario, new widget generation
    raw = dict(at.session_state["raw"])
    raw["rl_weights"] = [3.0, 2.0, 0.5, 1.0, 0.0, 0.5, 0.2]
    raw["dispatch"] = "rl"
    at.session_state["raw"] = raw
    at.session_state["ver"] = at.session_state["ver"] + 1
    at.run()
    assert not at.exception, at.exception
    anim = next(r for r in at.radio if r.label == "Przydział wozideł w animacji")
    assert anim.value == "rl"


def test_simple_mode_every_preset():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    for step in ["① Kopalnia dziś", "② Organizacja zmiany", "③ Kalibracja", "④ Jak było dziś", "⑤ Optymalizacja RL",
                 "🧨 Strzelanie", "👷 Objazd sztygara", "⛽ Tankowanie"]:
        assert step in labels
    for name in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].select(name).run()
        at.sidebar.button[0].click().run()
        assert not at.exception, (name, at.exception)


def test_simple_mode_demo_shows_today():
    at = AppTest.from_file(APP, default_timeout=180).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Koszt na tonę" in labels and "Czekanie w kolejkach" in labels
    assert any("Gdzie tracimy" in m.value for m in at.markdown)


def test_simple_mode_variant_compares_with_today():
    at = AppTest.from_file(APP, default_timeout=180).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    at.radio(key="work_kind").set_value("Planowanie produkcji").run()
    next(b for b in at.button if "Utwórz wariant" in b.label).click().run()
    assert not at.exception, at.exception
    assert at.session_state["variant_yaml"]
    assert any("vs dziś" in (m.delta or "") for m in at.metric)


def test_scenario_file_roundtrip():
    import yaml

    at = AppTest.from_file(APP, default_timeout=180).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    raw = at.session_state["raw"]
    # the same data the download button produces, loaded back as a new mine
    text = yaml.safe_dump({**raw, "variant": raw, "_plik": {"format": "kruszyk-1"}}, allow_unicode=True)
    data = yaml.safe_load(text)
    assert data["site"]["roads"] and data["variant"]["name"] == raw["name"]
    assert any("Zapisz scenariusz" in b.proto.label for b in at.get("download_button"))


def test_map_edit_mode_select_and_delete():
    at = AppTest.from_file(APP, default_timeout=180).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    at.radio(key="draw_group").set_value("✏️ Edycja").run()
    assert not at.exception, at.exception
    site = at.session_state["site"]
    ex = site["excavators"][0]
    at.selectbox(key="sel_box").set_value(f"excavator:{ex['uid']}").run()
    assert not at.exception, at.exception
    n = len(site["excavators"])
    next(b for b in at.button if b.label == "🗑️ Usuń").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["site"]["excavators"]) == n - 1
    road = at.session_state["site"]["roads"][0]
    g = road.get("grade_draw_pct", 0.0)
    at.selectbox(key="sel_box").set_value(f"road:{road['uid']}").run()
    next(b for b in at.button if b.label.startswith("⇅")).click().run()
    assert not at.exception, at.exception
    assert at.session_state["site"]["roads"][0]["grade_draw_pct"] == -g + 0.0


def test_planning_only_in_schedule_mode():
    at = AppTest.from_file(APP, default_timeout=300).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    at.radio(key="draw_group").set_value("⛏️ Urobek i koparki").run()
    assert not at.exception, at.exception
    assert not [r for r in at.radio if r.key == "pile_status"]  # one shift: piles only as they lie today
    assert "📅 Plan strzałów" not in at.radio(key="draw_group").options
    at.radio(key="work_kind").set_value("Planowanie produkcji").run()
    at.radio(key="draw_group").set_value("⛏️ Urobek i koparki").run()
    assert not at.exception, at.exception
    assert [r for r in at.radio if r.key == "pile_status"]  # schedule: a pile can be a planned blast
    assert "📅 Plan strzałów" in at.radio(key="draw_group").options


def test_advanced_editor_keeps_routes_to_second_crusher():
    """The excavator table must not turn map-built routes to other crushers into text (TypeError in the sim)."""
    import yaml

    from kruszyk.site import site_to_scenario

    raw = yaml.safe_load((Path(APP).parent / "scenarios" / "kruszyk_mapa_demo.yaml").read_text(encoding="utf-8"))
    s = raw["site"]
    s["crushers"] = [{"id": "C-2", "lat": 49.9946, "lon": 20.0240, "capacity_tph": 900, "hopper_t": 120,
                      "accepts": ["all"], "product": "tłuczeń 31,5–63"}]
    raw = site_to_scenario(s, raw)
    assert any(e.get("crusher_routes") for e in raw["excavators"])
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["raw"] = raw
    at.session_state["ver"] = 1
    at.run()
    at.toggle(key="advanced").set_value(True).run()
    assert not at.exception, at.exception
    assert all(isinstance(e.get("crusher_routes", {}), dict) for e in at.session_state["raw"]["excavators"])


def test_schedule_compares_with_rl_when_learned():
    at = AppTest.from_file(APP, default_timeout=300).run()
    name = next(o for o in at.sidebar.selectbox[0].options if "mapa demo" in o)
    at.sidebar.selectbox[0].select(name).run()
    at.sidebar.button[0].click().run()
    at.radio(key="work_kind").set_value("Planowanie produkcji").run()
    assert not at.exception, at.exception
    assert not [t for t in at.toggle if t.key == "per_rl"]  # nothing learned yet: only a hint where to learn
    raw = dict(at.session_state["raw"])
    raw["rl_weights"] = [3.0, 2.0, 0.5, 1.0, 0.0, 0.5, 0.2]  # what step ③ of the shift module stores in the mine
    at.session_state["raw"] = raw
    at.run()
    assert at.toggle(key="per_rl").value
    at.number_input(key="per_days").set_value(2).run()
    next(b for b in at.button if "Policz okres" in b.label).click().run()
    assert not at.exception, at.exception
    assert at.session_state["period"][3] is not None
    assert any("Wywiezione z RL" in m.label for m in at.metric)


def test_production_goal_backwards_and_applied():
    at = AppTest.from_file(APP, default_timeout=300).run()  # the default mine
    at.radio(key="work_kind").set_value("Planowanie produkcji").run()
    assert not at.exception, at.exception
    at.number_input(key="per_days").set_value(3).run()
    next(b for b in at.button if "Policz plan wstecz" in b.label).click().run()
    assert not at.exception, at.exception
    gp = at.session_state["prod_gp"][1]
    assert gp.feed_t > 0 and gp.products
    next(b for b in at.button if "Zastosuj w harmonogramie" in b.label).click().run()
    assert not at.exception, at.exception
    assert at.number_input(key="per_target").value == float(gp.feed_t)
    assert any("Zastosowany cel produkcyjny" in s.value for s in at.success)
    next(b for b in at.button if "Policz okres" in b.label).click().run()
    assert not at.exception, at.exception
    assert any("Cel produkcyjny – wykonanie" in m.value for m in at.markdown)


def test_calibration_screen_fits_and_saves():
    at = AppTest.from_file(APP, default_timeout=300).run()  # the default mine, shift module
    assert any("nieskalibrowany" in w.value for w in at.warning)
    at.number_input(key="cal_tonnes_t").set_value(3500).run()
    at.number_input(key="cal_trips").set_value(95).run()
    next(b for b in at.button if "Dopasuj model" in b.label).click().run()
    assert not at.exception, at.exception
    assert any("Model po kalibracji" in s.value for s in at.success)
    next(b for b in at.button if "Zapisz kalibrację" in b.label).click().run()
    assert not at.exception, at.exception
    assert at.session_state["raw"]["calibration"]["measured"]["tonnes_t"] == 3500
    assert any("Model skalibrowany" in c.value for c in at.caption)


def test_chance_of_the_plan():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="work_kind").set_value("Planowanie produkcji").run()
    at.number_input(key="per_days").set_value(2).run()
    at.number_input(key="per_target").set_value(8000.0).run()
    at.selectbox(key="mc_n").set_value(10).run()
    next(b for b in at.button if "Policz szansę" in b.label).click().run()
    assert not at.exception, at.exception
    assert any(m.label.startswith("Szansa:") for m in at.metric)


def test_clicked_object_can_be_edited():
    at = AppTest.from_file(APP, default_timeout=300).run()  # the default mine, map in step ①
    site = at.session_state["site"]
    pile = site["piles"][0]
    old_id = pile["id"]
    at.session_state["info_sel"] = f"pile:{pile['uid']}"
    at.run()
    assert not at.exception, at.exception
    at.number_input(key=f"of_pile:{pile['uid']}_tonnage_t").set_value(12345.0)
    at.text_input(key=f"of_pile:{pile['uid']}_id").set_value("U-99")
    next(b for b in at.button if "Zastosuj zmiany" in b.label).click().run()
    assert not at.exception, at.exception
    edited = next(p for p in at.session_state["site"]["piles"] if p["uid"] == pile["uid"])
    assert edited["tonnage_t"] == 12345.0 and edited["id"] == "U-99"
    exs = at.session_state["site"]["excavators"]
    assert all(e.get("pile") != old_id for e in exs) and any(e.get("pile") == "U-99" for e in exs)  # references renamed
    road = at.session_state["site"]["roads"][0]
    one_lane = bool(road.get("single_lane", False))
    at.session_state["info_sel"] = f"road:{road['uid']}"
    at.run()
    at.checkbox(key=f"of_road:{road['uid']}_single_lane").set_value(not one_lane)
    next(b for b in at.button if "Zastosuj zmiany" in b.label).click().run()
    assert not at.exception, at.exception
    assert next(r for r in at.session_state["site"]["roads"] if r["uid"] == road["uid"])["single_lane"] != one_lane


def test_every_kind_of_object_opens_its_form():
    at = AppTest.from_file(APP, default_timeout=300).run()
    site = at.session_state["site"]
    sels = [f"road:{site['roads'][0]['uid']}", f"pile:{site['piles'][0]['uid']}",
            f"excavator:{site['excavators'][0]['uid']}", f"area:{site['areas'][0]['uid']}", "crusher:main",
            f"crusher2:{site['crushers'][0]['uid']}", f"bay:{site['passing_bays'][0]['uid']}",
            f"parking:{site['parking'][0]['excavator']}"]
    for sel in sels:
        at.session_state["info_sel"] = sel
        at.run()
        assert not at.exception, (sel, at.exception)
        assert any("Zastosuj zmiany" in b.label for b in at.button), sel
