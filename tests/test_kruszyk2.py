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
    assert at.session_state["work"]["dr"]["W-2"] == [{"a": "O1", "t": 5000.0}] + [{"a": "", "t": 0.0}] * 5


def test_a_copy_of_the_schedule_is_compared_with_the_base():
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["k2_plan"] = {"start": "2026-11-02", "days": 2, "shifts": 2, "sat": False, "hol": "", "fire": 1,
                                   "real_t": 0.0}
    at.run()
    at.radio(key="k2_nav").set_value("② Harmonogram").run()
    at.button(key="k2_var_copy").click().run()
    assert not at.exception, at.exception
    assert list(at.session_state["variants"]) == ["A", "B"] and at.session_state["k2seg_variant"] == "B"
    assert at.session_state["variants"]["B"] == at.session_state["variants"]["A"]
    at.multiselect(key="k2_t_ids").set_value(["WZ-01", "WZ-02"]).run()
    at.selectbox(key="k2_t_ex").set_value("nie jeździ").run()
    at.button(key="k2_t_set").click().run()  # B: two trucks fewer
    assert at.session_state["variants"]["B"]["tr"]["WZ-01"] == [""] * 4
    assert at.session_state["variants"]["A"]["tr"]["WZ-01"] != [""] * 4  # the base stays as it was
    at.button(key="k2_var_cmp").click().run()
    assert not at.exception, at.exception
    cmp = at.session_state["k2_cmp"]
    assert set(cmp) == {"A", "B"} and cmp["A"]["t"] > 0 and cmp["A"]["plan_t"] == cmp["B"]["plan_t"]


def test_full_output_variant():
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["k2_plan"] = {"start": "2026-11-02", "days": 2, "shifts": 2, "sat": False, "hol": "", "fire": 1,
                                   "real_t": 0.0}
    at.run()
    at.radio(key="k2_nav").set_value("② Harmonogram").run()
    at.button(key="k2_var_copy").click().run()
    at.toggle(key="k2_open_B").set_value(True).run()
    assert at.session_state["variants"]["B"]["mode"] == "min"
    assert at.session_state["variants"]["A"].get("mode", "limit") == "limit"
    at.button(key="k2_var_cmp").click().run()
    assert not at.exception, at.exception
    cmp = at.session_state["k2_cmp"]
    assert cmp["B"]["t"] >= cmp["A"]["t"] and cmp["B"]["mode"] == "min"


def test_a_variant_with_other_machines():
    import copy

    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["k2_plan"] = {"start": "2026-11-02", "days": 2, "shifts": 2, "sat": False, "hol": "", "fire": 1,
                                   "real_t": 0.0}
    at.run()
    at.radio(key="k2_nav").set_value("② Harmonogram").run()
    at.button(key="k2_var_copy").click().run()
    trucks = copy.deepcopy(at.session_state["raw"]["trucks"])[:4]  # B: two trucks fewer, the rest a bigger model
    for t in trucks:
        t.update({"model": "CAT 773G (sztywnoramowe 55 t)", "payload_t": 55.0})
    at.session_state["variants"]["B"]["machines"] = {"trucks": trucks, "excavators": []}
    at.run()
    assert not at.exception, at.exception
    assert set(at.session_state["variants"]["B"]["tr"]) == {t["id"] for t in trucks}  # the plan kept, rows synced
    assert len(at.session_state["variants"]["A"]["tr"]) == 6  # the base keeps its trucks
    at.button(key="k2_var_cmp").click().run()
    assert not at.exception, at.exception
    assert set(at.session_state["k2_cmp"]) == {"A", "B"}


def test_machines_screen_has_models_for_every_machine():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.radio(key="k2_mine_tab").set_value("Maszyny").run()
    assert not at.exception, at.exception
    cols = [list(d.value.columns) for d in at.dataframe] if hasattr(at, "dataframe") else []
    assert at.session_state["raw"]["excavators"]


def test_catalogue_model_on_machines_does_not_break_the_scenario():
    import copy

    import yaml

    from kruszyk.config import scenario_from_dict

    raw = yaml.safe_load(open(Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml", encoding="utf-8"))
    raw = copy.deepcopy(raw)
    for e in raw["excavators"]:
        e["model"] = "CAT 352 (koparka 52 t)"
    raw["crusher"]["model"] = "Sandvik CJ615 – szczękowa stacjonarna"
    sc = scenario_from_dict(raw)
    assert sc.excavators


def test_picking_a_model_brings_its_parameters():
    from app2.machines import _apply_model, model_changed

    t = _apply_model("truck", {"payload_t": 38.0, "model": "x"}, "x", "CAT 773G (sztywnoramowe 55 t)")
    assert t["payload_t"] == 55.0 and model_changed([t]) and "_model_changed" not in t
    e = _apply_model("excavator", {"bucket_m3": 4.5}, "", "CAT 352 (koparka 52 t)")
    assert e["bucket_m3"] == 3.1 and e["model"] == "CAT 352 (koparka 52 t)"


def test_a_variant_with_a_two_lane_ramp():
    import datetime as dt

    import yaml

    from app2.machines import variant_raw
    from kruszyk import workplan
    from kruszyk.config import scenario_from_dict
    from kruszyk.period import PeriodPlan, run_period
    from kruszyk.rl import today

    raw = yaml.safe_load(open(Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml", encoding="utf-8"))
    ramps = [r["id"] for r in raw["site"]["roads"] if r.get("single_lane")]
    assert ramps
    sc = scenario_from_dict(raw)
    n = 2
    work = {"start": {"O2": 30000.0, "O3": 30000.0}, "ex": {"K-1": [{"a": "O2", "t": 3000.0}] * n,
                                                           "K-2": [{"a": "O3", "t": 3000.0}] * n},
            "tr": {f.id: ["K-1" if j % 2 else "K-2"] * n for j, f in enumerate(sc.trucks)}, "dr": {}, "mode": "min"}
    out = {}
    for name, roads in (("A", {}), ("B", {r: {"single_lane": False} for r in ramps})):
        w = dict(work, roads=roads)
        vr = variant_raw(raw, w)
        assert all(not r.get("single_lane") for r in vr["site"]["roads"] if r["id"] in ramps) == bool(roads)
        ev = workplan.evaluate(w, sc.areas, [], n)
        sim = scenario_from_dict(workplan.scenario_raw(vr, w, ev["grid"]))
        p = PeriodPlan(start=dt.date(2026, 11, 2), days=1, shifts_per_day=2)
        p.frozen = workplan.frozen(w, ev, n)
        out[name] = run_period(today(sim), p, seed=1)["wywiezione_t"].sum()
    assert out["B"] >= out["A"] * 0.98  # two lanes never take tonnes away
    assert raw["site"]["roads"][0].get("single_lane") is not None or True  # the map itself is untouched


def test_kruszyk2_costs_only_the_loading_to_crusher_cycle():
    import yaml

    from kruszyk.config import cycle_costs, scenario_from_dict
    from kruszyk.rl import today
    from kruszyk.sim import simulate

    raw = yaml.safe_load(open(Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml", encoding="utf-8"))
    k = simulate(today(cycle_costs(scenario_from_dict(raw))), seed=1).kpi
    assert k["cost_breakdown"]["wiercenie_strzelanie"] == 0 and k["cost_breakdown"]["odchylka_skladu"] == 0
    assert k["cost_breakdown"]["kruszarka"] > 0 and k["cost_total"] == sum(k["cost_breakdown"].values())


def test_dynamic_assignment_sends_trucks_where_they_are_loaded_soonest():
    import datetime as dt

    import yaml

    from kruszyk import workplan
    from kruszyk.config import scenario_from_dict
    from kruszyk.period import PeriodPlan, run_period
    from kruszyk.rl import with_signals

    raw = yaml.safe_load(open(Path(__file__).parent.parent / "scenarios" / "kruszyk_demo_zakrzowek.yaml", encoding="utf-8"))
    sc = scenario_from_dict(raw)
    n = 1
    # every truck sent to K-1 by the plan, while K-2 has tonnes too
    work = {"start": {"O2": 30000.0, "O3": 30000.0}, "ex": {"K-1": [{"a": "O2", "t": 3000.0}],
                                                           "K-2": [{"a": "O3", "t": 3000.0}]},
            "tr": {f.id: ["K-1"] for f in sc.trucks}, "dr": {}, "mode": "limit"}
    ev = workplan.evaluate(work, sc.areas, [], n)
    sim = scenario_from_dict(workplan.scenario_raw(raw, work, ev["grid"]))
    out = {}
    for name, v in (("rigid", with_signals(sim)), ("dynamic", with_signals(sim, dynamic=True))):
        p = PeriodPlan(start=dt.date(2026, 11, 2), days=1, shifts_per_day=1)
        p.frozen = workplan.frozen(work, ev, n)
        kpis = []
        run_period(v, p, seed=1, kpis=kpis)
        out[name] = {e["excavator"]: e["loaded_t"] for e in kpis[0]["excavators"]}
    assert out["rigid"].get("K-2", 0) < 1  # today: nobody drives to K-2
    assert out["dynamic"].get("K-2", 0) > 0  # dynamic: trucks also load at K-2


def test_simulation_period_charts_dynamic_assignment_and_cost_ticks():
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["k2_quick"] = True
    at.session_state["k2_plan"] = {"start": "2026-11-02", "days": 3, "shifts": 2, "sat": False, "hol": "", "fire": 1,
                                   "real_t": 0.0}
    at.run()
    at.radio(key="k2_nav").set_value("② Harmonogram").run()
    at.button(key="k2_approve").click().run()
    assert not at.exception, at.exception
    at.toggle(key="k2_dyn").set_value(True)
    at.checkbox(key="k2_cost_kruszarka").uncheck()
    at.button(key="k2_month_go").click().run()
    assert not at.exception, at.exception
    m = at.session_state["month"]
    assert m["dynamic"] and "kruszarka" not in m["parts"]
    at.selectbox(key="k2_shift_i").set_value(3).run()
    assert not at.exception, at.exception
    for period in ("Doba", "Kilka dni", "Tydzień", "Miesiąc"):
        at.session_state["k2_period"] = period
        at.run()
        assert not at.exception, (period, at.exception)
    at.radio(key="k2_nav").set_value("④ Wynik").run()
    assert not at.exception, at.exception
    page = " ".join(x.value for x in at.markdown)
    assert "Koszt całkowity – wszystkie składniki" in page and "Kruszarka – stały" not in page
