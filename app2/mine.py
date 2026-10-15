"""① Kopalnia: the map (drawn on the satellite / orthophoto) and the machines."""
from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from kruszyk import catalog
from . import state, theme
from .theme import card

MAP, MACHINES = "Mapa", "Maszyny"


def step_mine() -> None:
    c = st.columns([3, 2], vertical_alignment="bottom")
    with c[0]:
        theme.header("Kopalnia", "Rysujesz na ortofoto – punkty przyczepiają się do dróg i obiektów. Model maszyny z "
                                 "katalogu wpisuje typowe parametry, popraw na Wasze realne.")
    with c[1]:
        tab = st.radio("Widok", [MAP, MACHINES], horizontal=True, key="k2_mine_tab", label_visibility="collapsed")
    if tab == MAP:
        from .mapview import map_screen

        map_screen()
    else:
        machines()


def _num(v, default=0.0) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else default
    except (TypeError, ValueError):
        return default


def machines() -> None:
    raw = copy.deepcopy(state.raw())
    v = st.session_state.get("ver", 0)

    # ---------------------------------------------------------------- trucks
    with card("trucks"):
        trucks = raw.get("trucks", [])
        n = sum(int(t.get("count", 1)) for t in trucks)
        cap = sum(int(t.get("count", 1)) * float(t.get("payload_t", 0)) for t in trucks)
        theme.h2(f"Wozidła <span class='k2-note' style='font-weight:400'>razem <b>{n} szt.</b> · {cap:,.0f} t ładowności</span>".replace(",", " "))
        df = pd.DataFrame([{"id": t.get("id", ""), "model": t.get("model", ""), "sztuk": int(t.get("count", 1)),
                            "ladownosc_t": t.get("payload_t"), "masa_t": t.get("empty_mass_t"), "moc_kw": t.get("power_kw"),
                            "vmax_kmh": t.get("max_speed_kmh"), "pal_pelny": t.get("fuel_lph_loaded"),
                            "pal_pusty": t.get("fuel_lph_empty"), "pal_jalowy": t.get("fuel_lph_idle")} for t in trucks])
        ed = st.data_editor(df, key=f"k2_trucks_{v}", num_rows="dynamic", hide_index=True, width="stretch",
                            column_config={"id": "ID", "model": "model", "sztuk": st.column_config.NumberColumn("sztuk", min_value=1, step=1),
                                           "ladownosc_t": "ładowność [t]", "masa_t": "masa pusta [t]", "moc_kw": "moc [kW]",
                                           "vmax_kmh": "v max [km/h]", "pal_pelny": "paliwo pełny [l/h]",
                                           "pal_pusty": "paliwo pusty [l/h]", "pal_jalowy": "paliwo jałowy [l/h]"})
        c = st.columns([3, 1, 1])
        model = c[0].selectbox("Dodaj z katalogu", catalog.names("truck"), key=f"k2_tcat_{v}")
        cnt = c[1].number_input("sztuk", 1, 30, 1, key=f"k2_tcnt_{v}")
        add = c[2].button("➕ Dodaj", key=f"k2_tadd_{v}", width="stretch")
    new_trucks = []
    for (_, r), old in zip(ed.iterrows(), trucks + [{}] * len(ed)):
        t = dict(old)
        t.update({"model": r["model"] or t.get("model", "wozidło"), "count": int(_num(r["sztuk"], 1)) or 1,
                  "payload_t": _num(r["ladownosc_t"], 36), "empty_mass_t": _num(r["masa_t"], 30),
                  "power_kw": _num(r["moc_kw"], 350), "max_speed_kmh": _num(r["vmax_kmh"], 45),
                  "fuel_lph_loaded": _num(r["pal_pelny"], 45), "fuel_lph_empty": _num(r["pal_pusty"], 25),
                  "fuel_lph_idle": _num(r["pal_jalowy"], 8)})
        if r["id"]:
            t["id"] = str(r["id"])
        new_trucks.append(t)
    if add:
        p = catalog.params("truck", model)
        used = {t.get("id") for t in new_trucks}
        k = len(new_trucks)
        for _ in range(int(cnt)):
            k += 1
            while f"WZ-{k:02d}" in used:
                k += 1
            new_trucks.append({**p, "model": model, "count": 1, "id": f"WZ-{k:02d}", "excavators": []})
    raw["trucks"] = new_trucks
    if add:
        state.set_raw(raw)
        st.rerun()

    # ---------------------------------------------------------------- excavators
    with card("ex"):
        theme.h2("Koparki i ładowarki")
        exs = raw.get("excavators", [])
        df = pd.DataFrame([{"id": e["id"], "lyzka_m3": e.get("bucket_m3"), "cykl_s": e.get("swing_cycle_s"),
                            "wypelnienie": e.get("fill_factor", 0.9), "podstawienie_s": e.get("spot_time_s", 40.0)} for e in exs])
        ed = st.data_editor(df, key=f"k2_ex_{v}", hide_index=True, width="stretch", disabled=["id"],
                            column_config={"id": "ID", "lyzka_m3": "łyżka [m³]", "cykl_s": "cykl łyżki [s]",
                                           "wypelnienie": "wypełnienie łyżki", "podstawienie_s": "podstawienie wozidła [s]"})
        st.caption("Koparki dodajesz na mapie. Model z katalogu: kliknij koparkę na mapie.")
    for e, (_, r) in zip(exs, ed.iterrows()):
        e.update({"bucket_m3": _num(r["lyzka_m3"], 4), "swing_cycle_s": _num(r["cykl_s"], 30),
                  "fill_factor": _num(r["wypelnienie"], 0.9), "spot_time_s": _num(r["podstawienie_s"], 40)})
    for se in (raw.get("site") or {}).get("excavators", []):  # the map keeps the machine data too
        x = next((e for e in exs if e["id"] == se.get("id")), None)
        if x:
            se.update({k_: x[k_] for k_ in ("bucket_m3", "swing_cycle_s", "fill_factor") if k_ in x})

    c1, c2 = st.columns([3, 2])
    # ---------------------------------------------------------------- crushers
    with c1, card("cr"):
        theme.h2("Kruszarki")
        cr = raw.setdefault("crusher", {})
        extra = raw.get("crushers_extra") or []
        rows = [{"id": cr.get("id", "C-1"), "wydajnosc": cr.get("capacity_tph"), "kosz": cr.get("hopper_t"),
                 "wyladunek_s": cr.get("dump_time_s", 60.0), "kwh_t": cr.get("kwh_per_t", 1.1)}]
        rows += [{"id": c.get("id"), "wydajnosc": c.get("capacity_tph"), "kosz": c.get("hopper_t"),
                  "wyladunek_s": c.get("dump_time_s", 60.0), "kwh_t": c.get("kwh_per_t", 1.1)} for c in extra]
        ed = st.data_editor(pd.DataFrame(rows), key=f"k2_cr_{v}", hide_index=True, width="stretch", disabled=["id"],
                            column_config={"id": "ID", "wydajnosc": "wydajność [t/h]", "kosz": "kosz [t]",
                                           "wyladunek_s": "wyładunek [s]", "kwh_t": "energia [kWh/t]"})
        for c_, (_, r) in zip([cr] + extra, ed.iterrows()):
            c_.update({"capacity_tph": _num(r["wydajnosc"], 400), "hopper_t": _num(r["kosz"], 100),
                       "dump_time_s": _num(r["wyladunek_s"], 60), "kwh_per_t": _num(r["kwh_t"], 1.1)})
    # ---------------------------------------------------------------- drill rigs
    with c2, card("rigs"):
        theme.h2("Wiertnice")
        rigs = raw.get("drills") or [{"id": "W-1", "in_work": True, "m_per_shift": 250.0, "speed_kmh": 2.0}]
        ed = st.data_editor(pd.DataFrame([{"id": r["id"], "w_pracy": bool(r.get("in_work", True)),
                                           "mb_zm": r.get("m_per_shift", 250.0), "kmh": r.get("speed_kmh", 2.0)} for r in rigs]),
                            key=f"k2_rigs_{v}", num_rows="dynamic", hide_index=True, width="stretch",
                            column_config={"id": "ID", "w_pracy": "w pracy", "mb_zm": "postęp [mb/zmianę]",
                                           "kmh": "przejazd [km/h]"})
        raw["drills"] = [{"id": str(r["id"] or f"W-{i + 1}"), "in_work": bool(r["w_pracy"]),
                          "m_per_shift": _num(r["mb_zm"], 250), "speed_kmh": _num(r["kmh"], 2)}
                         for i, (_, r) in enumerate(ed.iterrows())]

    # ---------------------------------------------------------------- costs
    with card("costs"):
        theme.h2("Koszty")
        co = raw.setdefault("costs", {})
        c = st.columns(6)
        fields = [("truck_per_h", "wozidło [zł/h]", 300.0), ("excavator_per_h", "koparka [zł/h]", 900.0),
                  ("crusher_per_h", "kruszarka [zł/h]", 1200.0), ("fuel_per_l", "paliwo [zł/l]", 6.5),
                  ("energy_per_kwh", "energia [zł/kWh]", 0.9), ("drill_blast_per_t", "wiercenie + strzał [zł/t]", 3.0)]
        for col, (k_, label, d) in zip(c, fields):
            co[k_] = col.number_input(label, 0.0, 100_000.0, float(co.get(k_, d)), key=f"k2_cost_{k_}_{v}")
        st.caption("Stawki maszyn bez paliwa (paliwo liczy się osobno z jazdy i postojów).")

    with st.expander("Awarie maszyn (z rejestru: ile razy i ile godzin w miesiącu)"):
        from app_calibration import reliability_editor

        reliability_editor(raw, state.scenario(), "k2rel")

    if st.button("💾 Zapisz maszyny", type="primary", key="k2_machines_save"):
        state.set_raw(raw)
        st.session_state.pop("month", None)  # the machines changed: the month has to be calculated again
        st.success("✅ Maszyny zapisane.")
        st.rerun()
