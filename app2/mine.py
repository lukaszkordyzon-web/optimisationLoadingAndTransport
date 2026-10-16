"""① Kopalnia: the map (drawn on the satellite / orthophoto) and the machines."""
from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from kruszyk import catalog
from . import state, theme
from .machines import crusher_editor, excavator_editor, map_models, model_changed, sync_site, truck_editor
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
        new_trucks = truck_editor(trucks, f"k2_trucks_{v}")
        st.caption("Każde wozidło w osobnym wierszu. Wybierz model z listy – parametry wpiszą się same; popraw je "
                   "na Wasze realne. „(własny)” = wpisujesz wszystko sam.")
        c = st.columns([3, 1, 1])
        model = c[0].selectbox("Dodaj z katalogu", catalog.names("truck"), key=f"k2_tcat_{v}")
        cnt = c[1].number_input("sztuk", 1, 30, 1, key=f"k2_tcnt_{v}")
        add = c[2].button("➕ Dodaj", key=f"k2_tadd_{v}", width="stretch")
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
    if model_changed(new_trucks):  # a model picked: show its parameters in the table at once
        state.set_raw(raw)
        st.rerun()
    if add:
        state.set_raw(raw)
        st.rerun()

    # ---------------------------------------------------------------- excavators
    with card("ex"):
        theme.h2("Koparki i ładowarki")
        exs = raw.get("excavators", [])
        area_ids = [x["id"] for x in (raw.get("site") or {}).get("areas", []) if x.get("id")]
        raw["excavators"] = excavator_editor(exs, f"k2_ex_{v}", area_ids, map_models(raw))
        if model_changed(raw["excavators"]):
            sync_site(raw)
            state.set_raw(raw)
            st.rerun()
        st.caption("Koparki dodajesz na mapie. Tu wybierasz model z katalogu i poprawiasz parametry.")
    sync_site(raw)

    c1, c2 = st.columns([3, 2])
    # ---------------------------------------------------------------- crushers
    with c1, card("cr"):
        theme.h2("Kruszarki")
        cr = raw.setdefault("crusher", {})
        cr.setdefault("model", ((raw.get("site") or {}).get("crusher") or {}).get("model", ""))
        extra = raw.get("crushers_extra") or []
        done = crusher_editor([cr] + extra, f"k2_cr_{v}")
        changed = model_changed(done)
        raw["crusher"], raw["crushers_extra"] = done[0], done[1:]
        sync_site(raw)
        if changed:
            state.set_raw(raw)
            st.rerun()
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
        from kruszyk.config import Costs

        cur = Costs(**{k_: x for k_, x in co.items() if k_ in Costs.__dataclass_fields__})
        rates = {"truck": cur.truck_rates(), "excavator": cur.excavator_rates()}
        for kind, title in (("truck", "Wozidło"), ("excavator", "Koparka")):
            c = st.columns([0.8, 1, 1, 1, 1.4], vertical_alignment="bottom")
            c[0].markdown(f'<div class="k2-h4" style="padding-bottom:10px">{title}</div>', unsafe_allow_html=True)
            op, am, sv = rates[kind]
            vals = [c[1].number_input("operator / kierowca [zł/h zmiany]", 0.0, 10_000.0, round(op, 1), 5.0,
                                      key=f"k2_cost_{kind}_op_{v}"),
                    c[2].number_input("amortyzacja – koszt odtworzenia [zł/mth]", 0.0, 10_000.0, round(am, 1), 5.0,
                                      key=f"k2_cost_{kind}_am_{v}",
                                      help="Cena nowej maszyny ÷ motogodziny jej życia. Liczy się tylko, gdy silnik "
                                           "pracuje."),
                    c[3].number_input("serwis, opony, części [zł/mth]", 0.0, 10_000.0, round(sv, 1), 5.0,
                                      key=f"k2_cost_{kind}_sv_{v}")]
            for k_, val in zip(("operator_per_h", "amort_per_mth", "service_per_mth"), vals):
                co[f"{kind}_{k_}"] = float(val)
            c[4].markdown(f'<div class="k2-note" style="padding-bottom:10px">razem przy pracującym silniku '
                          f'<b>{sum(vals):,.0f} zł/h</b>; na postoju z wyłączonym silnikiem {vals[0]:,.0f} zł/h</div>'
                          .replace(",", " "), unsafe_allow_html=True)
        c = st.columns(4)
        fields = [("crusher_per_h", "kruszarka [zł/h]", 1200.0), ("fuel_per_l", "paliwo [zł/l]", 6.5),
                  ("energy_per_kwh", "energia [zł/kWh]", 0.9), ("drill_blast_per_t", "wiercenie + strzał [zł/t]", 3.0)]
        for col, (k_, label, d) in zip(c, fields):
            co[k_] = col.number_input(label, 0.0, 100_000.0, float(co.get(k_, d)), key=f"k2_cost_{k_}_{v}")
        st.caption("Stawki bez paliwa (paliwo liczy się osobno z jazdy i postojów). mth = motogodzina: amortyzacja i serwis "
                   "liczą się tylko wtedy, gdy silnik pracuje – także w kolejce na biegu jałowym.")

    with st.expander("Awarie maszyn (z rejestru: ile razy i ile godzin w miesiącu)"):
        from app_calibration import reliability_editor

        reliability_editor(raw, state.scenario(), "k2rel")

    if st.button("💾 Zapisz maszyny", type="primary", key="k2_machines_save"):
        state.set_raw(raw)
        st.session_state.pop("month", None)  # the machines changed: the month has to be calculated again
        st.success("✅ Maszyny zapisane.")
        st.rerun()
