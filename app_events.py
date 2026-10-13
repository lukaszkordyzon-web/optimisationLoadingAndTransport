"""Shift and events editor: shift times, breaks, blasting, foreman's round, refuelling, speed practice."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st


def _time(label: str, value: str, key: str, **kw) -> str:
    h, m = (int(x) for x in str(value or "06:00").split(":"))
    t = st.time_input(label, dt.time(h, m), key=key, step=300, **kw)
    return t.strftime("%H:%M")


def _blasting(raw: dict, site: dict | None, k) -> None:
    """Today's blast: place, crew times, evacuation; how excavators get out of the way."""
    bl = raw.setdefault("blasting", {})
    c = st.columns([1, 2])
    bl["enabled"] = c[0].toggle("Strzelanie na tej zmianie", bool(bl.get("enabled", False)), key=k("bl_en"))
    places = {"": "– wybierz –"}
    for group, label in (("piles", "usyp"), ("areas", "obszar"), ("planned_piles", "planowany strzał")):
        for o in (site or {}).get(group, []):
            if o.get("id") and len(o.get("polygon") or []) >= 3:
                places[o["id"]] = f"{o['id']} ({label})"
    cur = bl.get("place", "") if bl.get("place", "") in places else ""
    bl["place"] = c[1].selectbox("Miejsce strzału", list(places), index=list(places).index(cur),
                                 format_func=places.get, key=k("bl_pl"), disabled=not bl["enabled"],
                                 help="tu jedzie ekipa strzałowa (po drogach); wozidła i koparki zjeżdżają")
    if bl["enabled"] and not bl["place"]:
        st.info("Wybierz miejsce strzału – bez niego ekipa nie jeździ po mapie, ale zamknięcie wyrobiska w podanych "
                "godzinach i tak obowiązuje.")
    c = st.columns(5)
    with c[0]:
        bl["crew_arrive"] = _time("Przyjazd ekipy", bl.get("crew_arrive", "09:00"), k("bl_a"))
    with c[1]:
        bl["fire_from"] = _time("Odpalenie od", bl.get("fire_from", "11:00"), k("bl_f"))
    with c[2]:
        bl["fire_to"] = _time("Odpalenie do (otwarcie)", bl.get("fire_to", "11:20"), k("bl_t"))
    with c[3]:
        bl["evacuation_min"] = st.number_input("Ewakuacja przed [min]", 0.0, 120.0, float(bl.get("evacuation_min", 10)),
                                               5.0, key=k("bl_e"), help="tyle wcześniej wozidła i koparki muszą zjechać")
    with c[4]:
        bl["crew_leave"] = _time("Wyjazd ekipy", bl.get("crew_leave", "11:40"), k("bl_l"))
    c = st.columns(2)
    bl["retreat_m"] = c[0].number_input("Bez miejsca postoju koparka cofa się o [m]", 0.0, 2000.0,
                                        float(bl.get("retreat_m", 200)), 50.0, key=k("bl_r"))
    tr0 = raw.setdefault("traffic", {})
    tr0["excavator_move_kmh"] = c[1].number_input("Prędkość przejazdu koparki [km/h]", 0.5, 15.0,
                                                  float(tr0.get("excavator_move_kmh", 2.5)), 0.5, key=k("bl_v"))
    st.caption("Koparka zjeżdża przed ewakuacją na **miejsce postoju** (znacznik 🅿️ na mapie, najkrótszą drogą) "
               "i wraca po otwarciu – ten czas wydłuża jej postój. Ekipa, sztygar i cysterna jeżdżą po "
               "narysowanych drogach (najkrótszą trasą).")
    st.caption("Ekipa jedzie samochodem po drogach do miejsca strzelania. **Mijanie:** jadący w dół ustępuje "
               "jadącemu w górę – gdy ekipa wyjeżdża pod górę, wozidło zjeżdżające w dół musi jej ustąpić. "
               "Od „ewakuacji” do „otwarcia” żadne wozidło nie wjeżdża do wyrobiska, koparki stoją.")


def events_editor(raw: dict, prefix: str, excavator_ids: list[str], site: dict | None, title: str | None = None,
                  plan: bool = False) -> None:
    """Edits raw['shift_plan'], ['blasting'], ['inspection'], ['refueling'], ['traffic'], ['variability'] in place.
    `plan` – for the period: blasting comes from the blast plan (each blast its own times and place)."""
    ver = st.session_state.get("ver", 0)
    k = lambda name: f"{prefix}_{ver}_{name}"  # noqa: E731
    if title:
        st.markdown(f"### {title}")
    st.caption("Zmiana to nie 8 godzin wożenia: rozruch maszyn, przerwy, strzelanie, objazd sztygara, cysterna. "
               "Wszystko to zabiera czas i zaburza ruch wozideł." + (" Dotyczy każdej zmiany okresu; godziny i miejsca "
                                                                     "strzałów wynikają z planu strzałów." if plan else ""))
    names = ["🕐 Godziny i przerwy", "🧨 Strzelanie", "👷 Objazd sztygara", "⛽ Tankowanie", "🚦 Prędkości i mijanie"]
    if plan:
        names.remove("🧨 Strzelanie")
    tabs = dict(zip(names, st.tabs(names)))
    tabs = [tabs.get(n) for n in ["🕐 Godziny i przerwy", "🧨 Strzelanie", "👷 Objazd sztygara", "⛽ Tankowanie",
                                  "🚦 Prędkości i mijanie"]]

    with tabs[0]:
        sp = raw.setdefault("shift_plan", {})
        c = st.columns(3)
        sp["start_time"] = _time("Początek zmiany", sp.get("start_time", "06:00"), k("start"))
        raw["shift_hours"] = float(c[0].number_input("Długość zmiany [h]", 1.0, 24.0, float(raw.get("shift_hours", 8)), 0.5,
                                                     key=k("hours")))
        sp["startup_min"] = c[1].number_input("Rozruch [min]", 0.0, 120.0, float(sp.get("startup_min", 15)), 5.0, key=k("su"),
                                             help="przejęcie maszyn, obsługa codzienna, rozgrzanie – zanim ruszy pierwszy kurs")
        sp["end_early_min"] = c[2].number_input("Koniec pracy przed końcem zmiany [min]", 0.0, 120.0,
                                               float(sp.get("end_early_min", 10)), 5.0, key=k("ee"),
                                               help="ostatni kurs musi się skończyć przed tym czasem")
        st.markdown("Przerwy (wozidła i koparki stoją):")
        base_key = k("breaks_base")
        if base_key not in st.session_state:
            st.session_state[base_key] = pd.DataFrame(sp.get("breaks", [{"start": "10:00", "min": 20}]) or [],
                                                      columns=["start", "min"])
        ed = st.data_editor(st.session_state[base_key], num_rows="dynamic", hide_index=True, key=k("breaks"),
                            column_config={"start": st.column_config.TextColumn("godzina (GG:MM)"),
                                           "min": st.column_config.NumberColumn("czas [min]", min_value=0)})
        sp["breaks"] = [{"start": str(r["start"]), "min": float(r["min"])} for _, r in ed.dropna().iterrows()
                        if ":" in str(r["start"])]

    if tabs[1] is not None:  # one shift only; the period takes blasts from the plan
        with tabs[1]:
            _blasting(raw, site, k)

    with tabs[2]:
        ins = raw.setdefault("inspection", {})
        ins["enabled"] = st.toggle("Objazd sztygara", bool(ins.get("enabled", False)), key=k("in_en"))
        c = st.columns([1, 3, 1])
        with c[0]:
            ins["start"] = _time("Wyjazd", ins.get("start", "09:30"), k("in_s"))
        ins["stops"] = c[1].multiselect("Kolejne punkty objazdu", excavator_ids,
                                        [x for x in ins.get("stops", []) if x in excavator_ids] or excavator_ids,
                                        key=k("in_p"))
        ins["stop_min"] = c[2].number_input("Postój [min]", 0.0, 120.0, float(ins.get("stop_min", 10)), 5.0, key=k("in_m"))
        st.caption("Samochód sztygara jedzie od kruszarki przez kolejne koparki i wraca. Te same zasady mijania co ekipy.")

    with tabs[3]:
        rf = raw.setdefault("refueling", {})
        st.markdown("**Wozidła** – zatankowane przed zmianą; w trakcie zjeżdżają na stację przy kruszarce, gdy "
                    "spadną do rezerwy (zużycie liczone z jazdy i postojów).")
        c = st.columns(3)
        c[0].caption("Pojemność zbiornika – przy każdym wozidle (tabela „Wozidła”).")
        rf["truck_reserve_pct"] = c[1].number_input("Rezerwa [%]", 0.0, 80.0, float(rf.get("truck_reserve_pct", 15)), 5.0,
                                                    key=k("rf_res"))
        rf["truck_refuel_min"] = c[2].number_input("Tankowanie wozidła [min]", 1.0, 60.0, float(rf.get("truck_refuel_min", 12)),
                                                   1.0, key=k("rf_tmin"))
        st.markdown("**Koparki** – domyślnie tankowane przed zmianą. Jeśli w trakcie zmiany wjeżdża cysterna, wpisz "
                    "godziny: cysterna objeżdża koparki, każda stoi na czas tankowania, a na drogach obowiązuje mijanie.")
        c = st.columns(2)
        times = c[0].text_input("Godziny wjazdu cysterny (np. 13:00, 17:30; puste = brak)",
                                ", ".join(rf.get("tanker_times", [])), key=k("rf_tt"))
        rf["tanker_times"] = [x.strip() for x in times.split(",") if ":" in x]
        rf["excavator_refuel_min"] = c[1].number_input("Tankowanie koparki [min]", 1.0, 120.0,
                                                       float(rf.get("excavator_refuel_min", 20)), 5.0, key=k("rf_emin"))
        rf["at_parking"] = st.checkbox("Koparki zjeżdżają do tankowania na swoje miejsce postoju (🅿️)",
                                       bool(rf.get("at_parking", False)), key=k("rf_park"))

    with tabs[4]:
        v = raw.setdefault("variability", {})
        tr = raw.setdefault("traffic", {})
        st.markdown("**Limity prędkości** ustawiasz przy każdej drodze (tabela „Drogi i rampy”) – są obowiązujące. "
                    "Tu ustawiasz, jak jest naprawdę:")
        c = st.columns(4)
        v["speeding_share"] = c[0].slider("Kierowców jeżdżących szybciej [%]", 0, 100, int(100 * float(v.get("speeding_share", 0.3))),
                                          5, key=k("sp_sh")) / 100
        v["speeding_pct"] = c[1].slider("…o ile ponad limit [%]", 0, 50, int(v.get("speeding_pct", 10)), 1, key=k("sp_pct"))
        tr["yield_s"] = c[2].number_input("Ustąpienie pierwszeństwa [s]", 0.0, 300.0, float(tr.get("yield_s", 30)), 5.0,
                                          key=k("tr_y"), help="ile traci wozidło, gdy musi przepuścić pojazd jadący w górę")
        tr["light_speed_kmh"] = c[3].number_input("Prędkość aut osobowych [km/h]", 5.0, 80.0, float(tr.get("light_speed_kmh", 30)),
                                                  5.0, key=k("tr_v"), help="nigdy powyżej limitu drogi")
