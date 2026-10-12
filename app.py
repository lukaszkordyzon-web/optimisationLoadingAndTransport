"""Panel scenariuszy – uruchomienie: streamlit run app.py"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from kruszyk.analytics import match_factor_report
from kruszyk.config import dump_scenario, scenario_from_dict
from kruszyk.ml import generate_dataset, train_travel_model
from kruszyk.optimize import replicate, sweep_fragmentation, sweep_trucks
from kruszyk.sim import simulate
from kruszyk.views import STATE_LABELS, driver_instruction, operator_view, route_profile, snapshot

from app_map import render_live_map, render_map_tab

SCEN_DIR = Path(__file__).parent / "scenarios"
STATE_COLORS = {
    "jazda_pusty": "#8ecae6", "jazda_pelny": "#219ebc", "kolejka_koparka": "#e63946",
    "podstawienie": "#f4a261", "zaladunek": "#2a9d8f", "kolejka_kruszarka": "#d62828",
    "wyladunek": "#264653", "wyladunek_halda": "#6c757d", "awaria": "#000000",
    "postoj_brak_urobku": "#adb5bd",
}
BASE_TS = pd.Timestamp("2026-01-01 06:00")

st.set_page_config(page_title="Kruszyk – załadunek, transport, kruszenie", page_icon="⛏️", layout="wide")


# ------------------------------------------------------------------ state
def presets() -> dict[str, Path]:
    return {yaml.safe_load(p.read_text(encoding="utf-8")).get("name", p.stem): p
            for p in sorted(SCEN_DIR.glob("*.yaml"))}


def set_raw(raw: dict):
    st.session_state.raw = raw
    st.session_state.ver = st.session_state.get("ver", 0) + 1  # new widget keys -> widgets reload
    st.session_state.pop("res", None)


if "raw" not in st.session_state:
    p = presets()
    first = next((k for k, v in p.items() if v.stem == "kruszyk_bazowy"), next(iter(p)))
    set_raw(yaml.safe_load(p[first].read_text(encoding="utf-8")))

raw = st.session_state.raw
V = st.session_state.ver


def k(name: str) -> str:
    return f"{V}_{name}"


# ---------------------------------------------------------------- cached runs
@st.cache_data(show_spinner=False)
def run_sim(yaml_text: str, seed: int):
    return simulate(scenario_from_dict(yaml.safe_load(yaml_text)), seed=seed)


@st.cache_data(show_spinner=False)
def run_replicate(yaml_text: str, reps: int):
    return replicate(scenario_from_dict(yaml.safe_load(yaml_text)), reps)


@st.cache_data(show_spinner=False)
def run_sweep(yaml_text: str, n_min: int, n_max: int, reps: int):
    return sweep_trucks(scenario_from_dict(yaml.safe_load(yaml_text)), n_min, n_max, reps)


@st.cache_data(show_spinner=False)
def run_frag(yaml_text: str, values: tuple, reps: int):
    return sweep_fragmentation(scenario_from_dict(yaml.safe_load(yaml_text)), list(values), reps)


@st.cache_data(show_spinner=False)
def run_snapshots(yaml_text: str, seed: int, step_min: int):
    res = run_sim(yaml_text, seed)
    frames = []
    for m in range(0, int(res.scenario.shift_hours * 60), step_min):
        s = snapshot(res, m * 60.0)
        if len(s):
            s["minuta"] = m
            frames.append(s)
    return pd.concat(frames, ignore_index=True)


@st.cache_resource(show_spinner=False)
def run_train(yaml_text: str, shifts: int):
    sc = scenario_from_dict(yaml.safe_load(yaml_text))
    data = generate_dataset(sc, shifts)
    model, metrics = train_travel_model(data)
    return data, model, metrics


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⛏️ Kruszyk")
    st.caption("Optymalizacja łańcucha odstrzał → koparka → wozidła → kruszarka (dane syntetyczne)")
    p = presets()
    choice = st.selectbox("Scenariusz przykładowy", list(p))
    if st.button("Wczytaj scenariusz", use_container_width=True):
        set_raw(yaml.safe_load(p[choice].read_text(encoding="utf-8")))
        st.rerun()
    up = st.file_uploader("…albo wczytaj własny plik YAML", type=["yaml", "yml"])
    if up is not None and st.session_state.get("uploaded") != up.file_id:
        st.session_state.uploaded = up.file_id
        set_raw(yaml.safe_load(up.getvalue().decode("utf-8")))
        st.rerun()
    st.divider()
    seed = st.number_input("Ziarno losowania (inna zmiana = inne ziarno)", 0, 10_000, int(raw.get("seed", 42)))


# ---------------------------------------------------------------- editor
def records(df: pd.DataFrame) -> list[dict]:
    df = df.dropna(how="all")
    out = []
    for r in df.to_dict("records"):
        out.append({kk: (v.item() if hasattr(v, "item") else v) for kk, v in r.items() if not pd.isna(v)})
    return out


def editor() -> dict:
    new = copy.deepcopy(raw)
    c1, c2, c3, c4 = st.columns(4)
    new["name"] = c1.text_input("Nazwa scenariusza", raw.get("name", ""), key=k("name"))
    new["shift_hours"] = c2.number_input("Długość zmiany [h]", 1.0, 24.0, float(raw["shift_hours"]), 0.5, key=k("shift"))
    new["dispatch"] = c3.radio("Przydział wozideł", ["dynamic", "fixed"],
                               index=0 if raw.get("dispatch", "dynamic") == "dynamic" else 1, key=k("disp"),
                               format_func=lambda x: {"dynamic": "automatyczny (system)", "fixed": "stały do koparki"}[x],
                               horizontal=True)
    new["loose_density_t_m3"] = c4.number_input("Gęstość urobku spulchn. [t/m³]", 1.0, 3.0,
                                                float(raw.get("loose_density_t_m3", 1.75)), 0.05, key=k("dens"))
    new["seed"] = int(seed)

    st.subheader("Odstrzały i koparki")
    c1, c2 = st.columns([2, 3])
    new["blasts"] = records(c1.data_editor(pd.DataFrame(raw["blasts"]), num_rows="dynamic", key=k("blasts"),
        column_config={"id": "Odstrzał", "tonnage_t": st.column_config.NumberColumn("Tonaż [t]", min_value=0),
                       "fragmentation": st.column_config.NumberColumn("Rozdrobnienie 0=dobre 1=bryły",
                                                                      min_value=0.0, max_value=1.0, step=0.05)},
        use_container_width=True))
    route_ids = list(raw["routes"])
    blast_ids = [b["id"] for b in new["blasts"] if "id" in b]
    ex_df = pd.DataFrame(raw["excavators"])
    for col, default in (("fill_factor", 0.9), ("spot_time_s", 40.0)):
        if col not in ex_df:
            ex_df[col] = default
    new["excavators"] = records(c2.data_editor(ex_df, num_rows="dynamic", key=k("ex"), use_container_width=True,
        column_config={"id": "Koparka", "bucket_m3": "Łyżka [m³]", "swing_cycle_s": "Cykl łyżki [s]",
                       "blast": st.column_config.SelectboxColumn("Odstrzał", options=blast_ids),
                       "route": st.column_config.SelectboxColumn("Trasa", options=route_ids),
                       "fill_factor": "Wsp. napełnienia", "spot_time_s": "Podstawienie [s]"}))

    st.subheader("Wozidła")
    tr_df = pd.DataFrame(raw["trucks"])
    new["trucks"] = records(st.data_editor(tr_df, num_rows="dynamic", key=k("trucks"), use_container_width=True,
        column_config={"model": "Model", "count": st.column_config.NumberColumn("Liczba", min_value=0, step=1),
                       "payload_t": "Ładowność [t]", "empty_mass_t": "Masa własna [t]", "power_kw": "Moc [kW]",
                       "max_speed_kmh": "V max [km/h]", "drive_efficiency": "Sprawność napędu",
                       "fuel_lph_loaded": "Paliwo pełny [l/h]", "fuel_lph_empty": "Paliwo pusty [l/h]",
                       "fuel_lph_idle": "Paliwo postój [l/h]"}))
    for t in new["trucks"]:
        t["count"] = int(t.get("count", 0))

    st.subheader("Trasy (od frontu do kosza kruszarki; nachylenie + = pod górę z ładunkiem)")
    seg_rows = [{"route": rid, **s} for rid, r in raw["routes"].items() for s in r["segments"]]
    seg_df = st.data_editor(pd.DataFrame(seg_rows), num_rows="dynamic", key=k("segs"), use_container_width=True,
        column_config={"route": "Trasa", "name": "Odcinek", "length_m": "Długość [m]",
                       "grade_pct": "Nachylenie [%]", "speed_limit_kmh": "Limit [km/h]"})
    routes: dict = {}
    for r in records(seg_df):
        rid = r.pop("route", None)
        if rid:
            routes.setdefault(rid, {"segments": []})["segments"].append(r)
    for rid, r in routes.items():  # keep map geometry of routes built from the map
        if raw["routes"].get(rid, {}).get("path"):
            r["path"] = raw["routes"][rid]["path"]
    new["routes"] = routes

    st.subheader("Kruszarka, hałda buforowa, doradztwo prędkości")
    c1, c2, c3 = st.columns(3)
    cr = dict(raw["crusher"])
    with c1:
        st.markdown("**Kruszarka**")
        cr["capacity_tph"] = st.number_input("Wydajność nominalna [t/h]", 50.0, 10000.0, float(cr["capacity_tph"]), 50.0, key=k("cap"))
        cr["hopper_t"] = st.number_input("Pojemność kosza [t]", 10.0, 2000.0, float(cr["hopper_t"]), 10.0, key=k("hop"))
        cr["dump_time_s"] = st.number_input("Czas wyładunku [s]", 10.0, 300.0, float(cr.get("dump_time_s", 60)), 5.0, key=k("dump"))
        cr["fragmentation_penalty"] = st.slider("Spadek wydajności przy złym rozdrobnieniu", 0.0, 0.8,
                                                float(cr.get("fragmentation_penalty", 0.4)), 0.05, key=k("pen"))
    new["crusher"] = cr
    sp = dict(raw.get("stockpile", {}))
    with c2:
        st.markdown("**Hałda buforowa + ładowarka**")
        sp["enabled"] = st.toggle("Włączona", bool(sp.get("enabled", False)), key=k("sp_en"))
        sp["divert_wait_s"] = st.number_input("Zrzut na hałdę, gdy czekanie > [s]", 0.0, 3600.0, float(sp.get("divert_wait_s", 180)), 30.0, key=k("sp_w"))
        sp["loader_tph"] = st.number_input("Wydajność ładowarki [t/h]", 0.0, 3000.0, float(sp.get("loader_tph", 450)), 50.0, key=k("sp_l"))
        sp["initial_t"] = st.number_input("Zapas na hałdzie na start [t]", 0.0, 50000.0, float(sp.get("initial_t", 0)), 100.0, key=k("sp_i"))
    new["stockpile"] = sp
    pc = dict(raw.get("pacing", {}))
    with c3:
        st.markdown("**Doradztwo prędkości**")
        st.caption("Wozidło, które i tak czekałoby pod koszem, dostaje komunikat „zwolnij” – mniej postoju na silniku.")
        pc["enabled"] = st.toggle("Włączone", bool(pc.get("enabled", False)), key=k("pc_en"))
        pc["max_slowdown"] = st.slider("Maks. wydłużenie jazdy", 0.0, 0.5, float(pc.get("max_slowdown", 0.25)), 0.05, key=k("pc_m"))
    new["pacing"] = pc

    with st.expander("Zmienność i awarie"):
        v = dict(raw.get("variability", {}))
        c = st.columns(4)
        v["driver_sd"] = c[0].number_input("Różnice między kierowcami (sd)", 0.0, 0.5, float(v.get("driver_sd", 0.07)), 0.01, key=k("v1"))
        v["trip_sd"] = c[1].number_input("Losowość przejazdu (sd)", 0.0, 0.5, float(v.get("trip_sd", 0.08)), 0.01, key=k("v2"))
        v["rain_probability"] = c[2].number_input("Szansa deszczu na zmianie", 0.0, 1.0, float(v.get("rain_probability", 0.2)), 0.05, key=k("v3"))
        v["truck_mtbf_h"] = c[3].number_input("Średnio h między awariami wozidła", 1.0, 1000.0, float(v.get("truck_mtbf_h", 40)), 5.0, key=k("v4"))
        new["variability"] = v
    with st.expander("Koszty [PLN]"):
        cs = dict(raw.get("costs", {}))
        c = st.columns(4)
        labels = {"truck_per_h": "Wozidło [PLN/h]", "excavator_per_h": "Koparka [PLN/h]", "crusher_per_h": "Kruszarka [PLN/h]",
                  "fuel_per_l": "Paliwo [PLN/l]", "energy_per_kwh": "Energia [PLN/kWh]",
                  "drill_blast_per_t": "Wiercenie+strzelanie [PLN/t]", "drill_blast_fine_extra_per_t": "Dopłata za drobny odstrzał [PLN/t]"}
        defaults = {"truck_per_h": 300, "excavator_per_h": 900, "crusher_per_h": 1200, "fuel_per_l": 6.5,
                    "energy_per_kwh": 0.9, "drill_blast_per_t": 3.0, "drill_blast_fine_extra_per_t": 2.5}
        for i, (key, lab) in enumerate(labels.items()):
            cs[key] = c[i % 4].number_input(lab, 0.0, 100000.0, float(cs.get(key, defaults[key])), key=k("c" + key))
        new["costs"] = cs
    return new


tabs = st.tabs(["🗺️ Mapa kopalni", "⚙️ Scenariusz", "📊 Wyniki zmiany", "📍 Podgląd na żywo", "⚖️ Porównanie wariantów",
                "💥 Odstrzał ↔ kruszarka", "🤖 Model ETA"])

with tabs[1]:
    edited = editor()
    try:
        sc = scenario_from_dict(copy.deepcopy(edited))
        yaml_text = dump_scenario(sc)
        st.session_state.raw_edited = edited
        ok = True
    except Exception as e:  # noqa: BLE001 – show any config problem to the user
        st.error(f"Błąd w scenariuszu: {e}")
        ok = False
    if ok:
        st.divider()
        r = match_factor_report(sc)
        st.subheader("Szybki kalkulator (bez symulacji)")
        c = st.columns(5)
        c[0].metric("Match factor", r["match_factor"], help="<1 koparki czekają, >1 wozidła się kolejkują")
        c[1].metric("Potrzeba wozideł", r["trucks_needed"])
        c[2].metric("Rekomendacja", r["recommended_trucks"])
        c[3].metric("Wąskie gardło", r["bottleneck"])
        c[4].metric("Oczekiwane t/h", r["expected_tph"])
        st.dataframe(pd.DataFrame(r["excavators"]).rename(columns={
            "spot_s": "podstawienie [s]", "load_s": "załadunek [s]", "haul_s": "jazda pełny [s]",
            "dump_s": "wyładunek [s]", "return_s": "powrót [s]", "cycle_s": "cykl [s]",
            "trucks_needed": "wozideł potrzeba", "excavator_max_tph": "max koparki [t/h]"}),
            hide_index=True, use_container_width=True)
        st.download_button("💾 Pobierz scenariusz (YAML)", yaml_text, file_name="scenariusz.yaml")

with tabs[0]:
    render_map_tab(st.session_state.get("raw_edited", raw) if ok else raw, set_raw)

if not ok:
    st.stop()


# ---------------------------------------------------------------- results
def gantt(res) -> go.Figure:
    df = res.states.copy()
    df["od"] = BASE_TS + pd.to_timedelta(df["start_s"], unit="s")
    df["do"] = BASE_TS + pd.to_timedelta(df["end_s"], unit="s")
    df["stan"] = df["state"].map(STATE_LABELS)
    colors = {STATE_LABELS[s]: c for s, c in STATE_COLORS.items()}
    fig = px.timeline(df, x_start="od", x_end="do", y="truck", color="stan", color_discrete_map=colors,
                      hover_data={"excavator": True})
    fig.update_yaxes(categoryorder="category descending", title=None)
    fig.update_layout(height=60 + 28 * len(res.truck_ids), legend_title=None, margin=dict(t=10, b=10))
    return fig


with tabs[2]:
    res = run_sim(yaml_text, int(seed))
    kp = res.kpi
    st.caption(f"Jedna zmiana {sc.shift_hours:g} h · {'deszcz' if kp['rain'] else 'sucho'} · ziarno {seed} "
               "(zmień ziarno w panelu bocznym, żeby zobaczyć inną zmianę)")
    c = st.columns(6)
    c[0].metric("Wywiezione", f"{kp['hauled_t']:,.0f} t".replace(",", " "))
    c[1].metric("Wydajność", f"{kp['throughput_tph']:.0f} t/h")
    c[2].metric("Koszt", f"{kp['cost_per_t']:.2f} zł/t" if kp["cost_per_t"] else "–")
    c[3].metric("Kolejki / cykl", f"{kp['avg_queue_per_cycle_min']:.1f} min")
    c[4].metric("Kruszarka pracuje", f"{100 * kp['crusher_utilization']:.0f} %")
    c[5].metric("Paliwo", f"{kp['fuel_l_per_t']:.2f} l/t" if kp["fuel_l_per_t"] else "–")
    c = st.columns(6)
    c[0].metric("Pokruszone", f"{kp['crushed_t']:,.0f} t".replace(",", " "))
    c[1].metric("Kolejka pod kruszarką", f"{kp['truck_queue_crusher_h']:.1f} h")
    c[2].metric("Kolejka pod koparkami", f"{kp['truck_queue_excavator_h']:.1f} h")
    c[3].metric("Kruszarka bez materiału", f"{kp['crusher_starved_h']:.1f} h")
    c[4].metric("Na hałdę", f"{kp['stockpile_in_t']:.0f} t")
    c[5].metric("Awarie wozideł", f"{kp['truck_breakdown_h']:.1f} h")

    st.markdown("**Koparki**")
    st.dataframe(pd.DataFrame(kp["excavators"]).rename(columns={
        "excavator": "koparka", "loaded_t": "załadowano [t]", "remaining_t": "zostało [t]",
        "utilization": "wykorzystanie", "waiting_for_trucks_h": "czeka na wozidła [h]",
        "blast_done_at_h": "odstrzał wybrany po [h]"}), hide_index=True, use_container_width=True)

    st.markdown("**Oś czasu wozideł** – czerwone = czekanie")
    st.plotly_chart(gantt(res), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kosz kruszarki i hałda**")
        h = res.hopper.copy()
        h["godzina"] = h["t_s"] / 3600
        fig = go.Figure()
        fig.add_scatter(x=h["godzina"], y=h["level_t"], name="kosz [t]", fill="tozeroy")
        if sc.stockpile.enabled:
            fig.add_scatter(x=h["godzina"], y=h["stockpile_t"], name="hałda [t]")
        fig.add_hline(y=sc.crusher.hopper_t, line_dash="dot", annotation_text="pojemność kosza")
        fig.update_layout(height=320, xaxis_title="godzina zmiany", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Struktura kosztu [PLN/t]**")
        cb = pd.Series(kp["cost_breakdown"]) / max(kp["hauled_t"], 1)
        fig = px.bar(cb[cb > 0].sort_values(), orientation="h", labels={"value": "PLN/t", "index": ""})
        fig.update_layout(height=320, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Pobierz dane syntetyczne (CSV)"):
        c = st.columns(4)
        c[0].download_button("Przejazdy odcinków", res.segments.to_csv(index=False), "segments.csv")
        c[1].download_button("Stany wozideł", res.states.to_csv(index=False), "states.csv")
        c[2].download_button("Załadunki", res.loads.to_csv(index=False), "loads.csv")
        c[3].download_button("Kosz kruszarki", res.hopper.to_csv(index=False), "hopper.csv")


# ---------------------------------------------------------------- live view
def profile_figure(snap: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for rid in sc.routes:
        prof = route_profile(sc, rid)
        fig.add_scatter(x=prof["dist_m"], y=prof["elev_m"], mode="lines", name=f"trasa {rid}",
                        line=dict(width=6, color="#ced4da"), hoverinfo="skip", showlegend=False)
        ex_here = [e.id for e in sc.excavators if e.route == rid]
        fig.add_annotation(x=prof["dist_m"].iloc[0], y=prof["elev_m"].iloc[0], text="⛏️ " + ", ".join(ex_here),
                           showarrow=False, yshift=18)
    fig.add_annotation(x=0, y=0, text="🏭 kruszarka", showarrow=False, yshift=18)
    if len(snap):
        snap = snap.copy()
        # trucks standing at the same spot (queue) are spread out so their labels stay readable
        snap["dist_m"] = snap["dist_m"] + snap.groupby(["dist_m", "elev_m"]).cumcount() * 60.0
        for state, g in snap.groupby("state"):
            fig.add_scatter(x=g["dist_m"], y=g["elev_m"], mode="markers+text", text=g["truck"],
                            textposition="top center", name=STATE_LABELS.get(state, state),
                            marker=dict(size=14, color=STATE_COLORS.get(state, "#888")))
    fig.update_layout(height=380, xaxis_title="odległość do kruszarki [m]", yaxis_title="wysokość względna [m]",
                      xaxis_autorange="reversed", margin=dict(t=40), legend=dict(orientation="h", y=1.12))
    return fig


with tabs[3]:
    shift_min = int(sc.shift_hours * 60)
    minute = st.slider("Moment zmiany [min od startu]", 0, shift_min - 1, min(180, shift_min - 1), 1)
    t = minute * 60.0
    snap = snapshot(res, t)
    clock = (BASE_TS + pd.Timedelta(seconds=t)).strftime("%H:%M")
    hop = res.hopper[res.hopper["t_s"] <= t]
    level = hop["level_t"].iloc[-1] if len(hop) else 0.0
    st.markdown(f"### 🕐 {clock} · kosz kruszarki: {level:.0f} / {sc.crusher.hopper_t:.0f} t")
    render_live_map(sc, snap, STATE_COLORS, STATE_LABELS)
    st.plotly_chart(profile_figure(snap), use_container_width=True)

    st.markdown("#### 📱 Ekran operatora koparki")
    cols = st.columns(len(sc.excavators))
    for col, ex in zip(cols, sc.excavators):
        ov = operator_view(snap, ex.id)
        with col.container(border=True):
            st.markdown(f"**{ex.id}** · odstrzał {ex.blast}")
            st.write(f"Pod łyżką: **{ov['loading'] or '— brak wozidła —'}**")
            if ov["waiting"]:
                st.write("Czekają: " + ", ".join(ov["waiting"]))
            if ov["incoming"]:
                for truck, eta, seg in ov["incoming"]:
                    eta_txt = f"{int(eta // 60)}:{int(eta % 60):02d}" if eta is not None and pd.notna(eta) else "?"
                    st.write(f"🚚 {truck} za **{eta_txt}** ({seg})")
            else:
                st.write("Nikt nie nadjeżdża")
            if not ov["loading"] and not ov["waiting"]:
                st.caption("Koparka może przygotować urobek / przegarnąć front")

    st.markdown("#### 📱 Ekrany kierowców")
    if len(snap):
        d = snap.copy()
        d["polecenie"] = d.apply(driver_instruction, axis=1)
        d["od [min]"] = d["since_min"].round(1)
        st.dataframe(d[["truck", "status", "excavator", "segment", "polecenie", "od [min]"]].rename(columns={
            "truck": "wozidło", "excavator": "koparka", "segment": "odcinek"}), hide_index=True, use_container_width=True)

    with st.expander("▶️ Animacja całej zmiany"):
        step = st.select_slider("Krok animacji [min]", [1, 2, 5, 10], value=2)
        frames = run_snapshots(yaml_text, int(seed), step)
        frames["stan"] = frames["state"].map(STATE_LABELS)
        rng_x = max(r.length_m for r in sc.routes.values()) * 1.05
        ymin = min(route_profile(sc, rid)["elev_m"].min() for rid in sc.routes)
        fig = px.scatter(frames, x="dist_m", y="elev_m", animation_frame="minuta", animation_group="truck",
                         color="state", color_discrete_map=STATE_COLORS, text="truck", hover_data=["stan", "excavator"],
                         range_x=[rng_x, -100], range_y=[ymin - 20, 30])
        for rid in sc.routes:
            prof = route_profile(sc, rid)
            fig.add_scatter(x=prof["dist_m"], y=prof["elev_m"], mode="lines", line=dict(width=6, color="#ced4da"),
                            showlegend=False, hoverinfo="skip")
        fig.update_traces(marker=dict(size=14), textposition="top center", selector=dict(mode="markers+text"))
        fig.update_layout(height=450, xaxis_title="odległość do kruszarki [m]", yaxis_title="wysokość [m]")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- comparisons
def variant_yaml(base_raw: dict, **changes) -> str:
    r = copy.deepcopy(base_raw)
    if "dispatch" in changes:
        r["dispatch"] = changes["dispatch"]
    if "stockpile" in changes:
        r.setdefault("stockpile", {})["enabled"] = changes["stockpile"]
    if "pacing" in changes:
        r.setdefault("pacing", {})["enabled"] = changes["pacing"]
    return dump_scenario(scenario_from_dict(r))


with tabs[4]:
    base_raw = st.session_state.raw_edited
    st.subheader("Ile wozideł? – przegląd liczby wozideł dla wariantów organizacji")
    c = st.columns(4)
    n_min = c[0].number_input("Od", 1, 60, max(1, sc.truck_count - 4))
    n_max = c[1].number_input("Do", 1, 60, sc.truck_count + 4)
    reps = c[2].number_input("Powtórzeń (zmian) na punkt", 1, 20, 3)
    target = c[3].number_input("Cel wydajności [t/h] (0 = brak)", 0, 20000, 0)
    variants = {
        "obecny scenariusz": {},
        "przydział stały do koparki": {"dispatch": "fixed"},
        "przydział automatyczny": {"dispatch": "dynamic"},
        "+ hałda buforowa": {"stockpile": True},
        "+ doradztwo prędkości": {"pacing": True},
        "+ hałda + doradztwo": {"stockpile": True, "pacing": True},
    }
    chosen = st.multiselect("Warianty", list(variants), default=["obecny scenariusz", "+ hałda buforowa"])
    if st.button("▶️ Uruchom przegląd", type="primary"):
        st.session_state.sweep = None
        frames = []
        with st.spinner("Symulacja wariantów…"):
            for name in chosen:
                df = run_sweep(variant_yaml(base_raw, **variants[name]), int(n_min), int(n_max), int(reps))
                df["wariant"] = name
                frames.append(df)
        st.session_state.sweep = pd.concat(frames) if frames else None
    sw = st.session_state.get("sweep")
    if sw is not None:
        c1, c2 = st.columns(2)
        fig = px.line(sw, x="trucks", y="throughput_tph", color="wariant", markers=True,
                      labels={"trucks": "liczba wozideł", "throughput_tph": "wywóz [t/h]"})
        if target:
            fig.add_hline(y=target, line_dash="dot", annotation_text="cel")
        c1.plotly_chart(fig, use_container_width=True)
        c2.plotly_chart(px.line(sw, x="trucks", y="cost_per_t", color="wariant", markers=True,
                                labels={"trucks": "liczba wozideł", "cost_per_t": "koszt [PLN/t]"}), use_container_width=True)
        c1.plotly_chart(px.line(sw, x="trucks", y="avg_queue_per_cycle_min", color="wariant", markers=True,
                                labels={"trucks": "liczba wozideł", "avg_queue_per_cycle_min": "kolejki [min/cykl]"}), use_container_width=True)
        c2.plotly_chart(px.line(sw, x="trucks", y="excavator_utilization", color="wariant", markers=True,
                                labels={"trucks": "liczba wozideł", "excavator_utilization": "wykorzystanie koparek"}), use_container_width=True)
        best = []
        for name, g in sw.groupby("wariant"):
            g2 = g[g["throughput_tph"] >= target] if target else g
            g2 = g2 if len(g2) else g
            b = g2.loc[g2["cost_per_t"].idxmin()]
            best.append({"wariant": name, "najlepiej wozideł": int(b["trucks"]), "t/h": round(b["throughput_tph"]),
                         "PLN/t": round(b["cost_per_t"], 2), "kolejki min/cykl": round(b["avg_queue_per_cycle_min"], 1)})
        st.markdown("**Rekomendacje** (najniższy koszt na tonę" + (f" przy ≥ {target} t/h)" if target else ")"))
        st.dataframe(pd.DataFrame(best), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Porównanie zapisanych scenariuszy")
    p = presets()
    picks = st.multiselect("Scenariusze", ["(bieżący – edytowany)"] + list(p), default=["(bieżący – edytowany)"] + list(p)[:2])
    reps2 = st.number_input("Powtórzeń (zmian)", 1, 20, 5, key="reps2")
    if st.button("▶️ Porównaj scenariusze"):
        rows = []
        with st.spinner("Symulacja…"):
            for name in picks:
                text = yaml_text if name.startswith("(bieżący") else dump_scenario(
                    scenario_from_dict(yaml.safe_load(p[name].read_text(encoding="utf-8"))))
                rows.append({"scenariusz": name, **run_replicate(text, int(reps2))})
        st.session_state.compare = pd.DataFrame(rows)
    cmp_df = st.session_state.get("compare")
    if cmp_df is not None:
        show = cmp_df[["scenariusz", "throughput_tph", "crushed_tph", "cost_per_t", "avg_queue_per_cycle_min",
                       "excavator_utilization", "crusher_utilization", "fuel_l_per_t", "stockpile_end_t"]]
        st.dataframe(show.rename(columns={"throughput_tph": "wywóz t/h", "crushed_tph": "kruszenie t/h",
                                          "cost_per_t": "PLN/t", "avg_queue_per_cycle_min": "kolejki min/cykl",
                                          "excavator_utilization": "wyk. koparek", "crusher_utilization": "wyk. kruszarki",
                                          "fuel_l_per_t": "paliwo l/t", "stockpile_end_t": "hałda na koniec t"}),
                     hide_index=True, use_container_width=True)
        cost_cols = [c for c in cmp_df if c.startswith("koszt_")]
        long = cmp_df.melt(id_vars="scenariusz", value_vars=cost_cols, var_name="składnik", value_name="PLN")
        long["składnik"] = long["składnik"].str.replace("koszt_", "")
        st.plotly_chart(px.bar(long, x="scenariusz", y="PLN", color="składnik", title="Koszt zmiany wg składników"),
                        use_container_width=True)


# ---------------------------------------------------------------- blast vs crusher
with tabs[5]:
    st.subheader("Mine-to-mill: ile warto dopłacić do odstrzału?")
    st.caption("Lepsze rozdrobnienie (0) = droższe wiercenie i strzelanie, ale szybszy załadunek, wyższa wydajność "
               "i mniej energii na kruszarce. Wszystkie odstrzały w scenariuszu dostają tę samą wartość.")
    reps3 = st.number_input("Powtórzeń (zmian) na punkt", 1, 10, 2, key="reps3")
    if st.button("▶️ Przelicz rozdrobnienie", type="primary"):
        with st.spinner("Symulacja…"):
            st.session_state.frag = run_frag(yaml_text, tuple(round(x * 0.1, 1) for x in range(11)), int(reps3))
    fr = st.session_state.get("frag")
    if fr is not None:
        c1, c2 = st.columns(2)
        cost_cols = [c for c in fr if c.startswith("koszt_")]
        per_t = fr[["fragmentation"] + cost_cols].copy()
        for ccol in cost_cols:
            per_t[ccol] = per_t[ccol] / fr["hauled_t"]
        long = per_t.melt(id_vars="fragmentation", var_name="składnik", value_name="PLN/t")
        long["składnik"] = long["składnik"].str.replace("koszt_", "")
        fig = px.area(long, x="fragmentation", y="PLN/t", color="składnik",
                      labels={"fragmentation": "rozdrobnienie (0 = drobne, 1 = bryły)"})
        best = fr.loc[fr["cost_per_t"].idxmin()]
        fig.add_vline(x=best["fragmentation"], line_dash="dot", annotation_text="optimum")
        c1.plotly_chart(fig, use_container_width=True)
        c2.plotly_chart(px.line(fr, x="fragmentation", y=["throughput_tph", "crushed_tph"], markers=True,
                                labels={"fragmentation": "rozdrobnienie", "value": "t/h", "variable": ""}),
                        use_container_width=True)
        st.success(f"Najniższy koszt całkowity: rozdrobnienie ≈ {best['fragmentation']:.1f} → "
                   f"{best['cost_per_t']:.2f} PLN/t, {best['throughput_tph']:.0f} t/h")


# ---------------------------------------------------------------- ML
with tabs[6]:
    st.subheader("Model czasu przejazdu i ETA pod koparkę")
    st.caption("Uczy się na danych z wielu symulowanych zmian – tak samo będzie uczył się na danych z telefonów. "
               "Porównanie z modelem czysto fizycznym (bez stylu kierowcy, deszczu, pory zmiany).")
    shifts = st.slider("Liczba zmian do nauki", 5, 100, 30, 5)
    if st.button("▶️ Generuj dane i trenuj", type="primary"):
        with st.spinner("Generowanie danych i trening…"):
            st.session_state.ml = run_train(yaml_text, shifts)
    if st.session_state.get("ml"):
        data, model, m = st.session_state.ml
        c = st.columns(4)
        c[0].metric("Rekordów (odcinków)", f"{m['train_rows'] + m['test_rows']:,}".replace(",", " "))
        c[1].metric("Błąd modelu ML", f"{m['mae_model_s']:.1f} s/odcinek")
        c[2].metric("Błąd modelu fizycznego", f"{m['mae_physics_baseline_s']:.1f} s/odcinek")
        c[3].metric("Poprawa", f"{100 * (1 - m['mae_model_s'] / m['mae_physics_baseline_s']):.0f} %")
        sample = data.sample(min(3000, len(data)), random_state=0).copy()
        sample["przewidywany_s"] = model.predict_s(sample)
        st.plotly_chart(px.scatter(sample, x="duration_s", y="przewidywany_s", color="segment", opacity=0.5,
                                   labels={"duration_s": "rzeczywisty czas [s]", "przewidywany_s": "przewidywany [s]"}),
                        use_container_width=True)
        st.markdown("**Kalkulator ETA**")
        c = st.columns(5)
        tr = c[0].selectbox("Wozidło", sorted(data["truck"].unique()))
        ex_id = c[1].selectbox("Koparka", [e.id for e in sc.excavators])
        ex_cfg = next(e for e in sc.excavators if e.id == ex_id)
        segs = [s.name for s in reversed(sc.routes[ex_cfg.route].segments)]
        seg_i = c[2].selectbox("Aktualny odcinek", range(len(segs)), format_func=lambda i: segs[i])
        prog = c[3].slider("Przejechane z odcinka", 0.0, 1.0, 0.5)
        rain = c[4].toggle("Deszcz")
        eta = model.eta_to_excavator_s(sc, tr, ex_id, seg_i, prog, rain, 2.0)
        st.info(f"🚚 {tr} będzie pod {ex_id} za **{int(eta // 60)} min {int(eta % 60)} s**")
