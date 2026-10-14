"""Panel scenariuszy – uruchomienie: streamlit run app.py"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from kruszyk.analytics import match_factor_report
from kruszyk.config import dump_scenario, scenario_from_dict
from kruszyk.ml import generate_dataset, train_travel_model
from kruszyk.optimize import replicate, sweep_fragmentation, sweep_trucks
from kruszyk.sim import simulate
from kruszyk.anim import animation_html
from kruszyk.rl import POLICIES
from kruszyk.rl import explain as rl_explain
from kruszyk.rl import explain_advisor, explain_excavators
from kruszyk.rl import train as rl_train
from kruszyk.views import STATE_LABELS, driver_instruction, operator_view, route_profile, snapshot

from app_map import render_live_map, render_map_tab
from app_simple import animation_page, render_simple, run_policy

SCEN_DIR = Path(__file__).parent / "scenarios"
STATE_COLORS = {
    "jazda_pusty": "#8ecae6", "jazda_pelny": "#219ebc", "kolejka_koparka": "#e63946",
    "podstawienie": "#f4a261", "zaladunek": "#2a9d8f", "kolejka_kruszarka": "#d62828",
    "wyladunek": "#264653", "wyladunek_halda": "#6c757d", "awaria": "#000000",
    "postoj_brak_urobku": "#adb5bd", "wyjazd_z_frontu": "#e9c46a", "mijanka": "#ff4d6d", "inne_przestoje": "#7f5539",
    "parking": "#90be6d", "cel_osiagniety": "#43aa8b",
}
BASE_TS = pd.Timestamp("2026-01-01 06:00")

st.set_page_config(page_title="Kruszyk – załadunek, transport, kruszenie", page_icon="⛏️", layout="wide")


# ------------------------------------------------------------------ state
DEFAULT_SCENARIO = "kruszyk_demo_zakrzowek"  # the mine opened at start


def presets() -> dict[str, Path]:
    return {yaml.safe_load(p.read_text(encoding="utf-8")).get("name", p.stem): p
            for p in sorted(SCEN_DIR.glob("*.yaml"))}


def set_raw(raw: dict, new_mine: bool = False):
    st.session_state.raw = raw
    st.session_state.ver = st.session_state.get("ver", 0) + 1  # new widget keys -> widgets reload
    st.session_state.pop("res", None)
    if new_mine:  # another mine loaded: variant, policies, plans and results belong to the previous one
        for key in ("variant_raw", "variant_yaml", "rl_by_target", "rl_result", "period", "prod_gp", "prod_applied",
                    "cal_result", "mc_res", "goal_res", "info_sel", "prod_mix_df", "prod_y_ids", "prod_y_df"):
            st.session_state.pop(key, None)


def new_project(name: str) -> dict:
    """An empty mine with the usual settings of the default mine."""
    from kruszyk.project import new_project as make

    return make(yaml.safe_load(presets_by_stem()[DEFAULT_SCENARIO].read_text(encoding="utf-8")), name)


def presets_by_stem() -> dict[str, Path]:
    return {p.stem: p for p in sorted(SCEN_DIR.glob("*.yaml"))}


def slug(name: str) -> str:
    import re
    import unicodedata

    s_ = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s_).strip("_").lower() or "kopalnia"


def export_scenario() -> str:
    """Everything needed to restore the work: scenario + current map (also unsaved changes) + variant."""
    from kruszyk.site import site_to_scenario

    data = copy.deepcopy(st.session_state.raw)
    site = st.session_state.get("site")
    if site and site != data.get("site"):
        try:  # unsaved map changes: build the routes so the file is complete
            data = site_to_scenario(copy.deepcopy(site), data)
        except ValueError:
            data["site"] = copy.deepcopy(site)  # incomplete map – keep the drawing anyway
    if st.session_state.get("variant_raw") is not None:
        data["variant"] = copy.deepcopy(st.session_state.variant_raw)
    data["_plik"] = {"format": "kruszyk-1", "zapisano": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    from kruszyk.config import yaml_dump

    return yaml_dump(data)


def import_scenario(text: str) -> None:
    data = yaml.safe_load(text) or {}
    variant = data.pop("variant", None)
    data.pop("_plik", None)
    set_raw(data, new_mine=True)
    if variant:
        st.session_state.variant_raw = variant
        st.session_state.variant_ver = st.session_state.get("variant_ver", 0) + 1


if "raw" not in st.session_state:
    p = presets()
    first = next((k for k, v in p.items() if v.stem == DEFAULT_SCENARIO), next(iter(p)))
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
    choice = st.selectbox("Scenariusz przykładowy", list(p),
                          index=next((i for i, v in enumerate(p.values()) if v.stem == DEFAULT_SCENARIO), 0))
    if st.button("Wczytaj scenariusz", width="stretch"):
        set_raw(yaml.safe_load(p[choice].read_text(encoding="utf-8")), new_mine=True)
        st.rerun()
    with st.expander("🆕 Nowy projekt"):
        st.caption("Pusta kopalnia: mapę, drogi, usypy, maszyny i kruszarki opisujesz od zera (krok ① Kopalnia dziś). "
                   "Ustawienia zmiany, kosztów i losowości są typowe – do poprawienia. **Bieżąca kopalnia zostanie "
                   "zamknięta** – zapisz ją wcześniej do pliku.")
        new_name = st.text_input("Nazwa kopalni", "", key="new_project_name", placeholder="np. Kopalnia Wapienia Zawada")
        if st.button("Załóż nowy projekt", width="stretch", disabled=not new_name.strip(), key="new_project_go"):
            set_raw(new_project(new_name), new_mine=True)
            st.rerun()
    st.divider()
    st.markdown("**📁 Mój scenariusz**")
    st.caption("Zapisz ustawioną kopalnię do pliku i wczytaj ją następnym razem – bez wyklikiwania od nowa. "
               "Plik zawiera mapę, drogi, usypy, wozidła, zdarzenia, wariant i nauczoną politykę.")
    st.download_button("💾 Zapisz scenariusz do pliku", export_scenario(), width="stretch",
                       file_name=f"{slug(raw.get('name', 'kopalnia'))}.kruszyk.yaml", mime="text/yaml")
    up = st.file_uploader("📂 Wczytaj scenariusz z pliku", type=["yaml", "yml"])
    if up is not None and st.session_state.get("uploaded") != up.file_id:
        st.session_state.uploaded = up.file_id
        import_scenario(up.getvalue().decode("utf-8"))
        st.rerun()
    st.divider()
    advanced = st.toggle("🔧 Tryb zaawansowany", value=False, key="advanced",
                         help="Wszystkie parametry i analizy: wyniki zmiany, podgląd na żywo z ekranami operatora "
                              "i kierowców, porównania wariantów, szczegóły uczenia, odstrzał ↔ kruszarka, model ETA.")
    seed = int(raw.get("seed", 42))
    if advanced:
        seed = st.number_input("Ziarno losowania (inna zmiana = inne ziarno)", 0, 10_000, seed)

if not advanced:
    render_simple(raw, set_raw, render_map_tab)
    st.stop()


# ---------------------------------------------------------------- editor
def records(df: pd.DataFrame) -> list[dict]:
    df = df.dropna(how="all")
    out = []
    for r in df.to_dict("records"):
        out.append({kk: (v.item() if hasattr(v, "item") else v) for kk, v in r.items()
                    if isinstance(v, (list, dict)) or not pd.isna(v)})
    return out


def editor() -> dict:
    new = copy.deepcopy(raw)
    c1, c2, c3, c4 = st.columns(4)
    new["name"] = c1.text_input("Nazwa scenariusza", raw.get("name", ""), key=k("name"))
    new["shift_hours"] = c2.number_input("Długość zmiany [h]", 1.0, 24.0, float(raw["shift_hours"]), 0.5, key=k("shift"))
    modes = ["natural", "fixed", "dynamic"] + (["rl"] if raw.get("rl_weights") else [])
    cur = raw.get("dispatch", "dynamic")
    new["dispatch"] = c3.selectbox("Przydział wozideł", modes, index=modes.index(cur) if cur in modes else 2,
                                   key=k("disp"), format_func=POLICIES.get,
                                   help="naturalny = obecna praktyka (kierowca wraca do „swojej” koparki, czasem jedzie "
                                        "gdzie indziej, bez informacji); heurystyka = system wysyła tam, gdzie najszybciej "
                                        "załadują; RL = polityka wyuczona w zakładce „Uczenie RL”")
    new["loose_density_t_m3"] = c4.number_input("Gęstość urobku spulchn. [t/m³]", 1.0, 3.0,
                                                float(raw.get("loose_density_t_m3", 1.75)), 0.05, key=k("dens"))
    new["seed"] = int(seed)

    st.subheader("Odstrzały i koparki")
    c1, c2 = st.columns([2, 3])
    psd_by_id = {b["id"]: b["psd"] for b in raw["blasts"] if b.get("psd")}  # curves are edited on the map screen
    new["blasts"] = records(c1.data_editor(pd.DataFrame([{k_: v for k_, v in b.items() if k_ != "psd"} for b in raw["blasts"]]),
                                           num_rows="dynamic", key=k("blasts"),
        column_config={"id": "Odstrzał", "tonnage_t": st.column_config.NumberColumn("Tonaż [t]", min_value=0),
                       "fragmentation": st.column_config.NumberColumn("Rozdrobnienie 0=dobre 1=bryły",
                                                                      min_value=0.0, max_value=1.0, step=0.05)},
        width="stretch"))
    for b in new["blasts"]:
        if b.get("id") in psd_by_id:
            b["psd"] = psd_by_id[b["id"]]
    if psd_by_id:
        c1.caption("Usypy z krzywą uziarnienia: rozdrobnienie liczone z krzywej (edycja na ekranie mapy).")
    route_ids = list(raw["routes"])
    blast_ids = [b["id"] for b in new["blasts"] if "id" in b]
    # nested fields set on the map screen (spotting, routes to crushers, digging path) do not survive the table
    nested = ("spotting", "crusher_routes", "dig_routes", "dig_path")
    keep_by_id = {e["id"]: {f: e[f] for f in nested if e.get(f)} for e in raw["excavators"]}
    spot_by_id = {i: v["spotting"] for i, v in keep_by_id.items() if "spotting" in v}
    ex_df = pd.DataFrame([{k_: v for k_, v in e.items() if k_ not in nested} for e in raw["excavators"]])
    for col, default in (("fill_factor", 0.9), ("spot_time_s", 40.0)):
        if col not in ex_df:
            ex_df[col] = default
    new["excavators"] = records(c2.data_editor(ex_df, num_rows="dynamic", key=k("ex"), width="stretch",
        column_config={"id": "Koparka", "bucket_m3": "Łyżka [m³]", "swing_cycle_s": "Cykl łyżki [s]",
                       "blast": st.column_config.SelectboxColumn("Odstrzał", options=blast_ids),
                       "route": st.column_config.SelectboxColumn("Trasa", options=route_ids),
                       "fill_factor": "Wsp. napełnienia", "spot_time_s": "Podstawienie [s]"}))
    for e in new["excavators"]:
        e.update(keep_by_id.get(e.get("id"), {}))
    if spot_by_id:
        c2.caption("Koparki ze sposobami podstawiania (ekran mapy): „Podstawienie [s]” nie jest używane.")

    st.subheader("Wozidła")
    st.caption("**Koparki** – do których koparek grupa może jeździć, np. `K-1, K-2` (puste = wszystkie). "
               "**Dwa usypy w kursie** – np. `K-1:0.5, K-2:0.5` = pół ładunku z K-1, potem przejazd i pół z K-2 "
               "(wymaga tras z mapy).")
    tr_df = pd.DataFrame(raw["trucks"]) if raw["trucks"] else pd.DataFrame(
        columns=["id", "model", "count", "payload_t", "empty_mass_t", "power_kw", "max_speed_kmh"])
    tr_df["excavators"] = pd.Series([", ".join(x) if isinstance(x, list) else ""
                                     for x in tr_df.get("excavators", [None] * len(tr_df))], dtype="object")
    tr_df["blend"] = pd.Series([", ".join(f"{a}:{b:g}" for a, b in x.items()) if isinstance(x, dict) else ""
                                for x in tr_df.get("blend", [None] * len(tr_df))], dtype="object")
    new["trucks"] = records(st.data_editor(tr_df, num_rows="dynamic", key=k("trucks"), width="stretch",
        column_config={"model": "Model", "count": st.column_config.NumberColumn("Liczba", min_value=0, step=1),
                       "excavators": st.column_config.TextColumn("Koparki (dozwolone)"),
                       "blend": st.column_config.TextColumn("Dwa usypy w kursie"),
                       "payload_t": "Ładowność [t]", "empty_mass_t": "Masa własna [t]", "power_kw": "Moc [kW]",
                       "max_speed_kmh": "V max [km/h]", "drive_efficiency": "Sprawność napędu",
                       "fuel_lph_loaded": "Paliwo pełny [l/h]", "fuel_lph_empty": "Paliwo pusty [l/h]",
                       "fuel_lph_idle": "Paliwo postój [l/h]"}))
    for t in new["trucks"]:
        t["count"] = int(t.get("count", 0))
        t["excavators"] = [x.strip() for x in str(t.get("excavators", "")).split(",") if x.strip()]
        blend = {}
        for part in str(t.get("blend", "")).split(","):
            if ":" in part:
                a, b = part.split(":", 1)
                try:
                    blend[a.strip()] = float(b.replace("%", "").strip())
                except ValueError:
                    pass
        t["blend"] = blend

    st.subheader("Trasy (od frontu do kosza kruszarki; nachylenie + = pod górę z ładunkiem)")
    seg_rows = [{"route": rid, **s} for rid, r in raw["routes"].items() for s in r["segments"]]
    seg_df = st.data_editor(pd.DataFrame(seg_rows), num_rows="dynamic", key=k("segs"), width="stretch",
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
        cs["truck_variable_share"] = c[3].number_input(
            "Wozidło: część zależna od motogodzin [%]", 0.0, 100.0, 100 * float(cs.get("truck_variable_share", 0.4)),
            5.0, key=k("ctruck_var"), help="serwis, opony, zużycie – tyle stawki oszczędza wozidło zaparkowane "
                                          "z wyłączonym silnikiem (decyzja polityki RL); reszta to kierowca i leasing") / 100
        new["costs"] = cs
    return new


tabs = st.tabs(["🗺️ Mapa kopalni", "🎬 Animacja", "⚙️ Scenariusz", "📊 Wyniki zmiany", "📍 Podgląd na żywo",
                "⚖️ Porównanie wariantów", "🧠 Uczenie RL", "💥 Odstrzał ↔ kruszarka", "🤖 Model ETA"])

with tabs[2]:
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
            hide_index=True, width="stretch")
        st.download_button("💾 Pobierz scenariusz (YAML)", yaml_text, file_name="scenariusz.yaml")

with tabs[2]:
    from app_events import events_editor

    with st.expander("🕐 Organizacja zmiany – godziny, przerwy, strzelanie, objazd, tankowanie, prędkości"):
        # edits the scenario kept in the session: the next run's editor copy carries it
        events_editor(st.session_state.raw, "eva", [e["id"] for e in raw.get("excavators", [])], raw.get("site"))

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


with tabs[3]:
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
        "blast_done_at_h": "odstrzał wybrany po [h]"}), hide_index=True, width="stretch")

    st.markdown("**Oś czasu wozideł** – czerwone = czekanie")
    st.plotly_chart(gantt(res), width="stretch")

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
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Struktura kosztu [PLN/t]**")
        cb = pd.Series(kp["cost_breakdown"]) / max(kp["hauled_t"], 1)
        fig = px.bar(cb[cb > 0].sort_values(), orientation="h", labels={"value": "PLN/t", "index": ""})
        fig.update_layout(height=320, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

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


with tabs[4]:
    shift_min = int(sc.shift_hours * 60)
    minute = st.slider("Moment zmiany [min od startu]", 0, shift_min - 1, min(180, shift_min - 1), 1)
    t = minute * 60.0
    snap = snapshot(res, t)
    clock = (BASE_TS + pd.Timedelta(seconds=t)).strftime("%H:%M")
    hop = res.hopper[res.hopper["t_s"] <= t]
    level = hop["level_t"].iloc[-1] if len(hop) else 0.0
    st.markdown(f"### 🕐 {clock} · kosz kruszarki: {level:.0f} / {sc.crusher.hopper_t:.0f} t")
    render_live_map(sc, snap, STATE_COLORS, STATE_LABELS)
    st.plotly_chart(profile_figure(snap), width="stretch")

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
            "truck": "wozidło", "excavator": "koparka", "segment": "odcinek"}), hide_index=True, width="stretch")

    if st.toggle("▶️ Animacja całej zmiany (liczy się kilka sekund)"):
        step = st.select_slider("Krok animacji [min]", [2, 5, 10], value=5)
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
        st.plotly_chart(fig, width="stretch")


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


with tabs[5]:
    base_raw = st.session_state.raw_edited
    st.subheader("Ile wozideł? – przegląd liczby wozideł dla wariantów organizacji")
    c = st.columns(4)
    n_min = c[0].number_input("Od", 1, 60, max(1, sc.truck_count - 4))
    n_max = c[1].number_input("Do", 1, 60, sc.truck_count + 4)
    reps = c[2].number_input("Powtórzeń (zmian) na punkt", 1, 20, 3)
    target = c[3].number_input("Cel wydajności [t/h] (0 = brak)", 0, 20000, 0)
    variants = {
        "obecny scenariusz": {},
        "przydział naturalny": {"dispatch": "natural"},
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
        c1.plotly_chart(fig, width="stretch")
        c2.plotly_chart(px.line(sw, x="trucks", y="cost_per_t", color="wariant", markers=True,
                                labels={"trucks": "liczba wozideł", "cost_per_t": "koszt [PLN/t]"}), width="stretch")
        c1.plotly_chart(px.line(sw, x="trucks", y="avg_queue_per_cycle_min", color="wariant", markers=True,
                                labels={"trucks": "liczba wozideł", "avg_queue_per_cycle_min": "kolejki [min/cykl]"}), width="stretch")
        c2.plotly_chart(px.line(sw, x="trucks", y="excavator_utilization", color="wariant", markers=True,
                                labels={"trucks": "liczba wozideł", "excavator_utilization": "wykorzystanie koparek"}), width="stretch")
        best = []
        for name, g in sw.groupby("wariant"):
            g2 = g[g["throughput_tph"] >= target] if target else g
            g2 = g2 if len(g2) else g
            b = g2.loc[g2["cost_per_t"].idxmin()]
            best.append({"wariant": name, "najlepiej wozideł": int(b["trucks"]), "t/h": round(b["throughput_tph"]),
                         "PLN/t": round(b["cost_per_t"], 2), "kolejki min/cykl": round(b["avg_queue_per_cycle_min"], 1)})
        st.markdown("**Rekomendacje** (najniższy koszt na tonę" + (f" przy ≥ {target} t/h)" if target else ")"))
        st.dataframe(pd.DataFrame(best), hide_index=True, width="stretch")

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
                     hide_index=True, width="stretch")
        cost_cols = [c for c in cmp_df if c.startswith("koszt_")]
        long = cmp_df.melt(id_vars="scenariusz", value_vars=cost_cols, var_name="składnik", value_name="PLN")
        long["składnik"] = long["składnik"].str.replace("koszt_", "")
        st.plotly_chart(px.bar(long, x="scenariusz", y="PLN", color="składnik", title="Koszt zmiany wg składników"),
                        width="stretch")


# ---------------------------------------------------------------- blast vs crusher
with tabs[7]:
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
        c1.plotly_chart(fig, width="stretch")
        c2.plotly_chart(px.line(fr, x="fragmentation", y=["throughput_tph", "crushed_tph"], markers=True,
                                labels={"fragmentation": "rozdrobnienie", "value": "t/h", "variable": ""}),
                        width="stretch")
        st.success(f"Najniższy koszt całkowity: rozdrobnienie ≈ {best['fragmentation']:.1f} → "
                   f"{best['cost_per_t']:.2f} PLN/t, {best['throughput_tph']:.0f} t/h")


# ---------------------------------------------------------------- ML
with tabs[8]:
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
                        width="stretch")
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


# ---------------------------------------------------------------- animation on the map
with tabs[1]:
    if not any(r.path for r in sc.routes.values()):
        st.info("Animacja jeździ po drogach narysowanych na mapie. Narysuj kopalnię w zakładce **🗺️ Mapa kopalni** "
                "i kliknij **Zbuduj trasy**, albo wczytaj w panelu bocznym scenariusz **„mapa demo”**.")
    else:
        options = ["natural", "fixed", "dynamic"] + (["rl"] if sc.rl_weights else [])
        c1, c2 = st.columns([3, 2])
        policy = c1.radio("Przydział wozideł w animacji", options, horizontal=True, format_func=POLICIES.get,
                          index=options.index(sc.dispatch) if sc.dispatch in options else 0, key=k("anim_policy"))
        c2.caption("Cała zmiana symulowana w ułamku sekundy, odtwarzana przyspieszona. Koszty liczone na bieżąco; "
                   "porównanie z przydziałem naturalnym **tego samego dnia** (ta sama pogoda, awarie, kierowcy)."
                   + ("" if sc.rl_weights else " Polityka RL pojawi się po treningu w zakładce 🧠 Uczenie RL."))
        components.html(animation_page(yaml_text, int(seed), policy), height=780, scrolling=False)
        a, b = run_policy(yaml_text, int(seed), policy).kpi, run_policy(yaml_text, int(seed), "natural").kpi
        rows = [("Wywiezione [t]", "hauled_t", 0), ("Wydajność [t/h]", "throughput_tph", 0), ("Koszt [zł/t]", "cost_per_t", 2),
                ("Paliwo [l/t]", "fuel_l_per_t", 3), ("Kolejki [min/cykl]", "avg_queue_per_cycle_min", 1),
                ("Kolejka pod kruszarką [h]", "truck_queue_crusher_h", 1), ("Kolejka pod koparkami [h]", "truck_queue_excavator_h", 1)]
        st.dataframe(pd.DataFrame([{"wskaźnik (cała zmiana)": n, POLICIES[policy]: round(a[key] or 0, d),
                                    "naturalny": round(b[key] or 0, d)} for n, key, d in rows]),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------- reinforcement learning
with tabs[6]:
    st.subheader("Uczenie ze wzmocnieniem – przydział wozideł do koparek")
    st.markdown(
        "Agent decyduje po każdym wyładunku, **do której koparki** jedzie wozidło. Widzi to, co system wie w tej chwili: "
        "przewidywane czekanie pod każdą koparką, dojazd, odwóz, czas załadunku, pozostały urobek, ile wozideł już tam "
        "jedzie i czy to „własna” koparka kierowcy. Uczy się na tysiącach symulowanych zmian z losowością "
        "(pogoda, kierowcy, awarie, czasy załadunku) – nagrodą jest **niższy koszt na tonę**. "
        "Metoda: przeszukiwanie polityk metodą entropii krzyżowej (CEM).")
    allowed_multi = any(len(f.excavators or [e.id for e in sc.excavators]) > 1 and not f.blend for f in sc.trucks)
    if len(sc.excavators) < 2 or not allowed_multi:
        st.warning("Agent ma czego się uczyć dopiero wtedy, gdy wozidła mogą wybierać spośród co najmniej dwóch koparek.")
    c = st.columns(5)
    iters = c[0].number_input("Iteracje", 2, 60, 10)
    pop = c[1].number_input("Polityk na iterację", 4, 40, 12)
    eps = c[2].number_input("Zmian na politykę", 1, 10, 2)
    ep_h = c[3].number_input("Długość zmiany w nauce [h]", 1.0, 12.0, min(4.0, float(sc.shift_hours)), 1.0)
    ev = c[4].number_input("Zmian w teście końcowym", 1, 20, 4)
    c = st.columns(3)
    ex_learn = c[2].toggle("⛏️ Ucz zarządzania koparkami (ile pracuje, ostatnia łyżka)", True, key="ex_rl_learn")
    adv_on = c[0].toggle("📱 Zalecenia w czasie rzeczywistym dla agenta", True, key="adv_rl_on")
    adv_learn = c[1].toggle("🧠 Ucz też ustawień zaleceń (zapas przyjazdu, zwalnianie)", True, key="adv_rl_learn",
                            disabled=not adv_on)
    n_sims = int(iters * (pop + 2) * eps + eps * 2 + 5 * ev)
    st.caption(f"≈ {n_sims} symulowanych zmian, zwykle 15–60 s.")
    if st.button("🧠 Ucz agenta", type="primary"):
        bar, live = st.progress(0.0, "start"), st.empty()
        rows: list[dict] = []

        def cb(it, n, row):
            rows.append(row)
            bar.progress(it / n, f"iteracja {it}/{n}: polityka {row['koszt_polityka']:.2f} zł/t, "
                                  f"naturalny {row['koszt_naturalny']:.2f} zł/t")
            live.line_chart(pd.DataFrame(rows).set_index("iteracja")[["koszt_polityka", "koszt_naturalny"]])

        result = rl_train(sc, iterations=int(iters), population=int(pop), episodes=int(eps), shift_hours=float(ep_h),
                          eval_episodes=int(ev), callback=cb, advisor=adv_on, learn_advisor=adv_learn,
                          learn_excavators=ex_learn)
        live.empty()
        bar.empty()
        st.session_state.rl_result = {"result": result, "scenario": sc.name}

    rr = st.session_state.get("rl_result")
    if rr:
        r = rr["result"]
        h = r.history
        st.markdown(f"#### Przebieg uczenia · {rr['scenario']}")
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=h["iteracja"], y=h["koszt_naturalny"], name="naturalny (te same zmiany)", line=dict(color="#9aa0a6", dash="dot"))
        fig.add_scatter(x=h["iteracja"], y=h["koszt_populacja_sredni"], name="próbowane polityki (średnio)", line=dict(color="#f4a261"))
        fig.add_scatter(x=h["iteracja"], y=h["koszt_polityka"], name="polityka agenta", line=dict(color="#d62828", width=4))
        fig.update_layout(title="Koszt na tonę [zł/t]", height=340, xaxis_title="iteracja (0 = agent nienauczony)",
                          legend=dict(orientation="h", y=-0.3), margin=dict(t=40))
        c1.plotly_chart(fig, width="stretch")
        hh = h[h["iteracja"] > 0]
        fig = go.Figure(go.Bar(x=hh["iteracja"], y=hh["oszczednosc_pct"],
                               marker_color=["#2a9d8f" if v > 0 else "#d62828" for v in hh["oszczednosc_pct"]]))
        fig.update_layout(title="Oszczędność vs przydział naturalny na tych samych zmianach [%]", height=340,
                          xaxis_title="iteracja", yaxis_title="%", margin=dict(t=40))
        c2.plotly_chart(fig, width="stretch")
        c = st.columns(4)
        for col, (key, title) in zip(c, [("throughput_tph", "Wydajność [t/h]"), ("fuel_l_per_t", "Paliwo [l/t]"),
                                         ("avg_queue_per_cycle_min", "Kolejki [min/cykl]"),
                                         ("excavator_utilization", "Wykorzystanie koparek")]):
            fig = go.Figure()
            fig.add_scatter(x=h["iteracja"], y=h[f"naturalny_{key}"], name="naturalny", line=dict(color="#9aa0a6", dash="dot"))
            fig.add_scatter(x=h["iteracja"], y=h[key], name="agent", line=dict(color="#d62828", width=3))
            fig.update_layout(title=title, height=260, showlegend=False, margin=dict(t=40, l=10, r=10))
            col.plotly_chart(fig, width="stretch")

        st.markdown("#### Test końcowy na nowych, pełnych zmianach")
        comp = r.comparison
        nat = comp.iloc[0]
        show = pd.DataFrame({
            "polityka": comp["polityka"],
            "koszt [zł/t]": comp["cost_per_t"].round(2),
            "vs naturalny": (100 * (comp["cost_per_t"] / nat["cost_per_t"] - 1)).round(1).astype(str) + " %",
            "wydajność [t/h]": comp["throughput_tph"].round(0),
            "paliwo [l/t]": comp["fuel_l_per_t"].round(3),
            "kolejki [min/cykl]": comp["avg_queue_per_cycle_min"].round(1),
            "wykorzystanie koparek": comp["excavator_utilization"].round(2),
        })
        st.dataframe(show, hide_index=True, width="stretch")
        best = comp.iloc[-1]
        hours = sc.shift_hours
        saving = (nat["cost_per_t"] - best["cost_per_t"]) * best["hauled_t"]
        st.success(f"Polityka RL: {best['cost_per_t']:.2f} zł/t vs {nat['cost_per_t']:.2f} zł/t przy przydziale naturalnym, "
                   f"wydajność {best['throughput_tph']:.0f} vs {nat['throughput_tph']:.0f} t/h. "
                   f"Przy tej samej ilości urobku to ok. **{saving:,.0f} zł na zmianę** ({hours:g} h).".replace(",", " "))
        c1, c2 = st.columns([3, 2])
        c1.markdown("**Czego nauczył się agent** (dodatnia waga = unika koparek z dużą wartością cechy)")
        c1.dataframe(rl_explain(r.weights), hide_index=True, width="stretch")
        if r.excavators:
            c1.markdown("**Wyuczone zarządzanie koparkami**  \n" + "  \n".join(explain_excavators(r.excavators)))
        if r.advisor:
            c1.markdown("**Wyuczone zalecenia**  \n" + "  \n".join(explain_advisor(r.advisor)))
        with c2:
            st.markdown("**Wdrożenie polityki**")
            st.caption("Zapisuje wyuczone wagi w scenariuszu i ustawia przydział „polityka RL”. Potem zobaczysz ją "
                       "w 🎬 Animacji, wynikach i porównaniach.")
            if st.button("✅ Zastosuj politykę RL w scenariuszu"):
                new_raw = copy.deepcopy(st.session_state.raw_edited)
                new_raw["rl_weights"] = r.weights
                new_raw["dispatch"] = "rl"
                if r.advisor:
                    new_raw["advisor"] = {**new_raw.get("advisor", {}), **r.advisor, "enabled": True}
                if r.excavators:
                    new_raw["excavator_policy"] = {**new_raw.get("excavator_policy", {}), **r.excavators}
                set_raw(new_raw)
                st.rerun()
