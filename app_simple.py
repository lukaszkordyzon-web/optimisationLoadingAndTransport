"""Simple, guided mode of the panel: 1) the mine, 2) how it works today, 3) optimisation."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from kruszyk.anim import animation_html
from kruszyk.config import dump_scenario, scenario_from_dict
from kruszyk.psd import PRESETS, preset_of
from kruszyk.rl import POLICIES
from kruszyk.rl import train as rl_train
from kruszyk.sim import simulate
from kruszyk.site import site_to_scenario

DEMO = Path(__file__).parent / "scenarios" / "kruszyk_mapa_demo.yaml"


# ---------------------------------------------------------------- cached runs (shared with the advanced mode)
@st.cache_data(show_spinner=False)
def run_policy(yaml_text: str, seed: int, policy: str):
    sc_ = scenario_from_dict(yaml.safe_load(yaml_text))
    sc_.dispatch = policy
    return simulate(sc_, seed=seed)


TEMPLATE = Path(__file__).parent / "kruszyk" / "anim_template.html"


def animation_page(yaml_text: str, seed: int, policy: str, compare: bool = True) -> str:
    # the template's modification time is part of the cache key, so a changed player is never served stale
    return _animation_page(yaml_text, seed, policy, compare, TEMPLATE.stat().st_mtime)


def compare_page(yaml_text: str, policy: str, base_yaml: str, seed: int, label: str) -> str:
    """Animation of one scenario/policy with the live comparison against 'today' (base scenario, natural)."""
    return _compare_page(yaml_text, policy, base_yaml, seed, label, TEMPLATE.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _compare_page(yaml_text: str, policy: str, base_yaml: str, seed: int, label: str, template_version: float) -> str:
    res = run_policy(yaml_text, seed, policy)
    base = run_policy(base_yaml, seed, "natural")
    return animation_html(res, base, label, "dziś")


@st.cache_data(show_spinner=False)
def _animation_page(yaml_text: str, seed: int, policy: str, compare: bool, template_version: float) -> str:
    res_p = run_policy(yaml_text, seed, policy)
    base = run_policy(yaml_text, seed, "natural") if compare and policy != "natural" else None
    return animation_html(res_p, base, f"Przydział: {POLICIES[policy]}", "dziś (naturalny)")


# ---------------------------------------------------------------- helpers
def _has_map_routes(sc) -> bool:
    return any(r.path for r in sc.routes.values())


def _demo_button(set_raw, key: str):
    if st.button("📂 Wczytaj przykładową kopalnię (mapa demo)", key=key):
        set_raw(yaml.safe_load(DEMO.read_text(encoding="utf-8")), new_mine=True)
        st.rerun()


def _losses(k: dict) -> list[str]:
    """Where the time and money go – in plain words."""
    ex = k["excavators"]
    out = []
    q_ex, q_cr = k["truck_queue_excavator_h"], k["truck_queue_crusher_h"]
    idle_ex = sum(e["waiting_for_trucks_h"] for e in ex)
    if q_ex > 0.2:
        out.append(f"🚚⏳ Wozidła stały w kolejce **pod koparkami** łącznie **{q_ex:.1f} h**.")
    if q_cr > 0.2:
        out.append(f"🚚⏳ Wozidła stały w kolejce **pod kruszarką** łącznie **{q_cr:.1f} h**.")
    if idle_ex > 0.2:
        worst = max(ex, key=lambda e: e["waiting_for_trucks_h"])
        out.append(f"⛏️💤 Koparki czekały na wozidła łącznie **{idle_ex:.1f} h** "
                   f"(najdłużej {worst['excavator']}: {worst['waiting_for_trucks_h']:.1f} h).")
    if k["crusher_starved_h"] > 0.3:
        out.append(f"🏭💤 Kruszarka stała bez materiału **{k['crusher_starved_h']:.1f} h**.")
    if k["truck_breakdown_h"] > 0.2:
        out.append(f"🔧 Awarie wozideł: **{k['truck_breakdown_h']:.1f} h**.")
    return out or ["✅ Brak wyraźnych strat – łańcuch jest dobrze zbalansowany."]


def _tiles(k: dict, ref: dict | None = None):
    items = [("Wywiezione", "hauled_t", "{:,.0f} t", False), ("Wydajność", "throughput_tph", "{:,.0f} t/h", False),
             ("Koszt na tonę", "cost_per_t", "{:.2f} zł/t", True), ("Paliwo na tonę", "fuel_l_per_t", "{:.3f} l/t", True),
             ("Czekanie w kolejkach", "avg_queue_per_cycle_min", "{:.1f} min/kurs", True)]
    cols = st.columns(len(items))
    for col, (label, key, fmt, lower_better) in zip(cols, items):
        v = k.get(key) or 0
        delta = None
        if ref is not None and ref.get(key):
            d = 100 * (v / ref[key] - 1)
            delta = f"{d:+.1f} % vs dziś"
        col.metric(label, fmt.format(v).replace(",", " "), delta,
                   delta_color="inverse" if lower_better else "normal")


# ---------------------------------------------------------------- the three steps
def step_today(sc, yaml_text: str, seed: int, set_raw):
    st.markdown("### ② Jak jest dziś")
    st.markdown("Symulacja **jednej zmiany tak, jak pracujecie obecnie**: kierowca po wyładunku wraca do „swojej” "
                "koparki, czasem jedzie gdzie indziej, nikt nie wie, gdzie jest kolejka. Do tego losowość: deszcz, "
                "różni kierowcy, awarie, różne czasy załadunku.")
    if not _has_map_routes(sc):
        st.info("Najpierw zbuduj kopalnię w kroku **① Kopalnia** (i kliknij „Zapisz kopalnię”) – albo wczytaj przykład.")
        _demo_button(set_raw, "demo_today")
        return
    k = run_policy(yaml_text, seed, "natural").kpi
    _tiles(k)
    st.markdown("**Gdzie tracimy:**  \n" + "  \n".join(_losses(k)))
    st.caption("▶ Kliknij **Start** na panelu obok mapy – zmiana leci przyspieszona, kropki to wozidła "
               "(kolor = co robią), liczniki pokazują koszty na bieżąco.")
    components.html(animation_page(yaml_text, seed, "natural", False), height=760, scrolling=False)


def _snapshot_editor(key: str, make_df, **kwargs) -> pd.DataFrame:
    """data_editor over a fixed starting table (kept in the session), so edits are never applied twice."""
    base_key = f"{key}__base"
    if base_key not in st.session_state:
        st.session_state[base_key] = make_df()
    return st.data_editor(st.session_state[base_key], key=key, **kwargs)


def _parse_list(text) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def _parse_blend(text) -> dict:
    out = {}
    for part in str(text or "").split(","):
        if ":" in part:
            a, b = part.split(":", 1)
            try:
                out[a.strip()] = float(b.replace("%", "").strip())
            except ValueError:
                pass
    return out


def step_variant(base_raw: dict, base_yaml: str | None, seed: int):
    st.markdown("### ③ Wariant – co jeśli")
    st.markdown("Kopia stanu obecnego, w której **zmieniasz** np. liczbę wozideł, przypisanie do koparek, drogi "
                "(włączasz drogę planowaną, zamykasz istniejącą, zmieniasz nachylenie), wydajność kruszarki albo "
                "uziarnienie usypów – i od razu widzisz wynik obok stanu obecnego. Stan obecny się nie zmienia.")
    if base_yaml is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia**.")
        return
    var = st.session_state.get("variant_raw")
    if var is None:
        if st.button("➕ Utwórz wariant (kopia stanu obecnego)", type="primary"):
            st.session_state.variant_raw = copy.deepcopy(base_raw)
            st.session_state.variant_ver = st.session_state.get("variant_ver", 0) + 1
            st.rerun()
        return
    vv = st.session_state.get("variant_ver", 0)
    c1, c2 = st.columns([3, 1])
    var["name"] = c1.text_input("Nazwa wariantu", var.get("name", "wariant") if var.get("name") != base_raw.get("name")
                                else "Wariant: " + base_raw.get("name", ""), key=f"vname_{vv}")
    if c2.button("🗑️ Usuń wariant"):
        st.session_state.pop("variant_raw")
        st.session_state.pop("variant_rl", None)
        st.rerun()

    st.markdown("**🚚 Wozidła** – zmień liczbę, ładowność albo przypisania (`K-1, K-2`; dwa usypy w kursie: `K-1:0.5, K-2:0.5`).")
    ed = _snapshot_editor(
        f"vtrucks_{vv}",
        lambda: pd.DataFrame([{"model": f.get("model", ""), "count": int(f.get("count", 0)),
                               "payload_t": float(f.get("payload_t", 50)),
                               "excavators": ", ".join(f.get("excavators") or []),
                               "blend": ", ".join(f"{a}:{b:g}" for a, b in (f.get("blend") or {}).items())}
                              for f in var.get("trucks", [])]),
        num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"model": "typ wozidła", "count": st.column_config.NumberColumn("liczba", min_value=0, step=1),
                       "payload_t": st.column_config.NumberColumn("ładowność [t]", min_value=5.0),
                       "excavators": "koparki (puste = wszystkie)", "blend": "dwa usypy w kursie"})
    old = st.session_state[f"vtrucks_{vv}__base_fleets"] if f"vtrucks_{vv}__base_fleets" in st.session_state else None
    if old is None:
        old = st.session_state[f"vtrucks_{vv}__base_fleets"] = copy.deepcopy(var.get("trucks", []))
    fleets = []
    for i, r in ed.reset_index(drop=True).iterrows():
        if pd.isna(r["payload_t"]) or not str(r["model"] or "").strip():
            continue
        base = dict(old[i]) if i < len(old) else dict(old[0]) if old else {}
        payload = float(r["payload_t"])
        if base.get("payload_t") and abs(float(base["payload_t"]) - payload) > 1e-6:
            scale = payload / float(base["payload_t"])
            for key in ("empty_mass_t", "power_kw", "fuel_lph_loaded", "fuel_lph_empty"):
                if key in base:
                    base[key] = round(float(base[key]) * scale, 1)
        fleets.append({**base, "model": str(r["model"]).strip(), "count": int(r["count"] or 0), "payload_t": payload,
                       "excavators": _parse_list(r["excavators"]), "blend": _parse_blend(r["blend"])})
    var["trucks"] = fleets

    site = var.get("site")
    if site:
        st.markdown("**🛣️ Drogi** – odznacz „w użyciu”, żeby zamknąć drogę; zaznacz przy drodze planowanej, żeby ją otworzyć.")
        ed = _snapshot_editor(
            f"vroads_{vv}",
            lambda: pd.DataFrame([{"name": r.get("name", ""), "kind": r.get("kind", "droga"),
                                   "grade_pct": float(r.get("grade_pct", 0)), "speed_limit_kmh": float(r.get("speed_limit_kmh", 30)),
                                   "active": bool(r.get("active", True))} for r in site["roads"]]),
            hide_index=True, width="stretch", disabled=["name", "kind"],
            column_config={"name": "droga", "kind": "rodzaj", "grade_pct": "nachylenie [%]",
                           "speed_limit_kmh": "limit [km/h]", "active": st.column_config.CheckboxColumn("w użyciu")})
        for r, (_, row) in zip(site["roads"], ed.iterrows()):
            r["grade_pct"], r["speed_limit_kmh"], r["active"] = float(row["grade_pct"]), float(row["speed_limit_kmh"]), bool(row["active"])
        st.markdown("**⛰️ Usypy** – ilość i uziarnienie.")
        ed = _snapshot_editor(
            f"vpiles_{vv}",
            lambda: pd.DataFrame([{"id": p["id"], "material": p.get("material", ""), "tonnage_t": float(p.get("tonnage_t", 0)),
                                   "uziarnienie": preset_of(p["psd"]) if p.get("psd") else "własna"} for p in site["piles"]]),
            hide_index=True, width="stretch", disabled=["id", "material"],
            column_config={"id": "usyp", "material": "materiał", "tonnage_t": "ilość [t]",
                           "uziarnienie": st.column_config.SelectboxColumn(options=list(PRESETS) + ["własna"])})
        for p, (_, row) in zip(site["piles"], ed.iterrows()):
            p["tonnage_t"] = float(row["tonnage_t"])
            if row["uziarnienie"] in PRESETS:
                p["psd"] = PRESETS[row["uziarnienie"]]
    c1, c2, _ = st.columns([1, 1, 2])
    cr = var.setdefault("crusher", {})
    cr["capacity_tph"] = c1.number_input("🏭 Kruszarka – wydajność [t/h]", 50.0, 10000.0, float(cr.get("capacity_tph", 1000)),
                                         50.0, key=f"vcap_{vv}")
    cr["hopper_t"] = c2.number_input("Kosz kruszarki [t]", 10.0, 2000.0, float(cr.get("hopper_t", 150)), 10.0, key=f"vhop_{vv}")

    try:
        built = site_to_scenario(copy.deepcopy(site), copy.deepcopy(var)) if site else copy.deepcopy(var)
        vsc = scenario_from_dict(built)
        vyaml = dump_scenario(vsc)
    except Exception as e:  # noqa: BLE001 – show any config problem to the user
        st.error(f"Wariant nie daje się policzyć: {e}")
        return
    st.session_state.variant_yaml = vyaml
    st.markdown("#### Wariant vs dziś (ta sama zmiana: pogoda, awarie, kierowcy)")
    k = run_policy(vyaml, seed, "natural").kpi
    _tiles(k, run_policy(base_yaml, seed, "natural").kpi)
    st.markdown("**Gdzie tracimy w wariancie:**  \n" + "  \n".join(_losses(k)))
    if _has_map_routes(vsc):
        components.html(compare_page(vyaml, "natural", base_yaml, seed, "Wariant (przydział jak dziś)"),
                        height=760, scrolling=False)


def step_optimize(base_raw: dict, base_yaml: str | None, seed: int):
    st.markdown("### ④ Optymalizacja (uczenie ze wzmocnieniem)")
    st.markdown(
        "Program **uczy się sam**, do której koparki wysyłać każde wozidło po wyładunku, żeby **koszt na tonę był jak "
        "najniższy**. Przy każdej decyzji patrzy na to, co system wie w tej chwili: gdzie jest kolejka, ile jest jazdy, "
        "jak szybko ładuje dana koparka, ile urobku zostało. Ćwiczy na setkach symulowanych zmian i z rundy na rundę "
        "wybiera lepsze reguły. Wynik porównujemy zawsze z **dziś** (stan obecny, przydział jak dziś).")
    if base_yaml is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia**.")
        return
    targets = {"stan obecny": base_yaml}
    if st.session_state.get("variant_yaml") and st.session_state.get("variant_raw") is not None:
        targets["wariant"] = st.session_state.variant_yaml
    which = st.radio("Co optymalizujemy?", list(targets), horizontal=True, key="rl_target")
    yaml_text = targets[which]
    sc = scenario_from_dict(yaml.safe_load(yaml_text))
    if len(sc.excavators) < 2:
        st.warning("Optymalizacja przydziału ma sens, gdy są co najmniej 2 koparki.")
        return
    store = st.session_state.setdefault("rl_by_target", {})
    plain = copy.deepcopy(sc)
    plain.rl_weights, plain.dispatch = None, "natural"
    mine_key = dump_scenario(plain)  # the mine itself, without a learned policy stored in it
    if st.button("🚀 Optymalizuj przydział wozideł", type="primary"):
        bar = st.progress(0.0, "Uczenie…")
        chart = st.empty()
        rows: list[dict] = []

        def cb(it, n, row):
            rows.append(row)
            bar.progress(it / n, f"Runda nauki {it} z {n}: {row['koszt_polityka']:.2f} zł/t "
                                  f"(bez optymalizacji {row['koszt_naturalny']:.2f} zł/t)")
            chart.line_chart(pd.DataFrame(rows).set_index("iteracja")[["koszt_polityka", "koszt_naturalny"]]
                             .rename(columns={"koszt_polityka": "po nauce", "koszt_naturalny": "bez optymalizacji"}))

        res = rl_train(sc, iterations=8, population=10, episodes=2, shift_hours=min(4.0, sc.shift_hours),
                       eval_episodes=3, callback=cb)
        bar.empty()
        chart.empty()
        store[which] = {"result": res, "key": mine_key}
        if which == "stan obecny":
            base_raw["rl_weights"] = res.weights
        st.rerun()
    entry = store.get(which)
    if not entry or entry["key"] != mine_key:
        st.caption("Kliknij przycisk – nauka trwa zwykle 15–40 s.")
        return
    res = entry["result"]
    sc.rl_weights, sc.dispatch = res.weights, "rl"
    rl_yaml = dump_scenario(sc)
    today = run_policy(base_yaml, seed, "natural").kpi
    best = run_policy(rl_yaml, seed, "rl").kpi
    st.markdown(f"#### Wynik: {which} + nauczony przydział vs dziś (ta sama zmiana)")
    _tiles(best, today)
    saving = (today["cost_per_t"] - best["cost_per_t"]) * best["hauled_t"]
    st.success(f"**{best['cost_per_t']:.2f} zł/t** zamiast {today['cost_per_t']:.2f} zł/t i "
               f"**{best['hauled_t'] - today['hauled_t']:+,.0f} t** urobku na zmianę – ok. **{saving:,.0f} zł** "
               "oszczędności na zmianę.".replace(",", " "))
    h = res.history
    fig = go.Figure()
    fig.add_scatter(x=h["iteracja"], y=h["koszt_naturalny"], name="bez optymalizacji", line=dict(color="#9aa0a6", dash="dot"))
    fig.add_scatter(x=h["iteracja"], y=h["koszt_polityka"], name="program w trakcie nauki", line=dict(color="#d62828", width=4))
    fig.update_layout(title="Jak program się uczył – koszt zł/t w kolejnych rundach (0 = przed nauką)", height=300,
                      xaxis_title="runda nauki", yaxis_title="zł/t", margin=dict(t=40), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Zobacz, jak jeżdżą po optymalizacji")
    components.html(compare_page(rl_yaml, "rl", base_yaml, seed, f"{which} + nauczony przydział"), height=760, scrolling=False)


def render_simple(raw: dict, set_raw, render_map_tab) -> None:
    st.markdown("Cztery kroki: **① opisz kopalnię (stan obecny) → ② zobacz, jak jest dziś → ③ sprawdź wariant "
                "„co jeśli” → ④ zoptymalizuj.** Szczegółowe ustawienia i analizy są w **Trybie zaawansowanym** "
                "(przełącznik w panelu bocznym).")
    tabs = st.tabs(["① Kopalnia – stan obecny", "② Jak jest dziś", "③ Wariant – co jeśli", "④ Optymalizacja"])
    with tabs[0]:
        render_map_tab(raw, set_raw, simple=True)
    seed = int(raw.get("seed", 42))
    try:
        sc = scenario_from_dict(copy.deepcopy(raw))
        yaml_text = dump_scenario(sc)
        if not _has_map_routes(sc):
            raise ValueError("brak tras z mapy")
    except Exception as e:  # noqa: BLE001
        sc, yaml_text = None, None
        msg = str(e)
    with tabs[1]:
        if sc is None:
            st.info(f"Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia** ({msg}) – albo wczytaj przykład.")
            _demo_button(set_raw, "demo_today")
        else:
            step_today(sc, yaml_text, seed, set_raw)
    with tabs[2]:
        step_variant(raw, yaml_text, seed)
    with tabs[3]:
        step_optimize(raw, yaml_text, seed)
