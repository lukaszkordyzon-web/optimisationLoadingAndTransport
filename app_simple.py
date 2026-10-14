"""Simple, guided mode of the panel: 1) the mine, 2) how it works today, 3) optimisation."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from kruszyk.anim import animation_html
from kruszyk.config import Advisor, ExcavatorPolicy, dump_scenario, scenario_from_dict
from kruszyk.psd import PRESETS, preset_of
from kruszyk.rl import POLICIES
from kruszyk.rl import explain_advisor, explain_excavators
from kruszyk.rl import train as rl_train
from kruszyk.sim import simulate
from kruszyk.site import normalize_roads, site_to_scenario

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


def _crushers_summary(k: dict) -> None:
    """Production per crusher vs its plan (shown when there are several crushers or a plan / product is set)."""
    cs = k.get("crushers") or []
    if len(cs) < 2 and not any(c.get("target_t") or c.get("product") for c in cs):
        return
    rows = [{"kruszarka": c["crusher"], "produkt": c.get("product") or "–",
             "przyjmuje z": ", ".join(c.get("accepts") or []) or "wszystkich",
             "skruszone [t]": round(c["crushed_t"]), "plan [t]": round(c["target_t"]) if c.get("target_t") else None,
             "wykonanie planu [%]": c.get("plan_pct"), "energia [kWh/t]": c.get("kwh_per_t"),
             "wykorzystanie [%]": round(100 * c["utilization"]),
             "bez nadawy [h]": c["starved_h"]} for c in cs]
    st.markdown("**🏭 Kruszarki – produkcja**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


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
    if k.get("truck_blasting_h", 0) > 0.05:
        out.append(f"🧨 Strzelanie: wozidła stały **{k['truck_blasting_h']:.1f} h**, koparki "
                   f"**{k['excavator_down_h'].get('strzelanie', 0):.1f} h**.")
    if k.get("truck_break_h", 0) > 0.05:
        out.append(f"⏸ Przerwy: wozidła **{k['truck_break_h']:.1f} h** łącznie.")
    if k.get("excavator_down_h", {}).get("tankowanie", 0) > 0.05:
        out.append(f"⛽ Tankowanie koparek z cysterny: **{k['excavator_down_h']['tankowanie']:.1f} h** postoju.")
    if k.get("relocations", 0):
        out.append(f"⤳ Przestawianie koparek wzdłuż frontu: **{k['relocations']}** razy, "
                   f"**{60 * k.get('excavator_down_h', {}).get('przestawienie', 0):.0f} min** postoju koparek.")
    if k.get("truck_refuels", 0):
        out.append(f"⛽ Wozidła zjeżdżały tankować **{k['truck_refuels']}** razy.")
    if k.get("yields", 0):
        out.append(f"🚙 Ustępowanie pierwszeństwa autom (ekipa, sztygar, cysterna): **{k['yields']}** razy, "
                   f"{60 * k['yield_h']:.0f} min.")
    if k.get("bay_stops", 0):
        out.append(f"⇆ Mijanki: wozidła czekały na mijankach **{k['bay_stops']}** razy, łącznie "
                   f"**{60 * k.get('bay_wait_h', 0):.0f} min** (jeden pas; pierwszeństwo ma jadący pod górę).")
    if k.get("spot_exit_h", 0) > 0.2:
        out.append(f"↩️ Manewry pod koparkami (podstawienie i wyjazd): **{k['spot_exit_h']:.1f} h** czasu wozideł.")
    if k.get("feed_target"):
        mix = ", ".join(f"{b} {k['feed_share'].get(b, 0):.0f} % (cel {t:.0f} %)" for b, t in k["feed_target"].items())
        out.append(f"🧪 Skład nadawy: {mix} – odchyłka **{k['feed_dev_pp']:.1f} pp**.")
    if k.get("parked_excavator_h", 0) > 0.05:
        out.append(f"🅿️ Koparki odstawione (za mało wozideł): **{k['parked_excavator_h']:.1f} h**.")
    if k.get("parked_truck_h", 0) > 0.05:
        out.append(f"🅿️ Wozidła zaparkowane zamiast stać w kolejce do kosza: **{k['parked_truck_h']:.1f} h** "
                   "(silnik wyłączony – bez paliwa i zużycia).")
    if k.get("over_limit_pct", 0) > 0.5:
        out.append(f"🚦 **{k['over_limit_pct']:.0f} %** przejazdów odcinków powyżej limitu prędkości "
                   f"({k['speeders']} kierowców jeździ szybciej).")
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
    st.markdown("### ⑤ Jak było dziś")
    st.markdown("Odtworzenie **jednej zmiany tak, jak pracujecie obecnie**: kierowca po wyładunku wraca do „swojej” "
                "koparki, czasem jedzie gdzie indziej, nikt nie wie, gdzie jest kolejka. Do tego losowość: deszcz, "
                "różni kierowcy, awarie, różne czasy załadunku.")
    st.caption("Źródło danych: model syntetyczny. Po podłączeniu telemetrii (GPS) ten widok odtwarza rzeczywistą "
               "zmianę – te same wskaźniki.")
    if not _has_map_routes(sc):
        st.info("Najpierw zbuduj kopalnię w kroku **① Kopalnia** (i kliknij „Zapisz kopalnię”) – albo wczytaj przykład.")
        _demo_button(set_raw, "demo_today")
        return
    from app_calibration import status_line

    status_line(sc)
    k = run_policy(yaml_text, seed, "natural").kpi
    _tiles(k)
    _goal_line(k, sc)
    _crushers_summary(k)
    st.markdown("**Gdzie tracimy:**  \n" + "  \n".join(_losses(k)))
    st.caption("▶ Kliknij **Start** na panelu obok mapy – zmiana leci przyspieszona, kropki to wozidła "
               "(kolor = co robią), liczniki pokazują koszty na bieżąco.")
    components.html(animation_page(yaml_text, seed, "natural", False), height=760, scrolling=False)


def _clock(sc, h: float) -> str:
    hh, mm = (int(x) for x in sc.shift_plan.start_time.split(":"))
    m = hh * 60 + mm + int(round(h * 60))
    return f"{m // 60 % 24:02d}:{m % 60:02d}"


def _goal_line(k: dict, sc) -> None:
    """Did the shift reach its goal (rules of the shift)?"""
    g = sc.shift_goal or {}
    if not k.get("goal_t"):
        return
    what = (f"{g['t']:,.0f} t „{g['item']}” (≈ {k['goal_t']:,.0f} t nadawy)" if g.get("kind") == "product"
            else f"{k['goal_t']:,.0f} t przez zakład").replace(",", " ")
    if k.get("goal_reached_h") is not None:
        st.success(f"🎯 Cel zmiany {what}: **osiągnięty o {_clock(sc, k['goal_reached_h'])}**"
                   + (" – potem wozidła parkują." if g.get("stop", True) else "."))
    else:
        st.error(f"🎯 Cel zmiany {what}: **niezrealizowany – {k['goal_pct']:.0f} %** "
                 f"({k['hauled_t']:,.0f} t).".replace(",", " "))
    for m in k.get("excavator_moves") or []:
        st.caption(f"⛏️ {m['excavator']} przejechał z {m['from']} na {m['to']} o {_clock(sc, m['start_h'])} "
                   f"({60 * (m['end_h'] - m['start_h']):.0f} min).")


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
    st.markdown("### ② Wariant – co jeśli")
    st.markdown("Kopia stanu obecnego, w której **zmieniasz** np. liczbę wozideł, przypisanie do koparek, drogi "
                "(włączasz drogę planowaną, zamykasz istniejącą, zmieniasz nachylenie), wydajność kruszarki albo "
                "uziarnienie usypów – i porównujesz go ze stanem obecnym **na wielu tych samych zmianach** (wynik "
                "jednej zmiany bywa przypadkiem). Poniżej też: **ile wozideł** opłaca się w tej kopalni. Stan obecny "
                "się nie zmienia. **Efekt na cały okres** policzysz w kroku **③ Harmonogram** (wybierz tam „wariant”).")
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

    st.markdown("**🚚 Wozidła** – każde osobno (nr); zmień ładowność, wyłącz z pracy (liczba 0), dodaj wiersz albo "
                "zmień przypisania (`K-1, K-2`; dwa usypy w kursie: `K-1:0.5, K-2:0.5`).")
    ed = _snapshot_editor(
        f"vtrucks_{vv}",
        lambda: pd.DataFrame([{"id": f.get("id", ""), "model": f.get("model", ""), "count": int(f.get("count", 0)),
                               "payload_t": float(f.get("payload_t", 50)),
                               "excavators": ", ".join(f.get("excavators") or []),
                               "blend": ", ".join(f"{a}:{b:g}" for a, b in (f.get("blend") or {}).items())}
                              for f in var.get("trucks", [])]),
        num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"id": "nr", "model": "typ wozidła", "count": st.column_config.NumberColumn("liczba", min_value=0, step=1),
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
        base["id"] = "" if pd.isna(r.get("id")) else str(r.get("id") or "").strip()
        fleets.append({**base, "model": str(r["model"]).strip(), "count": int(r["count"] or 0), "payload_t": payload,
                       "excavators": _parse_list(r["excavators"]), "blend": _parse_blend(r["blend"])})
    var["trucks"] = fleets

    site = var.get("site")
    if site:
        normalize_roads(site)
        st.markdown("**🛣️ Drogi** – odznacz „w użyciu”, żeby zamknąć drogę; zaznacz przy drodze planowanej, żeby ją otworzyć.")
        ed = _snapshot_editor(
            f"vroads_{vv}",
            lambda: pd.DataFrame([{"name": r.get("name", ""), "kind": r.get("kind", "droga"),
                                   "grade_draw_pct": float(r.get("grade_draw_pct", 0)),
                                   "speed_limit_kmh": float(r.get("speed_limit_kmh", 30)),
                                   "single_lane": bool(r.get("single_lane")),
                                   "active": bool(r.get("active", True))} for r in site["roads"]]),
            hide_index=True, width="stretch", disabled=["name", "kind"],
            column_config={"name": "droga", "kind": "rodzaj", "grade_draw_pct": "nachylenie wzdłuż rysowania [%]",
                           "speed_limit_kmh": "limit [km/h]",
                           "single_lane": st.column_config.CheckboxColumn(
                               "jeden pas", help="ruch wahadłowy – mijanie tylko na mijankach z mapy"),
                           "active": st.column_config.CheckboxColumn("w użyciu")})
        for r, (_, row) in zip(site["roads"], ed.iterrows()):
            r["grade_draw_pct"], r["speed_limit_kmh"], r["active"] = (float(row["grade_draw_pct"]), float(row["speed_limit_kmh"]),
                                                                      bool(row["active"]))
            r["single_lane"] = bool(row["single_lane"])
        if site.get("excavators"):
            from kruszyk.config import SPOT_MODES

            st.markdown("**🚚 Podstawianie pod koparki** – procent wozideł danym sposobem (np. co da ładowanie dwustronne).")
            modes = list(SPOT_MODES)
            ed = _snapshot_editor(
                f"vspot_{vv}",
                lambda: pd.DataFrame([{"koparka": e["id"], **{m: (e.get("spotting") or {}).get(m) for m in modes}}
                                      for e in site["excavators"]]),
                hide_index=True, width="stretch", disabled=["koparka"],
                column_config={m: st.column_config.NumberColumn(f"{SPOT_MODES[m]['short']} [%]", help=SPOT_MODES[m]["label"], min_value=0.0,
                                                                max_value=100.0, step=5.0) for m in modes})
            for e, (_, row) in zip(site["excavators"], ed.iterrows()):
                mix = {m: float(row[m]) for m in modes if not pd.isna(row[m]) and float(row[m]) > 0}
                if mix:
                    e["spotting"] = mix
                else:
                    e.pop("spotting", None)
        st.markdown("**⛰️ Usypy** – ilość i uziarnienie.")
        ed = _snapshot_editor(
            f"vpiles_{vv}",
            lambda: pd.DataFrame([{"id": p["id"], "material": p.get("material", ""), "tonnage_t": float(p.get("tonnage_t", 0)),
                                   "target_share": p.get("target_share"),
                                   "uziarnienie": preset_of(p["psd"]) if p.get("psd") else "własna"} for p in site["piles"]]),
            hide_index=True, width="stretch", disabled=["id", "material"],
            column_config={"id": "usyp", "material": "materiał", "tonnage_t": "ilość [t]",
                           "target_share": st.column_config.NumberColumn("udział w nadawie [%]", min_value=0.0, max_value=100.0),
                           "uziarnienie": st.column_config.SelectboxColumn(options=list(PRESETS) + ["własna"])})
        for p, (_, row) in zip(site["piles"], ed.iterrows()):
            p["tonnage_t"] = float(row["tonnage_t"])
            p["target_share"] = None if pd.isna(row["target_share"]) or not row["target_share"] else float(row["target_share"])
            if row["uziarnienie"] in PRESETS:
                p["psd"] = PRESETS[row["uziarnienie"]]
    with st.expander("🕐 Zmiana i zdarzenia w wariancie (np. strzelanie o innej godzinie, inny objazd, cysterna)"):
        from app_events import events_editor

        events_editor(var, f"v{vv}", [e["id"] for e in (site or {}).get("excavators", [])] or
                      [e["id"] for e in var.get("excavators", [])], site)
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
    # today rebuilt from the map exactly like the variant: an unchanged copy must compare as identical
    try:
        cmp_base = dump_scenario(scenario_from_dict(
            site_to_scenario(copy.deepcopy(base_raw["site"]), copy.deepcopy(base_raw)) if base_raw.get("site")
            else copy.deepcopy(base_raw)))
    except Exception:  # noqa: BLE001 – the map can't be rebuilt: compare with the stored routes
        cmp_base = base_yaml
    _variant_compare(cmp_base, vyaml, seed)
    _variant_show(cmp_base, vyaml, vsc, seed)
    _fleet_section(cmp_base, vyaml, seed)


def _variant_show(base_yaml: str, vyaml: str, vsc, seed: int) -> None:
    """The comparison as a film: one of the compared shifts on the map, today and the variant side by side."""
    from ui import card

    with card("var_show"):
        st.markdown("#### 🎬 Zobacz różnicę na mapie")
        st.caption("Animacja pokazuje, **dlaczego** wariant wychodzi lepiej albo gorzej: gdzie wozidła stoją w kolejce, "
                   "gdzie czekają na mijankach, kiedy kruszarka czeka na urobek – obok liczniki ton i kosztów na żywo. "
                   "Do pokazania kierownictwu, sztygarom, kierowcom. O tym, czy wariant jest lepszy, decyduje porównanie "
                   "na wielu zmianach powyżej.")
        got = st.session_state.get("vcmp")
        options = {"zmiana przykładowa": seed}
        if got:
            ps = got[1]["per_shift"].copy()
            ps["d"] = ps["wariant_zl_t"] - ps["dzis_zl_t"]
            typical = ps.iloc[(ps["d"] - ps["d"].median()).abs().argsort().iloc[0]]
            options = {f"typowa (zmiana {int(typical['zmiana'])} z porównania)": int(typical["seed"]),
                       f"największa przewaga wariantu (zmiana {int(ps.loc[ps['d'].idxmin(), 'zmiana'])})":
                           int(ps.loc[ps["d"].idxmin(), "seed"]),
                       f"najgorsza dla wariantu (zmiana {int(ps.loc[ps['d'].idxmax(), 'zmiana'])})":
                           int(ps.loc[ps["d"].idxmax(), "seed"])}
        which = st.selectbox("Którą zmianę pokazać", list(options), key="vshow_which")
        s_ = options[which]
        k = run_policy(vyaml, s_, "natural").kpi
        _tiles(k, run_policy(base_yaml, s_, "natural").kpi)
        with st.expander("Gdzie tracimy w wariancie na tej zmianie"):
            _crushers_summary(k)
            st.markdown("  \n".join(_losses(k)))
        if _has_map_routes(vsc):
            components.html(compare_page(vyaml, "natural", base_yaml, s_, "Wariant (przydział jak dziś)"),
                            height=760, scrolling=False)


def _pct(x: float) -> str | None:
    return None if abs(x) < 0.05 else f"{x:+.1f} %".replace(".", ",")  # no arrow for "no change"


def _variant_compare(base_yaml: str, vyaml: str, seed: int) -> None:
    """Variant vs today on many shifts (same weather, breakdowns, drivers) – with a verdict that is not chance."""
    from kruszyk.compare import N_SHIFTS, compare, noise_note
    from ui import card

    with card("var_cmp"):
        st.markdown("#### ⚖️ Wariant vs dziś – na wielu zmianach")
        st.caption("Oba scenariusze liczone na tych samych zmianach: ta sama pogoda, te same awarie i kierowcy – "
                   "różnica wynika ze zmiany, nie z losu. Przydział wozideł jak dziś (sztygar).")
        c = st.columns([1, 1, 2])
        n = c[0].selectbox("Ile zmian porównać", N_SHIFTS, index=1, key="vcmp_n")
        key = hash((base_yaml, vyaml, n, seed))
        if c[1].button("▶️ Porównaj", type="primary", key="vcmp_go"):
            bar = st.progress(0.0, "liczę…")
            res = compare(scenario_from_dict(yaml.safe_load(base_yaml)), scenario_from_dict(yaml.safe_load(vyaml)),
                          n=n, seed=seed, progress=lambda i, k_: bar.progress(i / k_, f"zmiana {i} z {k_}"))
            bar.empty()
            st.session_state.vcmp = (key, res)
        got = st.session_state.get("vcmp")
        if not got:
            c[2].caption("Kliknij **Porównaj** – kilka sekund.")
            return
        if got[0] != key:
            st.warning("Wariant zmienił się od ostatniego porównania – kliknij **Porównaj** ponownie.")
        r = got[1]
        a, b = r["base"], r["variant"]
        m = st.columns(4)
        d_t, d_cost = _pct(r["d_t_pct"]), _pct(r["d_cost_pct"])
        m[0].metric("Tony na zmianę (średnio)", f"{b['t']:,.0f} t".replace(",", " "), d_t and d_t + " vs dziś")
        m[1].metric("Koszt na tonę", f"{b['zl_t']:.2f} zł/t".replace(".", ","), d_cost and d_cost + " vs dziś",
                    delta_color="inverse")
        m[2].metric("Paliwo na tonę", f"{b['paliwo_l_t']:.3f} l/t".replace(".", ","),
                    _pct(100 * (b["paliwo_l_t"] / a["paliwo_l_t"] - 1)) if a["paliwo_l_t"] else None, delta_color="inverse")
        m[3].metric("Kolejka pod kruszarką", f"{b['kolejka_kruszarka_h']:.1f} h".replace(".", ","),
                    f"{b['kolejka_kruszarka_h'] - a['kolejka_kruszarka_h']:+.1f} h".replace(".", ",")
                    if abs(b["kolejka_kruszarka_h"] - a["kolejka_kruszarka_h"]) >= 0.05 else None,
                    delta_color="inverse")
        lines = []
        for what, verdict, win, lose, unit in (("koszt na tonę", r["cost_verdict"], r["cost_better"], r["cost_worse"],
                                                "taniej"),
                                               ("tony", r["t_verdict"], r["t_more"], r["t_less"], "więcej ton")):
            if verdict == "lepiej":
                lines.append(f"✅ **{what}: wariant lepszy** – {unit} na **{win} z {r['n']}** zmian.")
            elif verdict == "gorzej":
                lines.append(f"❌ **{what}: wariant gorszy** – gorzej na **{lose} z {r['n']}** zmian.")
            else:
                lines.append(f"➖ **{what}: bez pewnej różnicy** – lepiej na {win}, gorzej na {lose} z {r['n']} zmian "
                             "(to może być przypadek).")
        st.markdown("  \n".join(lines))
        st.caption(f"Typowa zmiana (P10–P90): dziś {a['t_p10']:,.0f}–{a['t_p90']:,.0f} t, wariant "
                   f"{b['t_p10']:,.0f}–{b['t_p90']:,.0f} t. ".replace(",", " ") + noise_note(r["n"]))
        ps = r["per_shift"]
        fig = go.Figure()
        fig.add_scatter(x=ps["zmiana"], y=ps["dzis_t"], name="dziś", mode="lines+markers", line=dict(color="#9aa0a6"))
        fig.add_scatter(x=ps["zmiana"], y=ps["wariant_t"], name="wariant", mode="lines+markers",
                        line=dict(color="#c8102e", width=3))
        fig.update_layout(height=260, margin=dict(t=30, b=10), title="Tony na każdej z porównanych zmian",
                          xaxis_title="zmiana", yaxis_title="t", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")


def _fleet_section(base_yaml: str, vyaml: str, seed: int) -> None:
    """How many trucks: the same mine with fewer / more trucks on the same shifts."""
    from kruszyk.compare import N_SHIFTS, fleet_sweep
    from ui import card

    with card("var_fleet"):
        st.markdown("#### 🚚 Ile wozideł? – ta sama kopalnia z mniejszą i większą flotą")
        which = st.radio("Dla", ["stanu obecnego", "wariantu"], horizontal=True, key="vfleet_which")
        sc = scenario_from_dict(yaml.safe_load(base_yaml if which == "stanu obecnego" else vyaml))
        now = sc.truck_count
        c = st.columns([2, 1, 1])
        lo, hi = c[0].slider("Liczba wozideł", 1, max(12, now + 6), (max(1, now - 2), now + 3), key="vfleet_range")
        n = c[1].selectbox("Zmian na wariant", N_SHIFTS, index=0, key="vfleet_n")
        key = hash((base_yaml if which == "stanu obecnego" else vyaml, lo, hi, n, seed))
        if c[2].button("▶️ Policz", key="vfleet_go"):
            counts = list(range(lo, hi + 1))
            bar = st.progress(0.0, "liczę…")
            tab, rec = fleet_sweep(sc, counts, n=n, seed=seed,
                                   progress=lambda i, k_: bar.progress(i / k_, f"zmiana {i} z {k_}"))
            bar.empty()
            st.session_state.vfleet = (key, tab, rec, now)
        got = st.session_state.get("vfleet")
        if not got:
            st.caption(f"Teraz w pracy: **{now}** wozideł. Kliknij **Policz** – każda flota liczona na tych samych zmianach.")
            return
        if got[0] != key:
            st.warning("Ustawienia zmieniły się – kliknij **Policz** ponownie.")
        _, tab, rec, now = got
        show = tab.rename(columns={"wozidla": "wozidła", "t": "tony / zmianę", "t_p10": "P10 [t]", "t_p90": "P90 [t]",
                                   "zl_t": "zł/t", "paliwo_l_t": "paliwo l/t", "kolejka_kruszarka_h": "kolejka pod koszem [h]",
                                   "kolejka_koparki_h": "kolejka pod koparkami [h]", "kruszarka_pct": "kruszarka [%]"})
        st.dataframe(show.round(2), hide_index=True, width="stretch")
        fig = go.Figure()
        fig.add_bar(x=tab["wozidla"], y=tab["t"], name="tony na zmianę", marker_color="#9fb3c8")
        fig.add_scatter(x=tab["wozidla"], y=tab["zl_t"], name="zł/t", yaxis="y2", mode="lines+markers",
                        line=dict(color="#c8102e", width=3))
        fig.update_layout(height=280, margin=dict(t=30, b=10), xaxis_title="wozidła", yaxis_title="t / zmianę",
                          yaxis2=dict(title="zł/t", overlaying="y", side="right"), legend=dict(orientation="h"),
                          title="Tony i koszt tony w zależności od liczby wozideł")
        st.plotly_chart(fig, width="stretch")
        msg = (f"💰 Najtańsza tona: **{rec['najtansza']} wozideł** ({rec['najtansza_zl_t']:.2f} zł/t, "
               f"{rec['najtansza_t']:,.0f} t/zmianę). 🏭 Prawie pełna produkcja (97 % maksimum): od "
               f"**{rec['pelna_produkcja']} wozideł** ({rec['pelna_t']:,.0f} t).").replace(",", " ")
        st.success(msg)
        if now > rec["pelna_produkcja"]:
            st.info(f"Dziś pracuje {now} wozideł – {now - rec['pelna_produkcja']} więcej, niż potrzeba do pełnej "
                    "produkcji: dodatkowe stoją w kolejkach (rezerwa na awarie kosztuje – patrz zł/t).")
        elif now < rec["pelna_produkcja"]:
            st.info(f"Dziś pracuje {now} wozideł – dołożenie do {rec['pelna_produkcja']} podnosi produkcję.")


def step_optimize(base_raw: dict, base_yaml: str | None, seed: int):
    st.markdown("### ⑥ Optymalizacja RL – decyzje na żywo")
    st.markdown(
        "Program **uczy się sam** decyzji na zmianie, żeby **koszt na tonę był jak najniższy**. Ćwiczy na setkach "
        "symulowanych zmian i z rundy na rundę wybiera lepsze reguły. Wynik porównujemy zawsze z **dziś**. Po wdrożeniu "
        "te decyzje trafiają na żywo na telefony kierowców i operatorów koparek; nauczona polityka zapisuje się w "
        "kopalni – w **Planowaniu produkcji** widać, ile daje na cały miesiąc.")
    st.markdown("| Decyzja | Kiedy |\n|---|---|\n"
                "| 🚚 **dokąd jedzie wozidło** po wyładunku | zawsze – patrzy na kolejki, jazdę, tempo koparek, urobek |\n"
                "| 🅿️ **wozidło parkuje** z wyłączonym silnikiem | gdy po kursie i tak stałoby w kolejce do kosza |\n"
                "| ⛏️ **koparka zostaje odstawiona** | gdy wozideł jest za mało na wszystkie koparki |\n"
                "| 🪣 **bez ostatniej łyżki** | gdy wozidło prawie pełne, a w kolejce czekają inne |\n"
                "| 📱 **zalecana prędkość** | żeby dojechać, gdy koparka jest wolna, a nie stać w kolejce |")
    if base_yaml is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia**.")
        return
    targets = {"stan obecny": base_yaml}
    which = "stan obecny"
    advise = st.toggle("📱 Zalecenia w czasie rzeczywistym", True, key="rl_advisor",
                       help="Każde wozidło dostaje zalecaną prędkość, żeby dojechać pod koparkę dokładnie, gdy będzie "
                            "wolna (nie wcześniej – kolejka, nie później – koparka czeka). Operator koparki dostaje "
                            "komunikat, kto i za ile podjedzie oraz co robić: nabrać łyżkę i czekać, przegarnąć front…")
    yaml_text = targets[which]
    sc = scenario_from_dict(yaml.safe_load(yaml_text))
    sc.advisor.enabled = advise
    if len(sc.excavators) < 2:
        st.warning("Optymalizacja przydziału ma sens, gdy są co najmniej 2 koparki.")
        return
    store = st.session_state.setdefault("rl_by_target", {})
    plain = copy.deepcopy(sc)
    plain.rl_weights, plain.dispatch = None, "natural"
    plain.advisor = Advisor(enabled=advise)  # learned advice settings stored in the mine don't make it "another mine"
    plain.excavator_policy = ExcavatorPolicy()
    mine_key = dump_scenario(plain)  # the mine itself, without a learned policy stored in it
    if st.button("🚀 Optymalizuj decyzje zmiany (nauka RL)", type="primary"):
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
                       eval_episodes=3, callback=cb, advisor=advise, learn_advisor=True, learn_excavators=True)
        bar.empty()
        chart.empty()
        # the policy is used only if it really beats today's practice on full shifts (not just in training)
        trial = _with_policy(sc, res)
        seeds = [seed + 101 + i for i in range(4)]
        from kruszyk.optimize import kpi_row
        from kruszyk.rl import aggregate

        before = aggregate([kpi_row(run_policy(base_yaml, s_, "natural").kpi) for s_ in seeds])["objective"]
        after = aggregate([kpi_row(run_policy(dump_scenario(trial), s_, "rl").kpi) for s_ in seeds])["objective"]
        no_gain = after > before * (1 - RL_MIN_GAIN)
        store[which] = {"result": res, "key": mine_key, "no_gain": no_gain, "test": (before, after)}
        if which == "stan obecny" and no_gain:
            base_raw.pop("rl_weights", None)  # keep today's practice – nothing to gain here
        elif which == "stan obecny":
            base_raw["rl_weights"] = res.weights
            if res.advisor:
                base_raw.setdefault("advisor", {}).update(res.advisor)
            if res.excavators:
                base_raw.setdefault("excavator_policy", {}).update(res.excavators)
        st.rerun()
    entry = store.get(which)
    if not entry or entry["key"] != mine_key:
        st.caption("Kliknij przycisk – nauka trwa zwykle 15–40 s.")
        return
    if advise:
        st.info("**Komunikaty na terminalach operatorów:**  \n"
                "🚚 **Kierowca:** „Jedź do K-2, zalecana prędkość 22 km/h – będziesz, gdy koparka skończy WZ-04” "
                "(zamiast pędzić i stać w kolejce: mniej paliwa, mniej zużycia).  \n"
                "⛏️ **Operator:** „WZ-03 za 1:40 → nabierz łyżkę i czekaj” – pierwsza łyżka gotowa w chwili "
                "podstawienia skraca załadunek o jeden cykl; „brak wozideł przez 5 min → przegarnij front”.  \n"
                "Te komunikaty widać na żywo w panelu animacji poniżej (📱 Ekrany operatorów koparek).")
    res = entry["result"]
    sc = _with_policy(sc, res)
    rl_yaml = dump_scenario(sc)
    _recommendations(entry, base_yaml, sc, seed)
    today = run_policy(base_yaml, seed, "natural").kpi
    best = run_policy(rl_yaml, seed, "rl").kpi
    if entry.get("no_gain"):
        before, after = entry["test"]
        st.warning(f"**Nauczony przydział nie obniża kosztu tej kopalni** ({after:.2f} zł/t wobec {before:.2f} zł/t dziś, "
                   "średnio z 4 pełnych zmian) – zostaje przydział jak dziś, a polityka RL **nie jest zapisywana**.  \n"
                   + "  \n".join(bottleneck(today)))
    st.markdown(f"#### Wynik: {which} + nauczony przydział vs dziś (ta sama zmiana)")
    _tiles(best, today)
    _goal_line(best, sc)
    saving = (today["cost_per_t"] - best["cost_per_t"]) * best["hauled_t"]
    if not entry.get("no_gain"):
        st.success(f"**{best['cost_per_t']:.2f} zł/t** zamiast {today['cost_per_t']:.2f} zł/t i "
                   f"**{best['hauled_t'] - today['hauled_t']:+,.0f} t** urobku na zmianę – ok. **{saving:,.0f} zł** "
                   "oszczędności na zmianę.".replace(",", " "))
    _decisions(best, today, sc)
    lessons = (explain_excavators(res.excavators) if res.excavators else []) + \
        (explain_advisor(res.advisor) if res.advisor else [])
    if lessons:
        st.markdown("#### Czego agent się nauczył")
        st.markdown("  \n".join(lessons))
    if best.get("feed_target"):
        mix = ", ".join(f"{b} {best['feed_share'].get(b, 0):.0f} % (cel {t:.0f} %)" for b, t in best["feed_target"].items())
        st.markdown(f"🧪 **Skład nadawy po optymalizacji:** {mix} – odchyłka **{best['feed_dev_pp']:.1f} pp** "
                    f"(dziś {today.get('feed_dev_pp', 0):.1f} pp). Wozidła przypisane na sztywno jeżdżą po swojemu – "
                    "skład koryguje się wozidłami wolnymi.")
    comp = res.comparison.set_index("polityka")
    if len(comp) >= 5:
        only, full = comp.iloc[-2], comp.iloc[-1]
        diff = 100 * (full["cost_per_t"] / only["cost_per_t"] - 1)
        st.caption(f"Test na nowych zmianach: sam przydział {only['cost_per_t']:.2f} zł/t, przydział + zarządzanie "
                   f"koparkami{' + zalecenia' if res.advisor else ''} {full['cost_per_t']:.2f} zł/t ({diff:+.1f} %).")
    h = res.history
    fig = go.Figure()
    fig.add_scatter(x=h["iteracja"], y=h["koszt_naturalny"], name="bez optymalizacji", line=dict(color="#9aa0a6", dash="dot"))
    fig.add_scatter(x=h["iteracja"], y=h["koszt_polityka"], name="program w trakcie nauki", line=dict(color="#d62828", width=4))
    fig.update_layout(title="Jak program się uczył – koszt zł/t w kolejnych rundach (0 = przed nauką)", height=300,
                      xaxis_title="runda nauki", yaxis_title="zł/t", margin=dict(t=40), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Zobacz, jak jeżdżą po optymalizacji")
    components.html(compare_page(rl_yaml, "rl", base_yaml, seed, f"{which} + nauczony przydział"), height=760, scrolling=False)


def _rec_numbers(x: dict) -> str:
    return (f"koszt tony **{x['d_cost_pct']:+.1f} %** ({x['zl_t']:.2f} zamiast {x['today_zl_t']:.2f} zł/t), "
            f"tony {x['d_t_pct']:+.1f} %, taniej na {x['wins']} z {x['n']} zmian").replace(".", ",")


def _recommendations(entry: dict, base_yaml: str, sc_policy, seed: int) -> None:
    """What the mine can change tomorrow without a system, and what only the live system gives – every
    organisational recommendation checked on the same shifts as today's practice."""
    from kruszyk.advice import recommend
    from ui import card

    if "advice" not in entry:
        bar = st.progress(0.0, "Sprawdzam rekomendacje na 10 zmianach…")
        entry["advice"] = recommend(scenario_from_dict(yaml.safe_load(base_yaml)), sc_policy, seed=seed,
                                    progress=lambda f, t: bar.progress(min(f, 1.0), t))
        bar.empty()
    a = entry["advice"]
    with card("rl_recs"):
        st.markdown("#### 📋 Rekomendacje dla kopalni")
        st.markdown("**✅ Do wprowadzenia od jutra – bez żadnego systemu**")
        if a["organisational"]:
            for r in a["organisational"]:
                small = " _(mały efekt)_" if abs(r["d_cost_pct"]) < 1 else ""
                st.markdown(f"**{r['title']}**{small}  \n{r['why']}  \n{_rec_numbers(r)}.")
                if r.get("changes"):
                    st.caption("Zmiany przydziału: " + "; ".join(
                        f"{t}: {', '.join(old) or 'bez przydziału'} → {new}" for t, old, new in r["changes"]))
        else:
            st.caption("Nie ma zmian organizacyjnych, które bez systemu dają pewną oszczędność – to, co robi polityka, "
                       "wymaga informacji na żywo.")
        for t in a.get("tradeoffs", []):
            st.markdown(f"⚖️ **Do decyzji: {t['title']}** – taniej, ale mniej produkcji: {_rec_numbers(t)}. Opłaca "
                        "się tylko, jeśli tych ton i tak nie sprzedacie.")
        st.markdown("**📱 Wymaga systemu na żywo** (telefony w wozidłach i koparkach)")
        st.markdown("  \n".join(f"• {x['title']} – {x['detail']}" for x in a["system"]))
        f = a["full"]
        msg = f"Cały system na żywo vs dziś: {_rec_numbers(f)}."
        if f["d_t_pct"] < -1.5:
            st.warning(msg + " **Mniej ton**: polityka oszczędza na koszcie tony kosztem produkcji. Jeśli liczy się "
                             "produkcja, ustaw **cel zmiany** w kroku ② Plan zmiany i naucz politykę ponownie.")
        elif f["verdict"] == "lepiej":
            st.success(msg)
        else:
            st.info(msg + " Różnica niepewna – system na żywo niewiele tu daje.")
        st.caption(f"Każda rekomendacja sprawdzona w symulacji na tych samych {a['n']} zmianach (pogoda, awarie, "
                   "kierowcy) co praca jak dziś. Wyniki modelu – przed decyzją inwestycyjną sprawdź je w kroku "
                   "Planowanie produkcji → ② Wariant (20–30 zmian).")


def _decisions(k: dict, today: dict, sc) -> None:
    """What the learned policy did on this shift that today's practice doesn't do."""
    c = sc.costs
    st.markdown("#### Co polityka zrobiła na tej zmianie")
    cols = st.columns(4)
    park = k.get("parked_truck_h", 0.0)
    cols[0].metric("🅿️ Wozidła na parkingu", f"{park:.1f} h",
                   f"−{park * c.truck_per_h * c.truck_variable_share:,.0f} zł zużycia".replace(",", " ") if park else None,
                   delta_color="off", help="silnik wyłączony zamiast stania w kolejce do kosza: bez paliwa i bez "
                                           f"zużycia ({100 * c.truck_variable_share:.0f} % stawki wozidła)")
    cols[1].metric("⏳ Kolejka pod kruszarką", f"{k.get('truck_queue_crusher_h', 0):.1f} h",
                   f"{k.get('truck_queue_crusher_h', 0) - today.get('truck_queue_crusher_h', 0):+.1f} h vs dziś",
                   delta_color="inverse")
    cols[2].metric("⛏️ Koparki odstawione", f"{k.get('parked_excavator_h', 0):.1f} h")
    cols[3].metric("🪣 Kursy bez ostatniej łyżki", f"{k.get('topup_skips', 0)}")
    off = k.get("off_rule_trips") or {}
    if off:
        lines = [f"{t}: " + ", ".join(f"{n}× do {e}" for e, n in sorted(v.items())) for t, v in sorted(off.items())]
        st.markdown("**🔀 Zmiany przydziału względem reguł** (wozidła bez 🔒, kursy do innej koparki niż przypisana):  \n"
                    + "  \n".join(lines))
    elif any(f.excavators and not f.assign_locked for f in sc.trucks):
        st.caption("🔀 Polityka nie zmieniła przydziału wozideł – plan zmiany jest tu dobry.")


DAY, SCHEDULE = "Analiza zmiany", "Planowanie produkcji"


RL_MIN_GAIN = 0.005  # the learned policy must be at least 0.5 % cheaper than today to be used


def _with_policy(sc, res):
    """The scenario with the learned dispatch, advice and excavator management."""
    out = copy.deepcopy(sc)
    out.rl_weights, out.dispatch = res.weights, "rl"
    for key, val in (res.advisor or {}).items():
        setattr(out.advisor, key, val)
    for key, val in (res.excavators or {}).items():
        setattr(out.excavator_policy, key, val)
    return out


def bottleneck(k: dict) -> list[str]:
    """Why the dispatch has no room to improve: what limits the shift."""
    out = []
    cycles = max(k.get("cycles") or 1, 1)
    busy = max(k.get("crushers") or [{}], key=lambda c: c.get("utilization", 0))
    if busy.get("utilization", 0) >= 0.85:
        out.append(f"🏭 Ogranicza **kruszarka {busy['crusher']}** – pracuje na {100 * busy['utilization']:.0f} %, wozidła "
                   f"stoją pod koszem średnio **{60 * k['truck_queue_crusher_h'] / cycles:.0f} min na kurs**. Przydział "
                   "wozideł nie przerobi więcej, niż przyjmie kruszarka.")
        idle = [c["crusher"] for c in k.get("crushers", []) if c.get("utilization", 0) < 0.1]
        if idle:
            out.append(f"↪️ {', '.join(idle)} prawie stoi – sprawdź, z których usypów przyjmuje urobek.")
    if k.get("other_delay_h", 0) > 0.3 * (k.get("trucks", 1) * 8):
        out.append(f"⏸️ Dużą część czasu wozideł zajmują **przestoje z kalibracji** ({60 * k['other_delay_h'] / cycles:.0f} min "
                   "na kurs) – ich przydział tego nie zmieni.")
    if max((e.get("utilization", 0) for e in k.get("excavators", [])), default=0) >= 0.85:
        out.append("⛏️ Koparki ładują prawie bez przerwy – ogranicza załadunek.")
    out.append("💡 Realne dźwignie: **mniej wozideł** (ten sam urobek taniej – Planowanie produkcji → dobór floty), "
               "druga linia kruszenia przyjmująca ten urobek, większa wydajność wąskiego gardła.")
    return out


def render_simple(raw: dict, set_raw, render_map_tab) -> None:
    kind = st.radio("Moduł", [DAY, SCHEDULE], horizontal=True, key="work_kind",
                    help="Analiza zmiany: kalibracja modelu, przebieg zmiany i optymalizacja przydziału (RL). "
                         "Planowanie produkcji: plan na tydzień / miesiąc – cel produkcyjny, strzelania, flota, "
                         "zmiany, szansa realizacji.")
    schedule = kind == SCHEDULE
    seed = int(raw.get("seed", 42))
    try:
        sc = scenario_from_dict(copy.deepcopy(raw))
        yaml_text = dump_scenario(sc)
        if not _has_map_routes(sc):
            raise ValueError("brak tras z mapy")
    except Exception as e:  # noqa: BLE001
        sc, yaml_text = None, None
        msg = str(e)
    if schedule:
        st.caption("Kolejno: kopalnia i plan strzałów → wariant „co jeśli” → harmonogram (cel produkcji, kalendarz, "
                   "dobór floty, szansa realizacji).")
        tabs = st.tabs(["① Kopalnia i plan strzałów", "② Wariant – co jeśli", "③ Harmonogram"])
        with tabs[0]:
            render_map_tab(raw, set_raw, simple=True, schedule=True)
        with tabs[1]:
            step_variant(raw, yaml_text, seed)
        with tabs[2]:
            from app_period import step_period

            step_period(raw, yaml_text, seed)
        return
    st.caption("Kolejno: stan wyjściowy kopalni → plan zmiany → organizacja zmiany → kalibracja → odtworzenie zmiany → optymalizacja RL. "
               "Wszystkie parametry: **Tryb zaawansowany** (panel boczny).")
    tabs = st.tabs(["① Stan wyjściowy kopalni", "② Plan zmiany (reguły)", "③ Organizacja zmiany", "④ Kalibracja", "⑤ Jak było dziś",
                    "⑥ Optymalizacja RL"])
    with tabs[0]:
        render_map_tab(raw, set_raw, simple=True, schedule=False)
    with tabs[1]:
        from app_rules import step_rules

        step_rules(raw, set_raw)
    with tabs[2]:
        from app_events import events_editor

        events_editor(raw, "ev", [e["id"] for e in (raw.get("site") or {}).get("excavators", [])]
                      or [e["id"] for e in raw.get("excavators", [])], raw.get("site"), "③ Organizacja zmiany")
    with tabs[3]:
        from app_calibration import reliability_editor, step_calibration

        step_calibration(raw, sc, set_raw)
        if sc is not None:
            st.divider()
            reliability_editor(raw, sc, "rel_day")
    with tabs[4]:
        if sc is None:
            st.info(f"Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia** ({msg}) – albo wczytaj przykład.")
            _demo_button(set_raw, "demo_today")
        else:
            step_today(sc, yaml_text, seed, set_raw)
    with tabs[5]:
        step_optimize(raw, yaml_text, seed)
