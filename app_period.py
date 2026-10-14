"""Step "Month": the shift simulation run day after day for a week / month, with the mine state carried over."""
from __future__ import annotations

import calendar
import copy
import datetime as dt
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kruszyk.config import scenario_from_dict
from kruszyk.goal import GOAL_ITEMS, PRODUCTS, YIELD_PRESETS, apply_plan, default_yields, plan_backwards, products_made
from kruszyk.period import GOALS, WEEKDAYS, PeriodPlan, PileBlasts, optimize_period, run_period, summary

RL_LABEL = "📱 z systemem RL"


def _next_month(today: dt.date) -> dt.date:
    return (today.replace(day=1) + dt.timedelta(days=32)).replace(day=1)


def _holidays(text: str, year: int) -> list[dt.date]:
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            d, m = (int(x) for x in part.split(".")[:2])
            out.append(dt.date(year, m, d))
        except ValueError:
            pass
    return out


def step_period(base_raw: dict, base_yaml: str | None, seed: int) -> None:
    st.markdown("### 🗓️ Harmonogram na tydzień / miesiąc")
    st.markdown("**Kolejność:** 🎯 **cel produkcyjny** (ile ton jakich produktów) → program **liczy wstecz od "
                "kruszarek**: ile nadawy na każdą kruszarkę, z których usypów i obszarów, ile strzelać i wiercić, ile "
                "wozideł – najtaniej → **zastosuj** → ▶️ **Policz okres** sprawdza plan zmiana po zmianie (pogoda, "
                "awarie, kolejki) → ⚙️ **dobór floty i zmian** wybiera najtańszy wariant, który zdąży.")
    if base_yaml is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia**.")
        return
    for k_, v_ in st.session_state.pop("per_apply", {}).items():  # "apply this plan" from the goal tab
        st.session_state[k_] = v_
    st.session_state.setdefault("per_shifts", 2)
    variant = st.session_state.get("variant_yaml")
    c1, c2 = st.columns(2)
    which = c1.radio("Kopalnia", ["stan obecny"] + (["wariant z kroku ②"] if variant else []), horizontal=True,
                     key="per_which")
    sc = scenario_from_dict(yaml_load(variant if which.startswith("wariant") else base_yaml))
    if not sc.rl_weights and base_raw.get("rl_weights"):  # a variant made before learning: use the mine's policy
        sc.rl_weights = list(base_raw["rl_weights"])
    sc.dispatch = "natural"  # the schedule is planned on how trucks run today
    from app_calibration import status_line

    status_line(sc)
    with c2:
        if sc.rl_weights:
            compare_rl = st.toggle("📱 Porównaj z systemem przydziału na żywo (RL)", True, key="per_rl",
                                   help="Ten sam plan policzony drugi raz z przydziałem i zaleceniami nauczonymi w "
                                        "module „Analiza zmiany” – widać, ile ton i złotych na okres daje wdrożenie.")
        else:
            compare_rl = False
            st.caption("📱 Polityka RL nie jest jeszcze nauczona – zrób to w module **Analiza zmiany → ⑤ Optymalizacja RL**, "
                       "a tu zobaczysz, ile daje na cały okres.")
    st.caption("Baza harmonogramu: obecny sposób przydziału (wozidło wraca do przypisanej koparki)"
               + (" – obok to samo z systemem RL." if compare_rl else "."))

    applied = st.session_state.get("prod_applied")
    if applied is not None:
        apply_plan(sc, applied)  # crusher feed targets, accepted piles and feed shares from the production goal
        c1, c2 = st.columns([4, 1])
        c1.success(f"🎯 Zastosowany cel produkcyjny: **{applied.total_t:,.0f} t** produktów → nadawa "
                   f"**{applied.feed_t:,.0f} t**; udziały usypów i plany kruszarek ustawione wstecz od kruszarek."
                   .replace(",", " "))
        if c2.button("Wyłącz", key="prod_off"):
            st.session_state.pop("prod_applied")
            _reset_tables()
            st.rerun()

    t_prod, t_cal, t_old, t_new, t_ex, t_goal = st.tabs(
        ["🎯 Cel produkcyjny", "🗓️ Kalendarz", "🪨 Usypy leżące",
         f"📍 Obszary i strzały z mapy ({len(sc.planned_blasts) + len(sc.areas)})",
         "⛏️ Koparki, wozidła, wiertnice", "⚙️ Dobór floty i zmian"])
    with t_cal:
        c = st.columns(5)
        start = c[0].date_input("Od dnia", _next_month(dt.date.today()), key="per_start", format="DD.MM.YYYY")
        days = c[1].number_input("Liczba dni (kalendarzowych)", 1, 92,
                                 calendar.monthrange(start.year, start.month)[1] if start.day == 1 else 30, key="per_days")
        shifts = c[2].selectbox("Zmian na dobę", [1, 2, 3], key="per_shifts",
                                help=f"każda po {sc.shift_hours:g} h – pierwsza od {sc.shift_plan.start_time}")
        sat = c[3].checkbox("Soboty robocze", key="per_sat")
        target = c[4].number_input("Cel nadawy na okres [t] (0 = bez celu)", 0.0, 10_000_000.0, 0.0, 5000.0,
                                   key="per_target", help="ustawia się sam po zastosowaniu celu produkcyjnego")
        hol = st.text_input("Dni wolne (dd.mm, po przecinku)", "", key="per_hol", placeholder="np. 11.11, 24.12, 25.12")
        with st.expander("🕐 Organizacja zmiany – godziny, przerwy, objazd, tankowanie, prędkości"):
            from app_events import events_editor

            events_editor(base_raw, "evp", [e.id for e in sc.excavators], sc.site, plan=True)

    with t_old:
        st.caption("Urobek, który leży dziś po wcześniejszych odstrzałach – wybiera się go do końca. Nowy urobek "
                   "przybywa tylko z kolejnych strzałów: w obszarach wydobycia i planowanych (zakładka obok). "
                   "**Udział w nadawie** – ile procent nadawy ma iść z tego usypu; liczą się tylko usypy, z których na "
                   "danej zmianie da się ładować (udziały przeliczane do 100 %).")
        base_key = f"per_piles__{hashlib.md5(','.join(b.id for b in sc.blasts).encode()).hexdigest()}"
        if base_key not in st.session_state:
            st.session_state[base_key] = pd.DataFrame([{"usyp": b.id, "material": b.material, "teraz_t": b.tonnage_t,
                                                        "udzial": b.target_share} for b in sc.blasts])
        ed = st.data_editor(st.session_state[base_key], key=base_key + "_ed", hide_index=True, width="stretch",
                            disabled=["usyp", "material", "teraz_t"],
                            column_config={"usyp": "usyp", "material": "materiał",
                                           "teraz_t": st.column_config.NumberColumn("leży teraz [t]", format="%.0f"),
                                           "udzial": st.column_config.NumberColumn("udział w nadawie [%]", min_value=0.0,
                                                                                   max_value=100.0, step=5.0)})
    with t_new:
        st.markdown("**🟧 Obszary wydobycia – strzały ustala program**")
        if sc.areas:
            akey = f"per_areas__{hashlib.md5(repr([(a['id'], a.get('reserves_t'), a.get('blast_max_t')) for a in sc.areas]).encode()).hexdigest()}"
            if akey not in st.session_state:
                st.session_state[akey] = pd.DataFrame([{
                    "obszar": a["id"], "poziom": a.get("level_m"), "material": a.get("material", ""),
                    "zasoby_t": float(a.get("reserves_t") or 0),
                    "max_t": float(a.get("blast_max_t") or 20000), "ponizej_t": float(a.get("reblast_below_t") or 0),
                    "odstep": int(a.get("min_gap_days") or 1), "dziennie": int(a.get("max_per_day") or 1),
                    "udzial": a.get("target_share"), "tpm": float(a.get("t_per_m") or 50.0)} for a in sc.areas])
            aed = st.data_editor(st.session_state[akey], key=akey + "_ed", hide_index=True, width="stretch",
                                 disabled=["obszar", "material"],
                                 column_config={"obszar": "obszar", "material": "materiał",
                                                "poziom": st.column_config.NumberColumn("poziom [m]"),
                                                "zasoby_t": st.column_config.NumberColumn("zasoby [t]", min_value=0.0),
                                                "max_t": st.column_config.NumberColumn("największy strzał [t]", min_value=1000.0),
                                                "ponizej_t": st.column_config.NumberColumn("strzelaj poniżej [t]", min_value=0.0),
                                                "odstep": st.column_config.NumberColumn("odstęp [dni rob.]", min_value=1, step=1),
                                                "dziennie": st.column_config.NumberColumn("strzałów / dzień", min_value=1, step=1),
                                                "udzial": st.column_config.NumberColumn("udział w nadawie [%]",
                                                                                        min_value=0.0, max_value=100.0),
                                                "tpm": st.column_config.NumberColumn("t/mb", min_value=5.0)})
            for a, (_, r) in zip(sc.areas, aed.iterrows()):
                a.update({"reserves_t": float(r["zasoby_t"] or 0), "blast_max_t": float(r["max_t"] or 20000),
                          "reblast_below_t": float(r["ponizej_t"] or 0), "min_gap_days": int(r["odstep"] or 1),
                          "max_per_day": int(r["dziennie"] or 1), "t_per_m": float(r["tpm"] or 50.0),
                          "level_m": None if pd.isna(r["poziom"]) else float(r["poziom"]),
                          "target_share": None if pd.isna(r["udzial"]) or not r["udzial"] else float(r["udzial"])})
            lead = st.number_input("Minimalne wyprzedzenie frontu piętra wyższego [m]", 0.0, 500.0, 30.0, 5.0, key="per_lead",
                                   help="strzał na niższym poziomie dopiero, gdy wszystko z wyższego poziomu bliżej niż "
                                        "ta odległość jest już odstrzelone i odebrane")
            st.caption("Program dzieli obszar na pasy (od strony drogi) wielkości największego strzału. Strzela, gdy "
                       "urobku z obszaru zostaje mniej niż próg albo koparki nie mają co ładować – z zachowaniem "
                       "odstępu i limitu dziennego. Godziny strzału – z obszaru (ustawiasz na mapie).")
        else:
            st.caption("Brak obszarów – narysuj je w kroku ① (grupa „📅 Plan strzałów” → „Obszar wydobycia”).")
        st.markdown("**📅 Planowane strzały – konkretny dzień i miejsce**")
        st.caption("Rysujesz je na mapie w kroku ①. W dniu strzału pojawia się nowy usyp. Dzień, tony i udział "
                   "możesz tu zmienić bez rysowania.")
        if sc.planned_blasts:
            pkey = f"per_planned__{hashlib.md5(repr([(p['id'], p.get('blast_day'), p.get('target_share')) for p in sc.planned_blasts]).encode()).hexdigest()}"
            if pkey not in st.session_state:
                st.session_state[pkey] = pd.DataFrame([{"strzal": p["id"], "material": p.get("material", ""),
                                                        "dzien": int(p.get("blast_day") or 1), "ile_t": float(p["tonnage_t"]),
                                                        "udzial": p.get("target_share")} for p in sc.planned_blasts])
            ped = st.data_editor(st.session_state[pkey], key=pkey + "_ed", hide_index=True, width="stretch",
                                 disabled=["strzal", "material"],
                                 column_config={"strzal": "strzał", "material": "materiał",
                                                "dzien": st.column_config.NumberColumn("w dniu roboczym nr", min_value=1, step=1),
                                                "ile_t": st.column_config.NumberColumn("ile ton", min_value=0.0, step=1000.0),
                                                "udzial": st.column_config.NumberColumn("udział w nadawie [%]", min_value=0.0,
                                                                                        max_value=100.0, step=5.0)})
            for p, (_, r) in zip(sc.planned_blasts, ped.iterrows()):
                p["blast_day"], p["tonnage_t"] = int(r["dzien"] or 1), float(r["ile_t"] or 0)
                p["target_share"] = None if pd.isna(r["udzial"]) or not r["udzial"] else float(r["udzial"])
        else:
            st.caption("brak planowanych strzałów")
    with t_ex:
        st.session_state.setdefault("per_trucks", int(sc.truck_count))
        n_trucks = st.number_input("Liczba wozideł (wszystkie typy razem)", 1, 60, step=1,
                                   key="per_trucks", help="więcej = najpierw wracają wozidła odstawione, potem kopie ostatniego; mniej = "
                                                         "odstawiane od końca listy")
        if n_trucks != sc.truck_count:
            sc = sc.with_truck_count(int(n_trucks))
        c = st.columns(2)
        move = c[0].checkbox("Koparka bez urobku przejeżdża na inny usyp", True, key="per_move",
                             help="na najbliższy odstrzelony usyp, na którym nikt nie pracuje")
        setup = c[1].number_input("Przygotowanie nowego frontu [min] (oprócz jazdy)", 0.0, 240.0, 15.0, 5.0,
                                  key="per_setup", disabled=not move)
        st.markdown("**🔩 Wiertnice** – strzał w nowym miejscu trzeba najpierw nawiercić. Metry = tony strzału / t/mb "
                    "(ustawiasz przy obszarze / strzale), czas = metry / wydajność. Następny pas obszaru można wiercić "
                    "dopiero po odebraniu poprzedniego strzału.")
        rigs = _rigs_table(base_raw, sc)
        drills = len(rigs)
        c = st.columns(2)
        clear = c[0].number_input("Odebrany, gdy zostało ≤ [%]", 0.0, 100.0, 10.0, 5.0, key="per_clear",
                                  disabled=drills == 0, help="poprzedniego strzału w obszarze – wtedy wiertnica może wejść")
        tpm = c[1].number_input("Domyślnie t/mb", 5.0, 500.0, 50.0, 5.0, key="per_tpm", disabled=drills == 0,
                                help="gdy obszar / strzał nie ma własnej wartości")
        from app_calibration import reliability_editor

        reliability_editor(base_raw, sc, "rel_per")
    plan = PeriodPlan(start=start, days=int(days), shifts_per_day=int(shifts), saturdays=sat,
                      holidays=_holidays(hol, start.year), target_t=float(target),
                      blasts=[PileBlasts(r["usyp"], 0.0, 0, 1,  # piles only get emptied – no new blasts in a dug-out place
                                         None if pd.isna(r["udzial"]) or not r["udzial"] else float(r["udzial"]))
                              for _, r in ed.iterrows()],
                      move_excavators=move, move_setup_min=float(setup), drills=int(drills), rigs=rigs,
                      drill_m_per_shift=float(np.mean([r["m_per_shift"] for r in rigs])) if rigs else 250.0,
                      clear_pct=float(clear), t_per_m=float(tpm),
                      lead_m=float(st.session_state.get("per_lead", 30.0)))
    with t_cal:
        _blast_calendar(plan, sc)
    n = len(plan.shifts())
    st.caption(f"{len(plan.working_days())} dni roboczych × {shifts} = **{n} zmian** · liczy się ok. {max(1, n // 3)} s")
    with t_goal:
        _goal_tab(sc, plan, seed, compare_rl)
    with t_prod:
        _production_tab(base_raw, sc, plan, n)

    key = hashlib.md5(repr((dump_key(sc), plan, seed, compare_rl)).encode()).hexdigest()
    if st.button("▶️ Policz okres", type="primary", disabled=n == 0):
        runs = 2 if compare_rl else 1
        bar = st.progress(0.0, "liczę zmiany…")
        df = run_period(sc, plan, seed=seed, progress=lambda i, k: bar.progress(i / k / runs, f"zmiana {i} z {k}"))
        df_rl = None
        if compare_rl:
            sc_rl = copy.deepcopy(sc)
            sc_rl.dispatch = "rl"
            df_rl = run_period(sc_rl, plan, seed=seed,
                               progress=lambda i, k: bar.progress(0.5 + i / k / 2, f"z systemem RL: zmiana {i} z {k}"))
        bar.empty()
        st.session_state.period = (key, df, plan, df_rl)
    res = st.session_state.get("period")
    if res:
        if res[0] != key:
            st.warning("Ustawienia zmieniły się od ostatniego liczenia – kliknij **Policz okres** ponownie.")
        if len(res) > 3 and res[3] is not None:
            _rl_compare(res[1], res[3], res[2], sc)
        _results(res[1], res[2], sc)
    st.divider()
    _chance_section(sc, plan, seed, key)


def _chance_section(sc, plan: PeriodPlan, seed: int, key: str) -> None:
    """Monte Carlo: the same plan many times with different weather, breakdowns, drivers – how sure is it?"""
    from kruszyk.montecarlo import chance, run_many

    st.markdown("#### 🎲 Szansa realizacji")
    st.caption("Jedno „Policz okres” to **jeden możliwy miesiąc**. Tu ten sam plan liczy się wiele razy, za każdym razem "
               "z innym losowaniem pogody, awarii, kierowców i czasów – widać, **jak pewny** jest wynik. To informacja: "
               "plan się nie zmienia.")
    gp = st.session_state.get("prod_applied")
    target = float(plan.target_t or 0.0)
    c = st.columns([1, 1, 2])
    n = c[0].selectbox("Przebiegów", [10, 20, 40], index=1, key="mc_n",
                       help="więcej = dokładniej, dłużej; 20 daje szansę z dokładnością ok. ±10–20 pp")
    calibrated = bool(sc.calibration.measured)
    model_unc = c[1].toggle("Niepewność kalibracji", calibrated, key="mc_model", disabled=not calibrated,
                            help="model z jednej zmiany też nie jest pewny – każdy przebieg dostaje trochę inne czasy "
                                 "i przestoje (±5 %, przestoje ±10 %)")
    per = max(1, len(plan.shifts()) // 4)
    from kruszyk.montecarlo import safe_workers

    w = safe_workers()
    c[2].caption(f"ok. {n * per // w // 60} min {n * per // w % 60} s ({w} równolegle) · cel nadawy: "
                 + (f"**{target:,.0f} t**".replace(",", " ") if target else "brak – ustaw w kalendarzu lub z celu produkcyjnego"))
    mc_key = repr((key, n, model_unc, gp.total_t if gp else None))
    if st.button("🎲 Policz szansę realizacji", key="mc_go", disabled=not plan.shifts()):
        bar = st.progress(0.0, "liczę przebiegi…")
        runs = run_many(sc, plan, n=n, seed=seed, model_sd=0.05 if model_unc else 0.0,
                        progress=lambda i, k: bar.progress(i / k, f"przebieg {i} z {k}"))
        bar.empty()
        st.session_state.mc_res = (mc_key, chance(runs, plan, target, goal=gp))
    got = st.session_state.get("mc_res")
    if not got:
        return
    if got[0] != mc_key:
        st.info("Plan zmienił się od ostatniego liczenia – policz szansę ponownie.")
    _chance_results(got[1])


def _chance_results(r: dict) -> None:
    def t(x) -> str:
        return f"{x:,.0f}".replace(",", " ")
    c = st.columns(4)
    if r.get("p") is not None:
        p = 100 * r["p"]
        c[0].metric(f"Szansa: {t(r['target_t'])} t do {r['deadline'].strftime('%d.%m')}", f"{p:.0f} %",
                    f"± {100 * r['p_margin']:.0f} pp (z {r['n']} przebiegów)", delta_color="off")
    c[1].metric("Wywiezione (P10 / P50 / P90)", f"{t(r['t_p50'])} t", f"{t(r['t_p10'])} – {t(r['t_p90'])} t",
                delta_color="off", help="P10 = w 9 na 10 miesięcy co najmniej tyle; P90 = tylko 1 na 10 miesięcy więcej")
    c[2].metric("Koszt na tonę (P50)", f"{r['cost_p50']:.2f} zł/t", f"{r['cost_p10']:.2f} – {r['cost_p90']:.2f} zł/t",
                delta_color="off")
    if r.get("hit_p50"):
        c[3].metric("Cel zrobiony (typowo)", r["hit_p50"].strftime("%d.%m"),
                    f"9 na 10: do {r['hit_p90'].strftime('%d.%m')}" if r.get("hit_p90") else "w 1 na 10 – nie w okresie",
                    delta_color="off")
    if r.get("products"):
        st.markdown("**Szansa dla produktów z celu produkcyjnego**")
        st.dataframe(pd.DataFrame(r["products"]), hide_index=True, width="stretch",
                     column_config={"cel_t": st.column_config.NumberColumn("cel [t]", format="%.0f"),
                                    "szansa_pct": st.column_config.ProgressColumn("szansa [%]", min_value=0, max_value=100,
                                                                                  format="%d %%"),
                                    "p10_t": st.column_config.NumberColumn("P10 [t]", format="%.0f"),
                                    "p50_t": st.column_config.NumberColumn("P50 [t]", format="%.0f")})
    b = r["band"]
    fig = go.Figure()
    fig.add_scatter(x=b["data"], y=b["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=b["data"], y=b["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(33,158,188,0.25)", name="P10–P90 (8 na 10 miesięcy)")
    fig.add_scatter(x=b["data"], y=b["p50"], mode="lines", line=dict(color="#219ebc", width=3), name="P50 (typowo)")
    if r["target_t"]:
        fig.add_hline(y=r["target_t"], line_dash="dash", line_color="#e63946", annotation_text="cel")
        fig.add_vline(x=r["deadline"], line_dash="dot", line_color="#e63946")
    fig.update_layout(height=320, margin=dict(t=30, b=10), title="Tony narastająco – pasmo możliwych miesięcy",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    if r["drivers"]:
        st.markdown("**Co mają nieudane przebiegi:**  \n" + "  \n".join("• " + d for d in r["drivers"]))
    if r["t_p90"] - r["t_p10"] < 0.02 * max(r["t_p50"], 1):
        st.info("📏 Wyniki prawie się nie różnią – produkcję ogranicza stały element (zwykle kruszarka), a losowe "
                "zdarzenia go nie dotykają. W modelu psują się na razie tylko wozidła, więc ta pewność może być "
                "**zawyżona** – awarie kruszarek i koparek zmieniłyby obraz.")


RIG_DEFAULT = {"m_per_shift": 250.0, "speed_kmh": 2.0}


def _rigs_table(base_raw: dict, sc) -> list[dict]:
    """Every drill rig with its own output, speed and breakdowns. Returns the rigs in work."""
    from kruszyk.config import Reliability

    rigs = base_raw.get("drills")
    if rigs is None:
        rigs = base_raw["drills"] = [{"id": "W-1", "in_work": True, **RIG_DEFAULT}]
    need = st.session_state.pop("per_rigs_needed", 0)  # the production goal asked for more rigs
    while sum(1 for r in rigs if r.get("in_work", True)) < need:
        idle = next((r for r in rigs if not r.get("in_work", True)), None)
        if idle:
            idle["in_work"] = True
        else:
            k = len(rigs) + 1
            rigs.append({**(rigs[-1] if rigs else RIG_DEFAULT), "id": f"W-{k}", "in_work": True})
    rel_raw = base_raw.setdefault("reliability", {})
    rel = Reliability(**{k_: v_ for k_, v_ in rel_raw.items() if k_ in Reliability.__dataclass_fields__})

    def fail(rid: str) -> tuple:
        return rel.values(rid if rid in rel.machines else "drill", "drill")[:2]
    sig = abs(hash(repr(rigs) + repr(sorted(rel.machines.items())) + str(st.session_state.get("ver", 0)))) % 10**8
    key = f"per_rigs_{sig}"
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame([{"id": r["id"], "w_pracy": bool(r.get("in_work", True)),
                                               "m": float(r.get("m_per_shift") or 250), "v": float(r.get("speed_kmh") or 2),
                                               "fail_n": fail(r["id"])[0], "fail_h": fail(r["id"])[1]} for r in rigs],
                                             columns=["id", "w_pracy", "m", "v", "fail_n", "fail_h"])
    start = st.session_state[key]
    ed = st.data_editor(start, key=key + "_ed", num_rows="dynamic", hide_index=True, width="stretch",
                        column_config={"id": "wiertnica", "w_pracy": st.column_config.CheckboxColumn("w pracy"),
                                       "m": st.column_config.NumberColumn("wydajność [m / zmianę]", min_value=10.0),
                                       "v": st.column_config.NumberColumn("przejazd [km/h]", min_value=0.5),
                                       "fail_n": st.column_config.NumberColumn("awarii / mies.", min_value=0.0),
                                       "fail_h": st.column_config.NumberColumn("godzin awarii / mies.", min_value=0.0)})
    new, used = [], set()
    machines = rel_raw.setdefault("machines", {})
    for i, r in ed.reset_index(drop=True).iterrows():
        prev = new[-1] if new else {**RIG_DEFAULT}
        rid = str(r["id"] or "").strip() if not pd.isna(r["id"]) else ""
        k = len(new) + 1
        while not rid or rid in used:
            rid, k = f"W-{k}", k + 1
        used.add(rid)
        rig = {"id": rid, "in_work": bool(r["w_pracy"]) if not pd.isna(r["w_pracy"]) else True,
               "m_per_shift": float(r["m"]) if not pd.isna(r["m"]) else prev["m_per_shift"],
               "speed_kmh": float(r["v"]) if not pd.isna(r["v"]) else prev["speed_kmh"]}
        new.append(rig)
        if not pd.isna(r["fail_n"]) and not pd.isna(r["fail_h"]):
            was = (float(start["fail_n"].iloc[i]), float(start["fail_h"].iloc[i])) if i < len(start) else fail(rid)
            if (float(r["fail_n"]), float(r["fail_h"])) != was:
                machines[rid] = {"count": float(r["fail_n"]), "hours": float(r["fail_h"])}
    base_raw["drills"] = new
    working = [r for r in new if r["in_work"]]
    st.caption(f"W pracy: **{len(working)}** z {len(new)} wiertnic" + ("" if working else " – bez wiercenia (strzały "
                                                                          "w nowych miejscach od razu nawiercone)") + ".")
    return working


def _reset_tables() -> None:
    """Share tables are snapshots: drop them so they are rebuilt from the scenario."""
    for k_ in [k_ for k_ in st.session_state if str(k_).startswith(("per_piles__", "per_areas__", "per_planned__"))]:
        del st.session_state[k_]


def _production_tab(base_raw: dict, sc, plan: PeriodPlan, n_shifts: int) -> None:
    """Production goal of the period (tonnes of products) planned backwards from the crushers."""
    prod = base_raw.setdefault("production", {})  # kept in the mine file
    st.markdown("Ile ton **jakich produktów** trzeba zrobić w okresie. Program liczy **wstecz od kruszarek**: z uzysków "
                "każdej linii wie, ile nadawy potrzeba, a potem wybiera **najtańsze** źródła (koszt jazdy, kruszenia, "
                "wiercenia i strzelania) w granicach wydajności kruszarek i koparek oraz zasobów usypów i obszarów.")
    c1, c2 = st.columns([1, 2])
    total = c1.number_input("Produkty razem w okresie [t]", 1000.0, 10_000_000.0, float(prod.get("total_t", 200_000)),
                            10_000.0, key="prod_total", format="%.0f")
    mix0 = prod.get("mix") or {"grysy (2–16)": 75.0, "kliniec 16–31,5": 15.0, "mieszanki / podbudowa": 10.0}
    if "prod_mix_df" not in st.session_state:
        st.session_state.prod_mix_df = pd.DataFrame([{"produkt": k_, "udzial": v_} for k_, v_ in mix0.items()])
    with c2:
        mix_df = st.data_editor(st.session_state.prod_mix_df, key="prod_mix_ed", num_rows="dynamic", hide_index=True,
                                width="stretch",
                                column_config={"produkt": st.column_config.SelectboxColumn("produkt / grupa",
                                                                                           options=GOAL_ITEMS, required=True),
                                               "udzial": st.column_config.NumberColumn("udział [%]", min_value=0.0,
                                                                                       max_value=100.0, step=5.0)})
    mix = {str(r["produkt"]): float(r["udzial"] or 0) for _, r in mix_df.iterrows()
           if isinstance(r["produkt"], str) and not pd.isna(r["udzial"])}
    tot_pct = sum(mix.values())
    if abs(tot_pct - 100) > 0.5:
        st.warning(f"Udziały sumują się do **{tot_pct:.0f} %** – liczę z nich proporcjonalnie (do 100 %).")
    mix = {k_: 100 * v_ / tot_pct for k_, v_ in mix.items()} if tot_pct else {}
    for k_ in mix:
        st.caption(f"{k_}: **{total * mix[k_] / 100:,.0f} t**".replace(",", " ").replace("31 5", "31,5"))

    st.markdown("**🏭 Uzyski kruszarek** – na co rozdziela się 1 t nadawy [%] (reszta to podziarno / odsiew).")
    stored = prod.get("yields") or {}
    ids = [c.id for c in sc.crushers]
    if st.session_state.get("prod_y_ids") != ids:
        st.session_state.prod_y_ids = ids
        st.session_state.prod_y_df = pd.DataFrame(
            [{"produkt": p, **{c.id: float((stored.get(c.id) or default_yields(c.product)).get(p, 0.0))
                              for c in sc.crushers}} for p in PRODUCTS])
    c = st.columns([2, 1, 1])
    preset = c[0].selectbox("Typowe uzyski", list(YIELD_PRESETS), key="prod_preset")
    who = c[1].selectbox("dla kruszarki", ids, key="prod_preset_c")
    if c[2].button("Wstaw", key="prod_preset_go"):
        df = st.session_state.prod_y_df.copy()
        df[who] = [YIELD_PRESETS[preset].get(p, 0.0) for p in df["produkt"]]
        st.session_state.prod_y_df = df
        st.session_state.pop("prod_y_ed", None)
        st.rerun()
    y_df = st.data_editor(st.session_state.prod_y_df, key="prod_y_ed", hide_index=True, width="stretch",
                          disabled=["produkt"],
                          column_config={c.id: st.column_config.NumberColumn(
                              f"{c.id} [%]" + (f" · {c.product}" if c.product else ""), min_value=0.0, max_value=100.0)
                              for c in sc.crushers})
    yields = {c.id: {r["produkt"]: float(r[c.id]) for _, r in y_df.iterrows() if r[c.id] and not pd.isna(r[c.id])}
              for c in sc.crushers}
    for c in sc.crushers:
        tot = sum(yields[c.id].values())
        if tot > 100.5:
            st.warning(f"{c.id}: uzyski sumują się do {tot:.0f} % – więcej niż nadawa.")
    prod.update({"total_t": float(total), "mix": {k_: round(v_, 2) for k_, v_ in mix.items()}, "yields": yields})

    drills = plan.drills
    key = repr((total, mix, yields, n_shifts, dump_key(sc), plan))
    if st.button("🔁 Policz plan wstecz od kruszarek", type="primary", key="prod_go", disabled=not mix or not n_shifts):
        with st.spinner("liczę dzisiejszą zmianę (kalibracja transportu) i plan…"):
            today = _today_kpi(dump_key(sc), int(sc.seed))
            st.session_state.prod_gp = (key, plan_backwards(
                sc, total, mix, yields, n_shifts, None, plan.t_per_m, drills, plan.drill_m_per_shift, today))
    got = st.session_state.get("prod_gp")
    if not got:
        st.caption(f"Okres: **{n_shifts} zmian** (kalendarz w zakładce obok).")
        return
    if got[0] != key:
        st.info("Cel, uzyski albo kalendarz zmieniły się – policz plan ponownie.")
    _production_plan(got[1])


@st.cache_data(show_spinner=False, max_entries=8)
def _today_kpi(yaml_text: str, seed: int) -> dict:
    """One shift as it runs today – how many tonnes the fleet really hauls on these roads."""
    from kruszyk.sim import simulate

    sc_ = scenario_from_dict(yaml_load(yaml_text))
    sc_.dispatch = "natural"
    return {"hauled_t": float(simulate(sc_, seed=seed).kpi["hauled_t"])}


def _production_plan(gp) -> None:
    def t(x) -> str:
        return f"{x:,.0f}".replace(",", " ")
    head = (f"Nadawa **{t(gp.feed_t)} t** na kruszarki → **{t(sum(p['plan_t'] for p in gp.products))} t** produktów z "
            f"celu · koszt ok. **{t(gp.cost_zl)} zł**" + (f" (**{gp.cost_per_t:.2f} zł/t** produktu)" if gp.cost_per_t else ""))
    (st.success if gp.feasible else st.warning)(("✅ Cel do zrobienia. " if gp.feasible else "⚠️ Celu nie da się w pełni "
                                                 "zrobić. ") + head)
    for line in gp.limits:
        st.markdown("• " + line)
    c1, c2 = st.columns(2)
    c1.markdown("**Produkty: cel i plan**")
    pr = pd.DataFrame(gp.products)
    pr["nadwyzka_t"] = (pr["plan_t"] - pr["cel_t"]).clip(lower=0)
    c1.dataframe(pr, hide_index=True, width="stretch",
                 column_config={"nadwyzka_t": st.column_config.NumberColumn(
                                    "nadwyżka [t]", format="%.0f", help="produkt uboczny ponad cel – na skład / sprzedaż"),"cel_t": st.column_config.NumberColumn("cel [t]", format="%.0f"),
                                "plan_t": st.column_config.NumberColumn("z planu [t]", format="%.0f"),
                                "brak_t": st.column_config.NumberColumn("brak [t]", format="%.0f")})
    c2.markdown("**Kruszarki: nadawa**")
    c2.dataframe(pd.DataFrame(gp.crushers).drop(columns=["pelna"]), hide_index=True, width="stretch",
                 column_config={"nadawa_t": st.column_config.NumberColumn("w okresie [t]", format="%.0f"),
                                "na_zmiane_t": st.column_config.NumberColumn("na zmianę [t]", format="%.0f"),
                                "wykorzystanie_pct": "wykorzystanie [%]"})
    st.markdown("**Skąd urobek, ile strzelać i wiercić**")
    st.dataframe(pd.DataFrame(gp.sources), hide_index=True, width="stretch",
                 column_config={"skad": "skąd", "dostepne_t": st.column_config.NumberColumn("dostępne [t]", format="%.0f"),
                                "plan_t": st.column_config.NumberColumn("do wybrania [t]", format="%.0f"),
                                "udzial_pct": "udział w nadawie [%]",
                                "do_odstrzalu_t": st.column_config.NumberColumn("do odstrzelenia [t]", format="%.0f"),
                                "strzalow": "strzałów", "wiercenie_m": "wiercenie [m]"})
    with st.expander("Trasy urobku: skąd → która kruszarka"):
        st.dataframe(pd.DataFrame(gp.flows), hide_index=True, width="stretch",
                     column_config={"skad": "skąd", "t": st.column_config.NumberColumn("t", format="%.0f"),
                                    "odleglosc_m": "odległość [m]", "koszt_na_t": "koszt [zł/t]"})
    d, h = gp.drilling, gp.haulage
    c = st.columns(4)
    c[0].metric("Wiercenie", f"{t(d['metry'])} m", f"{d['zmian_wiertnicy']:g} zmian wiertnicy", delta_color="off")
    c[1].metric("Wiertnic potrzeba", d["wiertnic_potrzeba"], f"jest {d['wiertnic_jest']}", delta_color="off")
    c[2].metric("Wozideł potrzeba", h["wozidel_potrzeba"], f"jest {h['wozidel_jest']}", delta_color="off",
                help="z odległości, skalibrowane na dzisiejszej zmianie (rampy, mijanki, kolejki); dokładnie "
                     "sprawdza ▶️ Policz okres i ⚙️ Dobór floty")
    c[3].metric("Koparki załadują", f"{t(h['zaladunek_na_zmiane_t'])} t/zm.", delta_color="off")
    if st.button("✅ Zastosuj w harmonogramie", key="prod_apply"):
        st.session_state.prod_applied = gp
        st.session_state.per_apply = {"per_target": float(gp.feed_t), "goal_kind": "deadline", "goal_t": float(gp.feed_t),
                                      "per_trucks": max(1, int(h["wozidel_potrzeba"])),
                                      "per_rigs_needed": int(d["wiertnic_potrzeba"])}
        _reset_tables()
        st.rerun()
    st.caption("Zastosuj → udziały usypów i obszarów, plany kruszarek, cel nadawy, liczba wozideł i wiertnic ustawią się "
               "same. Potem ▶️ **Policz okres** pokaże, czy plan wychodzi zmiana po zmianie.")


def _rl_compare(df: pd.DataFrame, df_rl: pd.DataFrame, plan: PeriodPlan, sc) -> None:
    """The same period with the live dispatch system (RL): what the rollout gives management."""
    a, b = summary(df, plan, sc.crushers), summary(df_rl, plan, sc.crushers)
    if not a or not b:
        return
    st.markdown("#### 📱 Co daje system przydziału na żywo (RL) w tym okresie")
    c = st.columns(4)
    c[0].metric("Wywiezione z RL", f"{b['wywiezione_t']:,.0f} t".replace(",", " "),
                f"{b['wywiezione_t'] - a['wywiezione_t']:+,.0f} t vs dziś".replace(",", " "))
    if a["koszt_na_t"] and b["koszt_na_t"]:
        c[1].metric("Koszt na tonę z RL", f"{b['koszt_na_t']:.2f} zł/t",
                    f"{b['koszt_na_t'] - a['koszt_na_t']:+.2f} zł/t", delta_color="inverse")
        saving = (a["koszt_na_t"] - b["koszt_na_t"]) * b["wywiezione_t"]
        c[2].metric("Oszczędność w okresie", f"{saving / 1000:,.0f} tys. zł".replace(",", " "),
                    help="różnica zł/t × tony wywiezione z RL")
    if plan.target_t and b["realizacja_celu_pct"] is not None:
        c[3].metric("Realizacja celu z RL", f"{b['realizacja_celu_pct']:.0f} %",
                    f"{b['realizacja_celu_pct'] - (a['realizacja_celu_pct'] or 0):+.0f} pp")
    fig = go.Figure()
    for name, d, color in (("jak dziś", df, "#9aa0a6"), ("z systemem RL", df_rl, "#d62828")):
        cum = d.groupby("data")["wywiezione_t"].sum().cumsum()
        fig.add_scatter(x=list(cum.index), y=cum.values, mode="lines", name=name, line=dict(color=color, width=3))
    if plan.target_t:
        fig.add_hline(y=plan.target_t, line_dash="dash", line_color="#e63946", annotation_text="cel")
    fig.update_layout(height=280, margin=dict(t=30, b=10), title="Tony narastająco: jak dziś vs z systemem RL",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Ta sama pogoda, awarie i strzały w obu wersjach. Szczegóły poniżej – dla wersji „jak dziś”.")
    st.divider()


def _blast_schedule(meta: dict, sc) -> None:
    """Blasts fired in the period (planned + generated in extraction areas): table and map."""
    blasts = meta.get("blasts", [])
    for w in meta.get("waits", []):
        st.info("⛰️ " + w)
    if not blasts:
        st.caption("W okresie nie było strzałów w nowych miejscach (obszary / planowane strzały).")
        return
    times = {p["id"]: p for p in sc.planned_blasts} | {a["id"]: a for a in sc.areas}
    drilled = {d["strzal"]: d for d in meta.get("drilling", [])}
    rows = []
    for b in blasts:
        src = times.get(b["obszar"] or b["strzal"], {})
        rows.append({"data": b["data"].strftime("%d.%m"), "dzień": WEEKDAYS[b["data"].weekday()], "strzał": b["strzal"],
                     "obszar": b["obszar"] or "–", "ile [t]": round(b["t"]),
                     "skąd": "program (obszar)" if b["auto"] else "plan z mapy",
                     "nawiercony": (f"{drilled[b['strzal']]['data'].strftime('%d.%m')} z{drilled[b['strzal']]['zmiana']} "
                                    f"({drilled[b['strzal']]['wiertnica']})") if b["strzal"] in drilled else "wcześniej",
                     "opóźnienie [dni rob.]": b.get("opoznienie_dni") or None,
                     "przyjazd ekipy": src.get("crew_arrive") or sc.blasting.crew_arrive,
                     "odpalenie": f"{src.get('fire_from') or sc.blasting.fire_from}–{src.get('fire_to') or sc.blasting.fire_to}",
                     "wyjazd ekipy": src.get("crew_leave") or sc.blasting.crew_leave})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if sc.site:
        import folium
        from streamlit_folium import st_folium

        from app_map import base_map, draw_site
        from kruszyk.site import centroid

        m = base_map(sc.site)
        draw_site(m, sc.site)
        for b in blasts:
            if not b.get("polygon"):
                continue
            folium.Polygon([tuple(c) for c in b["polygon"]], color="#ff006e", weight=2, fill=True, fill_opacity=0.35,
                           tooltip=f"{b['strzal']} · {b['data'].strftime('%d.%m')} · {b['t']:,.0f} t".replace(",", " ")
                           ).add_to(m)
            folium.Marker(centroid(b["polygon"]), icon=folium.DivIcon(
                html=f"<div style='color:#fff;font-size:11px;font-weight:700;text-shadow:0 0 3px #000;white-space:nowrap'>"
                     f"{b['data'].strftime('%d.%m')}</div>")).add_to(m)
        st_folium(m, height=460, use_container_width=True, returned_objects=[], key="period_blast_map")


def _goal_tab(sc, plan: PeriodPlan, seed: int, compare_rl: bool) -> None:
    """Main goal of the period and automatic choice of the plan (fleet, shifts, Saturdays); RL shown alongside."""
    st.caption("Wybierz, **na czym Ci najbardziej zależy**, i co program może zmieniać. Policzy okres dla każdej "
               "kombinacji (ta sama pogoda i awarie dla wszystkich – uczciwe porównanie) i ułoży ranking planów. "
               "Strzały, kruszarki i koparki są takie, jak w zakładkach obok.")
    goal = st.radio("Cel główny", list(GOALS), format_func=GOALS.get, key="goal_kind")
    target_t, deadline = 0.0, None
    if goal == "deadline":
        c = st.columns(2)
        target_t = c[0].number_input("Ile ton trzeba zrobić", 1000.0, 10_000_000.0, 100_000.0, 5000.0, key="goal_t")
        last = plan.start + dt.timedelta(days=plan.days - 1)
        deadline = c[1].date_input("Do kiedy", last, min_value=plan.start, max_value=last, key="goal_date",
                                   format="DD.MM.YYYY")
    st.markdown("**Co program może zmieniać:**")
    c = st.columns(3)
    n0 = int(sc.truck_count)
    trucks = c[0].multiselect("Liczba wozideł", list(range(max(1, n0 - 4), n0 + 5)),
                              [x for x in (n0 - 2, n0, n0 + 2) if x >= 1], key="goal_trucks")
    shifts = c[1].multiselect("Zmian na dobę", [1, 2, 3], [plan.shifts_per_day], key="goal_shifts")
    sats = c[2].multiselect("Soboty", ["bez sobót", "z sobotami"], ["z sobotami" if plan.saturdays else "bez sobót"],
                            key="goal_sats")
    disp = ["natural"] + (["rl"] if compare_rl else [])
    combos = len(trucks) * len(shifts) * len(sats) * len(disp)
    if compare_rl:
        st.caption("Każdy plan liczony dwa razy: jak dziś i z systemem RL – ranking według wersji „jak dziś”.")
    per = max(1, plan.days * max(shifts or [1]) // 18)
    st.caption(f"**{combos}** planów do policzenia · ok. {combos * per // 60} min {combos * per % 60} s")
    if st.button("🎯 Dobierz plan", type="primary", disabled=combos == 0 or combos > 60, key="goal_run"):
        bar = st.progress(0.0, "liczę plany…")
        res = optimize_period(sc, plan, goal, trucks, shifts, [x == "z sobotami" for x in sats], disp,
                              target_t, deadline, seed, progress=lambda i, k: bar.progress(i / k, f"plan {i} z {k}"))
        bar.empty()
        st.session_state.goal_res = (res, goal, target_t, deadline)
    if combos > 60:
        st.warning("Za dużo kombinacji (max 60) – zawęź wybór.")
    if st.session_state.get("goal_res"):
        _goal_results(*st.session_state.goal_res)


def _goal_results(res: pd.DataFrame, goal: str, target_t: float, deadline) -> None:
    def name(r) -> str:
        return f"{r['wozidla']} wozideł · {r['zmian_na_dobe']} zm./dobę · {'z sobotami' if r['soboty'] else 'bez sobót'}"
    with_rl = res[res["przydzial"] == "rl"]
    res = res[res["przydzial"] != "rl"].reset_index(drop=True)
    res["miejsce"] = range(1, len(res) + 1)
    best = res.iloc[0]
    ok = goal != "deadline" or (best["cel_osiagniety"] is not None and best["cel_osiagniety"] <= deadline)
    def t(x) -> str:
        return f"{x:,.0f}".replace(",", " ")
    msg = f"**Najlepszy plan:** {name(best)} → **{t(best['wywiezione_t'])} t**, **{best['koszt_na_t']:.2f} zł/t**"
    if goal == "deadline":
        msg += (f", cel {t(target_t)} t osiągnięty **{best['cel_osiagniety'].strftime('%d.%m')}**" if ok else
                f" – **celu nie da się zrobić do {deadline.strftime('%d.%m')}** w żadnym planie "
                f"(najbliżej: {t(best['do_terminu_t'])} t)")
    (st.success if ok else st.warning)(msg)
    same = with_rl[(with_rl["wozidla"] == best["wozidla"]) & (with_rl["zmian_na_dobe"] == best["zmian_na_dobe"])
                   & (with_rl["soboty"] == best["soboty"])]
    if len(same):
        r = same.iloc[0]
        line = (f"📱 **Ten sam plan z systemem RL:** {t(r['wywiezione_t'])} t ({r['wywiezione_t'] - best['wywiezione_t']:+,.0f} t), "
                f"{r['koszt_na_t']:.2f} zł/t ({r['koszt_na_t'] - best['koszt_na_t']:+.2f} zł/t)").replace(",", " ")
        if goal == "deadline" and r["cel_osiagniety"] is not None:
            line += f", cel osiągnięty {r['cel_osiagniety'].strftime('%d.%m')}"
        st.info(line)
    # what limits production: if very different fleets give (almost) the same tonnes, it's the material
    top = res[(res["zmian_na_dobe"] == best["zmian_na_dobe"]) & (res["soboty"] == best["soboty"])]
    if top["wozidla"].nunique() > 1 and top["wywiezione_t"].max() < 1.03 * top["wywiezione_t"].min():
        st.info("💡 Przy tych samych zmianach różna liczba wozideł daje prawie tyle samo ton – **ograniczeniem jest "
                "urobek (strzały), nie flota**. Więcej ton da częstszy / większy strzał albo nowe miejsce strzału, "
                "a mniej wozideł obniży koszt.")
    show = res.drop(columns=["_df", "_score"]).copy()
    show["plan"] = [name(r) for _, r in res.iterrows()]
    show["cel_osiagniety"] = show["cel_osiagniety"].map(lambda d: d.strftime("%d.%m") if d is not None else "–")
    cols = ["miejsce", "plan", "wywiezione_t", "koszt_na_t", "koszt_zl", "zmian"] + \
        (["do_terminu_t", "cel_osiagniety"] if goal == "deadline" else [])
    st.dataframe(show[cols], hide_index=True, width="stretch",
                 column_config={"wywiezione_t": st.column_config.NumberColumn("wywiezione [t]", format="%.0f"),
                                "koszt_na_t": "zł/t", "koszt_zl": st.column_config.NumberColumn("koszt [zł]", format="%.0f"),
                                "do_terminu_t": st.column_config.NumberColumn("do terminu [t]", format="%.0f"),
                                "cel_osiagniety": "cel osiągnięty"})
    fig = go.Figure()
    for _, r in res.head(4).iterrows():
        cum = r["_df"].groupby("data")["wywiezione_t"].sum().cumsum()
        fig.add_scatter(x=list(cum.index), y=cum.values, mode="lines", name=f"#{r['miejsce']} {name(r)}")
    if target_t:
        fig.add_hline(y=target_t, line_dash="dash", line_color="#e63946", annotation_text="cel")
    if deadline:
        fig.add_vline(x=deadline, line_dash="dot", line_color="#e63946")
    fig.update_layout(height=360, margin=dict(t=30, b=10), title="Tony narastająco – 4 najlepsze plany",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    if st.button("✅ Ustaw najlepszy plan w kalendarzu", key="goal_apply"):
        st.session_state.per_apply = {"per_trucks": int(best["wozidla"]), "per_shifts": int(best["zmian_na_dobe"]),
                                      "per_sat": bool(best["soboty"])}
        st.rerun()


def _blast_calendar(plan: PeriodPlan, sc) -> None:
    """Which blasts fall on which working day, with their feed shares (a preview, before calculating)."""
    rows = []
    for i, d in enumerate(plan.working_days(), start=1):
        items = [(b.pile, b.tonnage_t, b.share) for b in plan.blasts
                 if b.every_days > 0 and i >= b.first_day and (i - b.first_day) % b.every_days == 0]
        items += [(p["id"], p["tonnage_t"], p.get("target_share")) for p in sc.planned_blasts
                  if int(p.get("blast_day") or 1) == i]
        for pid, t, sh in items:
            rows.append({"dzień rob.": i, "data": d.strftime("%d.%m"), "dzień": WEEKDAYS[d.weekday()],
                         "strzał": pid, "ile [t]": round(t), "udział w nadawie [%]": sh})
    st.markdown("**Kalendarz strzałów**")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(400, 38 + 35 * len(rows)))
    else:
        st.caption("brak strzałów w okresie – ustaw je w zakładkach obok")


def yaml_load(text: str) -> dict:
    import yaml

    return yaml.safe_load(text)


def dump_key(sc) -> str:
    from kruszyk.config import dump_scenario

    return dump_scenario(sc)


def _results(df: pd.DataFrame, plan: PeriodPlan, sc) -> None:
    s = summary(df, plan, sc.crushers)
    if not s:
        return
    c = st.columns(5)
    c[0].metric("Wywiezione", f"{s['wywiezione_t']:,.0f} t".replace(",", " "),
                f"{s['realizacja_celu_pct']:.0f} % celu" if s["realizacja_celu_pct"] is not None else None,
                delta_color="normal" if (s["realizacja_celu_pct"] or 100) >= 100 else "inverse")
    c[1].metric("Koszt na tonę (okres)", f"{s['koszt_na_t']:.2f} zł/t" if s["koszt_na_t"] else "–")
    c[2].metric("Koszt łącznie", f"{s['koszt_zl'] / 1000:,.0f} tys. zł".replace(",", " "))
    c[3].metric("Średnio na zmianę", f"{s['srednio_na_zmiane_t']:,.0f} t".replace(",", " "))
    c[4].metric("Zmiany z brakiem urobku", s["zmian_bez_urobku"])

    piles = [c[2:-2] for c in df.columns if c.startswith("z_") and c.endswith("_t") and df[c].sum() > 0]
    meta = dict(df.attrs)
    df = df.copy()
    df.attrs = {}  # blasts / drilling log are shown separately (tables can't carry them)
    df["etykieta"] = df["data"].map(lambda d: d.strftime("%d.%m")) + " " + df["dzien"] + " z" + df["zmiana"].astype(str)
    for w in meta.get("warnings", []):
        st.warning(w)
    r_find, r_blast, r_prod, r_mix, r_day = st.tabs(["💡 Wnioski", f"🧨 Harmonogram strzałów ({len(meta.get('blasts', []))})",
                                                     "📈 Produkcja i usypy", "🧪 Skład nadawy", "📋 Dzień po dniu"])
    with r_blast:
        _blast_schedule(meta, sc)
    with r_find:
        st.markdown("  \n".join(_findings(df, plan, sc, s)))
        gp = st.session_state.get("prod_applied")
        if gp is not None:
            crushed = {k_["kruszarka"]: k_["skruszone_t"] for k_ in s["kruszarki"]}
            made = products_made(crushed, gp.yields, [p["produkt"] for p in gp.products])
            st.markdown("**🎯 Cel produkcyjny – wykonanie w symulacji**")
            st.dataframe(pd.DataFrame([{"produkt": p["produkt"], "cel [t]": p["cel_t"], "wykonanie [t]": round(made[p["produkt"]]),
                                        "wykonanie [%]": round(100 * made[p["produkt"]] / p["cel_t"], 1) if p["cel_t"] else None}
                                       for p in gp.products]), hide_index=True, width="stretch")
        if len(sc.crushers) > 1 or any(c.target_t or c.product for c in sc.crushers):
            st.markdown("**🏭 Kruszarki za okres**")
            st.dataframe(pd.DataFrame(s["kruszarki"]), hide_index=True, width="stretch",
                         column_config={"skruszone_t": "skruszone [t]", "plan_t": "plan [t]",
                                        "wykonanie_pct": "wykonanie planu [%]"})
    with r_prod:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for p in piles:
            fig.add_bar(x=df["etykieta"], y=df[f"z_{p}_t"], name=f"z {p}")
        fig.add_scatter(x=df["etykieta"], y=df["wywiezione_t"].cumsum(), name="narastająco", mode="lines",
                        line=dict(color="#111", width=3), secondary_y=True)
        if plan.target_t:
            fig.add_scatter(x=[df["etykieta"].iloc[0], df["etykieta"].iloc[-1]], y=[plan.target_t / len(df), plan.target_t],
                            name="cel narastająco", mode="lines", line=dict(color="#e63946", dash="dash"), secondary_y=True)
        for _, r in df[df["strzal"] != ""].iterrows():
            fig.add_annotation(x=r["etykieta"], y=0, text="🧨", showarrow=False, yshift=-12)
        fig.update_layout(barmode="stack", height=420, margin=dict(t=30, b=10), legend=dict(orientation="h"),
                          title="Tony na zmianę (słupki) i narastająco (linia)")
        fig.update_yaxes(title_text="t / zmianę", secondary_y=False)
        fig.update_yaxes(title_text="t narastająco", secondary_y=True)
        st.plotly_chart(fig, width="stretch")
        fig2 = go.Figure()
        for p in piles:
            fig2.add_scatter(x=df["etykieta"], y=df[f"zostalo_{p}_t"], name=p, mode="lines+markers", line_shape="hv")
        fig2.update_layout(height=300, margin=dict(t=30, b=10), title="Ile zostało w usypach po każdej zmianie [t]",
                           legend=dict(orientation="h"))
        st.plotly_chart(fig2, width="stretch")
    with r_mix:
        shares = s.get("udzialy_pct", {})
        st.markdown(f"Średnia odchyłka od zadanego składu (ważona tonami): **{s.get('srednia_odchylka_skladu_pp', 0):.1f} pp**. "
                    "Cel na zmianę liczony tylko z usypów, z których wtedy dało się ładować.")
        if shares:
            fig3 = go.Figure(go.Bar(x=list(shares), y=list(shares.values()), text=[f"{v:.0f} %" for v in shares.values()]))
            fig3.update_layout(height=280, margin=dict(t=30, b=10), title="Udział usypów w nadawie za cały okres [%]")
            st.plotly_chart(fig3, width="stretch")
        st.dataframe(df[["etykieta", "wywiezione_t", "sklad", "sklad_cel", "sklad_odchylka_pp"]], hide_index=True,
                     width="stretch", column_config={"etykieta": "zmiana", "wywiezione_t": "wywiezione [t]",
                                                     "sklad": "skład (wykonanie)", "sklad_cel": "skład (cel)",
                                                     "sklad_odchylka_pp": "odchyłka [pp]"})
    with r_day:
        day = df.groupby(["data", "dzien"], as_index=False).agg(
            wywiezione_t=("wywiezione_t", "sum"), koszt_zl=("koszt_zl", "sum"), strzal=("strzal", lambda x: ", ".join(v for v in x if v)),
            przejazdy=("przejazdy_koparek", lambda x: ", ".join(v for v in x if v)),
            wiercenie=("wiercenie", lambda x: "; ".join(v for v in x if v)),
            brak_urobku=("skonczyl_sie_urobek", lambda x: "; ".join(v for v in x if v)),
            kruszarka_bez_nadawy_h=("kruszarka_bez_nadawy_h", "sum"), deszcz=("deszcz", "any"))
        day["koszt_na_t"] = (day["koszt_zl"] / day["wywiezione_t"].where(day["wywiezione_t"] > 0)).round(2)
        day["data"] = day["data"].map(lambda d: d.strftime("%d.%m"))
        st.dataframe(day, hide_index=True, width="stretch",
                     column_config={"wywiezione_t": st.column_config.NumberColumn("wywiezione [t]", format="%.0f"),
                                    "koszt_zl": st.column_config.NumberColumn("koszt [zł]", format="%.0f"),
                                    "koszt_na_t": "zł/t", "strzal": "strzał", "przejazdy": "przejazdy koparek",
                                    "brak_urobku": "koparki bez urobku",
                                    "kruszarka_bez_nadawy_h": "kruszarka bez nadawy [h]"})
        with st.expander("Zmiana po zmianie (wszystkie kolumny)"):
            st.dataframe(df.drop(columns=["etykieta"]), hide_index=True, width="stretch")
        st.download_button("⬇️ Pobierz wyniki (CSV)", df.drop(columns=["etykieta"]).to_csv(index=False).encode("utf-8"),
                           "kruszyk_okres.csv", "text/csv")


def _findings(df: pd.DataFrame, plan: PeriodPlan, sc, s: dict) -> list[str]:
    out = []
    if plan.target_t:
        miss = plan.target_t - s["wywiezione_t"]
        out.append(f"🎯 Cel {plan.target_t:,.0f} t: ".replace(",", " ") +
                   ("**zrobiony** ✅" if miss <= 0 else f"brakuje **{miss:,.0f} t** ({100 - s['realizacja_celu_pct']:.0f} %)".replace(",", " ")))
    empty = df[df["skonczyl_sie_urobek"] != ""]
    if len(empty):
        by_ex: dict[str, list[str]] = {}
        for _, r in empty.iterrows():
            for item in r["skonczyl_sie_urobek"].split(", "):
                by_ex.setdefault(item.split(" ")[0], []).append(f"{r['data'].strftime('%d.%m')} z{r['zmiana']}")
        for ex, when in by_ex.items():
            out.append(f"⛏️ **{ex}** zostaje bez urobku na **{len(when)}** zmianach "
                       f"({', '.join(when[:6])}{' …' if len(when) > 6 else ''}) – strzały za rzadko, za małe albo "
                       "brak wolnego odstrzelonego usypu, na który mogłaby przejechać.")
    mv = df[df["przejazdy_koparek"] != ""]
    for _, r in mv.iterrows():
        out.append(f"🚜 {r['data'].strftime('%d.%m')} z{r['zmiana']}: przejazd {r['przejazdy_koparek']}.")
    if s["kruszarka_bez_nadawy_h"] > 1:
        out.append(f"🏭 Kruszarka stała bez nadawy łącznie **{s['kruszarka_bez_nadawy_h']:.0f} h** w okresie.")
    zero = df[df["wywiezione_t"] <= 0]
    if len(zero):
        out.append(f"🛑 **{len(zero)}** zmian bez żadnego przewozu – wszystkie usypy puste.")
    rainy = int(df["deszcz"].sum())
    if rainy:
        out.append(f"🌧️ Deszcz na {rainy} z {len(df)} zmian (losowo, wg prawdopodobieństwa ze scenariusza).")
    return out or ["✅ Okres bez wyraźnych problemów."]
