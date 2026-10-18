"""Step "Calibration": the synthetic model tuned to what a real shift measured."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from kruszyk.calibrate import FIELDS as SHIFT_FIELDS
from kruszyk.calibrate import apply_to_raw, calibrate, calibrate_shifts, shift_stats
from kruszyk.config import RELIABILITY_KINDS, Reliability
from ui import card

FIELDS = [  # key, label, unit, help
    ("tonnes_t", "Produkcja na zmianę", "t", "waga przy kruszarce albo raport dyspozytora – **wymagane**"),
    ("trips", "Liczba kursów", "kursów", "ile wozideł wyładowano w kruszarce – z tego wychodzi faktyczny ładunek kursu"),
    ("trucks", "Wozideł w pracy", "szt.", "ile wozideł naprawdę jeździło na tej zmianie"),
    ("load_min", "Załadunek jednego wozidła", "min", "stoper: od pierwszej do ostatniej łyżki, średnio"),
    ("travel_min", "Jazda na kurs (pełny + pusty)", "min", "stoper: od koparki do kosza i z powrotem, bez czekania"),
    ("fuel_l", "Paliwo wozideł na zmianę", "l", "z tankowań – wszystkie wozidła razem"),
    ("kwh", "Energia kruszarek na zmianę", "kWh", "z licznika – wszystkie kruszarki razem"),
]
COMPARE = [("tonnes_t", "produkcja [t]"), ("trips", "kursy"), ("load_min", "załadunek [min]"),
           ("travel_min", "jazda na kurs [min]"), ("fuel_l", "paliwo [l]"), ("kwh", "energia [kWh]")]


def status_line(sc) -> None:
    """One line for the other screens: is the model tuned to reality?"""
    cal = sc.calibration
    if cal.measured.get("tonnes_t") and cal.n_shifts > 1:
        st.caption(f"⚖️ Model skalibrowany {cal.date} na **{cal.n_shifts} zmianach** (średnio "
                   f"**{cal.measured['tonnes_t']:,.0f} t**, typowy rozrzut ±{cal.measured_sd.get('tonnes_t', 0):,.0f} t)."
                   .replace(",", " "))
    elif cal.measured.get("tonnes_t"):
        st.caption(f"⚖️ Model skalibrowany {cal.date} na zmianie z produkcją **{cal.measured['tonnes_t']:,.0f} t**."
                   .replace(",", " "))
    else:
        st.warning("⚠️ Model nieskalibrowany – czasy, prędkości i przestoje to założenia, więc tony i koszty są "
                   "poglądowe. Wpisz dane z prawdziwej zmiany: moduł **Analiza zmiany → ④ Kalibracja**.")


ONE, MANY = "Jedna typowa zmiana", "📅 Kilka zmian – tabela (zalecane)"


def step_calibration(raw: dict, sc, set_raw) -> None:
    st.markdown("### ④ Kalibracja – model dopasowany do prawdziwych zmian")
    st.markdown("Dziś model opiera się na **założeniach**. Wpisz, co zmierzyliście na prawdziwych zmianach – program "
                "dopasuje model tak, żeby „jak było dziś” dawało **te same tony i kursy**. Wtedy optymalizacja RL i "
                "harmonogram liczą na **Waszej** kopalni, a nie na przykładzie. Najlepiej **tabela zmian** z tygodnia "
                "albo miesiąca: model dopasowuje się do średniej i do tego, jak bardzo zmiany się różnią.")
    if sc is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Stan wyjściowy kopalni**.")
        return
    mode = st.radio("Dane", [MANY, ONE], horizontal=True, key="cal_mode", label_visibility="collapsed")
    if mode == MANY:
        _many_shifts(raw, sc, set_raw)
    else:
        _one_shift(raw, sc, set_raw)


def _one_shift(raw: dict, sc, set_raw) -> None:
    old = dict(sc.calibration.measured)
    with card("cal_in"):
        st.markdown("#### 📋 Pomiary z jednej zmiany")
        c = st.columns(4)
        measured = {}
        for i, (key, label, unit, help_) in enumerate(FIELDS):
            default = float(old.get(key) or (sc.truck_count if key == "trucks" else 0.0))
            measured[key] = c[i % 4].number_input(f"{label} [{unit}]", 0.0, 1_000_000.0, default,
                                                  1.0 if key in ("trips", "trucks") else 10.0, key=f"cal_{key}",
                                                  help=help_, format="%.0f" if key != "load_min" and key != "travel_min"
                                                  else "%.1f")
        go = st.button("⚖️ Dopasuj model", type="primary", disabled=not measured["tonnes_t"], key="cal_go")
    if go:
        bar = st.progress(0.0, "liczę…")
        fitted, before, after = calibrate(sc, measured, progress=lambda f, t: bar.progress(f, t))
        bar.empty()
        st.session_state.cal_result = (fitted, before, after, measured)
    res = st.session_state.get("cal_result")
    if not res:
        if old:
            _explain(sc.calibration)
            if st.button("↩️ Usuń kalibrację (wróć do założeń)", key="cal_reset"):
                raw.pop("calibration", None)
                set_raw(raw)
                st.rerun()
        return
    fitted, before, after, m = res
    rows = []
    for key, label in COMPARE:
        rows.append({"wielkość": label, "pomiar": m.get(key) or None, "model przed": round(before[key], 1),
                     "model po kalibracji": round(after[key], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    gap = abs(after["tonnes_t"] - m["tonnes_t"]) / m["tonnes_t"] * 100
    (st.success if gap <= 3 else st.warning)(
        f"Model po kalibracji: **{after['tonnes_t']:,.0f} t** na zmianę (pomiar {m['tonnes_t']:,.0f} t, różnica "
        f"{gap:.1f} %).".replace(",", " "))
    if after.get("note"):
        st.warning("⚠️ " + after["note"])
    if after.get("limit"):
        st.info(f"🔎 Wąskie gardło po kalibracji: **{after['limit']}**. Kolejka wozideł pod kruszarką: "
                f"{after['queue_crusher_h']:.1f} h, pod koparkami: {after['queue_excavator_h']:.1f} h na zmianę (łącznie).")
    _explain(fitted.calibration, [f.payload_t for f in fitted.trucks], [f.payload_t for f in sc.trucks])
    if st.button("✅ Zapisz kalibrację w kopalni", key="cal_save"):
        apply_to_raw(raw, fitted)
        st.session_state.pop("cal_result")
        set_raw(raw)
        st.rerun()
    if raw.get("rl_weights"):
        st.caption("Po zapisaniu warto ponownie nauczyć politykę RL (krok ⑥) – uczyła się na modelu z założeń.")


SHIFT_COLS = {"data": "data / zmiana", "tonnes_t": "tony", "trips": "kursy", "trucks": "wozideł w pracy",
              "load_min": "załadunek [min]", "travel_min": "jazda na kurs [min]", "fuel_l": "paliwo [l]",
              "kwh": "energia [kWh]"}
LABELS = dict(COMPARE) | {"trucks": "wozidła w pracy"}


def _many_shifts(raw: dict, sc, set_raw) -> None:
    with card("cal_many"):
        st.markdown("#### 📅 Zmiany z tygodnia / miesiąca")
        st.caption("Jeden wiersz = jedna zmiana. Wymagane tylko **tony**; reszta – jeśli macie. Możesz wkleić kolumny z "
                   "Excela (zaznacz komórkę i Ctrl+V). Puste komórki są pomijane. Zmiany z awariami **zostaw** – "
                   "należą do rzeczywistości; model uwzględnia awarie z tabeli awaryjności. Najlepiej 10–30 zmian.")
        key = f"cal_tab_{st.session_state.get('ver', 0)}"
        if f"{key}__base" not in st.session_state:
            st.session_state[f"{key}__base"] = pd.DataFrame(
                [{"data": "", **{k: None for k in SHIFT_FIELDS}} for _ in range(10)])
        df = st.data_editor(st.session_state[f"{key}__base"], key=key, num_rows="dynamic", hide_index=True,
                            width="stretch",
                            column_config={"data": st.column_config.TextColumn(SHIFT_COLS["data"]),
                                           **{k: st.column_config.NumberColumn(SHIFT_COLS[k], min_value=0.0)
                                              for k in SHIFT_FIELDS}})
        stats = shift_stats(df)
        n = stats.get("tonnes_t", {}).get("n", 0)
        if n:
            st.dataframe(pd.DataFrame([{"wielkość": LABELS.get(k, k), "zmian": v["n"], "średnio": round(v["mean"], 1),
                                        "typowy rozrzut ±": round(v["rsd"], 1), "odchylenie std": round(v["sd"], 1),
                                        "P10–P90": f"{v['p10']:.0f}–{v['p90']:.0f}"} for k, v in stats.items()]),
                         hide_index=True, width="stretch")
            st.caption("Typowy rozrzut = rozstęp ćwiartkowy / 1,349 – nie zawyża go jedna zmiana z długą awarią "
                       "(odchylenie standardowe tak).")
        if 0 < n < 10:
            st.warning(f"Tylko {n} zmian – średnia jest niepewna, a rozrzut prawie niewiadomy. Dodaj zmiany (10+).")
        go_ = st.button(f"⚖️ Dopasuj model do {n} zmian", type="primary", disabled=n < 2, key="cal_many_go")
    if go_:
        bar = st.progress(0.0, "liczę…")
        st.session_state.cal_many = calibrate_shifts(sc, df, progress=lambda f, t: bar.progress(min(f, 1.0), t))
        bar.empty()
    res = st.session_state.get("cal_many")
    if not res:
        if sc.calibration.n_shifts > 1:
            _explain(sc.calibration)
        return
    fitted, stats, before, after = res
    rows = []
    for k, label in COMPARE:
        if k in stats:
            m = stats[k]
            rows.append({"wielkość": label, "zmierzone (średnio ± rozrzut)": f"{m['mean']:,.1f} ± {m['rsd']:,.1f}",
                         "model przed": f"{before[k]['mean']:,.1f} ± {before[k]['rsd']:,.1f}",
                         "model po kalibracji": f"{after[k]['mean']:,.1f} ± {after[k]['rsd']:,.1f}",
                         "P10–P90 zmierzone": f"{m['p10']:,.0f}–{m['p90']:,.0f}",
                         "P10–P90 model": f"{after[k]['p10']:,.0f}–{after[k]['p90']:,.0f}"})
    st.dataframe(pd.DataFrame(rows).replace(",", " ", regex=True), hide_index=True, width="stretch")
    m, a = stats["tonnes_t"], after["tonnes_t"]
    gap = 100 * abs(a["mean"] - m["mean"]) / m["mean"]
    (st.success if gap <= 3 else st.warning)(
        f"Model po kalibracji: średnio **{a['mean']:,.0f} t** na zmianę, typowo ±{a['rsd']:,.0f} t (zmierzone "
        f"{m['mean']:,.0f} ± {m['rsd']:,.0f} t; różnica średnich {gap:.1f} %).".replace(",", " "))
    if m["rsd"] and a["rsd"] < 0.6 * m["rsd"]:
        st.info("Zmiany w kopalni różnią się bardziej niż w modelu – sprawdź awaryjność (tabela poniżej) i czy w "
                "tabeli zmian nie ma zmian nietypowych (krótsza zmiana, postój zakładu).")
    elif m["rsd"] and a["rsd"] > 1.6 * m["rsd"]:
        st.info("Model ma większy rozrzut niż kopalnia – możliwe, że awaryjność w tabeli jest zawyżona.")
    fig = go.Figure()
    fig.add_box(y=[float(x) for x in df["tonnes_t"] if x == x and x], name="zmierzone", marker_color="#2f4156",
                boxpoints="all", jitter=0.4)
    fig.add_box(y=a["values"], name="model po kalibracji", marker_color="#c8102e", boxpoints="all", jitter=0.4)
    fig.update_layout(height=300, margin=dict(t=30, b=10), title="Tony na zmianę: zmierzone i z modelu",
                      yaxis_title="t / zmianę", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    _explain(fitted.calibration, [f.payload_t for f in fitted.trucks], [f.payload_t for f in sc.trucks])
    if st.button("✅ Zapisz kalibrację w kopalni", key="cal_many_save"):
        apply_to_raw(raw, fitted)
        st.session_state.pop("cal_many")
        set_raw(raw)
        st.rerun()


def _explain(cal, payload_after=None, payload_before=None) -> None:
    lines = []
    if payload_after and payload_before and payload_after != payload_before:
        lines.append(f"🚚 Faktyczny ładunek kursu: **{', '.join(f'{p:g}' for p in payload_after)} t** "
                     f"(w założeniach {', '.join(f'{p:g}' for p in payload_before)} t).")
    if cal.load_factor != 1.0:
        lines.append(f"⛏️ Załadunek trwa **×{cal.load_factor:.2f}** tego, co w założeniach.")
    if cal.travel_factor != 1.0:
        lines.append(f"🛣️ Jazda trwa **×{cal.travel_factor:.2f}** tego, co w założeniach.")
    if cal.delay_min_per_trip:
        lines.append(f"⏸️ **{cal.delay_min_per_trip:.1f} min na kurs** to przestoje, których model nie zna (czekanie na "
                     "polecenie, sprzątanie frontu, rozmowy, nieewidencjonowane postoje). W animacji: „inne przestoje”.")
        if cal.delay_min_per_trip > 10:
            lines.append("🔎 To dużo – sprawdź, czy liczba wozideł w pracy, przerwy i czasy jazdy są wpisane zgodnie "
                         "z rzeczywistością. Jeśli tak, ta strata czasu to największa rezerwa kopalni.")
    if cal.shift_sd:
        lines.append(f"📅 Zmiany różnią się między sobą bardziej, niż wyjaśniają awarie i pogoda: każda zmiana dostaje "
                     f"własne tempo jazdy i załadunku, typowo ±{100 * cal.shift_sd:.0f} % (załoga, urobek, organizacja dnia).")
    if cal.fuel_factor != 1.0:
        lines.append(f"⛽ Zużycie paliwa **×{cal.fuel_factor:.2f}** względem założeń.")
    if cal.kwh_factor != 1.0:
        lines.append(f"🏭 Energia kruszenia **×{cal.kwh_factor:.2f}** względem założeń.")
    if lines:
        st.markdown("**Co zmieniła kalibracja:**  \n" + "  \n".join(lines))


def reliability_editor(raw: dict, sc, key: str) -> None:
    """Breakdowns per machine: failures per month and their total hours (from the maintenance records)."""
    with card(f"rel_{key}"):
        _reliability_editor(raw, sc, key)


def _reliability_editor(raw: dict, sc, key: str) -> None:
    st.markdown("#### 🔧 Machine breakdowns")
    st.caption("For every machine: **breakdowns per month** and **total hours** they lasted – the average of the last "
               "3–6 months from the repair log. Only stops longer than about 15 min, **no planned services** "
               "(shorter stops are in the calibration). The length of one repair is random – usually short, sometimes "
               "longer than a shift; then the breakdown carries over to the next shifts.")
    rel = sc.reliability
    stored = (raw.get("reliability") or {}).get("machines") or {}
    from kruszyk.config import truck_units

    # (key, kind, name, where its value comes from when it has none of its own)
    rows = [(c.id, "crusher", f"Crusher {c.id}", c.id) for c in sc.crushers]
    rows += [(e.id, "excavator", f"Excavator {e.id}", e.id) for e in sc.excavators]
    rows += [(tid, "truck", f"Truck {tid} · {f.model}", tid if tid in stored else f"truck:{f.model}")
             for tid, f in truck_units(sc.trucks)]
    rigs = [r for r in raw.get("drills") or [] if r.get("id")]
    rows += [(r["id"], "drill", f"Drill rig {r['id']}", r["id"] if r["id"] in stored else "drill") for r in rigs] \
        or [("drill", "drill", "Drill rig (each)", "drill")]
    c = st.columns([1, 1, 2])
    enabled = c[0].toggle("Include breakdowns", rel.enabled, key=f"{key}_on")
    ref = c[1].number_input("Shifts per day in that period", 1.0, 3.0, float(rel.ref_shifts_per_day), 1.0,
                            key=f"{key}_ref", format="%.0f",
                            help="how many shifts per day the machines worked in the months the data come from – "
                                 "breakdowns are converted to working hours, so more shifts = more breakdowns")
    sig = repr(sorted(stored.items())) + repr([r[0] for r in rows]) + str(st.session_state.get("ver", 0))
    ed_key = f"{key}_tab_{abs(hash(sig)) % 10**8}"
    if f"{ed_key}__base" not in st.session_state:
        st.session_state[f"{ed_key}__base"] = pd.DataFrame([
            {"klucz": k_, "urzadzenie": name, "awarii": rel.values(src, kind)[0], "godzin": rel.values(src, kind)[1]}
            for k_, kind, name, src in rows])
    df = st.data_editor(st.session_state[f"{ed_key}__base"], key=ed_key, hide_index=True, width="stretch",
                        disabled=["klucz", "urzadzenie"], column_order=["urzadzenie", "awarii", "godzin"],
                        column_config={"urzadzenie": "machine",
                                       "awarii": st.column_config.NumberColumn("breakdowns / month", min_value=0.0, step=1.0),
                                       "godzin": st.column_config.NumberColumn("total [h / month]", min_value=0.0,
                                                                               step=1.0)})
    machines, assumed, lines = dict(stored), [], []  # machines not listed here (e.g. rigs) keep their data
    probe = Reliability(enabled=True, ref_shifts_per_day=ref, ref_days_per_month=rel.ref_days_per_month)
    for (k_, kind, name, src), (_, r) in zip(rows, df.iterrows()):
        n, h = float(r["awarii"] or 0), float(r["godzin"] or 0)
        d_n, d_h, d_assumed = rel.values(src, kind)
        if k_ in stored or (n, h) != (d_n, d_h) or not d_assumed:
            machines[k_] = {"count": n, "hours": h}
        else:
            assumed.append(name)
        probe.machines[k_] = {"count": n, "hours": h}
        mtbf, mttr = probe.rates(k_, kind, sc.shift_hours)
        if mtbf < float("inf"):
            lines.append(f"{name}: repair on average **{mttr:.1f} h**, a breakdown about every **{mtbf / sc.shift_hours:.0f} "
                         "working shifts**")
    raw["reliability"] = {"enabled": bool(enabled), "ref_shifts_per_day": float(ref),
                          "ref_days_per_month": float(rel.ref_days_per_month), "machines": machines}
    with st.expander("What this means for the model"):
        st.markdown("  \n".join(lines) or "no breakdowns")
    if assumed:
        st.caption("Assumptions (default values, to be corrected): " + ", ".join(assumed) + ".")
