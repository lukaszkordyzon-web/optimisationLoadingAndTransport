"""Step "Calibration": the synthetic model tuned to what a real shift measured."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from kruszyk.calibrate import apply_to_raw, calibrate

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
    if cal.measured.get("tonnes_t"):
        st.caption(f"⚖️ Model skalibrowany {cal.date} na zmianie z produkcją **{cal.measured['tonnes_t']:,.0f} t**."
                   .replace(",", " "))
    else:
        st.warning("⚠️ Model nieskalibrowany – czasy, prędkości i przestoje to założenia, więc tony i koszty są "
                   "poglądowe. Wpisz dane z prawdziwej zmiany: moduł **🚚 Zmiana → ② Kalibracja**.")


def step_calibration(raw: dict, sc, set_raw) -> None:
    st.markdown("### ② Kalibracja – model dopasowany do prawdziwej zmiany")
    st.markdown("Dziś model opiera się na **założeniach**. Wpisz, co zmierzyliście na jednej typowej zmianie. Program "
                "dopasuje model tak, żeby „jak było dziś” dawało **te same tony i kursy**. Wtedy optymalizacja RL i "
                "harmonogram liczą na **Waszej** kopalni, a nie na przykładzie. Wymagana jest tylko produkcja na "
                "zmianę; każda kolejna liczba poprawia dokładność.")
    if sc is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia dziś**.")
        return
    old = dict(sc.calibration.measured)
    c = st.columns(4)
    measured = {}
    for i, (key, label, unit, help_) in enumerate(FIELDS):
        default = float(old.get(key) or (sc.truck_count if key == "trucks" else 0.0))
        measured[key] = c[i % 4].number_input(f"{label} [{unit}]", 0.0, 1_000_000.0, default,
                                              1.0 if key in ("trips", "trucks") else 10.0, key=f"cal_{key}",
                                              help=help_, format="%.0f" if key != "load_min" and key != "travel_min"
                                              else "%.1f")
    if st.button("⚖️ Dopasuj model", type="primary", disabled=not measured["tonnes_t"], key="cal_go"):
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
        st.caption("Po zapisaniu warto ponownie nauczyć politykę RL (krok ④) – uczyła się na modelu z założeń.")


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
    if cal.fuel_factor != 1.0:
        lines.append(f"⛽ Zużycie paliwa **×{cal.fuel_factor:.2f}** względem założeń.")
    if cal.kwh_factor != 1.0:
        lines.append(f"🏭 Energia kruszenia **×{cal.kwh_factor:.2f}** względem założeń.")
    if lines:
        st.markdown("**Co zmieniła kalibracja:**  \n" + "  \n".join(lines))
