"""Step "Calibration": the synthetic model tuned to what a real shift measured."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from kruszyk.calibrate import apply_to_raw, calibrate
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
    if cal.measured.get("tonnes_t"):
        st.caption(f"⚖️ Model skalibrowany {cal.date} na zmianie z produkcją **{cal.measured['tonnes_t']:,.0f} t**."
                   .replace(",", " "))
    else:
        st.warning("⚠️ Model nieskalibrowany – czasy, prędkości i przestoje to założenia, więc tony i koszty są "
                   "poglądowe. Wpisz dane z prawdziwej zmiany: moduł **Analiza zmiany → ④ Kalibracja**.")


def step_calibration(raw: dict, sc, set_raw) -> None:
    st.markdown("### ④ Kalibracja – model dopasowany do prawdziwej zmiany")
    st.markdown("Dziś model opiera się na **założeniach**. Wpisz, co zmierzyliście na jednej typowej zmianie. Program "
                "dopasuje model tak, żeby „jak było dziś” dawało **te same tony i kursy**. Wtedy optymalizacja RL i "
                "harmonogram liczą na **Waszej** kopalni, a nie na przykładzie. Wymagana jest tylko produkcja na "
                "zmianę; każda kolejna liczba poprawia dokładność.")
    if sc is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Stan wyjściowy kopalni**.")
        return
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


def reliability_editor(raw: dict, sc, key: str) -> None:
    """Breakdowns per machine: failures per month and their total hours (from the maintenance records)."""
    with card(f"rel_{key}"):
        _reliability_editor(raw, sc, key)


def _reliability_editor(raw: dict, sc, key: str) -> None:
    st.markdown("#### 🔧 Awaryjność urządzeń")
    st.caption("Dla każdego urządzenia: **ile awarii w miesiącu** i **ile godzin łącznie** trwały – średnio z ostatnich "
               "3–6 miesięcy, z dziennika napraw. Tylko postoje dłuższe niż ok. 15 min, **bez przeglądów planowych** "
               "(krótsze postoje są w kalibracji). Czas jednej naprawy jest losowany – zwykle krótki, czasem dłuższy "
               "niż zmiana; wtedy awaria przechodzi na kolejne zmiany.")
    rel = sc.reliability
    stored = (raw.get("reliability") or {}).get("machines") or {}
    from kruszyk.config import truck_units

    # (key, kind, name, where its value comes from when it has none of its own)
    rows = [(c.id, "crusher", f"Kruszarka {c.id}", c.id) for c in sc.crushers]
    rows += [(e.id, "excavator", f"Koparka {e.id}", e.id) for e in sc.excavators]
    rows += [(tid, "truck", f"Wozidło {tid} · {f.model}", tid if tid in stored else f"truck:{f.model}")
             for tid, f in truck_units(sc.trucks)]
    rigs = [r for r in raw.get("drills") or [] if r.get("id")]
    rows += [(r["id"], "drill", f"Wiertnica {r['id']}", r["id"] if r["id"] in stored else "drill") for r in rigs] \
        or [("drill", "drill", "Wiertnica (każda)", "drill")]
    c = st.columns([1, 1, 2])
    enabled = c[0].toggle("Uwzględniaj awarie", rel.enabled, key=f"{key}_on")
    ref = c[1].number_input("Zmian na dobę w tym okresie", 1.0, 3.0, float(rel.ref_shifts_per_day), 1.0,
                            key=f"{key}_ref", format="%.0f",
                            help="przy ilu zmianach na dobę pracowały maszyny w miesiącach, z których są dane – "
                                 "program przelicza awarie na godziny pracy, więc więcej zmian = więcej awarii")
    sig = repr(sorted(stored.items())) + repr([r[0] for r in rows]) + str(st.session_state.get("ver", 0))
    ed_key = f"{key}_tab_{abs(hash(sig)) % 10**8}"
    if f"{ed_key}__base" not in st.session_state:
        st.session_state[f"{ed_key}__base"] = pd.DataFrame([
            {"klucz": k_, "urzadzenie": name, "awarii": rel.values(src, kind)[0], "godzin": rel.values(src, kind)[1]}
            for k_, kind, name, src in rows])
    df = st.data_editor(st.session_state[f"{ed_key}__base"], key=ed_key, hide_index=True, width="stretch",
                        disabled=["klucz", "urzadzenie"], column_order=["urzadzenie", "awarii", "godzin"],
                        column_config={"urzadzenie": "urządzenie",
                                       "awarii": st.column_config.NumberColumn("awarii / miesiąc", min_value=0.0, step=1.0),
                                       "godzin": st.column_config.NumberColumn("łącznie [h / miesiąc]", min_value=0.0,
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
            lines.append(f"{name}: naprawa średnio **{mttr:.1f} h**, awaria co ok. **{mtbf / sc.shift_hours:.0f} zmian** "
                         "pracy")
    raw["reliability"] = {"enabled": bool(enabled), "ref_shifts_per_day": float(ref),
                          "ref_days_per_month": float(rel.ref_days_per_month), "machines": machines}
    with st.expander("Co to znaczy dla modelu"):
        st.markdown("  \n".join(lines) or "brak awarii")
    if assumed:
        st.caption("Założenia (wartości domyślne, do poprawienia): " + ", ".join(assumed) + ".")
