"""Step ⑥ with the driving signals: the trucks go where the foreman sends them (as today); the system only tells
the drivers how fast to drive and when to wait with the engine off. Result: the same month with and without."""
from __future__ import annotations

import datetime as dt
import hashlib

import plotly.graph_objects as go
import streamlit as st
import yaml

from kruszyk.config import scenario_from_dict
from kruszyk.period import PeriodPlan, run_period
from kruszyk.rl import explain_signals, today, train_signals, with_signals
from ui import card

QUICK = {"iterations": 2, "population": 4, "episodes": 1, "days": 3}  # for tests (session flag "signals_quick")


def month_plan() -> PeriodPlan:
    """The month as set in production planning (or 4 weeks, 2 shifts a day, Monday–Friday)."""
    ss = st.session_state
    start = ss.get("per_start") or (dt.date.today() + dt.timedelta(days=7 - dt.date.today().weekday()))
    days = int(QUICK["days"] if ss.get("signals_quick") else ss.get("per_days") or 28)
    return PeriodPlan(start=start, days=days, shifts_per_day=int(ss.get("per_shifts") or 2),
                      saturdays=bool(ss.get("per_sat", False)))


def step_signals(base_raw: dict, base_yaml: str, seed: int) -> None:
    sc = scenario_from_dict(yaml.safe_load(base_yaml))
    plan = month_plan()
    n = len(plan.shifts())
    st.caption(f"Miesiąc: od {plan.start:%d.%m}, {n} zmian ({plan.shifts_per_day} na dobę"
               f"{', z sobotami' if plan.saturdays else ''}) – jak w Planowaniu produkcji.")
    key = hashlib.md5(repr((base_yaml, plan, seed)).encode()).hexdigest()
    res = st.session_state.get("signals")
    if st.button("🚀 Naucz sygnały i porównaj miesiąc", type="primary", key="signals_go"):
        size = QUICK if st.session_state.get("signals_quick") else {}
        bar = st.progress(0.0, "Miesiąc jak dziś…")
        states: list = []
        df_today = run_period(today(sc), plan, seed=seed, states=states,
                              progress=lambda i, k: bar.progress(0.3 * i / k, f"Miesiąc jak dziś: zmiana {i} z {k}"))
        r = train_signals(states, seed=seed, **{k: v for k, v in size.items() if k != "days"},
                          callback=lambda i, k, row: bar.progress(0.3 + 0.4 * i / k, f"Nauka sygnałów: runda {i} z {k}"))
        df_sig = run_period(with_signals(sc, r.params), plan, seed=seed,
                            progress=lambda i, k: bar.progress(0.7 + 0.3 * i / k, f"Miesiąc z sygnałami: zmiana {i} z {k}"))
        bar.empty()
        res = {"key": key, "params": r.params, "setup": r.setup, "today": df_today, "signals": df_sig}
        st.session_state.signals = res
        for k_, v_ in r.params.items():  # stored in the mine: the schedule compares with them too
            (base_raw.setdefault("advisor", {}) if k_ in ("arrival_margin_s", "max_slowdown", "crusher_pacing")
             else base_raw.setdefault("excavator_policy", {}))[k_] = v_
        base_raw["signals"] = True
    if not res:
        st.caption("Nauka i dwa miesiące liczą się ok. 1–2 min.")
        return
    if res["key"] != key:
        st.warning("Kopalnia albo miesiąc zmieniły się od ostatniego liczenia – kliknij przycisk ponownie.")
    a, b = res["today"], res["signals"]
    t0, t1 = float(a["wywiezione_t"].sum()), float(b["wywiezione_t"].sum())
    c0, c1 = float(a["koszt_zl"].sum()), float(b["koszt_zl"].sum())
    f0, f1 = float(a["paliwo_l"].sum()), float(b["paliwo_l"].sum())
    q0, q1 = float(a["kolejki_h"].sum()), float(b["kolejki_h"].sum())
    z0, z1 = c0 / t0 if t0 else 0.0, c1 / t1 if t1 else 0.0

    def num(x: float) -> str:
        return f"{x:,.0f}".replace(",", " ")
    with card("signals_month"):
        st.markdown("#### 📅 Ten sam miesiąc: jak dziś i z sygnałami jazdy")
        m = st.columns(5)
        m[0].metric("Wywiezione", f"{num(t1)} t", f"{num(t1 - t0)} t ({100 * (t1 / t0 - 1):+.1f} %)" if t0 else None)
        m[1].metric("Koszt tony", f"{z1:.2f} zł/t", f"{z1 - z0:+.2f} zł/t", delta_color="inverse")
        m[2].metric("Oszczędność", f"{num((z0 - z1) * t1)} zł", help="(koszt tony dziś − z sygnałami) × tony z sygnałami")
        m[3].metric("Paliwo", f"{num(f1)} l", f"{100 * (f1 / f0 - 1):+.0f} %" if f0 else None, delta_color="inverse")
        m[4].metric("Stanie w kolejkach", f"{num(q1)} h", f"{num(q1 - q0)} h", delta_color="inverse")
        fig = go.Figure()
        for name, d, color in (("jak dziś", a, "#9aa0a6"), ("z sygnałami", b, "#c8102e")):
            cum = d.groupby("data")["koszt_zl"].sum().cumsum() / 1000
            fig.add_scatter(x=list(cum.index), y=cum.values, mode="lines", name=name, line=dict(color=color, width=3))
        fig.update_layout(height=260, margin=dict(t=30, b=10), title="Koszt narastająco [tys. zł]",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Ten sam plan, strzały, pogoda i awarie. Wozidła jeżdżą tam, gdzie wysyła sztygar – zmienia się "
                   "tylko sposób jazdy.")
    with card("signals_what"):
        st.markdown("#### 📱 Sygnały dla kierowców")
        st.markdown("  \n".join(explain_signals(res["params"])))
        s = res["setup"]
        st.caption(f"Nauczone na {s['shifts']} zmianach symulacji, w sytuacjach z {s['states']} zmian miesiąca "
                   "jak dziś (pełne usypy, końcówki, koparka po przejeździe, deszcz, awarie).")
