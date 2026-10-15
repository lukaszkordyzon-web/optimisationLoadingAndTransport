"""③ Symulacja and ④ Wynik: the approved schedule driven twice – today (the foreman's orders, no signals) and with
the driving signals learned by RL (the same orders, other driving). Same shifts, weather and breakdowns."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from kruszyk.anim import split_html
from kruszyk.compare import sign_threshold
from kruszyk.period import run_period
from kruszyk.rl import explain_signals, today, train_signals, with_signals
from kruszyk.sim import simulate
from ui import card

from . import state
from .plan import labels, period_plan

SEED = 0
QUICK = {"iterations": 2, "population": 4, "episodes": 1}  # tests: session flag "k2_quick"


def _approved():
    appr = st.session_state.get("approved")
    if not appr:
        st.info("Najpierw ułóż i zatwierdź harmonogram w kroku **② Harmonogram**.")
        return None
    return appr


def _month_key(appr) -> str:
    return hashlib.md5(repr((state.raw_yaml(), appr["key"])).encode()).hexdigest()


def month(appr) -> dict | None:
    """Both months (today / with the signals) on the frozen schedule; the signals are learned on the shifts of the
    month worked as today. Computed on a click, kept until the mine or the schedule changes."""
    key = _month_key(appr)
    res = st.session_state.get("month")
    if res and res["key"] == key:
        return res
    st.caption("Najpierw program liczy miesiąc jak dziś, uczy się na nim sygnałów jazdy i liczy ten sam miesiąc z "
               "sygnałami – zwykle 1–3 min.")
    if not st.button("🚀 Policz miesiąc i naucz sygnały", type="primary", key="k2_month_go"):
        return None
    sc = state.scenario()
    plan = period_plan(appr["plan"])
    plan.frozen = appr["frozen"]
    bar = st.progress(0.0, "Miesiąc jak dziś…")
    states, kpi0 = [], []
    df0 = run_period(today(sc), plan, seed=SEED, states=states, kpis=kpi0,
                     progress=lambda i, k: bar.progress(0.35 * i / k, f"Miesiąc jak dziś: zmiana {i} z {k}"))
    size = QUICK if st.session_state.get("k2_quick") else {}
    r = train_signals(states, seed=SEED, **size,
                      callback=lambda i, k, row: bar.progress(0.35 + 0.3 * i / k, f"Nauka sygnałów: runda {i} z {k}"))
    kpi1 = []
    df1 = run_period(with_signals(sc, r.params), plan, seed=SEED, kpis=kpi1,
                     progress=lambda i, k: bar.progress(0.65 + 0.35 * i / k, f"Miesiąc z sygnałami: zmiana {i} z {k}"))
    bar.empty()
    res = {"key": key, "today": df0, "signals": df1, "params": r.params, "setup": r.setup, "states": states,
           "labels": labels(plan)}
    st.session_state.month = res
    st.session_state.pop("split", None)
    return res


@st.cache_data(show_spinner=False, max_entries=6)
def _split(state_yaml: str, params: tuple, seed: int) -> str:
    from kruszyk.config import scenario_from_dict
    import yaml

    sc = scenario_from_dict(yaml.safe_load(state_yaml))
    a = simulate(today(sc), seed=seed)
    b = simulate(with_signals(sc, dict(params)), seed=seed)
    return split_html(a, b, "Jak dziś", "Z sygnałami RL")


def step_simulation() -> None:
    st.markdown("## Symulacja: jak dziś i z sygnałami RL")
    appr = _approved()
    if not appr:
        return
    res = month(appr)
    if not res:
        return
    from kruszyk.config import dump_scenario

    a, b, lab = res["today"], res["signals"], res["labels"]
    opts = list(range(len(a)))

    def fmt(i: int) -> str:
        d = b["koszt_na_t"].iloc[i] - a["koszt_na_t"].iloc[i]
        return f"{lab[i]} · {a['wywiezione_t'].iloc[i]:,.0f} t · {d:+.2f} zł/t z RL".replace(",", " ")
    with card("k2_shiftpick"):
        c = st.columns([4, 1])
        i = c[0].selectbox("Zmiana do obejrzenia", opts, format_func=fmt, key="k2_shift")
        c[1].metric("Zmian w miesiącu", len(a))
    st_yaml = dump_scenario(res["states"][i])
    page = _split(st_yaml, tuple(sorted(res["params"].items())), SEED + 7919 * i)  # the seed of that shift in the month
    components.html(page, height=900, scrolling=True)
    c = st.columns([3, 1])
    c[0].caption("Lewa i prawa strona: ta sama zmiana z miesiąca – te same zlecenia sztygara, pogoda i awarie. "
                 "Po prawej kierowcy dostają sygnały jazdy (komunikaty pod mapą).")
    if c[1].button("Cały miesiąc →", key="k2_to_result", width="stretch"):
        st.session_state.nav_next = "④ Wynik"
        st.rerun()


def step_result() -> None:
    st.markdown("## Wynik miesiąca")
    appr = _approved()
    if not appr:
        return
    res = month(appr)
    if not res:
        return
    a, b = res["today"], res["signals"]
    t0, t1 = float(a["wywiezione_t"].sum()), float(b["wywiezione_t"].sum())
    c0, c1 = float(a["koszt_zl"].sum()), float(b["koszt_zl"].sum())
    f0, f1 = float(a["paliwo_l"].sum()), float(b["paliwo_l"].sum())
    q0, q1 = float(a["kolejki_h"].sum()), float(b["kolejki_h"].sum())
    z0, z1 = (c0 / t0 if t0 else 0.0), (c1 / t1 if t1 else 0.0)
    n = len(a)

    def num(x: float) -> str:
        return f"{x:,.0f}".replace(",", " ")
    st.caption(f"{res['labels'][0]} – {res['labels'][-1]} · {n} zmian · ten sam zatwierdzony harmonogram, strzały, "
               "pogoda i awarie. Wozidła jadą tam, gdzie wysyła sztygar – zmienia się tylko sposób jazdy.")
    m = st.columns(5)
    m[0].metric("Oszczędność w miesiącu", f"{num((z0 - z1) * t1)} zł", help="(koszt tony dziś − z sygnałami) × tony z sygnałami")
    m[1].metric("Koszt tony", f"{z1:.2f} zł/t", f"{z1 - z0:+.2f} zł/t vs dziś", delta_color="inverse")
    m[2].metric("Paliwo", f"{num(f1)} l", f"{100 * (f1 / f0 - 1):+.1f} %" if f0 else None, delta_color="inverse")
    m[3].metric("Wywiezione", f"{num(t1)} t", f"{100 * (t1 / t0 - 1):+.1f} %" if t0 else None)
    m[4].metric("Stanie w kolejkach", f"{num(q1)} h", f"{num(q1 - q0)} h", delta_color="inverse")

    c1_, c2_ = st.columns([3, 2])
    with c1_, card("k2_chart"):
        fig = go.Figure()
        for name, d, color in (("jak dziś", a, "#8A93A0"), ("z sygnałami RL", b, "#B3122A")):
            cum = d["koszt_zl"].cumsum() / 1000
            fig.add_scatter(x=res["labels"], y=cum, mode="lines", name=name, line=dict(color=color, width=3))
        fig.update_layout(height=320, margin=dict(t=40, b=10), title="Koszt narastająco [tys. zł]",
                          legend=dict(orientation="h", y=-0.25), xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width="stretch")
        week = pd.DataFrame({"tydzień": [pd.Timestamp(d).strftime("%G-W%V") for d in a["data"]],
                             "t_dzis": a["wywiezione_t"], "t_rl": b["wywiezione_t"], "k_dzis": a["koszt_zl"],
                             "k_rl": b["koszt_zl"], "taniej": (b["koszt_na_t"] < a["koszt_na_t"]).astype(int)})
        g = week.groupby("tydzień", sort=False).sum(numeric_only=True)
        g["zl_t_dzis"] = g["k_dzis"] / g["t_dzis"]
        g["zl_t_rl"] = g["k_rl"] / g["t_rl"]
        g["zmian"] = week.groupby("tydzień", sort=False).size()
        st.dataframe(g[["t_dzis", "t_rl", "zl_t_dzis", "zl_t_rl", "taniej", "zmian"]].round(2), width="stretch",
                     column_config={"t_dzis": "tony dziś", "t_rl": "tony z RL", "zl_t_dzis": "zł/t dziś",
                                    "zl_t_rl": "zł/t z RL", "taniej": "zmian taniej", "zmian": "zmian"})
    with c2_:
        with card("k2_signals"):
            st.markdown("#### 📱 Sygnały dla kierowców")
            st.markdown("  \n".join(explain_signals(res["params"])))
            s = res["setup"]
            st.caption(f"Nauczone na {s['shifts']} zmianach symulacji, w sytuacjach z {s['states']} zmian tego "
                       "miesiąca jak dziś (pełne usypy, końcówki, przejazdy koparek, deszcz, awarie).")
        with card("k2_sure"):
            wins = int((b["koszt_na_t"] < a["koszt_na_t"]).sum())
            need = sign_threshold(n)
            st.markdown("#### Pewność wyniku")
            verdict = ("różnica pewna" if wins >= need else "taniej z RL za rzadko, by mówić o pewnej różnicy"
                       if wins > n - need else "z RL drożej")
            st.markdown(f"Taniej z sygnałami na **{wins} z {n}** zmian – {verdict} (próg: {need} z {n}).")
        out = pd.DataFrame({"zmiana": res["labels"], "t_dzis": a["wywiezione_t"], "t_rl": b["wywiezione_t"],
                            "zl_t_dzis": a["koszt_na_t"], "zl_t_rl": b["koszt_na_t"], "paliwo_dzis": a["paliwo_l"],
                            "paliwo_rl": b["paliwo_l"]})
        st.download_button("⬇️ Pobierz wyniki zmian (CSV)", out.to_csv(index=False).encode("utf-8"), "wynik_miesiaca.csv",
                           "text/csv", key="k2_csv", width="stretch")
