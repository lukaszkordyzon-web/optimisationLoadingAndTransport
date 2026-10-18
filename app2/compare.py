"""③ Symulacja and ④ Wynik: the approved schedule driven twice – today (the foreman's orders, no signals) and with
the driving signals learned by RL (the same orders, other driving). Same shifts, weather and breakdowns."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from kruszyk.anim import split_html
from kruszyk.compare import sign_threshold
from kruszyk.period import run_period
from kruszyk.rl import today, train_signals, with_signals
from kruszyk.sim import simulate
from . import state, theme
from .plan import labels, period_plan
from .theme import card

SEED = 0
QUICK = {"iterations": 2, "population": 4, "episodes": 1}  # tests: session flag "k2_quick"


def _approved():
    appr = st.session_state.get("approved")
    if not appr:
        st.info("Najpierw ułóż i zatwierdź harmonogram w kroku **② Harmonogram**.")
        return None
    return appr


COSTS = [("wozidla_kierowcy", "Kierowcy"), ("paliwo", "Paliwo wozideł"), ("wozidla_serwis", "Serwis wozideł – od mth"),
         ("wozidla_amortyzacja", "Odtworzenie wozideł – od mth"), ("koparki_operatorzy", "Operatorzy koparek"),
         ("koparki_serwis", "Serwis koparek – od mth"), ("koparki_amortyzacja", "Odtworzenie koparek – od mth"),
         ("kruszarka", "Kruszarka – stały"), ("energia", "Energia kruszarki – od tony")]


def _parts() -> tuple:
    """The cost parts the planner counts (ticks); all when none is ticked."""
    got = tuple(k for k, _ in COSTS if st.session_state.get(f"k2_cost_{k}", True))
    return got or tuple(k for k, _ in COSTS)


def _dynamic() -> bool:
    return bool(st.session_state.get("k2_dyn", False))


def _month_key(appr) -> str:
    return hashlib.md5(repr((state.raw_yaml(), appr["key"], _weight(), _parts(), _dynamic())).encode()).hexdigest()


def _cost(df: pd.DataFrame, parts: tuple) -> pd.Series:
    """Every shift's cost counting only the chosen parts."""
    cols = [f"k_{k}" for k in parts if f"k_{k}" in df]
    return df[cols].sum(axis=1) if cols else df["koszt_zl"]


def _weight() -> float:
    """The RL goal from the slider: 0 = the cheapest tonne … 1 = the most tonnes (within the schedule)."""
    return int(st.session_state.get("k2_rl_w", 50)) / 100


GOAL_WORDS = {0: "najtańsza tona", 50: "pół na pół", 100: "najwięcej ton"}


def goal_label(w: float) -> str:
    v = round(100 * w)
    return GOAL_WORDS.get(v) or (f"bliżej najtańszej tony ({v})" if v < 50 else f"bliżej najwięcej ton ({v})")


def goal_slider() -> None:
    with card("rlgoal"):
        c = st.columns([1.2, 4, 1.2], vertical_alignment="center")
        c[0].markdown('<div class="k2-h4" style="margin:0">Cel nauki RL</div><div class="k2-note">najtańsza tona</div>',
                      unsafe_allow_html=True)
        c[1].slider("Cel nauki RL", 0, 100, 50, 10, key="k2_rl_w", label_visibility="collapsed",
                    help="Lewo: RL może odpuścić trochę ton, jeśli tona wyjdzie wyraźnie taniej. Środek: tańsza tona, ale "
                         "bez utraty ton. Prawo: każda tona się liczy – w granicach harmonogramu i wydajności koparek.")
        c[2].markdown('<div class="k2-note" style="text-align:right">najwięcej ton</div>', unsafe_allow_html=True)
        o = st.columns([2, 3], vertical_alignment="center")
        o[0].toggle("Dynamiczny przydział wozideł (z RL)", key="k2_dyn",
                    help="Jak dziś: wozidło wraca do koparki z harmonogramu. Włączone (tylko w przebiegu z RL): po "
                         "wyładunku jedzie tam, gdzie zostanie najszybciej załadowane – do koparek, które mają w tej "
                         "zmianie tony. Koparka nie stoi bez wozideł, gdy inne czekają w kolejce.")
        with o[1].popover(f"Koszty w wyliczeniu · {len(_parts())} z {len(COSTS)}", width="stretch"):
            theme.note("Zaznaczone koszty liczą się w koszcie tony, oszczędności i w nauce RL. Koszt całkowity "
                       "(wszystkie składniki) jest zawsze na końcu tabeli w ④ Wynik. Wiercenie i strzelanie nie są "
                       "liczone – jazda ich nie zmienia.")
            cols = st.columns(3)
            for j, (k, label) in enumerate(COSTS):
                cols[j % 3].checkbox(label, value=True, key=f"k2_cost_{k}")


def month(appr) -> dict | None:
    """Both months (today / with the signals) on the frozen schedule; the signals are learned on the shifts of the
    month worked as today. Computed on a click, kept until the mine or the schedule changes."""
    goal_slider()
    key = _month_key(appr)
    res = st.session_state.get("month")
    if res and res["key"] == key:
        return res
    with card("monthgo"):
        theme.h4("Policz miesiąc")
        theme.note("Program liczy miesiąc jak dziś według zatwierdzonego harmonogramu, uczy się na nim sygnałów jazdy "
                   f"i liczy ten sam miesiąc z sygnałami – zwykle 1–3 min. Cel nauki: <b>{goal_label(_weight())}</b>.")
        go_ = st.button("Policz miesiąc i naucz sygnały", type="primary", key="k2_month_go")
    if not go_:
        return None
    from kruszyk.config import cycle_costs, scenario_from_dict
    import yaml

    sc = cycle_costs(scenario_from_dict(yaml.safe_load(appr["raw"])))  # the mine as the plan puts it: piles in the areas
    plan = period_plan(appr["plan"])
    plan.frozen = appr["frozen"]
    bar = st.progress(0.0, "Miesiąc jak dziś…")
    states, kpi0 = [], []
    df0 = run_period(today(sc), plan, seed=SEED, states=states, kpis=kpi0,
                     progress=lambda i, k: bar.progress(0.35 * i / k, f"Miesiąc jak dziś: zmiana {i} z {k}"))
    size = QUICK if st.session_state.get("k2_quick") else {}
    parts, dyn = _parts(), _dynamic()
    r = train_signals(states, seed=SEED, **size, weight=_weight(), dynamic=dyn, parts=parts,
                      callback=lambda i, k, row: bar.progress(0.35 + 0.3 * i / k, f"Nauka sygnałów: runda {i} z {k}"))
    kpi1 = []
    df1 = run_period(with_signals(sc, r.params, dyn), plan, seed=SEED, kpis=kpi1,
                     progress=lambda i, k: bar.progress(0.65 + 0.35 * i / k, f"Miesiąc z sygnałami: zmiana {i} z {k}"))
    bar.empty()
    res = {"key": key, "today": df0, "signals": df1, "params": r.params, "setup": r.setup, "states": states,
           "labels": labels(plan), "parts": parts, "dynamic": dyn}
    st.session_state.month = res
    st.session_state.pop("split", None)
    return res


def _in_shift(r, parts: tuple, step_min: float = 5.0) -> pd.DataFrame:
    """One shift minute by minute: tonnes, cost per tonne (the chosen costs) and queueing, all cumulative."""
    from kruszyk.rl import selected_cost

    tl = r.timeline
    end = float(tl["t_s"].iloc[-1]) if len(tl) else 0.0
    grid = np.arange(0.0, end + 1.0, step_min * 60)
    # tonnes grow evenly between dumps (a dump is a step; as a ramp the cost per tonne does not saw up and down)
    ld = r.loads[r.loads["dumped_at_s"].notna()].sort_values("dumped_at_s") if "dumped_at_s" in r.loads else r.loads
    if len(ld):
        tonnes = np.interp(grid, np.concatenate([[0.0], ld["dumped_at_s"].to_numpy(dtype=float)]),
                           np.concatenate([[0.0], ld["tonnes"].cumsum().to_numpy(dtype=float)]))
    else:
        tonnes = np.zeros(len(grid))
    share = selected_cost(r.kpi, parts) / r.kpi["cost_total"] if r.kpi["cost_total"] else 1.0
    cost = (np.interp(grid, tl["t_s"], tl["cost"]) if len(tl) else np.zeros(len(grid))) * share
    q = r.states[r.states["state"].isin(["kolejka_koparka", "kolejka_kruszarka"])]
    st_, en_ = q["start_s"].to_numpy(), q["end_s"].to_numpy()
    queue = np.array([float(np.clip(np.minimum(en_, g) - st_, 0, None).sum()) / 3600 for g in grid])
    with np.errstate(invalid="ignore", divide="ignore"):
        # the first loads make the cost per tonne jump – it is shown once some material has gone through
        zl_t = np.where(tonnes > max(150.0, 0.05 * float(tonnes[-1] if len(tonnes) else 0)), cost / np.maximum(tonnes, 1),
                        np.nan)
    return pd.DataFrame({"t_s": grid, "t": tonnes, "zl_t": zl_t, "q": queue, "cost": cost})


@st.cache_data(show_spinner=False, max_entries=6)
def _split(state_yaml: str, params: tuple, seed: int, dynamic: bool = False, parts: tuple = ()) -> tuple:
    """Both runs of one shift: the side-by-side page and the shift's curves."""
    from kruszyk.config import cycle_costs, scenario_from_dict
    import yaml

    sc = cycle_costs(scenario_from_dict(yaml.safe_load(state_yaml)))
    a = simulate(today(sc), seed=seed)
    b = simulate(with_signals(sc, dict(params), dynamic), seed=seed)
    parts = parts or tuple(k for k, _ in COSTS)
    ca, cb = _in_shift(a, parts, 2.0), _in_shift(b, parts, 2.0)
    pack = lambda c: {k: [round(float(v), 3) for v in c[col]] for k, col in  # noqa: E731
                      (("s", "t_s"), ("t", "t"), ("cost", "cost"), ("q", "q"))}
    curves = {"shift_s": sc.shift_hours * 3600, "min_t": 150.0, "L": pack(ca), "R": pack(cb)}
    return split_html(a, b, "Jak dziś", "Z sygnałami RL", curves, prior="__PRIOR__")


PERIODS = ["Zmiana", "Doba", "Kilka dni", "Tydzień", "Miesiąc"]


def _window(a: pd.DataFrame, i: int, period: str, days: int) -> list[int]:
    """The shifts of the chosen period around shift i."""
    dates = [pd.Timestamp(d).normalize() for d in a["data"]]
    d0 = dates[i]
    if period == "Doba":
        keep = [k for k, d in enumerate(dates) if d == d0]
    elif period == "Kilka dni":
        keep = [k for k, d in enumerate(dates) if d0 <= d < d0 + pd.Timedelta(days=days)]
    elif period == "Tydzień":
        wk = d0.isocalendar()[:2]
        keep = [k for k, d in enumerate(dates) if d.isocalendar()[:2] == wk]
    elif period == "Miesiąc":
        keep = list(range(len(a)))
    else:
        keep = [i]
    return keep or [i]


def signal_boxes(p: dict) -> str:
    """The driving signals in plain words, as boxes."""
    m, sd, cp = p["arrival_margin_s"], p["max_slowdown"], p["crusher_pacing"]
    when = ("chwilę przed zwolnieniem koparki" if m > 5 else "chwilę po zwolnieniu koparki" if m < -5
            else "dokładnie, gdy koparka się zwolni")
    rows = [("Prędkość pod koparkę:", f"dojedź {when} – bez stania w kolejce (najwyżej {100 * sd:.0f} % wolniej, "
                                      "nigdy ponad limit)."),
            ("Przed kruszarką:", "nie zwalniaj – kosz nadąża." if cp < 0.15 else
             "zwolnij, gdy kosz jest pełny – zamiast czekać na biegu jałowym."),
            ("Postój:", "wozidła nie stają – każde jeździ bez przerwy." if p.get("truck_park_queue_min", -1) < 0 else
             f"„stań i wyłącz silnik”, gdy kurs skończyłby się kolejką dłuższą niż {p['truck_park_queue_min']:.0f} min; "
             f"zawsze jeździ co najmniej {100 * p['truck_min_working_share']:.0f} % wozideł każdej koparki."),
            ("Operator koparki:", "nabierz łyżkę przed podstawieniem wozidła.")]
    return "".join(f'<div class="k2-box"><b>{a}</b> {b}</div>' for a, b in rows)


def _prior(a: pd.DataFrame, b: pd.DataFrame, lab: list, i: int, period: str, days: int, parts: tuple) -> str:
    """The period's shifts before the one played (cumulative at each shift's end) for the live charts; 'null' for
    a single shift."""
    import json

    keep = [k for k in _window(a, i, period, days) if k < i]
    if period == "Zmiana" or not keep:
        return "null"
    side = lambda d: {"t": d["wywiezione_t"].iloc[keep].cumsum().round(1).tolist(),  # noqa: E731
                      "cost": _cost(d.iloc[keep], parts).cumsum().round(0).tolist(),
                      "q": d["kolejki_h"].iloc[keep].cumsum().round(2).tolist()}
    first = _window(a, i, period, days)[0]
    return json.dumps({"n": len(keep), "labels": [lab[k] for k in keep], "current": lab[i],
                       "title": f"{period}: {lab[first]} → {lab[i]} (do tej chwili)", "L": side(a), "R": side(b)})


def _shift_step(n: int, d: int) -> None:
    st.session_state.k2_shift_i = (int(st.session_state.get("k2_shift_i", 0)) + d) % n


def step_simulation() -> None:
    head = st.columns([4, 1.4], vertical_alignment="bottom")
    with head[0]:
        theme.header("Symulacja: jak dziś i z sygnałami RL", "Ta sama zmiana z miesiąca obok siebie – ten sam plan, "
                                                             "pogoda i awarie. Zmienia się tylko sposób jazdy.")
    appr = _approved()
    if not appr:
        return
    res = month(appr)
    if not res:
        return
    from kruszyk.config import dump_scenario

    a, b, lab = res["today"], res["signals"], res["labels"]
    parts, dyn = res.get("parts") or _parts(), bool(res.get("dynamic"))
    n = len(a)
    if int(st.session_state.get("k2_shift_i", 0)) >= n:
        st.session_state.k2_shift_i = 0
    with card("shiftbar"):
        c = st.columns([0.5, 2.6, 0.5, 2.4, 1.2], vertical_alignment="bottom")
        c[0].button("‹", key="k2_prev", width="stretch", help="poprzednia zmiana", on_click=_shift_step, args=(n, -1))
        c[1].selectbox("Zmiana", list(range(n)), key="k2_shift_i", format_func=lambda k: f"{k + 1}. {lab[k]}",
                       help="Wybierz zmianę z listy – możesz wpisać datę, np. 15.10.")
        c[2].button("›", key="k2_next", width="stretch", help="następna zmiana", on_click=_shift_step, args=(n, 1))
        period = c[3].segmented_control("Wykresy za okres", PERIODS, key="k2_period", default="Zmiana") or "Zmiana"
        days = int(c[4].number_input("dni", 2, 31, 3, key="k2_days", disabled=period != "Kilka dni"))
        i = int(st.session_state.get("k2_shift_i", 0))
        z0 = _cost(a.iloc[[i]], parts).iloc[0] / max(a["wywiezione_t"].iloc[i], 1)
        z1 = _cost(b.iloc[[i]], parts).iloc[0] / max(b["wywiezione_t"].iloc[i], 1)
        theme.note(f"{lab[i]} · jak dziś {a['wywiezione_t'].iloc[i]:,.0f} t · {z0:.2f} zł/t &nbsp;|&nbsp; z RL "
                   f"{b['wywiezione_t'].iloc[i]:,.0f} t · {z1:.2f} zł/t ({z1 - z0:+.2f})".replace(",", " "))
    st_yaml = dump_scenario(res["states"][i])
    page = _split(st_yaml, tuple(sorted(res["params"].items())), SEED + 7919 * i, dyn, tuple(parts))
    components.html(page.replace("__PRIOR__", _prior(a, b, lab, i, period, days, parts)), height=1240, scrolling=True)
    with head[1]:
        if st.button("Cały miesiąc →", type="primary", key="k2_to_result", width="stretch"):
            st.session_state.nav_next = "④ Wynik"
            st.rerun()


def step_result() -> None:
    head = st.columns([4, 2], vertical_alignment="bottom")
    appr = st.session_state.get("approved")
    res = st.session_state.get("month")
    with head[0]:
        sub = ""
        if res and appr and res["key"] == _month_key(appr):
            sub = (f"Wariant {appr.get('variant', 'A')}"
                   + (" (pełna wydajność koparek)" if (appr.get("work") or {}).get("mode") == "min" else "") + " · "
                   f"{res['labels'][0]} – {res['labels'][-1]} · {len(res['today'])} zmian · ten sam harmonogram, strzały, "
                   "pogoda i awarie. "
                   + ("Z RL wozidła są przydzielane do koparek dynamicznie – zmienia się jazda i przydział."
                      if res.get("dynamic") else "Wozidła jeżdżą do tych samych koparek – zmienia się tylko sposób jazdy."))
        theme.header("Ten sam miesiąc: jak dziś i z sygnałami RL", sub)
    appr = _approved()
    if not appr:
        return
    res = month(appr)
    if not res:
        return
    p = st.session_state.get("k2_plan", {})
    with head[1]:
        st.markdown('<div style="text-align:right">' + theme.chip("Harmonogram zamrożony") + " " +
                    theme.chip("Model zgodny z realną produkcją" if p.get("real_t") else "Model bez kontroli realnej produkcji",
                               ok=bool(p.get("real_t"))) + "</div>", unsafe_allow_html=True)
    a, b = res["today"], res["signals"]
    parts = tuple(res.get("parts") or _parts())
    k0s, k1s = _cost(a, parts), _cost(b, parts)  # every shift's cost counting the ticked parts
    t0, t1 = float(a["wywiezione_t"].sum()), float(b["wywiezione_t"].sum())
    c0, c1 = float(k0s.sum()), float(k1s.sum())
    f0, f1 = float(a["paliwo_l"].sum()), float(b["paliwo_l"].sum())
    q0, q1 = float(a["kolejki_h"].sum()), float(b["kolejki_h"].sum())
    z0, z1 = (c0 / t0 if t0 else 0.0), (c1 / t1 if t1 else 0.0)
    n = len(a)

    def num(x: float) -> str:
        return f"{x:,.0f}".replace(",", " ")
    m = st.columns(5)
    m[0].metric("Oszczędność w miesiącu", f"{num(c0 - c1)} zł",
                help="Ile mniej kosztował miesiąc z RL – zaznaczone koszty, te same zmiany. Dodatkowe tony nie są tu "
                     "wyceniane: są w „Wywiezione”.")
    m[1].metric("Koszt tony", f"{z1:.2f} zł/t", f"{z1 - z0:+.2f} zł/t (dziś {z0:.2f})", delta_color="inverse",
                help="Zaznaczone koszty (Koszty w wyliczeniu) – bez wiercenia i strzelania, jazda ich nie zmienia.")
    m[2].metric("Paliwo", f"{num(f1)} l", f"{100 * (f1 / f0 - 1):+.1f} % (dziś {num(f0)} l)" if f0 else None,
                delta_color="inverse")
    m[3].metric("Wywiezione", f"{num(t1)} t", f"{100 * (t1 / t0 - 1):+.1f} % (dziś {num(t0)} t)" if t0 else None)
    m[4].metric("Stanie w kolejkach", f"{num(q1)} h", f"{num(q1 - q0)} h (dziś {num(q0)} h)", delta_color="inverse")

    target = float(p.get("target_t") or 0)
    planned = float(appr.get("planned_t") or 0)
    if target or planned:
        txt = []
        if planned:
            txt.append(f"<b>Plan harmonogramu {num(planned)} t:</b> jak dziś wywiezione {100 * t0 / planned:.0f} % · "
                       f"z sygnałami {100 * t1 / planned:.0f} %")
        if target:
            txt.append(f"<b>Cel miesiąca {num(target)} t:</b> jak dziś {100 * t0 / target:.0f} % · "
                       f"z sygnałami {100 * t1 / target:.0f} %")
        st.markdown('<div class="k2-box" style="background:#fff;border:1px solid #D9DDE3">' + "<br>".join(txt) + "</div>",
                    unsafe_allow_html=True)
    c1_, c2_ = st.columns([3, 2])
    with c1_, card("chart"):
        theme.h4("Koszt narastająco")
        fig = go.Figure()
        for name, d, color in (("jak dziś", a, "#8A93A0"), ("z sygnałami", b, "#B3122A")):
            fig.add_scatter(x=res["labels"], y=_cost(d, parts).cumsum() / 1000, mode="lines", name=name,
                            line=dict(color=color, width=3))
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.12),
                          xaxis=dict(showticklabels=False), yaxis_title="tys. zł", plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig, width="stretch")
        week = pd.DataFrame({"w": [pd.Timestamp(d).strftime("%d.%m") for d in a["data"]],
                             "wk": [pd.Timestamp(d).isocalendar().week for d in a["data"]],
                             "t0": a["wywiezione_t"], "t1": b["wywiezione_t"], "k0": k0s, "k1": k1s,
                             "win": (k1s / b["wywiezione_t"].clip(lower=1) < k0s / a["wywiezione_t"].clip(lower=1))
                             .astype(int)})
        rows = []
        for _, g in week.groupby("wk", sort=False):
            rows.append([f"{g['w'].iloc[0]}–{g['w'].iloc[-1]}", f"{num(g['t0'].sum())} / {num(g['t1'].sum())}",
                         f"{g['k0'].sum() / g['t0'].sum():.2f} / {g['k1'].sum() / g['t1'].sum():.2f}",
                         f"{int(g['win'].sum())} z {len(g)}"])
        st.markdown(theme.table([("Tydzień", ""), ("Tony dziś / z RL", "n"), ("zł/t dziś / z RL", "n"),
                                 ("Zmian taniej", "n")], rows, numeric={1, 2, 3}, sans={0}), unsafe_allow_html=True)
    if all(f"k_{k}" in a for k, _ in COSTS):
        with card("savings"):
            theme.h4("Skąd oszczędność", "cały miesiąc; zaznaczone koszty, bez wiercenia i strzelania. Kierowcy i "
                                         "operatorzy są opłaceni tak samo – liczy się paliwo i to, co zależy od "
                                         "motogodzin")
            rows = []
            for k, label in COSTS:
                if k in parts:
                    x0, x1 = float(a[f"k_{k}"].sum()), float(b[f"k_{k}"].sum())
                    rows.append([label, num(x0), num(x1), f"{x1 - x0:+,.0f}".replace(",", " ")])
            rows.append(["Razem – zaznaczone koszty", num(c0), num(c1), f"{c1 - c0:+,.0f}".replace(",", " ")])
            s0, s1 = float(a["koszt_zl"].sum()), float(b["koszt_zl"].sum())
            if len(parts) < len(COSTS):
                rows.append(["Koszt całkowity – wszystkie składniki", num(s0), num(s1),
                             f"{s1 - s0:+,.0f}".replace(",", " ")])
            m0, m1 = float(a["motogodziny_wozidel"].sum()), float(b["motogodziny_wozidel"].sum())
            rows.append(["Motogodziny wozideł [mth]", num(m0), num(m1), f"{m1 - m0:+,.0f}".replace(",", " ")])
            st.markdown(theme.table([("Składnik", ""), ("Jak dziś [zł]", "n"), ("Z RL [zł]", "n"), ("Różnica", "n")],
                                    rows, numeric={1, 2, 3}, sans={0}), unsafe_allow_html=True)
    with c2_:
        with card("signals"):
            theme.h4("Sygnały dla kierowców")
            st.markdown(signal_boxes(res["params"]), unsafe_allow_html=True)
            s = res["setup"]
            theme.note(f"Cel nauki: <b>{goal_label(s.get('weight', 0.5))}</b>. "
                       f"Nauczone na {s['shifts']} zmianach symulacji, w sytuacjach z {s['states']} zmian tego miesiąca "
                       "(pełne usypy, końcówki, przejazdy koparek, deszcz, awarie).")
        with card("sure"):
            wins = int((k1s / b["wywiezione_t"].clip(lower=1) < k0s / a["wywiezione_t"].clip(lower=1)).sum())
            need = sign_threshold(n)
            theme.h4("Pewność wyniku")
            verdict = ("różnica pewna" if wins >= need else "za mało, by mówić o pewnej różnicy" if wins > n - need
                       else "z sygnałami drożej")
            st.markdown(f'<div style="font-size:14px">Taniej z sygnałami na <b>{wins} z {n}</b> zmian – {verdict} '
                        f"(próg: {need} z {n}).</div>", unsafe_allow_html=True)
        out = pd.DataFrame({"zmiana": res["labels"], "t_dzis": a["wywiezione_t"], "t_rl": b["wywiezione_t"],
                            "zl_t_dzis": a["koszt_na_t"], "zl_t_rl": b["koszt_na_t"], "paliwo_dzis": a["paliwo_l"],
                            "paliwo_rl": b["paliwo_l"]})
        st.download_button("Pobierz wyniki zmian (CSV)", out.to_csv(index=False).encode("utf-8"), "wynik_miesiaca.csv",
                           "text/csv", key="k2_csv", width="stretch")
