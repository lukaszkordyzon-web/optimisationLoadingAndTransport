"""② Harmonogram: the planner sets where every excavator loads on every shift; the program adds the drilling and
blasting of every new pile, shows when it is there and what does not add up. The approved schedule is frozen: the
month with today's driving and the month with the driving signals both follow it."""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kruszyk.period import PeriodPlan
from kruszyk.schedule import (EMPTY, NOT_YET, area_blast_t, area_of, check, depletion, frozen, hauled_plan,
                              plan_blasts, propose)
from . import state, theme
from .theme import card

grid_component = components.declare_component("k2_grid", path=str(Path(__file__).parent / "grid_component"))

WD = ["pn", "wt", "śr", "cz", "pt", "so", "nd"]
ROMAN = {1: "I", 2: "II", 3: "III"}
FIRE = {"1 zmiana – urobek od następnej zmiany": 1, "między zmianami – urobek od razu": 0, "2 zmiany": 2}
START_COLORS = [("#DCE8F7", "#1D4E89"), ("#D6EFEA", "#11655A"), ("#E4E9F2", "#2B3F66")]
AREA_COLORS = [("#FBE3CC", "#8A4B12"), ("#E7DDF5", "#5A3A8C"), ("#FFF0B8", "#6E5600"), ("#F6D9E4", "#8A2650"),
               ("#DDF0D5", "#2F5E1E"), ("#FCE1D8", "#8A2E12")]


# ------------------------------------------------------------------ period
def plan_settings() -> dict:
    p = st.session_state.setdefault("k2_plan", {})
    today = dt.date.today()
    p.setdefault("start", (today + dt.timedelta(days=7 - today.weekday())).isoformat())
    p.setdefault("days", 28)
    p.setdefault("shifts", 2)
    p.setdefault("sat", False)
    p.setdefault("hol", "")
    p.setdefault("fire", 1)
    p.setdefault("real_t", 0.0)
    p.setdefault("target_t", 0.0)
    return p


def _holidays(text: str, year: int) -> list[dt.date]:
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        try:
            d, m = (int(x) for x in part.strip().split(".")[:2])
            out.append(dt.date(year, m, d))
        except ValueError:
            continue
    return out


def period_plan(p: dict) -> PeriodPlan:
    start = dt.date.fromisoformat(p["start"])
    return PeriodPlan(start=start, days=int(p["days"]), shifts_per_day=int(p["shifts"]), saturdays=bool(p["sat"]),
                      holidays=_holidays(p["hol"], start.year), move_setup_min=15.0)


def labels(plan: PeriodPlan) -> list[str]:
    return [f"{WD[d.weekday()]} {d:%d.%m} {ROMAN.get(s, s)}" for d, _, s in plan.shifts()]


def rates_per_shift() -> tuple[dict, float]:
    from app_rules import loading_rates

    r, hours = loading_rates(state.raw_yaml())
    return {k: v * hours for k, v in r.items()}, hours


# ------------------------------------------------------------------ screen
def step_plan() -> None:
    raw = state.raw()
    sc = state.scenario()
    p = plan_settings()
    v = st.session_state.get("ver", 0)
    head = st.columns([4, 1.6], vertical_alignment="bottom")
    with head[0]:
        theme.header("Harmonogram miesiąca", "Układasz go Ty. Program dolicza wiercenie i strzał, pilnuje dostępności "
                                             "usypów i wiertnic.")
    if not sc.excavators:
        st.info("Najpierw narysuj kopalnię (koparki, usypy, drogi) w kroku **① Kopalnia**.")
        return
    rates, _ = rates_per_shift()

    # ---------------------------------------------------------------- period
    with card("period"):
        c = st.columns([0.7, 1.3, 0.7, 0.9, 1.0, 1.1, 1.1, 1.4], vertical_alignment="bottom")
        c[0].markdown('<div class="k2-h4" style="padding-bottom:10px">Okres</div>', unsafe_allow_html=True)
        p["start"] = c[1].date_input("Start", dt.date.fromisoformat(p["start"]), format="DD.MM.YYYY", key="k2_start").isoformat()
        p["days"] = c[2].number_input("Liczba dni", 1, 92, int(p["days"]), key="k2_days")
        p["shifts"] = c[3].selectbox("Zmian na dobę", [1, 2, 3], index=int(p["shifts"]) - 1, key="k2_shifts")
        p["sat"] = c[4].checkbox("Soboty robocze", bool(p["sat"]), key="k2_sat")
        p["hol"] = c[5].text_input("Dni wolne", p["hol"], key="k2_hol", placeholder="11.11, 24.12")
        p["real_t"] = c[6].number_input("Realnie na zmianę [t]", 0.0, 50_000.0, float(p["real_t"]), 100.0, format="%.0f", key="k2_real",
                                        help="Ile ton robicie realnie na zmianę – program sprawdzi, czy model daje tyle samo.")
        model_t = sum(rates.values())
        with c[7]:
            if p["real_t"] and model_t:
                diff = 100 * (model_t / p["real_t"] - 1)
                if abs(diff) <= 5:
                    st.markdown(theme.chip("model zgodny"), unsafe_allow_html=True)
                else:
                    st.markdown(theme.chip(f"model {model_t:,.0f} t ({diff:+.0f} %)".replace(",", " "), ok=False),
                                unsafe_allow_html=True)
                    if st.button("Dopasuj model", key="k2_fit"):
                        _fit_model(raw, sc, p["real_t"])
            elif model_t:
                theme.note(f"model: ok. {model_t:,.0f} t/zmianę".replace(",", " "))
        g = st.columns([0.7, 1.3, 5.6], vertical_alignment="bottom")
        g[0].markdown('<div class="k2-h4" style="padding-bottom:10px">Cel</div>', unsafe_allow_html=True)
        p["target_t"] = g[1].number_input("Cel na miesiąc [t]", 0.0, 10_000_000.0, float(p["target_t"]), 5000.0,
                                          format="%.0f", key="k2_target",
                                          help="Ile ton nadawy ma przejść w okresie (0 = bez celu).")
        target_slot = g[2].empty()
    msg = st.session_state.pop("k2_fit_msg", None)
    if msg:
        st.success("✅ " + msg)
    plan = period_plan(p)
    shifts = plan.shifts()
    n = len(shifts)
    lab = labels(plan)
    if n == 0:
        st.warning("W tym okresie nie ma zmian roboczych.")
        return

    # ---------------------------------------------------------------- start state, drilling, areas
    start_t = {b.id: float(b.tonnage_t) for b in sc.blasts}
    rigs = [r for r in (raw.get("drills") or []) if r.get("in_work", True)]
    c1, c2 = st.columns([2, 3])
    with c1, card("startstate"):
        theme.h4(f"Stan na start · {lab[0]}", "z mapy")
        on = {}
        for e in sc.excavators:
            on.setdefault(e.blast, []).append(e.id)
        st.markdown(theme.table([("Usyp leży", ""), ("Ton", "n"), ("Koparka", "")],
                                [[b, f"{t:,.0f}".replace(",", " "), ", ".join(on.get(b, [])) or "–"]
                                 for b, t in start_t.items()], numeric={1}), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        theme.h4("Wiercenie i strzelanie")
        theme.note((" · ".join(f"{r['id']}: <b>{float(r.get('m_per_shift') or 250):.0f} mb/zm.</b>" for r in rigs)
                    + " (z Maszyn)") if rigs else "Brak wiertnic w pracy – odstrzały bez wiercenia (Maszyny).")
        names = list(FIRE)
        cur = next((k for k, x in FIRE.items() if x == int(p["fire"])), names[0])
        p["fire"] = FIRE[st.selectbox("Strzał (z ewakuacją i przewietrzeniem)", names, index=names.index(cur),
                                      key="k2_fire")]
    with c2, card("areas"):
        theme.h4("Obszary eksploatacji", "każdy odstrzał to nowy usyp: A/1, A/2 …")
        _areas_editor(raw, sc, rigs, int(p["fire"]), v)

    # ---------------------------------------------------------------- the grid
    key = repr((n, tuple(e.id for e in sc.excavators), p["start"], int(p["shifts"])))
    grid = st.session_state.get("grid")
    if grid is None or st.session_state.get("grid_key") != key or set(grid) != {e.id for e in sc.excavators}:
        grid = propose(sc.excavators, start_t, sc.areas, rates, n, sc.blasting.max_blast_t())
        st.session_state.grid, st.session_state.grid_key = grid, key
    grid = {ex: (cells + [cells[-1]] * n)[:n] for ex, cells in grid.items()}
    st.session_state.grid = grid
    blasts = plan_blasts(grid, sc.areas, rigs, n, sc.blasting.max_blast_t(), int(p["fire"]))
    issues = check(grid, blasts, start_t, rates, n)
    left = depletion(grid, blasts, start_t, rates, n)
    planned = hauled_plan(grid, blasts, start_t, rates, n)
    total = sum(planned)
    target = float(p["target_t"] or 0)
    if target:
        pct = 100 * total / target
        target_slot.markdown(theme.chip(f"plan da ok. {total:,.0f} t · {pct:.0f} % celu".replace(",", " "), ok=pct >= 100),
                             unsafe_allow_html=True)
    else:
        target_slot.markdown(f'<div class="k2-note" style="padding-bottom:10px">plan da ok. <b>{total:,.0f} t</b> '
                             "(szacunek z dzisiejszego tempa)</div>".replace(",", " "), unsafe_allow_html=True)

    main, side = st.columns([3.2, 1])
    with main, card("grid"):
        h = st.columns([2.2, 1.4, 1.6], vertical_alignment="center")
        h[0].markdown('<div class="k2-h4">Harmonogram</div>', unsafe_allow_html=True)
        with h[1]:
            day_view = st.radio("Dokładność", ["Dzień", "Zmiana"], horizontal=True, key="k2seg_view",
                                label_visibility="collapsed") == "Dzień"
        if h[2].button("↺ Ułóż propozycję od nowa", key="k2_repropose", width="stretch"):
            st.session_state.grid = propose(sc.excavators, start_t, sc.areas, rates, n, sc.blasting.max_blast_t())
            st.rerun()
        data = _grid_payload(grid, blasts, left, start_t, rigs, shifts, lab, day_view, sc, planned, target)
        val = grid_component(data=data, key="k2_gridc", default=None)
        if val and val.get("nonce") != st.session_state.get("k2_grid_nonce"):
            st.session_state.k2_grid_nonce = val["nonce"]
            _set(val["ex"], int(val["a"]), int(val["b"]), val["pile"])
            st.rerun()
        with st.expander("Edycja zakresem lub w tabeli"):
            _editor(grid, blasts, sc, start_t, lab, n)
    with side:
        with card("issues"):
            short = target - total if target else 0.0
            theme.h4(f"Do poprawy · {len(issues) + (1 if short > 0 else 0)}")
            if short > 0:
                st.markdown(f'<div class="k2-issue"><b>Plan nie zrobi celu – brakuje ok. {short:,.0f} t</b><br>'
                            "Dołóż odstrzał, zmianę na dobę albo soboty – albo sprawdź, czy koparki nie stoją bez "
                            "urobku.</div>".replace(",", " "), unsafe_allow_html=True)
            if not issues and short <= 0:
                theme.note("Każda koparka ma urobek na każdej zmianie (szacunek z dzisiejszego tempa ładowania).")
            for j, it in enumerate(issues):
                _issue(it, grid, blasts, sc, start_t, lab, n, j)

    sig = hashlib.md5(repr((state.raw_yaml(), grid, p)).encode()).hexdigest()
    appr = st.session_state.get("approved")
    with head[1]:
        if appr and appr["key"] == sig:
            st.markdown(theme.chip("Harmonogram zatwierdzony"), unsafe_allow_html=True)
        if st.button("Zatwierdź i porównaj z RL →", type="primary", key="k2_approve", width="stretch"):
            st.session_state.approved = {"key": sig, "frozen": frozen(grid, blasts, n), "plan": dict(p),
                                         "grid": copy.deepcopy(grid), "blasts": blasts}
            st.session_state.pop("month", None)
            st.session_state.nav_next = "③ Symulacja"
            st.rerun()


def _fit_model(raw: dict, sc, real_t: float) -> None:
    from kruszyk.calibrate import calibrate

    with st.spinner("Dopasowuję model do realnej produkcji…"):
        fitted, _, after = calibrate(sc, {"tonnes_t": real_t, "trucks": sc.truck_count})
    new = copy.deepcopy(raw)
    new["calibration"] = dataclasses.asdict(fitted.calibration)
    state.set_raw(new)
    st.session_state.k2_fit_msg = (f"Model dopasowany: {after['tonnes_t']:,.0f} t na zmianę "
                                   f"(przestoje {fitted.calibration.delay_min_per_trip:.1f} min na kurs).").replace(",", " ")
    st.rerun()


def _areas_editor(raw: dict, sc, rigs: list, fire: int, v: int) -> None:
    if not sc.areas:
        st.info("Brak obszarów eksploatacji – narysuj je na mapie (① Kopalnia → „Obszar wydobycia”). Bez nich "
                "harmonogram korzysta tylko z usypów, które leżą dziś.")
        return
    mps = max([float(r.get("m_per_shift") or 250) for r in rigs] or [250.0])
    cap = sc.blasting.max_blast_t()
    rows = []
    for a in sc.areas:
        t = area_blast_t(a, cap)
        m = t / float(a.get("t_per_m") or 50)
        d = 0 if not rigs else -(-m // mps)
        rows.append({"obszar": a["id"], "poziom": a.get("level_m"), "odstrzal_t": float(a.get("blast_max_t") or 20000),
                     "tpm": float(a.get("t_per_m") or 50), "zasoby_t": float(a.get("reserves_t") or 0),
                     "mb": round(m), "wiercenie": f"{d:.0f} zm.", "do_urobku": f"{d + fire:.0f} zm."})
    ed = st.data_editor(pd.DataFrame(rows), key=f"k2_areas_{v}", hide_index=True, width="stretch",
                        disabled=["obszar", "poziom", "mb", "wiercenie", "do_urobku"],
                        column_config={"obszar": "obszar", "poziom": st.column_config.NumberColumn("poziom [m]", format="%.0f"),
                                       "odstrzal_t": st.column_config.NumberColumn("odstrzał [t]", min_value=500.0, step=500.0),
                                       "tpm": st.column_config.NumberColumn("t/mb", min_value=5.0),
                                       "zasoby_t": st.column_config.NumberColumn("zasoby [t]", min_value=0.0, step=5000.0),
                                       "mb": "metrów wiercenia", "wiercenie": "wiercenie",
                                       "do_urobku": "od wiercenia do urobku"})
    if cap < float("inf"):
        st.caption(f"Limit MW: największy strzał {cap:,.0f} t.".replace(",", " "))
    changed = False
    new = copy.deepcopy(raw)
    for _, r in ed.iterrows():
        upd = {"blast_max_t": float(r["odstrzal_t"]), "t_per_m": float(r["tpm"]), "reserves_t": float(r["zasoby_t"])}
        for src in (new.get("areas") or []) + ((new.get("site") or {}).get("areas") or []):
            if src.get("id") == r["obszar"] and any(float(src.get(k) or 0) != val for k, val in upd.items()):
                src.update(upd)
                changed = True
    if changed:
        state.set_raw(new)
        st.rerun()


# ------------------------------------------------------------------ the grid as a table
def _colors(start_t: dict, blasts: dict) -> dict:
    out = {p: START_COLORS[i % len(START_COLORS)] for i, p in enumerate(start_t)}
    areas = sorted({b["area"] for b in blasts.values()})
    for p, b in blasts.items():
        out[p] = AREA_COLORS[(areas.index(b["area"]) * 2 + int(p.rsplit("/", 1)[1])) % len(AREA_COLORS)]
    return out


def _grid_payload(grid, blasts, left, start_t, rigs, shifts, lab, day_view, sc, planned=None, target=0.0) -> dict:
    """Everything the interactive grid draws: columns (days or shifts), rows with coloured cells, the piles an
    excavator can be sent to with when they are there."""
    n = len(shifts)
    colors = _colors(start_t, blasts)
    if day_view:
        groups, heads, cur = [], [], None
        for i, (d, _, _) in enumerate(shifts):
            if cur != d:
                cur = d
                groups.append([])
                heads.append(f"{WD[d.weekday()]} {d:%d.%m}")
            groups[-1].append(i)
    else:
        groups, heads = [[i] for i in range(n)], list(lab)

    def cell(text, bg, fg, title="", red=False, bold=False):
        return {"text": text, "bg": bg, "fg": fg, "title": title, "red": red, "bold": bold}

    def bad(ex, i):
        p = grid[ex][i]
        return (p in blasts and blasts[p]["avail"] > i) or left.get(p, [0] * n)[i] <= 1

    rows = [{"kind": "group", "name": "KOPARKI – planujesz Ty"}]
    for ex, cells in grid.items():
        out = []
        for g in groups:
            ps = [cells[i] for i in g]
            text = ps[0] if len(set(ps)) == 1 else " ⇢".join(dict.fromkeys(ps))
            bg, fg = colors.get(ps[-1], ("#FFFFFF", "#1B2733"))
            red = any(bad(ex, i) for i in g)
            why = ""
            if red:
                p = next(cells[i] for i in g if bad(ex, i))
                why = (f" – {p} dostępny dopiero od {lab[min(blasts[p]['avail'], n - 1)]}"
                       if p in blasts and blasts[p]["avail"] > g[0] else f" – na {p} nie zostanie urobku (szacunek)")
            out.append(cell(text, bg, fg, f"{ex}: {text}{why}", red))
        rows.append({"kind": "ex", "name": ex, "cells": out})
    if rigs:
        rows.append({"kind": "group", "name": "WIERTNICE – liczy program"})
        for r in rigs:
            out = []
            for g in groups:
                jobs = list(dict.fromkeys(p for p, b in blasts.items() for (rid, x) in b["drill"] if rid == r["id"] and x in g))
                out.append(cell(", ".join(jobs) or "–", "#EDEFF2" if jobs else "#FFFFFF", "#3A4452" if jobs else "#8A93A0",
                                f"{r['id']}: " + ("wierci " + ", ".join(jobs) if jobs else "wolna")))
            rows.append({"kind": "rig", "name": r["id"], "cells": out})
    if blasts:
        rows.append({"kind": "group", "name": "ODSTRZAŁY – liczy program"})
        for p, b in sorted(blasts.items(), key=lambda kv: kv[1]["avail"]):
            out = []
            for g in groups:
                if any(x in g for x in b["fire"]):
                    out.append(cell("strzał", "#1B2733", "#FFFFFF", f"{p}: strzał", bold=True))
                elif any(x in g for _, x in b["drill"]):
                    out.append(cell("wierc.", "#EDEFF2", "#3A4452", f"{p}: wiercenie"))
                elif any(x >= b["avail"] for x in g):
                    lft = left[p][g[0]]
                    out.append(cell(f"{lft / 1000:.1f} kt" if lft > 1 else "wybrany", "#E3F1E6" if lft > 1 else "#F3F4F6",
                                    "#1E5A2C" if lft > 1 else "#5B6573", f"{p}: leży ok. {lft:,.0f} t".replace(",", " ")))
                else:
                    out.append(cell("", "#FFFFFF", "#FFFFFF"))
            rows.append({"kind": "blast", "name": f"{p} · {b['t'] / 1000:.0f} kt · {b['m']} mb", "cells": out})
    rows.append({"kind": "group", "name": "USYPY Z DZIŚ"})
    for p in start_t:
        out = []
        for g in groups:
            lft = left[p][g[0]]
            out.append(cell(f"{lft / 1000:.1f} kt" if lft > 1 else "wybrany", "#E3F1E6" if lft > 1 else "#F3F4F6",
                            "#1E5A2C" if lft > 1 else "#5B6573", f"{p}: leży ok. {lft:,.0f} t".replace(",", " ")))
        rows.append({"kind": "pile", "name": p, "cells": out})
    if planned:
        rows.append({"kind": "group", "name": "PLAN NARASTAJĄCO" + (f" · cel {target / 1000:,.0f} kt".replace(",", " ")
                                                                   if target else "")})
        out, acc = [], 0.0
        for g in groups:
            acc += sum(planned[i] for i in g)
            due = target * (g[-1] + 1) / n if target else 0.0
            ok = not target or acc >= due * 0.98
            out.append(cell(f"{acc / 1000:.1f} kt", "#E3F1E6" if ok else "#FFF0B8", "#1E5A2C" if ok else "#6E5600",
                            f"do końca: ok. {acc:,.0f} t".replace(",", " ") +
                            (f" · w tempie celu powinno być {due:,.0f} t".replace(",", " ") if target else ""), bold=True))
        rows.append({"kind": "sum", "name": "wywiezione [kt]", "cells": out})
    opts, _ = _options(grid, blasts, sc, start_t)
    piles = []
    for p in opts:
        is_new = p not in start_t and p not in blasts
        piles.append({"id": p, "isNew": is_new, "avail": None if p in start_t or is_new else blasts[p]["avail"],
                      "left": left.get(p)})
    return {"day": day_view, "heads": heads, "groups": groups, "labels": lab, "off": [False] * len(heads),
            "rows": rows, "grid": grid, "piles": piles}


# ------------------------------------------------------------------ editing
def _options(grid, blasts, sc, start_t) -> tuple[list[str], dict]:
    used = {p for cells in grid.values() for p in cells}
    opts, desc = [], {}
    for p, t in start_t.items():
        opts.append(p)
        desc[p] = f"{p} · leży dziś ({t:,.0f} t)".replace(",", " ")
    areas = {a["id"]: a for a in sc.areas}
    for a_id in areas:
        ks = sorted(area_of(p, areas)[1] for p in used | set(blasts) if area_of(p, areas) and p.startswith(f"{a_id}/"))
        for k in ks:
            pid = f"{a_id}/{k}"
            opts.append(pid)
            b = blasts.get(pid)
            desc[pid] = f"{pid} · odstrzał z obszaru {a_id}" + (" · w harmonogramie" if b else "")
        nxt = f"{a_id}/{(ks[-1] if ks else 0) + 1}"
        opts.append(nxt)
        desc[nxt] = f"{nxt} · NOWY odstrzał z obszaru {a_id} – program doda wiercenie i strzał"
    return opts, desc


def _set(ex: str, a: int, b: int, pile: str) -> None:
    g = st.session_state.grid
    for i in range(a, b + 1):
        if i < len(g[ex]):
            g[ex][i] = pile


def _editor(grid, blasts, sc, start_t, lab, n) -> None:
    opts, desc = _options(grid, blasts, sc, start_t)
    c = st.columns([1, 1.4, 1.4, 3, 1.2])
    ex = c[0].selectbox("Koparka", list(grid), key="k2_e_ex")
    a = c[1].selectbox("Od zmiany", range(n), format_func=lambda i: lab[i], key="k2_e_from")
    b = c[2].selectbox("Do zmiany", range(n), index=n - 1, format_func=lambda i: lab[i], key="k2_e_to")
    pile = c[3].selectbox("Ładuje z", opts, format_func=lambda x: desc.get(x, x), key="k2_e_pile")
    with c[4]:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("Ustaw", type="primary", key="k2_e_set", width="stretch"):
            _set(ex, min(a, b), max(a, b), pile)
            st.rerun()
    with st.expander("Edytuj w tabeli (wszystkie zmiany naraz)"):
        df = pd.DataFrame({lab[i]: [grid[e][i] for e in grid] for i in range(n)}, index=list(grid))
        ver = st.session_state.get("k2_tab_ver", 0)
        ed = st.data_editor(df, key=f"k2_tab_{ver}", width="stretch",
                            column_config={c_: st.column_config.SelectboxColumn(c_, options=opts) for c_ in df.columns})
        if not ed.equals(df):
            for e in grid:
                st.session_state.grid[e] = [str(ed.loc[e, lab[i]] or grid[e][i]) for i in range(n)]
            st.session_state.k2_tab_ver = ver + 1
            st.rerun()


def _issue(it, grid, blasts, sc, start_t, lab, n, j) -> None:
    ex, i, p = it["ex"], it["shift"], it["pile"]
    if it["kind"] == NOT_YET:
        avail = it["avail"]
        prev = grid[ex][i - 1] if i > 0 else next((e.blast for e in sc.excavators if e.id == ex), p)
        st.markdown(f'<div class="k2-issue"><b>{ex} · {lab[i]}: {p} dostępny dopiero od {lab[min(avail, n - 1)]}</b><br>'
                    "Wiercenie i strzał nie zdążą – wiertnice zajęte albo za mało czasu.</div>", unsafe_allow_html=True)
        if prev != p and st.button(f"Zostaw {ex} na {prev} do tego czasu", key=f"k2_fix_{j}", width="stretch"):
            _set(ex, i, min(avail, n) - 1, prev)
            st.rerun()
    elif it["kind"] == EMPTY:
        areas = {a["id"]: a for a in sc.areas}
        e = next((x for x in sc.excavators if x.id == ex), None)
        allowed = [a for a in areas if e is None or e.may_work(a)]
        nxt = None
        if allowed:
            a_id = allowed[0]
            ks = [area_of(q, areas)[1] for q in set(blasts) | {x for cells in grid.values() for x in cells}
                  if area_of(q, areas) and q.startswith(f"{a_id}/")]
            nxt = f"{a_id}/{max(ks or [0]) + 1}"
        st.markdown(f'<div class="k2-issue w"><b>{ex} · {lab[i]}: na {p} nie zostanie urobku</b><br>'
                    "Szacunek z dzisiejszego tempa ładowania – koparka by stała.</div>", unsafe_allow_html=True)
        if nxt and st.button(f"Od tej zmiany {ex} na nowy odstrzał {nxt}", key=f"k2_fix_{j}", width="stretch"):
            end = i
            while end + 1 < n and grid[ex][end + 1] == p:
                end += 1
            _set(ex, i, end, nxt)
            st.rerun()
