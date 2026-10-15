"""② Harmonogram: the planner sets the work – on every day, in which area every excavator loads and how many tonnes,
and to which excavator every truck drives (or that it stands). The program numbers the piles of every area, adds
the drilling and blasting of every new pile in time, and shows what does not add up. The approved schedule is frozen:
the month with today's driving and the month with the driving signals both follow it."""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kruszyk import workplan
from kruszyk.config import yaml_dump
from kruszyk.period import PeriodPlan
from kruszyk.schedule import area_blast_t
from . import state, theme
from .theme import card

grid_component = components.declare_component("k2_grid", path=str(Path(__file__).parent / "grid_component"))

WD = ["pn", "wt", "śr", "cz", "pt", "so", "nd"]
ROMAN = {1: "I", 2: "II", 3: "III"}
FIRE = {"1 zmiana – urobek od następnej zmiany": 1, "między zmianami – urobek od razu": 0, "2 zmiany": 2}
AREA_COLORS = [("#FBE3CC", "#8A4B12"), ("#E7DDF5", "#5A3A8C"), ("#D6EFEA", "#11655A"), ("#F6D9E4", "#8A2650"),
               ("#FFF0B8", "#6E5600"), ("#DDF0D5", "#2F5E1E")]
EX_COLORS = [("#DCE8F7", "#1D4E89"), ("#FCE1D8", "#8A2E12"), ("#E4E9F2", "#2B3F66"), ("#E3F1E6", "#1E5A2C"),
             ("#F3E8D2", "#6B4A12")]
STAND = "postój"
OFF = "nie jeździ"


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


@st.cache_data(show_spinner=False, max_entries=8)
def _rates(sim_yaml: str) -> tuple[dict, float]:
    from app_rules import loading_rates

    return loading_rates(sim_yaml)


def rates_per_shift(raw: dict, sc, near: dict, cap: float) -> dict[str, float]:
    """Tonnes every excavator loads on a shift worked as today when the material does not run out."""
    ids = [a["id"] for a in sc.areas]
    if not ids:
        return {}
    grid = {}
    for j, e in enumerate(sc.excavators):
        allowed = [a for a in ids if e.may_work(a)] or ids
        grid[e.id] = [f"{near.get(e.id) if near.get(e.id) in allowed else allowed[j % len(allowed)]}/1"]
    try:
        probe = workplan.scenario_raw(raw, {"start": {a: 1e6 for a in ids}}, grid, cap)
    except (ValueError, KeyError):
        return {}
    probe["blasting"]["enabled"] = False  # an ordinary shift, without a blast
    r, hours = _rates(yaml_dump(probe))
    return {k: v * hours for k, v in r.items()}


def _work(sc, raw: dict, n: int, cap: float) -> tuple[dict, dict]:
    """The plan being edited (kept in the session), fitted to the period; a first one when there is none."""
    start, near = workplan.nearest_areas(raw.get("site") or {})
    rates = rates_per_shift(raw, sc, near, cap)
    trucks = [f.id for f in sc.trucks if f.id and f.count]
    w = st.session_state.get("work")
    if w is None or set(w.get("ex") or {}) != {e.id for e in sc.excavators} or set(w.get("tr") or {}) != set(trucks):
        if not start:
            start = {a["id"]: area_blast_t(a, cap) for a in sc.areas}
        w = workplan.default(sc.excavators, trucks, sc.areas, rates, n, start, near, cap)
    w = workplan.fit(w, n)
    st.session_state.work = w
    return w, rates


# ------------------------------------------------------------------ screen
def step_plan() -> None:
    raw = state.raw()
    sc = state.scenario()
    p = plan_settings()
    head = st.columns([4, 1.6], vertical_alignment="bottom")
    with head[0]:
        theme.header("Harmonogram miesiąca", "Ustawiasz pracę: koparka – obszar i tony, wozidło – do której koparki. "
                                             "Usypy, wiercenie i strzały program układa sam.")
    if not sc.excavators or not sc.areas:
        st.info("Najpierw narysuj kopalnię w kroku **① Kopalnia**: drogi, kruszarkę, koparki i obszary wydobycia.")
        return
    cap = sc.blasting.max_blast_t()

    # ---------------------------------------------------------------- period
    with card("period"):
        c = st.columns([0.7, 1.3, 0.7, 0.9, 1.0, 1.1, 1.1, 1.4], vertical_alignment="bottom")
        c[0].markdown('<div class="k2-h4" style="padding-bottom:10px">Okres</div>', unsafe_allow_html=True)
        p["start"] = c[1].date_input("Start", dt.date.fromisoformat(p["start"]), format="DD.MM.YYYY", key="k2_start").isoformat()
        p["days"] = c[2].number_input("Liczba dni", 1, 92, int(p["days"]), key="k2_days")
        p["shifts"] = c[3].selectbox("Zmian na dobę", [1, 2, 3], index=int(p["shifts"]) - 1, key="k2_shifts")
        p["sat"] = c[4].checkbox("Soboty robocze", bool(p["sat"]), key="k2_sat")
        p["hol"] = c[5].text_input("Dni wolne", p["hol"], key="k2_hol", placeholder="11.11, 24.12")
        p["real_t"] = c[6].number_input("Realnie na zmianę [t]", 0.0, 50_000.0, float(p["real_t"]), 100.0, format="%.0f",
                                        key="k2_real", help="Ile ton robicie realnie na zmianę – program sprawdzi, czy "
                                                            "model daje tyle samo.")
        real_slot = c[7].empty()
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
    work, rates = _work(sc, raw, n, cap)
    model_t = sum(rates.values())
    with real_slot.container():
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

    # ---------------------------------------------------------------- areas, drilling
    rigs = [r for r in (raw.get("drills") or []) if r.get("in_work", True)]
    c1, c2 = st.columns([3, 1.25])
    with c1, card("areas"):
        theme.h4("Obszary", "ile urobku leży na start, wielkość odstrzału, zasoby")
        _areas_editor(raw, sc, rigs, int(p["fire"]), work)
    with c2, card("drill"):
        theme.h4("Wiercenie i strzelanie")
        theme.note((" · ".join(f"{r['id']}: <b>{float(r.get('m_per_shift') or 250):.0f} mb/zm.</b>" for r in rigs)
                    + " (z Maszyn)") if rigs else "Brak wiertnic w pracy – odstrzały bez wiercenia (Maszyny).")
        names = list(FIRE)
        cur = next((k for k, x in FIRE.items() if x == int(p["fire"])), names[0])
        p["fire"] = FIRE[st.selectbox("Urobek dostępny", names, index=names.index(cur), key="k2_fire")]
        _blast_hours(raw)
        theme.note("W dniu strzału wozidła i koparki schodzą z wyrobiska na czas ewakuacji i strzału; przejazd koparki "
                   "na nowy usyp jest wliczony.")

    # ---------------------------------------------------------------- the grid
    ev = workplan.evaluate(work, sc.areas, rigs, n, cap, int(p["fire"]), rates)
    total = sum(ev["planned"])
    target = float(p["target_t"] or 0)
    if target:
        pct = 100 * total / target
        target_slot.markdown(theme.chip(f"plan: {total:,.0f} t · {pct:.0f} % celu".replace(",", " "), ok=pct >= 100),
                             unsafe_allow_html=True)
    else:
        target_slot.markdown(f'<div class="k2-note" style="padding-bottom:10px">plan: <b>{total:,.0f} t</b> '
                             "(suma ton koparek)</div>".replace(",", " "), unsafe_allow_html=True)

    main, side = st.columns([3.2, 1])
    with main, card("grid"):
        h = st.columns([2.2, 1.4, 1.6], vertical_alignment="center")
        h[0].markdown('<div class="k2-h4">Harmonogram</div>', unsafe_allow_html=True)
        with h[1]:
            day_view = st.radio("Dokładność", ["Dzień", "Zmiana"], horizontal=True, key="k2seg_view",
                                label_visibility="collapsed") == "Dzień"
        if h[2].button("↺ Ułóż propozycję od nowa", key="k2_repropose", width="stretch"):
            st.session_state.pop("work", None)
            st.rerun()
        data = _grid_payload(work, ev, rigs, shifts, lab, day_view, sc, rates, target, int(p["shifts"]))
        val = grid_component(data=data, key="k2_gridc", default=None)
        if val and val.get("nonce") != st.session_state.get("k2_grid_nonce"):
            st.session_state.k2_grid_nonce = val["nonce"]
            a, b = _span(int(val["a"]), int(val["b"]), val, shifts)
            if val.get("kind") == "tr":
                _set_trucks(val["ids"], a, b, val["ex"])
            else:
                _set_ex(val["ex"], a, b, val["area"], float(val.get("t") or 0))
            st.rerun()
        with st.expander("Edycja zakresem (bez myszy)"):
            _editor(work, sc, lab, n)
    with side, card("issues"):
        short = target - total if target else 0.0
        issues = ev["issues"]
        theme.h4(f"Do poprawy · {len(issues) + (1 if short > 0 else 0)}")
        if short > 0:
            st.markdown(f'<div class="k2-issue"><b>Plan nie zrobi celu – brakuje {short:,.0f} t</b><br>'
                        "Dołóż ton koparkom, zmianę na dobę albo soboty.</div>".replace(",", " "), unsafe_allow_html=True)
        if not issues and short <= 0:
            theme.note("Wszystko się zgadza: urobek jest na czas, każda koparka z tonami ma wozidła.")
        for j, it in enumerate(issues):
            _issue(it, work, lab, n, j)

    sig = hashlib.md5(repr((state.raw_yaml(), work, p)).encode()).hexdigest()
    appr = st.session_state.get("approved")
    with head[1]:
        if appr and appr["key"] == sig:
            st.markdown(theme.chip("Harmonogram zatwierdzony"), unsafe_allow_html=True)
        if st.button("Zatwierdź i porównaj z RL →", type="primary", key="k2_approve", width="stretch"):
            try:
                sim_raw = workplan.scenario_raw(raw, work, ev["grid"], cap)
            except ValueError as e:
                st.error(f"Nie da się policzyć tego planu: {e}")
                return
            st.session_state.approved = {"key": sig, "frozen": workplan.frozen(work, ev, n), "plan": dict(p),
                                         "work": copy.deepcopy(work), "raw": yaml_dump(sim_raw), "planned_t": total}
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


def _blast_hours(raw: dict) -> None:
    """Hours of the blasting works on a blast day (the scenario's `blasting`)."""
    bl = raw.get("blasting") or {}

    def tm(key, default):
        try:
            return dt.datetime.strptime(str(bl.get(key) or default), "%H:%M").time()
        except ValueError:
            return dt.datetime.strptime(default, "%H:%M").time()
    v = st.session_state.get("ver", 0)
    c = st.columns(2)
    new = {"crew_arrive": c[0].time_input("Ekipa strzałowa od", tm("crew_arrive", "09:00"), key=f"k2_bl_ca_{v}",
                                          step=dt.timedelta(minutes=15)),
           "crew_leave": c[1].time_input("Ekipa do", tm("crew_leave", "11:40"), key=f"k2_bl_cl_{v}",
                                         step=dt.timedelta(minutes=15)),
           "fire_from": c[0].time_input("Strzał od", tm("fire_from", "11:00"), key=f"k2_bl_ff_{v}",
                                        step=dt.timedelta(minutes=5)),
           "fire_to": c[1].time_input("Strzał do", tm("fire_to", "11:20"), key=f"k2_bl_ft_{v}",
                                      step=dt.timedelta(minutes=5))}
    new = {k: x.strftime("%H:%M") for k, x in new.items()}
    new["evacuation_min"] = float(c[0].number_input("Ewakuacja przed [min]", 0.0, 120.0,
                                                    float(bl.get("evacuation_min") or 10.0), 5.0, format="%.0f",
                                                    key=f"k2_bl_ev_{v}"))
    if any(str(bl.get(k)) != str(x) for k, x in new.items() if k != "evacuation_min") or \
            float(bl.get("evacuation_min") or 10.0) != new["evacuation_min"]:
        r = copy.deepcopy(raw)
        r.setdefault("blasting", {}).update(new)
        state.set_raw(r)
        st.rerun()


def _areas_editor(raw: dict, sc, rigs: list, fire: int, work: dict) -> None:
    mps = max([float(r.get("m_per_shift") or 250) for r in rigs] or [250.0])
    cap = sc.blasting.max_blast_t()
    rows = []
    for a in sc.areas:
        t = area_blast_t(a, cap)
        m = t / float(a.get("t_per_m") or 50)
        d = 0 if not rigs else -(-m // mps)
        rows.append({"obszar": a["id"], "poziom": a.get("level_m"), "start_t": float(work["start"].get(a["id"], 0.0)),
                     "odstrzal_t": float(a.get("blast_max_t") or 20000), "tpm": float(a.get("t_per_m") or 50),
                     "zasoby_t": float(a.get("reserves_t") or 0),
                     "wierc_od": None if a.get("drill_below_t") is None else float(a["drill_below_t"]),
                     "mb": round(m), "wiercenie": f"{d:.0f} zm.",
                     "do_urobku": f"{d + fire:.0f} zm."})
    ver = f"{st.session_state.get('ver', 0)}_{st.session_state.get('k2_area_ver', 0)}"
    ed = st.data_editor(pd.DataFrame(rows), key=f"k2_areas_{ver}", hide_index=True, width="stretch",
                        disabled=["obszar", "poziom", "mb", "wiercenie", "do_urobku"],
                        column_config={"obszar": "obszar", "poziom": st.column_config.NumberColumn("poziom [m]", format="%.0f"),
                                       "start_t": st.column_config.NumberColumn("leży na start [t]", min_value=0.0,
                                                                                step=1000.0, format="%.0f"),
                                       "odstrzal_t": st.column_config.NumberColumn("odstrzał [t]", min_value=500.0,
                                                                                   step=500.0, format="%.0f"),
                                       "tpm": st.column_config.NumberColumn("t/mb", min_value=5.0),
                                       "zasoby_t": st.column_config.NumberColumn("zasoby [t]", min_value=0.0, step=5000.0,
                                                                                 format="%.0f"),
                                       "wierc_od": st.column_config.NumberColumn(
                                           "wierć, gdy zostaje [t]", min_value=0.0, step=1000.0, format="%.0f",
                                           help="Wiercenie następnego odstrzału zaczyna się, gdy w obszarze zostaje do "
                                                "wybrania tyle ton. Puste = program wierci w ostatniej chwili, tak by "
                                                "urobek był na czas."),
                                       "mb": "mb wiercenia", "wiercenie": "wiercenie", "do_urobku": "od wiercenia do urobku"})
    theme.note("Na start leży usyp <b>/1</b> (np. O1/1). Każdy kolejny odstrzał to następny usyp: O1/2, O1/3 …"
               + (f" · limit MW: największy strzał {cap:,.0f} t".replace(",", " ") if cap < float("inf") else ""))
    started = False
    for _, r in ed.iterrows():
        if float(r["start_t"] or 0) != float(work["start"].get(r["obszar"], 0.0)):
            work["start"][r["obszar"]] = float(r["start_t"] or 0)
            started = True
    if started:
        st.session_state.k2_area_ver = st.session_state.get("k2_area_ver", 0) + 1
        st.rerun()
    changed = False
    new = copy.deepcopy(raw)
    for _, r in ed.iterrows():
        upd = {"blast_max_t": float(r["odstrzal_t"]), "t_per_m": float(r["tpm"]), "reserves_t": float(r["zasoby_t"])}
        below = None if pd.isna(r["wierc_od"]) else float(r["wierc_od"])
        for src in (new.get("areas") or []) + ((new.get("site") or {}).get("areas") or []):
            if src.get("id") != r["obszar"]:
                continue
            if any(float(src.get(k) or 0) != val for k, val in upd.items()) or src.get("drill_below_t") != below:
                src.update(upd)
                if below is None:
                    src.pop("drill_below_t", None)
                else:
                    src["drill_below_t"] = below
                changed = True
    if changed:
        state.set_raw(new)
        st.rerun()


# ------------------------------------------------------------------ the grid
def _num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _kt(x: float) -> str:
    return f"{x / 1000:.1f}".rstrip("0").rstrip(".") if x < 10_000 else f"{x / 1000:.0f}"


def _grid_payload(work, ev, rigs, shifts, lab, day_view, sc, rates, target, spd) -> dict:
    """Everything the interactive grid draws: columns (days or shifts), rows with coloured cells, and what the lists
    of the excavator / truck cells offer."""
    n = len(shifts)
    grid, blasts, left = ev["grid"], ev["blasts"], ev["left"]
    area_ids = [a["id"] for a in sc.areas]
    acol = {a: AREA_COLORS[j % len(AREA_COLORS)] for j, a in enumerate(area_ids)}
    ex_ids = list(work["ex"])
    ecol = {x: EX_COLORS[j % len(EX_COLORS)] for j, x in enumerate(ex_ids)}
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

    def cell(text, bg, fg, title="", red=False, bold=False, **kw):
        return {"text": text, "bg": bg, "fg": fg, "title": title, "red": red, "bold": bold, **kw}

    trucks_on = [{x: sum(1 for cells in work["tr"].values() if cells[i] == x) for x in ex_ids} for i in range(n)]
    bad: dict[tuple[str, int], str] = {}
    for it in ev["issues"]:
        k = it["kind"]
        if k == workplan.LATE:
            for i in range(it["shift"], min(it["avail"], n)):
                if grid[it["ex"]][i] == it["pile"]:
                    bad[(it["ex"], i)] = f"{it['pile']} będzie dopiero od {lab[min(it['avail'], n - 1)]}"
        elif k == workplan.EXHAUSTED:
            bad[(it["ex"], it["shift"])] = f"obszar {it['area']} bez zasobów"
    for x in ex_ids:
        for i in range(n):
            a, t = workplan.cell(work, x, i)
            if t and rates.get(x) and t > 1.1 * rates[x]:
                bad.setdefault((x, i), f"więcej niż ładuje (ok. {_num(rates[x])} t na zmianę)")
            if t and not trucks_on[i][x]:
                bad.setdefault((x, i), "nie jeździ do niej żadne wozidło")

    rows = [{"kind": "group", "name": "KOPARKI – obszar i tony · ustawiasz Ty"}]
    for x in ex_ids:
        out = []
        for g in groups:
            cs = [workplan.cell(work, x, i) for i in g]
            tot = sum(t for _, t in cs)
            areas = list(dict.fromkeys(a for a, _ in cs if a))
            why = next((bad[(x, i)] for i in g if (x, i) in bad), "")
            per = round(tot / len(g)) if g else 0
            vals = {"a": areas[0] if areas else "", "t": per}
            if not areas:
                out.append(cell(STAND, "#F3F4F6", "#5B6573", f"{x}: postój" + (f" – {why}" if why else ""),
                                red=bool(why), v=vals))
                continue
            bg, fg = acol.get(areas[-1], ("#FFFFFF", "#1B2733"))
            piles = ", ".join(dict.fromkeys(grid[x][i] for i in g if grid[x][i]))
            k = max(trucks_on[i][x] for i in g)
            title = (f"{x}: {'/'.join(areas)} – {_num(tot)} t ({_num(per)} t na zmianę) · ładuje z {piles} · "
                     f"wozideł: {k}" + (f" – {why}" if why else ""))
            out.append(cell(f"{'/'.join(areas)} {_kt(tot)}", bg, fg, title, red=bool(why), v=vals))
        rows.append({"kind": "ex", "name": x, "cells": out})
    rows.append({"kind": "group", "name": "WOZIDŁA – do której koparki · ustawiasz Ty"})
    for tid, cells in work["tr"].items():
        out = []
        for g in groups:
            xs = list(dict.fromkeys(cells[i] for i in g))
            idle = [cells[i] for i in g if cells[i] and workplan.cell(work, cells[i], i)[1] <= 0]
            if xs == [""]:
                out.append(cell("—", "#F3F4F6", "#8A93A0", f"{tid}: nie jeździ"))
                continue
            bg, fg = ecol.get(next(x for x in xs if x), ("#FFFFFF", "#1B2733"))
            text = "/".join(x or "—" for x in xs)
            out.append(cell(text, bg, fg, f"{tid}: jeździ do {text}" + (f" – {idle[0]} stoi" if idle else ""),
                            red=bool(idle)))
        rows.append({"kind": "tr", "name": tid, "cells": out})
    for x in ex_ids:
        out = []
        for g in groups:
            k = trucks_on[g[0]][x]
            out.append(cell(str(k) if k else "–", "#FFFFFF", "#3A4452" if k else "#8A93A0", f"do {x}: {k} wozideł"))
        rows.append({"kind": "sum", "name": f"wozideł do {x}", "cells": out})
    if rigs:
        rows.append({"kind": "group", "name": "WIERTNICE – układa program"})
        for r in rigs:
            out = []
            for g in groups:
                jobs = list(dict.fromkeys(p for p, b in blasts.items() for (rid, x) in b["drill"] if rid == r["id"] and x in g))
                out.append(cell(", ".join(jobs) or "–", "#EDEFF2" if jobs else "#FFFFFF", "#3A4452" if jobs else "#8A93A0",
                                f"{r['id']}: " + ("wierci " + ", ".join(jobs) if jobs else "wolna")))
            rows.append({"kind": "rig", "name": r["id"], "cells": out})
    rows.append({"kind": "group", "name": "USYPY I ODSTRZAŁY – układa program"})

    def order(p):
        a, k = p.rsplit("/", 1)
        return (area_ids.index(a) if a in area_ids else 99, int(k))
    for p in sorted(left, key=order):
        b = blasts.get(p)
        out = []
        for g in groups:
            if b and any(x in g for x in b["fire"]):
                out.append(cell("strzał", "#1B2733", "#FFFFFF", f"{p}: strzał", bold=True))
            elif b and any(x in g for _, x in b["drill"]):
                out.append(cell("wierc.", "#EDEFF2", "#3A4452", f"{p}: wiercenie"))
            elif b and all(x < b["avail"] for x in g):
                out.append(cell("", "#FFFFFF", "#FFFFFF"))
            else:
                lft = left[p][g[0]]
                out.append(cell(f"{_kt(lft)} kt" if lft > 1 else "wybrany", "#E3F1E6" if lft > 1 else "#F3F4F6",
                                "#1E5A2C" if lft > 1 else "#5B6573", f"{p}: leży ok. {_num(lft)} t"))
        name = f"{p} · na start" if not b else f"{p} · {_kt(b['t'])} kt · {b['m']} mb"
        rows.append({"kind": "pile", "name": name, "cells": out})
    planned = ev["planned"]
    rows.append({"kind": "group", "name": "PLAN NARASTAJĄCO" + (f" · cel {_kt(target)} kt" if target else "")})
    out, acc = [], 0.0
    for g in groups:
        acc += sum(planned[i] for i in g)
        due = target * (g[-1] + 1) / n if target else 0.0
        ok = not target or acc >= due * 0.98
        out.append(cell(f"{acc / 1000:.1f}", "#E3F1E6" if ok else "#FFF0B8", "#1E5A2C" if ok else "#6E5600",
                        f"do końca: {_num(acc)} t" + (f" · w tempie celu powinno być {_num(due)} t" if target else ""),
                        bold=True))
    rows.append({"kind": "sum", "name": "wywiezione [kt]", "cells": out})
    areas = []
    for a in sc.areas:
        info = f"na start {_kt(work['start'].get(a['id'], 0.0))} kt · odstrzał {_kt(area_blast_t(a, sc.blasting.max_blast_t()))} kt"
        areas.append({"id": a["id"], "info": info, "bg": acol[a["id"]][0], "fg": acol[a["id"]][1]})
    exs = [{"id": e.id, "allowed": [a for a in area_ids if e.may_work(a)] or area_ids, "rate": round(rates.get(e.id, 0.0)),
            "bg": ecol[e.id][0], "fg": ecol[e.id][1]} for e in sc.excavators if e.id in ecol]
    return {"day": day_view, "spd": spd, "heads": heads, "groups": groups, "labels": lab, "off": [False] * len(heads),
            "rows": rows, "areas": areas, "exs": exs}


# ------------------------------------------------------------------ editing
def _span(a: int, b: int, val: dict, shifts: list) -> tuple[int, int]:
    """The shifts a change from the list goes to: the marked ones, all of them, or the marked ones with `fwd` working
    days after and `back` before."""
    if val.get("scope") == "all":
        return 0, len(shifts) - 1
    if val.get("scope") == "around":
        lo_day = shifts[a][1] - int(val.get("back") or 0)
        hi_day = shifts[b][1] + int(val.get("fwd") or 0)
        days = [i for i, (_, wd, _) in enumerate(shifts) if lo_day <= wd <= hi_day]
        return days[0], days[-1]
    return a, b


def _set_ex(ex: str, a: int, b: int, area: str, t_shift: float) -> None:
    w = st.session_state.work
    for i in range(a, b + 1):
        if i < len(w["ex"][ex]):
            w["ex"][ex][i] = {"a": area if area != STAND else "", "t": t_shift if area and area != STAND else 0.0}


def _set_trucks(ids: list[str], a: int, b: int, ex: str) -> None:
    w = st.session_state.work
    for tid in ids:
        for i in range(a, b + 1):
            if tid in w["tr"] and i < len(w["tr"][tid]):
                w["tr"][tid][i] = "" if ex == OFF else ex


def _editor(work, sc, lab, n) -> None:
    theme.h4("Koparka")
    c = st.columns([1, 1.4, 1.4, 1.2, 1.2, 1])
    ex = c[0].selectbox("Koparka", list(work["ex"]), key="k2_e_ex")
    a = c[1].selectbox("Od zmiany", range(n), format_func=lambda i: lab[i], key="k2_e_from")
    b = c[2].selectbox("Do zmiany", range(n), index=n - 1, format_func=lambda i: lab[i], key="k2_e_to")
    area = c[3].selectbox("Obszar", [a_["id"] for a_ in sc.areas] + [STAND], key="k2_e_area")
    t = c[4].number_input("Ton na zmianę", 0.0, 20_000.0, 1000.0, 100.0, format="%.0f", key="k2_e_t")
    with c[5]:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("Ustaw", type="primary", key="k2_e_set", width="stretch"):
            _set_ex(ex, min(a, b), max(a, b), area, t)
            st.rerun()
    theme.h4("Wozidła")
    c = st.columns([2.2, 1.4, 1.4, 1.2, 1])
    ids = c[0].multiselect("Wozidła", list(work["tr"]), key="k2_t_ids")
    a = c[1].selectbox("Od zmiany", range(n), format_func=lambda i: lab[i], key="k2_t_from")
    b = c[2].selectbox("Do zmiany", range(n), index=n - 1, format_func=lambda i: lab[i], key="k2_t_to")
    x = c[3].selectbox("Jeździ do", list(work["ex"]) + [OFF], key="k2_t_ex")
    with c[4]:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("Ustaw", type="primary", key="k2_t_set", width="stretch", disabled=not ids):
            _set_trucks(ids, min(a, b), max(a, b), x)
            st.rerun()


def _issue(it, work, lab, n, j) -> None:
    k, ex, i = it["kind"], it.get("ex"), it["shift"]
    many = f" ({it['shifts']} zmian)" if it.get("shifts", 1) > 1 else ""
    if k == workplan.LATE:
        st.markdown(f'<div class="k2-issue"><b>{ex} · {lab[i]}: {it["pile"]} będzie dopiero od '
                    f'{lab[min(it["avail"], n - 1)]}</b><br>Wiercenie i strzał nie zdążą – wiertnice zajęte albo za mało '
                    "czasu. Zmniejsz tony wcześniej albo zacznij w innym obszarze.</div>", unsafe_allow_html=True)
    elif k == workplan.EXHAUSTED:
        st.markdown(f'<div class="k2-issue"><b>Obszar {it["area"]} kończy się od {lab[i]}</b><br>Zasoby nie wystarczą na '
                    "zaplanowane tony – zwiększ zasoby w tabeli obszarów albo przenieś koparkę.</div>",
                    unsafe_allow_html=True)
    elif k == workplan.OVER:
        r = max(100.0, round(it["rate"], -2))
        st.markdown(f'<div class="k2-issue w"><b>{ex}: {_num(it["t"])} t na zmianę{many}</b><br>Realnie ładuje ok. '
                    f'{_num(it["rate"])} t – reszta planu nie zostanie wywieziona.</div>', unsafe_allow_html=True)
        if st.button(f"Ustaw {ex} na {_num(r)} t", key=f"k2_fix_{j}", width="stretch"):
            for s in range(n):
                c = work["ex"][ex][s]
                if c.get("a") and float(c.get("t") or 0) > 1.1 * it["rate"]:
                    c["t"] = r
            st.rerun()
    elif k == workplan.NO_TRUCKS:
        st.markdown(f'<div class="k2-issue"><b>{ex} od {lab[i]}: brak wozideł{many}</b><br>Koparka ma tony, ale nie '
                    "jeździ do niej żadne wozidło.</div>", unsafe_allow_html=True)
    elif k == workplan.IDLE_TRUCK:
        st.markdown(f'<div class="k2-issue w"><b>{it["truck"]} od {lab[i]} jeździ do {ex}, a {ex} stoi{many}</b><br>'
                    "Wozidło nie ma skąd ładować.</div>", unsafe_allow_html=True)
        if st.button(f"{it['truck']}: nie jeździ, gdy {ex} stoi", key=f"k2_fix_{j}", width="stretch"):
            cells = work["tr"][it["truck"]]
            for s in range(n):
                if cells[s] == ex and workplan.cell(work, ex, s)[1] <= 0:
                    cells[s] = ""
            st.rerun()
