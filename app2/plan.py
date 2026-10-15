"""② Harmonogram: the planner sets the work – on every day, in which area every excavator loads and how many tonnes,
and to which excavator every truck drives (or that it stands). The program numbers the piles of every area, adds
the drilling and blasting of every new pile in time, and shows what does not add up. The approved schedule is frozen:
the month with today's driving and the month with the driving signals both follow it."""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import math

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
NO_ORDER = "bez zlecenia"


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


@st.cache_data(show_spinner=False, max_entries=8)
def _travel(site_yaml: str, rigs_yaml: str) -> dict:
    import yaml

    return workplan.area_travel_h(yaml.safe_load(site_yaml), yaml.safe_load(rigs_yaml) or [])


def travel_h(raw: dict, rigs: list) -> dict:
    """Hours a rig drives between the areas along the roads of the map."""
    if not raw.get("site"):
        return {}
    return _travel(yaml_dump(raw["site"]), yaml_dump(rigs))


def _work(sc, raw: dict, n: int, cap: float, rigs: list, fire: int) -> tuple[dict, dict]:
    """The plan being edited (kept in the session), fitted to the period; a first one when there is none."""
    start, near = workplan.nearest_areas(raw.get("site") or {})
    rates = rates_per_shift(raw, sc, near, cap)
    trucks = [f.id for f in sc.trucks if f.id and f.count]
    w = st.session_state.get("work")
    if w is None or set(w.get("ex") or {}) != {e.id for e in sc.excavators} or set(w.get("tr") or {}) != set(trucks):
        if not start:
            start = {a["id"]: area_blast_t(a, cap) for a in sc.areas}
        w = workplan.default(sc.excavators, trucks, sc.areas, rates, n, start, near, cap, rigs, fire)
    w = workplan.fit(w, n)
    if set(w["dr"]) != {r["id"] for r in rigs}:  # rigs added / removed (or a plan from before the rigs were planned)
        w["dr"] = workplan.propose_drilling(w, sc.areas, rigs, n, cap, fire)
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
    rigs = [r for r in (raw.get("drills") or []) if r.get("in_work", True)]
    work, rates = _work(sc, raw, n, cap, rigs, int(p["fire"]))
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
    c1, c2 = st.columns([3, 1.25])
    with c1, card("areas"):
        theme.h4("Obszary", "ile urobku leży na start, wielkość odstrzału, zasoby")
        _areas_editor(raw, sc, rigs, int(p["fire"]), work)
    with c2, card("drill"):
        theme.h4("Wiercenie i strzelanie")
        theme.note((" · ".join(f"{r['id']}: <b>{float(r.get('m_per_shift') or 250):.0f} mb/zm.</b>, "
                              f"{float(r.get('speed_kmh') or 2.0):.1f} km/h" for r in rigs)
                    + " (z Maszyn)") if rigs else "Brak wiertnic w pracy – dodaj je w Maszynach.")
        names = list(FIRE)
        cur = next((k for k, x in FIRE.items() if x == int(p["fire"])), names[0])
        p["fire"] = FIRE[st.selectbox("Urobek dostępny", names, index=names.index(cur), key="k2_fire")]
        _blast_hours(raw)
        theme.note("W dniu strzału wozidła i koparki schodzą z wyrobiska na czas ewakuacji i strzału; przejazd koparki "
                   "na nowy usyp jest wliczony.")

    # ---------------------------------------------------------------- the grid
    ev = workplan.evaluate(work, sc.areas, rigs, n, cap, int(p["fire"]), rates, travel_h(raw, rigs), sc.shift_hours)
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
            if val.get("kind") == "fire":
                _move_fire(val["pile"], val.get("delta"))
            else:
                a, b = _span(int(val["a"]), int(val["b"]), val, shifts)
                if val.get("kind") == "tr":
                    _set_trucks(val["ids"], a, b, val["ex"])
                else:
                    if val.get("kind") == "dr":
                        _set_ex(val["ex"], a, b, val["area"], float(val.get("t") or 0), "dr",
                                [d for d, _, _ in shifts] if day_view else None)
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
            theme.note("Wszystko się zgadza: urobek jest odstrzelony na czas, każda koparka z tonami ma wozidła.")
        for j, it in enumerate(issues):
            _issue(it, work, lab, n, j, ev)

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
    cap = sc.blasting.max_blast_t()
    rows = []
    for a in sc.areas:
        t = area_blast_t(a, cap)
        rows.append({"obszar": a["id"], "poziom": a.get("level_m"), "start_t": float(work["start"].get(a["id"], 0.0)),
                     "odstrzal_t": float(a.get("blast_max_t") or 20000), "tpm": float(a.get("t_per_m") or 50),
                     "zasoby_t": float(a.get("reserves_t") or 0), "mb": round(t / float(a.get("t_per_m") or 50)),
                     "wiert": " · ".join(f"{r['id']} {_kt(workplan.rig_rate(r, a))} kt" for r in rigs) or "–"})
    ver = f"{st.session_state.get('ver', 0)}_{st.session_state.get('k2_area_ver', 0)}"
    ed = st.data_editor(pd.DataFrame(rows), key=f"k2_areas_{ver}", hide_index=True, width="stretch",
                        disabled=["obszar", "poziom", "mb", "wiert"],
                        column_config={"obszar": "obszar", "poziom": st.column_config.NumberColumn("poziom [m]", format="%.0f"),
                                       "start_t": st.column_config.NumberColumn("leży na start [t]", min_value=0.0,
                                                                                step=1000.0, format="%.0f"),
                                       "odstrzal_t": st.column_config.NumberColumn("odstrzał [t]", min_value=500.0,
                                                                                   step=500.0, format="%.0f"),
                                       "tpm": st.column_config.NumberColumn("t/mb", min_value=5.0),
                                       "zasoby_t": st.column_config.NumberColumn("zasoby [t]", min_value=0.0, step=5000.0,
                                                                                 format="%.0f"),
                                       "mb": "mb na 1 odstrzał",
                                       "wiert": st.column_config.TextColumn(
                                           "wiertnica na zmianę", help="Ile ton wiertnica odwierci w tym obszarze przez "
                                                                       "całą zmianę (mb/zm. × t/mb).")})
    theme.note("Na start leży usyp <b>/1</b> (np. O1/1). Odwiercone tony strzela się, gdy uzbiera się odstrzał – albo "
               "gdy wiertnice schodzą z obszaru; każdy strzał to następny usyp: O1/2, O1/3 …"
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
        for src in (new.get("areas") or []) + ((new.get("site") or {}).get("areas") or []):
            if src.get("id") == r["obszar"] and any(float(src.get(k) or 0) != val for k, val in upd.items()):
                src.update(upd)
                src.pop("drill_below_t", None)
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
    of the rig / excavator / truck / blast cells offer."""
    n = len(shifts)
    grid, blasts = ev["grid"], ev["blasts"]
    area_ids = [a["id"] for a in sc.areas]
    by_area = {a["id"]: a for a in sc.areas}
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

    def work_row(key, part, why_of):
        """A row the planner sets: area and tonnes on every day / shift."""
        out = []
        for g in groups:
            cs = [workplan.cell(work, key, i, part) for i in g]
            tot = sum(t for _, t in cs)
            areas = list(dict.fromkeys(a for a, _ in cs if a))
            why = next((w for w in (why_of(i) for i in g) if w), "")
            per = round(tot / len(g)) if g else 0
            vals = {"a": areas[0] if areas else "", "t": per}
            if not areas:
                out.append(cell(STAND, "#F3F4F6", "#5B6573", f"{key}: postój" + (f" – {why}" if why else ""),
                                red=bool(why), v=vals))
                continue
            bg, fg = acol.get(areas[-1], ("#FFFFFF", "#1B2733"))
            out.append(cell(f"{'/'.join(areas)} {_kt(tot)}", bg, fg, f"{key}: {'/'.join(areas)} – {_num(tot)} t "
                            f"({_num(per)} t na zmianę)" + (f" – {why}" if why else ""), red=bool(why), v=vals))
        return out

    trucks_on = [{x: sum(1 for cells in work["tr"].values() if cells[i] == x) for x in ex_ids} for i in range(n)]

    def ex_why(x):
        def f(i):
            a, t = workplan.cell(work, x, i)
            if not t:
                return ""
            if t - ev["quota"][x][i] > 1:
                return f"w {a} brakuje odstrzelonego urobku: dostanie {_num(ev['quota'][x][i])} t"
            if rates.get(x) and t > 1.1 * rates[x]:
                return f"więcej niż ładuje (ok. {_num(rates[x])} t na zmianę)"
            if not trucks_on[i][x]:
                return "nie jeździ do niej żadne wozidło"
            return ""
        return f

    def rig_row(rid):
        """Orders the planner gave (strong colour) and the drilling that carries them on (light, with an arrow)."""
        rr, out = ev["rig_rows"][rid], []
        for g in groups:
            orders = [rr[i]["order"] for i in g if rr[i]["order"]]
            drill: dict[str, float] = {}
            for i in g:
                for a, d in rr[i]["drill"]:
                    drill[a] = drill.get(a, 0.0) + d
            drive = sum(rr[i]["drive_h"] for i in g)
            wait = rr[g[-1]]["waiting"]
            title = (f"{rid}: " + (f"zlecenie {', '.join(f'{a} {_num(t)} t' for a, t in orders)} · " if orders else "")
                     + (f"wierci {', '.join(f'{a} {_num(d)} t' for a, d in drill.items())}" if drill else "nie wierci")
                     + (f" · przejazd {drive * 60:.0f} min" if drive else "") + (f" · zostaje {_num(wait)} t" if wait else ""))
            vals = {"a": orders[0][0] if orders else "", "t": orders[0][1] if orders else 0}
            if orders:
                bg, fg = acol.get(orders[-1][0], ("#FFFFFF", "#1B2733"))
                out.append(cell(" + ".join(f"{a} {_kt(t)}" for a, t in orders), bg, fg, title, bold=True, v=vals))
            elif any(rr[i]["stop"] for i in g):
                out.append(cell(STAND, "#F3F4F6", "#5B6573", f"{rid}: postój", v=vals))
            elif drill:
                a = next(iter(drill))
                bg, fg = acol.get(a, ("#FFFFFF", "#1B2733"))
                out.append(cell(f"→ {'/'.join(drill)} {_kt(sum(drill.values()))}", "#FFFFFF", fg, title,
                                v=vals, bd=bg))
            else:
                out.append(cell("–", "#FFFFFF", "#8A93A0", f"{rid}: wolna", v=vals))
        return out

    rows = [{"kind": "group", "name": "WIERTNICE – zlecenia: obszar i tony · ustawiasz Ty"}]
    for r in rigs:
        rows.append({"kind": "dr", "name": r["id"], "cells": rig_row(r["id"])})
    rows.append({"kind": "group", "name": "STRZAŁY – z odwierconych ton · kliknij, żeby przesunąć"})
    for a_id in area_ids:
        out = []
        for g in groups:
            fired = [(p, b) for p, b in blasts.items() if b["area"] == a_id and any(x in g for x in b["fire"] or [b["avail"] - 1])]
            if fired:
                p, b = fired[0]
                moved = f" (przesunięty o {b['delay']} zm.)" if b["delay"] else ""
                txt = " ".join(f"{q.rsplit('/', 1)[1]}·{_kt(c['t'])}" for q, c in fired)
                out.append(cell(f"✸ {txt}", "#1B2733", "#FFFFFF",
                                f"strzał {', '.join(q for q, _ in fired)}: {_num(b['t'])} t · {b['m']} mb{moved} – "
                                f"urobek od {lab[min(b['avail'], n - 1)]}", bold=True, p=p))
            else:
                out.append(cell("", "#FFFFFF", "#FFFFFF"))
        rows.append({"kind": "fire", "name": a_id, "cells": out})
    rows.append({"kind": "group", "name": "URABEK ODSTRZELONY W OBSZARACH"})
    for a_id in area_ids:
        out = []
        for g in groups:
            lft = ev["area_left"][a_id][g[0]]
            miss = sum(ev["short"][a_id][i] for i in g)
            if miss > 1:
                out.append(cell(f"−{_kt(miss)}", "#FDE2E2", "#8C0E20", f"{a_id}: koparkom zabraknie {_num(miss)} t",
                                red=True))
            else:
                out.append(cell(f"{_kt(lft)} kt" if lft > 1 else "–", "#E3F1E6" if lft > 1 else "#FFFFFF",
                                "#1E5A2C" if lft > 1 else "#8A93A0", f"{a_id}: leży ok. {_num(lft)} t"))
        rows.append({"kind": "pile", "name": a_id, "cells": out})
    rows.append({"kind": "group", "name": "KOPARKI – obszar i tony · ustawiasz Ty"})
    for x in ex_ids:
        rows.append({"kind": "ex", "name": x, "cells": work_row(x, "ex", ex_why(x))})
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
    cap = sc.blasting.max_blast_t()
    areas = []
    for a in sc.areas:
        info = f"leży {_kt(ev['area_left'][a['id']][0])} kt · odstrzał {_kt(area_blast_t(a, cap))} kt"
        areas.append({"id": a["id"], "info": info, "bg": acol[a["id"]][0], "fg": acol[a["id"]][1]})
    exs = [{"id": e.id, "allowed": [a for a in area_ids if e.may_work(a)] or area_ids, "rate": round(rates.get(e.id, 0.0)),
            "bg": ecol[e.id][0], "fg": ecol[e.id][1]} for e in sc.excavators if e.id in ecol]
    rigs_ = [{"id": r["id"], "allowed": area_ids, "rates": {a: round(workplan.rig_rate(r, by_area[a])) for a in area_ids}}
             for r in rigs]
    blasts_ = {p: {"t": b["t"], "m": b["m"], "delay": b["delay"], "avail": lab[min(b["avail"], n - 1)],
                   "done": lab[min(b["done"], n - 1)]} for p, b in blasts.items()}
    return {"day": day_view, "spd": spd, "heads": heads, "groups": groups, "labels": lab, "off": [False] * len(heads),
            "rows": rows, "areas": areas, "exs": exs, "rigs": rigs_, "blasts": blasts_}


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


def _set_ex(key: str, a: int, b: int, area: str, t_shift: float, part: str = "ex", days: list | None = None) -> None:
    """Excavators: area and tonnes on every shift. Rigs: an order (tonnes to drill) – once per day in the day view
    (`days` = the date of every shift), "postój" stops the rig, "bez zlecenia" clears the order."""
    w = st.session_state.work
    for i in range(a, b + 1):
        if i >= len(w[part][key]):
            continue
        if part == "dr":
            first = days is None or i == a or days[i] != days[i - 1]
            if area == STAND:
                w[part][key][i] = {"a": "", "t": 0.0, "stop": True}
            elif area == NO_ORDER or not first:
                w[part][key][i] = {"a": "", "t": 0.0}
            else:
                w[part][key][i] = {"a": area, "t": t_shift}
        else:
            w[part][key][i] = {"a": area if area != STAND else "", "t": t_shift if area and area != STAND else 0.0}


def _set_trucks(ids: list[str], a: int, b: int, ex: str) -> None:
    w = st.session_state.work
    for tid in ids:
        for i in range(a, b + 1):
            if tid in w["tr"] and i < len(w["tr"][tid]):
                w["tr"][tid][i] = "" if ex == OFF else ex


def _move_fire(pile: str, delta) -> None:
    """A blast later (or back towards the program's time): delta in shifts, None = as the program puts it."""
    fire = st.session_state.work.setdefault("fire", {})
    d = 0 if delta is None else max(0, int(fire.get(pile, 0)) + int(delta))
    if d:
        fire[pile] = d
    else:
        fire.pop(pile, None)


def _editor(work, sc, lab, n) -> None:
    areas = [a_["id"] for a_ in sc.areas]
    for title, part, pre in (("Wiertnica", "dr", "k2_d"), ("Koparka", "ex", "k2_e")):
        if not work.get(part):
            continue
        theme.h4(title)
        c = st.columns([1, 1.4, 1.4, 1.2, 1.2, 1])
        key = c[0].selectbox(title, list(work[part]), key=f"{pre}_ex")
        a = c[1].selectbox("Od zmiany", range(n), format_func=lambda i: lab[i], key=f"{pre}_from")
        b = c[2].selectbox("Do zmiany", range(n), index=n - 1, format_func=lambda i: lab[i], key=f"{pre}_to")
        area = c[3].selectbox("Obszar", areas + [STAND] + ([NO_ORDER] if part == "dr" else []), key=f"{pre}_area")
        t = c[4].number_input("Ton do odwiercenia" if part == "dr" else "Ton na zmianę", 0.0, 200_000.0, 1000.0, 100.0,
                              format="%.0f", key=f"{pre}_t")
        with c[5]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("Ustaw", type="primary", key=f"{pre}_set", width="stretch"):
                _set_ex(key, min(a, b), max(a, b), area, t, part, [0] * n if part == "dr" else None)
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


def _issue(it, work, lab, n, j, ev) -> None:
    k, i = it["kind"], it["shift"]
    k_ = it.get("shifts", 1)
    many = f" ({k_} {'zmiany' if k_ % 10 in (2, 3, 4) and k_ % 100 not in (12, 13, 14) else 'zmian'})" if k_ > 1 else ""
    if k == workplan.SHORT:
        a = it["area"]
        drill = [r for r, cells in work.get("dr", {}).items() if any(workplan.cell(work, r, s, "dr")[0] == a for s in range(n))]
        st.markdown(f'<div class="k2-issue"><b>{it["ex"]} od {lab[i]}: w {a} zabraknie {_num(it["t"])} t urobku{many}</b>'
                    f"<br>Koparki mają więcej ton, niż jest odstrzelone. Dołóż wiercenia w {a} wcześniej"
                    + (f" ({', '.join(drill)} wierci tam później)" if drill else "") + " albo zmniejsz tony koparek.</div>",
                    unsafe_allow_html=True)
    elif k == workplan.EXHAUSTED:
        st.markdown(f'<div class="k2-issue"><b>Obszar {it["area"]}: od {lab[i]} wiercenie przekracza zasoby</b><br>'
                    "Zwiększ zasoby w tabeli obszarów albo przenieś wiertnicę.</div>", unsafe_allow_html=True)
    elif k == workplan.UNFINISHED:
        st.markdown(f'<div class="k2-issue w"><b>{it["rig"]}: {_num(it["t"])} t w {it["area"]} nie zostanie odwierconych '
                    "do końca okresu</b><br>Zleceń jest więcej, niż wiertnica zdąży – przesuń część na drugą wiertnicę "
                    "albo zmniejsz zlecenia.</div>", unsafe_allow_html=True)
    elif k == workplan.OVER:
        ex = it["ex"]
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
        st.markdown(f'<div class="k2-issue"><b>{it["ex"]} od {lab[i]}: brak wozideł{many}</b><br>Koparka ma tony, ale nie '
                    "jeździ do niej żadne wozidło.</div>", unsafe_allow_html=True)
    elif k == workplan.IDLE_TRUCK:
        ex = it["ex"]
        st.markdown(f'<div class="k2-issue w"><b>{it["truck"]} od {lab[i]} jeździ do {ex}, a {ex} stoi{many}</b><br>'
                    "Wozidło nie ma skąd ładować.</div>", unsafe_allow_html=True)
        if st.button(f"{it['truck']}: nie jeździ, gdy {ex} stoi", key=f"k2_fix_{j}", width="stretch"):
            cells = work["tr"][it["truck"]]
            for s in range(n):
                if cells[s] == ex and workplan.cell(work, ex, s)[1] <= 0:
                    cells[s] = ""
            st.rerun()
