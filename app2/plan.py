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
    vs = st.session_state.setdefault("variants", {})
    act = st.session_state.get("k2seg_variant") or "A"
    if act not in vs:
        act = next(iter(vs), "A")
        st.session_state.k2seg_variant = act
    w = vs.get(act, st.session_state.get("work"))
    if w is None or set(w.get("ex") or {}) != {e.id for e in sc.excavators} or set(w.get("tr") or {}) != set(trucks):
        if not start:
            start = {a["id"]: area_blast_t(a, cap) for a in sc.areas}
        w = workplan.default(sc.excavators, trucks, sc.areas, rates, n, start, near, cap, rigs, fire)
    w = workplan.fit(w, n)
    if set(w["dr"]) != {r["id"] for r in rigs}:  # rigs added / removed (or a plan from before the rigs were planned)
        w["dr"] = workplan.propose_drilling(w, sc.areas, rigs, n, cap, fire)
    st.session_state.work = vs[act] = w  # the variant being edited
    return w, rates


# ------------------------------------------------------------------ variants of the schedule
def _variant_name(name: str, first: str) -> str:
    return f"{name} (bazowy)" if name == first else name


def _copy_variant() -> None:
    vs = st.session_state.variants
    act = st.session_state.get("k2seg_variant") or next(iter(vs))
    name = next(c for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in vs)
    vs[name] = copy.deepcopy(vs[act])
    st.session_state.k2seg_variant = name


def _drop_variant() -> None:
    vs = st.session_state.variants
    act = st.session_state.get("k2seg_variant")
    if len(vs) > 1 and act in vs:
        del vs[act]
        st.session_state.k2seg_variant = next(iter(vs))


def _diff(a: dict, b: dict, lab: list) -> list[str]:
    """What a variant changes against the base one, in a few words per machine."""
    out = []
    for ar in sorted(set(a["start"]) | set(b["start"])):
        if float(a["start"].get(ar, 0)) != float(b["start"].get(ar, 0)):
            out.append(f"{ar}: na start {_num(b['start'].get(ar, 0))} t zamiast {_num(a['start'].get(ar, 0))} t")
    for part in ("dr", "ex", "tr"):
        for k in sorted(set(a.get(part) or {}) | set(b.get(part) or {})):
            ca, cb = (a.get(part) or {}).get(k) or [], (b.get(part) or {}).get(k) or []
            diff = [i for i in range(min(len(ca), len(cb))) if ca[i] != cb[i]]
            if diff:
                out.append(f"{k}: inaczej na {len(diff)} zm. (od {lab[diff[0]]})")
    if a.get("fire") != b.get("fire"):
        out.append("inne przesunięcia strzałów")
    return out or ["bez zmian"]


def _month_today(raw: dict, sc, work: dict, p: dict, rigs: list, cap: float) -> dict:
    """The month of a variant worked as today (the same luck as every other variant – only the plan differs)."""
    from kruszyk.config import scenario_from_dict
    from kruszyk.period import run_period
    from kruszyk.rl import today

    plan = period_plan(p)
    n = len(plan.shifts())
    work = workplan.fit(work, n)
    ev = workplan.evaluate(work, sc.areas, rigs, n, cap, int(p["fire"]), None, travel_h(raw, rigs), sc.shift_hours)
    sim = scenario_from_dict(workplan.scenario_raw(raw, work, ev["grid"], cap))
    plan.frozen = workplan.frozen(work, ev, n)
    df = run_period(today(sim), plan, seed=0)
    t, cost = float(df["wywiezione_t"].sum()), float(df["koszt_zl"].sum())
    return {"plan_t": sum(ev["planned"]), "t": t, "cost": cost, "zl_t": cost / t if t else 0.0,
            "fuel": float(df["paliwo_l"].sum()), "queue": float(df["kolejki_h"].sum()), "issues": len(ev["issues"])}


def _variants_card(raw, sc, p, rigs, cap, lab) -> None:
    vs = st.session_state.variants
    names = list(vs)
    first = names[0]
    with card("variants"):
        c = st.columns([0.9, 3.2, 1.0, 1.0, 1.8], vertical_alignment="center")
        c[0].markdown('<div class="k2-h4" style="margin:0">Warianty</div>', unsafe_allow_html=True)
        with c[1]:
            st.radio("Wariant", names, horizontal=True, key="k2seg_variant", label_visibility="collapsed",
                     format_func=lambda x: _variant_name(x, first))
        c[2].button("⧉ Kopia", key="k2_var_copy", on_click=_copy_variant, width="stretch",
                    help="Nowy wariant – dokładna kopia tego, który widzisz; zmieniasz w nim, co chcesz.")
        c[3].button("Usuń", key="k2_var_drop", on_click=_drop_variant, width="stretch", disabled=len(names) < 2)
        go_ = c[4].button("Porównaj warianty", type="primary", key="k2_var_cmp", width="stretch",
                          disabled=len(names) < 2,
                          help="Każdy wariant liczony jak dziś – ten sam miesiąc, pogoda i awarie; różni się tylko plan.")
        theme.note("Harmonogram poniżej to wybrany wariant. „Kopia” tworzy nowy z tym samym planem – zmieniasz w nim "
                   "wiercenie, koparki albo wozidła i porównujesz. Potem wybrany wariant zatwierdzasz do porównania z RL.")
        cache = st.session_state.setdefault("k2_cmp", {})
        sig = {k: hashlib.md5(repr((state.raw_yaml(), w, p)).encode()).hexdigest() for k, w in vs.items()}
        if go_:
            bar = st.progress(0.0, "Liczę warianty…")
            for j, (k, w) in enumerate(vs.items()):
                if cache.get(k, {}).get("sig") != sig[k]:
                    bar.progress(j / len(vs), f"Wariant {k}: miesiąc jak dziś…")
                    cache[k] = {"sig": sig[k], **_month_today(raw, sc, w, p, rigs, cap)}
            bar.empty()
        done = [k for k in names if cache.get(k, {}).get("sig") == sig[k]]
        if len(done) >= 2:
            base = cache[first] if first in done else cache[done[0]]
            rows = []
            for k in done:
                r = cache[k]
                dz, dc = r["zl_t"] - base["zl_t"], r["cost"] - base["cost"]
                rows.append([_variant_name(k, first), _num(r["plan_t"]), _num(r["t"]), f"{r['zl_t']:.2f}",
                             "–" if k == first else f"{dz:+.2f}", _num(r["cost"]),
                             "–" if k == first else f"{dc:+,.0f}".replace(",", " "), _num(r["fuel"]), _num(r["queue"]),
                             str(r["issues"])])
            st.markdown(theme.table([("Wariant", ""), ("Plan [t]", "n"), ("Wywiezione [t]", "n"), ("zł/t", "n"),
                                     ("zł/t vs A", "n"), ("Koszt [zł]", "n"), ("Koszt vs A [zł]", "n"),
                                     ("Paliwo [l]", "n"), ("Kolejki [h]", "n"), ("Do poprawy", "n")], rows,
                                    numeric=set(range(1, 10)), sans={0}), unsafe_allow_html=True)
            for k in done:
                if k != first:
                    theme.note(f"<b>{k} względem {first}:</b> " + " · ".join(_diff(vs[first], vs[k], lab)))
        elif cache and not go_:
            theme.note("Zmieniłeś coś od ostatniego porównania – kliknij „Porównaj warianty”, żeby policzyć od nowa.")


# ------------------------------------------------------------------ screen
def step_plan() -> None:
    raw = state.raw()
    sc = state.scenario()
    p = plan_settings()
    head = st.columns([4, 1.6], vertical_alignment="bottom")
    with head[0]:
        theme.header("Harmonogram miesiąca", "Ustawiasz pracę: wiertnica i koparka – obszar i tony, wozidło – do której koparki. "
                                             "Wiercenie dni, strzały i usypy program liczy sam.")
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

    _variants_card(raw, sc, p, rigs, cap, lab)

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
            st.session_state.variants[st.session_state.get("k2seg_variant") or "A"] = None
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
                                         "work": copy.deepcopy(work), "raw": yaml_dump(sim_raw), "planned_t": total,
                                         "variant": st.session_state.get("k2seg_variant") or "A"}
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
    """Hours of the blasting works on a blast day (the scenario's `blasting`): the firing at an hour and how long it
    takes – the end is worked out, so it can never come before the start."""
    bl = raw.get("blasting") or {}

    def tm(key, default) -> dt.time:
        try:
            return dt.datetime.strptime(str(bl.get(key) or default), "%H:%M").time()
        except ValueError:
            return dt.datetime.strptime(default, "%H:%M").time()

    def mins(t: dt.time) -> int:
        return t.hour * 60 + t.minute

    v = st.session_state.get("ver", 0)
    f0, f1 = mins(tm("fire_from", "11:00")), mins(tm("fire_to", "11:20"))
    c = st.columns(2)
    fire = c[0].time_input("Strzał o", tm("fire_from", "11:00"), key=f"k2_bl_ff_{v}", step=dt.timedelta(minutes=5))
    dur = int(c[1].number_input("Strzał trwa [min]", 5, 240, max(5, f1 - f0) if f1 > f0 else 20, 5,
                                key=f"k2_bl_dur_{v}"))
    crew_a = c[0].time_input("Ekipa przyjeżdża", tm("crew_arrive", "09:00"), key=f"k2_bl_ca_{v}",
                             step=dt.timedelta(minutes=15))
    crew_l = c[1].time_input("Ekipa odjeżdża", tm("crew_leave", "11:40"), key=f"k2_bl_cl_{v}",
                             step=dt.timedelta(minutes=15))
    evac = float(c[0].number_input("Ewakuacja przed strzałem [min]", 0.0, 120.0, float(bl.get("evacuation_min") or 10.0),
                                   5.0, format="%.0f", key=f"k2_bl_ev_{v}"))
    end = mins(fire) + dur
    new = {"fire_from": fire.strftime("%H:%M"), "fire_to": f"{end // 60 % 24:02d}:{end % 60:02d}",
           "crew_arrive": crew_a.strftime("%H:%M"), "crew_leave": crew_l.strftime("%H:%M"), "evacuation_min": evac}
    c[1].markdown(f'<div class="k2-note" style="padding-top:30px">strzał <b>{new["fire_from"]}–{new["fire_to"]}</b>, '
                  f'wyrobisko zamknięte od {(mins(fire) - int(evac)) // 60:02d}:{(mins(fire) - int(evac)) % 60:02d}</div>',
                  unsafe_allow_html=True)
    if mins(crew_a) > mins(fire) - evac:
        st.warning("Ekipa przyjeżdża po rozpoczęciu ewakuacji – ustaw wcześniejszy przyjazd.")
    if mins(crew_l) < end:
        st.warning("Ekipa odjeżdża przed końcem strzału – ustaw późniejszy odjazd.")
    if any(str(bl.get(k)) != str(x) for k, x in new.items() if k != "evacuation_min") or \
            float(bl.get("evacuation_min") or 10.0) != evac:
        r = copy.deepcopy(raw)
        r.setdefault("blasting", {}).update(new)
        state.set_raw(r)
        st.rerun()


def _areas_editor(raw: dict, sc, rigs: list, fire: int, work: dict) -> None:
    cap = sc.blasting.max_blast_t()
    rows = []
    for a in sc.areas:
        rows.append({"obszar": a["id"], "poziom": a.get("level_m"), "start_t": float(work["start"].get(a["id"], 0.0)),
                     "tpm": float(a.get("t_per_m") or 50), "zasoby_t": float(a.get("reserves_t") or 0),
                     "mb10": round(10000 / float(a.get("t_per_m") or 50)),
                     "wiert": " · ".join(f"{r['id']} {_tt(workplan.rig_rate(r, a))}" for r in rigs) or "–"})
    ver = f"{st.session_state.get('ver', 0)}_{st.session_state.get('k2_area_ver', 0)}"
    ed = st.data_editor(pd.DataFrame(rows), key=f"k2_areas_{ver}", hide_index=True, width="stretch",
                        disabled=["obszar", "poziom", "mb10", "wiert"],
                        column_config={"obszar": "obszar", "poziom": st.column_config.NumberColumn("poziom [m]", format="%.0f"),
                                       "start_t": st.column_config.NumberColumn("leży na start [t]", min_value=0.0,
                                                                                step=1000.0, format="%.0f"),
                                       "tpm": st.column_config.NumberColumn("t/mb", min_value=5.0),
                                       "zasoby_t": st.column_config.NumberColumn("zasoby [t]", min_value=0.0, step=5000.0,
                                                                                 format="%.0f"),
                                       "mb10": "mb na 10 tys. t",
                                       "wiert": st.column_config.TextColumn(
                                           "wiertnica na zmianę", help="Ile ton wiertnica odwierci w tym obszarze przez "
                                                                       "całą zmianę (mb/zm. × t/mb).")})
    theme.note("Na start leży usyp <b>/1</b> (np. O1/1). Każde zlecenie wiercenia to jeden strzał tej wielkości – po "
               "odwierceniu; każdy strzał to następny usyp: O1/2, O1/3 …"
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
        upd = {"t_per_m": float(r["tpm"]), "reserves_t": float(r["zasoby_t"])}
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
    """Thousands of tonnes as a planner writes them: 2,2 · 15 · 0,5."""
    v = f"{x / 1000:.1f}".rstrip("0").rstrip(".") if x < 10_000 else f"{x / 1000:.0f}"
    return v.replace(".", ",")


def _tt(x: float) -> str:
    return f"{_kt(x)} tys. t"


def _grid_payload(work, ev, rigs, shifts, lab, day_view, sc, rates, target, spd) -> dict:
    """Everything the interactive grid draws: columns (days or shifts), rows with coloured cells, and what the lists
    of the rig / excavator / truck / blast cells offer."""
    n = len(shifts)
    blasts = ev["blasts"]
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
            out.append(cell(f"{'/'.join(areas)}: {_tt(tot)}", bg, fg, f"{key}: {'/'.join(areas)} – {_num(tot)} t "
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

    def rig_rows(rid):
        """Two rows of a rig: the orders the planner gives, and what the rig really drills on every day – an order
        bigger than a day goes on into the next days, orders wait in a queue."""
        rr, orders_row, drill_row = ev["rig_rows"][rid], [], []
        for g in groups:
            orders = [rr[i]["order"] for i in g if rr[i]["order"]]
            vals = {"a": orders[0][0] if orders else "", "t": sum(t for _, t in orders)}
            if orders:
                bg, fg = acol.get(orders[-1][0], ("#FFFFFF", "#1B2733"))
                orders_row.append(cell("\n".join(f"{a}: {_tt(t)}" for a, t in orders), bg, fg,
                                       f"{rid}: zlecenie " + ", ".join(f"{a} {_num(t)} t" for a, t in orders), bold=True,
                                       v=vals))
            elif any(rr[i]["stop"] for i in g):
                orders_row.append(cell(STAND, "#F3F4F6", "#5B6573", f"{rid}: postój", v=vals))
            else:
                orders_row.append(cell("", "#FFFFFF", "#8A93A0", f"{rid}: bez nowego zlecenia", v=vals))
            drill: dict[str, float] = {}
            for i in g:
                for a, d in rr[i]["drill"]:
                    drill[a] = drill.get(a, 0.0) + d
            drive = sum(rr[i]["drive_h"] for i in g)
            wait = rr[g[-1]]["waiting"]
            title = (f"{rid} wierci: " + (", ".join(f"{a} {_num(d)} t" for a, d in drill.items()) if drill else "nic")
                     + (f" · przejazd {drive * 60:.0f} min" if drive else "")
                     + (f" · zostaje na kolejne dni {_num(wait)} t" if wait > 1 else ""))
            if drill:
                a = next(iter(drill))
                bg, fg = acol.get(a, ("#FFFFFF", "#1B2733"))
                text = "\n".join(f"{a}: {_tt(d)}" for a, d in drill.items()) + (f"\nzostaje {_tt(wait)}" if wait > 1 else "")
                drill_row.append(cell(text, "#FFFFFF", fg, title, bd=bg))
            else:
                drill_row.append(cell("–", "#FFFFFF", "#8A93A0", title))
        return orders_row, drill_row

    rows = [{"kind": "group", "name": "WIERTNICE – zlecenia ustawiasz Ty, wiercenie liczy program"}]
    for r in rigs:
        orders_row, drill_row = rig_rows(r["id"])
        rows.append({"kind": "dr", "name": f"{r['id']} · zlecenia", "id": r["id"], "cells": orders_row})
        rows.append({"kind": "drx", "name": "↳ wierci", "cells": drill_row})
    rows.append({"kind": "group", "name": "STRZAŁY – zlecenie = strzał, po odwierceniu · kliknij, żeby przesunąć"})
    bl = sc.blasting
    fire_at = f"{bl.fire_from}–{bl.fire_to}"
    for a_id in area_ids:
        out = []
        for g in groups:
            fired = [(p, b) for p, b in blasts.items() if b["area"] == a_id and any(x in g for x in b["fire"] or [b["avail"] - 1])]
            if fired:
                p, b = fired[0]
                moved = f" (przesunięty o {b['delay']} zm.)" if b["delay"] else ""
                txt = "\n".join(f"strzał {_tt(c['t'])}" for _, c in fired) + f"\n{fire_at}"
                out.append(cell(txt, "#1B2733", "#FFFFFF",
                                "\n".join(f"strzał {q} (usyp {q}): {_num(c['t'])} t · {c['m']} mb" for q, c in fired)
                                + f" · {fire_at}{moved} – urobek od {lab[min(b['avail'], n - 1)]}", bold=True, p=p))
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
                out.append(cell(f"brak {_tt(miss)}", "#FDE2E2", "#8C0E20", f"{a_id}: koparkom zabraknie {_num(miss)} t",
                                red=True))
            else:
                out.append(cell(_tt(lft) if lft > 1 else "–", "#E3F1E6" if lft > 1 else "#FFFFFF",
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
    rows.append({"kind": "group", "name": "PLAN NARASTAJĄCO" + (f" · cel {_tt(target)}" if target else "")})
    out, acc = [], 0.0
    for g in groups:
        acc += sum(planned[i] for i in g)
        due = target * (g[-1] + 1) / n if target else 0.0
        ok = not target or acc >= due * 0.98
        out.append(cell(f"{acc / 1000:.1f}".replace(".", ","), "#E3F1E6" if ok else "#FFF0B8", "#1E5A2C" if ok else "#6E5600",
                        f"do końca: {_num(acc)} t" + (f" · w tempie celu powinno być {_num(due)} t" if target else ""),
                        bold=True))
    rows.append({"kind": "sum", "name": "wywiezione [tys. t]", "cells": out})
    areas = []
    for a in sc.areas:
        info = f"leży na start {_tt(ev['area_left'][a['id']][0])}"
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
    elif k == workplan.TOO_BIG:
        st.markdown(f'<div class="k2-issue"><b>{it["rig"]} · {lab[i]}: strzał {_num(it["t"])} t w {it["area"]}</b><br>'
                    f"Limit materiału wybuchowego pozwala na ok. {_num(it['cap'])} t w jednym strzale – podziel "
                    "zlecenie na dwa.</div>", unsafe_allow_html=True)
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
