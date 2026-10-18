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
from .machines import excavator_editor, map_models, model_changed, road_editor, truck_editor, variant_raw
from .theme import card

grid_component = components.declare_component("k2_grid", path=str(Path(__file__).parent / "grid_component"))

WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ROMAN = {1: "I", 2: "II", 3: "III"}
FIRE = {"1 shift – blasted rock from the next shift": 1, "between shifts – blasted rock at once": 0, "2 shifts": 2}
AREA_COLORS = [("#FBE3CC", "#8A4B12"), ("#E7DDF5", "#5A3A8C"), ("#D6EFEA", "#11655A"), ("#F6D9E4", "#8A2650"),
               ("#FFF0B8", "#6E5600"), ("#DDF0D5", "#2F5E1E")]
EX_COLORS = [("#DCE8F7", "#1D4E89"), ("#FCE1D8", "#8A2E12"), ("#E4E9F2", "#2B3F66"), ("#E3F1E6", "#1E5A2C"),
             ("#F3E8D2", "#6B4A12")]
STAND = "standby"
OFF = "idle"
NO_ORDER = "no order"
# the same choices as older versions of the grid component send them
STAND_ALL = {STAND, "postój"}
OFF_ALL = {OFF, "nie jeździ"}
NO_ORDER_ALL = {NO_ORDER, "bez zlecenia"}


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
    if w is not None and set(w.get("ex") or {}) == {e.id for e in sc.excavators} and set(w.get("tr") or {}) != set(trucks):
        old = w.get("tr") or {}
        w["tr"] = {t: old.get(t, [""] * n) for t in trucks}  # other trucks: the new ones stand until planned
    if w is None or set(w.get("ex") or {}) != {e.id for e in sc.excavators}:
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
    return f"{name} (base)" if name == first else name


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


def _variant_machines(raw: dict, w: dict, act: str) -> None:
    """Other trucks / excavator models in this variant only (the mine in ① Kopalnia stays as it is)."""
    vr = variant_raw(raw, w)
    ver = st.session_state.get("k2_vm_ver", 0)
    theme.note("You change this variant only: the model, the number of trucks (add / delete a row) and the excavator "
               "parameters. A new truck stands until you assign it to an excavator in the schedule.")
    trucks = truck_editor(vr.get("trucks", []), f"k2_vm_tr_{act}_{ver}")
    area_ids = [x["id"] for x in (raw.get("site") or {}).get("areas", []) if x.get("id")]
    exs = excavator_editor(vr.get("excavators", []), f"k2_vm_ex_{act}_{ver}", area_ids, map_models(vr))
    c = st.columns(2)
    picked = model_changed(trucks) | model_changed(exs)  # a model picked: its parameters go in at once
    if picked or c[0].button("Apply in variant", type="primary", key=f"k2_vm_set_{act}", width="stretch"):
        w["machines"] = {"trucks": trucks, "excavators": [{k: e[k] for k in ("id", "bucket_m3", "swing_cycle_s",
                                                                         "fill_factor", "spot_time_s", "model")
                                                          if k in e} for e in exs]}
        st.session_state.k2_vm_ver = ver + 1
        st.rerun()
    if c[1].button("Machines as in the mine", key=f"k2_vm_reset_{act}", width="stretch", disabled=not w.get("machines")):
        w.pop("machines", None)
        st.session_state.k2_vm_ver = ver + 1
        st.rerun()


def _diff(a: dict, b: dict, lab: list) -> list[str]:
    """What a variant changes against the base one, in a few words per machine."""
    out = []
    for ar in sorted(set(a["start"]) | set(b["start"])):
        if float(a["start"].get(ar, 0)) != float(b["start"].get(ar, 0)):
            out.append(f"{ar}: {_num(b['start'].get(ar, 0))} t at start instead of {_num(a['start'].get(ar, 0))} t")
    for part in ("dr", "ex", "tr"):
        for k in sorted(set(a.get(part) or {}) | set(b.get(part) or {})):
            ca, cb = (a.get(part) or {}).get(k) or [], (b.get(part) or {}).get(k) or []
            diff = [i for i in range(min(len(ca), len(cb))) if ca[i] != cb[i]]
            if diff:
                out.append(f"{k}: different on {len(diff)} shifts (from {lab[diff[0]]})")
    if a.get("fire") != b.get("fire"):
        out.append("other blast shifts")
    if a.get("roads") != b.get("roads"):
        out.append("other roads: " + ", ".join(sorted((b.get("roads") or {}))) if b.get("roads") else "roads as on the map")
    if a.get("machines") != b.get("machines"):
        mb = (b.get("machines") or {}).get("trucks")
        out.append("other machines" + (f" ({len(mb)} trucks)" if mb is not None else ""))
    return out or ["no changes"]


def _month_today(raw: dict, sc, work: dict, p: dict, rigs: list, cap: float) -> dict:
    """The month of a variant worked as today (the same luck as every other variant – only the plan differs)."""
    from kruszyk.config import cycle_costs, scenario_from_dict
    from kruszyk.period import run_period
    from kruszyk.rl import today

    plan = period_plan(p)
    n = len(plan.shifts())
    work = workplan.fit(work, n)
    ev = workplan.evaluate(work, sc.areas, rigs, n, cap, int(p["fire"]), None, travel_h(raw, rigs), sc.shift_hours)
    sim = cycle_costs(scenario_from_dict(workplan.scenario_raw(raw, work, ev["grid"], cap)))
    plan.frozen = workplan.frozen(work, ev, n)
    df = run_period(today(sim), plan, seed=0)
    t, cost = float(df["wywiezione_t"].sum()), float(df["koszt_zl"].sum())
    crush_cap = sum(float(c.capacity_tph) for c in sim.crushers) * sim.shift_hours * n
    return {"plan_t": sum(ev["planned"]), "t": t, "cost": cost, "zl_t": cost / t if t else 0.0,
            "fuel": float(df["paliwo_l"].sum()), "queue": float(df["kolejki_h"].sum()), "issues": len(ev["issues"]),
            "mode": work.get("mode", "limit"), "shifts": n, "crusher_use": t / crush_cap if crush_cap else 0.0,
            "ex_wait_h": float(df["koparki_czekaly_h"].sum()), "ex_h": len(sim.excavators) * sim.shift_hours * n,
            "no_material": int((df["skonczyl_sie_urobek"] != "").sum())}


def _limits(r: dict) -> str:
    """What stops a variant loading more (with the plan's tonnes as a minimum)."""
    if r["no_material"] >= 0.15 * r["shifts"]:
        return (f"blasted rock – the excavators ran out of blasted rock on {r['no_material']} shifts; drilling does not "
                "keep up with the higher output")
    if r["crusher_use"] >= 0.85:
        return f"crusher – {100 * r['crusher_use']:.0f} % utilised"
    if r["ex_h"] and r["ex_wait_h"] >= 0.25 * r["ex_h"]:
        return f"trucks – the excavators waited {r['ex_wait_h']:,.0f} h for trucks".replace(",", " ")
    return "excavators – working at full capacity"


def _variants_card(raw, p, rigs, cap, lab) -> None:
    from kruszyk.config import scenario_from_dict

    vs = st.session_state.variants
    names = list(vs)
    first = names[0]
    with card("variants"):
        c = st.columns([0.9, 3.2, 1.0, 1.0, 1.8], vertical_alignment="center")
        c[0].markdown('<div class="k2-h4" style="margin:0">Variants</div>', unsafe_allow_html=True)
        with c[1]:
            st.radio("Variant", names, horizontal=True, key="k2seg_variant", label_visibility="collapsed",
                     format_func=lambda x: _variant_name(x, first))
        c[2].button("⧉ Copy", key="k2_var_copy", on_click=_copy_variant, width="stretch",
                    help="A new variant – an exact copy of the one you see; change in it whatever you want.")
        c[3].button("Delete", key="k2_var_drop", on_click=_drop_variant, width="stretch", disabled=len(names) < 2)
        go_ = c[4].button("Compare variants", type="primary", key="k2_var_cmp", width="stretch",
                          disabled=len(names) < 2,
                          help="Every variant run as Baseline – the same month, weather and breakdowns; only the plan differs.")
        act = st.session_state.get("k2seg_variant") or first
        w = vs[act]
        t1, t2 = st.columns(2)
        drill = t1.toggle(f"Plan drilling in variant {act}", value=w.get("drill", True), key=f"k2_drill_{act}",
                          help="Off: blasted rock is available all the time – no drill rigs, blasts or rock shortages. "
                               "Only the excavator – truck – crusher system counts (e.g. to check what it gives "
                               "at full capacity).")
        if drill != w.get("drill", True):
            w["drill"] = drill
            st.rerun()
        open_ = t2.toggle(f"Full capacity in variant {act} – excavators do not stop once the plan is done",
                          value=w.get("mode") == "min", key=f"k2_open_{act}",
                          help="Off: an excavator loads the tonnes in the schedule and stops. On: once the plan is done it keeps "
                               "loading as long as it has blasted rock, trucks and time – the comparison shows what the mine "
                               "can give, what limits it and what an extra tonne costs.")
        if open_ != (w.get("mode") == "min"):
            w["mode"] = "min" if open_ else "limit"
            st.rerun()
        with st.expander(f"Machines in variant {act}" + (" · changed" if w.get("machines") else " · as in the mine")):
            _variant_machines(raw, w, act)
        with st.expander(f"Roads in variant {act}" + (" · changed" if w.get("roads") else " · as on the map")):
            theme.note("Ramp capacity: single lane = trucks wait at passing bays (uphill has priority), "
                       "two lanes = they pass anywhere. “in use” off = you test a detour. Changes this variant only.")
            roads = road_editor(raw, w.get("roads") or {}, f"k2_vroads_{act}_{st.session_state.get('k2_vm_ver', 0)}")
            if roads != (w.get("roads") or {}):
                w["roads"] = roads
                st.rerun()
        theme.note("The schedule below is the selected variant. “Copy” creates a new one with the same plan – change the "
                   "drilling, excavators or trucks in it and compare. Then approve the chosen variant for the comparison with RL.")
        cache = st.session_state.setdefault("k2_cmp", {})
        sig = {k: hashlib.md5(repr((state.raw_yaml(), w, p)).encode()).hexdigest() for k, w in vs.items()}
        if go_:
            bar = st.progress(0.0, "Running variants…")
            for j, (k, w) in enumerate(vs.items()):
                if cache.get(k, {}).get("sig") != sig[k]:
                    bar.progress(j / len(vs), f"Variant {k}: Baseline month…")
                    vr = variant_raw(raw, w)
                    cache[k] = {"sig": sig[k], **_month_today(vr, scenario_from_dict(copy.deepcopy(vr)), w, p, rigs, cap)}
            bar.empty()
        done = [k for k in names if cache.get(k, {}).get("sig") == sig[k]]
        if len(done) >= 2:
            base = cache[first] if first in done else cache[done[0]]
            rows = []
            for k in done:
                r = cache[k]
                label = _variant_name(k, first) + (" · full capacity" if r.get("mode") == "min" else "")
                dz, dc = r["zl_t"] - base["zl_t"], r["cost"] - base["cost"]
                rows.append([label, _num(r["plan_t"]), _num(r["t"]), f"{r['zl_t']:.2f}",
                             "–" if k == first else f"{dz:+.2f}", _num(r["cost"]),
                             "–" if k == first else f"{dc:+,.0f}".replace(",", " "), _num(r["fuel"]), _num(r["queue"]),
                             str(r["issues"])])
            st.markdown(theme.table([("Variant", ""), ("Plan [t]", "n"), ("Hauled [t]", "n"), ("PLN/t", "n"),
                                     ("PLN/t vs A", "n"), ("Cost [PLN]", "n"), ("Cost vs A [PLN]", "n"),
                                     ("Fuel [l]", "n"), ("Queues [h]", "n"), ("To fix", "n")], rows,
                                    numeric=set(range(1, 10)), sans={0}), unsafe_allow_html=True)
            for k in done:
                r = cache[k]
                lines = []
                if k != first:
                    lines.append(" · ".join(_diff(vs[first], vs[k], lab)))
                if r.get("mode") == "min":
                    over = r["t"] - r["plan_t"]
                    lines.append(f"above plan <b>{over:+,.0f} t</b>".replace(",", " ") + f" · limited by: {_limits(r)}")
                if k != first and r["t"] - base["t"] > 100:
                    extra = (r["cost"] - base["cost"]) / (r["t"] - base["t"])
                    lines.append(f"an extra tonne costs <b>{extra:.2f} PLN</b> (average {r['zl_t']:.2f} PLN/t)")
                if lines:
                    theme.note(f"<b>{k}:</b> " + " · ".join(lines))
        elif cache and not go_:
            theme.note("You have changed something since the last comparison – click “Compare variants” to run it again.")


# ------------------------------------------------------------------ screen
def step_plan() -> None:
    from kruszyk.config import scenario_from_dict

    base = state.raw()
    vs0 = st.session_state.get("variants") or {}
    w0 = vs0.get(st.session_state.get("k2seg_variant") or "A")
    raw = variant_raw(base, w0)  # the mine as the open variant has it (its own machines, if changed)
    sc = state.scenario() if raw is base else scenario_from_dict(copy.deepcopy(raw))
    p = plan_settings()
    head = st.columns([4, 1.6], vertical_alignment="bottom")
    with head[0]:
        theme.header("Monthly schedule", "You set the work: drill rig and excavator – area and tonnes, truck – to which excavator. "
                                             "The program works out the drilling days, blasts and muck piles itself.")
    if not sc.excavators or not sc.areas:
        st.info("First draw the mine in step **① Mine**: roads, crusher, excavators and extraction areas.")
        return
    cap = sc.blasting.max_blast_t()

    # ---------------------------------------------------------------- period
    with card("period"):
        c = st.columns([0.7, 1.3, 0.7, 0.9, 1.0, 1.1, 1.1, 1.4], vertical_alignment="bottom")
        c[0].markdown('<div class="k2-h4" style="padding-bottom:10px">Period</div>', unsafe_allow_html=True)
        p["start"] = c[1].date_input("Start", dt.date.fromisoformat(p["start"]), format="DD.MM.YYYY", key="k2_start").isoformat()
        p["days"] = c[2].number_input("Days", 1, 92, int(p["days"]), key="k2_days")
        p["shifts"] = c[3].selectbox("Shifts per day", [1, 2, 3], index=int(p["shifts"]) - 1, key="k2_shifts")
        p["sat"] = c[4].checkbox("Working Saturdays", bool(p["sat"]), key="k2_sat")
        p["hol"] = c[5].text_input("Days off", p["hol"], key="k2_hol", placeholder="11.11, 24.12")
        p["real_t"] = c[6].number_input("Real per shift [t]", 0.0, 50_000.0, float(p["real_t"]), 100.0, format="%.0f",
                                        key="k2_real", help="How many tonnes you really produce per shift – the program checks whether "
                                                            "the model gives the same.")
        real_slot = c[7].empty()
        g = st.columns([0.7, 1.3, 5.6], vertical_alignment="bottom")
        g[0].markdown('<div class="k2-h4" style="padding-bottom:10px">Target</div>', unsafe_allow_html=True)
        p["target_t"] = g[1].number_input("Monthly target [t]", 0.0, 10_000_000.0, float(p["target_t"]), 5000.0,
                                          format="%.0f", key="k2_target",
                                          help="How many tonnes of crusher feed should go through in the period (0 = no target).")
        target_slot = g[2].empty()
    msg = st.session_state.pop("k2_fit_msg", None)
    if msg:
        st.success("✅ " + msg)
    plan = period_plan(p)
    shifts = plan.shifts()
    n = len(shifts)
    lab = labels(plan)
    if n == 0:
        st.warning("There are no working shifts in this period.")
        return
    rigs = [r for r in (raw.get("drills") or []) if r.get("in_work", True)]
    work, rates = _work(sc, raw, n, cap, rigs, int(p["fire"]))
    model_t = sum(rates.values())
    with real_slot.container():
        if p["real_t"] and model_t:
            diff = 100 * (model_t / p["real_t"] - 1)
            if abs(diff) <= 5:
                st.markdown(theme.chip("model matches"), unsafe_allow_html=True)
            else:
                st.markdown(theme.chip(f"model {model_t:,.0f} t ({diff:+.0f} %)".replace(",", " "), ok=False),
                            unsafe_allow_html=True)
                if st.button("Fit model", key="k2_fit"):
                    _fit_model(base, sc, p["real_t"])
        elif model_t:
            theme.note(f"model: approx. {model_t:,.0f} t/shift".replace(",", " "))

    # ---------------------------------------------------------------- areas, drilling
    c1, c2 = st.columns([3, 1.25])
    with c1, card("areas"):
        theme.h4("Areas", "blasted rock lying at start, blast size, reserves")
        _areas_editor(base, sc, rigs, int(p["fire"]), work)
    with c2, card("drill"):
        theme.h4("Drilling and blasting")
        theme.note((" · ".join(f"{r['id']}: <b>{float(r.get('m_per_shift') or 250):.0f} m/shift</b>, "
                              f"{float(r.get('speed_kmh') or 2.0):.1f} km/h" for r in rigs)
                    + " (from Machines)") if rigs else "No drill rigs in work – add them in Machines.")
        names = list(FIRE)
        cur = next((k for k, x in FIRE.items() if x == int(p["fire"])), names[0])
        p["fire"] = FIRE[st.selectbox("Blasted rock available", names, index=names.index(cur), key="k2_fire")]
        _blast_hours(base)
        theme.note("On a blast day trucks and excavators leave the pit for the evacuation and the blast; the "
                   "excavator's move to the new muck pile is included.")

    _variants_card(base, p, rigs, cap, lab)

    # ---------------------------------------------------------------- the grid
    ev = workplan.evaluate(work, sc.areas, rigs, n, cap, int(p["fire"]), rates, travel_h(raw, rigs), sc.shift_hours)
    total = sum(ev["planned"])
    target = float(p["target_t"] or 0)
    if target:
        pct = 100 * total / target
        target_slot.markdown(theme.chip(f"plan: {total:,.0f} t · {pct:.0f} % of target".replace(",", " "), ok=pct >= 100),
                             unsafe_allow_html=True)
    else:
        target_slot.markdown(f'<div class="k2-note" style="padding-bottom:10px">plan: <b>{total:,.0f} t</b> '
                             "(sum of excavator tonnes)</div>".replace(",", " "), unsafe_allow_html=True)

    main, side = st.columns([3.2, 1])
    with main, card("grid"):
        h = st.columns([2.2, 1.4, 1.6], vertical_alignment="center")
        h[0].markdown('<div class="k2-h4">Schedule</div>', unsafe_allow_html=True)
        with h[1]:
            day_view = st.radio("Resolution", ["Day", "Shift"], horizontal=True, key="k2seg_view",
                                label_visibility="collapsed") == "Day"
        if h[2].button("↺ Rebuild the proposal", key="k2_repropose", width="stretch"):
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
        with st.expander("Edit by range (without the mouse)"):
            _editor(work, sc, lab, n)
    with side, card("issues"):
        short = target - total if target else 0.0
        issues = ev["issues"]
        theme.h4(f"To fix · {len(issues) + (1 if short > 0 else 0)}")
        if short > 0:
            st.markdown(f'<div class="k2-issue"><b>The plan misses the target – {short:,.0f} t short</b><br>'
                        "Add tonnes to the excavators, a shift per day or Saturdays.</div>".replace(",", " "), unsafe_allow_html=True)
        if not issues and short <= 0:
            theme.note("Everything adds up: the rock is blasted in time, every excavator with tonnes has trucks.")
        for j, it in enumerate(issues):
            _issue(it, work, lab, n, j, ev)

    sig = hashlib.md5(repr((state.raw_yaml(), work, p)).encode()).hexdigest()
    appr = st.session_state.get("approved")
    with head[1]:
        if appr and appr["key"] == sig:
            st.markdown(theme.chip("Schedule approved"), unsafe_allow_html=True)
        if st.button("Approve and compare with RL →", type="primary", key="k2_approve", width="stretch"):
            try:
                sim_raw = workplan.scenario_raw(raw, work, ev["grid"], cap)
            except ValueError as e:
                st.error(f"This plan cannot be run: {e}")
                return
            st.session_state.approved = {"key": sig, "frozen": workplan.frozen(work, ev, n), "plan": dict(p),
                                         "work": copy.deepcopy(work), "raw": yaml_dump(sim_raw), "planned_t": total,
                                         "variant": st.session_state.get("k2seg_variant") or "A"}
            st.session_state.pop("month", None)
            st.session_state.nav_next = "③ Simulation"
            st.rerun()


def _fit_model(raw: dict, sc, real_t: float) -> None:
    from kruszyk.calibrate import calibrate

    with st.spinner("Fitting the model to real production…"):
        fitted, _, after = calibrate(sc, {"tonnes_t": real_t, "trucks": sc.truck_count})
    new = copy.deepcopy(raw)
    new["calibration"] = dataclasses.asdict(fitted.calibration)
    state.set_raw(new)
    st.session_state.k2_fit_msg = (f"Model fitted: {after['tonnes_t']:,.0f} t per shift "
                                   f"(delays {fitted.calibration.delay_min_per_trip:.1f} min per trip).").replace(",", " ")
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
    fire = c[0].time_input("Blast at", tm("fire_from", "11:00"), key=f"k2_bl_ff_{v}", step=dt.timedelta(minutes=5))
    dur = int(c[1].number_input("Blast lasts [min]", 5, 240, max(5, f1 - f0) if f1 > f0 else 20, 5,
                                key=f"k2_bl_dur_{v}"))
    crew_a = c[0].time_input("Crew arrives", tm("crew_arrive", "09:00"), key=f"k2_bl_ca_{v}",
                             step=dt.timedelta(minutes=15))
    crew_l = c[1].time_input("Crew leaves", tm("crew_leave", "11:40"), key=f"k2_bl_cl_{v}",
                             step=dt.timedelta(minutes=15))
    evac = float(c[0].number_input("Evacuation before blast [min]", 0.0, 120.0, float(bl.get("evacuation_min") or 10.0),
                                   5.0, format="%.0f", key=f"k2_bl_ev_{v}"))
    end = mins(fire) + dur
    new = {"fire_from": fire.strftime("%H:%M"), "fire_to": f"{end // 60 % 24:02d}:{end % 60:02d}",
           "crew_arrive": crew_a.strftime("%H:%M"), "crew_leave": crew_l.strftime("%H:%M"), "evacuation_min": evac}
    c[1].markdown(f'<div class="k2-note" style="padding-top:30px">blast <b>{new["fire_from"]}–{new["fire_to"]}</b>, '
                  f'pit closed from {(mins(fire) - int(evac)) // 60:02d}:{(mins(fire) - int(evac)) % 60:02d}</div>',
                  unsafe_allow_html=True)
    if mins(crew_a) > mins(fire) - evac:
        st.warning("The crew arrives after the evacuation starts – set an earlier arrival.")
    if mins(crew_l) < end:
        st.warning("The crew leaves before the blast ends – set a later departure.")
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
                        column_config={"obszar": "area", "poziom": st.column_config.NumberColumn("level [m]", format="%.0f"),
                                       "start_t": st.column_config.NumberColumn("lying at start [t]", min_value=0.0,
                                                                                step=1000.0, format="%.0f"),
                                       "tpm": st.column_config.NumberColumn("t/m", min_value=5.0),
                                       "zasoby_t": st.column_config.NumberColumn("reserves [t]", min_value=0.0, step=5000.0,
                                                                                 format="%.0f"),
                                       "mb10": "m per 10 kt",
                                       "wiert": st.column_config.TextColumn(
                                           "drill rig per shift", help="How many tonnes a drill rig drills in this area in "
                                                                       "a whole shift (m/shift × t/m).")})
    theme.note("At start muck pile <b>/1</b> lies there (e.g. O1/1). Every drilling order is one blast of that size – once "
               "drilled; every blast is the next muck pile: O1/2, O1/3 …"
               + (f" · explosives limit: largest blast {cap:,.0f} t".replace(",", " ") if cap < float("inf") else ""))
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
    return f"{_kt(x)} kt"


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
                out.append(cell(STAND, "#F3F4F6", "#5B6573", f"{key}: standby" + (f" – {why}" if why else ""),
                                red=bool(why), v=vals))
                continue
            bg, fg = acol.get(areas[-1], ("#FFFFFF", "#1B2733"))
            out.append(cell(f"{'/'.join(areas)}: {_tt(tot)}", bg, fg, f"{key}: {'/'.join(areas)} – {_num(tot)} t "
                            f"({_num(per)} t per shift)" + (f" – {why}" if why else ""), red=bool(why), v=vals))
        return out

    trucks_on = [{x: sum(1 for cells in work["tr"].values() if cells[i] == x) for x in ex_ids} for i in range(n)]

    def ex_why(x):
        def f(i):
            a, t = workplan.cell(work, x, i)
            if not t:
                return ""
            if t - ev["quota"][x][i] > 1:
                return f"not enough blasted rock in {a}: it gets {_num(ev['quota'][x][i])} t"
            if rates.get(x) and t > 1.1 * rates[x]:
                return f"more than it loads (approx. {_num(rates[x])} t per shift)"
            if not trucks_on[i][x]:
                return "no truck drives to it"
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
                                       f"{rid}: order " + ", ".join(f"{a} {_num(t)} t" for a, t in orders), bold=True,
                                       v=vals))
            elif any(rr[i]["stop"] for i in g):
                orders_row.append(cell(STAND, "#F3F4F6", "#5B6573", f"{rid}: standby", v=vals))
            else:
                orders_row.append(cell("", "#FFFFFF", "#8A93A0", f"{rid}: no new order", v=vals))
            drill: dict[str, float] = {}
            for i in g:
                for a, d in rr[i]["drill"]:
                    drill[a] = drill.get(a, 0.0) + d
            drive = sum(rr[i]["drive_h"] for i in g)
            wait = rr[g[-1]]["waiting"]
            title = (f"{rid} drills: " + (", ".join(f"{a} {_num(d)} t" for a, d in drill.items()) if drill else "nothing")
                     + (f" · drive {drive * 60:.0f} min" if drive else "")
                     + (f" · {_num(wait)} t left for the next days" if wait > 1 else ""))
            if drill:
                a = next(iter(drill))
                bg, fg = acol.get(a, ("#FFFFFF", "#1B2733"))
                text = "\n".join(f"{a}: {_tt(d)}" for a, d in drill.items()) + (f"\nleft {_tt(wait)}" if wait > 1 else "")
                drill_row.append(cell(text, "#FFFFFF", fg, title, bd=bg))
            else:
                drill_row.append(cell("–", "#FFFFFF", "#8A93A0", title))
        return orders_row, drill_row

    rows = []
    drilling = work.get("drill", True)
    if not drilling:
        rows.append({"kind": "group", "name": "DRILLING NOT PLANNED – blasted rock available all the time"})
    else:
        rows.append({"kind": "group", "name": "DRILL RIGS – you set the orders, the program works out the drilling"})
    for r in rigs if drilling else []:
        orders_row, drill_row = rig_rows(r["id"])
        rows.append({"kind": "dr", "name": f"{r['id']} · orders", "id": r["id"], "cells": orders_row})
        rows.append({"kind": "drx", "name": "↳ drills", "cells": drill_row})
    if drilling:
        rows.append({"kind": "group", "name": "BLASTS – order = blast, once drilled · click to move"})
        bl = sc.blasting
        fire_at = f"{bl.fire_from}–{bl.fire_to}"
        for a_id in area_ids:
            out = []
            for g in groups:
                fired = [(p, b) for p, b in blasts.items() if b["area"] == a_id and any(x in g for x in b["fire"] or [b["avail"] - 1])]
                if fired:
                    p, b = fired[0]
                    moved = f" (moved by {b['delay']} shifts)" if b["delay"] else ""
                    txt = "\n".join(f"blast {_tt(c['t'])}" for _, c in fired) + f"\n{fire_at}"
                    out.append(cell(txt, "#1B2733", "#FFFFFF",
                                    "\n".join(f"blast {q} (muck pile {q}): {_num(c['t'])} t · {c['m']} m" for q, c in fired)
                                    + f" · {fire_at}{moved} – blasted rock from {lab[min(b['avail'], n - 1)]}", bold=True, p=p))
                else:
                    out.append(cell("", "#FFFFFF", "#FFFFFF"))
            rows.append({"kind": "fire", "name": a_id, "cells": out})
        rows.append({"kind": "group", "name": "BLASTED ROCK IN THE AREAS"})
        for a_id in area_ids:
            out = []
            for g in groups:
                lft = ev["area_left"][a_id][g[0]]
                miss = sum(ev["short"][a_id][i] for i in g)
                if miss > 1:
                    out.append(cell(f"short {_tt(miss)}", "#FDE2E2", "#8C0E20", f"{a_id}: the excavators will be {_num(miss)} t short",
                                    red=True))
                else:
                    out.append(cell(_tt(lft) if lft > 1 else "–", "#E3F1E6" if lft > 1 else "#FFFFFF",
                                    "#1E5A2C" if lft > 1 else "#8A93A0", f"{a_id}: approx. {_num(lft)} t lying"))
            rows.append({"kind": "pile", "name": a_id, "cells": out})
    rows.append({"kind": "group", "name": "EXCAVATORS – area and tonnes · set by you"})
    for x in ex_ids:
        rows.append({"kind": "ex", "name": x, "cells": work_row(x, "ex", ex_why(x))})
    rows.append({"kind": "group", "name": "TRUCKS – to which excavator · set by you"})
    for tid, cells in work["tr"].items():
        out = []
        for g in groups:
            xs = list(dict.fromkeys(cells[i] for i in g))
            idle = [cells[i] for i in g if cells[i] and workplan.cell(work, cells[i], i)[1] <= 0]
            if xs == [""]:
                out.append(cell("—", "#F3F4F6", "#8A93A0", f"{tid}: idle"))
                continue
            bg, fg = ecol.get(next(x for x in xs if x), ("#FFFFFF", "#1B2733"))
            text = "/".join(x or "—" for x in xs)
            out.append(cell(text, bg, fg, f"{tid}: drives to {text}" + (f" – {idle[0]} is standing" if idle else ""),
                            red=bool(idle)))
        rows.append({"kind": "tr", "name": tid, "cells": out})
    for x in ex_ids:
        out = []
        for g in groups:
            k = trucks_on[g[0]][x]
            out.append(cell(str(k) if k else "–", "#FFFFFF", "#3A4452" if k else "#8A93A0", f"to {x}: {k} trucks"))
        rows.append({"kind": "sum", "name": f"trucks to {x}", "cells": out})
    planned = ev["planned"]
    rows.append({"kind": "group", "name": "CUMULATIVE PLAN" + (f" · target {_tt(target)}" if target else "")})
    out, acc = [], 0.0
    for g in groups:
        acc += sum(planned[i] for i in g)
        due = target * (g[-1] + 1) / n if target else 0.0
        ok = not target or acc >= due * 0.98
        out.append(cell(f"{acc / 1000:.1f}".replace(".", ","), "#E3F1E6" if ok else "#FFF0B8", "#1E5A2C" if ok else "#6E5600",
                        f"to date: {_num(acc)} t" + (f" · on target pace it should be {_num(due)} t" if target else ""),
                        bold=True))
    rows.append({"kind": "sum", "name": "hauled [kt]", "cells": out})
    areas = []
    for a in sc.areas:
        info = f"lying at start {_tt(ev['area_left'][a['id']][0])}"
        areas.append({"id": a["id"], "info": info, "bg": acol[a["id"]][0], "fg": acol[a["id"]][1]})
    exs = [{"id": e.id, "allowed": [a for a in area_ids if e.may_work(a)] or area_ids, "rate": round(rates.get(e.id, 0.0)),
            "bg": ecol[e.id][0], "fg": ecol[e.id][1]} for e in sc.excavators if e.id in ecol]
    rigs_ = [{"id": r["id"], "allowed": area_ids, "rates": {a: round(workplan.rig_rate(r, by_area[a])) for a in area_ids}}
             for r in rigs]
    blasts_ = {p: {"t": b["t"], "m": b["m"], "delay": b["delay"], "avail": lab[min(b["avail"], n - 1)],
                   "done": lab[min(b["done"], n - 1)]} for p, b in blasts.items()}
    return {"day": day_view, "spd": spd, "heads": heads, "groups": groups, "labels": lab, "off": [False] * len(heads),
            "rows": rows, "areas": areas, "exs": exs, "rigs": rigs_, "blasts": blasts_,
            "v_stand": STAND, "v_off": OFF, "v_no_order": NO_ORDER}  # the values the grid sends back


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
            if area in STAND_ALL:
                w[part][key][i] = {"a": "", "t": 0.0, "stop": True}
            elif area in NO_ORDER_ALL or not first:
                w[part][key][i] = {"a": "", "t": 0.0}
            else:
                w[part][key][i] = {"a": area, "t": t_shift}
        else:
            w[part][key][i] = {"a": area if area not in STAND_ALL else "", "t": t_shift if area and area not in STAND_ALL else 0.0}


def _set_trucks(ids: list[str], a: int, b: int, ex: str) -> None:
    w = st.session_state.work
    for tid in ids:
        for i in range(a, b + 1):
            if tid in w["tr"] and i < len(w["tr"][tid]):
                w["tr"][tid][i] = "" if ex in OFF_ALL else ex


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
    for title, part, pre in (("Drill rig", "dr", "k2_d"), ("Excavator", "ex", "k2_e")):
        if not work.get(part):
            continue
        theme.h4(title)
        c = st.columns([1, 1.4, 1.4, 1.2, 1.2, 1])
        key = c[0].selectbox(title, list(work[part]), key=f"{pre}_ex")
        a = c[1].selectbox("From shift", range(n), format_func=lambda i: lab[i], key=f"{pre}_from")
        b = c[2].selectbox("To shift", range(n), index=n - 1, format_func=lambda i: lab[i], key=f"{pre}_to")
        area = c[3].selectbox("Area", areas + [STAND] + ([NO_ORDER] if part == "dr" else []), key=f"{pre}_area")
        t = c[4].number_input("Tonnes to drill" if part == "dr" else "Tonnes per shift", 0.0, 200_000.0, 1000.0, 100.0,
                              format="%.0f", key=f"{pre}_t")
        with c[5]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("Set", type="primary", key=f"{pre}_set", width="stretch"):
                _set_ex(key, min(a, b), max(a, b), area, t, part, [0] * n if part == "dr" else None)
                st.rerun()
    theme.h4("Trucks")
    c = st.columns([2.2, 1.4, 1.4, 1.2, 1])
    ids = c[0].multiselect("Trucks", list(work["tr"]), key="k2_t_ids")
    a = c[1].selectbox("From shift", range(n), format_func=lambda i: lab[i], key="k2_t_from")
    b = c[2].selectbox("To shift", range(n), index=n - 1, format_func=lambda i: lab[i], key="k2_t_to")
    x = c[3].selectbox("Drives to", list(work["ex"]) + [OFF], key="k2_t_ex")
    with c[4]:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("Set", type="primary", key="k2_t_set", width="stretch", disabled=not ids):
            _set_trucks(ids, min(a, b), max(a, b), x)
            st.rerun()


def _issue(it, work, lab, n, j, ev) -> None:
    k, i = it["kind"], it["shift"]
    k_ = it.get("shifts", 1)
    many = f" ({k_} shifts)" if k_ > 1 else ""
    if k == workplan.SHORT:
        a = it["area"]
        drill = [r for r, cells in work.get("dr", {}).items() if any(workplan.cell(work, r, s, "dr")[0] == a for s in range(n))]
        st.markdown(f'<div class="k2-issue"><b>{it["ex"]} from {lab[i]}: {a} will be {_num(it["t"])} t of blasted rock short{many}</b>'
                    f"<br>The excavators have more tonnes than is blasted. Add drilling in {a} earlier"
                    + (f" ({', '.join(drill)} drills there later)" if drill else "") + " or reduce the excavator tonnes.</div>",
                    unsafe_allow_html=True)
    elif k == workplan.EXHAUSTED:
        st.markdown(f'<div class="k2-issue"><b>Area {it["area"]}: from {lab[i]} drilling exceeds the reserves</b><br>'
                    "Increase the reserves in the areas table or move the drill rig.</div>", unsafe_allow_html=True)
    elif k == workplan.TOO_BIG:
        st.markdown(f'<div class="k2-issue"><b>{it["rig"]} · {lab[i]}: blast {_num(it["t"])} t in {it["area"]}</b><br>'
                    f"The explosives limit allows approx. {_num(it['cap'])} t in one blast – split "
                    "the order in two.</div>", unsafe_allow_html=True)
    elif k == workplan.UNFINISHED:
        st.markdown(f'<div class="k2-issue w"><b>{it["rig"]}: {_num(it["t"])} t in {it["area"]} will not be drilled '
                    "by the end of the period</b><br>There are more orders than the drill rig can manage – move some to another drill rig "
                    "or reduce the orders.</div>", unsafe_allow_html=True)
    elif k == workplan.OVER:
        ex = it["ex"]
        r = max(100.0, round(it["rate"], -2))
        st.markdown(f'<div class="k2-issue w"><b>{ex}: {_num(it["t"])} t per shift{many}</b><br>It really loads approx. '
                    f'{_num(it["rate"])} t – the rest of the plan will not be hauled.</div>', unsafe_allow_html=True)
        if st.button(f"Set {ex} to {_num(r)} t", key=f"k2_fix_{j}", width="stretch"):
            for s in range(n):
                c = work["ex"][ex][s]
                if c.get("a") and float(c.get("t") or 0) > 1.1 * it["rate"]:
                    c["t"] = r
            st.rerun()
    elif k == workplan.NO_TRUCKS:
        st.markdown(f'<div class="k2-issue"><b>{it["ex"]} from {lab[i]}: no trucks{many}</b><br>The excavator has tonnes, but no truck '
                    "drives to it.</div>", unsafe_allow_html=True)
    elif k == workplan.IDLE_TRUCK:
        ex = it["ex"]
        st.markdown(f'<div class="k2-issue w"><b>{it["truck"]} from {lab[i]} drives to {ex}, but {ex} is standing{many}</b><br>'
                    "The truck has nowhere to load from.</div>", unsafe_allow_html=True)
        if st.button(f"{it['truck']}: idle while {ex} is standing", key=f"k2_fix_{j}", width="stretch"):
            cells = work["tr"][it["truck"]]
            for s in range(n):
                if cells[s] == ex and workplan.cell(work, ex, s)[1] <= 0:
                    cells[s] = ""
            st.rerun()
