"""② Harmonogram: the planner sets where every excavator loads on every shift; the program adds the drilling and
blasting of every new pile, shows when it is there and what does not add up. The approved schedule is frozen: the
month with today's driving and the month with the driving signals both follow it."""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import html

import pandas as pd
import streamlit as st

from kruszyk.period import PeriodPlan
from kruszyk.schedule import EMPTY, NOT_YET, area_blast_t, area_of, check, depletion, frozen, plan_blasts, propose
from ui import card

from . import state

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
    st.markdown("## Harmonogram miesiąca")
    st.caption("Układasz go Ty: gdzie ładuje każda koparka. Program dolicza wiercenie i strzał każdego nowego "
               "odstrzału, pilnuje dostępności usypów i wiertnic.")
    raw = state.raw()
    sc = state.scenario()
    if not sc.excavators:
        st.info("Najpierw narysuj kopalnię (koparki, usypy, drogi) w kroku **① Kopalnia**.")
        return
    p = plan_settings()
    v = st.session_state.get("ver", 0)
    rates, _ = rates_per_shift()

    # ---------------------------------------------------------------- period
    with card("k2_period"):
        c = st.columns([1.3, 0.8, 0.9, 0.9, 1.1, 1.1, 1.4])
        p["start"] = c[0].date_input("Start", dt.date.fromisoformat(p["start"]), format="DD.MM.YYYY", key="k2_start").isoformat()
        p["days"] = c[1].number_input("Dni", 1, 92, int(p["days"]), key="k2_days")
        p["shifts"] = c[2].selectbox("Zmian na dobę", [1, 2, 3], index=int(p["shifts"]) - 1, key="k2_shifts")
        p["sat"] = c[3].checkbox("Soboty robocze", bool(p["sat"]), key="k2_sat")
        p["hol"] = c[4].text_input("Dni wolne (dd.mm)", p["hol"], key="k2_hol", placeholder="11.11, 24.12")
        p["real_t"] = c[5].number_input("Realnie na zmianę [t]", 0.0, 50_000.0, float(p["real_t"]), 100.0, key="k2_real",
                                        help="Ile ton robicie realnie na zmianę – program sprawdzi, czy model daje tyle samo.")
        model_t = sum(rates.values())
        with c[6]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if p["real_t"] and model_t:
                diff = 100 * (model_t / p["real_t"] - 1)
                if abs(diff) <= 5:
                    st.success(f"model zgodny ({model_t:,.0f} t)".replace(",", " "))
                else:
                    st.warning(f"model {model_t:,.0f} t ({diff:+.0f} %)".replace(",", " "))
                    if st.button("Dopasuj model", key="k2_fit"):
                        _fit_model(raw, sc, p["real_t"])
            elif model_t:
                st.caption(f"model: ok. {model_t:,.0f} t na zmianę".replace(",", " "))
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

    # ---------------------------------------------------------------- start state and drilling
    start_t = {b.id: float(b.tonnage_t) for b in sc.blasts}
    rigs = [r for r in (raw.get("drills") or []) if r.get("in_work", True)]
    c1, c2 = st.columns([2, 3])
    with c1, card("k2_startstate"):
        st.markdown(f"#### Stan na start · {lab[0]}")
        on = {}
        for e in sc.excavators:
            on.setdefault(e.blast, []).append(e.id)
        st.dataframe(pd.DataFrame([{"usyp": b, "leży [t]": round(t), "koparki": ", ".join(on.get(b, []))}
                                   for b, t in start_t.items()]), hide_index=True, width="stretch")
        st.markdown("**Wiercenie i strzelanie**")
        st.caption(" · ".join(f"{r['id']}: {float(r.get('m_per_shift') or 250):.0f} mb/zm." for r in rigs) +
                   " (z Maszyn)" if rigs else "Brak wiertnic w pracy – odstrzały liczone bez wiercenia (Maszyny).")
        names = list(FIRE)
        cur = next((k for k, x in FIRE.items() if x == int(p["fire"])), names[0])
        p["fire"] = FIRE[st.selectbox("Strzał (z ewakuacją i przewietrzeniem)", names, index=names.index(cur),
                                      key="k2_fire")]
    with c2, card("k2_areas"):
        st.markdown("#### Obszary eksploatacji")
        st.caption("Każdy odstrzał to nowy usyp: A/1, A/2 … Szare kolumny liczy program.")
        _areas_editor(raw, sc, rigs, int(p["fire"]), v)

    # ---------------------------------------------------------------- the grid
    key = repr((n, tuple(e.id for e in sc.excavators), p["start"], int(p["shifts"])))
    grid = st.session_state.get("grid")
    if grid is None or st.session_state.get("grid_key") != key or set(grid) != {e.id for e in sc.excavators}:
        grid = propose(sc.excavators, start_t, sc.areas, rates, n, sc.blasting.max_blast_t())
        st.session_state.grid, st.session_state.grid_key = grid, key
    grid = {ex: (cells + [cells[-1]] * n)[:n] for ex, cells in grid.items()}
    blasts = plan_blasts(grid, sc.areas, rigs, n, sc.blasting.max_blast_t(), int(p["fire"]))
    issues = check(grid, blasts, start_t, rates, n)
    left = depletion(grid, blasts, start_t, rates, n)

    with card("k2_grid"):
        h = st.columns([3, 2, 2])
        h[0].markdown("#### Harmonogram")
        day_view = h[1].radio("Dokładność", ["Dzień", "Zmiana"], horizontal=True, key="k2_view",
                              label_visibility="collapsed") == "Dzień"
        if h[2].button("↺ Ułóż propozycję od nowa", key="k2_repropose", width="stretch"):
            st.session_state.grid = propose(sc.excavators, start_t, sc.areas, rates, n, sc.blasting.max_blast_t())
            st.rerun()
        st.markdown(_grid_html(grid, blasts, left, start_t, rigs, shifts, lab, day_view, issues), unsafe_allow_html=True)
        _editor(grid, blasts, sc, start_t, lab, n)

    # ---------------------------------------------------------------- what to correct
    with card("k2_issues"):
        if issues:
            st.markdown(f"#### Do poprawy · {len(issues)}")
            for j, it in enumerate(issues):
                _issue(it, grid, blasts, sc, start_t, lab, n, j)
        else:
            st.markdown("#### Do poprawy · 0")
            st.caption("Każda koparka ma urobek na każdej zmianie (szacunek z dzisiejszego tempa ładowania).")

    sig = hashlib.md5(repr((state.raw_yaml(), grid, p)).encode()).hexdigest()
    appr = st.session_state.get("approved")
    c = st.columns([3, 2])
    if appr and appr["key"] == sig:
        c[0].success("✅ Harmonogram zatwierdzony – w kroku ③ oba miesiące idą według niego.")
    elif appr:
        c[0].warning("Harmonogram zmienił się od zatwierdzenia – zatwierdź go ponownie.")
    if c[1].button("✅ Zatwierdź i porównaj z RL →", type="primary", key="k2_approve", width="stretch"):
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


def _cell(text: str, bg: str, fg: str, title: str = "", strong: bool = False, red: bool = False) -> str:
    style = f"background:{bg};color:{fg};"
    if red:
        style = "background:#FDE2E2;color:#8C0E20;box-shadow:inset 0 0 0 2px #B3122A;"
    if strong:
        style += "font-weight:700;"
    return (f'<td title="{html.escape(title)}" style="{style}padding:6px 4px;text-align:center;'
            f'border-left:1px solid #EEF0F3;border-bottom:1px solid #EEF0F3;white-space:nowrap">{html.escape(text)}</td>')


def _grid_html(grid, blasts, left, start_t, rigs, shifts, lab, day_view, issues) -> str:
    n = len(shifts)
    colors = _colors(start_t, blasts)
    bad = set()
    for i in range(n):
        for ex, cells in grid.items():
            p = cells[i]
            if (p in blasts and blasts[p]["avail"] > i) or (left.get(p, [0] * n)[i] <= 1):
                bad.add((ex, i))
    # columns: shifts, or days (the shifts of a day merged)
    if day_view:
        cols, cur = [], None
        for i, (d, _, _) in enumerate(shifts):
            if cur is None or cur[0] != d:
                cur = (d, [])
                cols.append(cur)
            cur[1].append(i)
        heads = [f"{WD[d.weekday()]} {d:%d.%m}" for d, _ in cols]
        groups = [ix for _, ix in cols]
    else:
        heads, groups = lab, [[i] for i in range(n)]
    th = "position:sticky;left:0;background:#F7F8FA;z-index:1;"
    rows = [f'<tr><th style="{th}padding:6px 10px;border-bottom:1px solid #D9DDE3"></th>' +
            "".join(f'<th style="padding:6px 4px;background:#F7F8FA;border-bottom:1px solid #D9DDE3;'
                    f'border-left:1px solid #EEF0F3;white-space:nowrap;font-weight:600">{h}</th>' for h in heads) + "</tr>"]

    def group_row(title):
        rows.append(f'<tr><td colspan="{len(heads) + 1}" style="background:#F7F8FA;font-size:11px;font-weight:700;'
                    f'letter-spacing:.06em;color:#4A5562;padding:6px 10px">{title}</td></tr>')

    def name_cell(name):
        return (f'<td style="position:sticky;left:0;background:#fff;z-index:1;padding:6px 10px;font-weight:700;'
                f'border-bottom:1px solid #EEF0F3;white-space:nowrap">{html.escape(name)}</td>')

    group_row("KOPARKI – planujesz Ty")
    for ex, cells in grid.items():
        tds = []
        for g in groups:
            ps = [cells[i] for i in g]
            text = ps[0] if len(set(ps)) == 1 else " ⇢".join(dict.fromkeys(ps))
            bg, fg = colors.get(ps[-1], ("#fff", "#1B2733"))
            red = any((ex, i) in bad for i in g)
            why = ""
            if red:
                p = next(cells[i] for i in g if (ex, i) in bad)
                why = (f" – {p} dostępny dopiero od {lab[blasts[p]['avail']]}" if p in blasts and blasts[p]["avail"] > g[0]
                       else f" – {p} już wybrany (szacunek)")
            tds.append(_cell(text, bg, fg, f"{ex}, {heads[groups.index(g)]}: {text}{why}", red=red))
        rows.append("<tr>" + name_cell(ex) + "".join(tds) + "</tr>")
    if rigs:
        group_row("WIERTNICE – liczy program")
        for r in rigs:
            tds = []
            for g in groups:
                jobs = [p for p, b in blasts.items() for (rid, x) in b["drill"] if rid == r["id"] and x in g]
                txt = ", ".join(dict.fromkeys(jobs)) or "–"
                tds.append(_cell(txt, "#EDEFF2" if jobs else "#fff", "#3A4452" if jobs else "#8A93A0",
                                 f"{r['id']}: {'wierci ' + txt if jobs else 'wolna'}"))
            rows.append("<tr>" + name_cell(r["id"]) + "".join(tds) + "</tr>")
    if blasts:
        group_row("ODSTRZAŁY – liczy program")
        for p, b in sorted(blasts.items(), key=lambda kv: kv[1]["avail"]):
            tds = []
            for g in groups:
                drill = any(x in g for _, x in b["drill"])
                fire = any(x in g for x in b["fire"])
                avail = any(x >= b["avail"] for x in g)
                if fire:
                    tds.append(_cell("strzał", "#1B2733", "#fff", f"{p}: strzał", strong=True))
                elif drill:
                    tds.append(_cell("wierc.", "#EDEFF2", "#3A4452", f"{p}: wiercenie"))
                elif avail:
                    lft = left[p][g[0]]
                    tds.append(_cell(f"{lft / 1000:.1f} kt" if lft > 1 else "wybrany",
                                     "#E3F1E6" if lft > 1 else "#F3F4F6", "#1E5A2C" if lft > 1 else "#5B6573",
                                     f"{p}: leży ok. {lft:,.0f} t".replace(",", " ")))
                else:
                    tds.append(_cell("", "#fff", "#fff"))
            title = f"{p} · {b['t']:,.0f} t · {b['m']} mb".replace(",", " ")
            rows.append("<tr>" + name_cell(title) + "".join(tds) + "</tr>")
    group_row("USYPY Z DZIŚ")
    for p in start_t:
        tds = []
        for g in groups:
            lft = left[p][g[0]]
            tds.append(_cell(f"{lft / 1000:.1f} kt" if lft > 1 else "wybrany", "#E3F1E6" if lft > 1 else "#F3F4F6",
                             "#1E5A2C" if lft > 1 else "#5B6573", f"{p}: leży ok. {lft:,.0f} t".replace(",", " ")))
        rows.append("<tr>" + name_cell(p) + "".join(tds) + "</tr>")
    legend = ('<div style="display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:#3A4452;margin-top:8px">'
              '<span>⬜ szare: wiercenie</span><span>⬛ czarne: strzał</span><span>🟩 zielone: leży (kt = tys. t)</span>'
              '<span style="color:#8C0E20">czerwona ramka: konflikt – najedź, żeby zobaczyć dlaczego</span></div>')
    return ('<div style="overflow-x:auto;border:1px solid #E3E6EA;border-radius:8px;background:#fff">'
            '<table style="border-collapse:collapse;font-size:12px;font-family:\'IBM Plex Mono\',ui-monospace,monospace">'
            + "".join(rows) + "</table></div>" + legend)


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
            desc[pid] = f"{pid} · odstrzał z obszaru {a_id}" + (f" · w harmonogramie" if b else "")
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
    c = st.columns([4, 2])
    if it["kind"] == NOT_YET:
        avail = it["avail"]
        prev = grid[ex][i - 1] if i > 0 else next((e.blast for e in sc.excavators if e.id == ex), p)
        c[0].error(f"**{ex} · {lab[i]}:** {p} będzie dostępny dopiero od **{lab[min(avail, n - 1)]}** "
                   "(wiercenie i strzał nie zdążą – wiertnice zajęte albo za mało czasu).")
        if prev != p and c[1].button(f"Zostaw {ex} na {prev} do tego czasu", key=f"k2_fix_{j}", width="stretch"):
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
        c[0].warning(f"**{ex} · {lab[i]}:** na {p} nie zostanie urobku (szacunek z dzisiejszego tempa) – "
                     "koparka by stała.")
        if nxt and c[1].button(f"Od {lab[i]}: {ex} na nowy odstrzał {nxt}", key=f"k2_fix_{j}", width="stretch"):
            end = i
            while end + 1 < n and grid[ex][end + 1] == p:
                end += 1
            _set(ex, i, end, nxt)
            st.rerun()
