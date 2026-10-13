"""Step "Month": the shift simulation run day after day for a week / month, with the mine state carried over."""
from __future__ import annotations

import calendar
import copy
import datetime as dt
import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kruszyk.config import scenario_from_dict
from kruszyk.period import PeriodPlan, PileBlasts, run_period, summary

DISPATCH_LABELS = {"natural": "jak dziś (kierowcy wracają do „swojej” koparki)",
                   "dynamic": "system przydziela (heurystyka)", "rl": "polityka RL (wyuczona w kroku ④)"}


def _next_month(today: dt.date) -> dt.date:
    return (today.replace(day=1) + dt.timedelta(days=32)).replace(day=1)


def _holidays(text: str, year: int) -> list[dt.date]:
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            d, m = (int(x) for x in part.split(".")[:2])
            out.append(dt.date(year, m, d))
        except ValueError:
            pass
    return out


def step_period(base_raw: dict, base_yaml: str | None, seed: int) -> None:
    st.markdown("### 📅 Tydzień / miesiąc")
    st.markdown("Ta sama symulacja zmiany, ale **dzień po dniu**: urobek w usypach ubywa, **strzały go dokładają** "
                "według harmonogramu, hałda i postęp koparek po froncie przechodzą na następną zmianę, a pogoda, "
                "awarie i kierowcy losują się na każdą zmianę osobno. Widać, **czy zrobicie plan** i **kiedy "
                "zabraknie urobku**.")
    if base_yaml is None:
        st.info("Najpierw opisz i zapisz kopalnię w kroku **① Kopalnia**.")
        return
    variant = st.session_state.get("variant_yaml")
    c1, c2 = st.columns(2)
    which = c1.radio("Kopalnia", ["stan obecny"] + (["wariant z kroku ③"] if variant else []), horizontal=True,
                     key="per_which")
    sc = scenario_from_dict(yaml_load(variant if which.startswith("wariant") else base_yaml))
    modes = ["natural", "dynamic"] + (["rl"] if sc.rl_weights else [])
    dispatch = c2.radio("Przydział wozideł", modes, format_func=DISPATCH_LABELS.get, key="per_dispatch")
    sc.dispatch = dispatch

    st.markdown("**🗓️ Kalendarz**")
    c = st.columns(5)
    start = c[0].date_input("Od dnia", _next_month(dt.date.today()), key="per_start", format="DD.MM.YYYY")
    days = c[1].number_input("Liczba dni (kalendarzowych)", 1, 92,
                             calendar.monthrange(start.year, start.month)[1] if start.day == 1 else 30, key="per_days")
    shifts = c[2].selectbox("Zmian na dobę", [1, 2, 3], index=1, key="per_shifts",
                            help=f"każda po {sc.shift_hours:g} h – pierwsza od {sc.shift_plan.start_time}")
    sat = c[3].checkbox("Soboty robocze", key="per_sat")
    target = c[4].number_input("Cel na okres [t] (0 = bez celu)", 0.0, 10_000_000.0, 0.0, 5000.0, key="per_target")
    hol = st.text_input("Dni wolne (dd.mm, po przecinku)", "", key="per_hol", placeholder="np. 11.11, 24.12, 25.12")

    st.markdown("**🧨 Harmonogram strzałów** – co ile dni roboczych i ile ton trafia do usypu. 0 = bez strzałów "
                "(tylko to, co jest w usypie teraz).")
    base_key = f"per_blasts__{hashlib.md5(','.join(b.id for b in sc.blasts).encode()).hexdigest()}"
    if base_key not in st.session_state:
        st.session_state[base_key] = pd.DataFrame([{"usyp": b.id, "material": b.material, "teraz_t": b.tonnage_t,
                                                    "co_dni": 3, "ile_t": b.tonnage_t, "pierwszy_dzien": 2}
                                                   for b in sc.blasts])
    ed = st.data_editor(st.session_state[base_key], key=base_key + "_ed", hide_index=True, width="stretch",
                        disabled=["usyp", "material", "teraz_t"],
                        column_config={"usyp": "usyp", "material": "materiał",
                                       "teraz_t": st.column_config.NumberColumn("jest teraz [t]", format="%.0f"),
                                       "co_dni": st.column_config.NumberColumn("strzał co [dni rob.]", min_value=0, step=1),
                                       "ile_t": st.column_config.NumberColumn("ile ton ze strzału", min_value=0.0, step=1000.0),
                                       "pierwszy_dzien": st.column_config.NumberColumn(
                                           "pierwszy strzał w dniu rob. nr", min_value=1, step=1)})
    st.markdown("**🧨📍 Nowe strzały w nowych miejscach** – rysujesz je na mapie w kroku ① (tryb „Planowany strzał "
                "(nowy usyp)”). W dniu strzału pojawia się nowy usyp; koparka, której skończył się urobek, "
                "**sama przejeżdża** na najbliższy odstrzelony usyp, na którym nikt nie pracuje.")
    if sc.planned_blasts:
        pkey = f"per_planned__{hashlib.md5(repr([(p['id'], p.get('blast_day')) for p in sc.planned_blasts]).encode()).hexdigest()}"
        if pkey not in st.session_state:
            st.session_state[pkey] = pd.DataFrame([{"strzal": p["id"], "material": p.get("material", ""),
                                                    "dzien": int(p.get("blast_day") or 1), "ile_t": float(p["tonnage_t"])}
                                                   for p in sc.planned_blasts])
        ped = st.data_editor(st.session_state[pkey], key=pkey + "_ed", hide_index=True, width="stretch",
                             disabled=["strzal", "material"],
                             column_config={"strzal": "strzał", "material": "materiał",
                                            "dzien": st.column_config.NumberColumn("w dniu roboczym nr", min_value=1, step=1),
                                            "ile_t": st.column_config.NumberColumn("ile ton", min_value=0.0, step=1000.0)})
        for p, (_, r) in zip(sc.planned_blasts, ped.iterrows()):
            p["blast_day"], p["tonnage_t"] = int(r["dzien"] or 1), float(r["ile_t"] or 0)
    else:
        st.caption("Na mapie nie ma planowanych strzałów – koparki pracują tylko na obecnych usypach.")
    c = st.columns(2)
    move = c[0].checkbox("Koparka bez urobku przejeżdża na inny usyp", True, key="per_move")
    setup = c[1].number_input("Przygotowanie nowego frontu [min] (oprócz jazdy)", 0.0, 240.0, 15.0, 5.0,
                              key="per_setup", disabled=not move)
    plan = PeriodPlan(start=start, days=int(days), shifts_per_day=int(shifts), saturdays=sat,
                      holidays=_holidays(hol, start.year), target_t=float(target),
                      blasts=[PileBlasts(r["usyp"], float(r["ile_t"] or 0), int(r["co_dni"] or 0),
                                         int(r["pierwszy_dzien"] or 1)) for _, r in ed.iterrows()],
                      move_excavators=move, move_setup_min=float(setup))
    n = len(plan.shifts())
    st.caption(f"{len(plan.working_days())} dni roboczych × {shifts} = **{n} zmian** · liczy się ok. {max(1, n // 3)} s")

    key = hashlib.md5(repr((dump_key(sc), plan, seed)).encode()).hexdigest()
    if st.button("▶️ Policz okres", type="primary", disabled=n == 0):
        bar = st.progress(0.0, "liczę zmiany…")
        df = run_period(sc, plan, seed=seed, progress=lambda i, k: bar.progress(i / k, f"zmiana {i} z {k}"))
        bar.empty()
        st.session_state.period = (key, df, plan)
    res = st.session_state.get("period")
    if not res:
        return
    if res[0] != key:
        st.warning("Ustawienia zmieniły się od ostatniego liczenia – kliknij **Policz okres** ponownie.")
    _results(res[1], res[2], sc)


def yaml_load(text: str) -> dict:
    import yaml

    return yaml.safe_load(text)


def dump_key(sc) -> str:
    from kruszyk.config import dump_scenario

    return dump_scenario(sc)


def _results(df: pd.DataFrame, plan: PeriodPlan, sc) -> None:
    s = summary(df, plan)
    if not s:
        return
    c = st.columns(5)
    c[0].metric("Wywiezione", f"{s['wywiezione_t']:,.0f} t".replace(",", " "),
                f"{s['realizacja_celu_pct']:.0f} % celu" if s["realizacja_celu_pct"] is not None else None,
                delta_color="normal" if (s["realizacja_celu_pct"] or 100) >= 100 else "inverse")
    c[1].metric("Koszt na tonę (okres)", f"{s['koszt_na_t']:.2f} zł/t" if s["koszt_na_t"] else "–")
    c[2].metric("Koszt łącznie", f"{s['koszt_zl'] / 1000:,.0f} tys. zł".replace(",", " "))
    c[3].metric("Średnio na zmianę", f"{s['srednio_na_zmiane_t']:,.0f} t".replace(",", " "))
    c[4].metric("Zmiany z brakiem urobku", s["zmian_bez_urobku"])

    piles = [b.id for b in sc.blasts] + [p["id"] for p in sc.planned_blasts if f"z_{p['id']}_t" in df]
    df = df.copy()
    df["etykieta"] = df["data"].map(lambda d: d.strftime("%d.%m")) + " " + df["dzien"] + " z" + df["zmiana"].astype(str)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for p in piles:
        fig.add_bar(x=df["etykieta"], y=df[f"z_{p}_t"], name=f"z {p}")
    fig.add_scatter(x=df["etykieta"], y=df["wywiezione_t"].cumsum(), name="narastająco", mode="lines",
                    line=dict(color="#111", width=3), secondary_y=True)
    if plan.target_t:
        fig.add_scatter(x=[df["etykieta"].iloc[0], df["etykieta"].iloc[-1]], y=[plan.target_t / len(df), plan.target_t],
                        name="cel narastająco", mode="lines", line=dict(color="#e63946", dash="dash"), secondary_y=True)
    for _, r in df[df["strzal"] != ""].iterrows():
        fig.add_annotation(x=r["etykieta"], y=0, text="🧨", showarrow=False, yshift=-12)
    fig.update_layout(barmode="stack", height=420, margin=dict(t=30, b=10), legend=dict(orientation="h"),
                      title="Tony na zmianę (słupki) i narastająco (linia)")
    fig.update_yaxes(title_text="t / zmianę", secondary_y=False)
    fig.update_yaxes(title_text="t narastająco", secondary_y=True)
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    for p in piles:
        fig2.add_scatter(x=df["etykieta"], y=df[f"zostalo_{p}_t"], name=p, mode="lines+markers", line_shape="hv")
    fig2.update_layout(height=300, margin=dict(t=30, b=10), title="Ile zostało w usypach po każdej zmianie [t]",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Wnioski:**  \n" + "  \n".join(_findings(df, plan, sc, s)))

    day = df.groupby(["data", "dzien"], as_index=False).agg(
        wywiezione_t=("wywiezione_t", "sum"), koszt_zl=("koszt_zl", "sum"), strzal=("strzal", lambda x: ", ".join(v for v in x if v)),
        przejazdy=("przejazdy_koparek", lambda x: ", ".join(v for v in x if v)),
        brak_urobku=("skonczyl_sie_urobek", lambda x: "; ".join(v for v in x if v)),
        kruszarka_bez_nadawy_h=("kruszarka_bez_nadawy_h", "sum"), deszcz=("deszcz", "any"))
    day["koszt_na_t"] = (day["koszt_zl"] / day["wywiezione_t"].where(day["wywiezione_t"] > 0)).round(2)
    day["data"] = day["data"].map(lambda d: d.strftime("%d.%m"))
    st.markdown("**Dzień po dniu**")
    st.dataframe(day, hide_index=True, width="stretch",
                 column_config={"wywiezione_t": st.column_config.NumberColumn("wywiezione [t]", format="%.0f"),
                                "koszt_zl": st.column_config.NumberColumn("koszt [zł]", format="%.0f"),
                                "koszt_na_t": "zł/t", "strzal": "strzał", "przejazdy": "przejazdy koparek",
                                "brak_urobku": "koparki bez urobku",
                                "kruszarka_bez_nadawy_h": "kruszarka bez nadawy [h]"})
    with st.expander("Zmiana po zmianie (wszystkie kolumny)"):
        st.dataframe(df.drop(columns=["etykieta"]), hide_index=True, width="stretch")
    st.download_button("⬇️ Pobierz wyniki (CSV)", df.drop(columns=["etykieta"]).to_csv(index=False).encode("utf-8"),
                       "kruszyk_okres.csv", "text/csv")


def _findings(df: pd.DataFrame, plan: PeriodPlan, sc, s: dict) -> list[str]:
    out = []
    if plan.target_t:
        miss = plan.target_t - s["wywiezione_t"]
        out.append(f"🎯 Cel {plan.target_t:,.0f} t: ".replace(",", " ") +
                   ("**zrobiony** ✅" if miss <= 0 else f"brakuje **{miss:,.0f} t** ({100 - s['realizacja_celu_pct']:.0f} %)".replace(",", " ")))
    empty = df[df["skonczyl_sie_urobek"] != ""]
    if len(empty):
        by_ex: dict[str, list[str]] = {}
        for _, r in empty.iterrows():
            for item in r["skonczyl_sie_urobek"].split(", "):
                by_ex.setdefault(item.split(" ")[0], []).append(f"{r['data'].strftime('%d.%m')} z{r['zmiana']}")
        for ex, when in by_ex.items():
            out.append(f"⛏️ **{ex}** zostaje bez urobku na **{len(when)}** zmianach "
                       f"({', '.join(when[:6])}{' …' if len(when) > 6 else ''}) – strzały za rzadko, za małe albo "
                       "brak wolnego odstrzelonego usypu, na który mogłaby przejechać.")
    mv = df[df["przejazdy_koparek"] != ""]
    for _, r in mv.iterrows():
        out.append(f"🚜 {r['data'].strftime('%d.%m')} z{r['zmiana']}: przejazd {r['przejazdy_koparek']}.")
    if s["kruszarka_bez_nadawy_h"] > 1:
        out.append(f"🏭 Kruszarka stała bez nadawy łącznie **{s['kruszarka_bez_nadawy_h']:.0f} h** w okresie.")
    zero = df[df["wywiezione_t"] <= 0]
    if len(zero):
        out.append(f"🛑 **{len(zero)}** zmian bez żadnego przewozu – wszystkie usypy puste.")
    rainy = int(df["deszcz"].sum())
    if rainy:
        out.append(f"🌧️ Deszcz na {rainy} z {len(df)} zmian (losowo, wg prawdopodobieństwa ze scenariusza).")
    return out or ["✅ Okres bez wyraźnych problemów."]
