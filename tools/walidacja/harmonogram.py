"""Does the period schedule give sensible answers on the test mine?

  1. a month with plenty of material: tonnes ≈ shifts × an average shift; the cheapest fleet = the one found shift by shift
  2. a month with two small piles and no blasting: production must stop when they are taken out
  3. the same with blasting every 2 working days: production must go on
  4. chance of the plan (Monte Carlo): P10 / P50 / P90 of the month, chance of several targets
  5. production goal planned backwards (LP): feasible, sensible feed and fleet

    PYTHONPATH=. python tools/walidacja/harmonogram.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kopalnia_testowa import make  # noqa: E402

from kruszyk.goal import plan_backwards  # noqa: E402
from kruszyk.montecarlo import chance, run_many  # noqa: E402
from kruszyk.period import PeriodPlan, PileBlasts, optimize_period, run_period, summary  # noqa: E402

OUT = Path(__file__).parent / "wyniki"
START = dt.date(2026, 11, 2)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    lines = []
    big = make(b1_t=300_000, b2_t=300_000)
    plan = PeriodPlan(start=START, days=28, shifts_per_day=2)
    n_sh = len(plan.shifts())
    res = optimize_period(big, plan, "cheapest", trucks=[2, 3, 4, 5, 6], shifts=[2], saturdays=[False],
                          dispatches=["natural"], seed=3)
    tab = res[["wozidla", "wywiezione_t", "koszt_na_t"]].sort_values("wozidla")
    tab.to_csv(OUT / "harmonogram_flota.csv", index=False)
    lines.append(f"1. Miesiąc ({n_sh} zmian), dużo urobku – flota: "
                 + "; ".join(f"{r.wozidla} woz. {r.wywiezione_t:,.0f} t {r.koszt_na_t:.2f} zł/t" for r in tab.itertuples())
                 + f" → najtańsza: {int(res.iloc[0]['wozidla'])} wozidła")
    df = run_period(big, plan, seed=3)
    lines.append(f"   3 wozidła: {df['wywiezione_t'].sum():,.0f} t = {df['wywiezione_t'].mean():,.0f} t na zmianę "
                 f"(sd {df['wywiezione_t'].std():,.0f})")

    small = make(b1_t=15_000, b2_t=15_000)
    df = run_period(small, plan, seed=3)
    s = summary(df, plan)
    stop = df.index[df["wywiezione_t"] < 100].min()
    lines.append(f"2. Dwa usypy po 15 000 t, bez strzałów: {s['wywiezione_t']:,.0f} t w okresie, produkcja staje od "
                 f"zmiany nr {stop + 1 if stop == stop else '–'} (zmian bez urobku: {s['zmian_bez_urobku']})")
    blasts = PeriodPlan(start=START, days=28, shifts_per_day=2,
                        blasts=[PileBlasts("B1", 10_000, every_days=2, first_day=2),
                                PileBlasts("B2", 10_000, every_days=2, first_day=3)])
    df = run_period(small, blasts, seed=3)
    s = summary(df, blasts)
    lines.append(f"3. To samo + strzał 10 000 t co 2 dni robocze na każdym poziomie: {s['wywiezione_t']:,.0f} t, "
                 f"zmian bez urobku: {s['zmian_bez_urobku']}")

    runs = run_many(big, plan, n=40, seed=11)
    totals = pd.Series([float(r["wywiezione_t"].sum()) for r in runs])
    lines.append(f"4. Szansa realizacji (40 wariantów miesiąca): P10 {totals.quantile(.1):,.0f} t, P50 "
                 f"{totals.quantile(.5):,.0f} t, P90 {totals.quantile(.9):,.0f} t")
    for target in (100_000, 110_000, 115_000, 120_000):
        c = chance(runs, plan, target_t=target)
        lines.append(f"   cel {target:,.0f} t → szansa {100 * c['p']:.0f} %")

    gp = plan_backwards(big, total_t=60_000, mix={"grysy (2–16)": 60.0, "kliniec 16–31,5": 20.0,
                                                  "piasek łamany 0–2": 10.0},
                        yields={"C-1": {"grys 2–8": 25.0, "grys 8–16": 30.0, "kliniec 16–31,5": 20.0,
                                        "mieszanka 0–31,5": 10.0, "piasek łamany 0–2": 10.0}}, shifts=n_sh)
    feed = sum(f["t"] for f in gp.flows) if gp.flows and "t" in gp.flows[0] else gp.feed_t
    lines.append(f"5. Cel 60 000 t produktów (60 % grysy, 20 % kliniec, 10 % piasek): wykonalny={gp.feasible}, "
                 f"nadawa {feed:,.0f} t, koszt pełny {gp.full_cost_per_t} zł/t produktu (zmienny {gp.cost_per_t}), transport: {gp.haulage}")
    text = "\n".join(lines).replace(",", " ")
    (OUT / "harmonogram.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
