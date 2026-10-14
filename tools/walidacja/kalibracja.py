"""Calibration on a table of shifts vs on one shift: which predicts the next weeks better?

A "hidden truth" = the test mine with parameters the calibration doesn't know (losses per trip, shift-to-shift
pace variation, slower driving). From it: 20 "measured" shifts (as a weigh scale and a dispatcher would give them),
then 40 future shifts. The base model is calibrated (a) on the table of 20 shifts, (b) on single shifts picked from
them (the old way) – and both predict the future 40 shifts: mean and P10–P90 of tonnes.

    PYTHONPATH=. python tools/walidacja/kalibracja.py   -> tools/walidacja/wyniki/kalibracja.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kopalnia_testowa import make  # noqa: E402

from kruszyk.calibrate import calibrate, calibrate_shifts, model_rows, spread_check  # noqa: E402

OUT = Path(__file__).parent / "wyniki"
TRUTHS = {  # name -> hidden calibration of the real mine
    "przestoje 6 min/kurs, zmienność 12 %": {"delay_min_per_trip": 6.0, "shift_sd": 0.12},
    "przestoje 3 min/kurs, bez zmienności": {"delay_min_per_trip": 3.0, "shift_sd": 0.0},
    "przestoje 4 min/kurs, zmienność 20 %": {"delay_min_per_trip": 4.0, "shift_sd": 0.2},
    "jazda wolniej o 15 %, przestoje 8 min, zmienność 10 %": {"travel_factor": 1.15, "delay_min_per_trip": 8.0,
                                                               "shift_sd": 0.1},
}
SINGLE = 5  # single-shift calibrations per truth


def run(name):
    truth = make()
    truth.dispatch = "natural"
    for k, v in TRUTHS[name].items():
        setattr(truth.calibration, k, v)
    measured, _ = model_rows(truth, tuple(range(500, 520)))
    future, _ = model_rows(truth, tuple(range(900, 940)))
    df = pd.DataFrame(measured)[["tonnes_t", "trips", "fuel_l"]]
    df["trucks"] = 3
    ft = np.array([r["tonnes_t"] for r in future])
    fitted, _, _, after = calibrate_shifts(make(), df)
    rows = [{"prawda": name, "kalibracja": "tabela 20 zmian", "prognoza_t": after["tonnes_t"]["mean"],
             "prognoza_p10": after["tonnes_t"]["p10"], "prognoza_p90": after["tonnes_t"]["p90"]}]
    rng = np.random.default_rng(1)
    for i in rng.choice(len(measured), SINGLE, replace=False):
        one = {k: measured[i][k] for k in ("tonnes_t", "trips")} | {"trucks": 3}
        f1, _, _ = calibrate(make(), one)
        f1.dispatch = "natural"
        chk = spread_check(f1)["tonnes_t"]
        rows.append({"prawda": name, "kalibracja": f"jedna zmiana ({measured[i]['tonnes_t']:.0f} t)",
                     "prognoza_t": chk["mean"], "prognoza_p10": chk["p10"], "prognoza_p90": chk["p90"]})
    for r in rows:
        r.update(przyszlosc_t=float(ft.mean()), przyszlosc_p10=float(np.quantile(ft, 0.1)),
                 przyszlosc_p90=float(np.quantile(ft, 0.9)))
        r["blad_sredniej_pct"] = 100 * (r["prognoza_t"] / r["przyszlosc_t"] - 1)
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        df = pd.DataFrame([r for rows in ex.map(run, TRUTHS) for r in rows])
    df.to_csv(OUT / "kalibracja.csv", index=False)
    for name, g in df.groupby("prawda", sort=False):
        f = g.iloc[0]
        print(f"\n{name}: przyszłe 40 zmian średnio {f.przyszlosc_t:.0f} t, P10–P90 {f.przyszlosc_p10:.0f}–{f.przyszlosc_p90:.0f}")
        for r in g.itertuples():
            print(f"  {r.kalibracja:28s} prognoza {r.prognoza_t:6.0f} t ({r.blad_sredniej_pct:+5.1f} %), "
                  f"P10–P90 {r.prognoza_p10:.0f}–{r.prognoza_p90:.0f}")
    many = df[df.kalibracja == "tabela 20 zmian"].blad_sredniej_pct.abs()
    one = df[df.kalibracja != "tabela 20 zmian"].blad_sredniej_pct.abs()
    print(f"\nśredni błąd prognozy: tabela 20 zmian {many.mean():.1f} % (maks. {many.max():.1f} %), "
          f"jedna zmiana {one.mean():.1f} % (maks. {one.max():.1f} %)")
