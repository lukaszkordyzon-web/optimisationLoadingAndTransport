"""Sensitivity of the shift model on the test mine: one parameter at a time, then two at a time.

    PYTHONPATH=. python tools/walidacja/czulosc.py      -> tools/walidacja/wyniki/czulosc.csv, siatka.csv
Every point is the mean of many shifts (different weather, drivers, breakdowns); every run is also checked for
basic physics (mass balance, utilisation <= 100 %, positive cost).
"""
from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kopalnia_testowa import BASE, make  # noqa: E402

from kruszyk.sim import simulate  # noqa: E402

OUT = Path(__file__).parent / "wyniki"
SEEDS = 20

FACTORS = {  # parameter -> levels (the base value is one of them)
    "trucks": [1, 2, 3, 4, 5, 6, 8],
    "crusher_tph": [250, 300, 400, 500, 600, 800],
    "bucket_m3": [2.5, 3.5, 4.5, 5.5, 6.5],
    "grade_pct": [4, 6, 8, 10, 12],
    "dist": [0.5, 1.0, 1.5, 2.0, 3.0],
    "payload_t": [28, 36, 45, 55],
    "frag": [0.1, 0.3, 0.5, 0.7, 0.9],
    "rain": [0.0, 0.5, 1.0],
    "breakdowns": [False, True],
    "delay_min": [0, 2, 5, 10],
}


def _params(p: dict) -> dict:
    p = dict(p)
    d = p.pop("dist", 1.0)
    for k in ("flat_m", "ramp1_m", "ramp2_m", "bench_m"):
        p[k] = BASE[k] * d
    return p


def one(args) -> dict:
    p, seed, dispatch = args
    sc = make(**_params(p))
    sc.dispatch = dispatch
    k = simulate(sc, seed=seed).kpi
    h = sc.shift_hours
    checks = {
        "ok_masa": k["crushed_t"] <= k["hauled_t"] + 1e-6 and k["hauled_t"] - k["crushed_t"] <= sc.crusher.hopper_t + 1,
        "ok_kruszarka": k["crushed_t"] <= sc.crusher.capacity_tph * h + 1,
        "ok_wykorzystanie": all(0 <= c["utilization"] <= 1.0001 for c in k["crushers"])
        and all(0 <= e["utilization"] <= 1.0001 for e in k["excavators"]),
        "ok_koszt": (k["cost_per_t"] or 0) > 0 if k["hauled_t"] else True,
    }
    return {**p, "seed": seed, "dispatch": dispatch, "t": k["hauled_t"], "zl_t": k["cost_per_t"],
            "paliwo_l_t": k["fuel_l_per_t"], "kwh_t": k["crusher_kwh_per_t"], "kruszarka_pct": 100 * k["crusher_utilization"],
            "kolejka_kr_h": k["truck_queue_crusher_h"], "kolejka_kop_h": k["truck_queue_excavator_h"],
            "kursy": k["cycles"], "awarie_h": k["truck_breakdown_h"] + k["crusher_breakdown_h"], **checks}


def run(points: list[dict], seeds: int, dispatch: str = "natural", workers: int = 4) -> pd.DataFrame:
    jobs = [(p, s, dispatch) for p in points for s in range(seeds)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=8))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pts = [{name: lvl} for name, levels in FACTORS.items() for lvl in levels]
    df = run(pts, SEEDS)
    df["czynnik"] = [next(iter(p)) for p in pts for _ in range(SEEDS)]
    df.to_csv(OUT / "czulosc.csv", index=False)
    grid = [{"trucks": n, "crusher_tph": c, "dist": d}
            for n, c, d in itertools.product([2, 3, 4, 5, 6, 7], [300, 400, 600, 800], [1.0, 2.0])]
    run(grid, 10).to_csv(OUT / "siatka.csv", index=False)
    print("symulacji:", len(df) + len(grid) * 10)
