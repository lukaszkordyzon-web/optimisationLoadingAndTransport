"""Does the comparison on many shifts give sure answers where one shift misleads?

Three "what if" questions on the test mine with small, typical differences. The true answer comes from 200 shifts.
Each question is then asked 40 times on different shifts: once comparing ONE shift (the old variant screen), once
comparing 20 shifts with the sign test (the new one). Counted: right / wrong / "no sure difference".

    PYTHONPATH=. python tools/walidacja/warianty.py   -> tools/walidacja/wyniki/warianty.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kopalnia_testowa import make  # noqa: E402

from kruszyk.compare import compare  # noqa: E402

OUT = Path(__file__).parent / "wyniki"
ONE_LANE = {"single_lane": True, "trucks": 6, "crusher_tph": 800, "bucket_m3": 3.0, "swing_s": 30}
QUESTIONS = {  # name -> (base, variant, measure: "cost" = cheaper per tonne? / "t" = more tonnes?)
    "4 zamiast 3 wozideł – taniej na tonę?": ({"trucks": 3}, {"trucks": 4}, "cost"),
    "4 zamiast 3 wozideł – więcej ton?": ({"trucks": 3}, {"trucks": 4}, "t"),
    "druga mijanka na rampie – więcej ton?": (ONE_LANE, {**ONE_LANE, "bays": 2}, "t"),
    "łyżka 5 m³ zamiast 4,5 – więcej ton?": ({}, {"bucket_m3": 5.0}, "t"),
    "wozidła 40 t zamiast 36 t – taniej na tonę?": ({}, {"payload_t": 40.0}, "cost"),
}
REPEATS, N = 40, 20


def answer(r: dict, measure: str) -> str:
    v = r["cost_verdict"] if measure == "cost" else r["t_verdict"]
    return {"lepiej": "tak", "gorzej": "nie"}.get(v, "nie wiadomo")


def one_shift(r: dict, measure: str) -> str:
    ps = r["per_shift"].iloc[0]
    if measure == "cost":
        return "tak" if ps["wariant_zl_t"] < ps["dzis_zl_t"] else "nie"
    return "tak" if ps["wariant_t"] > ps["dzis_t"] else "nie" if ps["wariant_t"] < ps["dzis_t"] else "nie wiadomo"


def job(args):
    q, rep = args
    base, var, measure = QUESTIONS[q]
    one = compare(make(**base), make(**var), n=1, seed=50_000 + 101 * rep)
    many = compare(make(**base), make(**var), n=N, seed=90_000 + 997 * rep)
    return {"pytanie": q, "powtorzenie": rep, "jedna_zmiana": one_shift(one, measure), "20_zmian": answer(many, measure)}


def truth(q):
    base, var, measure = QUESTIONS[q]
    r = compare(make(**base), make(**var), n=200, seed=7)
    d = r["d_cost_pct"] if measure == "cost" else r["d_t_pct"]
    good = d < 0 if measure == "cost" else d > 0
    return q, ("tak" if good else "nie"), d


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        truths = {q: (a, d) for q, a, d in ex.map(truth, QUESTIONS)}
        rows = list(ex.map(job, [(q, i) for q in QUESTIONS for i in range(REPEATS)], chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "warianty.csv", index=False)
    for q, g in df.groupby("pytanie", sort=False):
        right, d = truths[q]
        print(f"\n{q}  prawda (200 zmian): {right} ({d:+.1f} %)")
        for col in ("jedna_zmiana", "20_zmian"):
            c = g[col].value_counts()
            ok, bad = int(c.get(right, 0)), int(c.get("nie" if right == "tak" else "tak", 0))
            print(f"  {col:13s}: dobrze {ok}/{REPEATS}, źle {bad}/{REPEATS}, „nie wiadomo” {int(c.get('nie wiadomo', 0))}")
