"""Scenario comparison on many shifts: is a variant really better, or was it one lucky shift?

Both scenarios run on the same shifts (same weather, breakdowns, drivers – common random numbers), so the
difference comes from the change, not from chance. Cost per tonne over the shifts = all costs / all tonnes.
The verdict uses a sign test: the variant is better / worse only if it wins on clearly more shifts than a coin would.
"""
from __future__ import annotations

import copy
import math

import pandas as pd

from .config import Scenario
from .sim import simulate

N_SHIFTS = (10, 20, 30)


def seeds_for(seed: int, n: int) -> list[int]:
    return [seed + 1000 + 37 * i for i in range(n)]


def shift_rows(sc: Scenario, seeds: list[int], progress=None, done: int = 0, total: int | None = None) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(seeds):
        k = simulate(sc, seed=s).kpi
        rows.append({"seed": s, "t": k["hauled_t"], "koszt": k["cost_total"], "paliwo_l": k["fuel_l"],
                     "kolejka_kruszarka_h": k["truck_queue_crusher_h"], "kolejka_koparki_h": k["truck_queue_excavator_h"],
                     "kruszarka_pct": 100 * k["crusher_utilization"], "deszcz": k["rain"]})
        if progress:
            progress(done + i + 1, total or len(seeds))
    df = pd.DataFrame(rows)
    df["zl_t"] = df["koszt"] / df["t"].where(df["t"] > 0)
    return df


def summary(df: pd.DataFrame) -> dict:
    t = float(df["t"].sum())
    return {"t": float(df["t"].mean()), "t_p10": float(df["t"].quantile(0.1)), "t_p90": float(df["t"].quantile(0.9)),
            "zl_t": float(df["koszt"].sum()) / t if t else None,
            "paliwo_l_t": float(df["paliwo_l"].sum()) / t if t else None,
            "kolejka_kruszarka_h": float(df["kolejka_kruszarka_h"].mean()),
            "kolejka_koparki_h": float(df["kolejka_koparki_h"].mean()),
            "kruszarka_pct": float(df["kruszarka_pct"].mean())}


def sign_threshold(n: int, alpha: float = 0.05) -> int:
    """Wins (out of n) a coin reaches with probability < alpha: fewer can be chance (one-sided binomial)."""
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n < alpha:
            return k
    return n + 1


def compare(base: Scenario, variant: Scenario, n: int = 20, seed: int = 0, dispatch: str = "natural",
            progress=None) -> dict:
    """Base and variant on the same n shifts. Returns summaries, per-shift rows and the verdict."""
    seeds = seeds_for(seed, n)
    a, b = copy.deepcopy(base), copy.deepcopy(variant)
    a.dispatch = b.dispatch = dispatch
    da = shift_rows(a, seeds, progress, 0, 2 * n)
    db = shift_rows(b, seeds, progress, n, 2 * n)
    sa, sb = summary(da), summary(db)
    cost_better = (db["zl_t"] < da["zl_t"] - 1e-9).sum()
    cost_worse = (db["zl_t"] > da["zl_t"] + 1e-9).sum()
    t_more = (db["t"] > da["t"] + 1e-6).sum()
    t_less = (db["t"] < da["t"] - 1e-6).sum()
    need = sign_threshold(n)
    d_cost = 100 * (sb["zl_t"] / sa["zl_t"] - 1) if sa["zl_t"] and sb["zl_t"] else 0.0
    d_t = 100 * (sb["t"] / sa["t"] - 1) if sa["t"] else 0.0

    def verdict(better: int, worse: int) -> str:
        return "lepiej" if better >= need else "gorzej" if worse >= need else "bez pewnej różnicy"
    per_shift = pd.DataFrame({"zmiana": range(1, n + 1), "seed": seeds, "dzis_t": da["t"], "wariant_t": db["t"],
                              "dzis_zl_t": da["zl_t"], "wariant_zl_t": db["zl_t"], "deszcz": da["deszcz"]})
    return {"n": n, "base": sa, "variant": sb, "d_cost_pct": d_cost, "d_t_pct": d_t,
            "cost_better": int(cost_better), "cost_worse": int(cost_worse), "t_more": int(t_more), "t_less": int(t_less),
            "need": need, "cost_verdict": verdict(int(cost_better), int(cost_worse)),
            "t_verdict": verdict(int(t_more), int(t_less)), "per_shift": per_shift}


def fleet_sweep(sc: Scenario, counts: list[int], n: int = 20, seed: int = 0, dispatch: str = "natural",
                progress=None) -> tuple[pd.DataFrame, dict]:
    """The same mine with 1, 2, 3 … trucks on the same n shifts. Returns the table and the recommendation:
    the cheapest fleet and the smallest one that hauls (almost) as much as the largest."""
    seeds = seeds_for(seed, n)
    rows = []
    for j, c in enumerate(counts):
        v = sc.with_truck_count(int(c))
        v.dispatch = dispatch
        s = summary(shift_rows(v, seeds, progress, j * n, len(counts) * n))
        rows.append({"wozidla": int(c), **s})
    tab = pd.DataFrame(rows)
    best_cost = tab.loc[tab["zl_t"].idxmin()]
    top = float(tab["t"].max())
    enough = tab[tab["t"] >= 0.97 * top].iloc[0]
    return tab, {"najtansza": int(best_cost["wozidla"]), "najtansza_zl_t": float(best_cost["zl_t"]),
                 "najtansza_t": float(best_cost["t"]), "pelna_produkcja": int(enough["wozidla"]),
                 "pelna_t": float(enough["t"]), "max_t": top,
                 "plateau": bool((tab["t"].diff().iloc[1:] / tab["t"].iloc[:-1].values < 0.01).any())}


def noise_note(n: int) -> str:
    return (f"Wariant jest „lepszy”, jeśli wygrywa na co najmniej {sign_threshold(n)} z {n} zmian – mniej może być "
            "przypadkiem (rzut monetą).")
