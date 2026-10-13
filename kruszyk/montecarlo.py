"""Chance that a period plan is met: the same plan calculated many times with different random shifts.

Each run draws its own weather, breakdowns, drivers and loading times for every shift (the period simulation
does that from the seed). With `model_sd` the calibration itself is uncertain too: every run gets its own travel,
loading and delay factors around the calibrated ones – otherwise the chance comes out too optimistic.
"""
from __future__ import annotations

import copy
import datetime as dt
import os

import numpy as np
import pandas as pd

from .config import Scenario
from .period import PeriodPlan, run_period

SEED_STEP = 7919


def _one(args) -> pd.DataFrame:
    sc, plan, seed, model_sd = args
    if model_sd > 0:
        rng = np.random.default_rng(seed + 1)
        sc = copy.deepcopy(sc)
        cal = sc.calibration
        cal.travel_factor *= float(rng.lognormal(0, model_sd))
        cal.load_factor *= float(rng.lognormal(0, model_sd))
        cal.delay_min_per_trip *= float(rng.lognormal(0, 2 * model_sd))
    return run_period(sc, plan, seed=seed)


WORKER_MB = 450   # one worker process: Python + numpy/pandas/simpy + a period run
MAX_WORKERS = 4


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def safe_workers() -> int:
    """Worker processes this container can afford – CPUs it may use and the memory it has left.
    Hosted servers (e.g. Streamlit Community Cloud) report the host's cores but give the app ~1–3 GB: too many
    processes and the whole app is killed. Override with KRUSZYK_WORKERS."""
    if os.environ.get("KRUSZYK_WORKERS"):
        return max(1, int(os.environ["KRUSZYK_WORKERS"]))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    quota = _read("/sys/fs/cgroup/cpu.max")  # "max 100000" or "200000 100000"
    if quota and not quota.startswith("max"):
        q, per = quota.split()[:2]
        cpus = min(cpus, max(1, int(int(q) / int(per))))
    free_mb = None
    limit = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read("/sys/fs/cgroup/memory.current") or _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit and limit != "max" and used and int(limit) < 1 << 50:
        free_mb = (int(limit) - int(used)) / 2**20
    if free_mb is None:
        info = _read("/proc/meminfo") or ""
        avail = next((line.split()[1] for line in info.splitlines() if line.startswith("MemAvailable")), None)
        free_mb = int(avail) / 1024 if avail else 0.0
    by_memory = int(free_mb // WORKER_MB) - 1  # keep one share for the app itself
    return max(1, min(cpus, by_memory, MAX_WORKERS))


def run_many(sc: Scenario, plan: PeriodPlan, n: int = 20, seed: int = 0, model_sd: float = 0.0,
             progress=None, workers: int | None = None) -> list[pd.DataFrame]:
    """`n` independent runs of the period. Uses the CPU cores when it can, one after another otherwise."""
    jobs = [(sc, plan, seed + SEED_STEP * (k + 1), model_sd) for k in range(n)]
    workers = workers if workers is not None else min(n, safe_workers())
    out: list[pd.DataFrame] = []
    if workers > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
                for df in ex.map(_one, jobs):
                    out.append(df)
                    if progress:
                        progress(len(out), n)
            return out
        except Exception:  # noqa: BLE001 – no processes on this host: fall back to one after another
            out = []
    for job in jobs:
        out.append(_one(job))
        if progress:
            progress(len(out), n)
    return out


def _q(values, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def chance(runs: list[pd.DataFrame], plan: PeriodPlan, target_t: float = 0.0, deadline: dt.date | None = None,
           goal=None) -> dict:
    """What the runs say: chance of the target by the deadline, P10/P50/P90, the band, products, risk drivers.
    `goal` = the applied production goal (GoalPlan) – then the chance of every product as well."""
    deadline = deadline or plan.start + dt.timedelta(days=plan.days - 1)
    rows, cums = [], []
    for df in runs:
        day = df.groupby("data")["wywiezione_t"].sum().cumsum()
        cums.append(day)
        to_deadline = float(day[day.index <= deadline].iloc[-1]) if (day.index <= deadline).any() else 0.0
        hit = day[day >= target_t].index[0] if target_t and (day >= target_t).any() else None
        t = float(df["wywiezione_t"].sum())
        row = {"wywiezione_t": t, "koszt_na_t": float(df["koszt_zl"].sum()) / t if t else None,
               "do_terminu_t": to_deadline, "cel_osiagniety": hit,
               "ok": bool(target_t) and to_deadline >= target_t,
               "deszcz_zmian": int(df["deszcz"].sum()), "awarie_h": float(df["awarie_h"].sum()),
               **{c: float(df[c].sum()) for c in df.columns if c.startswith("awari") and c != "awarie_h"},
               "zmian_bez_urobku": int((df["skonczyl_sie_urobek"] != "").sum()),
               "kruszarka_bez_nadawy_h": float(df["kruszarka_bez_nadawy_h"].sum())}
        if goal is not None:
            from .goal import products_made

            crushed = {c: float(df[f"kr_{c}_t"].sum()) for c in goal.yields if f"kr_{c}_t" in df}
            made = products_made(crushed, goal.yields, [p["produkt"] for p in goal.products])
            row["produkty"] = made
        rows.append(row)
    res = pd.DataFrame(rows)
    n = len(res)
    out = {"n": n, "target_t": target_t, "deadline": deadline,
           "t_p10": _q(res["wywiezione_t"], 0.1), "t_p50": _q(res["wywiezione_t"], 0.5),
           "t_p90": _q(res["wywiezione_t"], 0.9),
           "cost_p10": _q(res["koszt_na_t"].dropna(), 0.1), "cost_p50": _q(res["koszt_na_t"].dropna(), 0.5),
           "cost_p90": _q(res["koszt_na_t"].dropna(), 0.9), "runs": res}
    if target_t:
        p = float(res["ok"].mean())
        out["p"] = p
        out["p_margin"] = 1.96 * float(np.sqrt(max(p * (1 - p), 0.25 / n) / n))  # 95 % range of the estimate
        hits = sorted(h for h in res["cel_osiagniety"] if h is not None)
        out["hit_p50"] = hits[len(hits) // 2] if len(hits) * 2 > n else None
        out["hit_p90"] = hits[int(np.ceil(0.9 * n)) - 1] if len(hits) >= int(np.ceil(0.9 * n)) else None
    if goal is not None:
        out["products"] = [{"produkt": p["produkt"], "cel_t": p["cel_t"],
                            "szansa_pct": round(100 * float(np.mean([r[p["produkt"]] >= p["cel_t"] for r in res["produkty"]]))),
                            "p10_t": round(_q([r[p["produkt"]] for r in res["produkty"]], 0.1)),
                            "p50_t": round(_q([r[p["produkt"]] for r in res["produkty"]], 0.5))} for p in goal.products]
    # the band: cumulative tonnes per day
    days = sorted(set().union(*[set(c.index) for c in cums]))
    mat = np.array([[float(c[c.index <= d].iloc[-1]) if (c.index <= d).any() else 0.0 for d in days] for c in cums])
    out["band"] = pd.DataFrame({"data": days, "p10": np.quantile(mat, 0.1, axis=0), "p50": np.quantile(mat, 0.5, axis=0),
                                "p90": np.quantile(mat, 0.9, axis=0)})
    # what the failed runs had more of
    drivers = []
    if target_t and 0 < res["ok"].sum() < n:
        bad, good = res[~res["ok"]], res[res["ok"]]
        checks = [("deszcz_zmian", "zmian z deszczem", ""), ("awarie_wozidel_h", "godzin awarii wozideł", " h"),
                  ("awarie_koparek_h", "godzin awarii koparek", " h"), ("awarie_wiertnic_h", "godzin awarii wiertnic", " h"),
                  ("zmian_bez_urobku", "zmian z koparką bez urobku", ""),
                  ("kruszarka_bez_nadawy_h", "godzin kruszarki bez nadawy", " h")]
        checks += [(c, f"godzin awarii kruszarki {c[len('awaria_'):-2]}", " h") for c in res.columns
                   if c.startswith("awaria_") and c.endswith("_h")]
        for col, label, unit in checks:
            if col not in res:
                continue
            b, g = float(bad[col].mean()), float(good[col].mean())
            if b > g * 1.15 and b - g >= 0.5:
                drivers.append(((b - g) / max(g, 1.0), f"{label}: średnio **{b:.1f}{unit}** w nieudanych przebiegach "
                                                       f"vs {g:.1f}{unit} w udanych"))
        drivers = [text for _, text in sorted(drivers, reverse=True)]
    out["drivers"] = drivers
    return out
