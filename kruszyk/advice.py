"""Recommendations for the mine from the learned policy – what to change, and what it is worth.

The learned policy shows what a live system would do. Part of it the mine can do tomorrow without any system
(fewer trucks, another assignment of trucks to excavators); part of it needs live information (parking at a full
crusher, advice to drivers and operators). Every organisational recommendation is checked on the same shifts as
today's practice before it is shown – the policy's ideas that don't work without the system are left out.
"""
from __future__ import annotations

import copy
from collections import Counter

from .compare import compare, seeds_for
from .config import Scenario, truck_units
from .sim import simulate

N = 10  # shifts every recommendation is checked on
KEEP_T_PCT = 1.5  # "the same production": at most this much less


def _worth(r: dict) -> bool:
    """Worth recommending: cheaper per tonne for sure, or clearly cheaper without hauling less."""
    return r["cost_verdict"] == "lepiej" or (r["d_cost_pct"] < -0.5 and r["t_verdict"] != "gorzej")


def _numbers(r: dict) -> dict:
    return {"d_cost_pct": r["d_cost_pct"], "d_t_pct": r["d_t_pct"], "wins": r["cost_better"], "n": r["n"],
            "zl_t": r["variant"]["zl_t"], "t": r["variant"]["t"], "today_zl_t": r["base"]["zl_t"]}


def recommend(today: Scenario, with_policy: Scenario, seed: int = 0, progress=None) -> dict:
    """`today` = the mine as it works now; `with_policy` = the same with the learned policy (dispatch "rl")."""
    def step(i, text):
        if progress:
            progress(i / 5, text)
    seeds = seeds_for(seed, N)
    out: dict = {"organisational": [], "tradeoffs": [], "system": [], "n": N}

    # what the policy does, on the shifts of the check
    step(0, "co robi polityka")
    pol = copy.deepcopy(with_policy)
    pol.dispatch = "rl"
    runs = [simulate(pol, seed=s) for s in seeds]
    hours = today.shift_hours
    parked = sum(r.kpi["parked_truck_h"] for r in runs) / N
    exc_parked = sum(r.kpi["parked_excavator_h"] for r in runs) / N

    # 1. fewer trucks: the policy keeps some parked – how many can go without losing production?
    n_now = today.truck_count
    k = int(round(parked / hours)) if parked else 0
    if k >= 1 and n_now > 1:
        tried = []
        for m in range(max(1, n_now - k), n_now):
            step(1, f"sprawdzam {m} wozideł zamiast {n_now}")
            tried.append((m, compare(today, today.with_truck_count(m), n=N, seed=seed)))
        keep = [(m, r) for m, r in tried if _worth(r) and r["d_t_pct"] >= -KEEP_T_PCT]
        if keep:  # the cheapest fleet that hauls (practically) as much as today
            m, r = min(keep, key=lambda x: x[1]["d_cost_pct"])
            out["organisational"].append({
                "kind": "fleet", "title": f"Flota {m} wozideł zamiast {n_now}",
                "why": f"Polityka trzyma średnio {parked / hours:.1f} wozidła na parkingu, bo kruszarka i tak nie "
                       f"przyjmie więcej. {n_now - m} wozideł mniej daje praktycznie tyle samo ton bez żadnego systemu.",
                **_numbers(r)})
        cheaper = [(m, r) for m, r in tried if _worth(r) and r["d_t_pct"] < -KEEP_T_PCT]
        if cheaper:  # even fewer trucks: cheaper per tonne, but less production – a choice, not a free gain
            m, r = min(cheaper, key=lambda x: x[1]["d_cost_pct"])
            out["tradeoffs"].append({"title": f"Flota {m} wozideł", **_numbers(r)})

    # 2. another assignment of trucks to excavators: where each truck loaded most under the policy
    step(2, "sprawdzam przydział wozideł do koparek")
    loads = [r.loads for r in runs if len(r.loads)]
    if loads and len(today.excavators) > 1:
        import pandas as pd

        all_loads = pd.concat(loads)
        main = {t: Counter(g["excavator"]).most_common(1)[0][0] for t, g in all_loads.groupby("truck")}
        proposal = copy.deepcopy(today)
        changed = []
        units = dict(truck_units(proposal.trucks))
        if all(f.id and f.count <= 1 for f in proposal.trucks):  # one entry per truck: assign each one
            for tid, f in units.items():
                if tid in main and f.count:
                    if (f.excavators or [None])[0] != main[tid] or len(f.excavators) != 1:
                        changed.append((tid, list(f.excavators), main[tid]))
                    f.excavators, f.assign_locked = [main[tid]], True
        split = Counter(main.values())
        if changed:
            r = compare(today, proposal, n=N, seed=seed)
            idle = [e.id for e in today.excavators if e.id not in split]
            note = (f" {', '.join(idle)} zostaje bez wozideł – można ją odstawić albo przenieść na inny front "
                    "(w tym wyniku jeszcze liczona jako pracująca)." if idle else "")
            if _worth(r):
                out["organisational"].append({
                    "kind": "assignment", "title": "Inny przydział wozideł do koparek: " + ", ".join(
                        f"{ex} ← {', '.join(sorted(t for t, e in main.items() if e == ex))}" for ex in sorted(split)),
                    "why": "Tak polityka rozdzielała wozidła, gdy widziała kolejki. Sztygar może ustawić ten przydział "
                           "na stałe, bez systemu." + note, "changes": changed, **_numbers(r)})

    # 3. what only the live system can do – and what the whole system gives
    step(3, "cały system na żywo vs dziś")
    full = compare(today, with_policy, n=N, seed=seed, variant_dispatch="rl")
    pol_cfg = with_policy.excavator_policy
    if parked > 0.05:
        out["system"].append({"title": "Parkowanie wozideł, gdy kruszarka jest pełna",
                              "detail": f"średnio {parked:.1f} h parkowania na zmianę – silnik wyłączony zamiast kolejki"})
    if exc_parked > 0.05:
        out["system"].append({"title": "Odstawianie koparki, gdy wozideł jest za mało",
                              "detail": f"średnio {exc_parked:.1f} h na zmianę"})
    if pol_cfg.topup_min_fill < 0.999:
        out["system"].append({"title": "Bez ostatniej łyżki, gdy w kolejce czekają inne",
                              "detail": f"wozidło pełne w ≥ {100 * pol_cfg.topup_min_fill:.0f} %"})
    if with_policy.advisor.enabled:
        out["system"].append({"title": "Zapowiedzi dla operatorów i zalecana prędkość kierowców",
                              "detail": "łyżka nabrana, zanim wozidło podjedzie; dojazd, gdy koparka jest wolna"})
    out["system"].append({"title": "Przydział na żywo (kolejki, awarie, ruch na rampach)",
                          "detail": "wozidło jedzie tam, gdzie szybciej załaduje"})
    out["full"] = _numbers(full) | {"verdict": full["cost_verdict"]}
    step(5, "gotowe")
    return out
