"""Queues seen on real sites: 2–3 trucks waiting under an excavator, trucks bunching on single-lane ramps at passing
bays. Test mine with single-lane ramps, a big crusher (not the bottleneck), slower excavators and 6 trucks;
feed mix 50/50 from both levels. Each policy on the same 30 shifts.

    PYTHONPATH=. python tools/walidacja/kolejki.py   -> tools/walidacja/wyniki/kolejki.csv
"""
from __future__ import annotations

import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kopalnia_testowa import make  # noqa: E402

from kruszyk.rl import explain_advisor, explain_excavators, train  # noqa: E402
from kruszyk.sim import simulate  # noqa: E402

OUT = Path(__file__).parent / "wyniki"
MINE = {"single_lane": True, "trucks": 6, "crusher_tph": 800, "bucket_m3": 3.0, "swing_s": 30, "share_b1": 50.0}
EVAL = range(700, 730)


def row(name, k):
    return {"polityka": name, "t": k["hauled_t"], "zl_t": k["cost_per_t"], "paliwo_l_t": k["fuel_l_per_t"],
            "kolejka_kop_h": k["truck_queue_excavator_h"], "mijanki_h": k["bay_wait_h"], "zatrzyman_mijanki": k["bay_stops"],
            "kolejka_kr_h": k["truck_queue_crusher_h"], "koparki_czekaly_h": sum(e["waiting_for_trucks_h"] for e in k["excavators"]),
            "parking_h": k["parked_truck_h"], "skad_B1_pct": k["feed_share"].get("B1", 0.0), "nabrane_lyzki": k["prefills"]}


def policy(advice: bool):
    sc = make(**MINE)
    res = train(sc, iterations=12, population=12, episodes=2, shift_hours=4.0, eval_episodes=2, seed=1,
                advisor=advice, learn_advisor=advice, learn_excavators=True)
    pol = copy.deepcopy(sc)
    pol.dispatch, pol.rl_weights, pol.advisor.enabled = "rl", res.weights, advice
    for k, v in (res.excavators or {}).items():
        setattr(pol.excavator_policy, k, v)
    for k, v in (res.advisor or {}).items():
        setattr(pol.advisor, k, v)
    name = "RL + zalecenia prędkości" if advice else "RL – sam przydział"
    lessons = explain_excavators(res.excavators or {}) + (explain_advisor(res.advisor) if res.advisor else [])
    return [{**row(name, simulate(pol, seed=s).kpi), "seed": s} for s in EVAL], name, lessons


def baseline(disp: str):
    sc = make(**MINE)
    sc.dispatch = disp
    name = {"natural": "jak dziś (ludzie)", "dynamic": "system – heurystyka"}[disp]
    return [{**row(name, simulate(sc, seed=s).kpi), "seed": s} for s in EVAL], name, []


def job(j):
    return baseline(j) if isinstance(j, str) else policy(j)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(job, ["natural", "dynamic", False, True]))
    df = pd.DataFrame([r for rows, _, _ in out for r in rows])
    df.to_csv(OUT / "kolejki.csv", index=False)
    df["koszt"] = df["zl_t"] * df["t"]
    tab = df.groupby("polityka", sort=False).mean(numeric_only=True).drop(columns=["seed", "koszt"])
    tab["zl_t"] = df.groupby("polityka", sort=False)["koszt"].sum() / df.groupby("polityka", sort=False)["t"].sum()
    print(tab.round(2).to_string())
    base = df[df.polityka == "jak dziś (ludzie)"].set_index("seed")["zl_t"]
    for name in df.polityka.unique()[1:]:
        d = df[df.polityka == name].set_index("seed")["zl_t"] - base
        print(f"{name}: mediana {d.median():+.2f} zł/t vs jak dziś, lepiej na {(d < 0).sum()}/30 zmianach")
    for _, name, lessons in out:
        if lessons:
            print(name, "\n  " + "\n  ".join(lessons))
    # infrastructure instead of dispatch: how much one more passing bay / a second lane is worth
    rows = []
    for label, kw in (("1 pas, 1 mijanka", {}), ("1 pas, 2 mijanki", {"bays": 2}), ("1 pas, 3 mijanki", {"bays": 3}),
                      ("2 pasy", {"single_lane": False})):
        for disp in ("natural", "dynamic"):
            for s_ in EVAL:
                sc = make(**{**MINE, **kw})
                sc.dispatch = disp
                k = simulate(sc, seed=s_).kpi
                rows.append({"rampa": label, "przydzial": disp, "t": k["hauled_t"], "zl_t": k["cost_per_t"],
                             "mijanki_h": k["bay_wait_h"]})
    inf = pd.DataFrame(rows)
    inf["koszt"] = inf["zl_t"] * inf["t"]
    infra = inf.groupby(["rampa", "przydzial"], sort=False).agg(t=("t", "mean"), koszt=("koszt", "sum"),
                                                                 suma_t=("t", "sum"), mijanki_h=("mijanki_h", "mean"))
    infra["zl_t"] = infra.pop("koszt") / infra.pop("suma_t")  # all costs / all tonnes (not a mean of shifts)
    infra = infra.round(2)
    infra.to_csv(OUT / "mijanki.csv")
    print(infra.to_string())
