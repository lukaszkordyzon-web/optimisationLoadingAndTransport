"""CLI: python -m kruszyk <command> <scenario.yaml> [options]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analytics import match_factor_report
from .config import load_scenario
from .ml import generate_dataset, train_travel_model
from .optimize import recommend, sweep_trucks
from .sim import simulate


def _print_table(rows) -> None:
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_calc(args):
    r = match_factor_report(load_scenario(args.scenario))
    _print_table(r.pop("excavators"))
    print()
    for k, v in r.items():
        print(f"{k:24s} {v}")


def cmd_simulate(args):
    sc = load_scenario(args.scenario)
    if args.trucks:
        sc = sc.with_truck_count(args.trucks)
    if args.dispatch:
        sc.dispatch = args.dispatch
    res = simulate(sc, seed=args.seed)
    k = dict(res.kpi)
    _print_table(k.pop("excavators"))
    print()
    for key, v in k.items():
        print(f"{key:28s} {v}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        res.segments.to_csv(out / "segments.csv", index=False)
        res.states.to_csv(out / "states.csv", index=False)
        res.loads.to_csv(out / "loads.csv", index=False)
        res.hopper.to_csv(out / "hopper.csv", index=False)
        (out / "kpi.json").write_text(json.dumps(res.kpi, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDane zapisane w {out}/")


def cmd_sweep(args):
    sc = load_scenario(args.scenario)
    summary = sweep_trucks(sc, args.min, args.max, args.replications, args.dispatch)
    print(summary.to_string(index=False))
    best = recommend(summary, args.target_tph)
    print(f"\nRekomendacja: {int(best['trucks'])} wozideł "
          f"({best['throughput_tph']:.0f} t/h, {best['cost_per_t']:.2f} PLN/t)")


def cmd_train(args):
    sc = load_scenario(args.scenario)
    data = generate_dataset(sc, args.shifts)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.out, index=False)
    model, metrics = train_travel_model(data)
    for k, v in metrics.items():
        print(f"{k:24s} {v}")
    ex = sc.excavators[0]
    eta = model.eta_to_excavator_s(sc, "WZ-01", ex.id, segment_index=1, progress=0.5,
                                   rain=False, hour_of_shift=2.0)
    print(f"\nPrzykład ETA: WZ-01 w połowie rampy w dół -> {ex.id}: {eta / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kruszyk", description="Optymalizacja załadunku, transportu i kruszenia")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("calc", help="kalkulator analityczny: match factor, potrzebna liczba wozideł")
    s.add_argument("scenario")
    s.set_defaults(func=cmd_calc)

    s = sub.add_parser("simulate", help="symulacja jednej zmiany + dane syntetyczne")
    s.add_argument("scenario")
    s.add_argument("--trucks", type=int)
    s.add_argument("--dispatch", choices=["natural", "fixed", "dynamic"])
    s.add_argument("--seed", type=int)
    s.add_argument("--out", help="katalog na CSV z danymi syntetycznymi")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("sweep", help="dobór liczby wozideł symulacją")
    s.add_argument("scenario")
    s.add_argument("--min", type=int, default=2)
    s.add_argument("--max", type=int, default=14)
    s.add_argument("--replications", type=int, default=5)
    s.add_argument("--dispatch", choices=["natural", "fixed", "dynamic"])
    s.add_argument("--target-tph", type=float)
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("train", help="generuj dane z wielu zmian i trenuj model czasów przejazdu (ETA)")
    s.add_argument("scenario")
    s.add_argument("--shifts", type=int, default=40)
    s.add_argument("--out", help="ścieżka CSV na zbiór danych")
    s.set_defaults(func=cmd_train)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
