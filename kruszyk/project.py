"""A new, empty project: the usual settings of a template mine, no map objects and no machines."""
from __future__ import annotations

import copy

# what a new project keeps from a template mine: settings, not the mine itself
KEEP = ("shift_hours", "seed", "loose_density_t_m3", "rolling_resistance_pct", "variability", "costs",
        "shift_plan", "traffic", "inspection", "refueling", "spotting", "pacing")
# demo project: the former Zakrzówek limestone quarry in Kraków (closed in the 1990s, no operator, no client data)
DEMO_NAME = "Kamieniołom Demo – Zakrzówek (dane fikcyjne)"
DEMO_CENTER, DEMO_ZOOM = (50.0378, 19.9128), 16


def empty_site(center=(52.0, 19.4), zoom: int = 6) -> dict:
    return {"center": list(center), "zoom": zoom, "crusher": None, "piles": [], "excavators": [], "roads": []}


def new_project(template: dict, name: str, center=None, zoom: int = 16) -> dict:
    """An empty mine with the usual settings (shift, costs, variability) taken from `template`."""
    raw = {k: copy.deepcopy(template[k]) for k in KEEP if k in template}
    raw.update({"name": name.strip() or "Nowa kopalnia", "dispatch": "natural", "blasts": [], "excavators": [],
                "trucks": [], "routes": {}, "links": {}, "planned_blasts": [], "areas": [], "crushers_extra": [],
                "crusher": {"capacity_tph": 800.0, "hopper_t": 150.0, "dump_time_s": 60.0},
                "stockpile": {"enabled": False}, "blasting": {"enabled": False}, "drills": [],
                "site": empty_site(center, zoom) if center else empty_site()})
    raw["refueling"] = {k: v for k, v in (raw.get("refueling") or {}).items() if k != "tanker_times"}
    raw["inspection"] = {"enabled": False}
    return raw
