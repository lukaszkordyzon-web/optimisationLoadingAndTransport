"""Shift organisation: today's blast goes to the place chosen for it."""
from pathlib import Path

import yaml

from kruszyk.config import scenario_from_dict
from kruszyk.sim import simulate
from kruszyk.site import blast_point, centroid

MINE = Path(__file__).parent.parent / "scenarios" / "kruszyk_kopalnia.yaml"


def test_crew_drives_to_the_chosen_blast_place():
    raw = yaml.safe_load(MINE.read_text(encoding="utf-8"))
    area = raw["site"]["areas"][1]
    raw["blasting"] = {"enabled": True, "place": area["id"]}
    sc = scenario_from_dict(raw)
    site = dict(sc.site, blast_place=area["id"])
    assert blast_point(site) == centroid(area["polygon"])
    crew = [v for v in simulate(sc, seed=1).light_vehicles if v["kind"] == "strzelanie"]
    assert crew and crew[0]["frames"]
