import copy
from pathlib import Path

import pytest

from kruszyk.config import dump_scenario, load_scenario, scenario_from_dict
from kruszyk.sim import simulate
from kruszyk.site import build_route, haversine_m, parse_coords, polygon_area_m2, site_to_scenario
from kruszyk.views import snapshot

import yaml

SCENARIO = Path(__file__).parent.parent / "scenarios" / "kruszyk_bazowy.yaml"

# synthetic quarry: crusher at C, haul road 1 km east, then a ramp ~600 m south down into the pit
C = (50.0, 20.0)
J = (50.0, 20.014)      # junction ~1 km east
P = (49.9946, 20.014)   # pit bottom ~600 m south of J


def demo_site():
    return {
        "center": list(C), "zoom": 16,
        "crusher": {"lat": C[0], "lon": C[1]},
        "roads": [
            # drawn from the crusher outwards on purpose: drawing direction must not matter
            {"id": "R1", "name": "droga główna", "kind": "droga", "grade_pct": 1.0, "speed_limit_kmh": 40,
             "coords": [list(C), [50.0, 20.007], list(J)]},
            {"id": "R2", "name": "rampa", "kind": "rampa", "grade_pct": 9.0, "speed_limit_kmh": 25,
             "coords": [[J[0], J[1] + 0.0001], list(P)]},  # starts ~7 m from J -> snapped
        ],
        "piles": [{"id": "U-01", "material": "wapień", "tonnage_t": 5000, "fragmentation": 0.4,
                   "loose_density_t_m3": 1.7,
                   "polygon": [[49.9940, 20.0135], [49.9940, 20.0145], [49.9935, 20.0145], [49.9935, 20.0135]]}],
        "excavators": [{"id": "K-1", "lat": 49.9943, "lon": 20.014, "pile": "U-01", "bucket_m3": 6, "swing_cycle_s": 30}],
    }


@pytest.mark.parametrize("text,expected", [
    ("https://www.google.com/maps/@50.8712345,20.6283456,500m/data=!3m1!1e3", (50.8712345, 20.6283456)),
    ("50.8712, 20.6283", (50.8712, 20.6283)),
    ("https://maps.google.com/?q=50.1,19.9", (50.1, 19.9)),
    ("https://www.google.com/maps/place/X/data=!3d50.25!4d19.75", (50.25, 19.75)),
    ("Kruszyk", None),
])
def test_parse_coords(text, expected):
    assert parse_coords(text) == expected


def test_route_from_map():
    site = demo_site()
    r = build_route(site, (49.9943, 20.014), "T")
    names = [s["name"] for s in r["segments"]]
    assert names[:1] == ["front"] and "rampa" in names and "droga główna" in names
    total = sum(s["length_m"] for s in r["segments"])
    expected = haversine_m((49.9943, 20.014), P) + haversine_m(J, P) + haversine_m(C, J)
    assert total == pytest.approx(expected, rel=0.03)
    # ramp keeps its grade towards the crusher regardless of drawing direction
    assert next(s for s in r["segments"] if s["name"] == "rampa")["grade_pct"] == 9.0
    assert r["path"][0] == [49.9943, 20.014] and r["path"][-1] == list(C)


def test_disconnected_roads_are_reported():
    site = demo_site()
    site["roads"][1]["coords"][0] = [50.003, 20.03]  # ramp far away from the haul road
    with pytest.raises(ValueError, match="brak połączenia"):
        build_route(site, (49.9943, 20.014), "T")


def test_site_to_scenario_runs_simulation():
    raw = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    new = site_to_scenario(demo_site(), raw)
    sc = scenario_from_dict(copy.deepcopy(new))
    assert [b.id for b in sc.blasts] == ["U-01"] and sc.blasts[0].material == "wapień"
    assert sc.routes["T-K-1"].path
    # round trip through YAML keeps the map
    sc2 = scenario_from_dict(yaml.safe_load(dump_scenario(sc)))
    assert sc2.site["piles"][0]["material"] == "wapień" and sc2.routes["T-K-1"].path
    res = simulate(sc)
    assert res.kpi["hauled_t"] > 0
    snap = snapshot(res, 3600)
    assert snap["lat"].between(49.99, 50.01).all()


def test_polygon_area():
    sq = [[50.0, 20.0], [50.0, 20.0014], [49.99910, 20.0014], [49.99910, 20.0]]  # ~100 x 100 m
    assert polygon_area_m2(sq) == pytest.approx(10_000, rel=0.03)


def test_drawing_type_follows_shape():
    from app_map import add_drawing, empty_site

    s = empty_site()
    line = {"type": "LineString", "coordinates": [[20, 50], [20.01, 50]]}
    poly = {"type": "Polygon", "coordinates": [[[20, 50], [20.001, 50], [20.001, 50.001], [20, 50]]]}
    point = {"type": "Point", "coordinates": [20, 50]}
    assert "drogę" in add_drawing(s, line, "Usyp (materiał)", {"id": "U-01"})  # shape wins over selection
    add_drawing(s, poly, "Usyp (materiał)", {"id": "U-01", "material": "wapień"})
    add_drawing(s, poly, "Usyp (materiał)", {"id": "U-01", "material": "wapień"})  # duplicate id renamed
    add_drawing(s, point, "Koparka", {"id": "K-1", "pile": "U-01"})
    add_drawing(s, point, "Kruszarka", {})
    assert [p["id"] for p in s["piles"]] == ["U-01", "U-02"]
    assert s["piles"][0]["material"] == "wapień"
    assert [e["id"] for e in s["excavators"]] == ["K-1"] and s["crusher"]
