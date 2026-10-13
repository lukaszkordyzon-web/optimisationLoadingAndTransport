"""Editing objects on the map: selection by click, move, redraw, delete, rename references."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import yaml  # noqa: E402

from app_map import (apply_edit_drawing, delete_obj, ensure_uids, find_obj, rename_refs, sel_from_click,  # noqa: E402
                     site_items)

DEMO = Path(__file__).parent.parent / "scenarios" / "kruszyk_mapa_demo.yaml"


@pytest.fixture
def site():
    s = yaml.safe_load(DEMO.read_text())["site"]
    ensure_uids(s)
    return s


def test_click_tooltip_selects_object(site):
    road = site["roads"][0]
    assert sel_from_click(site, f"🛣️ {road['id']} · {road['name']} · …", {"lat": 0, "lng": 0}) == f"road:{road['uid']}"
    ex = site["excavators"][1]
    assert sel_from_click(site, f"⛏️ {ex['id']} · koparka → U-02", None) == f"excavator:{ex['uid']}"
    assert sel_from_click(site, "🏭 Kruszarka", None) == "crusher:"
    p = site["parking"][0]
    assert sel_from_click(site, f"🅿️ postój {p['excavator']} · miejsce postoju koparki", None) == f"parking:{p['excavator']}"
    # no tooltip → nearest object
    assert sel_from_click(site, None, {"lat": ex["lat"], "lng": ex["lon"]}) == f"excavator:{ex['uid']}"
    assert sel_from_click(site, None, {"lat": 0.0, "lng": 0.0}) is None


def test_every_item_is_findable(site):
    for sel in site_items(site):
        assert find_obj(site, sel) is not None, sel


def test_move_and_redraw(site):
    ex = site["excavators"][0]
    msg = apply_edit_drawing(site, f"excavator:{ex['uid']}", {"type": "Point", "coordinates": [20.5, 50.5]})
    assert msg.startswith("✅") and (ex["lat"], ex["lon"]) == (50.5, 20.5)
    assert apply_edit_drawing(site, f"excavator:{ex['uid']}", {"type": "LineString", "coordinates": []}).startswith("⚠️")
    road = site["roads"][0]
    apply_edit_drawing(site, f"road:{road['uid']}", {"type": "LineString", "coordinates": [[20.0, 50.0], [20.001, 50.0]]})
    assert road["coords"] == [[50.0, 20.0], [50.0, 20.001]]
    pile = site["piles"][0]
    apply_edit_drawing(site, f"pile:{pile['uid']}", {"type": "Polygon", "coordinates": [[[20, 50], [20.001, 50],
                                                                                         [20.001, 50.001], [20, 50]]]})
    assert len(pile["polygon"]) == 3


def test_delete_cleans_references(site):
    ex = next(e for e in site["excavators"] if any(p["excavator"] == e["id"] for p in site["parking"]))
    delete_obj(site, f"excavator:{ex['uid']}")
    assert ex["id"] not in {e["id"] for e in site["excavators"]}
    assert all(p["excavator"] != ex["id"] for p in site["parking"])
    pile = site["piles"][0]
    users = [e for e in site["excavators"] if e.get("pile") == pile["id"]]
    delete_obj(site, f"pile:{pile['uid']}")
    assert all(e["pile"] == "(najbliższy)" for e in users)
    n = len(site["roads"])
    delete_obj(site, f"road:{site['roads'][0]['uid']}")
    assert len(site["roads"]) == n - 1
    delete_obj(site, "crusher:")
    assert site["crusher"] is None


def test_rename_updates_references(site):
    ex = site["excavators"][0]
    old = ex["id"]
    site.setdefault("parking", []).append({"excavator": old, "lat": 0, "lon": 0})
    rename_refs(site, "excavators", old, "NEW")
    assert all(p["excavator"] != old for p in site["parking"])
    pile = site["piles"][0]
    users = [e for e in site["excavators"] if e.get("pile") == pile["id"]]
    rename_refs(site, "piles", pile["id"], "U-X")
    assert users and all(e["pile"] == "U-X" for e in users)


def test_passing_bay_snaps_to_road_and_marks_it_single_lane(site):
    from app_map import add_bay
    from kruszyk.site import along_m

    road = next(r for r in site["roads"] if r["id"] == "R4")
    road["single_lane"] = False
    a, b = road["coords"][0], road["coords"][1]
    msg = add_bay(site, (a[0] + b[0]) / 2 + 0.0001, (a[1] + b[1]) / 2)  # ~10 m off the road
    assert msg.startswith("✅") and road["single_lane"]
    bay = site["passing_bays"][-1]
    assert along_m(road["coords"], (bay["lat"], bay["lon"]))[1] < 1.0  # snapped onto the road
    assert add_bay(site, 0.0, 0.0).startswith("⚠️")
    sel = sel_from_click(site, f"⇆ {bay['id']} · mijanka", None)
    assert sel == f"bay:{bay['uid']}"
    apply_edit_drawing(site, sel, {"type": "Point", "coordinates": [b[1], b[0]]})
    assert find_obj(site, sel) is not None  # moved, still the same bay
    delete_obj(site, sel)
    assert bay["uid"] not in {x["uid"] for x in site["passing_bays"]}


def test_second_crusher_on_the_map(site):
    from app_map import add_drawing

    msg = add_drawing(site, {"type": "Point", "coordinates": [20.024, 49.9946]}, "Kruszarka",
                      {"new_crusher": True, "id": "C-2", "capacity_tph": 900.0, "hopper_t": 120.0,
                       "accepts": ["U-03"], "product": "tłuczeń", "target_t": 3000.0})
    assert msg.startswith("✅") and site["crushers"][0]["accepts"] == ["U-03"]
    main_before = dict(site["crusher"])
    add_drawing(site, {"type": "Point", "coordinates": [20.0, 50.0]}, "Kruszarka", {})  # moves the main one
    assert (site["crusher"]["lat"], site["crusher"]["lon"]) == (50.0, 20.0)
    assert site["crusher"].get("elev_m") == main_before.get("elev_m")  # other settings kept
    c2 = site["crushers"][0]
    sel = sel_from_click(site, "🏭 C-2 · kruszarka · 900 t/h", None)
    assert sel == f"crusher2:{c2['uid']}"
    assert sel_from_click(site, "🏭 C-1 · kruszarka główna", None) == "crusher:"
    delete_obj(site, sel)
    assert site["crushers"] == []


def test_pile_lying_or_planned_blast(site):
    from app_map import add_drawing

    poly = {"type": "Polygon", "coordinates": [[[20.030, 49.993], [20.031, 49.993], [20.031, 49.9925], [20.030, 49.993]]]}
    n_p, n_pl = len(site["piles"]), len(site.get("planned_piles", []))
    add_drawing(site, poly, "Usyp (materiał)", {"status": "lying", "id": "U-09", "material": "wapień",
                                                "tonnage_t": 5000.0, "level": "Poziom dolny"})
    assert len(site["piles"]) == n_p + 1 and "status" not in site["piles"][-1]
    msg = add_drawing(site, poly, "Usyp (materiał)", {"status": "planned", "id": "S-09", "material": "wapień",
                                                      "tonnage_t": 12000.0, "blast_day": 3, "fire_from": "10:00",
                                                      "fire_to": "10:15", "crew_arrive": "08:00", "level": "Poziom dolny"})
    assert msg.startswith("✅") and len(site["planned_piles"]) == n_pl + 1
    pl = site["planned_piles"][-1]
    assert pl["blast_day"] == 3 and pl["fire_from"] == "10:00" and "status" not in pl


def test_properties_of_every_object(site):
    import yaml as _y

    from app_map import object_properties, site_items

    raw = _y.safe_load(DEMO.read_text())
    for sel in site_items(site):
        props = object_properties(site, sel, raw)
        assert props[0][0] == "title" and len(props) >= 3, sel
        assert all(isinstance(v, str) for _, v in props)
    road = next(r for r in site["roads"] if r["kind"] == "rampa")
    p = dict(object_properties(site, f"road:{road['uid']}", raw))
    assert p["rzędna początku (1. punkt)"] != "–" and "m" in p["długość"]
    ex = site["excavators"][0]
    assert dict(object_properties(site, f"excavator:{ex['uid']}", raw))["ładuje z usypu"] == ex["pile"]


def test_levels_follow_the_ramp_inlet(site):
    from kruszyk.site import sync_levels

    R = {r["id"]: r for r in site["roads"]}
    R["R1"]["grade_draw_pct"], R["R1"]["inlet_elev_m"] = 0.0, 300.0
    R["R2"]["grade_draw_pct"], R["R2"]["inlet_elev_m"] = -10.0, 300.0
    site.pop("levels", None)
    sync_levels(site)
    z = sorted((lv["elev_m"] for lv in site["levels"]), reverse=True)
    assert z[0] == 300.0 and z[-1] == pytest.approx(300 - 0.10 * 636, abs=1)  # top road and the ramp's foot
    assert site["crusher"].get("elev_set") is None  # the crusher follows the inlet, not 0 m
