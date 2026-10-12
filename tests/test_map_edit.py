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
    d = site["dedicated"][0]
    assert sel_from_click(site, f"〰️ trasa {d['for']} · dedykowana", None) == f"dedicated:{d['for']}"
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
