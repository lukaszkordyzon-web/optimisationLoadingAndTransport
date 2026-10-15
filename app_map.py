"""Map screen: calibrate the site, draw roads/ramps, piles, excavators and the crusher."""
from __future__ import annotations

import copy

import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import Template
from folium.plugins import Draw
from streamlit_folium import st_folium

from kruszyk import catalog
from kruszyk.config import PRODUCTS, SPOT_MODES
from kruszyk.psd import DEFAULT_PRESET, PRESETS, SIEVES_MM, clean, preset_of, rosin_rammler, summary
from kruszyk.site import (SNAP_M, along_m, level_elev, analyze_site, normalize_roads, centroid, haversine_m, parse_coords,
                          point_along, polygon_area_m2, polyline_length_m, site_to_scenario)

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
# Polish orthophoto and shaded terrain from Geoportal (GUGiK, open data) – the default background
GEOPORTAL_ORTO = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
GEOPORTAL_RELIEF = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WMS/ShadedRelief"
GEOPORTAL_ATTR = "Geoportal.gov.pl (GUGiK)"
WHITE_TILE = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
              "PchI7wAAAABJRU5ErkJggg==")
MATERIALS = ["wapień", "dolomit", "piaskowiec", "granit", "bazalt", "melafir", "żwir", "piasek", "nadkład", "inny"]
ROAD_COLORS = {"rampa": "#d500f9", "droga": "#ffffff"}  # good contrast on light quarry imagery
PLANNED_COLOR = "#9e9e9e"
SKETCH_COLOR = "#ffd600"
MODES = {
    "Droga / rampa": "polyline",
    "Usyp (materiał)": "polygon",
    "Planowany strzał (nowy usyp)": "polygon",
    "Obszar wydobycia (strzały automatycznie)": "polygon",
    "Koparka": "marker",
    "Kruszarka": "marker",
    "Miejsce strzelania": "marker",
    "Miejsce postoju koparki": "marker",
    "Mijanka": "marker",
    "Kierunek urabiania koparki": "polyline",
    "Trasa dedykowana": "polyline",
    "✏️ Edytuj / przesuń / usuń": None,
}
EDIT = "✏️ Edytuj / przesuń / usuń"
# the drawing modes grouped by topic (first choose the group, then what to draw)
MODE_GROUPS = {
    "🛣️ Drogi": ["Droga / rampa", "Mijanka", "Kruszarka"],
    "⛏️ Urobek i koparki": ["Usyp (materiał)", "Koparka", "Kierunek urabiania koparki", "Miejsce postoju koparki"],
    "📅 Plan strzałów": ["Obszar wydobycia (strzały automatycznie)"],
    "✏️ Edycja": [EDIT],
}
DEDICATED_FOR = {"ekipa": "🧨 ekipa strzałowa", "sztygar": "👷 sztygar", "cysterna": "⛽ cysterna"}
DIRECTIONS = {"⬇ w dół": -1, "⬆ pod górę": 1, "➡ płasko": 0}


def _dflt(key: str, fallback):
    """A default from the project's catalogue (raw['defaults'], e.g. the demo project), else `fallback`."""
    return (st.session_state.get("raw") or {}).get("defaults", {}).get(key, fallback)


def _next_from_catalogue(kind: str, used: set) -> dict:
    """The next machine of the project's catalogue that is not on the map yet ({} = none)."""
    return next((m for m in _dflt(kind, []) if m.get("id") not in used), {})


def _preset_index(default_name: str = DEFAULT_PRESET) -> int:
    name = _dflt("psd_preset", default_name)
    return list(PRESETS).index(name) if name in PRESETS else list(PRESETS).index(DEFAULT_PRESET)


def _material_index() -> int:
    m = _dflt("material", MATERIALS[0])
    return MATERIALS.index(m) if m in MATERIALS else 0


def empty_site() -> dict:
    return {"center": [52.0, 19.4], "zoom": 6, "crusher": None, "piles": [], "excavators": [], "roads": []}


def ensure_uids(site: dict) -> None:
    """Every road / pile / excavator gets a stable hidden id, so tables and edits always hit the right object
    (also after deleting or reordering). Duplicate road ids are renamed."""
    import uuid

    seen = set()
    for key in ("roads", "piles", "excavators", "passing_bays", "planned_piles", "crushers", "areas"):
        for o in site.get(key, []):
            if not o.get("uid"):
                o["uid"] = uuid.uuid4().hex[:10]
    bay_ids: set = set()
    for b in site.get("passing_bays", []):
        if not b.get("id") or b["id"] in bay_ids:
            b["id"] = next(f"M{i}" for i in range(1, 10_000)
                           if f"M{i}" not in bay_ids | {x.get("id") for x in site["passing_bays"]})
        bay_ids.add(b["id"])
    for r in site.get("roads", []):
        if not r.get("id") or r["id"] in seen:
            r["id"] = next(f"R{i}" for i in range(1, 10_000) if f"R{i}" not in seen | {x.get("id") for x in site["roads"]})
        seen.add(r["id"])


DEFAULT_FLAGS = {"active": True, "single_lane": False, "t_per_m": 50.0, "drilled": False, "first_drilled": True}


def _plain(site: dict) -> dict:
    """Site without the hidden helper ids – for "are there unsaved changes?"."""
    out = {}
    for k, v in site.items():
        if isinstance(v, list):
            out[k] = [{kk: vv for kk, vv in o.items() if kk != "uid" and vv is not None and not (kk in DEFAULT_FLAGS and vv == DEFAULT_FLAGS[kk])}
                      if isinstance(o, dict) else o for o in v]
        else:
            out[k] = v
    return out


def get_site(raw: dict) -> dict:
    if "site" not in st.session_state or st.session_state.get("site_ver") != st.session_state.ver:
        old = st.session_state.get("site")
        new = copy.deepcopy(raw.get("site") or empty_site())
        new.pop("dedicated", None)  # everyone drives on the drawn roads
        new.pop("blast", None)      # the blast place comes from the planned blast (pile) itself
        normalize_roads(new)
        ensure_uids(new)
        st.session_state.site = new
        st.session_state.site_ver = st.session_state.ver
        # rebuild the map only when the view really changes (another scenario), not after "build routes"
        if old is None or (old.get("center"), old.get("zoom")) != (new.get("center"), new.get("zoom")):
            st.session_state.map_key = st.session_state.get("map_key", 0) + 1
        st.session_state.obj_ver = st.session_state.get("obj_ver", 0) + 1
    return st.session_state.site


def base_map(site: dict, height_zoom: int | None = None) -> folium.Map:
    m = folium.Map(location=site["center"], zoom_start=height_zoom or site.get("zoom", 16), tiles=None,
                   max_zoom=21, control_scale=True)
    folium.WmsTileLayer(GEOPORTAL_ORTO, layers="Raster", fmt="image/jpeg", name="Ortofoto (Geoportal)",
                        attr=GEOPORTAL_ATTR, max_zoom=21, overlay=False).add_to(m)
    folium.WmsTileLayer(GEOPORTAL_RELIEF, layers="Raster", fmt="image/png", name="Relief terenu (Geoportal)",
                        attr=GEOPORTAL_ATTR, max_zoom=21, overlay=False).add_to(m)
    folium.TileLayer(WHITE_TILE, attr=" ", name="Plan (bez podkładu)", max_zoom=22).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Mapa (OSM)", max_zoom=21, max_native_zoom=19).add_to(m)
    folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satelita Esri (zapas)", max_zoom=21, max_native_zoom=19).add_to(m)
    return m


def draw_network(m, info: dict) -> None:
    """Junctions (white) and dead ends (red) of the road network, with computed elevations."""
    g = info.get("graph")
    if g is None:
        return
    for n in info.get("junctions", []):
        folium.CircleMarker(g.nodes[n], radius=6, color="#000", weight=1, fill=True, fill_color="#fff", fill_opacity=1,
                            tooltip=f"węzeł · rzędna {info['elev'].get(n, float('nan')):.1f} m").add_to(m)
    for n in info.get("dead_ends", []):
        folium.CircleMarker(g.nodes[n], radius=6, color="#000", weight=1, fill=True, fill_color="#e63946",
                            fill_opacity=1, tooltip="ślepy koniec – niepołączona droga? (koniec ≤ 30 m od innej "
                                                    "drogi tworzy węzeł)").add_to(m)


def _direction_arrow(m, pts: list, grade: float, tip: str) -> None:
    """Arrow in the middle of a road showing the drawing direction; ↗ green = climbs that way, ↘ orange = descends."""
    import math

    if len(pts) < 2:
        return
    i = max(0, len(pts) // 2 - 1)
    a, b = pts[i], pts[i + 1]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ang = math.degrees(math.atan2((b[0] - a[0]), (b[1] - a[1]) * math.cos(math.radians(a[0]))))
    color = "#2a9d8f" if grade > 0.05 else ("#f77f00" if grade < -0.05 else "#555")
    label = f"{grade:+g}%" if abs(grade) >= 0.05 else ""
    folium.Marker(mid, tooltip=tip, icon=folium.DivIcon(
        icon_size=(60, 24), icon_anchor=(12, 12),
        html=f"<div style='display:flex;align-items:center;gap:2px;white-space:nowrap'>"
             f"<span style='display:inline-block;transform:rotate({-ang:.0f}deg);font-size:20px;color:{color};"
             f"text-shadow:0 0 3px #fff,0 0 3px #fff'>➤</span>"
             f"<span style='font-size:11px;font-weight:700;color:#fff;text-shadow:0 0 3px #000'>{label}</span></div>")
    ).add_to(m)


def draw_site(m, site: dict, plans: bool = True) -> None:
    """Add site objects to a map or a FeatureGroup."""
    for r in site.get("roads", []):
        pts = [tuple(c) for c in r["coords"]]
        active = r.get("active", True)
        g = float(r.get("grade_draw_pct", 0.0))
        slope = "płasko" if abs(g) < 0.05 else (f"w górę {g:g} %" if g > 0 else f"w dół {-g:g} %")
        tip = (f"🛣️ {r.get('id')} · {r['name']} · {r.get('kind')} · wzdłuż strzałki: {slope} · limit {r.get('speed_limit_kmh', '?')} km/h · "
               f"{polyline_length_m(r['coords']):.0f} m" + (" · JEDEN PAS – mijanki" if r.get("single_lane") else "")
               + ("" if active else " · PLANOWANA (dziś nieużywana)"))
        folium.PolyLine(pts, color="#000000", weight=10, opacity=0.55, tooltip=tip).add_to(m)  # dark casing
        folium.PolyLine(pts, color=ROAD_COLORS.get(r.get("kind"), "#ffffff") if active else PLANNED_COLOR, weight=6,
                        dash_array=("14 6" if r.get("single_lane") else None) if active else "10 8", tooltip=tip).add_to(m)
        _direction_arrow(m, pts, g, tip)
    for d in site.get("dedicated", []):
        colors = {"ekipa": "#ff006e", "sztygar": "#3a86ff", "cysterna": "#8338ec"}
        folium.PolyLine([tuple(c) for c in d["coords"]], color=colors.get(d.get("for"), "#ffbe0b"), weight=4,
                        dash_array="6 6", tooltip=f"〰️ trasa {d.get('for')} · dedykowana: {DEDICATED_FOR.get(d.get('for'), 'koparka ' + str(d.get('for')))}"
                        ).add_to(m)
    for b in site.get("passing_bays", []):
        folium.Marker((b["lat"], b["lon"]), tooltip=f"⇆ {b.get('id')} · mijanka",
                      icon=folium.DivIcon(html=f"<div style='font-size:12px;font-weight:700;color:#fff;background:#ff4d6d;"
                                               f"border:2px solid #000;border-radius:4px;padding:0 3px;white-space:nowrap'>"
                                               f"⇆ {b.get('id')}</div>")).add_to(m)
    for p in site.get("parking", []):
        folium.Marker((p["lat"], p["lon"]), tooltip=f"🅿️ postój {p.get('excavator')} · miejsce postoju koparki",
                      icon=folium.DivIcon(html=f"<div style='font-size:13px;font-weight:700;color:#fff;"
                                               f"text-shadow:0 0 3px #000;white-space:nowrap'>🅿️ {p.get('excavator')}</div>")
                      ).add_to(m)
    for p in site.get("piles", []):
        folium.Polygon([tuple(c) for c in p["polygon"]], color="#00e5ff", fill=True, fill_opacity=0.25,
                       tooltip=f"⛰️ {p['id']} · usyp: {p.get('material', '')} · {p.get('tonnage_t', 0):,.0f} t").add_to(m)
        folium.Marker(centroid(p["polygon"]), tooltip=f"⛰️ {p['id']} · usyp: {p.get('material', '')}", icon=folium.DivIcon(
            html=f"<div style='color:#fff;font-weight:700;text-shadow:0 0 3px #000;white-space:nowrap'>"
                 f"{p['id']} · {p.get('material', '')}</div>")).add_to(m)
    for a in site.get("areas", []) if plans else []:
        tip = (f"🟧 {a['id']} · obszar wydobycia · {a.get('material', '')} · zasoby {a.get('reserves_t', 0):,.0f} t · "
               f"strzał ≤ {a.get('blast_max_t', 0):,.0f} t".replace(",", " "))
        folium.Polygon([tuple(c) for c in a["polygon"]], color="#ff9f1c", weight=3, dash_array="4 6", fill=True,
                       fill_opacity=0.08, tooltip=tip).add_to(m)
        folium.Marker(centroid(a["polygon"]), tooltip=tip, icon=folium.DivIcon(
            html=f"<div style='color:#ff9f1c;font-weight:800;text-shadow:0 0 3px #000;white-space:nowrap'>"
                 f"🟧 {a['id']}</div>")).add_to(m)
    for p in site.get("planned_piles", []) if plans else []:
        tip = (f"📅 {p['id']} · planowany strzał w dniu roboczym {p.get('blast_day', 1)} · {p.get('material', '')} · "
               f"{p.get('tonnage_t', 0):,.0f} t".replace(",", " "))
        folium.Polygon([tuple(c) for c in p["polygon"]], color="#ff006e", weight=3, dash_array="8 6", fill=True,
                       fill_opacity=0.12, tooltip=tip).add_to(m)
        folium.Marker(centroid(p["polygon"]), tooltip=tip, icon=folium.DivIcon(
            html=f"<div style='color:#fff;font-weight:700;text-shadow:0 0 3px #000;white-space:nowrap'>"
                 f"🧨 {p['id']} · dzień {p.get('blast_day', 1)}</div>")).add_to(m)
    for e in site.get("excavators", []):
        dp = e.get("dig_path") or []
        if len(dp) >= 2:
            tip = (f"⤳ urabianie {e['id']} · kierunek urabiania: start przy koparce → koniec (strzałka); "
                   f"przestawienie co {e.get('relocate_every_t', 600):g} t, {e.get('relocate_min', 4):g} min")
            folium.PolyLine([tuple(c) for c in dp], color="#ff9f1c", weight=5, dash_array="2 8", tooltip=tip).add_to(m)
            _direction_arrow(m, [tuple(c) for c in dp[-2:]], 0.0, tip)
    for e in site.get("excavators", []):
        folium.Marker((e["lat"], e["lon"]), tooltip=f"⛏️ {e['id']} · koparka → {e.get('pile', '?')}",
                      icon=folium.Icon(color="orange", icon="wrench", prefix="fa")).add_to(m)
    if site.get("crusher"):
        c = site["crusher"]
        acc = ", ".join(c.get("accepts") or []) or "wszystkich usypów"
        folium.Marker((c["lat"], c["lon"]), tooltip=f"🏭 {c.get('id', 'C-1')} · kruszarka główna · przyjmuje z: {acc}",
                      icon=folium.Icon(color="red", icon="industry", prefix="fa")).add_to(m)
    for c in site.get("crushers", []):
        acc = ", ".join(c.get("accepts") or []) or "wszystkich usypów"
        folium.Marker((c["lat"], c["lon"]), tooltip=f"🏭 {c['id']} · kruszarka · {c.get('capacity_tph', 0):g} t/h · "
                                                    f"przyjmuje z: {acc}",
                      icon=folium.Icon(color="darkred", icon="industry", prefix="fa")).add_to(m)
    if site.get("blast"):
        b = site["blast"]
        folium.Marker((b["lat"], b["lon"]), tooltip="🧨 Miejsce strzelania",
                      icon=folium.DivIcon(html="<div style='font-size:22px'>🧨</div>")).add_to(m)


def _latlon(coords: list) -> list:
    return [[round(c[1], 7), round(c[0], 7)] for c in coords]  # GeoJSON is [lon, lat]


DEFAULTS = {
    "Droga / rampa": {"kind": "droga", "grade_draw_pct": 0.0, "speed_limit_kmh": 40.0, "single_lane": False},
    "Usyp (materiał)": {"material": "inny", "tonnage_t": 10_000.0, "psd": PRESETS[DEFAULT_PRESET],
                        "loose_density_t_m3": 1.75},
    "Koparka": {"pile": "(najbliższy)", "bucket_m3": 5.0, "swing_cycle_s": 30.0},
}
GEOM_MODE = {"LineString": "Droga / rampa", "Polygon": "Usyp (materiał)"}


def add_drawing(site: dict, g: dict, mode: str, attrs: dict) -> str:
    """Add a freshly drawn geometry to the site. The shape decides the object type; form parameters
    are used when they belong to that type, otherwise defaults (editable later in the tables)."""
    target = GEOM_MODE.get(g["type"])
    if g["type"] == "Point":
        target = mode if mode in ("Kruszarka", "Miejsce strzelania", "Miejsce postoju koparki", "Mijanka") else "Koparka"
    if g["type"] == "LineString" and mode in ("Trasa dedykowana", "Kierunek urabiania koparki"):
        target = mode
    if g["type"] == "Polygon" and mode == "Obszar wydobycia (strzały automatycznie)":
        site.setdefault("areas", [])
        ids = {a_["id"] for a_ in site["areas"]}
        a = dict(attrs)
        if not a.get("id") or a["id"] in ids:
            a["id"] = next(f"O{i}" for i in range(1, 1000) if f"O{i}" not in ids)
        poly = _latlon(g["coordinates"][0][:-1])
        if a.get("level_m") is None:
            a["level_m"] = level_at(site, centroid(poly))
        if not a.get("reserves_t"):
            a["reserves_t"] = round(polygon_area_m2(poly) * float(a.get("bench_m", 15)) * float(a.get("solid_density_t_m3", 2.6)), -2)
        site["areas"].append({**a, "polygon": poly})
        ensure_uids(site)
        n = -(-a["reserves_t"] // a["blast_max_t"])
        return (f"✅ Obszar {a['id']}: {a['reserves_t']:,.0f} t ≈ {n:.0f} strzałów po ≤ {a['blast_max_t']:,.0f} t – "
                "terminy ustali harmonogram").replace(",", " ")
    if g["type"] == "Polygon" and (mode == "Planowany strzał (nowy usyp)"
                                   or (mode == "Usyp (materiał)" and attrs.get("status") == "planned")):
        attrs = {k: v for k, v in attrs.items() if k != "status"}
        site.setdefault("planned_piles", [])
        ids = {p["id"] for p in site["piles"]} | {p["id"] for p in site["planned_piles"]}
        a = dict(attrs)
        if not a.get("id") or a["id"] in ids:
            a["id"] = next(f"S-{i:02d}" for i in range(1, 1000) if f"S-{i:02d}" not in ids)
        site["planned_piles"].append({**a, "polygon": _latlon(g["coordinates"][0][:-1])})
        ensure_uids(site)
        return (f"✅ Planowany strzał {a['id']} w dniu roboczym {a.get('blast_day', 1)} – w module „Planowanie produkcji” "
                f"koparka bez urobku sama tu przejedzie")
    if target == "Kierunek urabiania koparki":
        ex = next((e for e in site["excavators"] if e["id"] == attrs.get("excavator")), None)
        if ex is None:
            return "⚠️ Najpierw dodaj koparkę, potem narysuj jej kierunek urabiania"
        set_dig_path(ex, _latlon(g["coordinates"]), attrs)
        return (f"✅ Kierunek urabiania koparki {ex['id']}: {polyline_length_m(ex['dig_path']):.0f} m – koparka "
                f"startuje na początku linii i co {ex['relocate_every_t']:g} t przesuwa się dalej")
    if target == "Trasa dedykowana":
        who = attrs.get("for", "sztygar")
        site.setdefault("dedicated", [])
        site["dedicated"] = [d for d in site["dedicated"] if d.get("for") != who]  # one route per role
        site["dedicated"].append({"for": who, "coords": _latlon(g["coordinates"])})
        return f"✅ Dodano trasę dedykowaną: {DEDICATED_FOR.get(who, 'przejazd koparki ' + who)}"
    if target == "Miejsce postoju koparki":
        lon, lat = g["coordinates"]
        who = attrs.get("excavator")
        if not who:
            return "⚠️ Najpierw dodaj koparkę, potem jej miejsce postoju"
        site.setdefault("parking", [])
        site["parking"] = [p for p in site["parking"] if p.get("excavator") != who]
        site["parking"].append({"excavator": who, "lat": lat, "lon": lon})
        return f"✅ Ustawiono miejsce postoju koparki {who} (strzelanie / tankowanie)"
    if target == "Mijanka":
        return add_bay(site, g["coordinates"][1], g["coordinates"][0])
    if target is None:
        return f"⚠️ Nieobsługiwany kształt: {g['type']}"
    a = {k: v for k, v in attrs.items() if k != "status"} if target == mode else dict(DEFAULTS.get(target, {}))
    note = "" if target == mode else " (parametry domyślne – popraw w tabeli poniżej)"
    if target == "Droga / rampa":
        a.setdefault("name", f"{a['kind']} {len(site['roads']) + 1}")
        rid = next(f"R{i}" for i in range(1, 10_000) if f"R{i}" not in {r.get("id") for r in site["roads"]})
        site["roads"].append({"id": rid, **a, "coords": _latlon(g["coordinates"])})
        ensure_uids(site)
        return f"✅ Dodano drogę „{a['name']}” ({polyline_length_m(site['roads'][-1]['coords']):.0f} m){note}"
    if target == "Usyp (materiał)":
        ids = {p["id"] for p in site["piles"]}
        if a.get("id") in ids or not a.get("id"):
            a["id"] = next(f"U-{i:02d}" for i in range(1, 1000) if f"U-{i:02d}" not in ids)
        site["piles"].append({**a, "polygon": _latlon(g["coordinates"][0][:-1])})
        ensure_uids(site)
        return f"✅ Dodano usyp {a['id']}{note}"
    lon, lat = g["coordinates"]
    if target == "Kruszarka":
        if attrs.get("new_crusher") and site.get("crusher"):
            site.setdefault("crushers", [])
            ids = {(site["crusher"] or {}).get("id", "C-1")} | {c["id"] for c in site["crushers"]}
            cid = attrs.get("id") if attrs.get("id") and attrs["id"] not in ids else \
                next(f"C-{i}" for i in range(2, 100) if f"C-{i}" not in ids)
            site["crushers"].append({"id": cid, "lat": lat, "lon": lon, "capacity_tph": attrs.get("capacity_tph", 800.0),
                                     "hopper_t": attrs.get("hopper_t", 150.0), "accepts": attrs.get("accepts", []),
                                     "product": attrs.get("product", ""), "target_t": attrs.get("target_t", 0.0),
                                     "kwh_per_t": attrs.get("kwh_per_t"),
                                     **{k: attrs[k] for k in ("model", "max_feed_mm", "dump_time_s") if attrs.get(k)}})
            ensure_uids(site)
            return f"✅ Dodano kruszarkę {cid} – zbuduj trasy (zapisz kopalnię), żeby wozidła do niej jeździły"
        old = site.get("crusher") or {}
        site["crusher"] = {**old, "lat": lat, "lon": lon, **({"level": attrs["level"]} if attrs.get("level") else {}),
                           **{k: attrs[k] for k in ("id", "product") if attrs.get(k) and not old.get(k)}}
        return "✅ Ustawiono kruszarkę główną"
    if target == "Miejsce strzelania":
        site["blast"] = {"lat": lat, "lon": lon}
        return "✅ Ustawiono miejsce strzelania – godziny ustaw w sekcji „Zmiana i zdarzenia”"
    ids = {e["id"] for e in site["excavators"]}
    if a.get("id") in ids or not a.get("id"):
        a["id"] = next(f"K-{i}" for i in range(1, 1000) if f"K-{i}" not in ids)
    site["excavators"].append({**a, "lat": lat, "lon": lon})
    ensure_uids(site)
    return f"✅ Dodano koparkę {a['id']}{note}"


# ---------- editing existing objects (select on the map → move / redraw / delete) ----------
POINT_KINDS = ("excavator", "crusher", "crusher2", "blast", "parking", "bay")


def site_items(site: dict) -> dict:
    """All editable objects: selection key → label. Keys are "kind:ident" (uid for roads/piles/excavators)."""
    items = {}
    for r in site.get("roads", []):
        items[f"road:{r['uid']}"] = f"🛣️ {r.get('id')} · {r.get('name', '')}" + ("" if r.get("active", True) else " (planowana)")
    for p in site.get("piles", []):
        items[f"pile:{p['uid']}"] = f"⛰️ usyp {p['id']} · {p.get('material', '')}"
    for a in site.get("areas", []):
        items[f"area:{a['uid']}"] = f"🟧 obszar wydobycia {a['id']}"
    for p in site.get("planned_piles", []):
        items[f"planned:{p['uid']}"] = f"📅 planowany strzał {p['id']} (dzień {p.get('blast_day', 1)})"
    for e in site.get("excavators", []):
        items[f"excavator:{e['uid']}"] = f"⛏️ koparka {e['id']}"
    if site.get("crusher"):
        items["crusher:"] = f"🏭 kruszarka główna {site['crusher'].get('id', 'C-1')}"
    for c in site.get("crushers", []):
        items[f"crusher2:{c['uid']}"] = f"🏭 kruszarka {c['id']}"
    if site.get("blast"):
        items["blast:"] = "🧨 miejsce strzelania"
    for e in site.get("excavators", []):
        if len(e.get("dig_path") or []) >= 2:
            items[f"dig:{e['uid']}"] = f"⤳ kierunek urabiania {e['id']}"
    for b in site.get("passing_bays", []):
        items[f"bay:{b['uid']}"] = f"⇆ mijanka {b.get('id')}"
    for p in site.get("parking", []):
        items[f"parking:{p['excavator']}"] = f"🅿️ miejsce postoju koparki {p['excavator']}"
    for d in site.get("dedicated", []):
        items[f"dedicated:{d['for']}"] = f"〰️ trasa dedykowana: {DEDICATED_FOR.get(d['for'], 'przejazd koparki ' + str(d['for']))}"
    return items


def find_obj(site: dict, sel: str | None):
    if not sel:
        return None
    kind, ident = sel.split(":", 1)
    if kind in ("road", "pile", "excavator"):
        return next((o for o in site.get(kind + "s", []) if o.get("uid") == ident), None)
    if kind == "bay":
        return next((o for o in site.get("passing_bays", []) if o.get("uid") == ident), None)
    if kind == "planned":
        return next((o for o in site.get("planned_piles", []) if o.get("uid") == ident), None)
    if kind == "crusher2":
        return next((o for o in site.get("crushers", []) if o.get("uid") == ident), None)
    if kind == "area":
        return next((o for o in site.get("areas", []) if o.get("uid") == ident), None)
    if kind == "dig":
        e = next((o for o in site.get("excavators", []) if o.get("uid") == ident), None)
        return {"coords": e["dig_path"], "_ex": e} if e and len(e.get("dig_path") or []) >= 2 else None
    if kind in ("crusher", "blast"):
        return site.get(kind)
    if kind == "parking":
        return next((p for p in site.get("parking", []) if p.get("excavator") == ident), None)
    if kind == "dedicated":
        return next((d for d in site.get("dedicated", []) if str(d.get("for")) == ident), None)
    return None


def sel_from_click(site: dict, tooltip: str | None, latlng: dict | None) -> str | None:
    """Clicked object → selection key: by the tooltip prefix, otherwise the nearest object (≤ 40 m)."""
    head = (tooltip or "").strip().split(" · ")[0].strip()
    by_id = {"🛣️": ("road", "roads"), "⛰️": ("pile", "piles"), "⛏️": ("excavator", "excavators"),
             "⇆": ("bay", "passing_bays"), "📅": ("planned", "planned_piles"), "🟧": ("area", "areas")}
    for icon, (kind, key) in by_id.items():
        if head.startswith(icon):
            oid = head[len(icon):].strip()
            o = next((o for o in site.get(key, []) if str(o.get("id")) == oid), None)
            if o:
                return f"{kind}:{o['uid']}"
    if head.startswith("⤳"):
        who = head.split("urabianie", 1)[-1].strip()
        e = next((o for o in site.get("excavators", []) if o["id"] == who), None)
        if e:
            return f"dig:{e['uid']}"
    if head.startswith("🏭"):
        cid = head[len("🏭"):].strip()
        extra = next((c for c in site.get("crushers", []) if c["id"] == cid), None)
        if extra:
            return f"crusher2:{extra['uid']}"
        if site.get("crusher"):
            return "crusher:"
    if head.startswith("🧨") and site.get("blast"):
        return "blast:"
    if head.startswith("🅿️"):
        who = head.split("postój", 1)[-1].strip()
        if any(p.get("excavator") == who for p in site.get("parking", [])):
            return f"parking:{who}"
    if head.startswith("〰️"):
        who = head.split("trasa", 1)[-1].strip()
        if any(str(d.get("for")) == who for d in site.get("dedicated", [])):
            return f"dedicated:{who}"
    if not latlng or latlng.get("lat") is None:
        return None
    q = (latlng["lat"], latlng["lng"])
    cands = []
    for sel in site_items(site):
        o = find_obj(site, sel)
        kind = sel.split(":", 1)[0]
        if kind in POINT_KINDS:
            pts = [(o["lat"], o["lon"])]
        elif kind in ("pile", "planned", "area"):
            pts = [tuple(c) for c in o["polygon"]] + [centroid(o["polygon"])]
        else:
            pts = [tuple(c) for c in o["coords"]]
        cands.append((min(haversine_m(q, p) for p in pts), sel))
    best = min(cands, default=None)
    return best[1] if best and best[0] <= 40 else None


def rename_refs(site: dict, kind: str, old, new) -> None:
    """Keep references consistent after renaming a pile or an excavator."""
    if old == new:
        return
    if kind == "piles":
        for e in site.get("excavators", []):
            if e.get("pile") == old:
                e["pile"] = new
    if kind == "excavators":
        for p in site.get("parking", []):
            if p.get("excavator") == old:
                p["excavator"] = new
        for d in site.get("dedicated", []):
            if d.get("for") == old:
                d["for"] = new


def delete_obj(site: dict, sel: str) -> str:
    kind, ident = sel.split(":", 1)
    o = find_obj(site, sel)
    if o is None:
        return "⚠️ Nie ma już takiego obiektu"
    if kind == "road":
        site["roads"] = [r for r in site["roads"] if r is not o]
        return f"🗑️ Usunięto drogę {o.get('id')}"
    if kind == "pile":
        site["piles"] = [p for p in site["piles"] if p is not o]
        for e in site.get("excavators", []):
            if e.get("pile") == o["id"]:
                e["pile"] = "(najbliższy)"
        return f"🗑️ Usunięto usyp {o['id']} (jego koparki ładują teraz z najbliższego)"
    if kind == "excavator":
        site["excavators"] = [e for e in site["excavators"] if e is not o]
        site["parking"] = [p for p in site.get("parking", []) if p.get("excavator") != o["id"]]
        site["dedicated"] = [d for d in site.get("dedicated", []) if d.get("for") != o["id"]]
        return f"🗑️ Usunięto koparkę {o['id']} (razem z jej miejscem postoju i trasą przejazdu)"
    if kind in ("crusher", "blast"):
        site[kind] = None
        return "🗑️ Usunięto " + ("kruszarkę" if kind == "crusher" else "miejsce strzelania")
    if kind == "bay":
        site["passing_bays"] = [b for b in site["passing_bays"] if b is not o]
        return f"🗑️ Usunięto mijankę {o.get('id')}"
    if kind == "area":
        site["areas"] = [a for a in site["areas"] if a is not o]
        return f"🗑️ Usunięto obszar wydobycia {o['id']}"
    if kind == "crusher2":
        site["crushers"] = [c for c in site["crushers"] if c is not o]
        return f"🗑️ Usunięto kruszarkę {o['id']}"
    if kind == "planned":
        site["planned_piles"] = [p for p in site["planned_piles"] if p is not o]
        return f"🗑️ Usunięto planowany strzał {o.get('id')}"
    if kind == "dig":
        o["_ex"].pop("dig_path", None)
        return f"🗑️ Usunięto kierunek urabiania koparki {o['_ex']['id']} (koparka stoi w miejscu)"
    if kind == "parking":
        site["parking"] = [p for p in site["parking"] if p is not o]
        return f"🗑️ Usunięto miejsce postoju koparki {ident}"
    site["dedicated"] = [d for d in site["dedicated"] if d is not o]
    return "🗑️ Usunięto trasę dedykowaną"


def apply_edit_drawing(site: dict, sel: str, g: dict) -> str:
    """The next drawing after „Przesuń” / „Przerysuj” becomes the new position / geometry of the object."""
    kind = sel.split(":", 1)[0]
    o = find_obj(site, sel)
    if o is None:
        return "⚠️ Nie ma już takiego obiektu"
    if kind in POINT_KINDS:
        if g["type"] != "Point":
            return "⚠️ Aby przesunąć obiekt, postaw znacznik (📍) w nowym miejscu"
        if kind == "bay":  # a passing bay stays on its road
            site["passing_bays"] = [b for b in site["passing_bays"] if b is not o]
            msg = add_bay(site, g["coordinates"][1], g["coordinates"][0])
            if not msg.startswith("✅"):
                site["passing_bays"].append(o)
                return msg
            site["passing_bays"][-1].update({"id": o.get("id"), "uid": o.get("uid")})
            return f"✅ Przesunięto mijankę {o.get('id')}"
        lon, lat = g["coordinates"]
        if kind == "excavator" and len(o.get("dig_path") or []) >= 2:  # the digging line moves with it
            dlat, dlon = lat - o["lat"], lon - o["lon"]
            o["dig_path"] = [[round(a + dlat, 7), round(b + dlon, 7)] for a, b in o["dig_path"]]
        o["lon"], o["lat"] = lon, lat
        return "✅ Przesunięto obiekt"
    if kind == "dig":
        if g["type"] != "LineString":
            return "⚠️ Aby przerysować kierunek urabiania, narysuj linię (/)"
        set_dig_path(o["_ex"], _latlon(g["coordinates"]))
        return f"✅ Przerysowano kierunek urabiania koparki {o['_ex']['id']}"
    if kind in ("road", "dedicated"):
        if g["type"] != "LineString":
            return "⚠️ Aby przerysować drogę / trasę, narysuj linię (/)"
        o["coords"] = _latlon(g["coordinates"])
        o.pop("z", None)  # the old point elevations belong to the old line
        return f"✅ Przerysowano ({polyline_length_m(o['coords']):.0f} m)"
    if g["type"] != "Polygon":
        return "⚠️ Aby przerysować usyp, narysuj wielokąt (⬠)"
    o["polygon"] = _latlon(g["coordinates"][0][:-1])
    return f"✅ Przerysowano usyp {o['id']}"


def _fmt(v, unit: str = "", nd: int = 0) -> str:
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return "–"
    if isinstance(v, bool):
        return "tak" if v else "nie"
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}".replace(",", " ") + (f" {unit}" if unit else "")
    return str(v)


def _node_elev(info: dict, p) -> float | None:
    g, elev = info.get("graph"), info.get("elev") or {}
    if g is None or not elev:
        return None
    n = min(elev, key=lambda i: haversine_m(g.nodes[i], tuple(p)))
    return float(elev[n]) if haversine_m(g.nodes[n], tuple(p)) < 2 * SNAP_M else None


def object_properties(site: dict, sel: str, raw: dict) -> list[tuple[str, str]]:
    """Everything about one object – what was entered and what follows from it (heights, lengths, links).
    First item = the title."""
    o = find_obj(site, sel)
    kind = sel.split(":", 1)[0]
    info = analyze_site(site)
    lv = {x["name"]: x for x in site.get("levels") or []}
    P: list[tuple[str, str]] = []
    if kind == "road":
        L = polyline_length_m(o["coords"])
        z0, z1 = _node_elev(info, o["coords"][0]), _node_elev(info, o["coords"][-1])
        auto = next((x["name"] for x in site.get("levels") or [] if (o.get("name") or o.get("id")) in x.get("roads", [])), None)
        bays = [b["id"] for b in site.get("passing_bays", []) if along_m(o["coords"], (b["lat"], b["lon"]))[1] <= SNAP_M]
        P = [("title", f"🛣️ {o.get('kind', 'droga')} {o.get('id')} – {o.get('name', '')}"),
             ("rodzaj", o.get("kind")), ("długość", _fmt(L, "m")),
             ("nachylenie wzdłuż rysowania", _fmt(o.get("grade_draw_pct"), "%", 1)),
             ("rzędna początku (1. punkt)", _fmt(z0, "m", 1)), ("rzędna końca", _fmt(z1, "m", 1)),
             ("różnica wysokości", _fmt(None if z0 is None or z1 is None else z1 - z0, "m", 1)),
             ("rzędna wlotu (wpisana)", _fmt(o.get("inlet_elev_m"), "m", 1)), ("poziom (auto)", auto or "–"),
             ("limit prędkości", _fmt(o.get("speed_limit_kmh"), "km/h")), ("jeden pas", _fmt(bool(o.get("single_lane")))),
             ("mijanki", ", ".join(bays) or "–"), ("w użyciu", _fmt(o.get("active", True))),
             ("czas przejazdu przy limicie", _fmt(L / max(float(o.get("speed_limit_kmh") or 30), 1) * 3.6 / 60, "min", 1)),
             ("łuki (promień → prędkość w łuku)", _bends_text(o["coords"]))]
    elif kind in ("pile", "planned"):
        from kruszyk.psd import summary as psd_summary

        max_feed = float(raw.get("crusher", {}).get("max_feed_mm", 800))
        sm = psd_summary(o["psd"], max_feed) if o.get("psd") else {}
        exs = [e["id"] for e in site.get("excavators", []) if e.get("pile") == o["id"]]
        P = [("title", f"{'🧨 planowany odstrzał' if kind == 'planned' else '⛰️ usyp'} {o['id']}"),
             ("materiał", o.get("material")), ("ilość", _fmt(o.get("tonnage_t"), "t")),
             ("poziom", (o.get("level") or "–") + (" (auto)" if o.get("level_auto") else "")),
             ("rzędna", _fmt(o.get("level_m"), "m", 1)), ("powierzchnia", _fmt(polygon_area_m2(o["polygon"]), "m²")),
             ("gęstość spulchniona", _fmt(o.get("loose_density_t_m3"), "t/m³", 2)),
             ("uziarnienie", preset_of(o["psd"]) if o.get("psd") else "–"),
             ("P80", _fmt(sm.get("P80 [mm]"), "mm")), ("nadziarno", _fmt(sm.get("nadziarno [%]"), "%", 1)),
             ("udział w nadawie", _fmt(o.get("target_share"), "%")), ("koparki", ", ".join(exs) or "–"),
             ("kruszarki przyjmujące", ", ".join(c.get("id", "C-1") for c in [site.get("crusher") or {}] + site.get("crushers", [])
                                                   if c and (not c.get("accepts") or o["id"] in c["accepts"])) or "–")]
        if kind == "planned":
            m = float(o.get("tonnage_t") or 0) / float(o.get("t_per_m") or 50)
            P += [("dzień strzału (roboczy)", _fmt(o.get("blast_day"))), ("przyjazd ekipy", o.get("crew_arrive") or "–"),
                  ("odpalenie", f"{o.get('fire_from', '–')}–{o.get('fire_to', '–')}"),
                  ("ewakuacja przed", _fmt(o.get("evacuation_min"), "min")), ("wyjazd ekipy", o.get("crew_leave") or "–"),
                  ("t/mb", _fmt(o.get("t_per_m"), "t/mb")), ("metry do wywiercenia", _fmt(m, "m")),
                  ("nawiercony", _fmt(bool(o.get("drilled"))))]
    elif kind == "excavator":
        lvx = (info.get("levels") or {}).get(o["id"], {})
        park = next((p for p in site.get("parking", []) if p.get("excavator") == o["id"]), None)
        spot = ", ".join(f"{k} {v:g}%" for k, v in (o.get("spotting") or {}).items()) or "stały czas"
        P = [("title", f"⛏️ koparka {o['id']}"), ("ładuje z usypu", o.get("pile")),
             ("łyżka", _fmt(o.get("bucket_m3"), "m³", 1)), ("cykl łyżki", _fmt(o.get("swing_cycle_s"), "s")),
             ("rzędna (z dróg)", _fmt(lvx.get("elev_m"), "m", 1)), ("odległość od drogi", _fmt(lvx.get("dist_to_road_m"), "m")),
             ("sposób podstawiania", spot),
             ("kierunek urabiania", _fmt(polyline_length_m(o["dig_path"]), "m") if o.get("dig_path") else "–"),
             ("przestawienie", f"co {o.get('relocate_every_t', 600):g} t, {o.get('relocate_min', 4):g} min" if o.get("dig_path") else "–"),
             ("miejsce postoju", "tak" if park else "– (cofa się po swojej trasie)"),
             ("położenie", f"{o['lat']:.5f}, {o['lon']:.5f}")]
    elif kind in ("crusher", "crusher2"):
        main = kind == "crusher"
        cr = raw.get("crusher", {}) if main else o
        P = [("title", f"🏭 kruszarka {o.get('id', 'C-1')}{' (główna)' if main else ''}"),
             ("wydajność", _fmt(cr.get("capacity_tph"), "t/h")), ("kosz", _fmt(cr.get("hopper_t"), "t")),
             ("produkt", o.get("product") or "–"), ("energia", _fmt(cr.get("kwh_per_t", raw.get("crusher", {}).get("kwh_per_t")), "kWh/t", 2)),
             ("plan na zmianę", _fmt(o.get("target_t"), "t")), ("przyjmuje z", ", ".join(o.get("accepts") or []) or "wszystkich usypów"),
             ("rzędna", _fmt(site["crusher"].get("elev_m") if main else _node_elev(info, (o["lat"], o["lon"])), "m", 1)),
             ("położenie", f"{o['lat']:.5f}, {o['lon']:.5f}")]
    elif kind == "area":
        n = -(-float(o.get("reserves_t") or 0) // max(float(o.get("blast_max_t") or 1), 1))
        P = [("title", f"🟧 obszar wydobycia {o['id']}"), ("materiał", o.get("material")),
             ("poziom", (o.get("level") or "–") + (" (auto)" if o.get("level_auto") else "")),
             ("rzędna", _fmt(o.get("level_m"), "m", 1)), ("zasoby", _fmt(o.get("reserves_t"), "t")),
             ("powierzchnia", _fmt(polygon_area_m2(o["polygon"]), "m²")), ("największy strzał", _fmt(o.get("blast_max_t"), "t")),
             ("liczba strzałów (ok.)", _fmt(n)), ("strzelaj poniżej", _fmt(o.get("reblast_below_t"), "t")),
             ("odstęp między strzałami", _fmt(o.get("min_gap_days"), "dni rob.")),
             ("t/mb", _fmt(o.get("t_per_m"), "t/mb")),
             ("godziny strzału", f"ekipa {o.get('crew_arrive', '–')}, odpalenie {o.get('fire_from', '–')}–{o.get('fire_to', '–')}")]
    elif kind == "bay":
        road = next((r for r in site.get("roads", []) if along_m(r["coords"], (o["lat"], o["lon"]))[1] <= SNAP_M), None)
        P = [("title", f"⇆ mijanka {o.get('id')}"), ("na drodze", (road or {}).get("name", "–")),
             ("rzędna", _fmt(_node_elev(info, (o["lat"], o["lon"])), "m", 1)), ("położenie", f"{o['lat']:.5f}, {o['lon']:.5f}")]
    elif kind == "parking":
        P = [("title", f"🅿️ miejsce postoju koparki {o.get('excavator')}"), ("koparka", o.get("excavator")),
             ("położenie", f"{o['lat']:.5f}, {o['lon']:.5f}")]
    elif kind == "dig":
        e = o["_ex"]
        P = [("title", f"⤳ kierunek urabiania {e['id']}"), ("długość", _fmt(polyline_length_m(e["dig_path"]), "m")),
             ("przestawienie co", _fmt(e.get("relocate_every_t", 600), "t")), ("czas przestawienia", _fmt(e.get("relocate_min", 4), "min"))]
    else:
        P = [("title", sel)] + [(k, _fmt(v)) for k, v in o.items() if not isinstance(v, (list, dict))]
    return [(k, v if isinstance(v, str) else _fmt(v)) for k, v in P]


# what can be edited in the panel of a clicked object: (key, label, type, options)
#   num: (min, step) · opt: optional number · text · bool · sel: options · multi: options · time · level · psd
_TIMES = [("crew_arrive", "przyjazd ekipy", "time", None), ("fire_from", "odpalenie od", "time", None),
          ("fire_to", "odpalenie do", "time", None), ("evacuation_min", "ewakuacja przed [min]", "opt", (0.0, 5.0)),
          ("crew_leave", "wyjazd ekipy", "time", None)]
_FAIL = [("_fail_count", "awarii / miesiąc", "num", (0.0, 1.0)),
         ("_fail_hours", "łącznie godzin awarii / miesiąc", "num", (0.0, 1.0))]
RELIABILITY_KIND = {"excavator": "excavator", "crusher": "crusher", "crusher2": "crusher"}
SITE_LIST = {"road": "roads", "pile": "piles", "planned": "planned_piles", "excavator": "excavators",
             "crusher2": "crushers", "area": "areas", "bay": "passing_bays", "parking": "parking"}


def _form_fields(site: dict, kind: str, o: dict) -> list[tuple]:
    pile_ids = [p["id"] for p in site.get("piles", [])] + [p["id"] for p in site.get("planned_piles", [])]
    crusher = [("id", "nazwa", "text", None), ("model", "model (katalog)", "sel", [""] + catalog.names("crusher")),
               ("capacity_tph", "wydajność [t/h]", "num", (50.0, 50.0)),
               ("hopper_t", "kosz [t]", "num", (5.0, 5.0)), ("product", "co produkuje", "sel", [""] + list(PRODUCTS)),
               ("kwh_per_t", "energia [kWh/t]", "num", (0.1, 0.1)), ("target_t", "plan na zmianę [t]", "num", (0.0, 500.0)),
               ("accepts", "przyjmuje z (puste = wszystkie)", "multi", _pile_ids(site))]
    main_crusher = crusher + [("elev_m", "rzędna [m] (puste = z wlotów pochylni)", "elev", None)]
    return {
        "road": [("name", "nazwa", "text", None), ("kind", "rodzaj", "sel", ["droga", "rampa"]),
                 ("grade_draw_pct", "nachylenie wzdłuż rysowania [%]", "num", (-30.0, 0.5)),
                 ("speed_limit_kmh", "limit prędkości [km/h]", "num", (5.0, 5.0)),
                 ("inlet_elev_m", "rzędna wlotu [m] (pochylnia)", "opt", (-2000.0, 1.0)),
                 ("single_lane", "jeden pas (mijanki)", "bool", None), ("active", "w użyciu", "bool", None)],
        "pile": [("id", "nazwa", "text", None), ("material", "materiał", "sel", MATERIALS),
                 ("tonnage_t", "ilość [t]", "num", (0.0, 1000.0)), ("level", "poziom", "level", None),
                 ("loose_density_t_m3", "gęstość spulchniona [t/m³]", "opt", (1.0, 0.05)),
                 ("target_share", "udział w nadawie [%]", "opt", (0.0, 5.0)), ("psd", "uziarnienie", "psd", None)],
        "planned": [("id", "nazwa", "text", None), ("material", "materiał", "sel", MATERIALS),
                    ("tonnage_t", "ilość [t]", "num", (0.0, 1000.0)), ("level", "poziom", "level", None),
                    ("blast_day", "dzień roboczy strzału", "num", (1.0, 1.0)),
                    ("target_share", "udział w nadawie [%]", "opt", (0.0, 5.0)), ("t_per_m", "t/mb", "num", (5.0, 5.0)),
                    ("drilled", "nawiercony", "bool", None), ("psd", "uziarnienie", "psd", None), *_TIMES],
        "area": [("id", "nazwa", "text", None), ("material", "materiał", "sel", MATERIALS), ("level", "poziom", "level", None),
                 ("reserves_t", "zasoby [t]", "num", (0.0, 5000.0)), ("blast_max_t", "największy strzał [t]", "num", (1000.0, 1000.0)),
                 ("reblast_below_t", "strzelaj, gdy zostanie poniżej [t]", "num", (0.0, 1000.0)),
                 ("min_gap_days", "odstęp między strzałami [dni rob.]", "num", (1.0, 1.0)),
                 ("max_per_day", "strzałów na dzień", "num", (1.0, 1.0)),
                 ("target_share", "udział w nadawie [%]", "opt", (0.0, 5.0)), ("t_per_m", "t/mb", "num", (5.0, 5.0)),
                 ("first_drilled", "pierwszy pas nawiercony", "bool", None), *_TIMES],
        "excavator": [("id", "nazwa", "text", None), ("pile", "ładuje z usypu", "sel", pile_ids),
                      ("model", "model (katalog)", "sel", [""] + catalog.names("excavator")),
                      ("bucket_m3", "łyżka [m³]", "num", (0.5, 0.5)), ("swing_cycle_s", "cykl łyżki [s]", "num", (10.0, 1.0)),
                      ("relocate_every_t", "przestawienie co [t]", "num", (50.0, 50.0)),
                      ("relocate_min", "przestawienie [min]", "num", (0.0, 1.0)),
                      ("areas", "pracuje w (harmonogram; puste = wszędzie)", "multi", _pile_ids(site)), *_FAIL],
        "crusher": main_crusher + _FAIL, "crusher2": crusher + _FAIL,
        "parking": [("excavator", "koparka", "sel", [e["id"] for e in site.get("excavators", [])])],
        "bay": [("id", "nazwa", "text", None)],
        "dig": [("relocate_every_t", "przestawienie co [t]", "num", (50.0, 50.0)),
                ("relocate_min", "przestawienie [min]", "num", (0.0, 1.0))],
    }.get(kind, [])


def object_editor(site: dict, sel: str, raw: dict) -> None:
    """Edit every parameter of the clicked object; position and shape stay on the map (✏️ Edytuj / przesuń)."""
    kind = sel.split(":", 1)[0]
    o = find_obj(site, sel)
    if kind == "dig":
        o = o["_ex"]
    fields = _form_fields(site, kind, o)
    if not fields:
        return
    cr = raw.setdefault("crusher", {})
    defaults = {"relocate_every_t": 600.0, "relocate_min": 4.0, "active": True, "kind": "droga", "speed_limit_kmh": 40.0,
                "blast_day": 1, "t_per_m": 50.0, "min_gap_days": 1, "max_per_day": 1, "blast_max_t": 20000.0,
                "first_drilled": True, "capacity_tph": 800.0, "hopper_t": 150.0, "kwh_per_t": 1.1}

    from kruszyk.config import Reliability

    rel_raw = st.session_state.raw.setdefault("reliability", {})  # kept in the session scenario in every mode
    rel = Reliability(**{k_: v_ for k_, v_ in rel_raw.items() if k_ in Reliability.__dataclass_fields__})
    old_id = o.get("id", "C-1")

    def cur(key):
        if key.startswith("_fail") and kind in RELIABILITY_KIND:
            n_, h_, _ = rel.values(old_id, RELIABILITY_KIND[kind])
            return n_ if key == "_fail_count" else h_
        if kind == "crusher" and key in ("capacity_tph", "hopper_t", "kwh_per_t"):
            return cr.get(key, defaults[key])  # the main crusher's machine data live in the scenario
        if kind == "crusher" and key == "id":
            return o.get("id", "C-1")
        return o.get(key, defaults.get(key))

    with st.form(f"objform_{sel}_{st.session_state.obj_ver}", border=False):
        cols = st.columns(3)
        new = {}
        for n, (key, label, typ, opt) in enumerate(fields):
            c, v, wk = cols[n % 3], cur(key), f"of_{sel}_{key}"
            if typ == "num":
                new[key] = c.number_input(label, float(opt[0]), None, float(v if v is not None else opt[0]), float(opt[1]),
                                          key=wk)
            elif typ == "opt":
                new[key] = c.number_input(label, float(opt[0]), None, None if v is None else float(v), float(opt[1]), key=wk,
                                          placeholder="—")
            elif typ == "text":
                new[key] = c.text_input(label, str(v or ""), key=wk)
            elif typ == "bool":
                new[key] = c.checkbox(label, bool(v), key=wk)
            elif typ == "sel":
                options = list(opt) if v in opt or v is None else [v] + list(opt)
                new[key] = c.selectbox(label, options, index=options.index(v) if v in options else 0, key=wk,
                                       format_func=lambda x: x or "—")
            elif typ == "multi":
                new[key] = c.multiselect(label, opt, [x for x in (v or []) if x in opt], key=wk, disabled=not opt,
                                         placeholder="wybierz…" if opt else "brak usypów i obszarów – najpierw je narysuj")
            elif typ == "elev":
                from kruszyk.site import crusher_fixed

                new[key] = c.number_input(label, -2000.0, 5000.0, float(v) if crusher_fixed(o) and v is not None else None,
                                          0.5, key=wk, placeholder=f"z pochylni: {float(v or 0):.1f} m",
                                          help="Wpisz, jeśli znasz rzędną kruszarki. Puste = liczona z rzędnych wlotów "
                                               "pochylni i nachyleń dróg.")
            elif typ == "time":
                new[key] = c.text_input(f"{label} (GG:MM)", str(v or ""), key=wk, placeholder="jak w planie zmiany")
            elif typ == "level":
                with c:
                    new[key] = level_select(label, site, wk, v)
            elif typ == "psd":
                names = list(PRESETS) + (["własna"] if v and preset_of(v) not in PRESETS else [])
                now = preset_of(v) if v else DEFAULT_PRESET
                new[key] = c.selectbox(label, names, index=names.index(now) if now in names else 0, key=wk,
                                       help="własną krzywą wpiszesz w tabeli uziarnienia (sekcja 3)")
        ok = st.form_submit_button("💾 Zastosuj zmiany", type="primary")
    if not ok:
        return
    fails = (new.pop("_fail_count", None), new.pop("_fail_hours", None))
    if kind in RELIABILITY_KIND and fails[0] is not None:
        machines = rel_raw.setdefault("machines", {})
        mid = (new.get("id") or old_id).strip() or old_id
        if old_id != mid:
            machines.pop(old_id, None)
        if old_id in rel.machines or fails != (cur("_fail_count"), cur("_fail_hours")):
            machines[mid] = {"count": float(fails[0]), "hours": float(fails[1])}
    if "product" in new and new["product"] != (o.get("product") or "") and new["product"] in PRODUCTS:
        new["kwh_per_t"] = PRODUCTS[new["product"]]  # another product -> its typical energy
    if new.get("model") and new["model"] != o.get("model"):  # another model from the catalogue -> its parameters
        cat_kind = "excavator" if kind == "excavator" else "crusher"
        new.update(catalog.params(cat_kind, new["model"]))
        if kind == "crusher":
            for k_ in ("max_feed_mm", "dump_time_s"):
                cr[k_] = new.pop(k_)
    for key, val in new.items():
        if isinstance(val, str):
            val = val.strip()
        if key == "psd":
            if val in PRESETS:
                o["psd"] = PRESETS[val]
        elif key == "level":
            if val != o.get("level"):
                o["level"] = val
                o.pop("level_auto", None)
        elif key == "id":
            if val and val != cur("id"):
                rename_refs(site, SITE_LIST.get(kind, ""), o.get("id"), val)
                o["id"] = val
        elif kind == "crusher" and key in ("capacity_tph", "hopper_t", "kwh_per_t"):
            cr[key] = val  # the main crusher's machine data live in the scenario
        elif key == "elev_m":
            if val is None:  # follow the ramp inlets again
                o.pop("elev_set", None)
                o.pop("level", None)
            elif val != o.get("elev_m") or not o.get("elev_set"):
                o["elev_m"], o["elev_set"] = float(val), True
        elif val is None or val == "":
            o.pop(key, None)  # empty = not set (e.g. crew times from the shift plan)
        else:
            o[key] = val
    st.session_state.obj_ver += 1
    st.session_state.map_flash = "✅ Zmieniono parametry – zapisz kopalnię, żeby weszły do obliczeń."
    st.rerun()


MAX_RAMP_PCT = 10.0  # steeper stretches are marked in the profile


def road_points_editor(o: dict, net: dict) -> None:
    """Elevation of every drawn point of a road (bend) and its longitudinal profile. With point elevations every
    stretch has its own grade; without them the road has one grade along its whole length."""
    import plotly.graph_objects as go

    from kruszyk.site import chainage, road_z, set_road_z

    coords = o["coords"]
    ch = chainage(coords)
    given = road_z(o) is not None
    z = road_z(o) or [_node_elev(net, p) for p in coords]
    st.markdown("**📈 Punkty drogi – rzędne i profil podłużny**")
    st.caption(("✍️ Rzędne punktów wpisane – każdy odcinek ma nachylenie z różnicy rzędnych. " if given else
                "Teraz: jedno nachylenie na całej drodze, rzędne niżej są z niego wyliczone. Popraw rzędne punktów "
                "załamania i zapisz – każdy odcinek dostanie własne nachylenie. ")
               + "Punkt 1 = początek rysowania (strzałka na mapie).")
    df = pd.DataFrame({"punkt": range(1, len(coords) + 1), "odległość [m]": [round(c) for c in ch],
                       "rzędna [m]": [None if v is None else round(float(v), 2) for v in z]})
    ed = st.data_editor(df, key=f"rz_{o['uid']}_{st.session_state.obj_ver}", hide_index=True, width="stretch",
                        disabled=["punkt", "odległość [m]"], num_rows="fixed",
                        column_config={"rzędna [m]": st.column_config.NumberColumn(format="%.2f", step=0.1)})
    zz = [None if pd.isna(v) else float(v) for v in ed["rzędna [m]"]]
    if all(v is not None for v in zz) and len(zz) >= 2:
        grades = [100 * (zz[i] - zz[i - 1]) / (ch[i] - ch[i - 1]) if ch[i] > ch[i - 1] else 0.0
                  for i in range(1, len(zz))]
        fig = go.Figure()
        for i, g in enumerate(grades):
            steep = abs(g) > MAX_RAMP_PCT
            fig.add_scatter(x=[ch[i], ch[i + 1]], y=[zz[i], zz[i + 1]], mode="lines", showlegend=False,
                            line=dict(color="#c8102e" if steep else "#2f4156", width=5 if steep else 3),
                            hoverinfo="skip")
            fig.add_annotation(x=(ch[i] + ch[i + 1]) / 2, y=(zz[i] + zz[i + 1]) / 2, text=f"{g:+.1f} %",
                               showarrow=False, yshift=12, font=dict(color="#c8102e" if steep else "#2f4156"))
        fig.add_scatter(x=ch, y=zz, mode="markers+text", text=[str(i + 1) for i in range(len(zz))],
                        textposition="bottom center", marker=dict(size=8, color="#2f4156"), showlegend=False,
                        hovertemplate="punkt %{text}: %{x:.0f} m, rzędna %{y:.1f} m<extra></extra>")
        fig.update_layout(height=260, margin=dict(t=10, b=30, l=10, r=10), xaxis_title="odległość od początku [m]",
                          yaxis_title="rzędna [m]")
        st.plotly_chart(fig, width="stretch")
        if any(abs(g) > MAX_RAMP_PCT for g in grades):
            st.caption(f"🔴 Odcinki ponad {MAX_RAMP_PCT:g} % – sprawdź, czy wozidła tam wjadą z ładunkiem.")
    c1, c2 = st.columns([1, 1])
    if c1.button("💾 Zapisz rzędne punktów", key=f"rz_save_{o['uid']}", type="primary",
                 disabled=any(v is None for v in zz)):
        set_road_z(o, zz)
        st.session_state.obj_ver += 1
        st.session_state.map_flash = (f"✅ Zapisano rzędne {len(zz)} punktów drogi „{o.get('name') or o.get('id')}” – "
                                      "zapisz kopalnię, żeby weszły do obliczeń.")
        st.rerun()
    if given and c2.button("↺ Jedno nachylenie na całej drodze", key=f"rz_clear_{o['uid']}",
                           help="usuwa rzędne punktów; zostaje średnie nachylenie drogi"):
        set_road_z(o, None)
        st.session_state.obj_ver += 1
        st.rerun()
    if any(v is None for v in zz):
        st.caption("Uzupełnij rzędne wszystkich punktów (brak rzędnej = sieć dróg nie ma jeszcze punktu "
                   "odniesienia: wpisz rzędną kruszarki albo wlotu pochylni).")


def _bends_text(coords: list) -> str:
    from kruszyk.physics import curve_speed_kmh
    from kruszyk.site import group_bends, polyline_bends

    bends = group_bends(polyline_bends(coords))
    return ", ".join(f"R {r:.0f} m / {d:.0f}° → {curve_speed_kmh(r):.0f} km/h" for r, d in bends) or "brak (prosta)"


def highlight(m, site: dict, sel: str | None) -> None:
    """Yellow halo around the selected object (drawn under it)."""
    o = find_obj(site, sel)
    if o is None:
        return
    kind = sel.split(":", 1)[0]
    if kind in POINT_KINDS:
        folium.CircleMarker((o["lat"], o["lon"]), radius=22, color=SKETCH_COLOR, weight=4, fill=True,
                            fill_opacity=0.15).add_to(m)
    elif kind in ("pile", "planned", "area"):
        folium.Polygon([tuple(c) for c in o["polygon"]], color=SKETCH_COLOR, weight=6, fill=False).add_to(m)
    else:
        folium.PolyLine([tuple(c) for c in o["coords"]], color=SKETCH_COLOR, weight=18, opacity=0.6).add_to(m)


def edit_panel(site: dict) -> None:
    """Left column in the edit mode: choose an object (or click it on the map) and change / move / delete it."""
    items = site_items(site)
    if "sel_next" in st.session_state:  # set by a click on the map / a delete (the widget can't be set directly then)
        st.session_state.sel_box = st.session_state.pop("sel_next")
    if st.session_state.get("sel_box") not in items:
        st.session_state.sel_box = None
    sel = st.selectbox("Obiekt (albo kliknij go na mapie)", [None] + list(items), key="sel_box",
                       format_func=lambda k: "— wybierz —" if k is None else items[k])
    arm = st.session_state.get("edit_arm")
    if arm and arm[1] != sel:
        st.session_state.edit_arm = arm = None
    if not sel:
        st.caption("Kliknij drogę, usyp, koparkę lub inny znacznik na mapie – zaznaczy się na żółto.")
        return
    o = find_obj(site, sel)
    kind = sel.split(":", 1)[0]
    ver = st.session_state.obj_ver
    k = f"{sel}_{ver}"

    def put(field, value):
        if o.get(field) != value:
            o[field] = value
            st.session_state.obj_ver += 1

    if kind == "road":
        put("name", st.text_input("Nazwa", o.get("name", ""), key=f"en_{k}"))
        put("kind", st.selectbox("Rodzaj", ["droga", "rampa"], index=int(o.get("kind") == "rampa"), key=f"ek_{k}"))
        put("grade_draw_pct", float(st.number_input(
            "Nachylenie wzdłuż strzałki [%]", -25.0, 25.0, float(o.get("grade_draw_pct", 0.0)), 0.5, key=f"eg_{k}",
            help="+ = droga wznosi się w kierunku strzałki, − = opada")))
        if st.button("⇅ Zamień: w górę ↔ w dół", key=f"esw_{k}", help="Gdy droga została narysowana w złą stronę"):
            o["grade_draw_pct"] = -float(o.get("grade_draw_pct", 0.0)) + 0.0
            if o.get("z"):
                o["z"] = list(reversed(o["z"]))
            st.session_state.obj_ver += 1
            st.rerun()
        put("speed_limit_kmh", float(st.number_input("Limit prędkości [km/h]", 5.0, 80.0,
                                                     float(o.get("speed_limit_kmh", 40.0)), 5.0, key=f"el_{k}")))
        put("single_lane", st.checkbox("Jeden pas – ruch wahadłowy, mijanki", bool(o.get("single_lane")), key=f"e1_{k}",
                                       help="Między mijankami jedzie tylko jeden kierunek. Pierwszeństwo ma wozidło "
                                            "jadące pod górę, jadące w dół czeka na najbliższej mijance."))
        put("active", not st.checkbox("Droga planowana (dziś nieużywana)", not o.get("active", True), key=f"ea_{k}"))
    elif kind == "excavator":
        new_id = st.text_input("Oznaczenie", o["id"], key=f"ei_{k}").strip()
        if new_id and new_id != o["id"] and new_id not in {e["id"] for e in site["excavators"]}:
            rename_refs(site, "excavators", o["id"], new_id)
            put("id", new_id)
        piles = [p["id"] for p in site["piles"]] + ["(najbliższy)"]
        cur = o.get("pile") if o.get("pile") in piles else "(najbliższy)"
        put("pile", st.selectbox("Ładuje z usypu", piles, index=piles.index(cur), key=f"ep_{k}"))
        put("bucket_m3", float(st.number_input("Łyżka [m³]", 0.5, 30.0, float(o.get("bucket_m3", 5.0)), 0.5, key=f"eb_{k}")))
    elif kind == "pile":
        new_id = st.text_input("Oznaczenie", o["id"], key=f"ei_{k}").strip()
        if new_id and new_id != o["id"] and new_id not in {p["id"] for p in site["piles"]}:
            rename_refs(site, "piles", o["id"], new_id)
            put("id", new_id)
        mat = o.get("material") if o.get("material") in MATERIALS else "inny"
        put("material", st.selectbox("Materiał", MATERIALS, index=MATERIALS.index(mat), key=f"em_{k}"))
        put("tonnage_t", float(st.number_input("Ilość [t]", 0.0, 10_000_000.0, float(o.get("tonnage_t", 0.0)), 500.0,
                                               key=f"et_{k}")))

    c1, c2 = st.columns(2)
    if kind in POINT_KINDS:
        if c1.button("📍 Przesuń", key=f"emv_{k}", width="stretch"):
            st.session_state.edit_arm = arm = ("move", sel)
    else:
        if c1.button("✏️ Przerysuj", key=f"erd_{k}", width="stretch"):
            st.session_state.edit_arm = arm = ("redraw", sel)
    if c2.button("🗑️ Usuń", key=f"edl_{k}", width="stretch"):
        st.session_state.map_flash = delete_obj(site, sel)
        st.session_state.sel_next = None
        st.session_state.edit_arm = None
        st.session_state.obj_ver += 1
        st.rerun()
    if arm:
        tool = {"move": "postaw **znacznik (📍)** w nowym miejscu",
                "redraw": "narysuj od nowa **wielokąt (⬠)**" if kind in ("pile", "planned", "area") else "narysuj od nowa **linię (/)**"}
        st.info(f"Teraz {tool[arm[0]]} – obiekt przeniesie się tam sam.")
        if st.button("Anuluj", key=f"ecn_{k}"):
            st.session_state.edit_arm = None
            st.rerun()


def level_at(site: dict, point) -> float:
    """Elevation of the road network next to a point (from the grades of the drawn roads)."""
    info = analyze_site(site)
    g, elev = info.get("graph"), info.get("elev") or {}
    if g is None or not elev:
        return 0.0
    n = min(elev, key=lambda i: haversine_m(g.nodes[i], point))
    return round(float(elev[n]), 1)


def set_dig_path(ex: dict, coords: list, attrs: dict | None = None) -> None:
    """Digging line of an excavator: starts at the end nearer to the excavator, which is moved there."""
    pos = (ex["lat"], ex["lon"])
    if haversine_m(coords[-1], pos) < haversine_m(coords[0], pos):
        coords = list(reversed(coords))
    ex["dig_path"] = coords
    ex["lat"], ex["lon"] = coords[0]
    for key, default in (("relocate_every_t", 600.0), ("relocate_min", 4.0)):
        ex[key] = float((attrs or {}).get(key) or ex.get(key) or default)


def level_col(site: dict, label: str = "poziom"):
    return st.column_config.SelectboxColumn(label, options=level_names(site),
                                            help="poziom kopalni (dodasz je w „📏 Poziomy kopalni” nad mapą)")


def level_names(site: dict) -> list[str]:
    return [lv["name"] for lv in site.get("levels") or [] if lv.get("name")]


def level_select(label: str, site: dict, key: str, current: str | None = None) -> str | None:
    """Pick one of the mine levels (None = not set)."""
    names = level_names(site)
    if not names:
        return None
    opts = ["—"] + names
    idx = opts.index(current) if current in opts else 0
    v = st.selectbox(label, opts, index=idx, key=key,
                     format_func=lambda n: n if n == "—" else f"{n} ({level_elev(site, n):+g} m)")
    return None if v == "—" else v


def levels_editor(site: dict) -> None:
    """Mine levels with their elevations – the reference for roads, piles, areas and the crusher."""
    with st.expander(f"📏 Poziomy kopalni – rzędne ({len(level_names(site))})", expanded=False):
        st.caption("Rzędne liczą się od kruszarki i od **wlotów pochylni** (przy pochylni podajesz tylko rzędną "
                   "wlotu i nachylenie). Każda **droga pozioma** leży na rzędnej miejsca, gdzie styka się z pochylnią "
                   "– z niej powstaje poziom. Usypy i obszary przy takiej drodze dostają ten poziom same "
                   "(możesz go zmienić w tabeli).")
        autos = [lv for lv in site.get("levels") or [] if lv.get("auto")]
        if autos:
            st.markdown("**Wyznaczone automatycznie z dróg poziomych:**  \n" + "  \n".join(
                f"• **{lv['name']}** ({lv['elev_m']:+.1f} m) – {', '.join(lv.get('roads', []))}" for lv in autos))
        st.markdown("**Własne poziomy** (opcjonalnie – np. gdy poziom nie ma jeszcze drogi):")
        snap = f"levels_tbl_{st.session_state.obj_ver}"
        if snap not in st.session_state:
            st.session_state[snap] = pd.DataFrame([lv for lv in site.get("levels") or [] if not lv.get("auto")],
                                                  columns=["name", "elev_m"])
        ed = st.data_editor(st.session_state[snap], key=snap + "_ed", num_rows="dynamic", hide_index=True,
                            width="stretch",
                            column_config={"name": st.column_config.TextColumn("poziom", required=True),
                                           "elev_m": st.column_config.NumberColumn("rzędna [m]", required=True)})
        old = [lv.get("name") for lv in site.get("levels") or [] if not lv.get("auto")]
        new = [{"name": str(r["name"]).strip(), "elev_m": float(r["elev_m"])} for _, r in ed.iterrows()
               if r["name"] and not pd.isna(r["name"]) and not pd.isna(r["elev_m"])]
        for i, lv in enumerate(new):  # a renamed level keeps its objects
            if i < len(old) and old[i] and old[i] != lv["name"]:
                _rename_level(site, old[i], lv["name"])
        manual_now = [{"name": lv["name"], "elev_m": lv["elev_m"]} for lv in site.get("levels") or [] if not lv.get("auto")]
        if new != manual_now:
            site["levels"] = new + autos


def _rename_level(site: dict, old: str, new: str) -> None:
    for key in ("piles", "planned_piles", "areas"):
        for o in site.get(key, []):
            if o.get("level") == old:
                o["level"] = new
    for r in site.get("roads", []):
        for f in ("level_from", "level_to"):
            if r.get(f) == old:
                r[f] = new
    if (site.get("crusher") or {}).get("level") == old:
        site["crusher"]["level"] = new


def blast_times_form(prefix: str, defaults: dict | None = None) -> dict:
    """Crew times of a blast: arrival, firing window, evacuation, departure."""
    from app_events import _time

    d = {"crew_arrive": "08:00", "fire_from": "10:00", "fire_to": "10:15", "evacuation_min": 15.0,
         "crew_leave": "10:40", **(defaults or {})}
    st.markdown("**🧨 Godziny strzału**")
    c = st.columns(2)
    with c[0]:
        a = _time("Przyjazd ekipy", d["crew_arrive"], f"{prefix}_arr")
        f1 = _time("Odpalenie od", d["fire_from"], f"{prefix}_f1")
        ev = st.number_input("Ewakuacja przed [min]", 0.0, 120.0, float(d["evacuation_min"]), 5.0, key=f"{prefix}_ev")
    with c[1]:
        lv = _time("Wyjazd ekipy", d["crew_leave"], f"{prefix}_lv")
        f2 = _time("Odpalenie do (otwarcie)", d["fire_to"], f"{prefix}_f2")
    return {"crew_arrive": a, "fire_from": f1, "fire_to": f2, "evacuation_min": ev, "crew_leave": lv}


def _base_label(label: str) -> str:
    """'wydajność [t/h]' / 'przyjmuje z (puste = …)' -> 'wydajność' / 'przyjmuje z'."""
    return label.split("[")[0].split("(")[0].strip().lower()


def _pile_ids(site: dict) -> list[str]:
    return ([p["id"] for p in site.get("piles", [])] + [p["id"] for p in site.get("planned_piles", [])]
            + [a["id"] for a in site.get("areas", [])])


def add_bay(site: dict, lat: float, lon: float) -> str:
    """Passing bay: snapped onto the nearest road; that road becomes single-lane."""
    roads = [r for r in site.get("roads", []) if len(r.get("coords", [])) >= 2]
    if not roads:
        return "⚠️ Najpierw narysuj drogę – mijankę stawia się na drodze"
    best = min(((along_m(r["coords"], (lat, lon)), r) for r in roads), key=lambda x: x[0][1])
    (a, d), road = best
    if d > SNAP_M:
        return f"⚠️ Mijanka musi stać na drodze (najbliższa droga jest {d:.0f} m dalej)"
    plat, plon = point_along(road["coords"], a)
    site.setdefault("passing_bays", []).append({"lat": round(plat, 7), "lon": round(plon, 7)})
    ensure_uids(site)
    note = "" if road.get("single_lane") else " – droga oznaczona jako jednopasmowa (ruch wahadłowy)"
    road["single_lane"] = True
    return f"✅ Dodano mijankę {site['passing_bays'][-1]['id']} na drodze „{road.get('name', road.get('id'))}”{note}"


def editing_map(site: dict) -> folium.Map:
    """The drawing map. Its HTML must not depend on the selected mode or on the objects,
    otherwise st_folium rebuilds it on every change and the view/sketches are lost."""
    m = base_map(site)
    Draw(export=False, position="topleft",
         draw_options={"polyline": {"shapeOptions": {"color": SKETCH_COLOR, "weight": 4}},
                       "polygon": {"shapeOptions": {"color": SKETCH_COLOR}}, "marker": True,
                       "circle": False, "rectangle": False, "circlemarker": False},
         edit_options={"edit": False}).add_to(m)
    _ClearSketches().add_to(m)
    _Snapping().add_to(m)
    folium.LayerControl().add_to(m)
    return m


class _Snapping(folium.MacroElement):
    """Points drawn near what is already on the map stick to it: a click within 14 px of a road point, a pile corner,
    an excavator or the crusher takes exactly that point; near a road stretch – the point on the stretch (a branch
    road meets it). A ring shows where the point will stick."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
          var map = {{ this._parent.get_name() }};
          var PX = 14;
          var ring = L.circleMarker([0, 0], { radius: 9, color: '#B3122A', weight: 3, fill: false, interactive: false });
          function shapes() {
            var out = [];
            map.eachLayer(function (l) {
              if (l === ring || (l.options && l.options.interactive === false)) return;
              if (l instanceof L.Polyline) {
                var ll = l.getLatLngs();
                while (ll.length && Array.isArray(ll[0])) ll = ll[0];
                if (ll.length) out.push({ pts: ll, closed: l instanceof L.Polygon });
              } else if (l instanceof L.Marker || l instanceof L.CircleMarker) {
                out.push({ pts: [l.getLatLng()], closed: false });
              }
            });
            return out;
          }
          function snap(latlng, own) {
            var p = map.latLngToLayerPoint(latlng), best = null, bd = PX;
            var all = shapes();
            all.forEach(function (s) { s.pts.forEach(function (q) {
              if (own && own.indexOf(q) >= 0) return;
              var d = p.distanceTo(map.latLngToLayerPoint(q));
              if (d < bd) { bd = d; best = q; }
            }); });
            if (best) return { ll: L.latLng(best.lat, best.lng), hit: true };
            bd = PX * 0.8;
            all.forEach(function (s) {
              var n = s.pts.length;
              for (var i = 0; i + 1 < n + (s.closed ? 1 : 0); i++) {
                var a = map.latLngToLayerPoint(s.pts[i]), b = map.latLngToLayerPoint(s.pts[(i + 1) % n]);
                var c = L.LineUtil.closestPointOnSegment(p, a, b), d = p.distanceTo(c);
                if (d < bd) { bd = d; best = map.layerPointToLatLng(c); }
              }
            });
            return best ? { ll: best, hit: true } : { ll: latlng, hit: false };
          }
          var drawing = false;
          map.on('draw:drawstart', function () { drawing = true; });
          map.on('draw:drawstop', function () { drawing = false; map.removeLayer(ring); });
          map.on('mousemove', function (e) {
            if (!drawing) return;
            var r = snap(e.latlng);
            if (r.hit) { ring.setLatLng(r.ll); if (!map.hasLayer(ring)) ring.addTo(map); }
            else if (map.hasLayer(ring)) map.removeLayer(ring);
          });
          if (L.Draw && L.Draw.Polyline) {
            var addVertex = L.Draw.Polyline.prototype.addVertex;
            L.Draw.Polyline.prototype.addVertex = function (latlng) {
              var own = this._poly ? this._poly.getLatLngs() : [];
              return addVertex.call(this, snap(latlng, own.slice(0, Math.max(0, own.length - 0))).ll);
            };
          }
          if (L.Draw && L.Draw.Marker) {
            var move = L.Draw.Marker.prototype._onMouseMove;
            L.Draw.Marker.prototype._onMouseMove = function (e) {
              e.latlng = snap(e.latlng).ll;
              return move.call(this, e);
            };
          }
        })();
        {% endmacro %}
    """)


class _ClearSketches(folium.MacroElement):
    """A drawn sketch disappears a moment after it was drawn: by then it has been sent to the app and drawn as
    an object (or rejected). Otherwise the yellow sketch stays on the map – also after the object is deleted."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('draw:created', function (e) {
            var layer = e.layer;
            setTimeout(function () { if (layer && layer.remove) { layer.remove(); } }, 2500);
        });
        {% endmacro %}
    """)


FROM_RAMP = "(ze styku z pochylnią)"
ROAD_COLS = {"id": "id", "name": "nazwa", "kind": st.column_config.SelectboxColumn("rodzaj", options=["droga", "rampa"]),
             "grade_draw_pct": st.column_config.NumberColumn(
                 "nachylenie wzdłuż rysowania [%]", help="+ = droga wznosi się w kierunku rysowania (strzałka na mapie), "
                                                        "− = opada", min_value=-25.0, max_value=25.0, step=0.5),
             "speed_limit_kmh": "limit [km/h]",
             "single_lane": st.column_config.CheckboxColumn("jeden pas", help="ruch wahadłowy – wozidła mijają się "
                                                                             "tylko na mijankach"),
             "active": st.column_config.CheckboxColumn("w użyciu", help="odznacz = droga planowana / zamknięta")}
PILE_COLS = {"id": "usyp", "material": st.column_config.SelectboxColumn("materiał", options=MATERIALS),
             "tonnage_t": "ilość [t]", "fragmentation": "rozdrobnienie (0–1)", "loose_density_t_m3": "gęstość [t/m³]",
             "target_share": st.column_config.NumberColumn("udział w nadawie [%]", min_value=0.0, max_value=100.0,
                                                           help="ile procent nadawy ma iść z tego usypu (puste = bez celu)")}
TIME_COLS = {"crew_arrive": st.column_config.TextColumn("przyjazd ekipy"),
             "fire_from": st.column_config.TextColumn("odpalenie od"),
             "fire_to": st.column_config.TextColumn("odpalenie do"),
             "evacuation_min": st.column_config.NumberColumn("ewakuacja [min]", min_value=0.0),
             "crew_leave": st.column_config.TextColumn("wyjazd ekipy")}
EX_COLS = {"id": "koparka", "bucket_m3": "łyżka [m³]", "swing_cycle_s": "cykl łyżki [s]", "lat": "szer.", "lon": "dł.",
           "relocate_every_t": st.column_config.NumberColumn("przestawienie co [t]", min_value=50.0,
                                                             help="tylko z narysowanym kierunkiem urabiania"),
           "relocate_min": st.column_config.NumberColumn("przestawienie [min]", min_value=0.0)}


def checklist(site: dict, raw: dict) -> None:
    """What is already on the map and what is missing – in one line."""
    n_trucks = sum(int(f.get("count", 0)) for f in raw.get("trucks", []))
    items = [
        ("lokalizacja", site.get("zoom", 6) >= 12),
        ("kruszarka", bool(site.get("crusher"))),
        (f"drogi: {len(site['roads'])}", bool(site["roads"])),
        (f"usypy: {len(site['piles'])}", bool(site["piles"])),
        (f"koparki: {len(site['excavators'])}", bool(site["excavators"])),
        (f"wozidła: {n_trucks}", n_trucks > 0),
    ]
    st.markdown(" · ".join(f"{'✅' if ok else '⬜'} {name}" for name, ok in items))


TRUCK_COLS = {"id": "nr", "model": "typ", "count": None, "w_pracy": st.column_config.CheckboxColumn("w pracy"),
              "payload_t": st.column_config.NumberColumn("ładowność [t]", min_value=5.0, step=1.0),
              "empty_mass_t": st.column_config.NumberColumn("masa własna [t]", min_value=1.0),
              "power_kw": st.column_config.NumberColumn("moc [kW]", min_value=50.0),
              "max_speed_kmh": st.column_config.NumberColumn("v max [km/h]", min_value=5.0),
              "fuel_lph_loaded": st.column_config.NumberColumn("paliwo pełny [l/h]", min_value=0.0),
              "fuel_lph_empty": st.column_config.NumberColumn("paliwo pusty [l/h]", min_value=0.0),
              "fuel_lph_idle": st.column_config.NumberColumn("paliwo postój [l/h]", min_value=0.0),
              "tank_l": st.column_config.NumberColumn("zbiornik [l]", min_value=50.0),
              "excavators": st.column_config.TextColumn("koparki (puste = wszystkie)"),
              "fail_n": st.column_config.NumberColumn("awarii / mies.", min_value=0.0,
                                                      help="średnio z 3–6 miesięcy, postoje > 15 min, bez przeglądów"),
              "fail_h": st.column_config.NumberColumn("godzin awarii / mies.", min_value=0.0)}
TRUCK_NUM = ["payload_t", "empty_mass_t", "power_kw", "max_speed_kmh", "fuel_lph_loaded", "fuel_lph_empty",
             "fuel_lph_idle", "tank_l"]


def quick_fleet_crusher(raw: dict, site: dict) -> None:
    """Every truck as its own row with its own parameters; changes go straight to the scenario."""
    from kruszyk.config import Reliability, expand_trucks

    st.subheader("5. Wozidła")
    st.caption("Każde wozidło osobno – z własną ładownością, masą, mocą, zużyciem paliwa i zbiornikiem. **W pracy** "
               "odznacz, gdy wozidło stoi (np. w remoncie). **Koparki**: do których może jeździć, np. `K-1, K-2` "
               "(puste = do wszystkich). Nowy wiersz = kopia ostatniego wozidła. **Awaryjność** – dwie ostatnie kolumny: "
               "awarie w miesiącu i ich łączny czas (puste / domyślne = założenie).")
    ex_ids = {e["id"] for e in site.get("excavators", [])} | {e["id"] for e in raw.get("excavators", [])}
    rel_raw = raw.setdefault("reliability", {})
    rel = Reliability(**{k_: v_ for k_, v_ in rel_raw.items() if k_ in Reliability.__dataclass_fields__})
    with st.container(border=True):
        c = st.columns([3, 1, 1])
        model = c[0].selectbox("➕ Dodaj wozidła z katalogu", catalog.names("truck"), key="fleet_cat_model")
        n_add = c[1].number_input("ile", 1, 30, 1, key="fleet_cat_n")
        if c[2].button("Dodaj", key="fleet_cat_add", width="stretch"):
            trucks = expand_trucks(raw.get("trucks", []))
            used = {t["id"] for t in trucks}
            for _ in range(int(n_add)):
                tid = next(f"WZ-{i:02d}" for i in range(1, 1000) if f"WZ-{i:02d}" not in used)
                used.add(tid)
                trucks.append(catalog.apply("truck", {"id": tid, "count": 1, "excavators": []}, model))
            raw["trucks"] = trucks
            st.session_state.ver += 1
            st.rerun()
        st.caption("Zmiana **typu** w tabeli na model z katalogu wpisuje jego parametry (ładowność, masa, moc, paliwo, "
                   "zbiornik). Parametry orientacyjne – kalibracja dopasuje je do Waszej zmiany.")

    def fail(t: dict) -> tuple:
        key = t["id"] if t["id"] in rel.machines else f"truck:{t.get('model', '')}"
        return rel.values(key, "truck")[:2]
    # fixed starting table (the editor applies its edits on top of it), rebuilt when breakdowns change elsewhere
    sig = abs(hash(repr(sorted((k_, tuple(sorted(v_.items()))) for k_, v_ in rel.machines.items()))))
    base_key = f"qfleet_base_{st.session_state.ver}_{sig % 10**8}"
    if base_key not in st.session_state:
        trucks = expand_trucks(raw.get("trucks", []))
        st.session_state[base_key] = (trucks, pd.DataFrame(
            [{"id": t["id"], "model": t.get("model", ""), "w_pracy": bool(t.get("count", 1)),
              **{c: float(t.get(c) or 0) for c in TRUCK_NUM}, "excavators": ", ".join(t.get("excavators") or []),
              "fail_n": fail(t)[0], "fail_h": fail(t)[1]}
             for t in trucks], columns=["id", "model", "w_pracy", *TRUCK_NUM, "excavators", "fail_n", "fail_h"]))
    old, start = st.session_state[base_key]
    models = list(dict.fromkeys(catalog.names("truck") + [m for m in start["model"] if m]))
    ed = st.data_editor(start, num_rows="dynamic", key=f"qfleet_{base_key}", hide_index=True,
                        width="stretch", column_config={**{k_: v_ for k_, v_ in TRUCK_COLS.items() if v_ is not None},
                                                        "model": st.column_config.SelectboxColumn("typ", options=models)})
    trucks, used, retyped = [], set(), False
    for i, r in ed.reset_index(drop=True).iterrows():
        prev = trucks[-1] if trucks else (old[-1] if old else {
            "model": "wozidło 40 t", "payload_t": 40.0, "empty_mass_t": 30.0, "power_kw": 380, "max_speed_kmh": 45,
            "fuel_lph_loaded": 45, "fuel_lph_empty": 24, "fuel_lph_idle": 10, "tank_l": 700.0})
        base = dict(old[i]) if i < len(old) else dict(prev)
        tid = str(r["id"] or "").strip()
        n = len(trucks) + 1
        while not tid or tid in used:  # a new row or a repeated number: next free WZ-nn
            tid, n = f"WZ-{n:02d}", n + 1
        used.add(tid)
        row = {c: (float(r[c]) if not pd.isna(r[c]) else float(base.get(c) or 0)) for c in TRUCK_NUM}
        new_model = "" if pd.isna(r["model"]) else str(r["model"] or "")
        if new_model in catalog.TRUCKS and new_model != str(base.get("model", "")):  # a catalogue model chosen
            row.update({k_: float(v_) for k_, v_ in catalog.params("truck", new_model).items() if k_ in TRUCK_NUM})
            base["drive_efficiency"] = catalog.TRUCKS[new_model]["drive_efficiency"]
            retyped = True
        if i < len(old) and abs(float(old[i].get("payload_t", row["payload_t"])) - row["payload_t"]) > 1e-6 \
                and row["empty_mass_t"] == float(old[i].get("empty_mass_t", 0)):  # payload changed alone: scale the truck
            scale = row["payload_t"] / float(old[i]["payload_t"])
            row["empty_mass_t"] = round(row["empty_mass_t"] * scale, 1)
            row["power_kw"] = round(row["power_kw"] * scale)
        excs = [x.strip() for x in str(r["excavators"] or "").split(",") if x.strip() in ex_ids]
        model = str(r["model"] or "").strip() or str(base.get("model", "wozidło"))
        trucks.append({**{k_: v_ for k_, v_ in base.items() if k_ not in ("blend",)}, **row, "id": tid, "model": model,
                       "count": 1 if bool(r["w_pracy"]) or pd.isna(r["w_pracy"]) else 0, "excavators": excs,
                       **({"blend": base["blend"]} if base.get("blend") else {})})
    raw["trucks"] = trucks
    if retyped:  # show the model's parameters in the table
        st.session_state.ver += 1
        st.rerun()
    machines = rel_raw.setdefault("machines", {})
    for i, (t, (_, r)) in enumerate(zip(trucks, ed.reset_index(drop=True).iterrows())):
        n_, h_ = (float(r[c]) if not pd.isna(r[c]) else None for c in ("fail_n", "fail_h"))
        if n_ is None or h_ is None:
            continue
        was = (float(start["fail_n"].iloc[i]), float(start["fail_h"].iloc[i])) if i < len(start) else fail(t)
        if (n_, h_) != was:  # changed in this table (other tables keep what they set)
            machines[t["id"]] = {"count": n_, "hours": h_}
    st.caption(f"W pracy: **{sum(t['count'] for t in trucks)}** z {len(trucks)} wozideł. Kruszarki (wydajność, kosz, "
               "produkt, plan, z których usypów przyjmują) – w sekcji 3, zakładka „🛣️ Drogi, mijanki, kruszarki”.")


def psd_editor(site: dict, max_feed_mm: float, penalty: float) -> None:
    """Grading curve of every pile: pick a typical curve, set it from P80, or type sieve results."""
    import plotly.graph_objects as go

    st.markdown("#### Krzywe uziarnienia usypów")
    st.caption("Krzywa mówi, ile procent materiału przechodzi przez sito danej wielkości. Im grubszy urobek "
               "(P80 = rozmiar, poniżej którego jest 80 % materiału), tym **dłuższy załadunek** i **wolniejsza kruszarka**; "
               f"bryły większe niż wlot kruszarki ({max_feed_mm:.0f} mm) to nadziarno.")
    ids = [p["id"] for p in site["piles"]]
    c1, c2 = st.columns([2, 3])
    with c1:
        pid = st.selectbox("Usyp", ids, key="psd_pile")
        pile = next(p for p in site["piles"] if p["id"] == pid)
        ver_key = f"psd_ver_{pid}"
        st.session_state.setdefault(ver_key, 0)
        current = preset_of(pile.get("psd")) if pile.get("psd") else "własna"
        options = list(PRESETS) + ["własna"]
        choice = st.selectbox("Typowa krzywa", options, index=options.index(current),
                              key=f"psd_preset_{pid}_{st.session_state[ver_key]}")
        if choice != current and choice in PRESETS:
            pile["psd"] = PRESETS[choice]
            st.session_state[ver_key] += 1
            st.rerun()
        with st.expander("…albo z dwóch liczb: P80 i jednorodność"):
            p80 = st.number_input("P80 [mm]", 20.0, 2000.0, 300.0, 10.0, key=f"p80_{pid}")
            n = st.slider("Jednorodność n (mała = szeroki zakres ziaren)", 0.6, 2.5, 1.2, 0.1, key=f"n_{pid}")
            if st.button("Ustaw krzywą", key=f"rr_{pid}"):
                pile["psd"] = rosin_rammler(p80, n)
                st.session_state[ver_key] += 1
                st.rerun()
        st.caption("…albo wpisz wyniki przesiewu (możesz dodawać i usuwać wiersze):")
        base_key = f"psd_base_{pid}_{st.session_state[ver_key]}"
        if base_key not in st.session_state:
            st.session_state[base_key] = pd.DataFrame(clean(pile.get("psd") or PRESETS[DEFAULT_PRESET]),
                                                      columns=["sito [mm]", "przechodzi [%]"])
        ed = st.data_editor(st.session_state[base_key], num_rows="dynamic", hide_index=True, width="stretch",
                            key=f"psd_table_{pid}_{st.session_state[ver_key]}",
                            column_config={"przechodzi [%]": st.column_config.NumberColumn(min_value=0.0, max_value=100.0)})
        pts = clean(ed.dropna().values.tolist())
        if len(pts) >= 2:
            pile["psd"] = pts
    with c2:
        fig = go.Figure()
        for p in site["piles"]:
            if p.get("psd"):
                c = clean(p["psd"])
                fig.add_scatter(x=[x for x, _ in c], y=[y for _, y in c], mode="lines+markers", name=p["id"],
                                line=dict(width=5 if p["id"] == pid else 2))
        fig.add_vline(x=max_feed_mm, line_dash="dot", annotation_text="wlot kruszarki")
        fig.update_layout(height=320, xaxis=dict(type="log", title="sito [mm]"), yaxis=dict(title="przechodzi [%]", range=[0, 101]),
                          margin=dict(t=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        rows = [{"usyp": p["id"], "materiał": p.get("material", ""), **summary(p["psd"], max_feed_mm, penalty)}
                for p in site["piles"] if p.get("psd")]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_map_tab(raw_for_build: dict, set_raw, simple: bool = False, schedule: bool | None = None) -> None:
    """schedule: True = harmonogram (plan of blasts shown), False = one shift (hidden), None = everything."""
    site = get_site(raw_for_build)
    checklist(site, raw_for_build)

    st.subheader("1. Lokalizacja kopalni")
    c1, c2, c3 = st.columns([4, 1, 1])
    link = c1.text_input("Wklej link z Google Maps albo współrzędne „szer., dł.” (np. 50.8712, 20.6283)",
                         key="calib_link")
    zoom = c2.number_input("Przybliżenie", 3, 20, int(site.get("zoom", 16)))
    if c3.button("📍 Ustaw", width="stretch"):
        pos = parse_coords(link)
        if pos:
            site["center"], site["zoom"] = list(pos), int(zoom)
            st.session_state.map_key += 1  # the only moment the map is rebuilt on purpose
            st.rerun()
        else:
            st.error("Nie rozpoznano współrzędnych. W Google Maps: prawy klik na mapie → kliknij współrzędne (kopiują się).")

    st.subheader("2. Rysowanie obiektów")
    from kruszyk.site import sync_levels

    sync_levels(site)
    levels_editor(site)
    groups = [g for g in MODE_GROUPS if schedule is not False or g != "📅 Plan strzałów"]
    if st.session_state.get("group_next") in groups:  # "edit this object" from the properties panel
        st.session_state.draw_group = st.session_state.pop("group_next")
    if st.session_state.get("draw_group") not in groups:
        st.session_state.draw_group = groups[0]
    group = st.radio("Co robisz?", groups, horizontal=True, key="draw_group",
                     help="Najpierw grupa, potem po lewej – co dokładnie rysujesz.")
    c1, c2 = st.columns([1, 3])
    with c1:
        options = MODE_GROUPS[group]
        if len(options) > 1:
            mode = st.radio("Co dodajesz?", options, key=f"draw_mode_{list(MODE_GROUPS).index(group)}")
        else:
            mode = options[0]
            if mode != EDIT:
                st.markdown(f"**{mode}**")
        attrs: dict = {}
        if mode == "Droga / rampa":
            attrs["kind"] = st.selectbox("Rodzaj", ["droga", "rampa"])
            attrs["name"] = st.text_input("Nazwa", f"{attrs['kind']} {len(site['roads']) + 1}")
            direction = st.radio("📐 Rysuję drogę…", list(DIRECTIONS), horizontal=True,
                                 index=0 if attrs["kind"] == "rampa" else 2,
                                 help="Kierunek, w którym będziesz klikać punkty: zaczynasz na górze i idziesz w dół "
                                      "(⬇), zaczynasz na dole i idziesz w górę (⬆) albo droga jest płaska (➡). "
                                      "Na mapie strzałka pokaże kierunek rysowania.")
            if DIRECTIONS[direction] == 0:  # flat: 0 %, nothing to type
                grade = st.number_input("Nachylenie [%]", 0.0, 0.0, 0.0, disabled=True, key="road_grade_flat",
                                        help="Droga płaska ma 0 %. Wybierz ⬇ / ⬆, żeby wpisać nachylenie.")
            else:
                grade = st.number_input("Nachylenie [%]", 0.0, 25.0, 9.0 if attrs["kind"] == "rampa" else 2.0, 0.5,
                                        key="road_grade", help="Rampa: zwykle 8–10 %. Droga technologiczna: 0–3 %.")
            attrs["grade_draw_pct"] = DIRECTIONS[direction] * grade + 0.0
            if attrs["kind"] == "rampa":
                z = st.number_input("Rzędna wlotu [m] (pierwszy narysowany punkt)", -2000.0, 3000.0, value=None,
                                    step=0.5, key="road_inlet",
                                    help="Wystarczy rzędna wlotu – dalej liczy się z nachylenia. Puste = rzędna "
                                         "z sieci dróg od kruszarki.")
                if z is not None:
                    attrs["inlet_elev_m"] = z
            elif DIRECTIONS[direction] == 0:
                st.caption("Droga pozioma: jej poziom wyznaczy się sam – z rzędnej miejsca, gdzie styka się z pochylnią.")
            attrs["speed_limit_kmh"] = st.number_input("Limit prędkości [km/h]", 5.0, 80.0,
                                                       25.0 if attrs["kind"] == "rampa" else 40.0, 5.0)
            attrs["single_lane"] = st.checkbox(
                "Jeden pas – ruch wahadłowy (mijanki)",
                help="Wozidła nie miną się na tej drodze. Postaw na niej mijanki (tryb „Mijanka”). Pierwszeństwo ma "
                     "wozidło jadące pod górę – jadące w dół czeka na najbliższej mijance.")
            attrs["active"] = not st.checkbox("Droga planowana (dziś nieużywana)",
                                              help="Narysuj ją teraz, a włączysz w wariancie (moduł „Planowanie produkcji” → „② Wariant – co jeśli”).")
            st.caption("Narzędzie „linia” (/). Rysuj po osi drogi, zakończ klikając ostatni punkt. "
                       "Końce dróg, które się łączą, stawiaj blisko siebie (≤ 30 m).")
        elif mode == "Usyp (materiał)":
            planned = schedule is not False and st.radio(
                "Stan", ["leży – urobek gotowy", "planowany odstrzał"], horizontal=True, key="pile_status",
                help="Planowanie odstrzałów jest w module „Planowanie produkcji”.") == "planowany odstrzał"
            attrs["status"] = "planned" if planned else "lying"
            n_all = len(site["piles"]) + len(site.get("planned_piles", []))
            attrs["id"] = st.text_input("Oznaczenie usypu", f"{'S' if planned else 'U'}-{n_all + 1:02d}", key=f"pile_id_{planned}")
            attrs["material"] = st.selectbox("Materiał", MATERIALS, index=_material_index())
            lvl = level_select("Poziom", site, "pile_lvl")
            if lvl:
                attrs["level"] = lvl
            attrs["tonnage_t"] = st.number_input("Ile ton da strzał" if planned else "Ilość [t]", 0.0, 10_000_000.0,
                                                 15_000.0 if planned else 10_000.0, 500.0, key=f"pile_t_{planned}")
            preset = st.selectbox("Uziarnienie (krzywą można potem dokładnie ustawić)", list(PRESETS),
                                  index=_preset_index())
            attrs["psd"] = PRESETS[preset]
            attrs["loose_density_t_m3"] = st.number_input("Gęstość spulchniona [t/m³]", 1.0, 3.0,
                                                          float(_dflt("loose_density_t_m3", 1.75)), 0.05)
            share = st.number_input("Udział w nadawie [%] (0 = bez celu)", 0.0, 100.0, 0.0, 5.0,
                                    help="Oczekiwany procent nadawy z tego usypu. Steruje nim optymalizacja – tylko "
                                         "dla wozideł nieprzypisanych na sztywno do jednej koparki.")
            if share > 0:
                attrs["target_share"] = share
            if planned:
                attrs["blast_day"] = int(st.number_input("Strzał w dniu roboczym nr", 1, 300, 3, 1, key="pile_day",
                                                         help="licząc od pierwszego dnia harmonogramu (1 = pierwszy "
                                                              "dzień roboczy)"))
                c = st.columns(2)
                attrs["t_per_m"] = c[0].number_input("Ton na metr wiercenia [t/mb]", 5.0, 500.0,
                                                     float(_dflt("t_per_m", 50.0)), 1.0, key="pile_tpm")
                attrs["drilled"] = c[1].checkbox("Już nawiercony", key="pile_drilled")
                attrs.update(blast_times_form("pile"))
                st.caption("Narzędzie „wielokąt” (⬠). Obrysuj miejsce strzału (urobek po strzale). W dniu strzału "
                           "wyrobisko jest zamykane w podanych godzinach, ekipa jedzie w to miejsce, a koparka bez "
                           "urobku przejeżdża tu sama.")
            else:
                st.caption("Narzędzie „wielokąt” (⬠). Obrysuj usyp, zamknij klikając pierwszy punkt.")
        elif mode == "Planowany strzał (nowy usyp)":
            n_pl = len(site.get("planned_piles", []))
            attrs["id"] = st.text_input("Oznaczenie", f"S-{n_pl + 1:02d}")
            attrs["material"] = st.selectbox("Materiał", MATERIALS, index=_material_index(), key="pl_mat")
            attrs["tonnage_t"] = st.number_input("Ile ton da strzał", 0.0, 10_000_000.0, 15_000.0, 1000.0, key="pl_t")
            preset = st.selectbox("Spodziewane uziarnienie", list(PRESETS), index=_preset_index(), key="pl_psd")
            attrs["psd"] = PRESETS[preset]
            attrs["loose_density_t_m3"] = float(_dflt("loose_density_t_m3", 1.75))
            share = st.number_input("Udział w nadawie [%] (0 = bez celu)", 0.0, 100.0, 0.0, 5.0, key="pl_share",
                                    help="ile procent nadawy ma iść z tego usypu, gdy już będzie odstrzelony")
            if share > 0:
                attrs["target_share"] = share
            attrs["blast_day"] = int(st.number_input("Strzał w dniu roboczym nr", 1, 300, 3, 1, key="pl_day",
                                                     help="licząc od pierwszego dnia okresu w module „Planowanie produkcji” "
                                                          "(1 = pierwszy dzień roboczy)"))
            c = st.columns(2)
            attrs["t_per_m"] = c[0].number_input("Ton na metr wiercenia [t/mb]", 5.0, 500.0, float(_dflt("t_per_m", 50.0)),
                                                 1.0, key="pl_tpm",
                                                 help="z tego program liczy metry do wywiercenia i czas wiercenia")
            attrs["drilled"] = c[1].checkbox("Już nawiercony", key="pl_drilled")
            attrs.update(blast_times_form("pl"))
            st.caption("Narzędzie „wielokąt” (⬠). Obrysuj miejsce przyszłego strzału (urobek po strzale). Do dnia "
                       "strzału nic tu nie ma; potem koparka, której skończył się urobek, przejedzie tu sama "
                       "(najbliższa wolna). Wozidła ładują przy rogu usypu najbliższym drogi – doprowadź tam drogę. "
                       "W dniu strzału wyrobisko jest zamykane w podanych godzinach, a ekipa jedzie w to miejsce.")
        elif mode == "Obszar wydobycia (strzały automatycznie)":
            n_ar = len(site.get("areas", []))
            attrs["id"] = st.text_input("Oznaczenie obszaru", f"O{n_ar + 1}", key="ar_id")
            attrs["material"] = st.selectbox("Materiał", MATERIALS, index=_material_index(), key="ar_mat")
            preset = st.selectbox("Spodziewane uziarnienie", list(PRESETS), index=_preset_index(), key="ar_psd")
            attrs["psd"], attrs["loose_density_t_m3"] = PRESETS[preset], float(_dflt("loose_density_t_m3", 1.75))
            attrs["reserves_t"] = st.number_input("Zasoby w obszarze [t] (0 = policz z powierzchni)", 0.0, 50_000_000.0,
                                                  0.0, 10_000.0, key="ar_res")
            lvl = level_select("Poziom (piętro)", site, "ar_lvl")
            if lvl:
                attrs["level"], attrs["level_m"] = lvl, level_elev(site, lvl)
            else:
                st.caption("Bez poziomu rzędna zostanie odczytana z dróg przy obszarze (dodaj poziomy powyżej mapy).")
            c = st.columns(2)
            attrs["bench_m"] = c[0].number_input("Wysokość piętra [m]", 1.0, 50.0, float(_dflt("bench_m", 15.0)), 1.0,
                                                 key="ar_h")
            attrs["solid_density_t_m3"] = c[1].number_input("Gęstość w caliźnie [t/m³]", 1.5, 3.5,
                                                            float(_dflt("solid_density_t_m3", 2.6)), 0.05, key="ar_d")
            st.markdown("**Ograniczenia strzałów**")
            c = st.columns(2)
            attrs["blast_max_t"] = c[0].number_input("Największy strzał [t]", 1000.0, 1_000_000.0,
                                                     float(_dflt("blast_max_t", 20_000.0)), 1000.0, key="ar_max")
            attrs["reblast_below_t"] = c[1].number_input("Strzelaj, gdy zostanie mniej niż [t]", 0.0, 1_000_000.0,
                                                         float(_dflt("reblast_below_t", 8_000.0)), 1000.0, key="ar_below",
                                                         help="urobku z tego obszaru na usypach")
            attrs["min_gap_days"] = int(c[0].number_input("Odstęp między strzałami [dni rob.]", 1, 30,
                                                          int(_dflt("min_gap_days", 2)), 1, key="ar_gap"))
            attrs["max_per_day"] = int(c[1].number_input("Strzałów dziennie maks.", 1, 5, 1, 1, key="ar_mpd"))
            c = st.columns(2)
            attrs["t_per_m"] = c[0].number_input("Ton na metr wiercenia [t/mb]", 5.0, 500.0, float(_dflt("t_per_m", 50.0)),
                                                 1.0, key="ar_tpm",
                                                 help="z tego program liczy metry do wywiercenia i czas wiercenia")
            attrs["first_drilled"] = c[1].checkbox("Pierwszy pas już nawiercony", True, key="ar_fd")
            share = st.number_input("Udział w nadawie [%] (0 = bez celu)", 0.0, 100.0, 0.0, 5.0, key="ar_share")
            if share > 0:
                attrs["target_share"] = share
            attrs.update(blast_times_form("ar"))
            st.caption("Narzędzie „wielokąt” (⬠). Obrysuj obszar, który będziecie urabiać. W kroku harmonogramu program "
                       "**sam ustali terminy strzałów**: dzieli obszar na kolejne pasy (od strony drogi) wielkości "
                       "największego strzału i strzela, gdy urobku z obszaru zostaje mniej niż próg albo koparki nie "
                       "mają co ładować – z zachowaniem odstępu między strzałami. Kruszarki przyjmują urobek z obszaru, "
                       "jeśli zaznaczysz go w tabeli kruszarek.")
        elif mode == "Koparka":
            cat = _next_from_catalogue("excavators", {e["id"] for e in site["excavators"]})
            if cat:
                st.caption(f"Z katalogu projektu: **{cat.get('type', cat['id'])}**")
            attrs["id"] = st.text_input("Oznaczenie koparki", cat.get("id", f"K-{len(site['excavators']) + 1}"),
                                        key=f"ex_id_{cat.get('id', '')}")
            piles = [p["id"] for p in site["piles"]]
            attrs["pile"] = st.selectbox("Ładuje z usypu", piles or ["(najbliższy)"])
            model = st.selectbox("Model (katalog)", ["— własne parametry —"] + catalog.names("excavator"), key="ex_model",
                                 help="wybór modelu wpisuje typową łyżkę skalną i cykl – możesz je poprawić")
            if model in catalog.EXCAVATORS:
                cat = {**cat, **catalog.params("excavator", model)}
                attrs["model"] = model
            attrs["bucket_m3"] = st.number_input("Łyżka [m³]", 0.5, 30.0, float(cat.get("bucket_m3", 5.0)), 0.5,
                                                 key=f"ex_b_{cat.get('id', '')}_{model}")
            attrs["swing_cycle_s"] = st.number_input("Cykl łyżki [s]", 10.0, 90.0, float(cat.get("swing_cycle_s", 30.0)),
                                                     1.0, key=f"ex_c_{cat.get('id', '')}_{model}")
            if cat.get("fill_factor"):
                attrs["fill_factor"] = float(cat["fill_factor"])
            st.caption("Narzędzie „znacznik” (📍). Postaw go w miejscu pracy koparki.")
        elif mode == "Miejsce postoju koparki":
            ex_ids = [e["id"] for e in site["excavators"]]
            attrs["excavator"] = st.selectbox("Której koparki?", ex_ids) if ex_ids else None
            st.caption("Narzędzie „znacznik” (📍). Tu koparka zjeżdża na czas strzelania (i tankowania, jeśli tak "
                       "ustawisz). Dojedzie najkrótszą drogą; bez połączenia drogowego – na wprost. Bez znacznika koparka cofa się o kilkaset metrów swoją trasą.")
        elif mode == "Kruszarka":
            if site.get("crusher"):
                attrs["new_crusher"] = st.radio("Kruszarka", ["Przesuń główną", "Dodaj kolejną"], key="cr_which",
                                                horizontal=True) == "Dodaj kolejną"
            if not attrs.get("new_crusher"):
                lvl = level_select("Poziom kruszarki", site, "cr_lvl", (site.get("crusher") or {}).get("level"))
                if lvl:
                    attrs["level"] = lvl
            used = {(site.get("crusher") or {}).get("id", "C-1") if site.get("crusher") else None} | \
                {c["id"] for c in site.get("crushers", [])}
            cat = _next_from_catalogue("crushers", used)
            if cat:
                st.caption(f"Z katalogu projektu: **{cat.get('type', cat['id'])}**")
            if not attrs.get("new_crusher") and cat and not site.get("crusher"):
                attrs["id"], attrs["product"] = cat["id"], cat.get("product", "")
            if attrs.get("new_crusher"):
                n_cr = len(site.get("crushers", []))
                attrs["id"] = st.text_input("Oznaczenie", cat.get("id", f"C-{n_cr + 2}"), key=f"cr_id_{cat.get('id', '')}")
                model = st.selectbox("Model (katalog)", ["— własne parametry —"] + catalog.names("crusher"),
                                     key="cr_model", help="stacjonarne i mobilne; wpisuje wydajność, kosz i wlot")
                if model in catalog.CRUSHERS:
                    cat = {**cat, **catalog.params("crusher", model)}
                    attrs.update(model=model, max_feed_mm=cat["max_feed_mm"], dump_time_s=cat["dump_time_s"])
                attrs["capacity_tph"] = st.number_input("Wydajność [t/h]", 50.0, 10_000.0, float(cat.get("capacity_tph", 800.0)),
                                                        50.0, key=f"cr_cap_{cat.get('id', '')}_{model}")
                attrs["hopper_t"] = st.number_input("Kosz [t]", 5.0, 2000.0, float(cat.get("hopper_t", 150.0)), 5.0,
                                                    key=f"cr_hop_{cat.get('id', '')}_{model}")
                attrs["accepts"] = st.multiselect(
                    "Przyjmuje urobek z (puste = ze wszystkich)", _pile_ids(site), key="cr_acc")
                attrs["product"] = st.selectbox("Co produkuje", list(PRODUCTS), key=f"cr_prod_{cat.get('id', '')}",
                                                index=list(PRODUCTS).index(cat["product"]) if cat.get("product") in PRODUCTS
                                                else 0)
                attrs["kwh_per_t"] = st.number_input("Energia [kWh/t]", 0.1, 10.0, float(
                    cat["kwh_per_t"] if cat.get("kwh_per_t") and cat.get("product") == attrs["product"]
                    else PRODUCTS[attrs["product"]]),
                                                     0.1, key=f"cr_kwh_{attrs['product']}",
                                                     help="typowa dla wybranego produktu – możesz poprawić")
                attrs["target_t"] = st.number_input("Plan na zmianę [t] (0 = bez planu)", 0.0, 100_000.0, 0.0, 500.0,
                                                    key="cr_plan")
            st.caption("Narzędzie „znacznik” (📍). Postaw go na koszu kruszarki (miejsce wyładunku). Każda kruszarka "
                       "może przyjmować urobek ze wszystkich albo tylko z wybranych usypów – wozidło jedzie do "
                       "kruszarki, która przyjmuje jego materiał (przy kilku: dziś najbliższej, system – tej, gdzie "
                       "szybciej wyładuje).")
        elif mode == "Kierunek urabiania koparki":
            ex_ids = [e["id"] for e in site["excavators"]]
            attrs["excavator"] = st.selectbox("Której koparki?", ex_ids, key="dig_ex") if ex_ids else None
            attrs["relocate_every_t"] = st.number_input("Przestawienie koparki co [t]", 50.0, 20_000.0, 600.0, 50.0,
                                                        help="po tylu tonach koparka przesuwa się do przodu wzdłuż linii")
            attrs["relocate_min"] = st.number_input("Czas przestawienia [min]", 0.5, 60.0, 4.0, 0.5,
                                                    help="ile koparka nie ładuje przy każdym przestawieniu")
            st.caption("Narzędzie „linia” (/). Narysuj po froncie, którędy koparka będzie szła w trakcie urabiania: "
                       "od miejsca startu do miejsca, w którym skończy usyp. Koparka przeniesie się na początek linii; "
                       "wozidła dojeżdżają zawsze do jej aktualnego miejsca (dojazd po placu wydłuża się albo skraca). "
                       "Koniec ≈ cały urobek tej koparki z usypu.")
        elif mode == "Mijanka":
            st.caption("Narzędzie „znacznik” (📍). Postaw go na drodze w miejscu mijanki – przyklei się do drogi, "
                       "a droga zostanie oznaczona jako jednopasmowa. Odcinek między dwiema mijankami (albo mijanką "
                       "i końcem drogi) przejeżdża naraz tylko jeden kierunek; pierwszeństwo ma jadący pod górę.")
        elif mode == "Miejsce strzelania":
            st.caption("Narzędzie „znacznik” (📍). Postaw go w miejscu strzelania. Godziny przyjazdu ekipy, "
                       "odpalenia i wyjazdu ustawisz niżej w sekcji „Zmiana i zdarzenia”.")
        if mode == EDIT:
            edit_panel(site)
        else:
            st.info("Wybierz typ i parametry, potem narysuj obiekt – **dodaje się sam**. "
                "Linia = droga/rampa, wielokąt = usyp, znacznik = koparka (albo kruszarka, gdy wybrana).")
            st.caption("Żółty = szkic (znika chwilę po narysowaniu, gdy obiekt już jest). Dodane: drogi białe, rampy fioletowe, planowane "
                       "szare, usypy niebieskie. Żeby poprawić, przesunąć lub usunąć obiekt – wybierz "
                       "„✏️ Edytuj / przesuń / usuń” i kliknij go na mapie.")

    with c2:
        # objects are pushed as a dynamic layer: the map itself is not rebuilt when they change
        objects = folium.FeatureGroup(name="Obiekty")
        if mode == EDIT:
            highlight(objects, site, st.session_state.get("sel_box"))
        draw_site(objects, site, plans=schedule is not False)
        draw_network(objects, analyze_site(site))
        out = st_folium(editing_map(site), height=620, use_container_width=True,
                        key=f"map_{st.session_state.map_key}", feature_group_to_add=objects,
                        returned_objects=["all_drawings", "last_object_clicked",
                                          "last_object_clicked_tooltip"])

    out = out or {}
    click = (out.get("last_object_clicked_tooltip"), repr(out.get("last_object_clicked")))
    if out.get("last_object_clicked") and click != st.session_state.get("last_click"):
        st.session_state.last_click = click
        sel = sel_from_click(site, click[0], out.get("last_object_clicked"))
        if sel:
            st.session_state.info_sel = sel  # properties of the clicked object (any mode)
            if mode == EDIT and sel != st.session_state.get("sel_box"):
                st.session_state.sel_next = sel
                st.session_state.edit_arm = None
            st.rerun()
    # new sketches = drawings that were not there before. (Not "last_active_drawing": the map component also
    # reports a clicked existing object as the last drawing – that would add a copy of it.)
    seen = st.session_state.setdefault("seen_drawings", set())
    fresh = [f for f in (out.get("all_drawings") or []) if f.get("geometry") and repr(f["geometry"]) not in seen]
    geom = fresh[-1] if fresh else None
    for f in fresh:
        seen.add(repr(f["geometry"]))
    if geom:
        arm = st.session_state.get("edit_arm")
        if mode != EDIT:
            msg = add_drawing(site, geom["geometry"], mode, attrs)
        elif arm:
            msg = apply_edit_drawing(site, arm[1], geom["geometry"])
            st.session_state.edit_arm = None
        else:
            msg = ("⚠️ W trybie edycji nic nie dodaję – wybierz obiekt i kliknij „📍 Przesuń” / „✏️ Przerysuj”, "
                   "albo zmień tryb na rodzaj obiektu, który chcesz dodać.")
        st.session_state.map_flash = msg
        st.session_state.obj_ver += 1  # refresh the tables below, not the map
        st.rerun()
    flash = st.session_state.pop("map_flash", None)
    if flash:
        (st.success if flash.startswith("✅") else st.warning)(flash)
    info_sel = st.session_state.get("info_sel")
    if info_sel and find_obj(site, info_sel) is not None:
        props = object_properties(site, info_sel, raw_for_build)
        with st.container(border=True):
            c1, c2, c3 = st.columns([6, 1.3, 0.7])
            c1.markdown(f"**ℹ️ {props[0][1]}** – kliknięty obiekt")
            if c2.button("📍 Przesuń / przerysuj", key="info_edit"):
                st.session_state.group_next, st.session_state.sel_next = "✏️ Edycja", info_sel
                st.rerun()
            if c3.button("✖", key="info_close", help="zamknij"):
                st.session_state.pop("info_sel", None)
                st.rerun()
            object_editor(site, info_sel, raw_for_build)
            if info_sel.startswith("road:"):
                road_points_editor(find_obj(site, info_sel), analyze_site(site))
            in_form = {_base_label(f[1]) for f in _form_fields(site, info_sel.split(":", 1)[0], {})} | {"produkt"}
            props = [props[0]] + [p for p in props[1:] if _base_label(p[0]) not in in_form]
            if len(props) > 1:
                with st.expander("Wartości wyliczone – tylko podgląd (długości, rzędne, czasy; zmieniasz je na mapie "
                                 "albo w polach powyżej)"):
                    half = len(props) // 2
                    cols = st.columns(2)
                    for col, part in zip(cols, (props[1:half + 1], props[half + 1:])):
                        if part:
                            col.dataframe(pd.DataFrame(part, columns=["właściwość", "wartość"]).astype(str),
                                          hide_index=True, width="stretch")

    st.subheader("3. Obiekty na mapie")
    st.caption("Parametry poprawiasz w tabelach (wiersz można też usunąć). Położenie i kształt – na mapie, w trybie "
               "„✏️ Edytuj / przesuń / usuń”.")
    n_pl = len(site.get("planned_piles", [])) + len(site.get("areas", []))
    names = [f"🛣️ Drogi, mijanki, kruszarki ({len(site['roads'])})", f"⛰️ Usypy i uziarnienie ({len(site['piles'])})",
             f"⛏️ Koparki i podstawianie ({len(site['excavators'])})"]
    if schedule is not False:
        names.append(f"📅 Plan strzałów ({n_pl})")
    tabs = st.tabs(names)
    t_road, t_pile, t_ex = tabs[:3]
    t_ev = t_ex
    t_plan = tabs[3] if len(tabs) > 3 else None
    road_levels = {n: lv["name"] for lv in site.get("levels") or [] for n in lv.get("roads", [])}
    for r in site["roads"]:
        r.pop("level_pick", None)  # a table helper, never stored
    net = analyze_site(site)
    with t_road:
        crushers_table(site, raw_for_build)
        if site["roads"]:
            def shown_level(r) -> str:
                z = r.get("inlet_elev_m")
                if r.get("kind") == "rampa" or abs(float(r.get("grade_draw_pct") or 0.0)) >= 0.05:
                    lv = next((x for x in site.get("levels") or [] if z not in (None, "")
                               and abs(float(x.get("elev_m") or 0) - float(z)) < 0.05), None)
                    return lv["name"] if lv else FROM_RAMP
                return road_levels.get(r.get("name") or r.get("id"), FROM_RAMP)
            shown = {r["uid"]: shown_level(r) for r in site["roads"]}
            object_table(site, "roads", lambda: pd.DataFrame(
                [{k: v for k, v in r.items() if k not in ("coords", "uid")} | {"active": r.get("active", True),
                                                                               "single_lane": bool(r.get("single_lane")),
                                                                               "inlet_elev_m": r.get("inlet_elev_m")}
                 | {"level_pick": shown[r["uid"]],
                    "rzędna początku [m]": _node_elev(net, r["coords"][0]),
                    "rzędna końca [m]": _node_elev(net, r["coords"][-1]),
                    "długość [m]": round(polyline_length_m(r["coords"]))} for r in site["roads"]]),
                extra=("active", "single_lane", "inlet_elev_m", "level_pick"),
                disabled=["id", "długość [m]", "rzędna początku [m]", "rzędna końca [m]"],
                rerun_on=("inlet_elev_m", "grade_draw_pct", "active", "kind"),
                column_config={**ROAD_COLS, "inlet_elev_m": st.column_config.NumberColumn(
                    "rzędna wlotu [m]", help="pochylnia: rzędna pierwszego narysowanego punktu (wlotu). Droga "
                                             "płaska: zwykle puste – jej rzędna wynika ze styku z pochylnią; wpisz "
                                             "tylko dla drogi bez pochylni (np. poziom górny). Puste = z sieci."),
                               "level_pick": st.column_config.SelectboxColumn(
                                   "poziom", options=[FROM_RAMP] + level_names(site),
                                   help="droga płaska: wybierz poziom – droga dostanie jego rzędną. "
                                        f"„{FROM_RAMP}” = rzędna ze styku z pochylnią (kasuje wpisaną). "
                                        "Pochylnia: poziom jej wlotu."),
                               "rzędna początku [m]": st.column_config.NumberColumn(format="%.1f"),
                               "rzędna końca [m]": st.column_config.NumberColumn(format="%.1f")})
            uids, df0 = st.session_state[f"tbl_roads_{st.session_state.obj_ver}"]
            before = dict(zip(uids, df0["level_pick"]))  # what the table showed, not what levels say now
            changed = False
            for r in site["roads"]:
                pick = r.pop("level_pick", None)
                if pick is None or pick == before.get(r["uid"]):
                    continue
                r["inlet_elev_m"] = None if pick == FROM_RAMP else level_elev(site, pick)
                changed = True
            if changed:
                st.session_state.obj_ver += 1
                st.rerun()
            if net.get("warnings"):
                for w in net["warnings"]:
                    st.warning(w)
        else:
            st.caption("brak dróg")
        bays = site.get("passing_bays", [])
        st.markdown(f"**Mijanki:** {', '.join(b['id'] for b in bays) if bays else 'brak'}")
    with t_pile:
        if site["piles"]:
            max_feed = float(raw_for_build.get("crusher", {}).get("max_feed_mm", 800))
            rows = []
            for p in site["piles"]:
                row = {k: v for k, v in p.items() if k not in ("polygon", "psd", "fragmentation", "uid", "level_auto")}
                if p.get("psd"):
                    sm = summary(p["psd"], max_feed)
                    row |= {"uziarnienie": preset_of(p["psd"]), "P80 [mm]": sm["P80 [mm]"], "nadziarno [%]": sm["nadziarno [%]"]}
                row.setdefault("target_share", None)
                row["level"] = p.get("level")
                row["level_m"] = p.get("level_m")
                row["pow. [m²]"] = round(polygon_area_m2(p["polygon"]))
                rows.append(row)
            object_table(site, "piles", lambda: pd.DataFrame(rows), extra=("target_share", "level"), rerun_on=("level",),
                         disabled=["pow. [m²]", "uziarnienie", "P80 [mm]", "nadziarno [%]", "level_m"],
                         column_config={**PILE_COLS, "level": level_col(site),
                                        "level_m": st.column_config.NumberColumn("rzędna [m]", format="%.1f")})
            psd_editor(site, max_feed, float(raw_for_build.get("crusher", {}).get("fragmentation_penalty", 0.4)))
        else:
            st.caption("brak usypów")
    with t_ex:
        if site["excavators"]:
            object_table(site, "excavators", lambda: pd.DataFrame(
                [{k: v for k, v in e.items() if k not in ("uid", "spotting", "dig_path")}
                 | {"relocate_every_t": e.get("relocate_every_t") if e.get("dig_path") else None,
                    "relocate_min": e.get("relocate_min") if e.get("dig_path") else None}
                 for e in site["excavators"]]),
                extra=("relocate_every_t", "relocate_min"),
                disabled=["lat", "lon"],
                column_config={**EX_COLS, "pile": st.column_config.SelectboxColumn(
                    "ładuje z usypu", options=[p["id"] for p in site["piles"]] + ["(najbliższy)"])})
            st.caption("Kolumny „przestawienie” działają, gdy koparka ma narysowany kierunek urabiania.")
        else:
            st.caption("brak koparek")
        spotting_editor(site, st.session_state.raw)
    with t_ev:
        items = [f"miejsce postoju: {p['excavator']}" for p in site.get("parking", [])]
        if items:
            st.markdown("**Miejsca postoju koparek** (zjazd na strzelanie / tankowanie)  \n"
                        + "  \n".join(f"• {x}" for x in items))
            rm = st.multiselect("Usuń", items, key=f"rm_{st.session_state.obj_ver}")
            if rm and st.button("🗑️ Usuń zaznaczone", key=f"rmb_{st.session_state.obj_ver}"):
                site["parking"] = [p for p in site.get("parking", []) if f"miejsce postoju: {p['excavator']}" not in rm]
                st.session_state.obj_ver += 1
                st.rerun()
    if len(tabs) > 3:
        with t_plan:
            st.markdown("**🟧 Obszary wydobycia** – program sam ustala terminy strzałów (moduł „Planowanie produkcji”).")
            if site.get("areas"):
                object_table(site, "areas", lambda: pd.DataFrame(
                    [{"id": a["id"], "level": a.get("level"), "level_m": a.get("level_m"), "material": a.get("material", ""),
                  "reserves_t": a.get("reserves_t", 0.0),
                      "blast_max_t": a.get("blast_max_t", 20000.0), "reblast_below_t": a.get("reblast_below_t", 0.0),
                      "min_gap_days": int(a.get("min_gap_days", 1)), "max_per_day": int(a.get("max_per_day", 1)),
                      "target_share": a.get("target_share"), "t_per_m": a.get("t_per_m", 50.0),
                      "first_drilled": bool(a.get("first_drilled", True)), **{k: a.get(k) for k in TIME_COLS},
                      "pow. [m²]": round(polygon_area_m2(a["polygon"]))} for a in site["areas"]]),
                    extra=("target_share", "t_per_m", "first_drilled", "level", "level_m"), disabled=["pow. [m²]"],
                    rerun_on=("level",),
                    column_config={"id": "obszar", "material": st.column_config.SelectboxColumn("materiał", options=MATERIALS),
                                   "level": level_col(site),
                                   "level_m": st.column_config.NumberColumn(
                                       "rzędna [m]", help="z wybranego poziomu; bez poziomu – wpisz albo zostaw "
                                                          "puste (odczyt z dróg)"),
                                   "reserves_t": st.column_config.NumberColumn("zasoby [t]", min_value=0.0),
                                   "blast_max_t": st.column_config.NumberColumn("największy strzał [t]", min_value=1000.0),
                                   "reblast_below_t": st.column_config.NumberColumn("strzelaj poniżej [t]", min_value=0.0),
                                   "min_gap_days": st.column_config.NumberColumn("odstęp [dni rob.]", min_value=1, step=1),
                                   "max_per_day": st.column_config.NumberColumn("strzałów / dzień", min_value=1, step=1),
                                   "target_share": st.column_config.NumberColumn("udział w nadawie [%]", min_value=0.0,
                                                                                 max_value=100.0),
                                   "t_per_m": st.column_config.NumberColumn("t/mb", min_value=5.0),
                                   "first_drilled": st.column_config.CheckboxColumn("1. pas nawiercony"), **TIME_COLS})
            else:
                st.caption("brak – narysuj trybem „Obszar wydobycia (strzały automatycznie)”")
            st.markdown("**🧨 Planowane odstrzały** – usypy, które dopiero będą odstrzelone (dzień i godziny strzału).")
            if site.get("planned_piles"):
                object_table(site, "planned_piles", lambda: pd.DataFrame(
                    [{"id": p["id"], "material": p.get("material", ""), "level": p.get("level"),
                      "tonnage_t": p.get("tonnage_t", 0.0),
                      "blast_day": int(p.get("blast_day", 1)), "target_share": p.get("target_share"),
                      "t_per_m": p.get("t_per_m", 50.0), "drilled": bool(p.get("drilled")),
                      **{k: p.get(k) for k in TIME_COLS},
                      "uziarnienie": preset_of(p["psd"]) if p.get("psd") else "–"}
                     for p in site["planned_piles"]]),
                    extra=("target_share", "t_per_m", "drilled", "level", *TIME_COLS), disabled=["uziarnienie"],
                    rerun_on=("level",),
                    column_config={"id": "strzał", "material": st.column_config.SelectboxColumn("materiał", options=MATERIALS),
                                   "level": level_col(site),
                                   "tonnage_t": st.column_config.NumberColumn("ilość [t]", min_value=0.0),
                                   "blast_day": st.column_config.NumberColumn("dzień roboczy strzału", min_value=1, step=1),
                                   "target_share": st.column_config.NumberColumn(
                                       "udział w nadawie [%]", min_value=0.0, max_value=100.0,
                                       help="ile procent nadawy ma iść z tego usypu, gdy już jest odstrzelony"),
                               "t_per_m": st.column_config.NumberColumn("t/mb", min_value=5.0),
                               "drilled": st.column_config.CheckboxColumn("nawiercony"), **TIME_COLS})
            else:
                st.caption("brak – narysuj usyp ze stanem „planowany odstrzał”")

    st.subheader("4. Sieć dróg, poziomy i nachylenia")
    st.caption("Rzędne liczone od kruszarki wzdłuż nachyleń dróg – także dla odgałęzień ze środka rampy. "
               "Biały punkt = węzeł (skrzyżowanie), czerwony = ślepy koniec drogi.")
    c1, c2 = st.columns([1, 3])
    if site.get("crusher"):
        from kruszyk.site import crusher_fixed

        fixed = crusher_fixed(site["crusher"])
        elev = c1.number_input("Rzędna kruszarki [m]", -500.0, 5000.0, float(site["crusher"].get("elev_m", 0.0)), 1.0,
                               key=f"cr_elev_{st.session_state.obj_ver}",
                               help="Puste pole wlotu pochylni = rzędne liczone od kruszarki. Gdy pochylnie mają "
                                    "rzędne wlotu, rzędna kruszarki liczy się z nich – chyba że wpiszesz ją tu ręcznie.")
        if elev != float(site["crusher"].get("elev_m", 0.0)):
            site["crusher"]["elev_m"], site["crusher"]["elev_set"] = elev, True
            st.session_state.obj_ver += 1
            st.rerun()
        if fixed and site["crusher"].get("elev_set") and c1.button("↺ Licz z wlotów pochylni"):
            site["crusher"].pop("elev_set", None)
            st.session_state.obj_ver += 1
            st.rerun()
        c1.caption("✍️ ustawiona ręcznie" if fixed else "🔗 z wlotów pochylni / sieci dróg")
    info = analyze_site(site)
    with c2:
        if info["graph"] is not None:
            m = st.columns(3)
            m[0].metric("Węzły (skrzyżowania)", len(info["junctions"]))
            m[1].metric("Ślepe końce", len(info["dead_ends"]))
            m[2].metric("Długość dróg", f"{sum(polyline_length_m(r['coords']) for r in site['roads']):,.0f} m".replace(",", " "))
        for w in info["warnings"]:
            st.warning(w)
        if info["levels"]:
            pile_lvl = {p["id"]: p for p in site.get("piles", [])}
            rows = []
            for k_, v in info["levels"].items():
                ex = next((e for e in site["excavators"] if e["id"] == k_), {})
                p = pile_lvl.get(ex.get("pile"), {})
                row = {"koparka": k_, "rzędna z dróg [m]": v["elev_m"], "odległość od drogi [m]": v["dist_to_road_m"],
                       "usyp": ex.get("pile"), "poziom usypu": p.get("level") or "–", "rzędna usypu [m]": p.get("level_m")}
                if p.get("level_m") is not None and v["elev_m"] is not None and abs(p["level_m"] - v["elev_m"]) > 3:
                    st.warning(f"Koparka {k_}: usyp {ex.get('pile')} jest na poziomie {p['level']} ({p['level_m']:+g} m), "
                               f"a drogi dochodzą tam na {v['elev_m']:+g} m – sprawdź poziomy dróg / nachylenia.")
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if simple:
        quick_fleet_crusher(raw_for_build, site)

    st.subheader("6. Zapisz kopalnię" if simple else "5. Trasy z mapy → scenariusz")
    st.caption("Program wyznacza najkrótszą trasę od każdej koparki do kruszarki po narysowanych drogach, "
               "liczy długości i nachylenia odcinków. Po każdej zmianie na mapie kliknij przycisk ponownie.")
    if _plain(site) != _plain(raw_for_build.get("site") or {}) and (site.get("roads") or site.get("piles") or site.get("excavators")):
        st.warning("⚠️ Na mapie są zmiany, których jeszcze nie zapisano – kliknij przycisk poniżej.")
    c1, c2 = st.columns([1, 3])
    if c1.button("💾 Zapisz kopalnię" if simple else "🔁 Zbuduj trasy i przenieś do scenariusza", type="primary"):
        try:
            base = copy.deepcopy(raw_for_build)
            if st.session_state.raw.get("spotting"):  # edited above, kept in the session scenario
                base["spotting"] = copy.deepcopy(st.session_state.raw["spotting"])
            new = site_to_scenario(copy.deepcopy(site), base)
            set_raw(new)
            st.session_state.map_msg = True
            st.rerun()
        except ValueError as e:
            c2.error(str(e))
    if st.session_state.pop("map_msg", None):
        st.success("✅ Kopalnia zapisana. Dalej: **② Plan zmiany (reguły)**, **③ Organizacja zmiany** i **④ Kalibracja**." if simple else
                   "Scenariusz zaktualizowany z mapy – zobacz zakładki „Scenariusz” i „Wyniki zmiany”.")
    if raw_for_build.get("site") and any("path" in r for r in raw_for_build["routes"].values()):
        from kruszyk.physics import curve_speed_kmh

        rows = [{"trasa": rid, "odcinek": s["name"], "długość [m]": s["length_m"],
                 "nachylenie w stronę kruszarki [%]": s["grade_pct"],
                 "limit [km/h]": s.get("speed_limit_kmh"),
                 "łuki (R → km/h)": ", ".join(f"{r:.0f} m → {curve_speed_kmh(r):.0f}" for r, _ in s.get("curves") or [])}
                for rid, r in raw_for_build["routes"].items() for s in r["segments"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def spotting_editor(site: dict, raw: dict) -> None:
    """How trucks spot at each excavator (shares of the four ways) and how long the manoeuvres take."""
    st.markdown("#### 🚚 Podstawianie wozideł pod koparki")
    st.caption("Jak wozidła ustawiają się pod koparką i jak odjeżdżają. Podaj, jaki **procent** wozideł robi to danym "
               "sposobem (nie musi się sumować do 100 – liczą się proporcje). Puste = jeden stały czas podstawienia. "
               "**Dwustronnie**: następne wozidło cofa z drugiej strony, gdy poprzednie jest ładowane – koparka "
               "prawie nie czeka.")
    if not site["excavators"]:
        st.caption("brak koparek")
        return
    modes = list(SPOT_MODES)
    ver = st.session_state.obj_ver
    snap = f"spot_tbl_{ver}"
    if snap not in st.session_state:
        st.session_state[snap] = ([e["uid"] for e in site["excavators"]], pd.DataFrame(
            [{"koparka": e["id"], **{m: (e.get("spotting") or {}).get(m) for m in modes}} for e in site["excavators"]]))
    uids, df0 = st.session_state[snap]
    ed = st.data_editor(df0, key=f"spot_{ver}", hide_index=True, width="stretch", disabled=["koparka"],
                        column_config={m: st.column_config.NumberColumn(f"{SPOT_MODES[m]['short']} [%]", help=SPOT_MODES[m]["label"], min_value=0.0,
                                                                        max_value=100.0, step=5.0) for m in modes})
    by_uid = {e["uid"]: e for e in site["excavators"]}
    for i, row in ed.iterrows():
        e = by_uid.get(uids[i]) if i < len(uids) else None
        if e is None:
            continue
        mix = {m: float(row[m]) for m in modes if not pd.isna(row[m]) and float(row[m]) > 0}
        if mix:
            e["spotting"] = mix
        else:
            e.pop("spotting", None)
    sp = raw.setdefault("spotting", {})
    times = sp.setdefault("times", {})
    with st.expander("⏱️ Czasy manewrów (dla całej kopalni)"):
        k = f"spot_times_{st.session_state.get('ver', 0)}"
        if k + "_base" not in st.session_state:
            st.session_state[k + "_base"] = pd.DataFrame(
                [{"sposób": SPOT_MODES[m]["label"], **{f: {**SPOT_MODES[m], **(times.get(m) or {})}[f]
                                                       for f in ("in_s", "out_s", "extra_m")}} for m in modes])
        ted = st.data_editor(st.session_state[k + "_base"], key=k, hide_index=True, width="stretch", disabled=["sposób"],
                             column_config={"in_s": st.column_config.NumberColumn("wjazd / ustawienie [s]", min_value=0.0),
                                            "out_s": st.column_config.NumberColumn("wyjazd [s]", min_value=0.0),
                                            "extra_m": st.column_config.NumberColumn("dodatkowy objazd [m]", min_value=0.0)})
        for m, (_, row) in zip(modes, ted.iterrows()):
            diff = {f: float(row[f]) for f in ("in_s", "out_s", "extra_m")
                    if not pd.isna(row[f]) and float(row[f]) != SPOT_MODES[m][f]}
            if diff:
                times[m] = diff
            else:
                times.pop(m, None)
        sp["floor_speed_kmh"] = st.number_input("Prędkość na placu / spągu przy froncie [km/h]", 3.0, 30.0,
                                                float(sp.get("floor_speed_kmh", 10.0)), 1.0, key=k + "_v",
                                                help="dla dodatkowego objazdu (rondka) przy koparce")
        st.caption("Koparka czeka na wozidło: przy podstawianiu z jednej strony – cały wjazd (oraz wyjazd, gdy "
                   "wyjeżdża cofaniem albo tyłem z jednej strony); dwustronnie – tylko gdy nikt nie stoi po drugiej "
                   "stronie. Objazd (rondko) wydłuża tylko czas wozidła.")


def crushers_table(site: dict, raw: dict) -> None:
    """All crushers: capacity, hopper, what they produce, plan per shift and which piles they take material from."""
    st.markdown("**🏭 Kruszarki**")
    if not site.get("crusher"):
        st.caption("❌ brak – postaw kruszarkę na mapie (grupa „🛣️ Drogi” → Kruszarka)")
        return
    main, cr = site["crusher"], raw.setdefault("crusher", {})
    rows = [{"id": main.get("id", "C-1"), "rola": "główna", "capacity_tph": float(cr.get("capacity_tph", 1000)),
             "hopper_t": float(cr.get("hopper_t", 150)), "product": main.get("product") or None,
             "kwh_per_t": float(cr.get("kwh_per_t", 1.1)),
             "target_t": float(main.get("target_t") or 0.0), "accepts": list(main.get("accepts") or [])}]
    rows += [{"id": c["id"], "rola": "dodatkowa", "capacity_tph": float(c.get("capacity_tph", 800)),
              "hopper_t": float(c.get("hopper_t", 150)), "product": c.get("product") or None,
              "kwh_per_t": float(c.get("kwh_per_t") or cr.get("kwh_per_t", 1.1)),
              "target_t": float(c.get("target_t") or 0.0), "accepts": list(c.get("accepts") or [])}
             for c in site.get("crushers", [])]
    from kruszyk.config import Reliability

    rel_raw = st.session_state.raw.setdefault("reliability", {})
    rel = Reliability(**{k_: v_ for k_, v_ in rel_raw.items() if k_ in Reliability.__dataclass_fields__})
    for r in rows:
        r["fail_n"], r["fail_h"] = rel.values(r["id"], "crusher")[:2]
    sig = abs(hash(repr(sorted((k_, tuple(sorted(v_.items()))) for k_, v_ in rel.machines.items())))) % 10**8
    snap = f"crush_tbl_{st.session_state.obj_ver}_{st.session_state.get('ver', 0)}_{sig}"
    if snap not in st.session_state:
        st.session_state[snap] = pd.DataFrame(rows)
    ed = st.data_editor(st.session_state[snap], key=snap + "_ed", hide_index=True, width="stretch", disabled=["rola"],
                        column_config={
                            "id": "kruszarka", "rola": "rola",
                            "capacity_tph": st.column_config.NumberColumn("wydajność [t/h]", min_value=50.0),
                            "hopper_t": st.column_config.NumberColumn("kosz [t]", min_value=10.0),
                            "product": st.column_config.SelectboxColumn(
                                "co produkuje", options=list(PRODUCTS),
                                help="wybór produktu ustawia typową energochłonność (kWh/t) – możesz ją poprawić"),
                            "kwh_per_t": st.column_config.NumberColumn(
                                "energia [kWh/t]", min_value=0.1, max_value=10.0, step=0.1, format="%.2f",
                                help="zużycie energii na tonę przy dobrym rozdrobnieniu; grube bryły podnoszą je"),
                            "target_t": st.column_config.NumberColumn("plan na zmianę [t]", min_value=0.0),
                            "accepts": st.column_config.MultiselectColumn(
                                "przyjmuje z usypów (puste = wszystkie)", options=_pile_ids(site)),
                            "fail_n": st.column_config.NumberColumn("awarii / mies.", min_value=0.0,
                                                                    help="średnio z 3–6 miesięcy, postoje > 15 min"),
                            "fail_h": st.column_config.NumberColumn("godzin awarii / mies.", min_value=0.0)})
    start = st.session_state[snap]
    for i, r in ed.iterrows():  # breakdowns: only cells changed here (other tables keep what they set)
        if i < len(start) and not pd.isna(r["fail_n"]) and not pd.isna(r["fail_h"]):
            if (float(r["fail_n"]), float(r["fail_h"])) != (float(start["fail_n"].iloc[i]), float(start["fail_h"].iloc[i])):
                rel_raw.setdefault("machines", {})[str(r["id"])] = {"count": float(r["fail_n"]), "hours": float(r["fail_h"])}
    refresh = False
    for i, r in ed.iterrows():
        tgt = main if i == 0 else (site["crushers"][i - 1] if i - 1 < len(site.get("crushers", [])) else None)
        if tgt is None:
            continue
        acc = [a for a in (r["accepts"] if isinstance(r["accepts"], list) else []) if a]
        product = "" if r["product"] is None or pd.isna(r["product"]) else str(r["product"])
        kwh = float(r["kwh_per_t"]) if not pd.isna(r["kwh_per_t"]) else None
        if product != (tgt.get("product") or "") and product in PRODUCTS:
            kwh = PRODUCTS[product]  # another product chosen -> its typical energy
            refresh = True
        if kwh:
            if i == 0:
                cr["kwh_per_t"] = kwh
            else:
                tgt["kwh_per_t"] = kwh
        new = {"product": product, "target_t": float(r["target_t"] or 0.0), "accepts": acc,
               "id": str(r["id"] or "").strip() or tgt.get("id", "C-1")}
        defaults = {"product": "", "target_t": 0.0, "accepts": [], "id": "C-1"}
        for k_, v_ in new.items():  # only real changes (so a fresh mine doesn't look edited)
            if tgt.get(k_, defaults[k_]) != v_:
                tgt[k_] = v_
        if i == 0:  # the main crusher's parameters live in the scenario
            cr["capacity_tph"], cr["hopper_t"] = float(r["capacity_tph"]), float(r["hopper_t"])
        else:
            tgt["capacity_tph"], tgt["hopper_t"] = float(r["capacity_tph"]), float(r["hopper_t"])
    if refresh:  # show the energy of the newly chosen product
        st.session_state.pop(snap, None)
        st.session_state.obj_ver += 1
        st.rerun()
    st.caption("Wozidło wiezie urobek do kruszarki, która go przyjmuje; gdy przyjmuje kilka – dziś do najbliższej, "
               "przy przydziale systemowym / RL do tej, gdzie szybciej wyładuje. Kolejną kruszarkę dodasz na mapie: "
               "grupa „🛣️ Drogi” → Kruszarka → „Dodaj kolejną”.")


def object_table(site: dict, key: str, make_df, extra: tuple = (), rerun_on: tuple = (), **editor_kw) -> None:
    """Editable table of site objects. The editor works on a fixed snapshot (per object version) and every
    row is tied to its object by a hidden id – so deleting/editing a row always changes the right object."""
    ver = st.session_state.obj_ver
    snap_key = f"tbl_{key}_{ver}"
    if snap_key not in st.session_state:
        st.session_state[snap_key] = ([o["uid"] for o in site[key]], make_df())
    uids, df0 = st.session_state[snap_key]
    ed = st.data_editor(df0, num_rows="dynamic", key=f"{key}_{ver}", hide_index=True, width="stretch", **editor_kw)
    readonly = set(editor_kw.get("disabled") or [])
    by_uid = {o["uid"]: o for o in site[key]}
    before = {u: tuple(o.get(f) for f in rerun_on) for u, o in by_uid.items()}
    kept = set()
    for idx, row in ed.iterrows():
        if not isinstance(idx, (int, np.integer)) or idx >= len(uids):
            continue  # rows added in the table are ignored – geometry comes from the map
        obj = by_uid.get(uids[idx])
        if obj is None:
            continue
        kept.add(uids[idx])
        for col, val in row.items():
            if col in readonly or col in ("lat", "lon") or (col not in obj and col not in extra):
                continue
            if isinstance(val, (list, dict)) or not pd.isna(val):
                val = val.item() if hasattr(val, "item") else val
                if col == "id" and val != obj.get("id"):
                    rename_refs(site, key, obj.get("id"), val)
                obj[col] = val
            elif col in extra:
                obj[col] = None
    # rows deleted in the table disappear; objects added after the snapshot stay
    site[key] = [o for o in site[key] if o["uid"] in kept or o["uid"] not in uids]
    if rerun_on and any(tuple(o.get(f) for f in rerun_on) != before.get(o["uid"]) for o in site[key]):
        st.session_state.obj_ver += 1  # heights / levels changed: recompute the computed columns
        st.rerun()


def render_live_map(sc, snap: pd.DataFrame, colors: dict, labels: dict) -> None:
    """Truck positions on the satellite map (only when routes come from the map)."""
    if not sc.site or not any(r.path for r in sc.routes.values()):
        return
    m = base_map(sc.site)
    draw_site(m, sc.site)
    for r in sc.routes.values():
        if r.path:
            folium.PolyLine([tuple(p) for p in r.path], color="#ffffff", weight=2, dash_array="4").add_to(m)
    if len(snap) and "lat" in snap:
        for row in snap.dropna(subset=["lat"]).itertuples():
            folium.CircleMarker((row.lat, row.lon), radius=9, color="#000", weight=1, fill=True, fill_opacity=0.95,
                                fill_color=colors.get(row.state, "#888"),
                                tooltip=f"{row.truck}: {labels.get(row.state, row.state)}").add_to(m)
            folium.Marker((row.lat, row.lon), icon=folium.DivIcon(
                html=f"<div style='color:#fff;font-size:11px;font-weight:700;text-shadow:0 0 3px #000;"
                     f"transform:translate(10px,-18px)'>{row.truck}</div>")).add_to(m)
    st_folium(m, height=520, use_container_width=True, returned_objects=[], key="live_map")
