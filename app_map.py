"""Map screen: calibrate the site, draw roads/ramps, piles, excavators and the crusher."""
from __future__ import annotations

import copy

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from kruszyk.config import SPOT_MODES
from kruszyk.psd import DEFAULT_PRESET, PRESETS, SIEVES_MM, clean, preset_of, rosin_rammler, summary
from kruszyk.site import (SNAP_M, along_m, analyze_site, normalize_roads, centroid, haversine_m, parse_coords,
                          point_along, polygon_area_m2, polyline_length_m, site_to_scenario)

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
MATERIALS = ["wapień", "dolomit", "piaskowiec", "granit", "bazalt", "melafir", "żwir", "piasek", "nadkład", "inny"]
ROAD_COLORS = {"rampa": "#d500f9", "droga": "#ffffff"}  # good contrast on light quarry imagery
PLANNED_COLOR = "#9e9e9e"
SKETCH_COLOR = "#ffd600"
MODES = {
    "Droga / rampa": "polyline",
    "Usyp (materiał)": "polygon",
    "Planowany strzał (nowy usyp)": "polygon",
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
    "🧨 Strzelanie i objazdy": ["Miejsce strzelania", "Trasa dedykowana"],
    "📅 Plan strzałów": ["Planowany strzał (nowy usyp)"],
    "✏️ Edycja": [EDIT],
}
DEDICATED_FOR = {"ekipa": "🧨 ekipa strzałowa", "sztygar": "👷 sztygar", "cysterna": "⛽ cysterna"}
DIRECTIONS = {"⬇ w dół": -1, "⬆ pod górę": 1, "➡ płasko": 0}


def empty_site() -> dict:
    return {"center": [52.0, 19.4], "zoom": 6, "crusher": None, "piles": [], "excavators": [], "roads": []}


def ensure_uids(site: dict) -> None:
    """Every road / pile / excavator gets a stable hidden id, so tables and edits always hit the right object
    (also after deleting or reordering). Duplicate road ids are renamed."""
    import uuid

    seen = set()
    for key in ("roads", "piles", "excavators", "passing_bays", "planned_piles", "crushers"):
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


DEFAULT_FLAGS = {"active": True, "single_lane": False}


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
    folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satelita", max_zoom=21, max_native_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Mapa", max_zoom=21, max_native_zoom=19).add_to(m)
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


def draw_site(m, site: dict) -> None:
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
    for p in site.get("planned_piles", []):
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
    if g["type"] == "Polygon" and mode == "Planowany strzał (nowy usyp)":
        site.setdefault("planned_piles", [])
        ids = {p["id"] for p in site["piles"]} | {p["id"] for p in site["planned_piles"]}
        a = dict(attrs)
        if not a.get("id") or a["id"] in ids:
            a["id"] = next(f"S-{i:02d}" for i in range(1, 1000) if f"S-{i:02d}" not in ids)
        site["planned_piles"].append({**a, "polygon": _latlon(g["coordinates"][0][:-1])})
        ensure_uids(site)
        return (f"✅ Planowany strzał {a['id']} w dniu roboczym {a.get('blast_day', 1)} – w kroku „📅 Tydzień / "
                f"miesiąc” koparka bez urobku sama tu przejedzie")
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
    a = dict(attrs) if target == mode else dict(DEFAULTS.get(target, {}))
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
                                     "product": attrs.get("product", ""), "target_t": attrs.get("target_t", 0.0)})
            ensure_uids(site)
            return f"✅ Dodano kruszarkę {cid} – zbuduj trasy (zapisz kopalnię), żeby wozidła do niej jeździły"
        site["crusher"] = {**(site.get("crusher") or {}), "lat": lat, "lon": lon}
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
             "⇆": ("bay", "passing_bays"), "📅": ("planned", "planned_piles")}
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
        elif kind in ("pile", "planned"):
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
        return f"✅ Przerysowano ({polyline_length_m(o['coords']):.0f} m)"
    if g["type"] != "Polygon":
        return "⚠️ Aby przerysować usyp, narysuj wielokąt (⬠)"
    o["polygon"] = _latlon(g["coordinates"][0][:-1])
    return f"✅ Przerysowano usyp {o['id']}"


def highlight(m, site: dict, sel: str | None) -> None:
    """Yellow halo around the selected object (drawn under it)."""
    o = find_obj(site, sel)
    if o is None:
        return
    kind = sel.split(":", 1)[0]
    if kind in POINT_KINDS:
        folium.CircleMarker((o["lat"], o["lon"]), radius=22, color=SKETCH_COLOR, weight=4, fill=True,
                            fill_opacity=0.15).add_to(m)
    elif kind in ("pile", "planned"):
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
                "redraw": "narysuj od nowa **wielokąt (⬠)**" if kind in ("pile", "planned") else "narysuj od nowa **linię (/)**"}
        st.info(f"Teraz {tool[arm[0]]} – obiekt przeniesie się tam sam.")
        if st.button("Anuluj", key=f"ecn_{k}"):
            st.session_state.edit_arm = None
            st.rerun()


def set_dig_path(ex: dict, coords: list, attrs: dict | None = None) -> None:
    """Digging line of an excavator: starts at the end nearer to the excavator, which is moved there."""
    pos = (ex["lat"], ex["lon"])
    if haversine_m(coords[-1], pos) < haversine_m(coords[0], pos):
        coords = list(reversed(coords))
    ex["dig_path"] = coords
    ex["lat"], ex["lon"] = coords[0]
    for key, default in (("relocate_every_t", 600.0), ("relocate_min", 4.0)):
        ex[key] = float((attrs or {}).get(key) or ex.get(key) or default)


def _pile_ids(site: dict) -> list[str]:
    return [p["id"] for p in site.get("piles", [])] + [p["id"] for p in site.get("planned_piles", [])]


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
    folium.LayerControl().add_to(m)
    return m


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


def quick_fleet_crusher(raw: dict, site: dict) -> None:
    """Trucks and crusher edited right next to the map (simple mode); changes go straight to the scenario."""
    st.subheader("6. Wozidła")
    st.caption("Dodaj wiersz na każdy typ wozideł. **Koparki**: do których koparek mogą jeździć, np. `K-1, K-2` "
               "(puste = do wszystkich).")
    ex_ids = {e["id"] for e in site.get("excavators", [])} | {e["id"] for e in raw.get("excavators", [])}
    base_key = f"qfleet_base_{st.session_state.ver}"
    if base_key not in st.session_state:  # fixed starting table: the editor applies its edits on top of it
        st.session_state[base_key] = (
            copy.deepcopy(raw.get("trucks", [])),
            pd.DataFrame([{"model": f.get("model", ""), "count": int(f.get("count", 0)),
                           "payload_t": float(f.get("payload_t", 50)), "excavators": ", ".join(f.get("excavators") or [])}
                          for f in raw.get("trucks", [])], columns=["model", "count", "payload_t", "excavators"]))
    old, start = st.session_state[base_key]
    ed = st.data_editor(start, num_rows="dynamic",
                        key=f"qfleet_{st.session_state.ver}", hide_index=True, width="stretch",
                        column_config={"model": "typ wozidła", "count": st.column_config.NumberColumn("liczba", min_value=0, step=1),
                                       "payload_t": st.column_config.NumberColumn("ładowność [t]", min_value=5.0, step=5.0),
                                       "excavators": st.column_config.TextColumn("koparki (puste = wszystkie)")})
    fleets = []
    for i, r in ed.reset_index(drop=True).iterrows():
        if pd.isna(r["payload_t"]) or not str(r["model"] or "").strip():
            continue
        payload = float(r["payload_t"])
        base = dict(old[i]) if i < len(old) else {
            # typical values scaled with payload for a new truck type
            "empty_mass_t": round(payload * 0.75, 1), "power_kw": round(payload * 9.5), "max_speed_kmh": 45,
            "fuel_lph_loaded": round(payload * 1.1), "fuel_lph_empty": round(payload * 0.58), "fuel_lph_idle": 10}
        if i < len(old) and abs(float(old[i].get("payload_t", payload)) - payload) > 1e-6:
            scale = payload / float(old[i]["payload_t"])
            base["empty_mass_t"] = round(float(base.get("empty_mass_t", payload * 0.75)) * scale, 1)
            base["power_kw"] = round(float(base.get("power_kw", payload * 9.5)) * scale)
        excs = [x.strip() for x in str(r["excavators"] or "").split(",") if x.strip() in ex_ids]
        fleets.append({**base, "model": str(r["model"]).strip(), "count": int(r["count"] or 0), "payload_t": payload,
                       "excavators": excs})
    raw["trucks"] = fleets
    st.caption("Kruszarki (wydajność, kosz, produkt, plan, z których usypów przyjmują) – w sekcji 3, zakładka "
               "„🛣️ Drogi, mijanki, kruszarki”.")


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


def render_map_tab(raw_for_build: dict, set_raw, simple: bool = False) -> None:
    site = get_site(raw_for_build)
    checklist(site, raw_for_build)

    st.subheader("1. Kalibracja – gdzie jest kopalnia")
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
    group = st.radio("Co robisz?", list(MODE_GROUPS), horizontal=True, key="draw_group",
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
            grade = st.number_input("Nachylenie [%]", 0.0, 25.0, 9.0 if attrs["kind"] == "rampa" else 1.0, 0.5,
                                    disabled=DIRECTIONS[direction] == 0,
                                    help="Rampa: zwykle 8–10 %. Droga technologiczna: 0–3 %.")
            attrs["grade_draw_pct"] = DIRECTIONS[direction] * grade + 0.0
            attrs["speed_limit_kmh"] = st.number_input("Limit prędkości [km/h]", 5.0, 80.0,
                                                       25.0 if attrs["kind"] == "rampa" else 40.0, 5.0)
            attrs["single_lane"] = st.checkbox(
                "Jeden pas – ruch wahadłowy (mijanki)",
                help="Wozidła nie miną się na tej drodze. Postaw na niej mijanki (tryb „Mijanka”). Pierwszeństwo ma "
                     "wozidło jadące pod górę – jadące w dół czeka na najbliższej mijance.")
            attrs["active"] = not st.checkbox("Droga planowana (dziś nieużywana)",
                                              help="Narysuj ją teraz, a włączysz w kroku „Wariant – co jeśli”.")
            st.caption("Narzędzie „linia” (/). Rysuj po osi drogi, zakończ klikając ostatni punkt. "
                       "Końce dróg, które się łączą, stawiaj blisko siebie (≤ 30 m).")
        elif mode == "Usyp (materiał)":
            attrs["id"] = st.text_input("Oznaczenie usypu", f"U-{len(site['piles']) + 1:02d}")
            attrs["material"] = st.selectbox("Materiał", MATERIALS)
            attrs["tonnage_t"] = st.number_input("Ilość [t]", 0.0, 10_000_000.0, 10_000.0, 500.0)
            preset = st.selectbox("Uziarnienie (krzywą można potem dokładnie ustawić)", list(PRESETS),
                                  index=list(PRESETS).index(DEFAULT_PRESET))
            attrs["psd"] = PRESETS[preset]
            attrs["loose_density_t_m3"] = st.number_input("Gęstość spulchniona [t/m³]", 1.0, 3.0, 1.75, 0.05)
            share = st.number_input("Udział w nadawie [%] (0 = bez celu)", 0.0, 100.0, 0.0, 5.0,
                                    help="Oczekiwany procent nadawy z tego usypu. Steruje nim optymalizacja – tylko "
                                         "dla wozideł nieprzypisanych na sztywno do jednej koparki.")
            if share > 0:
                attrs["target_share"] = share
            st.caption("Narzędzie „wielokąt” (⬠). Obrysuj usyp, zamknij klikając pierwszy punkt.")
        elif mode == "Planowany strzał (nowy usyp)":
            n_pl = len(site.get("planned_piles", []))
            attrs["id"] = st.text_input("Oznaczenie", f"S-{n_pl + 1:02d}")
            attrs["material"] = st.selectbox("Materiał", MATERIALS, key="pl_mat")
            attrs["tonnage_t"] = st.number_input("Ile ton da strzał", 0.0, 10_000_000.0, 15_000.0, 1000.0, key="pl_t")
            preset = st.selectbox("Spodziewane uziarnienie", list(PRESETS), index=list(PRESETS).index(DEFAULT_PRESET),
                                  key="pl_psd")
            attrs["psd"] = PRESETS[preset]
            attrs["loose_density_t_m3"] = 1.75
            share = st.number_input("Udział w nadawie [%] (0 = bez celu)", 0.0, 100.0, 0.0, 5.0, key="pl_share",
                                    help="ile procent nadawy ma iść z tego usypu, gdy już będzie odstrzelony")
            if share > 0:
                attrs["target_share"] = share
            attrs["blast_day"] = int(st.number_input("Strzał w dniu roboczym nr", 1, 300, 3, 1, key="pl_day",
                                                     help="licząc od pierwszego dnia okresu w kroku „📅 Tydzień / "
                                                          "miesiąc” (1 = pierwszy dzień roboczy)"))
            st.caption("Narzędzie „wielokąt” (⬠). Obrysuj miejsce przyszłego strzału (urobek po strzale). Do dnia "
                       "strzału nic tu nie ma; potem koparka, której skończył się urobek, przejedzie tu sama "
                       "(najbliższa wolna). Wozidła ładują przy rogu usypu najbliższym drogi – doprowadź tam drogę.")
        elif mode == "Koparka":
            attrs["id"] = st.text_input("Oznaczenie koparki", f"K-{len(site['excavators']) + 1}")
            piles = [p["id"] for p in site["piles"]]
            attrs["pile"] = st.selectbox("Ładuje z usypu", piles or ["(najbliższy)"])
            attrs["bucket_m3"] = st.number_input("Łyżka [m³]", 0.5, 30.0, 5.0, 0.5)
            attrs["swing_cycle_s"] = st.number_input("Cykl łyżki [s]", 10.0, 90.0, 30.0, 1.0)
            st.caption("Narzędzie „znacznik” (📍). Postaw go w miejscu pracy koparki.")
        elif mode == "Trasa dedykowana":
            ex_ids = [e["id"] for e in site["excavators"]]
            options = list(DEDICATED_FOR) + ex_ids
            attrs["for"] = st.selectbox("Dla kogo?", options, format_func=lambda x: DEDICATED_FOR.get(x, f"⛏️ przejazd koparki {x}"))
            st.caption("Narzędzie „linia” (/). Trasa, którą jeździ tylko ten pojazd. **Ekipa:** od wjazdu do miejsca "
                       "strzelania. **Sztygar / cysterna:** cała runda – postoje przy koparkach ustawią się same "
                       "w najbliższych punktach trasy. **Koparka:** od koparki do jej miejsca postoju. Bez trasy "
                       "dedykowanej pojazd jedzie najkrótszą drogą po sieci.")
        elif mode == "Miejsce postoju koparki":
            ex_ids = [e["id"] for e in site["excavators"]]
            attrs["excavator"] = st.selectbox("Której koparki?", ex_ids) if ex_ids else None
            st.caption("Narzędzie „znacznik” (📍). Tu koparka zjeżdża na czas strzelania (i tankowania, jeśli tak "
                       "ustawisz). Dojedzie trasą dedykowaną, a gdy jej nie ma – najkrótszą drogą; bez połączenia "
                       "drogowego – na wprost. Bez znacznika koparka cofa się o kilkaset metrów swoją trasą.")
        elif mode == "Kruszarka":
            if site.get("crusher"):
                attrs["new_crusher"] = st.radio("Kruszarka", ["Przesuń główną", "Dodaj kolejną"], key="cr_which",
                                                horizontal=True) == "Dodaj kolejną"
            if attrs.get("new_crusher"):
                n_cr = len(site.get("crushers", []))
                attrs["id"] = st.text_input("Oznaczenie", f"C-{n_cr + 2}", key="cr_id")
                attrs["capacity_tph"] = st.number_input("Wydajność [t/h]", 50.0, 10_000.0, 800.0, 50.0, key="cr_cap")
                attrs["hopper_t"] = st.number_input("Kosz [t]", 10.0, 2000.0, 150.0, 10.0, key="cr_hop")
                attrs["accepts"] = st.multiselect(
                    "Przyjmuje urobek z (puste = ze wszystkich)", _pile_ids(site), key="cr_acc")
                attrs["product"] = st.text_input("Co produkuje", "", key="cr_prod", placeholder="np. kruszywo 0–31,5")
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
            st.caption("Żółty = szkic (znika po odświeżeniu strony). Dodane: drogi białe, rampy fioletowe, planowane "
                       "szare, usypy niebieskie. Żeby poprawić, przesunąć lub usunąć obiekt – wybierz "
                       "„✏️ Edytuj / przesuń / usuń” i kliknij go na mapie.")

    with c2:
        # objects are pushed as a dynamic layer: the map itself is not rebuilt when they change
        objects = folium.FeatureGroup(name="Obiekty")
        if mode == EDIT:
            highlight(objects, site, st.session_state.get("sel_box"))
        draw_site(objects, site)
        draw_network(objects, analyze_site(site))
        out = st_folium(editing_map(site), height=620, use_container_width=True,
                        key=f"map_{st.session_state.map_key}", feature_group_to_add=objects,
                        returned_objects=["last_active_drawing", "last_object_clicked",
                                          "last_object_clicked_tooltip"])

    out = out or {}
    click = (out.get("last_object_clicked_tooltip"), repr(out.get("last_object_clicked")))
    if out.get("last_object_clicked") and click != st.session_state.get("last_click"):
        st.session_state.last_click = click
        if mode == EDIT:
            sel = sel_from_click(site, click[0], out.get("last_object_clicked"))
            if sel and sel != st.session_state.get("sel_box"):
                st.session_state.sel_next = sel
                st.session_state.edit_arm = None
                st.rerun()
    geom = out.get("last_active_drawing")
    if geom and geom.get("geometry"):
        sig = repr(geom["geometry"])
        if sig != st.session_state.get("last_added_geom"):
            st.session_state.last_added_geom = sig
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

    st.subheader("3. Obiekty na mapie")
    st.caption("Parametry poprawiasz w tabelach (wiersz można też usunąć). Położenie i kształt – na mapie, w trybie "
               "„✏️ Edytuj / przesuń / usuń”.")
    n_pl = len(site.get("planned_piles", []))
    t_road, t_pile, t_ex, t_ev, t_plan = st.tabs([
        f"🛣️ Drogi, mijanki, kruszarki ({len(site['roads'])})", f"⛰️ Usypy i uziarnienie ({len(site['piles'])})",
        f"⛏️ Koparki i podstawianie ({len(site['excavators'])})", "🧨 Strzelanie, objazdy, postoje",
        f"📅 Plan strzałów ({n_pl})"])
    with t_road:
        crushers_table(site, raw_for_build)
        if site["roads"]:
            object_table(site, "roads", lambda: pd.DataFrame(
                [{k: v for k, v in r.items() if k not in ("coords", "uid")} | {"active": r.get("active", True),
                                                                               "single_lane": bool(r.get("single_lane"))}
                 | {"długość [m]": round(polyline_length_m(r["coords"]))} for r in site["roads"]]),
                extra=("active", "single_lane"), disabled=["id", "długość [m]"], column_config=ROAD_COLS)
        else:
            st.caption("brak dróg")
        bays = site.get("passing_bays", [])
        st.markdown(f"**Mijanki:** {', '.join(b['id'] for b in bays) if bays else 'brak'}")
    with t_pile:
        if site["piles"]:
            max_feed = float(raw_for_build.get("crusher", {}).get("max_feed_mm", 800))
            rows = []
            for p in site["piles"]:
                row = {k: v for k, v in p.items() if k not in ("polygon", "psd", "fragmentation", "uid")}
                if p.get("psd"):
                    sm = summary(p["psd"], max_feed)
                    row |= {"uziarnienie": preset_of(p["psd"]), "P80 [mm]": sm["P80 [mm]"], "nadziarno [%]": sm["nadziarno [%]"]}
                row.setdefault("target_share", None)
                row["pow. [m²]"] = round(polygon_area_m2(p["polygon"]))
                rows.append(row)
            object_table(site, "piles", lambda: pd.DataFrame(rows), extra=("target_share",),
                         disabled=["pow. [m²]", "uziarnienie", "P80 [mm]", "nadziarno [%]"], column_config=PILE_COLS)
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
        st.markdown(f"**Miejsce strzelania:** {'✅ ustawione' if site.get('blast') else '— brak'} "
                    "(godziny strzelania, objazd sztygara i tankowanie – w sekcji 5 „Zmiana i zdarzenia”)")
        items = ([f"trasa dedykowana: {DEDICATED_FOR.get(d['for'], 'przejazd koparki ' + str(d['for']))}"
                  for d in site.get("dedicated", [])] +
                 [f"miejsce postoju: {p['excavator']}" for p in site.get("parking", [])])
        if items:
            st.markdown("**Trasy dedykowane i miejsca postoju koparek**  \n" + "  \n".join(f"• {x}" for x in items))
            rm = st.multiselect("Usuń", items, key=f"rm_{st.session_state.obj_ver}")
            if rm and st.button("🗑️ Usuń zaznaczone", key=f"rmb_{st.session_state.obj_ver}"):
                site["dedicated"] = [d for d in site.get("dedicated", [])
                                     if f"trasa dedykowana: {DEDICATED_FOR.get(d['for'], 'przejazd koparki ' + str(d['for']))}" not in rm]
                site["parking"] = [p for p in site.get("parking", []) if f"miejsce postoju: {p['excavator']}" not in rm]
                st.session_state.obj_ver += 1
                st.rerun()
        else:
            st.caption("brak tras dedykowanych i miejsc postoju")
    with t_plan:
        st.caption("Przyszłe strzały w nowych miejscach. Liczą się w kroku „📅 Tydzień / miesiąc”: w dniu strzału "
                   "pojawia się nowy usyp, a koparka bez urobku sama na niego przejeżdża. Rysujesz je trybem "
                   "„Planowany strzał (nowy usyp)”.")
        if site.get("planned_piles"):
            object_table(site, "planned_piles", lambda: pd.DataFrame(
                [{"id": p["id"], "material": p.get("material", ""), "tonnage_t": p.get("tonnage_t", 0.0),
                  "blast_day": int(p.get("blast_day", 1)), "target_share": p.get("target_share"),
                  "uziarnienie": preset_of(p["psd"]) if p.get("psd") else "–"}
                 for p in site["planned_piles"]]),
                extra=("target_share",), disabled=["uziarnienie"],
                column_config={"id": "strzał", "material": st.column_config.SelectboxColumn("materiał", options=MATERIALS),
                               "tonnage_t": st.column_config.NumberColumn("ilość [t]", min_value=0.0),
                               "blast_day": st.column_config.NumberColumn("dzień roboczy strzału", min_value=1, step=1),
                               "target_share": st.column_config.NumberColumn(
                                   "udział w nadawie [%]", min_value=0.0, max_value=100.0,
                                   help="ile procent nadawy ma iść z tego usypu, gdy już jest odstrzelony")})
        else:
            st.caption("brak planowanych strzałów")

    st.subheader("4. Sieć dróg, poziomy i nachylenia")
    st.caption("Rzędne liczone od kruszarki wzdłuż nachyleń dróg – także dla odgałęzień ze środka rampy. "
               "Biały punkt = węzeł (skrzyżowanie), czerwony = ślepy koniec drogi.")
    c1, c2 = st.columns([1, 3])
    if site.get("crusher"):
        elev = c1.number_input("Rzędna kruszarki [m]", -500.0, 2000.0, float(site["crusher"].get("elev_m", 0.0)), 1.0,
                               help="np. rzędna terenu n.p.m.; poziomy koparek liczone względem niej")
        if elev != float(site["crusher"].get("elev_m", 0.0)):
            site["crusher"]["elev_m"] = elev
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
            st.dataframe(pd.DataFrame([{"koparka": k, "poziom (rzędna) [m]": v["elev_m"],
                                        "odległość od drogi [m]": v["dist_to_road_m"]} for k, v in info["levels"].items()]),
                         hide_index=True, width="stretch")

    from app_events import events_editor

    # always edits the scenario kept in the session (in the advanced mode raw_for_build is a per-run copy)
    events_editor(st.session_state.raw, "ev", [e["id"] for e in site.get("excavators", [])], site, number="5")
    if simple:
        quick_fleet_crusher(raw_for_build, site)

    st.subheader("7. Zapisz kopalnię" if simple else "6. Trasy z mapy → scenariusz")
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
        st.success("✅ Kopalnia zapisana. Przejdź do kroku **② Jak jest dziś**." if simple else
                   "Scenariusz zaktualizowany z mapy – zobacz zakładki „Scenariusz” i „Wyniki zmiany”.")
    if raw_for_build.get("site") and any("path" in r for r in raw_for_build["routes"].values()):
        rows = [{"trasa": rid, "odcinek": s["name"], "długość [m]": s["length_m"],
                 "nachylenie w stronę kruszarki [%]": s["grade_pct"],
                 "limit [km/h]": s.get("speed_limit_kmh")}
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
             "hopper_t": float(cr.get("hopper_t", 150)), "product": main.get("product", ""),
             "target_t": float(main.get("target_t") or 0.0), "accepts": list(main.get("accepts") or [])}]
    rows += [{"id": c["id"], "rola": "dodatkowa", "capacity_tph": float(c.get("capacity_tph", 800)),
              "hopper_t": float(c.get("hopper_t", 150)), "product": c.get("product", ""),
              "target_t": float(c.get("target_t") or 0.0), "accepts": list(c.get("accepts") or [])}
             for c in site.get("crushers", [])]
    snap = f"crush_tbl_{st.session_state.obj_ver}_{st.session_state.get('ver', 0)}"
    if snap not in st.session_state:
        st.session_state[snap] = pd.DataFrame(rows)
    ed = st.data_editor(st.session_state[snap], key=snap + "_ed", hide_index=True, width="stretch", disabled=["rola"],
                        column_config={
                            "id": "kruszarka", "rola": "rola",
                            "capacity_tph": st.column_config.NumberColumn("wydajność [t/h]", min_value=50.0),
                            "hopper_t": st.column_config.NumberColumn("kosz [t]", min_value=10.0),
                            "product": st.column_config.TextColumn("co produkuje"),
                            "target_t": st.column_config.NumberColumn("plan na zmianę [t]", min_value=0.0),
                            "accepts": st.column_config.MultiselectColumn(
                                "przyjmuje z usypów (puste = wszystkie)", options=_pile_ids(site))})
    for i, r in ed.iterrows():
        tgt = main if i == 0 else (site["crushers"][i - 1] if i - 1 < len(site.get("crushers", [])) else None)
        if tgt is None:
            continue
        acc = [a for a in (r["accepts"] if isinstance(r["accepts"], list) else []) if a]
        tgt.update({"product": str(r["product"] or ""), "target_t": float(r["target_t"] or 0.0), "accepts": acc})
        if str(r["id"] or "").strip():
            tgt["id"] = str(r["id"]).strip()
        if i == 0:  # the main crusher's parameters live in the scenario
            cr["capacity_tph"], cr["hopper_t"] = float(r["capacity_tph"]), float(r["hopper_t"])
        else:
            tgt["capacity_tph"], tgt["hopper_t"] = float(r["capacity_tph"]), float(r["hopper_t"])
    st.caption("Wozidło wiezie urobek do kruszarki, która go przyjmuje; gdy przyjmuje kilka – dziś do najbliższej, "
               "przy przydziale systemowym / RL do tej, gdzie szybciej wyładuje. Kolejną kruszarkę dodasz na mapie: "
               "grupa „🛣️ Drogi” → Kruszarka → „Dodaj kolejną”.")


def object_table(site: dict, key: str, make_df, extra: tuple = (), **editor_kw) -> None:
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
