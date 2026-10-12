"""Map screen: calibrate the site, draw roads/ramps, piles, excavators and the crusher."""
from __future__ import annotations

import copy

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from kruszyk.site import (analyze_site, centroid, parse_coords, polygon_area_m2, polyline_length_m,
                          site_to_scenario)

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
MATERIALS = ["wapień", "dolomit", "piaskowiec", "granit", "bazalt", "melafir", "żwir", "piasek", "nadkład", "inny"]
ROAD_COLORS = {"rampa": "#ff8800", "droga": "#ffe600"}
MODES = {
    "Droga / rampa": "polyline",
    "Usyp (materiał)": "polygon",
    "Koparka": "marker",
    "Kruszarka": "marker",
}


def empty_site() -> dict:
    return {"center": [52.0, 19.4], "zoom": 6, "crusher": None, "piles": [], "excavators": [], "roads": []}


def get_site(raw: dict) -> dict:
    if "site" not in st.session_state or st.session_state.get("site_ver") != st.session_state.ver:
        old = st.session_state.get("site")
        new = copy.deepcopy(raw.get("site") or empty_site())
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


def draw_site(m, site: dict) -> None:
    """Add site objects to a map or a FeatureGroup."""
    for r in site.get("roads", []):
        folium.PolyLine([tuple(c) for c in r["coords"]], color=ROAD_COLORS.get(r.get("kind"), "#ffe600"), weight=6,
                        tooltip=f"{r['name']} · {r.get('kind')} · {r.get('grade_pct', 0)} % · "
                                f"{polyline_length_m(r['coords']):.0f} m").add_to(m)
    for p in site.get("piles", []):
        folium.Polygon([tuple(c) for c in p["polygon"]], color="#00e5ff", fill=True, fill_opacity=0.25,
                       tooltip=f"Usyp {p['id']}: {p.get('material', '')} · {p.get('tonnage_t', 0):,.0f} t").add_to(m)
        folium.Marker(centroid(p["polygon"]), icon=folium.DivIcon(
            html=f"<div style='color:#fff;font-weight:700;text-shadow:0 0 3px #000;white-space:nowrap'>"
                 f"{p['id']} · {p.get('material', '')}</div>")).add_to(m)
    for e in site.get("excavators", []):
        folium.Marker((e["lat"], e["lon"]), tooltip=f"Koparka {e['id']} → {e.get('pile', '?')}",
                      icon=folium.Icon(color="orange", icon="wrench", prefix="fa")).add_to(m)
    if site.get("crusher"):
        c = site["crusher"]
        folium.Marker((c["lat"], c["lon"]), tooltip="Kruszarka",
                      icon=folium.Icon(color="red", icon="industry", prefix="fa")).add_to(m)


def _latlon(coords: list) -> list:
    return [[round(c[1], 7), round(c[0], 7)] for c in coords]  # GeoJSON is [lon, lat]


DEFAULTS = {
    "Droga / rampa": {"kind": "droga", "grade_pct": 0.0, "speed_limit_kmh": 40.0},
    "Usyp (materiał)": {"material": "inny", "tonnage_t": 10_000.0, "fragmentation": 0.3, "loose_density_t_m3": 1.75},
    "Koparka": {"pile": "(najbliższy)", "bucket_m3": 5.0, "swing_cycle_s": 30.0},
}
GEOM_MODE = {"LineString": "Droga / rampa", "Polygon": "Usyp (materiał)"}


def add_drawing(site: dict, g: dict, mode: str, attrs: dict) -> str:
    """Add a freshly drawn geometry to the site. The shape decides the object type; form parameters
    are used when they belong to that type, otherwise defaults (editable later in the tables)."""
    target = GEOM_MODE.get(g["type"])
    if g["type"] == "Point":
        target = "Kruszarka" if mode == "Kruszarka" else "Koparka"
    if target is None:
        return f"⚠️ Nieobsługiwany kształt: {g['type']}"
    a = dict(attrs) if target == mode else dict(DEFAULTS.get(target, {}))
    note = "" if target == mode else " (parametry domyślne – popraw w tabeli poniżej)"
    if target == "Droga / rampa":
        a.setdefault("name", f"{a['kind']} {len(site['roads']) + 1}")
        site["roads"].append({"id": f"R{len(site['roads']) + 1}", **a, "coords": _latlon(g["coordinates"])})
        return f"✅ Dodano drogę „{a['name']}” ({polyline_length_m(site['roads'][-1]['coords']):.0f} m){note}"
    if target == "Usyp (materiał)":
        ids = {p["id"] for p in site["piles"]}
        if a.get("id") in ids or not a.get("id"):
            a["id"] = next(f"U-{i:02d}" for i in range(1, 1000) if f"U-{i:02d}" not in ids)
        site["piles"].append({**a, "polygon": _latlon(g["coordinates"][0][:-1])})
        return f"✅ Dodano usyp {a['id']}{note}"
    lon, lat = g["coordinates"]
    if target == "Kruszarka":
        site["crusher"] = {"lat": lat, "lon": lon}
        return "✅ Ustawiono kruszarkę"
    ids = {e["id"] for e in site["excavators"]}
    if a.get("id") in ids or not a.get("id"):
        a["id"] = next(f"K-{i}" for i in range(1, 1000) if f"K-{i}" not in ids)
    site["excavators"].append({**a, "lat": lat, "lon": lon})
    return f"✅ Dodano koparkę {a['id']}{note}"


def editing_map(site: dict) -> folium.Map:
    """The drawing map. Its HTML must not depend on the selected mode or on the objects,
    otherwise st_folium rebuilds it on every change and the view/sketches are lost."""
    m = base_map(site)
    Draw(export=False, position="topleft",
         draw_options={"polyline": {"shapeOptions": {"color": "#ff00ff", "weight": 4}},
                       "polygon": {"shapeOptions": {"color": "#ff00ff"}}, "marker": True,
                       "circle": False, "rectangle": False, "circlemarker": False},
         edit_options={"edit": False}).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def render_map_tab(raw_for_build: dict, set_raw) -> None:
    site = get_site(raw_for_build)

    st.subheader("1. Kalibracja – gdzie jest kopalnia")
    c1, c2, c3 = st.columns([4, 1, 1])
    link = c1.text_input("Wklej link z Google Maps albo współrzędne „szer., dł.” (np. 50.8712, 20.6283)",
                         key="calib_link")
    zoom = c2.number_input("Przybliżenie", 3, 20, int(site.get("zoom", 16)))
    if c3.button("📍 Ustaw", use_container_width=True):
        pos = parse_coords(link)
        if pos:
            site["center"], site["zoom"] = list(pos), int(zoom)
            st.session_state.map_key += 1  # the only moment the map is rebuilt on purpose
            st.rerun()
        else:
            st.error("Nie rozpoznano współrzędnych. W Google Maps: prawy klik na mapie → kliknij współrzędne (kopiują się).")

    st.subheader("2. Rysowanie obiektów")
    c1, c2 = st.columns([1, 3])
    with c1:
        mode = st.radio("Co dodajesz?", list(MODES), key="draw_mode")
        attrs: dict = {}
        if mode == "Droga / rampa":
            attrs["kind"] = st.selectbox("Rodzaj", ["droga", "rampa"])
            attrs["name"] = st.text_input("Nazwa", f"{attrs['kind']} {len(site['roads']) + 1}")
            attrs["grade_pct"] = st.number_input("Nachylenie w stronę kruszarki [%] (+ pod górę)", -20.0, 20.0,
                                                 8.0 if attrs["kind"] == "rampa" else 0.0, 0.5)
            attrs["speed_limit_kmh"] = st.number_input("Limit prędkości [km/h]", 5.0, 80.0,
                                                       25.0 if attrs["kind"] == "rampa" else 40.0, 5.0)
            st.caption("Narzędzie „linia” (/). Rysuj po osi drogi, zakończ klikając ostatni punkt. "
                       "Końce dróg, które się łączą, stawiaj blisko siebie (≤ 30 m).")
        elif mode == "Usyp (materiał)":
            attrs["id"] = st.text_input("Oznaczenie usypu", f"U-{len(site['piles']) + 1:02d}")
            attrs["material"] = st.selectbox("Materiał", MATERIALS)
            attrs["tonnage_t"] = st.number_input("Ilość [t]", 0.0, 10_000_000.0, 10_000.0, 500.0)
            attrs["fragmentation"] = st.slider("Rozdrobnienie (0 = drobne, 1 = bryły)", 0.0, 1.0, 0.3, 0.05)
            attrs["loose_density_t_m3"] = st.number_input("Gęstość spulchniona [t/m³]", 1.0, 3.0, 1.75, 0.05)
            st.caption("Narzędzie „wielokąt” (⬠). Obrysuj usyp, zamknij klikając pierwszy punkt.")
        elif mode == "Koparka":
            attrs["id"] = st.text_input("Oznaczenie koparki", f"K-{len(site['excavators']) + 1}")
            piles = [p["id"] for p in site["piles"]]
            attrs["pile"] = st.selectbox("Ładuje z usypu", piles or ["(najbliższy)"])
            attrs["bucket_m3"] = st.number_input("Łyżka [m³]", 0.5, 30.0, 5.0, 0.5)
            attrs["swing_cycle_s"] = st.number_input("Cykl łyżki [s]", 10.0, 90.0, 30.0, 1.0)
            st.caption("Narzędzie „znacznik” (📍). Postaw go w miejscu pracy koparki.")
        else:
            st.caption("Narzędzie „znacznik” (📍). Postaw go na koszu kruszarki (miejsce wyładunku).")
        st.info("Wybierz typ i parametry, potem narysuj obiekt – **dodaje się sam**. "
                "Linia = droga/rampa, wielokąt = usyp, znacznik = koparka (albo kruszarka, gdy wybrana).")
        st.caption("Różowy = szkic (znika po odświeżeniu strony). Dodane: drogi żółte, rampy pomarańczowe, "
                   "usypy niebieskie. Parametry można poprawić w tabelach poniżej.")

    with c2:
        # objects are pushed as a dynamic layer: the map itself is not rebuilt when they change
        objects = folium.FeatureGroup(name="Obiekty")
        draw_site(objects, site)
        draw_network(objects, analyze_site(site))
        out = st_folium(editing_map(site), height=620, use_container_width=True,
                        key=f"map_{st.session_state.map_key}", feature_group_to_add=objects,
                        returned_objects=["last_active_drawing"])

    geom = (out or {}).get("last_active_drawing")
    if geom and geom.get("geometry"):
        sig = repr(geom["geometry"])
        if sig != st.session_state.get("last_added_geom"):
            st.session_state.last_added_geom = sig
            msg = add_drawing(site, geom["geometry"], mode, attrs)
            st.session_state.map_flash = msg
            st.session_state.obj_ver += 1  # refresh the tables below, not the map
            st.rerun()
    flash = st.session_state.pop("map_flash", None)
    if flash:
        (st.success if flash.startswith("✅") else st.warning)(flash)

    st.subheader("3. Obiekty na mapie (można poprawiać parametry i usuwać wiersze)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Drogi i rampy**")
        if site["roads"]:
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "coords"} | {"długość [m]": round(polyline_length_m(r["coords"]))}
                               for r in site["roads"]])
            ed = st.data_editor(df, num_rows="dynamic", key=f"roads_{st.session_state.obj_ver}",
                                disabled=["id", "długość [m]"], use_container_width=True, hide_index=True,
                                column_config={"kind": st.column_config.SelectboxColumn("rodzaj", options=["droga", "rampa"])})
            _sync(site, "roads", ed, "id")
        else:
            st.caption("brak")
        st.markdown("**Koparki**")
        if site["excavators"]:
            df = pd.DataFrame(site["excavators"])
            ed = st.data_editor(df, num_rows="dynamic", key=f"ex_{st.session_state.obj_ver}", hide_index=True,
                                disabled=["lat", "lon"], use_container_width=True,
                                column_config={"pile": st.column_config.SelectboxColumn(
                                    "usyp", options=[p["id"] for p in site["piles"]] + ["(najbliższy)"])})
            _sync(site, "excavators", ed, "lat")
        else:
            st.caption("brak")
        st.markdown(f"**Kruszarka:** {'✅ ustawiona' if site.get('crusher') else '❌ brak'}")
    with c2:
        st.markdown("**Usypy**")
        if site["piles"]:
            df = pd.DataFrame([{k: v for k, v in p.items() if k != "polygon"} | {"pow. [m²]": round(polygon_area_m2(p["polygon"]))}
                               for p in site["piles"]])
            ed = st.data_editor(df, num_rows="dynamic", key=f"piles_{st.session_state.obj_ver}", hide_index=True,
                                disabled=["pow. [m²]"], use_container_width=True,
                                column_config={"material": st.column_config.SelectboxColumn("materiał", options=MATERIALS)})
            _sync(site, "piles", ed, "polygon")
        else:
            st.caption("brak")

    st.subheader("4. Sieć dróg, poziomy i nachylenia")
    st.caption("Rzędne liczone od kruszarki wzdłuż nachyleń dróg – także dla odgałęzień ze środka rampy. "
               "Biały punkt = węzeł (skrzyżowanie), czerwony = ślepy koniec drogi.")
    c1, c2 = st.columns([1, 3])
    if site.get("crusher"):
        site["crusher"]["elev_m"] = c1.number_input("Rzędna kruszarki [m]", -500.0, 2000.0,
                                                    float(site["crusher"].get("elev_m", 0.0)), 1.0,
                                                    help="np. rzędna terenu n.p.m.; poziomy koparek liczone względem niej")
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
                         hide_index=True, use_container_width=True)

    st.subheader("5. Trasy z mapy → scenariusz")
    st.caption("Usypy stają się źródłami urobku, koparki dostają trasy do kruszarki (najkrótsza droga po narysowanej sieci). "
               "Wozidła, kruszarka, koszty zostają ze scenariusza.")
    c1, c2 = st.columns([1, 3])
    if c1.button("🔁 Zbuduj trasy i przenieś do scenariusza", type="primary"):
        try:
            new = site_to_scenario(copy.deepcopy(site), copy.deepcopy(raw_for_build))
            set_raw(new)
            st.session_state.map_msg = "Scenariusz zaktualizowany z mapy – zobacz zakładki Scenariusz i Wyniki."
            st.rerun()
        except ValueError as e:
            c2.error(str(e))
    if st.session_state.pop("map_msg", None):
        st.success("Scenariusz zaktualizowany z mapy – zobacz zakładki „Scenariusz” i „Wyniki zmiany”.")
    if raw_for_build.get("site") and any("path" in r for r in raw_for_build["routes"].values()):
        rows = [{"trasa": rid, "odcinek": s["name"], "długość [m]": s["length_m"], "nachylenie [%]": s["grade_pct"],
                 "limit [km/h]": s.get("speed_limit_kmh")}
                for rid, r in raw_for_build["routes"].items() for s in r["segments"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _sync(site: dict, key: str, edited: pd.DataFrame, geom_key: str) -> None:
    """Apply table edits (attributes, deleted rows) back to the site objects, keeping geometry."""
    old = site[key]
    keep = []  # rows added in the table are ignored – geometry comes from the map
    for idx, row in edited.iterrows():
        if idx >= len(old):
            continue
        obj = dict(old[idx])
        for col, val in row.items():
            if col in obj and not pd.isna(val) and col not in ("lat", "lon"):
                obj[col] = val.item() if hasattr(val, "item") else val
        keep.append(obj)
    site[key] = keep


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
