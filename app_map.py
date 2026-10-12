"""Map screen: calibrate the site, draw roads/ramps, piles, excavators and the crusher."""
from __future__ import annotations

import copy

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from kruszyk.site import centroid, parse_coords, polygon_area_m2, polyline_length_m, site_to_scenario

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
        st.session_state.site = copy.deepcopy(raw.get("site") or empty_site())
        st.session_state.site_ver = st.session_state.ver
        st.session_state.map_key = st.session_state.get("map_key", 0) + 1
    return st.session_state.site


def base_map(site: dict, height_zoom: int | None = None) -> folium.Map:
    m = folium.Map(location=site["center"], zoom_start=height_zoom or site.get("zoom", 16), tiles=None,
                   max_zoom=21, control_scale=True)
    folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satelita", max_zoom=21, max_native_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Mapa", max_zoom=21, max_native_zoom=19).add_to(m)
    return m


def draw_site(m: folium.Map, site: dict) -> None:
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
            st.session_state.map_key += 1
            st.rerun()
        else:
            st.error("Nie rozpoznano współrzędnych. W Google Maps: prawy klik na mapie → kliknij współrzędne (kopiują się).")

    st.subheader("2. Rysowanie obiektów")
    c1, c2 = st.columns([1, 3])
    with c1:
        mode = st.radio("Co rysujesz?", list(MODES), key="draw_mode")
        attrs: dict = {}
        if mode == "Droga / rampa":
            attrs["kind"] = st.selectbox("Rodzaj", ["droga", "rampa"])
            attrs["name"] = st.text_input("Nazwa", f"{attrs['kind']} {len(site['roads']) + 1}")
            attrs["grade_pct"] = st.number_input("Nachylenie w stronę kruszarki [%] (+ pod górę)", -20.0, 20.0,
                                                 8.0 if attrs["kind"] == "rampa" else 0.0, 0.5)
            attrs["speed_limit_kmh"] = st.number_input("Limit prędkości [km/h]", 5.0, 80.0,
                                                       25.0 if attrs["kind"] == "rampa" else 40.0, 5.0)
            st.caption("Rysuj linię po osi drogi. Końce dróg, które się łączą, stawiaj blisko siebie (≤ 30 m).")
        elif mode == "Usyp (materiał)":
            attrs["id"] = st.text_input("Oznaczenie usypu", f"U-{len(site['piles']) + 1:02d}")
            attrs["material"] = st.selectbox("Materiał", MATERIALS)
            attrs["tonnage_t"] = st.number_input("Ilość [t]", 0.0, 10_000_000.0, 10_000.0, 500.0)
            attrs["fragmentation"] = st.slider("Rozdrobnienie (0 = drobne, 1 = bryły)", 0.0, 1.0, 0.3, 0.05)
            attrs["loose_density_t_m3"] = st.number_input("Gęstość spulchniona [t/m³]", 1.0, 3.0, 1.75, 0.05)
            st.caption("Obrysuj usyp wielokątem.")
        elif mode == "Koparka":
            attrs["id"] = st.text_input("Oznaczenie koparki", f"K-{len(site['excavators']) + 1}")
            piles = [p["id"] for p in site["piles"]]
            attrs["pile"] = st.selectbox("Ładuje z usypu", piles or ["(najbliższy)"])
            attrs["bucket_m3"] = st.number_input("Łyżka [m³]", 0.5, 30.0, 5.0, 0.5)
            attrs["swing_cycle_s"] = st.number_input("Cykl łyżki [s]", 10.0, 90.0, 30.0, 1.0)
            st.caption("Postaw znacznik w miejscu pracy koparki.")
        else:
            st.caption("Postaw znacznik na koszu kruszarki (miejsce wyładunku).")
        add = st.button("➕ Dodaj narysowany obiekt", type="primary", use_container_width=True)

    with c2:
        m = base_map(site)
        draw_site(m, site)
        kind = MODES[mode]
        Draw(export=False, position="topleft",
             draw_options={"polyline": kind == "polyline", "polygon": kind == "polygon",
                           "marker": kind == "marker", "circle": False, "rectangle": False,
                           "circlemarker": False},
             edit_options={"edit": False}).add_to(m)
        folium.LayerControl().add_to(m)
        out = st_folium(m, height=620, use_container_width=True, key=f"map_{st.session_state.map_key}",
                        returned_objects=["last_active_drawing", "center", "zoom"])

    if add:
        geom = (out or {}).get("last_active_drawing")
        if not geom:
            st.warning("Najpierw narysuj obiekt narzędziem po lewej stronie mapy.")
        else:
            g = geom["geometry"]
            if out.get("center"):
                site["center"] = [out["center"]["lat"], out["center"]["lng"]]
                site["zoom"] = out.get("zoom", site.get("zoom", 16))
            try:
                if mode == "Droga / rampa" and g["type"] == "LineString":
                    site["roads"].append({"id": f"R{len(site['roads']) + 1}", **attrs, "coords": _latlon(g["coordinates"])})
                elif mode == "Usyp (materiał)" and g["type"] == "Polygon":
                    site["piles"].append({**attrs, "polygon": _latlon(g["coordinates"][0][:-1])})
                elif mode == "Koparka" and g["type"] == "Point":
                    lon, lat = g["coordinates"]
                    site["excavators"].append({**attrs, "lat": lat, "lon": lon})
                elif mode == "Kruszarka" and g["type"] == "Point":
                    lon, lat = g["coordinates"]
                    site["crusher"] = {"lat": lat, "lon": lon}
                else:
                    raise ValueError(f"Narysowano {g['type']}, a wybrany obiekt to „{mode}”")
                st.session_state.map_key += 1
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.subheader("3. Obiekty na mapie (można poprawiać parametry i usuwać wiersze)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Drogi i rampy**")
        if site["roads"]:
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "coords"} | {"długość [m]": round(polyline_length_m(r["coords"]))}
                               for r in site["roads"]])
            ed = st.data_editor(df, num_rows="dynamic", key=f"roads_{st.session_state.map_key}",
                                disabled=["id", "długość [m]"], use_container_width=True, hide_index=True,
                                column_config={"kind": st.column_config.SelectboxColumn("rodzaj", options=["droga", "rampa"])})
            _sync(site, "roads", ed, "id")
        else:
            st.caption("brak")
        st.markdown("**Koparki**")
        if site["excavators"]:
            df = pd.DataFrame(site["excavators"])
            ed = st.data_editor(df, num_rows="dynamic", key=f"ex_{st.session_state.map_key}", hide_index=True,
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
            ed = st.data_editor(df, num_rows="dynamic", key=f"piles_{st.session_state.map_key}", hide_index=True,
                                disabled=["pow. [m²]"], use_container_width=True,
                                column_config={"material": st.column_config.SelectboxColumn("materiał", options=MATERIALS)})
            _sync(site, "piles", ed, "polygon")
        else:
            st.caption("brak")

    st.subheader("4. Trasy z mapy → scenariusz")
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
