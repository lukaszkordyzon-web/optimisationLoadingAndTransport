"""Map screen: calibrate the site, draw roads/ramps, piles, excavators and the crusher."""
from __future__ import annotations

import copy

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from kruszyk.psd import DEFAULT_PRESET, PRESETS, SIEVES_MM, clean, preset_of, rosin_rammler, summary
from kruszyk.site import (analyze_site, normalize_roads, centroid, parse_coords, polygon_area_m2, polyline_length_m,
                          site_to_scenario)

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"
MATERIALS = ["wapień", "dolomit", "piaskowiec", "granit", "bazalt", "melafir", "żwir", "piasek", "nadkład", "inny"]
ROAD_COLORS = {"rampa": "#d500f9", "droga": "#ffffff"}  # good contrast on light quarry imagery
PLANNED_COLOR = "#9e9e9e"
SKETCH_COLOR = "#ffd600"
MODES = {
    "Droga / rampa": "polyline",
    "Usyp (materiał)": "polygon",
    "Koparka": "marker",
    "Kruszarka": "marker",
    "Miejsce strzelania": "marker",
    "Miejsce postoju koparki": "marker",
    "Trasa dedykowana": "polyline",
}
DEDICATED_FOR = {"ekipa": "🧨 ekipa strzałowa", "sztygar": "👷 sztygar", "cysterna": "⛽ cysterna"}
DIRECTIONS = {"⬇ w dół": -1, "⬆ pod górę": 1, "➡ płasko": 0}


def empty_site() -> dict:
    return {"center": [52.0, 19.4], "zoom": 6, "crusher": None, "piles": [], "excavators": [], "roads": []}


def get_site(raw: dict) -> dict:
    if "site" not in st.session_state or st.session_state.get("site_ver") != st.session_state.ver:
        old = st.session_state.get("site")
        new = copy.deepcopy(raw.get("site") or empty_site())
        normalize_roads(new)
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
        tip = (f"{r['name']} · {r.get('kind')} · wzdłuż strzałki: {slope} · limit {r.get('speed_limit_kmh', '?')} km/h · "
               f"{polyline_length_m(r['coords']):.0f} m" + ("" if active else " · PLANOWANA (dziś nieużywana)"))
        folium.PolyLine(pts, color="#000000", weight=10, opacity=0.55).add_to(m)  # dark casing
        folium.PolyLine(pts, color=ROAD_COLORS.get(r.get("kind"), "#ffffff") if active else PLANNED_COLOR, weight=6,
                        dash_array=None if active else "10 8", tooltip=tip).add_to(m)
        _direction_arrow(m, pts, g, tip)
    for d in site.get("dedicated", []):
        colors = {"ekipa": "#ff006e", "sztygar": "#3a86ff", "cysterna": "#8338ec"}
        folium.PolyLine([tuple(c) for c in d["coords"]], color=colors.get(d.get("for"), "#ffbe0b"), weight=4,
                        dash_array="6 6", tooltip=f"trasa dedykowana: {DEDICATED_FOR.get(d.get('for'), 'koparka ' + str(d.get('for')))}"
                        ).add_to(m)
    for p in site.get("parking", []):
        folium.Marker((p["lat"], p["lon"]), tooltip=f"miejsce postoju koparki {p.get('excavator')}",
                      icon=folium.DivIcon(html=f"<div style='font-size:13px;font-weight:700;color:#fff;"
                                               f"text-shadow:0 0 3px #000;white-space:nowrap'>🅿️ {p.get('excavator')}</div>")
                      ).add_to(m)
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
    if site.get("blast"):
        b = site["blast"]
        folium.Marker((b["lat"], b["lon"]), tooltip="Miejsce strzelania",
                      icon=folium.DivIcon(html="<div style='font-size:22px'>🧨</div>")).add_to(m)


def _latlon(coords: list) -> list:
    return [[round(c[1], 7), round(c[0], 7)] for c in coords]  # GeoJSON is [lon, lat]


DEFAULTS = {
    "Droga / rampa": {"kind": "droga", "grade_draw_pct": 0.0, "speed_limit_kmh": 40.0},
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
        target = mode if mode in ("Kruszarka", "Miejsce strzelania", "Miejsce postoju koparki") else "Koparka"
    if g["type"] == "LineString" and mode == "Trasa dedykowana":
        target = mode
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
    if target == "Miejsce strzelania":
        site["blast"] = {"lat": lat, "lon": lon}
        return "✅ Ustawiono miejsce strzelania – godziny ustaw w sekcji „Zmiana i zdarzenia”"
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
             "active": st.column_config.CheckboxColumn("w użyciu", help="odznacz = droga planowana / zamknięta")}
PILE_COLS = {"id": "usyp", "material": st.column_config.SelectboxColumn("materiał", options=MATERIALS),
             "tonnage_t": "ilość [t]", "fragmentation": "rozdrobnienie (0–1)", "loose_density_t_m3": "gęstość [t/m³]",
             "target_share": st.column_config.NumberColumn("udział w nadawie [%]", min_value=0.0, max_value=100.0,
                                                           help="ile procent nadawy ma iść z tego usypu (puste = bez celu)")}
EX_COLS = {"id": "koparka", "bucket_m3": "łyżka [m³]", "swing_cycle_s": "cykl łyżki [s]", "lat": "szer.", "lon": "dł."}


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
    st.subheader("6. Wozidła i kruszarka")
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
    c1, c2, _ = st.columns([1, 1, 2])
    cr = raw.setdefault("crusher", {})
    cr["capacity_tph"] = c1.number_input("Kruszarka – wydajność [t/h]", 50.0, 10000.0,
                                         float(cr.get("capacity_tph", 1000)), 50.0, key=f"qcap_{st.session_state.ver}")
    cr["hopper_t"] = c2.number_input("Kosz kruszarki [t]", 10.0, 2000.0, float(cr.get("hopper_t", 150)), 10.0,
                                     key=f"qhop_{st.session_state.ver}")


def psd_editor(site: dict, max_feed_mm: float, penalty: float) -> None:
    """Grading curve of every pile: pick a typical curve, set it from P80, or type sieve results."""
    import plotly.graph_objects as go

    st.subheader("3b. Krzywe uziarnienia usypów")
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
    c1, c2 = st.columns([1, 3])
    with c1:
        mode = st.radio("Co dodajesz?", list(MODES), key="draw_mode")
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
            st.caption("Narzędzie „znacznik” (📍). Postaw go na koszu kruszarki (miejsce wyładunku).")
        else:
            st.caption("Narzędzie „znacznik” (📍). Postaw go w miejscu strzelania. Godziny przyjazdu ekipy, "
                       "odpalenia i wyjazdu ustawisz niżej w sekcji „Zmiana i zdarzenia”.")
        st.info("Wybierz typ i parametry, potem narysuj obiekt – **dodaje się sam**. "
                "Linia = droga/rampa, wielokąt = usyp, znacznik = koparka (albo kruszarka, gdy wybrana).")
        st.caption("Żółty = szkic (znika po odświeżeniu strony). Dodane: drogi białe, rampy fioletowe, planowane szare, "
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
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "coords"} | {"active": r.get("active", True)}
                               | {"długość [m]": round(polyline_length_m(r["coords"]))} for r in site["roads"]])
            ed = st.data_editor(df, num_rows="dynamic", key=f"roads_{st.session_state.obj_ver}",
                                disabled=["id", "długość [m]"], width="stretch", hide_index=True,
                                column_config=ROAD_COLS)
            _sync(site, "roads", ed, "id", extra=("active",))
        else:
            st.caption("brak")
        st.markdown("**Koparki**")
        if site["excavators"]:
            df = pd.DataFrame(site["excavators"])
            ed = st.data_editor(df, num_rows="dynamic", key=f"ex_{st.session_state.obj_ver}", hide_index=True,
                                disabled=["lat", "lon"], width="stretch",
                                column_config={**EX_COLS, "pile": st.column_config.SelectboxColumn(
                                    "ładuje z usypu", options=[p["id"] for p in site["piles"]] + ["(najbliższy)"])})
            _sync(site, "excavators", ed, "lat")
        else:
            st.caption("brak")
        st.markdown(f"**Kruszarka:** {'✅ ustawiona' if site.get('crusher') else '❌ brak'} · "
                    f"**Miejsce strzelania:** {'✅' if site.get('blast') else '—'}")
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
    with c2:
        st.markdown("**Usypy**")
        if site["piles"]:
            max_feed = float(raw_for_build.get("crusher", {}).get("max_feed_mm", 800))
            rows = []
            for p in site["piles"]:
                row = {k: v for k, v in p.items() if k not in ("polygon", "psd", "fragmentation")}
                if p.get("psd"):
                    sm = summary(p["psd"], max_feed)
                    row |= {"uziarnienie": preset_of(p["psd"]), "P80 [mm]": sm["P80 [mm]"], "nadziarno [%]": sm["nadziarno [%]"]}
                row.setdefault("target_share", None)
                row["pow. [m²]"] = round(polygon_area_m2(p["polygon"]))
                rows.append(row)
            df = pd.DataFrame(rows)
            ed = st.data_editor(df, num_rows="dynamic", key=f"piles_{st.session_state.obj_ver}", hide_index=True,
                                disabled=["pow. [m²]", "uziarnienie", "P80 [mm]", "nadziarno [%]"],
                                width="stretch", column_config=PILE_COLS)
            _sync(site, "piles", ed, "polygon", extra=("target_share",))
        else:
            st.caption("brak")

    if site["piles"]:
        psd_editor(site, float(raw_for_build.get("crusher", {}).get("max_feed_mm", 800)),
                   float(raw_for_build.get("crusher", {}).get("fragmentation_penalty", 0.4)))

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
    if site != (raw_for_build.get("site") or {}) and (site.get("roads") or site.get("piles") or site.get("excavators")):
        st.warning("⚠️ Na mapie są zmiany, których jeszcze nie zapisano – kliknij przycisk poniżej.")
    c1, c2 = st.columns([1, 3])
    if c1.button("💾 Zapisz kopalnię" if simple else "🔁 Zbuduj trasy i przenieś do scenariusza", type="primary"):
        try:
            new = site_to_scenario(copy.deepcopy(site), copy.deepcopy(raw_for_build))
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


def _sync(site: dict, key: str, edited: pd.DataFrame, geom_key: str, extra: tuple = ()) -> None:
    """Apply table edits (attributes, deleted rows) back to the site objects, keeping geometry."""
    old = site[key]
    keep = []  # rows added in the table are ignored – geometry comes from the map
    for idx, row in edited.iterrows():
        if idx >= len(old):
            continue
        obj = dict(old[idx])
        for col, val in row.items():
            if (col in obj or col in extra) and not pd.isna(val) and col not in ("lat", "lon"):
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
