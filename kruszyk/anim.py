"""Animation data: truck trajectories on the drawn road network + live cost timeline.

The browser plays it back fast (Leaflet, no server round trips): every truck is a list of keyframes
[t, lat, lon, state]; positions between keyframes are interpolated.
"""
from __future__ import annotations

import base64
import json
import math

import numpy as np

from .site import analyze_site, haversine_m

STATE_CODES = ["jazda_pusty", "przejazd_miedzy_usypami", "jazda_pelny", "kolejka_koparka", "podstawienie",
               "zaladunek", "kolejka_kruszarka", "wyladunek", "wyladunek_halda", "awaria", "postoj_brak_urobku",
               "start"]
STATE_LABELS = {
    "jazda_pusty": "jedzie pusty", "przejazd_miedzy_usypami": "jedzie na 2. usyp", "jazda_pelny": "jedzie pełny",
    "kolejka_koparka": "czeka pod koparką", "podstawienie": "podstawia się", "zaladunek": "ładowany",
    "kolejka_kruszarka": "czeka pod kruszarką", "wyladunek": "wyładunek", "wyladunek_halda": "wyładunek na hałdę",
    "awaria": "awaria", "postoj_brak_urobku": "postój – brak urobku", "start": "przed startem",
}
STATE_COLORS = {
    "jazda_pusty": "#8ecae6", "przejazd_miedzy_usypami": "#b5838d", "jazda_pelny": "#219ebc",
    "kolejka_koparka": "#e63946", "podstawienie": "#f4a261", "zaladunek": "#2a9d8f",
    "kolejka_kruszarka": "#d62828", "wyladunek": "#264653", "wyladunek_halda": "#6c757d", "awaria": "#000000",
    "postoj_brak_urobku": "#adb5bd", "start": "#adb5bd",
}
QUEUES = {"kolejka_koparka", "kolejka_kruszarka"}


def _cum(path: list) -> list[float]:
    out = [0.0]
    for a, b in zip(path, path[1:]):
        out.append(out[-1] + haversine_m(a, b))
    return out


def _at(path: list, cum: list[float], d: float) -> tuple[float, float]:
    d = min(max(d, 0.0), cum[-1])
    for i in range(1, len(cum)):
        if d <= cum[i]:
            seg = cum[i] - cum[i - 1]
            f = 0.0 if seg == 0 else (d - cum[i - 1]) / seg
            a, b = path[i - 1], path[i]
            return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
    return tuple(path[-1])


def _offset(p, k: int) -> list[float]:
    """Spread trucks standing at the same spot so each stays visible."""
    ang = k * 2.4
    r = 0.00006 * (1 + k // 6)
    return [round(p[0] + r * math.sin(ang), 7), round(p[1] + r * math.cos(ang) * 1.5, 7)]


def trajectories(res) -> dict[str, list[list]]:
    sc = res.scenario
    routes = {**sc.routes, **sc.links}
    geo = {}
    for rid, r in routes.items():
        if r.path:
            cum = _cum(r.path)
            seg_len = [s.length_m for s in r.segments]
            scale = cum[-1] / sum(seg_len) if sum(seg_len) else 1.0
            bounds = np.concatenate([[0.0], np.cumsum(seg_len)]) * scale
            geo[rid] = (r.path, cum, bounds)
    crusher = (sc.site["crusher"]["lat"], sc.site["crusher"]["lon"])
    ex_pos = {e.id: tuple(routes[e.route].path[0]) for e in sc.excavators if e.route in geo}
    code = {s: i for i, s in enumerate(STATE_CODES)}

    out: dict[str, list[list]] = {}
    seg = res.segments
    for k, truck in enumerate(res.truck_ids):
        frames: list[list] = [[0.0, *_offset(crusher, k), code["start"]]]
        st = res.states[res.states["truck"] == truck].sort_values("start_s")
        tseg = seg[seg["truck"] == truck] if len(seg) else seg
        last = crusher
        for row in st.itertuples():
            state = row.state
            c = code.get(state, code["start"])
            if state in ("jazda_pusty", "przejazd_miedzy_usypami", "jazda_pelny"):
                leg = {"jazda_pusty": "pusty", "przejazd_miedzy_usypami": "miedzy", "jazda_pelny": "pelny"}[state]
                rows = tseg[(tseg["trip"] == row.trip) & (tseg["leg"] == leg)
                            & (tseg["start_s"] >= row.start_s - 1e-6) & (tseg["start_s"] < row.end_s)]
                for sr in rows.sort_values("start_s").itertuples():
                    if sr.route not in geo:
                        continue
                    path, cum, bounds = geo[sr.route]
                    d0, d1 = bounds[sr.seg_idx], bounds[sr.seg_idx + 1]
                    if not sr.forward:
                        d0, d1 = d1, d0
                    inner = [d for d in cum if min(d0, d1) < d < max(d0, d1)]
                    inner.sort(reverse=d0 > d1)
                    for d in [d0, *inner, d1]:
                        f = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                        lat, lon = _at(path, cum, d)
                        frames.append([round(sr.start_s + f * sr.duration_s, 1), round(lat, 7), round(lon, 7), c])
                    last = _at(path, cum, d1)
                continue
            if state in ("kolejka_koparka", "podstawienie", "zaladunek"):
                spot = ex_pos.get(row.excavator, last)
            elif state in ("kolejka_kruszarka", "wyladunek", "wyladunek_halda", "postoj_brak_urobku"):
                spot = crusher
            else:
                spot = last
            p = _offset(spot, k) if state in QUEUES or state == "postoj_brak_urobku" else [round(spot[0], 7), round(spot[1], 7)]
            frames.append([round(row.start_s, 1), *p, c])
            frames.append([round(row.end_s, 1), *p, c])
            last = spot
        out[truck] = frames
    return out


def network(sc) -> dict:
    site = sc.site or {}
    info = analyze_site(site) if site.get("roads") else {"graph": None, "elev": {}}
    g = info["graph"]
    nodes = []
    if g is not None:
        for n in info.get("junctions", []):
            nodes.append({"lat": g.nodes[n][0], "lon": g.nodes[n][1], "elev": round(info["elev"].get(n, 0.0), 1)})
    levels = info.get("levels", {})
    return {
        "roads": [{"coords": r["coords"], "kind": r.get("kind", "droga"), "name": r.get("name", ""),
                   "grade": r.get("grade_pct", 0)} for r in site.get("roads", [])],
        "piles": [{"polygon": p["polygon"], "id": p["id"], "material": p.get("material", "")} for p in site.get("piles", [])],
        "excavators": [{"id": e["id"], "lat": e["lat"], "lon": e["lon"],
                        "elev": (levels.get(e["id"]) or {}).get("elev_m")} for e in site.get("excavators", [])],
        "crusher": site.get("crusher"),
        "junctions": nodes,
        "center": site.get("center"), "zoom": site.get("zoom", 15),
    }


def _series(res) -> dict:
    tl = res.timeline
    if tl is None or tl.empty:
        return {}
    clean = lambda s: [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(float(v), 3) for v in s]  # noqa: E731
    return {"t": clean(tl["t_s"]), "dumped": clean(tl["dumped_t"]), "fuel": clean(tl["fuel_l"]),
            "cost": clean(tl["cost"]), "cpt": clean(tl["cost_per_t"]), "hopper": clean(tl["level_t"])}


def animation_payload(res, baseline=None, label: str = "", baseline_label: str = "naturalny") -> str:
    sc = res.scenario
    payload = {
        "shift_s": sc.shift_hours * 3600,
        "label": label, "baseline_label": baseline_label,
        "states": STATE_CODES, "labels": STATE_LABELS, "colors": STATE_COLORS,
        "queue_codes": [STATE_CODES.index(s) for s in QUEUES],
        "trucks": trajectories(res),
        "net": network(sc),
        "series": _series(res),
        "baseline": _series(baseline) if baseline is not None else None,
        "hopper_cap": sc.crusher.hopper_t,
        "excavators": [{"id": e.id, "blast": e.blast, "material": sc.blast(e.blast).material} for e in sc.excavators],
    }
    return json.dumps(payload, separators=(",", ":"))


def animation_html(res, baseline=None, label: str = "", baseline_label: str = "naturalny") -> str:
    from pathlib import Path

    here = Path(__file__).parent
    template = (here / "anim_template.html").read_text(encoding="utf-8")
    # Leaflet is bundled (kruszyk/vendor, BSD-2) so the animation works without access to any CDN
    css = (here / "vendor" / "leaflet.css").read_text(encoding="utf-8")
    layers_png = base64.b64encode((here / "vendor" / "layers.png").read_bytes()).decode()
    css = css.replace("url(images/layers.png)", f"url(data:image/png;base64,{layers_png})")
    css = css.replace("url(images/layers-2x.png)", f"url(data:image/png;base64,{layers_png})")
    js = (here / "vendor" / "leaflet.js").read_text(encoding="utf-8")
    html = template.replace('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">',
                            f"<style>{css}</style>")
    html = html.replace('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>',
                        f"<script>{js}</script>")
    return html.replace("__DATA__", animation_payload(res, baseline, label, baseline_label))
