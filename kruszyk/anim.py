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
               "start", "przerwa", "postoj_strzelanie", "koniec_zmiany", "tankowanie", "postoj_koparki"]
STATE_LABELS = {
    "jazda_pusty": "jedzie pusty", "przejazd_miedzy_usypami": "jedzie na 2. usyp", "jazda_pelny": "jedzie pełny",
    "kolejka_koparka": "czeka pod koparką", "podstawienie": "podstawia się", "zaladunek": "ładowany",
    "kolejka_kruszarka": "czeka pod kruszarką", "wyladunek": "wyładunek", "wyladunek_halda": "wyładunek na hałdę",
    "awaria": "awaria", "postoj_brak_urobku": "postój – brak urobku", "start": "rozruch / przed startem",
    "przerwa": "przerwa", "postoj_strzelanie": "czeka – strzelanie", "koniec_zmiany": "koniec zmiany",
    "tankowanie": "tankowanie", "postoj_koparki": "czeka – koparka stoi",
}
STATE_COLORS = {
    "jazda_pusty": "#8ecae6", "przejazd_miedzy_usypami": "#b5838d", "jazda_pelny": "#219ebc",
    "kolejka_koparka": "#e63946", "podstawienie": "#f4a261", "zaladunek": "#2a9d8f",
    "kolejka_kruszarka": "#d62828", "wyladunek": "#264653", "wyladunek_halda": "#6c757d", "awaria": "#000000",
    "postoj_brak_urobku": "#adb5bd", "start": "#adb5bd", "przerwa": "#ffd166", "postoj_strzelanie": "#ff006e",
    "koniec_zmiany": "#adb5bd", "tankowanie": "#8338ec", "postoj_koparki": "#fb8500",
}
QUEUES = {"kolejka_koparka", "kolejka_kruszarka", "postoj_koparki"}
CRUSHER_SIDE = {"kolejka_kruszarka", "wyladunek", "wyladunek_halda", "postoj_brak_urobku", "przerwa",
                "postoj_strzelanie", "koniec_zmiany", "tankowanie"}


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
        frames: list[list] = [[0.0, *_offset(crusher, k), code["start"], 0]]
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
                        frames.append([round(sr.start_s + f * sr.duration_s, 1), round(lat, 7), round(lon, 7), c,
                                       sr.speed_limit_kmh if sr.segment not in ("front", "plac_kruszarki", "dojazd") else 0])
                    last = _at(path, cum, d1)
                continue
            if state in ("kolejka_koparka", "podstawienie", "zaladunek", "postoj_koparki"):
                spot = ex_pos.get(row.excavator, last)
            elif state in CRUSHER_SIDE:
                spot = crusher
            else:
                spot = last
            spread = state in QUEUES or state in CRUSHER_SIDE - {"wyladunek", "wyladunek_halda"}
            p = _offset(spot, k) if spread else [round(spot[0], 7), round(spot[1], 7)]
            frames.append([round(row.start_s, 1), *p, c, 0])
            frames.append([round(row.end_s, 1), *p, c, 0])
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
                   "grade": r.get("grade_pct", 0), "active": r.get("active", True)} for r in site.get("roads", [])],
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


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def excavator_advice(res) -> dict[str, list[str]]:
    """What the operator's phone says, minute by minute: who is loaded, who comes next and when,
    and what to do meanwhile (fill the bucket / clean the face / wait)."""
    sc = res.scenario
    st = res.states
    out = {}
    n = int(sc.shift_hours * 60)
    for e in sc.excavators:
        blast = sc.blast(e.blast)
        one_pass = e.swing_cycle_s * (1 + 0.5 * blast.fragmentation)
        mine = st[(st["excavator"] == e.id) & st["state"].isin(["podstawienie", "zaladunek"])].sort_values("start_s")
        spots = mine[mine["state"] == "podstawienie"][["start_s", "truck"]].to_numpy()
        loading = mine[["start_s", "end_s", "truck", "state"]].to_numpy()
        down = [(a, b, r) for a, b, r in getattr(res, "excavator_down", {}).get(e.id, [])]
        lines, j = [], 0
        for m in range(n):
            t = m * 60.0
            w = next((r for a, b, r in down if a <= t < b), None)
            if w:
                lines.append({"przerwa": "⏸ przerwa", "strzelanie": "🔴 STRZELANIE – zjazd z frontu",
                              "tankowanie": "⛽ tankowanie z cysterny"}.get(w, w))
                continue
            cur = next((x for x in loading if x[0] <= t < x[1]), None)
            if cur is not None:
                lines.append(f"🟢 {'ładuje' if cur[3] == 'zaladunek' else 'podstawia się'} {cur[2]}")
                continue
            while j < len(spots) and spots[j][0] <= t:
                j += 1
            if j >= len(spots):
                lines.append("brak wozideł w drodze – przegarnij front")
                continue
            eta, truck = spots[j][0] - t, spots[j][1]
            if eta <= one_pass:
                lines.append(f"{truck} za {_mmss(eta)} → łyżka pełna, czekaj")
            elif eta <= one_pass + 90:
                lines.append(f"{truck} za {_mmss(eta)} → nabierz łyżkę")
            else:
                lines.append(f"{truck} za {_mmss(eta)} → przegarnij front / przygotuj urobek")
        out[e.id] = lines
    return out


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
        "light": res.light_vehicles,
        "closure": list(res.closure) if res.closure else None,
        "breaks": [list(b) for b in res.breaks],
        "advice": excavator_advice(res),
        "start_clock": sc.shift_plan.start_time,
        "t_start": res.t_start,
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
