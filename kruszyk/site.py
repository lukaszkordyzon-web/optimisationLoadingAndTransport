"""Site map: objects drawn on a satellite map -> routes, sources and excavators for the simulation.

Stored in the scenario YAML under `site`:

    site:
      center: [lat, lon]
      zoom: 16
      crusher: {lat, lon}
      piles:       # usypy / odstrzały – sources of material
        - {id, material, tonnage_t, fragmentation, loose_density_t_m3, polygon: [[lat, lon], ...]}
      excavators:  # loading points; each works on a pile
        - {id, lat, lon, pile, bucket_m3, swing_cycle_s}
      roads:       # haul roads and ramps, drawn as polylines
        - {id, name, kind: droga|rampa, grade_pct, speed_limit_kmh, coords: [[lat, lon], ...]}

`grade_pct` of a road is the grade in the direction TOWARDS THE CRUSHER (+ = uphill when loaded),
so the drawing direction does not matter.
"""
from __future__ import annotations

import heapq
import math
import re

R_EARTH = 6_371_000.0
SNAP_M = 30.0  # road ends closer than this are treated as one junction


# ---------------------------------------------------------------- geometry
def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(h))


def polyline_length_m(coords: list) -> float:
    return sum(haversine_m(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def polygon_area_m2(coords: list) -> float:
    if len(coords) < 3:
        return 0.0
    lat0 = sum(c[0] for c in coords) / len(coords)
    k = math.cos(math.radians(lat0))
    pts = [(math.radians(c[1]) * R_EARTH * k, math.radians(c[0]) * R_EARTH) for c in coords]
    return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2


def centroid(coords: list) -> tuple[float, float]:
    return (sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords))


def point_along(path: list, dist_m: float) -> tuple[float, float]:
    """Point at `dist_m` along a polyline (clamped)."""
    if dist_m <= 0:
        return tuple(path[0])
    for a, b in zip(path, path[1:]):
        d = haversine_m(a, b)
        if dist_m <= d:
            f = dist_m / d if d else 0
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        dist_m -= d
    return tuple(path[-1])


_COORD_PATTERNS = [
    r"@(-?\d+\.\d+),(-?\d+\.\d+)",            # google.com/maps/@50.1,20.2,15z
    r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",        # place links
    r"[?&](?:q|ll|query)=(-?\d+\.\d+),\s*(-?\d+\.\d+)",
    r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$",  # "50.123, 20.456"
]


def parse_coords(text: str) -> tuple[float, float] | None:
    """Read lat/lon from a Google Maps link or a plain 'lat, lon' string."""
    for pat in _COORD_PATTERNS:
        m = re.search(pat, text.strip())
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None


# ---------------------------------------------------------------- road graph
class RoadGraph:
    """Road polylines as a graph; vertices of different roads within SNAP_M are merged."""

    def __init__(self, roads: list[dict]):
        self.nodes: list[tuple[float, float]] = []
        self.adj: dict[int, list[tuple[int, float, int]]] = {}  # node -> [(node, length, road index)]
        self.roads = roads
        for ri, road in enumerate(roads):
            ids = [self._node(tuple(c)) for c in road["coords"]]
            for a, b in zip(ids, ids[1:]):
                if a == b:
                    continue
                d = haversine_m(self.nodes[a], self.nodes[b])
                self.adj.setdefault(a, []).append((b, d, ri))
                self.adj.setdefault(b, []).append((a, d, ri))

    def _node(self, p: tuple[float, float]) -> int:
        for i, q in enumerate(self.nodes):
            if haversine_m(p, q) < SNAP_M:
                return i
        self.nodes.append(p)
        return len(self.nodes) - 1

    def nearest(self, p: tuple[float, float]) -> tuple[int, float]:
        best = min(range(len(self.nodes)), key=lambda i: haversine_m(p, self.nodes[i]))
        return best, haversine_m(p, self.nodes[best])

    def shortest_path(self, src: int, dst: int) -> list[tuple[int, int, float, int]] | None:
        """Edges (from, to, length, road) of the shortest path, or None if not connected."""
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w, ri in self.adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, (u, w, ri)
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None
        edges, v = [], dst
        while v != src:
            u, w, ri = prev[v]
            edges.append((u, v, w, ri))
            v = u
        return list(reversed(edges))


def build_route(site: dict, start: tuple[float, float], route_id: str) -> dict:
    """Route from a loading point to the crusher: segments (grouped per road) + geometry."""
    roads = site.get("roads", [])
    if not roads:
        raise ValueError("Na mapie nie ma dróg")
    if not site.get("crusher"):
        raise ValueError("Na mapie nie ma kruszarki")
    crusher = (site["crusher"]["lat"], site["crusher"]["lon"])
    g = RoadGraph(roads)
    s_node, s_off = g.nearest(start)
    c_node, c_off = g.nearest(crusher)
    path = g.shortest_path(s_node, c_node)
    if path is None:
        raise ValueError(f"{route_id}: brak połączenia drogowego do kruszarki – połącz końce dróg (≤ {SNAP_M:.0f} m)")

    segments, geometry = [], [list(start)]
    if s_off > 5:
        segments.append({"name": "front", "length_m": round(s_off, 1), "grade_pct": 0.0, "speed_limit_kmh": 15.0})
    geometry.append(list(g.nodes[s_node]))
    for u, v, w, ri in path:
        road = roads[ri]
        name = road.get("name") or road.get("id") or f"droga {ri + 1}"
        if segments and segments[-1].get("_road") == ri:
            segments[-1]["length_m"] = round(segments[-1]["length_m"] + w, 1)
        else:
            segments.append({"name": name, "length_m": round(w, 1),
                             "grade_pct": float(road.get("grade_pct", 0.0)),
                             "speed_limit_kmh": float(road.get("speed_limit_kmh", 30.0)), "_road": ri})
        geometry.append(list(g.nodes[v]))
    if c_off > 5:
        segments.append({"name": "plac_kruszarki", "length_m": round(c_off, 1), "grade_pct": 0.0,
                         "speed_limit_kmh": 15.0})
    geometry.append(list(crusher))
    for s in segments:
        s.pop("_road", None)
    return {"segments": segments, "path": geometry}


def site_to_scenario(site: dict, raw: dict) -> dict:
    """Replace blasts/excavators/routes of a raw scenario with what is drawn on the map.
    Trucks, crusher parameters, costs etc. stay as they are."""
    piles = site.get("piles", [])
    excavators = site.get("excavators", [])
    if not piles:
        raise ValueError("Na mapie nie ma usypów")
    if not excavators:
        raise ValueError("Na mapie nie ma koparek")
    new = dict(raw)
    new["blasts"] = []
    for p in piles:
        b = {"id": p["id"], "tonnage_t": float(p.get("tonnage_t", 0)),
             "fragmentation": float(p.get("fragmentation", 0.3)), "material": p.get("material", "")}
        if p.get("loose_density_t_m3"):
            b["loose_density_t_m3"] = float(p["loose_density_t_m3"])
        new["blasts"].append(b)
    pile_ids = {p["id"] for p in piles}
    new["excavators"], new["routes"] = [], {}
    for e in excavators:
        pos = (e["lat"], e["lon"])
        pile = e.get("pile")
        if pile not in pile_ids:  # nearest pile by centroid
            pile = min(piles, key=lambda p: haversine_m(pos, centroid(p["polygon"])))["id"]
        rid = f"T-{e['id']}"
        new["routes"][rid] = build_route(site, pos, rid)
        new["excavators"].append({"id": e["id"], "bucket_m3": float(e.get("bucket_m3", 5.0)),
                                  "swing_cycle_s": float(e.get("swing_cycle_s", 30.0)),
                                  "blast": pile, "route": rid})
    new["site"] = site
    return new
