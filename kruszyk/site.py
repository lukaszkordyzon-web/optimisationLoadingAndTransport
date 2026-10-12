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
so the drawing direction does not matter. Roads form a network: an end of a road placed on another
road (≤ SNAP_M) – also in the middle of it – becomes a junction. Elevations of all junctions are
computed from the crusher (crusher.elev_m, default 0) along the grades, so a branch half-way down
a ramp starts at the right level and every excavator knows its level.
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
def _local(p, lat0: float) -> tuple[float, float]:
    k = math.cos(math.radians(lat0))
    return math.radians(p[1]) * R_EARTH * k, math.radians(p[0]) * R_EARTH


def project_on_segment(p, a, b) -> tuple[tuple[float, float], float, float]:
    """Closest point to p on segment a-b: (point, t in 0..1, distance m)."""
    lat0 = (a[0] + b[0]) / 2
    px, py = _local(p, lat0)
    ax, ay = _local(a, lat0)
    bx, by = _local(b, lat0)
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return q, t, haversine_m(p, q)


class RoadGraph:
    """Road polylines as a graph.

    * vertices of different roads within SNAP_M are merged into one node,
    * a dead end lying on another road (also mid-segment) splits that road -> T-junction,
    * extra points (excavators, crusher) are attached to the closest point of the network.
    """

    def __init__(self, roads: list[dict]):
        roads = [r for r in roads if r.get("active", True)]  # planned / closed roads are not driven on
        self.nodes: list[tuple[float, float]] = []
        self.edges: dict[tuple[int, int], tuple[float, int]] = {}  # (u<v) -> (length, road index)
        self.roads = roads
        for ri, road in enumerate(roads):
            ids = [self._node(tuple(c)) for c in road["coords"]]
            for a, b in zip(ids, ids[1:]):
                if a != b:
                    self._add_edge(a, b, ri)
        self._join_dead_ends()

    # -- construction
    def _node(self, p) -> int:
        for i, q in enumerate(self.nodes):
            if haversine_m(p, q) < SNAP_M:
                return i
        self.nodes.append(tuple(p))
        return len(self.nodes) - 1

    def _add_edge(self, a: int, b: int, ri: int):
        self.edges[(min(a, b), max(a, b))] = (haversine_m(self.nodes[a], self.nodes[b]), ri)

    def degree(self, n: int) -> int:
        return sum(1 for (a, b) in self.edges if n in (a, b))

    def _split(self, edge: tuple[int, int], n: int):
        _, ri = self.edges.pop(edge)
        self._add_edge(edge[0], n, ri)
        self._add_edge(n, edge[1], ri)

    def _nearest_edge(self, p, exclude_node: int | None = None):
        best = None
        for (a, b) in self.edges:
            if exclude_node in (a, b):
                continue
            q, t, d = project_on_segment(p, self.nodes[a], self.nodes[b])
            if best is None or d < best[3]:
                best = ((a, b), q, t, d)
        return best

    def _join_dead_ends(self):
        for n in range(len(self.nodes)):
            if self.degree(n) != 1:
                continue
            hit = self._nearest_edge(self.nodes[n], exclude_node=n)
            if hit and hit[3] < SNAP_M and 0.0 < hit[2] < 1.0:
                self._split(hit[0], n)  # T-junction: the dead end becomes a node of the other road

    def attach(self, p) -> tuple[int, float]:
        """Node for an outside point (excavator, crusher): closest network point, split if needed.
        Returns (node, distance from p to the network)."""
        if not self.edges:
            raise ValueError("Na mapie nie ma dróg")
        edge, q, t, d = self._nearest_edge(p)
        if t <= 0.0 or t >= 1.0 or haversine_m(q, self.nodes[edge[0] if t <= 0.5 else edge[1]]) < 3:
            return (edge[0] if t <= 0.5 else edge[1]), d
        self.nodes.append(q)
        n = len(self.nodes) - 1
        self._split(edge, n)
        return n, d

    def adj(self) -> dict[int, list[tuple[int, float, int]]]:
        out: dict[int, list] = {}
        for (a, b), (w, ri) in self.edges.items():
            out.setdefault(a, []).append((b, w, ri))
            out.setdefault(b, []).append((a, w, ri))
        return out

    # -- analysis
    def shortest_path(self, src: int, dst: int) -> list[tuple[int, int, float, int]] | None:
        """Edges (from, to, length, road) of the shortest path, or None if not connected."""
        adj = self.adj()
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w, ri in adj.get(u, []):
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

    def elevations(self, root: int, root_elev: float) -> tuple[dict[int, float], list[str]]:
        """Node elevations from the crusher along the shortest-path tree. Going away from the
        crusher on a road with grade g (towards the crusher) changes the height by -g*len/100.
        Returns elevations and warnings about loops whose grades do not close (> 3 m)."""
        adj = self.adj()
        elev, dist = {root: root_elev}, {root: 0.0}
        pq = [(0.0, root)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w, ri in adj.get(u, []):
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    elev[v] = elev[u] - float(self.roads[ri].get("grade_pct", 0.0)) * w / 100
                    heapq.heappush(pq, (d + w, v))
        warnings = []
        for (a, b), (w, ri) in self.edges.items():
            if a in elev and b in elev:
                g = float(self.roads[ri].get("grade_pct", 0.0))
                if g == 0 and abs(elev[a] - elev[b]) > 3:
                    name = self.roads[ri].get("name") or self.roads[ri].get("id")
                    warnings.append(f"Droga „{name}”: płaska, a jej końce wyszły na różnych wysokościach "
                                    f"({elev[a]:.0f} / {elev[b]:.0f} m) – sprawdź nachylenia w pętli")
        return elev, sorted(set(warnings))


def _grade(dz: float, length: float) -> float:
    return round(100 * dz / length, 2) if length > 0 else 0.0


def path_between(g: RoadGraph, start, end, elev: dict[int, float],
                 start_name: str = "front", end_name: str = "plac_kruszarki") -> dict:
    """Route start -> end over the network. Segments are grouped per road; each segment's grade is
    computed from node elevations IN THE DIRECTION OF TRAVEL (+ = uphill)."""
    s_node, s_off = g.attach(start)
    e_node, e_off = g.attach(end)
    path = g.shortest_path(s_node, e_node)
    if path is None:
        return None
    segments, geometry = [], [list(start)]
    if s_off > 5:
        segments.append({"name": start_name, "length_m": round(s_off, 1), "grade_pct": 0.0, "speed_limit_kmh": 15.0})
    geometry.append(list(g.nodes[s_node]))
    run = None
    for u, v, w, ri in path:
        road = g.roads[ri]
        name = road.get("name") or road.get("id") or f"droga {ri + 1}"
        dz = elev.get(v, 0.0) - elev.get(u, 0.0)
        if run and run["_road"] == ri:
            run["length_m"] += w
            run["_dz"] += dz
        else:
            run = {"name": name, "length_m": w, "_dz": dz, "speed_limit_kmh": float(road.get("speed_limit_kmh", 30.0)),
                   "_road": ri, "kind": road.get("kind", "droga")}
            segments.append(run)
        geometry.append(list(g.nodes[v]))
    if e_off > 5:
        segments.append({"name": end_name, "length_m": round(e_off, 1), "grade_pct": 0.0, "speed_limit_kmh": 15.0})
    geometry.append(list(end))
    for sgm in segments:
        if "_road" in sgm:
            sgm["grade_pct"] = _grade(sgm.pop("_dz"), sgm["length_m"])
            sgm["length_m"] = round(sgm["length_m"], 1)
            sgm.pop("_road")
            sgm.pop("kind")
    return {"segments": segments, "path": geometry,
            "elev_start_m": round(elev.get(s_node, 0.0), 1), "elev_end_m": round(elev.get(e_node, 0.0), 1)}


def analyze_site(site: dict) -> dict:
    """Network built from the drawing: graph, elevations, levels of excavators/piles, warnings."""
    roads = site.get("roads", [])
    out = {"graph": None, "elev": {}, "warnings": [], "levels": {}, "junctions": [], "dead_ends": []}
    if not roads:
        out["warnings"].append("Na mapie nie ma dróg")
        return out
    g = RoadGraph(roads)
    out["graph"] = g
    if not site.get("crusher"):
        out["warnings"].append("Na mapie nie ma kruszarki")
        return out
    c = site["crusher"]
    c_node, _ = g.attach((c["lat"], c["lon"]))
    attached = {c_node}
    for e in site.get("excavators", []):
        attached.add(g.attach((e["lat"], e["lon"]))[0])
    elev, warns = g.elevations(c_node, float(c.get("elev_m", 0.0)))
    out["elev"], out["warnings"] = elev, warns
    adj = g.adj()
    for n in range(len(g.nodes)):
        deg = len(adj.get(n, []))
        if deg >= 3:
            out["junctions"].append(n)
        elif deg == 1 and n not in attached:
            out["dead_ends"].append(n)
    unreachable = {g.roads[ri].get("name") or g.roads[ri].get("id")
                   for (a, b), (w, ri) in g.edges.items() if a not in elev}
    for name in sorted(unreachable):
        out["warnings"].append(f"Droga „{name}” nie ma połączenia z kruszarką")
    for e in site.get("excavators", []):
        n, d = g.attach((e["lat"], e["lon"]))
        out["levels"][e["id"]] = {"elev_m": round(elev[n], 1) if n in elev else None, "dist_to_road_m": round(d)}
        if n not in elev:
            out["warnings"].append(f"Koparka {e['id']}: brak dojazdu do kruszarki")
        elif d > 150:
            out["warnings"].append(f"Koparka {e['id']} stoi {d:.0f} m od najbliższej drogi")
    return out


def build_route(site: dict, start: tuple[float, float], route_id: str, end=None, end_name="plac_kruszarki") -> dict:
    """Route from a loading point to the crusher (or to `end`)."""
    roads = site.get("roads", [])
    if not roads:
        raise ValueError("Na mapie nie ma dróg")
    if not site.get("crusher"):
        raise ValueError("Na mapie nie ma kruszarki")
    c = site["crusher"]
    g = RoadGraph(roads)
    c_node, _ = g.attach((c["lat"], c["lon"]))
    target = tuple(end) if end is not None else (c["lat"], c["lon"])
    # attach the end points first: splitting a road creates nodes that need an elevation too
    g.attach(tuple(start))
    g.attach(target)
    elev, _ = g.elevations(c_node, float(c.get("elev_m", 0.0)))
    r = path_between(g, start, target, elev, end_name=end_name)
    if r is None:
        raise ValueError(f"{route_id}: brak połączenia drogowego – połącz drogi (koniec drogi ≤ {SNAP_M:.0f} m "
                         "od innej drogi tworzy węzeł)")
    return r


def site_to_scenario(site: dict, raw: dict) -> dict:
    """Replace blasts/excavators/routes of a raw scenario with what is drawn on the map.
    Trucks, crusher parameters, costs etc. stay as they are. Also builds links between every pair
    of excavators (for trucks that load at two piles in one trip)."""
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
        if p.get("psd"):
            b["psd"] = p["psd"]
        new["blasts"].append(b)
    pile_ids = {p["id"] for p in piles}
    new["excavators"], new["routes"], new["links"] = [], {}, {}
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
    for a in excavators:
        for b in excavators:
            if a["id"] != b["id"]:
                try:
                    new["links"][f"{a['id']}>{b['id']}"] = build_route(
                        site, (a["lat"], a["lon"]), f"{a['id']}>{b['id']}", end=(b["lat"], b["lon"]), end_name="dojazd")
                except ValueError:
                    pass  # not connected – such a pair can't be used for two-pile trips
    # keep fleet assignments only if they still point to existing excavators
    ex_ids = {e["id"] for e in excavators}
    for f in new.get("trucks", []):
        if f.get("excavators"):
            f["excavators"] = [x for x in f["excavators"] if x in ex_ids]
        if f.get("blend"):
            f["blend"] = {k: v for k, v in f["blend"].items() if k in ex_ids}
    new["site"] = site
    return new
