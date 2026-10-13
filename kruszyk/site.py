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
        - {id, name, kind: droga|rampa, grade_draw_pct, speed_limit_kmh, single_lane, coords: [[lat, lon], ...]}
      passing_bays:  # mijanki on single-lane roads
        - {lat, lon}

A single-lane road (`single_lane: true`) carries traffic one direction at a time between passing bays
(and road ends): every such stretch becomes a `section` of the routes; the simulation lets trucks into
it in one direction only, the truck going uphill has priority, the one coming down waits at the bay.

`grade_draw_pct` of a road is its grade IN THE DIRECTION IT WAS DRAWN (+ = the road climbs from its first
to its last point, - = it descends). Older maps used `grade_pct` = grade towards the crusher; they are
converted by `normalize_roads`. Roads form a network: an end of a road placed on another
road (≤ SNAP_M) – also in the middle of it – becomes a junction. Elevations of all junctions are
computed from the crusher (crusher.elev_m, default 0) along the grades, so a branch half-way down
a ramp starts at the right level and every excavator knows its level.
"""
from __future__ import annotations

import bisect
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


def along_m(coords: list, p) -> tuple[float, float]:
    """(distance along the polyline of the projection of p, distance of p from the polyline)."""
    best, acc = (0.0, math.inf), 0.0
    for a, b in zip(coords, coords[1:]):
        q, t, d = project_on_segment(p, tuple(a), tuple(b))
        seg = haversine_m(a, b)
        if d < best[1]:
            best = (acc + t * seg, d)
        acc += seg
    return best


def bays_on_road(site_or_bays, road: dict) -> list[float]:
    """Positions (m along the road) of the passing bays lying on a single-lane road."""
    bays = site_or_bays.get("passing_bays", []) if isinstance(site_or_bays, dict) else (site_or_bays or [])
    out = []
    for b in bays:
        a, d = along_m(road["coords"], (b["lat"], b["lon"]))
        if d <= SNAP_M:
            out.append(a)
    return sorted(out)


class RoadGraph:
    """Road polylines as a graph.

    * vertices of different roads within SNAP_M are merged into one node,
    * a dead end lying on another road (also mid-segment) splits that road -> T-junction,
    * extra points (excavators, crusher) are attached to the closest point of the network.
    """

    def __init__(self, roads: list[dict], include_planned: bool = False, bays: list | None = None):
        self.bays = bays or []
        if not include_planned:
            roads = [r for r in roads if r.get("active", True)]  # planned / closed roads are not driven on
        self.nodes: list[tuple[float, float]] = []
        self.edges: dict[tuple[int, int], tuple[float, int]] = {}  # (u<v) -> (length, road index)
        self.first: dict[tuple[int, int], int] = {}  # edge -> its node that comes first in the drawing
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
        """Edge drawn from a to b."""
        key = (min(a, b), max(a, b))
        self.edges[key] = (haversine_m(self.nodes[a], self.nodes[b]), ri)
        self.first[key] = a

    def degree(self, n: int) -> int:
        return sum(1 for (a, b) in self.edges if n in (a, b))

    def _split(self, edge: tuple[int, int], n: int):
        _, ri = self.edges.pop(edge)
        a = self.first.pop(edge)
        b = edge[1] if a == edge[0] else edge[0]
        self._add_edge(a, n, ri)  # keep the drawing direction on both halves
        self._add_edge(n, b, ri)

    def dz(self, u: int, v: int, w: float, ri: int) -> float:
        """Height change driving u -> v along a road edge of length w."""
        road = self.roads[ri]
        if "grade_draw_pct" in road:
            g = float(road.get("grade_draw_pct") or 0.0) * w / 100
            return g if self.first.get((min(u, v), max(u, v)), u) == u else -g
        return -float(road.get("grade_pct", 0.0)) * w / 100  # legacy: grade towards the crusher

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
                    elev[v] = elev[u] + self.dz(u, v, w, ri)
                    heapq.heappush(pq, (d + w, v))
        warnings = []
        for (a, b), (w, ri) in self.edges.items():
            if a in elev and b in elev:
                g = self.dz(a, b, w, ri)
                if g == 0 and abs(elev[a] - elev[b]) > 3:
                    name = self.roads[ri].get("name") or self.roads[ri].get("id")
                    warnings.append(f"Droga „{name}”: płaska, a jej końce wyszły na różnych wysokościach "
                                    f"({elev[a]:.0f} / {elev[b]:.0f} m) – sprawdź nachylenia w pętli")
        return elev, sorted(set(warnings))


def normalize_roads(site: dict) -> dict:
    """Give every road `grade_draw_pct` (grade along its drawing direction). Old roads stored `grade_pct`
    = grade towards the crusher: the magnitude is kept, the sign follows whether the road was drawn
    towards the crusher (its last point is nearer the crusher along the network) or away from it."""
    roads = site.get("roads", [])
    legacy = [r for r in roads if "grade_draw_pct" not in r]
    if not legacy:
        return site
    dist = {}
    if site.get("crusher"):
        g = RoadGraph(roads, include_planned=True)
        c = site["crusher"]
        root, _ = g.attach((c["lat"], c["lon"]))
        adj = g.adj()
        dist, pq = {root: 0.0}, [(0.0, root)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w, _ in adj.get(u, []):
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))

    def node(p):
        return next((n for n, q in enumerate(g.nodes) if haversine_m(p, q) < SNAP_M), None) if dist else None

    for r in legacy:
        a, b = node(tuple(r["coords"][0])), node(tuple(r["coords"][-1]))
        towards = a in dist and b in dist and dist[b] < dist[a]
        r["grade_draw_pct"] = float(r.get("grade_pct", 0.0)) * (1 if towards else -1) + 0.0
        r.pop("grade_pct", None)
    return site


def _grade(dz: float, length: float) -> float:
    return round(100 * dz / length, 2) if length > 0 else 0.0


def _lane_pieces(g: RoadGraph, u: int, v: int, w: float, dz: float, ri: int) -> list:
    """An edge u->v split at the passing bays of a single-lane road: [(length, dz, section key, direction)].
    Section key = road id + number of the stretch between bays; direction +1 = along the road drawing."""
    road = g.roads[ri]
    if not road.get("single_lane"):
        return [(w, dz, None, 0)]
    bays = bays_on_road(g.bays, road)
    au, _ = along_m(road["coords"], g.nodes[u])
    av, _ = along_m(road["coords"], g.nodes[v])
    sdir = 1 if av >= au else -1
    cuts = sorted((a for a in bays if min(au, av) < a < max(au, av)), reverse=sdir < 0)
    pts = [au, *cuts, av]
    span = abs(av - au) or 1.0
    rid = road.get("id") or f"droga{ri}"
    out = []
    for a, b in zip(pts, pts[1:]):
        f = abs(b - a) / span
        k = bisect.bisect_right(bays, (a + b) / 2)
        out.append((w * f, dz * f, f"{rid}#{k}", sdir))
    return [p for p in out if p[0] > 0.01] or [(w, dz, f"{rid}#{bisect.bisect_right(bays, au)}", sdir)]


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
        for piece_w, piece_dz, section, sdir in _lane_pieces(g, u, v, w, dz, ri):
            if run and run["_road"] == ri and run.get("section") == section:
                run["length_m"] += piece_w
                run["_dz"] += piece_dz
            else:
                run = {"name": name, "length_m": piece_w, "_dz": piece_dz,
                       "speed_limit_kmh": float(road.get("speed_limit_kmh", 30.0)), "_road": ri,
                       "kind": road.get("kind", "droga")}
                if section:
                    run["section"], run["section_dir"] = section, sdir
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
    normalize_roads(site)
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
    normalize_roads(site)
    roads = site.get("roads", [])
    if not roads:
        raise ValueError("Na mapie nie ma dróg")
    if not site.get("crusher"):
        raise ValueError("Na mapie nie ma kruszarki")
    c = site["crusher"]
    g = RoadGraph(roads, bays=site.get("passing_bays"))
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


DIG_STEPS = 30  # the digging line is divided into this many relocations at most


def dig_points(e: dict, tonnes: float | None = None) -> list[tuple[float, float]]:
    """Evenly spaced points of an excavator's digging line (start first), or [] without a line.
    One point per relocation: the excavator's share of the pile / tonnes between relocations."""
    path = e.get("dig_path") or []
    if len(path) < 2:
        return []
    L = polyline_length_m(path)
    every = float(e.get("relocate_every_t") or 600.0)
    n = math.ceil(tonnes / every) if tonnes else DIG_STEPS
    n = max(1, min(DIG_STEPS, n, int(L // 3) or 1))
    return [point_along(path, L * i / n) for i in range(n + 1)]


BLAST_TIMES = ("crew_arrive", "fire_from", "fire_to", "evacuation_min", "crew_leave")


def loading_point(site: dict, polygon: list) -> tuple[float, float]:
    """Where trucks load at a pile: its corner (or centre) closest to the road network."""
    cands = [tuple(c) for c in polygon] + [centroid(polygon)]
    roads = [r for r in site.get("roads", []) if r.get("active", True) and len(r.get("coords", [])) >= 2]
    if not roads:
        return cands[-1]
    return min(cands, key=lambda q: min(along_m(r["coords"], q)[1] for r in roads))


def _crusher_routes(site: dict, start, rid: str, routes: dict) -> dict:
    """Routes from a loading point to every additional crusher: {crusher id: route id}."""
    out = {}
    for c in site.get("crushers", []):
        r_id = f"{rid}>{c['id']}"
        try:
            routes[r_id] = build_route(site, tuple(start), r_id, end=(c["lat"], c["lon"]))
        except ValueError:
            continue
        out[c["id"]] = r_id
    return out


def site_to_scenario(site: dict, raw: dict) -> dict:
    """Replace blasts/excavators/routes of a raw scenario with what is drawn on the map.
    Trucks, crusher parameters, costs etc. stay as they are. Also builds links between every pair
    of excavators (for trucks that load at two piles in one trip)."""
    normalize_roads(site)
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
        if p.get("target_share"):
            b["target_share"] = float(p["target_share"])
        new["blasts"].append(b)
    pile_ids = {p["id"] for p in piles}
    new["excavators"], new["routes"], new["links"] = [], {}, {}
    for e in excavators:
        pos = (e["lat"], e["lon"])
        pile = e.get("pile")
        if pile not in pile_ids:  # nearest pile by centroid
            pile = min(piles, key=lambda p: haversine_m(pos, centroid(p["polygon"])))["id"]
        rid = f"T-{e['id']}"
        share = next((float(p.get("tonnage_t", 0)) for p in piles if p["id"] == pile), 0.0) / max(
            1, sum(1 for x in excavators if (x.get("pile") if x.get("pile") in pile_ids else None) == pile))
        dig = dig_points(e, share)
        if dig:
            pos = dig[0]
        new["routes"][rid] = build_route(site, pos, rid)
        ex = {"id": e["id"], "bucket_m3": float(e.get("bucket_m3", 5.0)),
              "swing_cycle_s": float(e.get("swing_cycle_s", 30.0)), "blast": pile, "route": rid}
        if dig:
            ex["dig_routes"], ex["dig_path"] = [rid], [list(p) for p in e["dig_path"]]
            for i, p in enumerate(dig[1:], start=1):
                new["routes"][f"{rid}@{i}"] = build_route(site, p, f"{rid}@{i}")
                ex["dig_routes"].append(f"{rid}@{i}")
            for key in ("relocate_every_t", "relocate_min"):
                if e.get(key) is not None:
                    ex[key] = float(e[key])
        if e.get("spotting"):
            ex["spotting"] = {k: float(v) for k, v in e["spotting"].items() if v}
        cr = _crusher_routes(site, pos, rid, new["routes"])
        if cr:
            ex["crusher_routes"] = cr
        new["excavators"].append(ex)
    for a in excavators:
        for b in excavators:
            if a["id"] != b["id"]:
                try:
                    new["links"][f"{a['id']}>{b['id']}"] = build_route(
                        site, (a["lat"], a["lon"]), f"{a['id']}>{b['id']}", end=(b["lat"], b["lon"]), end_name="dojazd")
                except ValueError:
                    pass  # not connected – such a pair can't be used for two-pile trips
    new["planned_blasts"] = []
    for p in site.get("planned_piles", []):  # future blasts: routes from their loading point
        if len(p.get("polygon") or []) < 3:
            continue
        load_at = loading_point(site, p["polygon"])
        rid = f"P-{p['id']}"
        try:
            new["routes"][rid] = build_route(site, load_at, rid)
        except ValueError:
            continue  # not connected to the roads yet
        pb = {"id": p["id"], "tonnage_t": float(p.get("tonnage_t", 0)), "material": p.get("material", ""),
              "blast_day": int(p.get("blast_day") or 1), "route": rid,
              "crusher_routes": _crusher_routes(site, load_at, rid, new["routes"]),
              "fragmentation": float(p.get("fragmentation", 0.3))}
        for key in ("psd", "loose_density_t_m3", "target_share", "polygon", "t_per_m", "drilled", *BLAST_TIMES):
            if p.get(key):
                pb[key] = p[key]
        new["planned_blasts"].append(pb)
    new["areas"] = [{k: v for k, v in a.items() if k != "uid"} for a in site.get("areas", [])
                    if len(a.get("polygon") or []) >= 3 and a.get("active", True)]
    # crushers: the main one keeps its parameters from the scenario; more crushers come from the map
    main = site.get("crusher") or {}
    new["crusher"] = {**raw.get("crusher", {}), "id": main.get("id") or raw.get("crusher", {}).get("id") or "C-1",
                      "accepts": list(main.get("accepts") or []), "product": main.get("product", ""),
                      "target_t": float(main.get("target_t") or 0.0)}
    base = {k: v for k, v in new["crusher"].items()
            if k not in ("id", "accepts", "capacity_tph", "hopper_t", "product", "target_t")}
    new["crushers_extra"] = [{**base, "id": c["id"], "capacity_tph": float(c.get("capacity_tph", 800)),
                              "hopper_t": float(c.get("hopper_t", 150)), "accepts": list(c.get("accepts") or []),
                              "product": c.get("product", ""), "target_t": float(c.get("target_t") or 0.0)}
                             for c in site.get("crushers", [])]
    # keep fleet assignments only if they still point to existing excavators
    ex_ids = {e["id"] for e in excavators}
    for f in new.get("trucks", []):
        if f.get("excavators"):
            f["excavators"] = [x for x in f["excavators"] if x in ex_ids]
        if f.get("blend"):
            f["blend"] = {k: v for k, v in f["blend"].items() if k in ex_ids}
    new["site"] = site
    return new


# ---------------------------------------------------------------- dedicated routes, excavator moves
LIGHT_ROLES = {"ekipa": "ekipa strzałowa", "sztygar": "sztygar", "cysterna": "cysterna"}
ON_ROAD_M = 20.0  # a dedicated route closer than this to a haul road shares that road (meets trucks there)


def _network(site: dict):
    """Graph of the roads in use with elevations from the crusher (None when not drawable yet)."""
    if not site.get("roads") or not site.get("crusher"):
        return None, {}
    normalize_roads(site)
    g = RoadGraph(site["roads"])
    c = site["crusher"]
    c_node, _ = g.attach((c["lat"], c["lon"]))
    elev, _ = g.elevations(c_node, float(c.get("elev_m", 0.0)))
    return g, elev


def _reverse_route(r: dict) -> dict:
    return {**r, "path": list(reversed(r["path"])),
            "segments": [{**s, "grade_pct": -s["grade_pct"], **({"section_dir": -s["section_dir"]} if "section_dir" in s else {})}
                         for s in reversed(r["segments"])]}


def polyline_route(site: dict, coords: list, name: str = "trasa dedykowana", step_m: float = 15.0,
                   break_points: list | None = None) -> dict:
    """A route along a drawn polyline. Parts lying on haul roads take that road's name, limit and
    grade (from the elevations), so light vehicles meet trucks there; parts off the roads keep `name`.
    `break_points` – points where a segment must end (stops of a round)."""
    g, elev = _network(site)
    samples = []  # (point, road index or None)
    for a, b in zip(coords, coords[1:]):
        d = haversine_m(a, b)
        n = max(1, int(d // step_m))
        for i in range(n):
            f = i / n
            samples.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    samples.append(tuple(coords[-1]))
    info = []
    last_z = 0.0
    for p in samples:
        ri, z = None, None
        if g is not None and g.edges:
            edge, q, t, d = g._nearest_edge(p)
            if d < ON_ROAD_M:
                ri = g.edges[edge][1]
                if edge[0] in elev and edge[1] in elev:
                    z = elev[edge[0]] + t * (elev[edge[1]] - elev[edge[0]])
        last_z = z if z is not None else last_z
        info.append((p, ri, last_z))
    # forced breaks at the samples closest to the stop points
    forced = set()
    for bp in break_points or []:
        forced.add(min(range(len(info)), key=lambda i: haversine_m(info[i][0], bp)))
    segments, cur = [], None
    for i in range(1, len(info)):
        (p0, ri0, z0), (p1, ri1, z1) = info[i - 1], info[i]
        road = g.roads[ri1] if (g is not None and ri1 is not None) else None
        nm = (road.get("name") or road.get("id")) if road else name
        limit = float(road.get("speed_limit_kmh", 30)) if road else 20.0
        w = haversine_m(p0, p1)
        if cur and cur["name"] == nm and (i - 1) not in forced:
            cur["length_m"] += w
            cur["_dz"] += z1 - z0
        else:
            cur = {"name": nm, "length_m": w, "_dz": z1 - z0, "speed_limit_kmh": limit}
            segments.append(cur)
    for sgm in segments:
        sgm["grade_pct"] = _grade(sgm.pop("_dz"), sgm["length_m"])
        sgm["length_m"] = round(sgm["length_m"], 1)
    route = {"segments": [s for s in segments if s["length_m"] > 0.5], "path": [list(c) for c in coords]}
    if break_points:
        # stop after the segment whose end is nearest to each break point (in order along the route)
        cum, ends = 0.0, []
        for sgm in route["segments"]:
            cum += sgm["length_m"]
            ends.append(cum)
        total = cum
        geo = polyline_length_m(coords)
        stops = []
        for bp in break_points:
            j = min(range(len(info)), key=lambda i: haversine_m(info[i][0], bp))
            d_along = sum(haversine_m(info[i - 1][0], info[i][0]) for i in range(1, j + 1)) * (total / geo if geo else 1)
            stops.append(min(range(len(ends)), key=lambda k: abs(ends[k] - d_along)))
        route["stop_after"] = stops
    return route


def dedicated(site: dict, role: str) -> list | None:
    """Coordinates of the route drawn for a role ('ekipa', 'sztygar', 'cysterna' or an excavator id)."""
    for d in site.get("dedicated", []):
        if d.get("for") == role and len(d.get("coords", [])) >= 2:
            return d["coords"]
    return None


def crew_route(site: dict) -> dict | None:
    """Blasting crew: blast point -> exit (driven backwards when arriving). Dedicated route if drawn."""
    if not site.get("blast"):
        return None
    bp = (site["blast"]["lat"], site["blast"]["lon"])
    coords = dedicated(site, "ekipa")
    if coords:
        # orient: start at the blast point
        if haversine_m(coords[-1], bp) < haversine_m(coords[0], bp):
            coords = list(reversed(coords))
        return polyline_route(site, coords, "trasa ekipy")
    try:
        return build_route(site, bp, "strzelanie")
    except ValueError:
        return None


def round_route(site: dict, role: str, stops: list[tuple[float, float]]) -> dict | None:
    """Dedicated round (foreman / tanker) with stops at the points closest to the given excavators."""
    coords = dedicated(site, role)
    if not coords:
        return None
    return polyline_route(site, coords, f"trasa: {LIGHT_ROLES.get(role, role)}", break_points=stops)


def excavator_move(site: dict, ex: dict, route_to_crusher_path: list | None, retreat_m: float = 200.0) -> dict | None:
    """How an excavator gets to its place for blasting/refuelling.
    Parking marker -> dedicated route if drawn, else shortest road path, else straight line.
    No marker -> it backs off `retreat_m` along its own route towards the crusher."""
    pos = (ex["lat"], ex["lon"])
    park = next((p for p in site.get("parking", []) if p.get("excavator") == ex["id"]), None)
    if park:
        target = (park["lat"], park["lon"])
        coords = dedicated(site, ex["id"])
        if coords:
            if haversine_m(coords[-1], pos) < haversine_m(coords[0], pos):
                coords = list(reversed(coords))
            r = polyline_route(site, coords, f"przejazd {ex['id']}")
        else:
            try:
                r = build_route(site, pos, ex["id"], end=target, end_name="miejsce postoju")
            except ValueError:
                d = haversine_m(pos, target)
                r = {"segments": [{"name": f"przejazd {ex['id']}", "length_m": round(d, 1), "grade_pct": 0.0,
                                   "speed_limit_kmh": 5.0}], "path": [list(pos), list(target)]}
        r["parking"] = True
        return r
    if not route_to_crusher_path or len(route_to_crusher_path) < 2:
        return None
    # back off along the route towards the crusher
    path, acc = [list(route_to_crusher_path[0])], 0.0
    for a, b in zip(route_to_crusher_path, route_to_crusher_path[1:]):
        d = haversine_m(a, b)
        if acc + d >= retreat_m:
            f = (retreat_m - acc) / d if d else 0
            path.append([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])])
            acc = retreat_m
            break
        path.append(list(b))
        acc += d
    return {"segments": [{"name": "odjazd z frontu", "length_m": round(acc, 1), "grade_pct": 0.0,
                          "speed_limit_kmh": 5.0}], "path": path, "parking": False}


# ---------------------------------------------------------------- extraction areas: automatic blasts
def _clip(poly: list, axis: tuple[float, float], lo: float, hi: float) -> list:
    """Part of a polygon (local metres) between lo <= dot(p, axis) <= hi (Sutherland–Hodgman, two half-planes)."""
    def cut(pts, keep, edge):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            ci, pi = keep(cur), keep(prev)
            if ci != pi:
                sp, sc_ = edge(prev), edge(cur)
                t = sp / (sp - sc_) if sp != sc_ else 0.0
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if ci:
                out.append(cur)
        return out

    dot = lambda p: p[0] * axis[0] + p[1] * axis[1]  # noqa: E731
    pts = cut(poly, lambda p: dot(p) >= lo, lambda p: dot(p) - lo)
    return cut(pts, lambda p: dot(p) <= hi, lambda p: dot(p) - hi) if pts else []


def area_strips(site: dict, polygon: list, n: int) -> list[list]:
    """Divide an extraction area into n blasts: strips across its longer axis, starting at the end nearer
    to the roads (that is where the first blast is fired). Returns lat/lon polygons in firing order."""
    if n < 1 or len(polygon) < 3:
        return []
    lat0, lon0 = centroid(polygon)
    k = math.cos(math.radians(lat0))
    to_xy = lambda c: (math.radians(c[1] - lon0) * R_EARTH * k, math.radians(c[0] - lat0) * R_EARTH)  # noqa: E731
    to_ll = lambda p: [round(lat0 + math.degrees(p[1] / R_EARTH), 7),  # noqa: E731
                       round(lon0 + math.degrees(p[0] / (R_EARTH * k)), 7)]
    xy = [to_xy(c) for c in polygon]
    # longer axis: principal direction of the vertices
    sxx = sum(x * x for x, _ in xy)
    syy = sum(y * y for _, y in xy)
    sxy = sum(x * y for x, y in xy)
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    axis = (math.cos(ang), math.sin(ang))
    proj = [x * axis[0] + y * axis[1] for x, y in xy]
    lo, hi = min(proj), max(proj)
    w = (hi - lo) / n
    strips = []
    for i in range(n):
        part = _clip(xy, axis, lo + i * w - 1e-6, lo + (i + 1) * w + 1e-6)
        if len(part) >= 3:
            strips.append([to_ll(p) for p in part])
    roads = [r for r in site.get("roads", []) if r.get("active", True) and len(r.get("coords", [])) >= 2]
    if roads and len(strips) > 1:
        def d(poly):
            c = centroid(poly)
            return min(along_m(r["coords"], c)[1] for r in roads)
        if d(strips[-1]) < d(strips[0]):
            strips.reverse()
    return strips
