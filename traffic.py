import ast
import math
import networkx as nx
from flood import edge_depth

SPEEDS = {"motorway": 60, "trunk": 50, "primary": 40, "secondary": 35, "tertiary": 30,
          "unclassified": 25, "residential": 20, "service": 15, "living_street": 10}
BASE_LOAD = {"motorway": 0.8, "trunk": 0.8, "primary": 0.75, "secondary": 0.65,
             "tertiary": 0.5, "unclassified": 0.4, "residential": 0.3,
             "service": 0.2, "living_street": 0.15}


def road_class(data):
    h = data.get("highway", "residential")
    if isinstance(h, str) and h.startswith("["):
        try:
            h = ast.literal_eval(h)
        except Exception:
            pass
    if isinstance(h, list):
        h = h[0]
    return str(h).replace("_link", "")


def demand(hour):
    """Simulated traffic demand: morning (9) and evening (18) peaks."""
    return 0.4 + 0.9 * math.exp(-((hour - 9) ** 2) / 4) + 0.9 * math.exp(-((hour - 18) ** 2) / 4)


def flood_factor(depth_cm):
    """1 = normal speed, 0 = road impassable."""
    if depth_cm >= 15:
        return 0.0
    if depth_cm <= 3:
        return 1.0
    return 1.0 - 0.7 * (depth_cm - 3) / 12


def make_routing_graph(g, depths, hour=18, blocked=()):
    """Copy of the road graph with travel time per road. Flooded or closed roads are removed."""
    blocked = {frozenset(p) for p in blocked}
    r = g.copy()
    for u, v, k, d in list(r.edges(keys=True, data=True)):
        cls = road_class(d)
        free_s = float(d["length"]) / (SPEEDS.get(cls, 20) / 3.6)
        ff = flood_factor(edge_depth(depths, u, v))
        if ff == 0.0 or frozenset((u, v)) in blocked:
            r.remove_edge(u, v, k)
            continue
        ratio = BASE_LOAD.get(cls, 0.3) * demand(hour) / max(ff, 0.25)
        d["free_s"] = free_s
        d["travel_s"] = free_s * (1 + 0.15 * ratio ** 4) / ff   # BPR congestion formula
    return r


def best_route(r, origin, dest):
    try:
        path = nx.shortest_path(r, origin, dest, weight="travel_s")
    except nx.NetworkXNoPath:
        return None, None, None
    time_s, length = 0.0, 0.0
    for a, b in zip(path[:-1], path[1:]):
        e = min(r[a][b].values(), key=lambda x: x["travel_s"])
        time_s += e["travel_s"]
        length += float(e["length"])
    return path, time_s, length