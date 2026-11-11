import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree

CENTER_LAT, CENTER_LON = 12.9352, 77.6245


def load_graph(path="koramangala_elev.graphml"):
    g = ox.load_graphml(path)
    for _, d in g.nodes(data=True):
        d["elevation"] = float(d["elevation"])
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])
    return g


def _xy_metres(g, nodes):
    pts = []
    for n in nodes:
        lon, lat = g.nodes[n]["x"], g.nodes[n]["y"]
        pts.append(((lon - CENTER_LON) * 111320 * np.cos(np.radians(CENTER_LAT)),
                    (lat - CENTER_LAT) * 110574))
    return np.array(pts)


def compute_susceptibility(g):
    """0..1 score per intersection: how likely water collects there."""
    nodes = list(g.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    elev = np.array([g.nodes[n]["elevation"] for n in nodes])
    tree = cKDTree(_xy_metres(g, nodes))
    pts = tree.data

    smooth = np.array([elev[tree.query_ball_point(p, 120)].mean() for p in pts])
    smooth = smooth + np.random.default_rng(42).uniform(0, 0.05, len(smooth))
    local_mean = np.array([elev[tree.query_ball_point(p, 400)].mean() for p in pts])
    depression = np.clip((local_mean - elev) / 6.0, 0, 1)

    und = g.to_undirected()
    acc = np.ones(len(nodes))
    for i in np.argsort(-smooth):               # highest ground first
        lower = [index[m] for m in und.neighbors(nodes[i]) if smooth[index[m]] < smooth[i]]
        if lower:
            j = min(lower, key=lambda k: smooth[k])
            acc[j] += acc[i]                    # pass water downhill
    acc_norm = np.log(acc) / np.log(acc.max())

    susc = 0.5 * acc_norm + 0.5 * depression
    susc = (susc - susc.min()) / (susc.max() - susc.min())
    return {n: float(susc[i]) for n, i in index.items()}


def flood_depths(susc, rain_mm_hr, drainage_mm_hr=20.0, boost=None):
    """Water depth in cm at each intersection."""
    boost = boost or {}
    out = {}
    for n, s in susc.items():
        excess = max(0.0, rain_mm_hr - drainage_mm_hr - boost.get(n, 0.0))
        out[n] = excess * (0.05 + 0.35 * s)
    return out


def edge_depth(depths, u, v):
    return max(depths[u], depths[v])