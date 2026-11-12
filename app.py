import numpy as np
import osmnx as ox
import pandas as pd
import pydeck as pdk
import streamlit as st
from flood import load_graph, compute_susceptibility, flood_depths, edge_depth
from traffic import make_routing_graph, best_route

st.set_page_config(page_title="NEURAL CITY", layout="wide")

ORIGIN = (12.9400, 77.6350)
HOSPITAL = (12.9300, 77.6200)


@st.cache_resource
def setup():
    g = load_graph()
    susc = compute_susceptibility(g)
    origin = ox.distance.nearest_nodes(g, X=ORIGIN[1], Y=ORIGIN[0])
    dest = ox.distance.nearest_nodes(g, X=HOSPITAL[1], Y=HOSPITAL[0])
    total, pairs = {}, {}
    for u, v, d in g.edges(data=True):
        name = d.get("name")
        if isinstance(name, str) and not name.startswith("["):
            total[name] = total.get(name, 0) + float(d["length"])
            pairs.setdefault(name, []).append((u, v))
    top = sorted(total, key=total.get, reverse=True)[:25]
    return g, susc, origin, dest, {n: pairs[n] for n in top}


g, susc, origin, dest, roads = setup()

DEFAULTS = {"rain": 80, "drain": 20, "hour": 18, "extra": 0, "closed": []}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


st.title("NEURAL CITY: Koramangala urban digital twin")
st.sidebar.header("Scenario controls")
st.sidebar.button("Reset demo", on_click=reset)
rain = st.sidebar.slider("Rainfall (mm/hr)", 0, 150, key="rain")
drain = st.sidebar.slider("Drainage capacity (mm/hr)", 5, 60, key="drain")
hour = st.sidebar.slider("Hour of day", 0, 23, key="hour")
extra = st.sidebar.slider("What-if: extra drainage in the worst flood spots (mm/hr)", 0, 40, key="extra")
closed = st.sidebar.multiselect("What-if: close roads", list(roads), key="closed")

threshold = np.quantile(list(susc.values()), 0.9)
boost = {n: extra for n, s in susc.items() if s >= threshold}
depths = flood_depths(susc, rain, drain, boost)
blocked = [p for name in closed for p in roads[name]]
blocked_set = {frozenset(p) for p in blocked}

dry = {n: 0.0 for n in susc}
_, t_normal, _ = best_route(make_routing_graph(g, dry, hour), origin, dest)
r = make_routing_graph(g, depths, hour, blocked)
path, t_now, _ = best_route(r, origin, dest)

rows, flooded = [], 0
for u, v, k in g.edges(keys=True):
    depth = edge_depth(depths, u, v)
    if frozenset((u, v)) in blocked_set:
        color = [170, 90, 255]
    elif depth >= 15:
        color = [230, 57, 70]; flooded += 1
    else:
        e = r.get_edge_data(u, v, k)
        ratio = e["travel_s"] / e["free_s"] if e else 3
        color = [90, 200, 120] if ratio < 1.2 else [250, 200, 60] if ratio < 1.6 else [240, 120, 40]
    rows.append({"sx": g.nodes[u]["x"], "sy": g.nodes[u]["y"],
                 "tx": g.nodes[v]["x"], "ty": g.nodes[v]["y"], "color": color})
edges = pd.DataFrame(rows)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flooded roads", flooded)
c2.metric("Closed roads (what-if)", len(blocked_set))
c3.metric("Ambulance time, normal day", f"{t_normal/60:.1f} min")
if path:
    c4.metric("Ambulance time, now", f"{t_now/60:.1f} min", f"+{(t_now - t_normal)/60:.1f} min", delta_color="inverse")
else:
    c4.metric("Ambulance time, now", "No route")
    st.warning("No flood-free route exists for the ambulance in this scenario.")

layers = [pdk.Layer("LineLayer", edges, get_source_position="[sx, sy]",
                    get_target_position="[tx, ty]", get_color="color", get_width=3)]
if path:
    coords = [[g.nodes[n]["x"], g.nodes[n]["y"]] for n in path]
    layers.append(pdk.Layer("PathLayer", [{"path": coords}], get_path="path",
                            get_color=[0, 200, 255], width_min_pixels=6))
points = pd.DataFrame([{"lon": ORIGIN[1], "lat": ORIGIN[0], "c": [0, 255, 0]},
                       {"lon": HOSPITAL[1], "lat": HOSPITAL[0], "c": [255, 255, 255]}])
layers.append(pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]",
                        get_fill_color="c", get_radius=60))

flood_cols = pd.DataFrame([
    {"lon": g.nodes[n]["x"], "lat": g.nodes[n]["y"], "depth": float(dep)}
    for n, dep in depths.items() if dep >= 15
])
if len(flood_cols) > 0:
    layers.append(pdk.Layer("ColumnLayer", flood_cols, get_position="[lon, lat]",
                            get_elevation="depth", elevation_scale=4, radius=18,
                            get_fill_color=[230, 57, 70, 210], extruded=True))

view = pdk.ViewState(latitude=12.9352, longitude=77.6245, zoom=14.5, pitch=55)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="dark"))
st.caption("Red = flooded, purple = closed, orange/yellow = slow, green = free flow, "
           "cyan = ambulance route (green dot = start, white dot = hospital), "
           "red columns = flood depth in 3D. "
           "Rainfall and traffic are simulated; terrain is SRTM 90 m data.")