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

# ---------- Mayor mode: where to send pumps ----------
from scipy.spatial import cKDTree
from flood import _xy_metres
from traffic import road_class, BASE_LOAD, demand

st.subheader("Mayor mode: where to send pumps")
pumps = st.slider("Mobile pumps available", 0, 10, 4)

node_list = list(g.nodes)
idx = {n: i for i, n in enumerate(node_list)}
xy = _xy_metres(g, node_list)
tree = cKDTree(xy)

load_sum, load_cnt = {}, {}
for u, v, d in g.edges(data=True):
    b = BASE_LOAD.get(road_class(d), 0.3)
    for n in (u, v):
        load_sum[n] = load_sum.get(n, 0) + b
        load_cnt[n] = load_cnt.get(n, 0) + 1
traffic_load = {n: load_sum[n] / load_cnt[n] * demand(hour) for n in load_sum}


def road_label(n):
    for _, _, d in list(g.edges(n, data=True)) + list(g.in_edges(n, data=True)):
        name = d.get("name")
        if isinstance(name, str) and not name.startswith("["):
            return name
    return "unnamed road"


priority = {n: depths[n] * (1 + traffic_load.get(n, 0.3)) for n in node_list if depths[n] >= 15}
chosen = []
for n in sorted(priority, key=priority.get, reverse=True):
    if len(chosen) >= pumps:
        break
    if all(np.linalg.norm(xy[idx[n]] - xy[idx[c]]) > 200 for c in chosen):
        chosen.append(n)

if not priority:
    st.info("No flooded intersections in this scenario, so no pumps are needed.")
elif not chosen:
    st.info("Set at least one pump to see suggestions.")
else:
    pump_boost = dict(boost)
    for c in chosen:
        for j in tree.query_ball_point(xy[idx[c]], 200):
            pump_boost[node_list[j]] = pump_boost.get(node_list[j], 0) + 30
    depths_after = flood_depths(susc, rain, drain, pump_boost)
    before = sum(1 for u, v, k in g.edges(keys=True) if edge_depth(depths, u, v) >= 15)
    after = sum(1 for u, v, k in g.edges(keys=True) if edge_depth(depths_after, u, v) >= 15)

    m1, m2, m3 = st.columns(3)
    m1.metric("Flooded roads without pumps", before)
    m2.metric("Flooded roads with pumps", after, f"{after - before}", delta_color="inverse")
    m3.metric("Pumps placed", len(chosen))

    table = pd.DataFrame([{
        "Rank": i + 1,
        "Near road": road_label(n),
        "Water depth (cm)": round(depths[n], 1),
        "Traffic load": round(traffic_load.get(n, 0.3), 2),
        "Action": "Deploy pump + divert traffic" if traffic_load.get(n, 0.3) > 0.5 else "Deploy pump",
    } for i, n in enumerate(chosen)])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Priority = water depth x traffic load, with pumps at least 200 m apart. "
               "Simple model: each pump adds 30 mm/hr of drainage within 200 m.")

    # ---------- AI insights ----------
from sklearn.ensemble import IsolationForest

st.subheader("AI insights")
tab1, tab2 = st.tabs(["Traffic forecast (graph neural network)",
                      "Infrastructure alerts (IoT anomaly detection)"])

with tab1:
    st.image("gnn_results.png")
    st.dataframe(pd.DataFrame({
        "Method": ["Persistence (no learning)", "Ridge regression (no graph)", "Graph neural network (GCN)"],
        "RMSE": [0.0391, 0.0374, 0.0315],
        "MAE": [0.0281, 0.0266, 0.0220],
    }), hide_index=True, use_container_width=True)
    st.caption("30-minute-ahead congestion forecast on unseen time. The graph neural network is 15.7% "
               "better than the best baseline. Trained on simulated traffic with spillover between "
               "neighbouring roads, so it shows the method works, not accuracy on real Bengaluru traffic.")

SENSORS = {
    "pipe_pressure_bar": ("Water pipe pressure", "Dispatch water-board crew: possible pipe leak"),
    "power_load_kw": ("Transformer load", "Alert electricity utility: transformer overload"),
    "drain_water_cm": ("Drain water level", "Clear drain and deploy pump: possible blockage"),
}


@st.cache_data
def sensor_alerts():
    data = pd.read_csv("sensor_data.csv", parse_dates=["time"])
    times = data["time"].tolist()
    rows = []
    for col, (label, action) in SENSORS.items():
        s = data[col]
        feats = pd.DataFrame({"value": s, "delta": s.diff().fillna(0),
                              "dev": s - s.rolling(24, min_periods=1).median()})
        flags = IsolationForest(contamination=0.10, random_state=0).fit_predict(feats) == -1
        start = None
        for i, f in enumerate(list(flags) + [False]):
            if f and start is None:
                start = i
            if not f and start is not None:
                if i - start >= 3:              # alert only on 3+ anomalies in a row
                    rows.append({"Sensor": label,
                                 "Alert from": times[start].strftime("%H:%M"),
                                 "Alert until": times[i - 1].strftime("%H:%M"),
                                 "Suggested action": action})
                start = None
    return pd.DataFrame(rows)


with tab2:
    alerts = sensor_alerts()
    if len(alerts) == 0:
        st.info("No alerts.")
    else:
        st.dataframe(alerts, hide_index=True, use_container_width=True)
    st.image("sensor_anomalies.png")
    st.caption("Simulated sensors (pipe pressure, transformer load, drain water level) with injected faults. "
               "Isolation Forest flags anomalies, and an alert is raised only when 3 or more flagged "
               "readings occur in a row, which filters out most false alarms.")
    