import time
import requests
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

# Load the road network saved in Step 2
graph = ox.load_graphml("koramangala.graphml")
nodes = list(graph.nodes(data=True))
print("Fetching elevation for", len(nodes), "intersections...")


def fetch_open_meteo(chunk):
    lats = ",".join(f"{float(d['y']):.5f}" for _, d in chunk)
    lons = ",".join(f"{float(d['x']):.5f}" for _, d in chunk)
    r = requests.get("https://api.open-meteo.com/v1/elevation",
                     params={"latitude": lats, "longitude": lons}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"open-meteo status {r.status_code}: {r.text[:150]}")
    return r.json()["elevation"]


def fetch_opentopo(chunk):
    locs = "|".join(f"{float(d['y']):.5f},{float(d['x']):.5f}" for _, d in chunk)
    r = requests.get("https://api.opentopodata.org/v1/srtm90m",
                     params={"locations": locs}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"opentopodata status {r.status_code}: {r.text[:150]}")
    return [item["elevation"] for item in r.json()["results"]]


services = [("open-meteo", fetch_open_meteo), ("opentopodata", fetch_opentopo)]
elevation = {}
CHUNK = 50

for i in range(0, len(nodes), CHUNK):
    chunk = nodes[i:i + CHUNK]
    values = None
    for name, fetch in services:
        for attempt in range(2):
            try:
                values = fetch(chunk)
                break
            except Exception as error:
                print(f"  {name} failed (try {attempt + 1}): {str(error)[:200]}")
                time.sleep(3)
        if values is not None:
            break
    if values is None:
        raise RuntimeError("Both elevation services failed. Copy the lines above and send them to me.")

    for (node_id, _), height in zip(chunk, values):
        elevation[node_id] = height
    print("Done", min(i + CHUNK, len(nodes)), "of", len(nodes))
    time.sleep(1.2)

# Fill any missing heights with the average
known = [h for h in elevation.values() if h is not None]
average = sum(known) / len(known)
for k in elevation:
    if elevation[k] is None:
        elevation[k] = average

# Attach the heights to the graph and save it
nx.set_node_attributes(graph, elevation, "elevation")
ox.save_graphml(graph, "koramangala_elev.graphml")

heights = list(elevation.values())
print("Lowest point (m):", min(heights))
print("Highest point (m):", max(heights))
print("Average (m):", round(sum(heights) / len(heights), 1))

# Draw the map: blue = low ground, red = high ground
xs = [float(d["x"]) for _, d in nodes]
ys = [float(d["y"]) for _, d in nodes]
zs = [elevation[n] for n, _ in nodes]
plt.figure(figsize=(9, 9))
scatter = plt.scatter(xs, ys, c=zs, cmap="RdYlBu_r", s=6)
plt.colorbar(scatter, label="Height above sea level (m)")
plt.title("Koramangala terrain: blue = low ground")
plt.axis("equal")
plt.savefig("koramangala_elevation.png", dpi=150)
print("Saved koramangala_elevation.png")