import matplotlib.pyplot as plt
import osmnx as ox
from flood import load_graph, compute_susceptibility, flood_depths, edge_depth
from traffic import make_routing_graph, best_route

ORIGIN = (12.9400, 77.6350)     # (lat, lon) where the emergency starts (approximate)
HOSPITAL = (12.9300, 77.6200)   # approximate hospital location
RAIN, HOUR = 100, 18

g = load_graph()
susc = compute_susceptibility(g)
origin = ox.distance.nearest_nodes(g, X=ORIGIN[1], Y=ORIGIN[0])
dest = ox.distance.nearest_nodes(g, X=HOSPITAL[1], Y=HOSPITAL[0])

dry = {n: 0.0 for n in susc}
wet = flood_depths(susc, RAIN)

p0, t0, l0 = best_route(make_routing_graph(g, dry, HOUR), origin, dest)
p1, t1, l1 = best_route(make_routing_graph(g, wet, HOUR), origin, dest)

print(f"Normal route (no rain): {t0/60:.1f} min, {l0/1000:.2f} km")
hits = sum(1 for a, b in zip(p0[:-1], p0[1:]) if edge_depth(wet, a, b) >= 15)
print(f"With {RAIN} mm/hr rain the normal route would cross {hits} flooded road segments")
if p1:
    print(f"Flood-aware route: {t1/60:.1f} min, {l1/1000:.2f} km")
else:
    print("No flood-free route exists")

colors = ["#e63946" if edge_depth(wet, u, v) >= 15 else "#444444" for u, v, k in g.edges(keys=True)]
fig, ax = ox.plot_graph(g, node_size=0, edge_color=colors, edge_linewidth=0.7,
                        bgcolor="black", show=False, close=False)
for path, color in [(p0, "#2a9d8f"), (p1, "#00b4d8")]:
    if path:
        ax.plot([g.nodes[n]["x"] for n in path], [g.nodes[n]["y"] for n in path], color=color, linewidth=3)
plt.savefig("ambulance_routes.png", dpi=150, facecolor="black")
print("Saved ambulance_routes.png")