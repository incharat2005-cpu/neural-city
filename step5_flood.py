import matplotlib.pyplot as plt
import osmnx as ox
from flood import load_graph, compute_susceptibility, flood_depths, edge_depth

g = load_graph()
susc = compute_susceptibility(g)

fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="black")
for ax, rain in zip(axes, [40, 80, 120]):
    depths = flood_depths(susc, rain)
    colors, flooded, slow = [], 0, 0
    for u, v, k in g.edges(keys=True):
        d = edge_depth(depths, u, v)
        if d >= 15:
            colors.append("#e63946"); flooded += 1
        elif d >= 7:
            colors.append("#f4a261"); slow += 1
        else:
            colors.append("#6c757d")
    print(f"Rain {rain} mm/hr -> flooded roads: {flooded}, slow roads: {slow}")
    ox.plot_graph(g, ax=ax, node_size=0, edge_color=colors, edge_linewidth=0.8,
                  bgcolor="black", show=False, close=False)
    ax.set_facecolor("black")
    ax.set_title(f"Rain {rain} mm/hr", color="white", fontsize=16)
plt.savefig("flood_map.png", dpi=130, facecolor="black")
print("Saved flood_map.png")