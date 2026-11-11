import osmnx as ox
import matplotlib.pyplot as plt

# Centre of Koramangala, Bengaluru (latitude, longitude)
center = (12.9352, 77.6245)

# Download the drivable road network within 2000 metres of the centre
graph = ox.graph_from_point(center, dist=2000, network_type="drive")

# Print basic facts so we know it worked
print("Intersections (nodes):", len(graph.nodes))
print("Road segments (edges):", len(graph.edges))

# Save the graph so we don't have to download it again
ox.save_graphml(graph, "koramangala.graphml")

# Draw the map and save it as a picture
fig, ax = ox.plot_graph(graph, node_size=4, edge_linewidth=0.8, show=False, close=False)
plt.savefig("koramangala_map.png", dpi=150)
print("Saved koramangala_map.png") 