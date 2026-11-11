from flood import load_graph, compute_susceptibility

g = load_graph()
s = compute_susceptibility(g)
values = list(s.values())
print("Intersections scored:", len(values))
print("Lowest score:", round(min(values), 2), " Highest score:", round(max(values), 2))
print("Intersections with score above 0.7:", sum(1 for v in values if v > 0.7))