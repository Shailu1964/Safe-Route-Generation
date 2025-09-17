import osmnx as ox
import joblib

# Define the location for the graph
place_name = "Pune, India"
GRAPH_FILE = 'pune_graph.pkl'

print(f"Fetching the road network for {place_name}...")

# 1. Fetch a simplified road network.
# We fetch only the main arterial roads and highways, which are much less data-intensive.
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)

print("Original graph stats:")
print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")


# 2. Further simplify the graph by removing small, isolated road clusters.
# This gets rid of tiny road segments that aren't connected to the main network.
G_simplified = G.copy()
isolated_nodes = [node for node, degree in dict(G_simplified.degree()).items() if degree < 2]
G_simplified.remove_nodes_from(isolated_nodes)


print("Simplified graph stats:")
print(f"Nodes: {len(G_simplified.nodes())}, Edges: {len(G_simplified.edges())}")

# 3. Save the new, smaller graph file
print(f"Saving the simplified graph to {GRAPH_FILE}...")
joblib.dump(G_simplified, GRAPH_FILE)

print("\nSimplified graph generation complete!")
print(f"The new '{GRAPH_FILE}' is much smaller and will use less memory.")