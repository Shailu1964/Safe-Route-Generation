import osmnx as ox
import networkx as nx
import joblib

# Download the drivable road network for Pune, India
print("Downloading road network for Pune, India...")
G = ox.graph_from_place("Pune, India", network_type="drive")

# Save the graph as a pickle file
output_file = "pune_graph.pkl"
joblib.dump(G, output_file)
print(f"Graph saved to {output_file}")
