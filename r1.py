from flask import Flask, request, jsonify
import osmnx as ox
import networkx as nx
import folium
import copy
from math import radians, cos, sin, sqrt, atan2
from geopy.geocoders import Nominatim
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the crime data from the CSV file
crime_data = pd.read_csv('preprocessed_Pune_crime_data.csv')

# Normalize column names (lowercase, no spaces)
crime_data.columns = crime_data.columns.str.strip().str.lower()

# Convert crime severity categories into numerical values
severity_mapping = {"Very Low": 0.1,"Low": 0.2, "Moderate": 0.5, "High": 0.7, "Very High": 1.0}
crime_data['crime_severity'] = crime_data['crimeseverity'].map(severity_mapping).fillna(0.1)

# Define today's date (1 Jan 2025)
today = datetime(2025, 1, 1)

# Add a cutoff mapping for each severity level
severity_cutoff_mapping = {
    "Very Low": today - timedelta(days=30),  # 1 month
    "Low": today - timedelta(days=90),      # 3 months
    "Moderate": today - timedelta(days=180), # 6 months
    "High": today - timedelta(days=270),    # 9 months
    "Very High": today - timedelta(days=365) # 1 year
}

# Ensure the 'date' column exists
if 'date of occurrence' not in crime_data.columns:
    raise KeyError("The 'date' column is missing from the crime data. Please ensure the CSV file contains a 'date' column.")

# Ensure the 'date' column is in datetime format
crime_data['date'] = pd.to_datetime(crime_data['date of occurrence'], errors='coerce')

# Filter crime data based on severity and cutoff dates
filtered_crime_data = pd.concat([
    crime_data[(crime_data['crimeseverity'] == severity) & (crime_data['date'] >= cutoff_date)]
    for severity, cutoff_date in severity_cutoff_mapping.items()
])

# Define crime spots with crime severity from filtered data
crime_spots = [(row['latitude'], row['longitude'], row['crime_severity']) for index, row in filtered_crime_data.iterrows()]

# Create graphs of Pune
G = ox.graph_from_place('Pune, India', network_type='drive')
G_safe = copy.deepcopy(G)
G_optimized = copy.deepcopy(G)

# Define the radii for crime spots
crime_radius = 0.005  # Approx. 200 meters

# Geolocator for converting addresses to coordinates
geolocator = Nominatim(user_agent="geoapiExercises")

def get_coordinates(location_name):
    """Function to get latitude and longitude for a given place name."""
    location = geolocator.geocode(location_name)
    if location:
        return (location.latitude, location.longitude)
    return None

def remove_crime_spots(graph, crime_locations, radius, max_severity):
    nodes_to_remove = []
    for crime_lat, crime_lon, severity in crime_locations:
        if severity > max_severity:
            nearby_nodes = [node for node, data in graph.nodes(data=True)
                            if abs(data['y'] - crime_lat) < radius and abs(data['x'] - crime_lon) < radius]
            nodes_to_remove.extend(nearby_nodes)
    graph.remove_nodes_from(nodes_to_remove)
    return graph

def adjust_weights_for_crime(graph, crime_locations, radius):
    for crime_lat, crime_lon, severity in crime_locations:
        nearby_nodes = [node for node, data in graph.nodes(data=True)
                        if abs(data['y'] - crime_lat) < radius and abs(data['x'] - crime_lon) < radius]
        
        for node in nearby_nodes:
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:  # If edge exists
                    for key in edge_data:
                        edge_data[key]['length'] *= (1 + severity)
    return graph

# Update the graph for safest and optimized routes
G_safe = remove_crime_spots(G_safe, crime_spots, crime_radius, max_severity=0.3)
G_optimized = remove_crime_spots(G_optimized, crime_spots, crime_radius, max_severity=0.8)

# Haversine function to calculate straight-line distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Function to calculate the A* route
def get_astar_route(graph, start, end):
    start_node = ox.distance.nearest_nodes(graph, start[1], start[0])
    end_node = ox.distance.nearest_nodes(graph, end[1], end[0])
    
    def heuristic(u, v):
        lat_u = graph.nodes[u]['y']
        lon_u = graph.nodes[u]['x']
        lat_v = graph.nodes[v]['y']
        lon_v = graph.nodes[v]['x']
        return haversine(lat_u, lon_u, lat_v, lon_v)
    
    route = nx.astar_path(graph, start_node, end_node, heuristic=heuristic, weight='length')
    return route

# Function to calculate route distance
def calculate_route_distance(graph, route):
    total_distance = 0
    for i in range(len(route) - 1):
        edge_data = graph.get_edge_data(route[i], route[i + 1])[0]
        total_distance += edge_data['length']
    return total_distance

# API to calculate routes
@app.route('/calculate-routes', methods=['GET'])
def calculate_routes_api():
    source = request.args.get('source')
    destination = request.args.get('destination')

    if not source or not destination:
        return jsonify({"error": "Missing required parameters: source and destination"}), 400
    
    start_point = get_coordinates(source)
    end_point = get_coordinates(destination)

    if not start_point or not end_point:
        return jsonify({"error": "Invalid source or destination"}), 400

    # Calculate routes
    fastest_route = get_astar_route(G, start_point, end_point)
    safest_route = get_astar_route(G_safe, start_point, end_point)
    optimized_route = get_astar_route(G_optimized, start_point, end_point)

    # Calculate distances
    fastest_route_distance = calculate_route_distance(G, fastest_route)
    safest_route_distance = calculate_route_distance(G_safe, safest_route)
    optimized_route_distance = calculate_route_distance(G_optimized, optimized_route)

    return jsonify({
        "fastest_route_distance_km": fastest_route_distance / 1000,
        "safest_route_distance_km": safest_route_distance / 1000,
        "optimized_route_distance_km": optimized_route_distance / 1000,
        "fastest_route": fastest_route,
        "safest_route": safest_route,
        "optimized_route": optimized_route
    })

if __name__ == '__main__':
    app.run(debug=True)
