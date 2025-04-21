from flask import Flask, request, jsonify
import osmnx as ox
import networkx as nx
import folium
import copy
from math import radians, cos, sin, sqrt, atan2
from geopy.geocoders import Nominatim
import pandas as pd
from datetime import datetime, timedelta
from utils import predict_crime_severity  # Import from utils.py

app = Flask(__name__)

# Load the crime data from the CSV file
crime_data = pd.read_csv('preprocessed_Pune_crime_data.csv')

# Normalize column names (lowercase, no spaces)
crime_data.columns = crime_data.columns.str.strip().str.lower()

# Ensure the 'date' column exists
if 'date of occurrence' not in crime_data.columns:
    raise KeyError("The 'date' column is missing from the crime data. Please ensure the CSV file contains a 'date' column.")

# Ensure the 'date' column is in datetime format
crime_data['date'] = pd.to_datetime(crime_data['date of occurrence'], errors='coerce', dayfirst=True)

# Define crime spots without severity from the dataset
crime_spots = [(row['latitude'], row['longitude']) for index, row in crime_data.iterrows()]

# Create graphs of Pune
G = ox.graph_from_place('Pune, India', network_type='drive')

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

def adjust_weights_for_crime(graph, crime_locations, radius):
    for crime_lat, crime_lon, severity in crime_locations:
        severity = float(severity)  # Ensure severity is numeric
        nearby_nodes = [node for node, data in graph.nodes(data=True)
                        if abs(data['y'] - crime_lat) < radius and abs(data['x'] - crime_lon) < radius]
        
        for node in nearby_nodes:
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:  # If edge exists
                    for key in edge_data:
                        edge_data[key]['length'] *= (1 + severity)
    return graph

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

# Predict crime severity dynamically for each crime spot
def get_dynamic_crime_spots(crime_locations, date, time, day):
    """Predict crime severity dynamically for each crime spot."""
    dynamic_crime_spots = []
    for lat, lon in crime_locations:
        severity = predict_crime_severity(lat, lon, date, time, day)
        dynamic_crime_spots.append((lat, lon, severity))
    return dynamic_crime_spots

# Update the graph for safest and optimized routes dynamically
def update_graphs_with_dynamic_severity(graph, crime_locations, date, time, day, radius):
    """Adjust graph weights dynamically based on predicted crime severity."""
    dynamic_crime_spots = get_dynamic_crime_spots(crime_locations, date, time, day)
    return adjust_weights_for_crime(graph, dynamic_crime_spots, radius)

# API to calculate routes dynamically
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

    # Get the current date, time, and day
    now = datetime.now()
    date = now.timestamp()
    time = now.hour
    day = now.weekday()

    # Update graphs dynamically using predicted crime severity
    safe_graph = update_graphs_with_dynamic_severity(copy.deepcopy(G), crime_spots, date, time, day, radius=0.005)
    optimized_graph = update_graphs_with_dynamic_severity(copy.deepcopy(G), crime_spots, date, time, day, radius=0.005)

    # Calculate routes
    fastest_route = get_astar_route(G, start_point, end_point)
    safest_route = get_astar_route(safe_graph, start_point, end_point)
    optimized_route = get_astar_route(optimized_graph, start_point, end_point)

    # Calculate distances
    fastest_route_distance = calculate_route_distance(G, fastest_route)
    safest_route_distance = calculate_route_distance(safe_graph, safest_route)
    optimized_route_distance = calculate_route_distance(optimized_graph, optimized_route)

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
