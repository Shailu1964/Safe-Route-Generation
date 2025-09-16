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
import joblib
import os
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache for crime severity predictions
CRIME_SEVERITY_CACHE = {}
CRIME_CACHE_EXPIRY = 300  # 5 minutes

app = Flask(__name__)

# Cache graph and crime data for faster loading
GRAPH_FILE = 'pune_graph.pkl'
CRIME_DATA_FILE = 'preprocessed_Pune_crime_data.pkl'

if os.path.exists(GRAPH_FILE):
    G = joblib.load(GRAPH_FILE)
else:
    G = ox.graph_from_place('Pune, India', network_type='drive')
    joblib.dump(G, GRAPH_FILE)

if os.path.exists(CRIME_DATA_FILE):
    crime_data = joblib.load(CRIME_DATA_FILE)
else:
    crime_data = pd.read_csv('preprocessed_Pune_crime_data.csv')
    joblib.dump(crime_data, CRIME_DATA_FILE)

# Normalize column names (lowercase, no spaces)
crime_data.columns = crime_data.columns.str.strip().str.lower()

# Ensure the 'date' column exists
if 'date of occurrence' not in crime_data.columns:
    raise KeyError("The 'date' column is missing from the crime data. Please ensure the CSV file contains a 'date' column.")

# Ensure the 'date' column is in datetime format
crime_data['date'] = pd.to_datetime(crime_data['date of occurrence'], errors='coerce', dayfirst=True)

# Define crime spots without severity from the dataset
crime_spots = [(row['latitude'], row['longitude']) for index, row in crime_data.iterrows()]

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
@lru_cache(maxsize=1024)
def get_crime_severity_cached(lat, lon, date, time, day):
    """Get crime severity with caching to avoid redundant predictions."""
    cache_key = f"{lat}_{lon}_{date}_{time}_{day}"
    current_time = time.time()
    
    # Check cache
    if cache_key in CRIME_SEVERITY_CACHE:
        cached_time, severity = CRIME_SEVERITY_CACHE[cache_key]
        if current_time - cached_time < CRIME_CACHE_EXPIRY:
            return severity
    
    # If not in cache or expired, calculate and cache
    severity = predict_crime_severity(lat, lon, date, time, day)
    CRIME_SEVERITY_CACHE[cache_key] = (current_time, severity)
    return severity

def get_dynamic_crime_spots(crime_locations, date, time, day):
    """Predict crime severity dynamically for each crime spot with parallel processing."""
    dynamic_crime_spots = []
    
    def process_spot(spot):
        lat, lon = spot
        severity = get_crime_severity_cached(lat, lon, date, time, day)
        return (lat, lon, severity)
    
    # Process spots in parallel with a timeout
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_spot, spot) for spot in crime_locations]
        for future in as_completed(futures, timeout=10):  # 10 second timeout
            try:
                result = future.result()
                dynamic_crime_spots.append(result)
            except Exception as e:
                logger.warning(f"Error processing crime spot: {e}")
    
    return dynamic_crime_spots

# Update the graph for safest and optimized routes dynamically
def update_graphs_with_dynamic_severity(graph, crime_locations, date, time, day, radius):
    """
    Adjust graph weights dynamically based on predicted crime severity.
    Optimized with timeout and better error handling.
    """
    try:
        start_time = time.time()
        logger.info("Starting dynamic crime severity update...")
        
        # Limit the number of crime locations to process
        max_crime_locations = 1000
        if len(crime_locations) > max_crime_locations:
            logger.warning(f"Too many crime locations ({len(crime_locations)}), sampling to {max_crime_locations}")
            crime_locations = crime_locations[:max_crime_locations]
        
        # Get dynamic crime spots with timeout
        dynamic_crime_spots = get_dynamic_crime_spots(tuple(map(tuple, crime_locations)), date, time, day)
        
        logger.info(f"Crime severity prediction completed in {time.time() - start_time:.2f} seconds")
        
        # Adjust graph weights
        return adjust_weights_for_crime(graph, dynamic_crime_spots, radius)
    except Exception as e:
        logger.error(f"Error in update_graphs_with_dynamic_severity: {e}")
        # Return original graph if there's an error
        return graph

def get_route_crime_details(graph, route, crime_data, radius):
    """
    Calculate total crime severity and collect recent crime types along a route.
    Optimized with spatial indexing and batch processing.
    """
    try:
        start_time = time.time()
        logger.info("Calculating route crime details...")
        
        total_severity = 0
        crime_types = set()
        current_date = datetime.now()
        
        # Sample nodes instead of using every node for better performance
        sample_every = max(1, len(route) // 50)  # Sample at most 50 points
        sampled_route = route[::sample_every]
        
        # Pre-calculate time ranges for severity levels
        time_ranges = {
            0.0: timedelta(weeks=6),    # Very Low
            0.2: timedelta(weeks=12),   # Low
            0.5: timedelta(weeks=18),   # Moderate
            0.7: timedelta(weeks=24),   # High
            1.0: timedelta(weeks=36)    # Very High
        }
        
        # Process nodes in batches
        batch_size = 10
        for i in range(0, len(sampled_route), batch_size):
            batch_nodes = sampled_route[i:i+batch_size]
            
            # Get coordinates for batch
            batch_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in batch_nodes]
            
            # Find nearby crimes for the entire batch
            for lat, lon in batch_coords:
                # Use vectorized operations for better performance
                nearby = crime_data[
                    (abs(crime_data['latitude'] - lat) < radius) &
                    (abs(crime_data['longitude'] - lon) < radius)
                ]
                
                # Process each nearby crime
                for _, crime in nearby.iterrows():
                    # Get severity from cache or calculate
                    crime_date = crime.get('date')
                    if not pd.isna(crime_date):
                        severity = get_crime_severity_cached(
                            crime['latitude'], 
                            crime['longitude'], 
                            crime_date.timestamp(), 
                            0, 
                            crime_date.weekday()
                        )
                        
                        # Check if crime is within time range
                        time_range = time_ranges.get(severity, time_ranges[0.0])
                        if crime_date >= current_date - time_range:
                            total_severity += severity
                            if 'crime type' in crime:
                                crime_types.add(crime['crime type'])
        
        logger.info(f"Route crime details calculated in {time.time() - start_time:.2f} seconds")
        return min(total_severity, 100), list(crime_types)  # Cap total severity
        
    except Exception as e:
        logger.error(f"Error in get_route_crime_details: {e}")
        return 0, []

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

    # Calculate crime details
    fastest_route_crime_details = get_route_crime_details(G, fastest_route, crime_data, radius=0.005)
    safest_route_crime_details = get_route_crime_details(safe_graph, safest_route, crime_data, radius=0.005)
    optimized_route_crime_details = get_route_crime_details(optimized_graph, optimized_route, crime_data, radius=0.005)

    return jsonify({
        "fastest_route_distance_km": fastest_route_distance / 1000,
        "safest_route_distance_km": safest_route_distance / 1000,
        "optimized_route_distance_km": optimized_route_distance / 1000,
        "fastest_route": fastest_route,
        "safest_route": safest_route,
        "optimized_route": optimized_route,
        "fastest_route_crime_details": {
            "total_severity": fastest_route_crime_details[0],
            "crime_types": fastest_route_crime_details[1]
        },
        "safest_route_crime_details": {
            "total_severity": safest_route_crime_details[0],
            "crime_types": safest_route_crime_details[1]
        },
        "optimized_route_crime_details": {
            "total_severity": optimized_route_crime_details[0],
            "crime_types": optimized_route_crime_details[1]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
