from flask import Flask
import osmnx as ox
import networkx as nx
from math import radians, cos, sin, sqrt, atan2
import pandas as pd
from datetime import datetime, timedelta
from utils import predict_crime_severity # This is still needed here for adjust_weights_for_crime
import joblib
import os

app = Flask(__name__)

# --- FUNCTIONS USED BY PRE-COMPUTATION AND THE APP ---

def adjust_weights_for_crime(graph, crime_locations, radius):
    """
    Adjusts edge weights in the graph based on proximity to crime spots.
    This is used by the pre-computation script.
    """
    for crime_lat, crime_lon, severity in crime_locations:
        severity = float(severity)
        nearby_nodes = [node for node, data in graph.nodes(data=True)
                        if abs(data['y'] - crime_lat) < radius and abs(data['x'] - crime_lon) < radius]
        
        for node in nearby_nodes:
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:
                    for key in edge_data:
                        edge_data[key]['length'] *= (1 + severity)
    return graph

def haversine(lat1, lon1, lat2, lon2):
    """
    Haversine function to calculate straight-line distance between two points.
    Used as the heuristic for the A* algorithm.
    """
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_astar_route(graph, start, end):
    """
    Calculates the A* route. Used by app.py for all route types.
    """
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

def calculate_route_distance(graph, route):
    """
    Calculates the total distance of a route. Used by app.py.
    """
    total_distance = 0
    for i in range(len(route) - 1):
        edge_data = graph.get_edge_data(route[i], route[i + 1])[0]
        total_distance += edge_data['length']
    return total_distance

def get_route_crime_details(graph, route, crime_data, radius):
    """
    Calculates total crime severity and collects recent crime types along a route.
    NOW USES PRE-COMPUTED SEVERITY FOR INSTANT RESULTS.
    """
    total_severity = 0
    crime_types = set()
    current_date = datetime.now()

    # Ensure the 'date' column exists for time-based filtering
    if 'date' not in crime_data.columns and 'date of occurrence' in crime_data.columns:
        crime_data['date'] = pd.to_datetime(crime_data['date of occurrence'], errors='coerce', dayfirst=True)

    for node in route:
        node_lat = graph.nodes[node]['y']
        node_lon = graph.nodes[node]['x']

        nearby_crimes = crime_data[
            (abs(crime_data['latitude'] - node_lat) < radius) &
            (abs(crime_data['longitude'] - node_lon) < radius)
        ]

        for _, crime in nearby_crimes.iterrows():
            # THE FIX: Read the pre-computed severity directly from the dataframe
            # This avoids calling the slow machine learning model
            severity = crime['precomputed_severity']
            
            # The time-based filtering remains the same
            time_range = timedelta(weeks=6)
            if severity == 0.2: time_range = timedelta(weeks=12)
            elif severity == 0.5: time_range = timedelta(weeks=18)
            elif severity == 0.7: time_range = timedelta(weeks=24)
            elif severity == 1.0: time_range = timedelta(weeks=36)

            # Check if the 'date' column is valid before comparing
            if 'date' in crime.index and pd.notna(crime['date']) and crime['date'] >= current_date - time_range:
                total_severity += severity
                crime_types.add(crime['crime type'])

    return total_severity, list(crime_types)