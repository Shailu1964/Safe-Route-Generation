from flask import Flask, render_template, request, jsonify
import os
import folium
from folium.plugins import HeatMap
from r2 import get_astar_route, remove_crime_spots, adjust_weights_for_crime, calculate_route_distance, crime_spots, G, G_safe, G_optimized
from opencage.geocoder import OpenCageGeocode
import networkx as nx  # Add this import
import joblib  # Add this import
import pandas as pd
from datetime import datetime
from utils import predict_crime_severity  # Import from utils.py

app = Flask(__name__)

# Initialize OpenCage API
OPENCAGE_API_KEY = '5fab302a78434c22b7424b380952c6cd'
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

# Load the trained model
model = joblib.load('crime_severity_model.pkl')

# Function to get coordinates from place names
def get_coordinates(place_name):
    result = geocoder.geocode(place_name)
    if result and len(result):
        lat = result[0]['geometry']['lat']
        lon = result[0]['geometry']['lng']
        return lat, lon
    return None, None

# Route for the homepage with form to input place names
@app.route('/')
def index():
    return render_template('index.html')

# Global variables to store generated routes and map data
generated_routes = {}
start_point = None
end_point = None

@app.route('/predict_crime', methods=['POST'])
def predict_crime():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    date = data.get('date')  # Expected in Unix timestamp format
    time = data.get('time')  # Expected as hour (0-23)
    day = data.get('day')    # Expected as numeric day (0-6)

    if not all([lat, lon, date, time, day]):
        return jsonify({"error": "Missing required parameters"}), 400

    severity = predict_crime_severity(lat, lon, date, time, day)
    return jsonify({"severity": severity})

@app.route('/generate_routes', methods=['POST'])
def generate_routes():
    global generated_routes, start_point, end_point

    start_place = request.form['start_place']
    end_place = request.form['end_place']

    # Clear previously generated routes
    generated_routes = {}

    # Get the coordinates for the start and end locations
    start_lat, start_lon = get_coordinates(start_place)
    end_lat, end_lon = get_coordinates(end_place)

    # If any of the places couldn't be geocoded, return an error
    if not start_lat or not end_lat:
        return "Error: Could not find one or both of the locations."

    start_point = (start_lat, start_lon)
    end_point = (end_lat, end_lon)

    # Predict crime severity for crime spots dynamically
    today = datetime.now()
    dynamic_crime_spots = []
    for lat, lon, _ in crime_spots:
        severity = predict_crime_severity(lat, lon, date=today.timestamp(), time=today.hour, day=today.weekday())
        dynamic_crime_spots.append((lat, lon, severity))

    # Adjust crime-related graphs for safe and optimized routes
    safe_graph = adjust_weights_for_crime(G_safe, dynamic_crime_spots, radius=0.005)
    optimized_graph = adjust_weights_for_crime(G_optimized, dynamic_crime_spots, radius=0.005)

    # Calculate all routes
    try:
        fastest_route = get_astar_route(G, start_point, end_point)
        generated_routes['Fastest'] = {
            'route': fastest_route,
            'distance': calculate_route_distance(G, fastest_route)
        }
    except nx.NetworkXNoPath:
        print("Fastest route not found.")

    try:
        safest_route = get_astar_route(safe_graph, start_point, end_point)
        generated_routes['Safest'] = {
            'route': safest_route,
            'distance': calculate_route_distance(safe_graph, safest_route)
        }
    except nx.NetworkXNoPath:
        print("Safest route not found.")

    try:
        optimized_route = get_astar_route(optimized_graph, start_point, end_point)
        generated_routes['Optimized'] = {
            'route': optimized_route,
            'distance': calculate_route_distance(optimized_graph, optimized_route)
        }
    except nx.NetworkXNoPath:
        print("Optimized route not found.")

    # Generate the initial map with all routes and heatmap
    return update_map(route_type='all', show_heatmap=True)

@app.route('/find_routes', methods=['POST'])
def find_routes(start_place=None, end_place=None, route_type=None, show_heatmap=None):
    if not start_place:
        start_place = request.form['start_place']
    if not end_place:
        end_place = request.form['end_place']
    if not route_type:
        route_type = request.form['route_type']
    if show_heatmap is None:
        show_heatmap = request.form['heatmap'] == 'yes'

    # Get the coordinates for the start and end locations
    start_lat, start_lon = get_coordinates(start_place)
    end_lat, end_lon = get_coordinates(end_place)

    # If any of the places couldn't be geocoded, return an error
    if not start_lat or not end_lat:
        return "Error: Could not find one or both of the locations."

    start_point = (start_lat, start_lon)
    end_point = (end_lat, end_lon)

    # Adjust crime-related graphs for safe and optimized routes
    safe_graph = adjust_weights_for_crime(G_safe, crime_spots, radius=0.005)
    optimized_graph = adjust_weights_for_crime(G_optimized, crime_spots, radius=0.005)

    # Initialize variables for routes and distances
    routes = {}
    distances = {}

    # Calculate routes based on the selected route type
    try:
        if route_type in ['all', 'fastest']:
            fastest_route = get_astar_route(G, start_point, end_point)
            routes['Fastest'] = fastest_route
            distances['Fastest'] = calculate_route_distance(G, fastest_route)
    except nx.NetworkXNoPath:
        print("Fastest route not found.")

    try:
        if route_type in ['all', 'safest']:
            safest_route = get_astar_route(safe_graph, start_point, end_point)
            routes['Safest'] = safest_route
            distances['Safest'] = calculate_route_distance(safe_graph, safest_route)
    except nx.NetworkXNoPath:
        print("Safest route not found.")

    try:
        if route_type in ['all', 'optimized']:
            optimized_route = get_astar_route(optimized_graph, start_point, end_point)
            routes['Optimized'] = optimized_route
            distances['Optimized'] = calculate_route_distance(optimized_graph, optimized_route)
    except nx.NetworkXNoPath:
        print("Optimized route not found.")

    # If no routes are found, return an error
    if not routes:
        return "Error: No routes could be calculated."

    # Create map with the available routes
    pune_map = folium.Map(location=start_point, zoom_start=12)

    def add_route_to_map(route, graph, color, route_type, distance):
        points = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
        folium.PolyLine(points, color=color, weight=5,
                        popup=f"{route_type} Route - Distance: {distance/1000:.2f} km").add_to(pune_map)

    # Add the available routes to the map
    for route_type, route in routes.items():
        color = {'Fastest': 'blue', 'Safest': 'green', 'Optimized': 'orange'}.get(route_type, 'gray')
        add_route_to_map(route, G if route_type == 'Fastest' else (safe_graph if route_type == 'Safest' else optimized_graph),
                         color, route_type, distances[route_type])

    # Add markers for start and end points
    folium.Marker(location=start_point, popup='Start', icon=folium.Icon(color='green')).add_to(pune_map)
    folium.Marker(location=end_point, popup='End', icon=folium.Icon(color='blue')).add_to(pune_map)

    # Add crime spots as a heat map if selected
    if show_heatmap:
        def add_crime_heatmap(crime_spots):
            heat_data = [[lat, lon, rate] for lat, lon, rate in crime_spots]
            HeatMap(heat_data, radius=15).add_to(pune_map)

        add_crime_heatmap(crime_spots)

    # Save the map to the static directory
    map_path = os.path.join('static', 'map', 'routes_map.html')
    pune_map.save(map_path)

    return render_template('map.html', map_file=map_path, start_place=start_place, end_place=end_place)

@app.route('/update_map', methods=['POST'])
def update_map(route_type=None, show_heatmap=None):
    global generated_routes, start_point, end_point

    if route_type is None:
        route_type = request.form['route_type']
    if show_heatmap is None:
        show_heatmap = request.form['heatmap'] == 'yes'

    # Create map with the available routes
    pune_map = folium.Map(location=start_point, zoom_start=12)

    def add_route_to_map(route, graph, color, route_type, distance):
        points = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
        folium.PolyLine(points, color=color, weight=5,
                        popup=f"{route_type} Route - Distance: {distance/1000:.2f} km").add_to(pune_map)

    # Add the selected routes to the map
    if route_type == 'all' or route_type == 'fastest':
        if 'Fastest' in generated_routes:
            add_route_to_map(
                generated_routes['Fastest']['route'], G, 'blue', 'Fastest', generated_routes['Fastest']['distance']
            )
    if route_type == 'all' or route_type == 'safest':
        if 'Safest' in generated_routes:
            add_route_to_map(
                generated_routes['Safest']['route'], G_safe, 'green', 'Safest', generated_routes['Safest']['distance']
            )
    if route_type == 'all' or route_type == 'optimized':
        if 'Optimized' in generated_routes:
            add_route_to_map(
                generated_routes['Optimized']['route'], G_optimized, 'orange', 'Optimized', generated_routes['Optimized']['distance']
            )

    # Add markers for start and end points
    folium.Marker(location=start_point, popup='Start', icon=folium.Icon(color='green')).add_to(pune_map)
    folium.Marker(location=end_point, popup='End', icon=folium.Icon(color='blue')).add_to(pune_map)

    # Add crime spots as a heat map if selected
    if show_heatmap:
        def add_crime_heatmap(crime_spots):
            heat_data = [[lat, lon, rate] for lat, lon, rate in crime_spots]
            HeatMap(heat_data, radius=15).add_to(pune_map)

        add_crime_heatmap(crime_spots)

    # Save the map to the static directory
    map_path = os.path.join('static', 'map', 'routes_map.html')
    pune_map.save(map_path)

    # Pass the current values of route_type and heatmap to the template
    return render_template('map.html', map_file=map_path, start_place=start_point, end_place=end_point,
                           route_type=route_type, heatmap='yes' if show_heatmap else 'no')

@app.route('/toggle_predicted_heatmap', methods=['POST'])
def toggle_predicted_heatmap():
    global crime_spots

    # Get the user's choice for showing the predicted heatmap
    show_predicted_heatmap = request.form['show_predicted_heatmap'] == 'yes'

    # Create map with the predicted crime severity heatmap if selected
    pune_map = folium.Map(location=start_point, zoom_start=12)

    # Add the predicted crime severity heatmap
    if show_predicted_heatmap:
        def add_predicted_heatmap(crime_spots):
            heat_data = [[lat, lon, rate] for lat, lon, rate in crime_spots]
            HeatMap(heat_data, radius=15).add_to(pune_map)

        add_predicted_heatmap(crime_spots)

    # Save the map to the static directory
    map_path = os.path.join('static', 'map', 'routes_map.html')
    pune_map.save(map_path)

    # Pass the current values to the template
    return render_template('map.html', map_file=map_path, start_place=start_point, end_place=end_point,
                           predicted_severity="Dynamic Severity", show_predicted_heatmap='yes' if show_predicted_heatmap else 'no')

if __name__ == '__main__':
    if not os.path.exists('static/map'):
        os.makedirs('static/map')
    app.run(debug=True)
