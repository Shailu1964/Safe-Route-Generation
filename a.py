from flask import Flask, render_template, request
import os
import folium
from folium.plugins import HeatMap
from r import get_astar_route, remove_crime_spots, adjust_weights_for_crime, calculate_route_distance, crime_spots, G, G_safe, G_optimized
from opencage.geocoder import OpenCageGeocode

app = Flask(__name__)

# Initialize OpenCage API
OPENCAGE_API_KEY = '5fab302a78434c22b7424b380952c6cd'
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

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

# Route to process the form and display the map
@app.route('/find_routes', methods=['POST'])
def find_routes():
    start_place = request.form['start_place']
    end_place = request.form['end_place']

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
    optimized_graph = remove_crime_spots(G_optimized, crime_spots, radius=0.005, max_severity=0.5)

    # Calculate routes using the A* algorithm
    fastest_route = get_astar_route(G, start_point, end_point)
    safest_route = get_astar_route(safe_graph, start_point, end_point)
    optimized_route = get_astar_route(optimized_graph, start_point, end_point)

    # Calculate distances for the routes
    fastest_distance = calculate_route_distance(G, fastest_route)
    safest_distance = calculate_route_distance(safe_graph, safest_route)
    optimized_distance = calculate_route_distance(optimized_graph, optimized_route)

    # Create map with the routes
    pune_map = folium.Map(location=start_point, zoom_start=12)

    def add_route_to_map(route, graph, color, route_type, distance):
        points = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
        folium.PolyLine(points, color=color, weight=5,
                        popup=f"{route_type} Route - Distance: {distance/1000:.2f} km").add_to(pune_map)

    # Add the routes to the map
    add_route_to_map(fastest_route, G, 'blue', 'Fastest', fastest_distance)
    add_route_to_map(safest_route, safe_graph, 'green', 'Safest', safest_distance)
    add_route_to_map(optimized_route, optimized_graph, 'orange', 'Optimized', optimized_distance)
    

    # Add markers for start and end points
    folium.Marker(location=start_point, popup='Start', icon=folium.Icon(color='green')).add_to(pune_map)
    folium.Marker(location=end_point, popup='End', icon=folium.Icon(color='blue')).add_to(pune_map)
    # for i, (lat, lon, rate) in enumerate(crime_spots):
    #     folium.Marker(location=(lat, lon), popup=f'Crime Spot {i+1} - Rate: {rate}', icon=folium.Icon(color='red')).add_to(pune_map)

    # Add crime spots to the map as a heat map
    def add_crime_heatmap(crime_spots):
        heat_data = [[lat, lon, rate] for lat, lon, rate in crime_spots]
        HeatMap(heat_data, radius=15).add_to(pune_map)
    
    # Add crime spots as a heat map
    add_crime_heatmap(crime_spots)


    # Save the map to the static directory
    map_path = os.path.join('static', 'map', 'routes_map.html')
    pune_map.save(map_path)

    return render_template('map.html', map_file=map_path)

if __name__ == '__main__':
    if not os.path.exists('static/map'):
        os.makedirs('static/map')
    app.run(debug=True)
