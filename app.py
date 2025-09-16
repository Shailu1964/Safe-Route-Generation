from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import folium
from folium.plugins import HeatMap, Fullscreen, LocateControl
from route_calculations import get_astar_route, adjust_weights_for_crime, calculate_route_distance, crime_spots, G, update_graphs_with_dynamic_severity, get_route_crime_details, crime_data
from opencage.geocoder import OpenCageGeocode
import networkx as nx
import joblib
from datetime import datetime
from utils import predict_crime_severity
import copy
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Get API key from environment variables
OPENCAGE_API_KEY = os.environ.get('OPENCAGE_API_KEY')
if not OPENCAGE_API_KEY:
    logger.error("OpenCage API key not found in environment variables!")
    # Fallback for development only - remove in production
    OPENCAGE_API_KEY = '5fab302a78434c22b7424b380952c6cd'

geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

# Ensure static directory exists
os.makedirs('static/map', exist_ok=True)

# Create a model singleton
class ModelSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                cls._instance = joblib.load('crime_severity_model.pkl')
                logger.info("Crime severity model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                cls._instance = None
        return cls._instance

# Function to get coordinates from place names with error handling
def get_coordinates(place_name):
    try:
        result = geocoder.geocode(place_name)
        if result and len(result):
            lat = result[0]['geometry']['lat']
            lon = result[0]['geometry']['lng']
            return lat, lon, None
        return None, None, "Location not found"
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return None, None, f"Geocoding error: {str(e)}"

# Store routes in session to handle multiple users
@app.route('/')
def index():
    # Generate a unique session ID if not already set
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/predict_crime', methods=['POST'])
def predict_crime():
    try:
        data = request.json
        required_fields = ['latitude', 'longitude', 'date', 'time', 'day']
        
        # Validate input
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing required parameter: {field}"}), 400
        
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        date = data['date']
        time = int(data['time'])
        day = int(data['day'])
        
        model = ModelSingleton.get_instance()
        if model is None:
            return jsonify({"error": "Model not available"}), 500
        
        severity = predict_crime_severity(lat, lon, date, time, day)
        return jsonify({"severity": severity})
    except Exception as e:
        logger.error(f"Error in predict_crime: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_routes', methods=['POST'])
def generate_routes():
    start_time = time.time()
    logger.info("Starting route generation...")
    
    try:
        # Set a timeout for the entire route generation (45 seconds)
        if time.time() - start_time > 45:
            raise TimeoutError("Route generation timed out")
            
        start_place = request.form['start_place']
        end_place = request.form['end_place']
        
        # Input validation
        if not start_place or not end_place:
            flash("Please enter both start and end locations")
            return redirect(url_for('index'))
        
        logger.info(f"Getting coordinates for {start_place} to {end_place}...")
        
        # Get coordinates with error handling and timeout
        try:
            start_lat, start_lon, start_error = get_coordinates(start_place)
            end_lat, end_lon, end_error = get_coordinates(end_place)
            
            if start_error:
                flash(f"Start location error: {start_error}")
                return redirect(url_for('index'))
            if end_error:
                flash(f"End location error: {end_error}")
                return redirect(url_for('index'))
                
        except Exception as e:
            logger.error(f"Error getting coordinates: {e}")
            flash("Error getting location coordinates. Please try again.")
            return redirect(url_for('index'))
        
        start_point = (start_lat, start_lon)
        end_point = (end_lat, end_lon)
        
        # Store in session
        session['start_point'] = start_point
        session['end_point'] = end_point
        session['start_place'] = start_place
        session['end_place'] = end_place
        
        # Get the current date, time, and day
        now = datetime.now()
        date = now.timestamp()
        time = now.hour
        day = now.weekday()
        
        routes = {}
        
        # Calculate routes with timeouts
        def calculate_route_with_timeout(route_func, graph, route_name, timeout=15):
            try:
                start = time.time()
                route = get_astar_route(graph, start_point, end_point)
                
                if time.time() - start > timeout:
                    logger.warning(f"{route_name} calculation took too long, skipping...")
                    return None
                    
                return {
                    'route': route,
                    'distance': calculate_route_distance(graph, route),
                    'calculation_time': time.time() - start
                }
            except Exception as e:
                logger.warning(f"Error calculating {route_name.lower()} route: {e}")
                return None
        
        logger.info("Calculating routes...")
        
        # 1. Fastest route (original graph)
        if time.time() - start_time < 40:  # Only proceed if we have time left
            fastest_data = calculate_route_with_timeout(get_astar_route, G, "Fastest")
            if fastest_data:
                routes['Fastest'] = fastest_data
        
        # 2. Safest route (with crime avoidance)
        if time.time() - start_time < 30:  # Only proceed if we have time left
            try:
                safe_graph = update_graphs_with_dynamic_severity(
                    copy.deepcopy(G), 
                    crime_spots, 
                    date, time, day, 
                    radius=0.005
                )
                if time.time() - start_time < 25:  # Check time again
                    safest_data = calculate_route_with_timeout(get_astar_route, safe_graph, "Safest")
                    if safest_data:
                        routes['Safest'] = safest_data
            except Exception as e:
                logger.error(f"Error creating safe graph: {e}")
        
        # 3. Optimized route (balance between safety and speed)
        if time.time() - start_time < 20:  # Only proceed if we have time left
            try:
                optimized_graph = update_graphs_with_dynamic_severity(
                    copy.deepcopy(G),
                    crime_spots,
                    date, time, day,
                    radius=0.002
                )
                if time.time() - start_time < 15:  # Check time again
                    optimized_data = calculate_route_with_timeout(get_astar_route, optimized_graph, "Optimized")
                    if optimized_data:
                        routes['Optimized'] = optimized_data
            except Exception as e:
                logger.error(f"Error creating optimized graph: {e}")
        
        # Store routes in session
        session['routes'] = routes
        
        # If no routes could be generated
        if not routes:
            flash("No routes could be generated between these locations. The locations might be too far apart or unreachable.")
            return redirect(url_for('index'))
        
        logger.info(f"Route generation completed in {time.time() - start_time:.2f} seconds")
        
        # Generate the initial map with all routes and heatmap
        return update_map(route_type='all', show_heatmap=True)
    
    except TimeoutError as te:
        logger.error(f"Route generation timed out: {te}")
        flash("The route calculation took too long. Please try locations that are closer together.")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in generate_routes: {e}", exc_info=True)
        flash("An error occurred while generating routes. Please try again with different locations.")
        return redirect(url_for('index'))

@app.route('/update_map', methods=['GET', 'POST'])
def update_map(route_type=None, show_heatmap=None):
    try:
        # Get route type and heatmap preference
        if request.method == 'POST':
            route_type = request.form.get('route_type', 'all')
            show_heatmap = request.form.get('heatmap') == 'yes'
        
        # For direct function calls with parameters
        if route_type is None:
            route_type = 'all'
        if show_heatmap is None:
            show_heatmap = True
        
        # Get data from session
        start_point = session.get('start_point')
        end_point = session.get('end_point')
        routes = session.get('routes', {})
        
        if not start_point or not end_point or not routes:
            flash("Please generate routes first")
            return redirect(url_for('index'))
        
        # Create map with the available routes
        pune_map = folium.Map(location=start_point, zoom_start=12)
        
        # Add plugins for better user interaction
        Fullscreen().add_to(pune_map)
        LocateControl().add_to(pune_map)
        
        # Function to add route to map with enhanced styling
        def add_route_to_map(route, graph, color, route_type, distance):
            # Calculate crime details for the route
            total_severity, crime_types = get_route_crime_details(graph, route, crime_data, radius=0.005)
            
            # Create a more informative popup with HTML styling
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; padding: 10px; min-width: 200px;">
                <h4 style="margin-top: 0; color: #333;">{route_type} Route</h4>
                <p><strong>Distance:</strong> {distance / 1000:.2f} km</p>
                <p><strong>Crime Risk Level:</strong> <span style="color: {'red' if total_severity > 10 else 'orange' if total_severity > 5 else 'green'}">
                    {'High' if total_severity > 10 else 'Medium' if total_severity > 5 else 'Low'}
                </span></p>
                <p><strong>Crime Types:</strong> {', '.join(crime_types) if crime_types else 'None'}</p>
            </div>
            """
            
            points = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
            folium.PolyLine(
                points, 
                color=color, 
                weight=5, 
                opacity=0.8,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{route_type} Route - {distance / 1000:.2f} km"
            ).add_to(pune_map)
        
        # Add the selected routes to the map
        route_colors = {
            'Fastest': 'blue',
            'Safest': 'green',
            'Optimized': 'orange'
        }
        
        for r_type, color in route_colors.items():
            if (route_type == 'all' or route_type == r_type.lower()) and r_type in routes:
                add_route_to_map(
                    routes[r_type]['route'], 
                    G, 
                    color, 
                    r_type, 
                    routes[r_type]['distance']
                )
        
        # Add markers for start and end points with more information
        start_place = session.get('start_place', 'Start')
        end_place = session.get('end_place', 'End')
        
        folium.Marker(
            location=start_point, 
            popup=f"<b>Start:</b> {start_place}", 
            tooltip="Start Location",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(pune_map)
        
        folium.Marker(
            location=end_point, 
            popup=f"<b>End:</b> {end_place}", 
            tooltip="End Location",
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(pune_map)
        
        # Add crime spots as a heat map with dynamic prediction
        if show_heatmap:
            now = datetime.now()
            date = now.timestamp()
            time = now.hour
            day = now.weekday()
            
            # Generate dynamic crime severity predictions
            dynamic_crime_spots = []
            for lat, lon in crime_spots:
                try:
                    severity = predict_crime_severity(lat, lon, date, time, day)
                    dynamic_crime_spots.append([lat, lon, severity])
                except Exception as e:
                    logger.error(f"Error predicting crime severity: {e}")
            
            # Add heatmap with gradient and legend
            HeatMap(
                dynamic_crime_spots, 
                radius=15,
                min_opacity=0.3,
                gradient={0.4: 'blue', 0.65: 'yellow', 0.9: 'red'},
                blur=10
            ).add_to(pune_map)
            
            # Add a legend for the heatmap
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; right: 50px; width: 180px; height: 90px; 
                            border:2px solid grey; z-index:9999; font-size:14px;
                            background-color: white; padding: 10px; border-radius: 5px;">
                    <p style="margin: 0 0 5px 0"><b>Crime Intensity</b></p>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 20px; height: 20px; background-color: blue; margin-right: 5px;"></div>
                        <span>Low</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 20px; height: 20px; background-color: yellow; margin-right: 5px;"></div>
                        <span>Medium</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div>
                        <span>High</span>
                    </div>
                </div>
            '''
            pune_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Generate a unique filename for each user's map
        user_id = session.get('user_id', 'default')
        map_filename = f"routes_map_{user_id}.html"
        map_path = os.path.join('static', 'map', map_filename)
        pune_map.save(map_path)
        
        # Pass the current values and route statistics to the template
        route_stats = {}
        for r_type, route_data in routes.items():
            total_severity, _ = get_route_crime_details(G, route_data['route'], crime_data, radius=0.005)
            route_stats[r_type] = {
                'distance': route_data['distance'] / 1000,  # km
                'severity': total_severity,
                'safety_level': 'High' if total_severity < 5 else 'Medium' if total_severity < 10 else 'Low'
            }
        
        return render_template(
            'map.html', 
            map_file=url_for('static', filename=f'map/{map_filename}'), 
            start_place=start_place,
            end_place=end_place,
            route_type=route_type, 
            heatmap='yes' if show_heatmap else 'no',
            route_stats=route_stats
        )
    
    except Exception as e:
        logger.error(f"Error in update_map: {e}")
        flash(f"An error occurred while generating the map: {str(e)}")
        return redirect(url_for('index'))

# Add a route to handle errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)