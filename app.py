from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import folium
from folium.plugins import HeatMap, Fullscreen, LocateControl
from route_calculations import get_astar_route, calculate_route_distance, get_route_crime_details
from opencage.geocoder import OpenCageGeocode
import networkx as nx
import joblib
import pandas as pd
from datetime import datetime
import uuid
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

OPENCAGE_API_KEY = os.environ.get('OPENCAGE_API_KEY', '5fab302a78434c22b7424b380952c6cd')
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

# --- FINAL, LIGHTWEIGHT DATA LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
G, safe_graph, optimized_graph, crime_data, heatmap_data = None, None, None, None, None

try:
    logger.info("Loading all pre-computed data...")
    G = joblib.load(os.path.join(BASE_DIR, 'pune_graph.pkl'))
    safe_graph = joblib.load(os.path.join(BASE_DIR, 'safe_graph.pkl'))
    optimized_graph = joblib.load(os.path.join(BASE_DIR, 'optimized_graph.pkl'))
    crime_data = joblib.load(os.path.join(BASE_DIR, 'preprocessed_Pune_crime_data.pkl'))
    heatmap_data = joblib.load(os.path.join(BASE_DIR, 'heatmap_data.pkl'))
    
    crime_data.columns = crime_data.columns.str.strip().str.lower()
    crime_data['date'] = pd.to_datetime(crime_data['date of occurrence'], errors='coerce', dayfirst=True)
    logger.info("All data loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not load data files. Run 'precompute_weighted_graphs.py' first. Error: {e}", exc_info=True)
# --- END OF DATA LOADING ---

os.makedirs('static/map', exist_ok=True)

def get_coordinates(place_name):
    try:
        result = geocoder.geocode(place_name)
        if result and len(result):
            return result[0]['geometry']['lat'], result[0]['geometry']['lng'], None
        return None, None, "Location not found"
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return None, None, f"Geocoding error: {str(e)}"

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/generate_routes', methods=['POST'])
def generate_routes():
    if not all([G, safe_graph, optimized_graph]):
        flash("Server error: Map data is not available. Please check the server logs.")
        return redirect(url_for('index'))

    try:
        start_place = request.form['start_place']
        end_place = request.form['end_place']

        if not start_place or not end_place:
            flash("Please enter both start and end locations")
            return redirect(url_for('index'))

        start_lat, start_lon, start_error = get_coordinates(start_place)
        end_lat, end_lon, end_error = get_coordinates(end_place)

        if start_error or end_error:
            flash(f"Location error: {start_error or end_error}")
            return redirect(url_for('index'))

        start_point = (start_lat, start_lon)
        end_point = (end_lat, end_lon)

        session.update({
            'start_point': start_point, 'end_point': end_point,
            'start_place': start_place, 'end_place': end_place
        })

        routes = {}
        try:
            fastest_route = get_astar_route(G, start_point, end_point)
            routes['Fastest'] = {'route': fastest_route, 'distance': calculate_route_distance(G, fastest_route)}
        except nx.NetworkXNoPath: logger.warning("Fastest route not found")
        
        try:
            safest_route = get_astar_route(safe_graph, start_point, end_point)
            routes['Safest'] = {'route': safest_route, 'distance': calculate_route_distance(safe_graph, safest_route)}
        except nx.NetworkXNoPath: logger.warning("Safest route not found")

        try:
            optimized_route = get_astar_route(optimized_graph, start_point, end_point)
            routes['Optimized'] = {'route': optimized_route, 'distance': calculate_route_distance(optimized_graph, optimized_route)}
        except nx.NetworkXNoPath: logger.warning("Optimized route not found")

        session['routes'] = routes

        if not routes:
            flash("No routes could be generated. Try different locations.")
            return redirect(url_for('index'))

        return update_map(route_type='all', show_heatmap=True)

    except Exception as e:
        logger.error(f"Error in generate_routes: {e}", exc_info=True)
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('index'))

@app.route('/update_map', methods=['GET', 'POST'])
def update_map(route_type=None, show_heatmap=None):
    try:
        if request.method == 'POST':
            route_type = request.form.get('route_type', 'all')
            show_heatmap = request.form.get('heatmap') == 'yes'
        
        if route_type is None: route_type = 'all'
        if show_heatmap is None: show_heatmap = True
        
        start_point = session.get('start_point')
        end_point = session.get('end_point')
        routes = session.get('routes', {})
        
        if not start_point or not end_point or not routes:
            flash("Please generate routes first")
            return redirect(url_for('index'))
        
        pune_map = folium.Map(location=start_point, zoom_start=12)
        
        Fullscreen().add_to(pune_map)
        LocateControl().add_to(pune_map)
        
        def add_route_to_map(route, graph, color, route_type, distance):
            total_severity, crime_types = get_route_crime_details(graph, route, crime_data, radius=0.005)
            popup_content = f"""<div style="font-family: Arial, sans-serif; padding: 10px; min-width: 200px;"><h4 style="margin-top: 0; color: #333;">{route_type} Route</h4><p><strong>Distance:</strong> {distance / 1000:.2f} km</p><p><strong>Crime Risk Level:</strong> <span style="color: {'red' if total_severity > 10 else 'orange' if total_severity > 5 else 'green'}">{'High' if total_severity > 10 else 'Medium' if total_severity > 5 else 'Low'}</span></p><p><strong>Crime Types:</strong> {', '.join(crime_types) if crime_types else 'None'}</p></div>"""
            points = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
            folium.PolyLine(points, color=color, weight=5, opacity=0.8, popup=folium.Popup(popup_content, max_width=300), tooltip=f"{route_type} Route - {distance / 1000:.2f} km").add_to(pune_map)
        
        route_colors = {'Fastest': 'blue', 'Safest': 'green', 'Optimized': 'orange'}
        for r_type, color in route_colors.items():
            if (route_type == 'all' or route_type == r_type.lower()) and r_type in routes:
                add_route_to_map(routes[r_type]['route'], G, color, r_type, routes[r_type]['distance'])
        
        start_place = session.get('start_place', 'Start')
        end_place = session.get('end_place', 'End')
        
        folium.Marker(location=start_point, popup=f"<b>Start:</b> {start_place}", tooltip="Start Location", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(pune_map)
        folium.Marker(location=end_point, popup=f"<b>End:</b> {end_place}", tooltip="End Location", icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(pune_map)
        
        if show_heatmap:
            if heatmap_data:
                HeatMap(
                    heatmap_data, 
                    radius=15, min_opacity=0.3,
                    gradient={0.4: 'blue', 0.65: 'yellow', 0.9: 'red'}, blur=10
                ).add_to(pune_map)
            
            legend_html = '''<div style="position: fixed; bottom: 50px; right: 50px; width: 180px; height: 90px; border:2px solid grey; z-index:9999; font-size:14px; background-color: white; padding: 10px; border-radius: 5px;"><p style="margin: 0 0 5px 0"><b>Crime Intensity</b></p><div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: blue; margin-right: 5px;"></div><span>Low</span></div><div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: yellow; margin-right: 5px;"></div><span>Medium</span></div><div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div><span>High</span></div></div>'''
            pune_map.get_root().html.add_child(folium.Element(legend_html))
        
        user_id = session.get('user_id', 'default')
        map_filename = f"routes_map_{user_id}.html"
        map_path = os.path.join('static', 'map', map_filename)
        pune_map.save(map_path)
        
        route_stats = {}
        for r_type, route_data in routes.items():
            total_severity, _ = get_route_crime_details(G, route_data['route'], crime_data, radius=0.005)
            route_stats[r_type] = {
                'distance': route_data['distance'] / 1000,
                'severity': total_severity,
                'safety_level': 'High' if total_severity < 5 else 'Medium' if total_severity < 10 else 'Low'
            }
        
        return render_template('map.html', map_file=url_for('static', filename=f'map/{map_filename}'), start_place=start_place, end_place=end_place, route_type=route_type, heatmap='yes' if show_heatmap else 'no', route_stats=route_stats)
    
    except Exception as e:
        logger.error(f"Error in update_map: {e}", exc_info=True)
        flash(f"An error occurred while generating the map: {str(e)}")
        return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e): return render_template('404.html'), 404
@app.errorhandler(500)
def server_error(e): return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)