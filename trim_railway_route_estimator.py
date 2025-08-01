import streamlit as st
import pandas as pd
import lz4.frame
import os
import folium
from streamlit_folium import st_folium
import networkx as nx
import osmnx as ox
from geopy.distance import geodesic
import hashlib
import numpy as np
import gc  # Garbage collection


# Load Thunderforest API key and check for local development
if 'THUNDERFOREST_API_KEY' not in st.session_state:
    if os.path.exists('thunderforest_key.py'):
        from thunderforest_key import THUNDERFOREST_API_KEY    
        st.session_state.THUNDERFOREST_API_KEY = THUNDERFOREST_API_KEY    
        st.success("THUNDERFOREST_API_KEY loaded successfully!")
    else:
        st.session_state.THUNDERFOREST_API_KEY = None
        st.info("No THUNDERFOREST_API_KEY available!")

# Check if running locally (create a local_config.py file for local development)
if 'IS_LOCAL_DEV' not in st.session_state:
    if os.path.exists('local_config.py'):
        try:
            from local_config import IS_LOCAL_DEVELOPMENT
            st.session_state.IS_LOCAL_DEV = IS_LOCAL_DEVELOPMENT
            st.success("🏠 Local development mode detected - Enhanced performance enabled!")
        except ImportError:
            st.session_state.IS_LOCAL_DEV = False
    else:
        st.session_state.IS_LOCAL_DEV = False

# Select number of waypoints to use in GUI
number_of_waypoints = 4

# Set page config
st.set_page_config(
    page_title="South Africa Railway Data",
    page_icon="🚂",
    layout="wide"
)

# Initialise some session state variables
if 'trip_distance' not in st.session_state:
    st.session_state.trip_distance = "100"  # Set default starting trip distance

# Initialize session state for dataframes (persistent in local dev, temporary in cloud)
if st.session_state.IS_LOCAL_DEV:
    # Local development: Keep graphs in memory for performance
    if 'railway_detailed' not in st.session_state:
        st.session_state.railway_detailed = None
    if 'railway_simple' not in st.session_state:
        st.session_state.railway_simple = None

if 'waypoints_df' not in st.session_state:
    st.session_state.waypoints_df = None    
if 'route_calculation_result' not in st.session_state:
    st.session_state.route_calculation_result = None     
if 'access_rates_df' not in st.session_state:
    st.session_state.access_rates_df = None    
if 'rates_calculation_result' not in st.session_state:
    st.session_state.rates_calculation_result = None 
if 'rates_calculation_df' not in st.session_state:
    st.session_state.rates_calculation_df = None    

# Route data persistence (keep lightweight data only)
if 'route_calculated' not in st.session_state:
    st.session_state.route_calculated = False
if 'route_points' not in st.session_state:
    st.session_state.route_points = None
if 'route_coords' not in st.session_state:
    st.session_state.route_coords = None
if 'route_distance' not in st.session_state:
    st.session_state.route_distance = None
if 'route_geometries' not in st.session_state:
    st.session_state.route_geometries = None

# Map caching and persistence
if 'cached_map_html' not in st.session_state:
    st.session_state.cached_map_html = None
if 'map_cache_key' not in st.session_state:
    st.session_state.map_cache_key = None
if 'map_needs_update' not in st.session_state:
    st.session_state.map_needs_update = True

def clear_graph_data():
    """Clear graph data from memory to free up space (only in cloud deployment)"""
    if not st.session_state.IS_LOCAL_DEV:
        # Only clear in cloud deployment
        if hasattr(st.session_state, 'railway_detailed'):
            delattr(st.session_state, 'railway_detailed')
        if hasattr(st.session_state, 'railway_simple'):
            delattr(st.session_state, 'railway_simple')
        
        # Force garbage collection
        gc.collect()
        
        st.info("Graph data cleared from memory to free up resources.")
    else:
        st.info("Local development mode: Graph data kept in memory for performance.")

@st.cache_data(max_entries=1, ttl=3600)  # Cache for 1 hour, max 1 entry
def load_lz4_data(filename):
    """Load data from LZ4 compressed file with caching"""
    try:
        if os.path.exists(filename):
            with lz4.frame.open(filename, 'rb') as f:
                data = pd.read_pickle(f)
            return data
        else:
            st.error(f"File {filename} not found!")
            return None
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

@st.cache_data(max_entries=2, ttl=7200)  # Cache for 2 hours
def load_excel_data(filename):
    """Load data from Excel file with caching"""
    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename)
            return df
        else:
            st.error(f"File {filename} not found!")
            return None
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

def parse_location_input(text):
    """Parse location input text into (lat, lon) tuple"""
    try:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return (lat, lon)
    except ValueError:
        pass
    return None

def find_detailed_path_between_nodes(G_detailed, start_node, end_node):
    """Find detailed path between two nodes"""
    try:
        detailed_path = nx.shortest_path(
            G_detailed, start_node, end_node, weight='length')
        detailed_distance = nx.shortest_path_length(
            G_detailed, start_node, end_node, weight='length')
        return detailed_path, detailed_distance
    except (nx.NetworkXNoPath, KeyError):
        return None, None

def calculate_hybrid_route(G_simple, G_detailed, start_node, end_node):
    """Calculate route using simplified graph, then get accurate distance using detailed graph"""
    try:
        simple_route = nx.shortest_path(
            G_simple, start_node, end_node, weight='length')
        simple_distance = nx.shortest_path_length(
            G_simple, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        return None, None, None, None

    detailed_full_route = []
    total_accurate_distance = 0

    for i in range(len(simple_route) - 1):
        current_node = simple_route[i]
        next_node = simple_route[i + 1]

        detailed_segment, segment_distance = find_detailed_path_between_nodes(
            G_detailed, current_node, next_node
        )

        if detailed_segment is None:
            fallback_distance = nx.shortest_path_length(
                G_simple, current_node, next_node, weight='length')
            total_accurate_distance += fallback_distance
            if detailed_full_route:
                detailed_full_route.extend([current_node, next_node])
            else:
                detailed_full_route.extend([current_node, next_node])
        else:
            total_accurate_distance += segment_distance
            if detailed_full_route:
                detailed_full_route.extend(detailed_segment[1:])
            else:
                detailed_full_route.extend(detailed_segment)

    return simple_route, detailed_full_route, total_accurate_distance, simple_distance

def calculate_od_pairs_hybrid(G_simple, G_detailed, nodes):
    """Calculate route between multiple points using hybrid approach"""
    simple_full_route = []
    detailed_full_route = []
    total_accurate_distance = 0
    segment_distances = []

    for i in range(len(nodes) - 1):
        orig_node = nodes[i]
        dest_node = nodes[i + 1]

        simple_segment, detailed_segment, segment_distance, simple_distance = calculate_hybrid_route(
            G_simple, G_detailed, orig_node, dest_node
        )

        if simple_segment is None:
            return None, None, None, None

        if simple_full_route:
            simple_full_route.extend(simple_segment[1:])
        else:
            simple_full_route.extend(simple_segment)

        if detailed_full_route:
            detailed_full_route.extend(detailed_segment[1:])
        else:
            detailed_full_route.extend(detailed_segment)

        segment_distances.append(segment_distance)
        total_accurate_distance += segment_distance

    return simple_full_route, detailed_full_route, total_accurate_distance, simple_distance, segment_distances

def get_route_geometry_improved(graph, route_nodes):
    """Get the geometry of a route and return lightweight coordinate data"""
    route_geometries = []

    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = graph.get_edge_data(u, v)

        if isinstance(edge_data, dict) and not any(key in edge_data for key in ['geometry', 'length']):
            edge_keys = list(edge_data.keys())
            edge_info = edge_data[edge_keys[0]]
        else:
            edge_info = edge_data

        if edge_info and 'geometry' in edge_info and edge_info['geometry'] is not None:
            try:
                coords = list(edge_info['geometry'].coords)
                coords_latlon = [(lat, lon) for lon, lat in coords]
                route_geometries.append(coords_latlon)
            except Exception:
                start_node = graph.nodes[u]
                end_node = graph.nodes[v]
                coords_latlon = [(start_node['y'], start_node['x']),
                                 (end_node['y'], end_node['x'])]
                route_geometries.append(coords_latlon)
        else:
            start_node = graph.nodes[u]
            end_node = graph.nodes[v]
            coords_latlon = [(start_node['y'], start_node['x']),
                             (end_node['y'], end_node['x'])]
            route_geometries.append(coords_latlon)

    return route_geometries    

def generate_cache_key():
    """Generate a cache key based on current route data"""
    if not st.session_state.route_calculated:
        return "no_route"
    
    # Create a hash of the route data
    route_data = {
        'points': st.session_state.route_points,
        'coords': st.session_state.route_coords,
        'distance': st.session_state.route_distance
    }
    
    cache_string = str(route_data)
    return hashlib.md5(cache_string.encode()).hexdigest()

def calculate_route():
    """Calculate route and store minimal data in session state"""
    
    # Parse location inputs
    locations = []
    valid_inputs = []
    
    for i in range(1, number_of_waypoints + 1): 
        text = st.session_state[f"coordinates_input_{i}"]
        if not text:
            continue

        coords = parse_location_input(text)
        if coords is not None:
            locations.append(coords)
            valid_inputs.append(i+1)
            continue
    
    if len(locations) < 2:
        st.error("Insufficient Locations. You must provide at least 2 valid locations to calculate a route.")
        return

    # Handle graph loading based on environment
    if st.session_state.IS_LOCAL_DEV:
        # Local development: Load once and keep in memory
        if st.session_state.railway_detailed is None:
            with st.spinner("Loading railway network data..."):
                st.session_state.railway_detailed = load_lz4_data("south_africa_railway_detailed.lz4")
        
        if st.session_state.railway_simple is None:
            with st.spinner("Loading railway network data..."):
                st.session_state.railway_simple = load_lz4_data("south_africa_railway_simple.lz4")
                
        railway_detailed = st.session_state.railway_detailed
        railway_simple = st.session_state.railway_simple
        
        if railway_detailed is None or railway_simple is None:
            st.error("Failed to load railway network data")
            return
    else:
        # Cloud deployment: Load temporarily for calculation
        with st.spinner("Loading railway network data..."):
            railway_detailed = load_lz4_data("south_africa_railway_detailed.lz4")
            railway_simple = load_lz4_data("south_africa_railway_simple.lz4")
            
            if railway_detailed is None or railway_simple is None:
                st.error("Failed to load railway network data")
                return
        
    try:
        # Snap points to railway graph
        points = []
        nodes = []
        coords = []
        snap_distances = []

        for i, point in enumerate(locations):
            node = ox.distance.nearest_nodes(railway_simple, X=point[1], Y=point[0])
            node_coords = (railway_simple.nodes[node]['y'], railway_simple.nodes[node]['x'])
            snap_dist = geodesic(point, node_coords).meters

            points.append(point)
            nodes.append(node)
            coords.append(node_coords)
            snap_distances.append(snap_dist)

        # Calculate route using hybrid approach
        simple_route, full_route, total_length, simple_distance, segment_distances = calculate_od_pairs_hybrid(
            railway_simple, railway_detailed, nodes
        )

        if simple_route is None:
            st.error("Error. No railway path found between the specified points.")
            return
        
        # Get route geometries for map display (before clearing graph data)
        G_latlon = ox.project_graph(railway_simple, to_crs='EPSG:4326')
        route_geometries = get_route_geometry_improved(G_latlon, simple_route)
        
        # Store minimal route data in session state
        st.session_state.route_calculated = True
        st.session_state.route_points = points
        st.session_state.route_coords = coords
        st.session_state.route_distance = total_length
        st.session_state.route_geometries = route_geometries  # Store lightweight geometry data
        
        # Mark that map needs update
        st.session_state.map_needs_update = True        
        
        # Update results label
        result_text = (
            f"**Route Summary:**  \n"
            f"Total distance: {total_length/1000:.2f} km  \n"  
        )
        
        for i, dist in enumerate(segment_distances):
            result_text += f"Segment {i+1}: {dist/1000:.2f} km  \n"

        for i, dist in enumerate(snap_distances):
            result_text += f"Snap distance {valid_inputs[i]}: {dist:.0f} m  \n"
        result_text = result_text[:-2]
        
        st.session_state.route_calculation_result = result_text       
        
        st.success("Route calculation success!")
                
        # Perform automatic rate calculation
        st.session_state.trip_distance = f"{total_length/1000:.2f}"
        calculate_rate()        

    except Exception as e:
        st.error(f"Error. Failed to calculate route: {str(e)}")
    finally:
        # Only clear graph data in cloud deployment
        if not st.session_state.IS_LOCAL_DEV:
            del railway_detailed, railway_simple, G_latlon
            gc.collect()
            st.info("Railway network data cleared from memory.")
        else:
            # In local dev, just clean up the temporary G_latlon
            if 'G_latlon' in locals():
                del G_latlon
                gc.collect()

def calculate_rate():
    """Calculate rates based on route distance and parameters""" 
    try:
        trip_distance = np.float64(st.session_state["trip_distance"].strip())
        trainkm_rate = np.float64(st.session_state["rate_tariff_per_trainkm_rand"].strip())
        gtk_rate = np.float64(st.session_state["rate_tariff_per_gtk_cents"].strip())
        
        if st.session_state.electric_locomotives:
            e_rate = np.float64(st.session_state["rate_e_rate_per_gtk_cents"].strip())
        else:
            e_rate = 0.0
                    
        loaded_train_mass = np.float64(st.session_state["loaded_train_mass"].strip())
        empty_train_mass = np.float64(st.session_state["empty_train_mass"].strip())
        annual_volume = np.float64(st.session_state["annual_volume"].strip()) 
                     
        payload_per_train = loaded_train_mass - empty_train_mass
        whole_trips_required = np.ceil(annual_volume / payload_per_train)        
            
        revenue = pd.DataFrame(columns=['Loaded', 'Empty']) 
        revenue.loc['GTK per trip'] = [loaded_train_mass * trip_distance, empty_train_mass * trip_distance]
        revenue.loc['Train.km revenue per trip'] = [trainkm_rate * trip_distance, trainkm_rate * trip_distance]
        revenue.loc['GTK revenue per trip'] = revenue.loc['GTK per trip'] * gtk_rate / 100
        revenue.loc['E-Rate revenue per trip'] = revenue.loc['GTK per trip'] * e_rate / 100
        revenue.loc['Total revenue per trip'] = revenue.loc[['Train.km revenue per trip', 'GTK revenue per trip', 'E-Rate revenue per trip'], :].sum()
        revenue.loc['Train.km revenue per annum'] = revenue.loc['Train.km revenue per trip'] * whole_trips_required
        revenue.loc['GTK revenue per annum'] = revenue.loc['GTK revenue per trip'] * whole_trips_required
        revenue.loc['E-Rate revenue per annum'] = revenue.loc['E-Rate revenue per trip'] * whole_trips_required
        revenue.loc['Total revenue per annum'] = revenue.loc[['Train.km revenue per annum', 'GTK revenue per annum', 'E-Rate revenue per annum'], :].sum()
        revenue['Combined'] = revenue['Empty'] + revenue['Loaded']
        total_revenue_per_annum = revenue.loc['Total revenue per annum', 'Combined']
        
        result_text = "**Calculation Summary:**  \n"
        result_text += f"Payload per train [ton]: {payload_per_train}  \n"
        result_text += f"Whole trips required to achieve annual volume: {whole_trips_required:.0f}  \n"           
        result_text += f"Total revenue per annum: R {total_revenue_per_annum:,.2f}"
        
        st.session_state.rates_calculation_result = result_text
        
        gtk_per_trip = revenue.loc[['GTK per trip']] 
        revenue_per_trip = revenue.loc[['Train.km revenue per trip', 'GTK revenue per trip', 'E-Rate revenue per trip', 'Total revenue per trip']]
        revenue_per_annum = revenue.loc[['Train.km revenue per annum', 'GTK revenue per annum', 'E-Rate revenue per annum', 'Total revenue per annum']]
        
        def format_thousands(x):
            try:
                return f"{x:,.0f}".replace(",", " ")
            except:
                return x
        
        # Format function: Rand with space thousands separator, no decimals
        def format_rands(x):
            try:
                return f"R {x:,.2f}".replace(",", " ")
            except:
                return x
        
        # Apply formatting using Styler 
        styled_gtk_per_trip = gtk_per_trip.style.format({col: format_thousands for col in gtk_per_trip.select_dtypes(include=np.number).columns})
        styled_revenue_per_trip = revenue_per_trip.style.format({col: format_rands for col in revenue_per_trip.select_dtypes(include=np.number).columns})
        styled_revenue_per_annum = revenue_per_annum.style.format({col: format_rands for col in revenue_per_annum.select_dtypes(include=np.number).columns})    
        
        st.session_state.rates_calculation_df = [styled_gtk_per_trip, styled_revenue_per_trip, styled_revenue_per_annum]
            
        st.success("Rate calculation success!")
        
    except Exception as e:
        st.error(f"Error in rate calculation: {str(e)}")

@st.cache_data(max_entries=2, ttl=3600)  # Cache map HTML for 1 hour
def create_cached_map_html(route_data, thunderforest_key):
    """Create and cache the map HTML using route data"""
    if not route_data or not route_data.get('calculated', False):
        return create_default_map_html(thunderforest_key)
    
    points = route_data['points']
    coords = route_data['coords'] 
    route_geometries = route_data['geometries']
    
    # Create a new map centered on the route
    center_lat = (points[0][0] + points[-1][0]) / 2
    center_lon = (points[0][1] + points[-1][1]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None)

    # Add base layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="ESRI Satellite",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)        

    if thunderforest_key:
        folium.TileLayer(
            tiles=f"https://tile.thunderforest.com/transport/{{z}}/{{x}}/{{y}}.png?apikey={thunderforest_key}",
            attr='Transport Map © Thunderforest, OpenStreetMap contributors',
            name='Transport Map',
            overlay=False,
            control=True
        ).add_to(m)

    # Create feature groups
    route_group = folium.FeatureGroup(name="Railway Route", show=True)
    location_markers_group = folium.FeatureGroup(name="Locations", show=True)
    snapped_markers_group = folium.FeatureGroup(name="Snapped Points", show=True)

    # Plot the railway route using stored geometries
    if route_geometries:
        for geometry in route_geometries:
            folium.PolyLine(
                geometry,
                color='red',
                weight=5,
                opacity=0.8,
            ).add_to(route_group)

    # Add location markers
    for i, point in enumerate(points):
        folium.CircleMarker(
            location=point,
            popup=f"Location {i+1}",
            radius=8,
            color='green',
            fill=True,
            fill_opacity=0.7,
            fill_color='green'
        ).add_to(location_markers_group)

    # Add snapped markers
    for i, coord in enumerate(coords):
        folium.CircleMarker(
            location=coord,
            popup=f"Snapped Location {i+1}",
            radius=6,
            color='red',
            fill=True,
            fill_opacity=0.7,
            fill_color='red'
        ).add_to(snapped_markers_group)

    # Add all feature groups to the map
    route_group.add_to(m)
    location_markers_group.add_to(m)
    snapped_markers_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m._repr_html_()

@st.cache_data(max_entries=1, ttl=7200)  # Cache default map for 2 hours
def create_default_map_html(thunderforest_key):
    """Create default map HTML without route"""
    center_lat, center_lon = -29.0, 24.0
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add base layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="ESRI Satellite",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)        

    if thunderforest_key:
        folium.TileLayer(
            tiles=f"https://tile.thunderforest.com/transport/{{z}}/{{x}}/{{y}}.png?apikey={thunderforest_key}",
            attr='Transport Map © Thunderforest, OpenStreetMap contributors',
            name='Transport Map',
            overlay=False,
            control=True
        ).add_to(m)
    
    return m._repr_html_()

def get_map_html():
    """Get the appropriate map HTML, using cache when possible"""
    current_cache_key = generate_cache_key()
    
    # Check if we need to update the map
    if (st.session_state.map_needs_update or 
        st.session_state.map_cache_key != current_cache_key or
        st.session_state.cached_map_html is None):
        
        # Create route data for caching
        if st.session_state.route_calculated:
            route_data = {
                'calculated': True,
                'points': st.session_state.route_points,
                'coords': st.session_state.route_coords,
                'geometries': st.session_state.route_geometries,
                'distance': st.session_state.route_distance
            }
            st.session_state.cached_map_html = create_cached_map_html(
                route_data, st.session_state.THUNDERFOREST_API_KEY
            )
        else:
            st.session_state.cached_map_html = create_default_map_html(
                st.session_state.THUNDERFOREST_API_KEY
            )
        
        # Update cache key and reset update flag
        st.session_state.map_cache_key = current_cache_key
        st.session_state.map_needs_update = False
    
    return st.session_state.cached_map_html

# Main app
def main():
    st.title("🚂 TRIM Railway Route Estimator")
    st.markdown("---")
    
    # Add memory management section
    col_mem1, col_mem2 = st.columns([3, 1])
    with col_mem1:
        if st.session_state.IS_LOCAL_DEV:
            st.info("🏠 **Local Development Mode**: Network files kept in memory for maximum performance.")
        else:
            st.info("☁️ **Cloud Mode**: Network files loaded temporarily during calculations to optimize memory usage.")
    with col_mem2:
        if st.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
            st.cache_data.clear()
            clear_graph_data()
            st.session_state.cached_map_html = None
            st.session_state.map_needs_update = True
            st.success("Cache cleared!")
    
    # Load data files
    with st.spinner("Loading configuration files..."):
        # Always load lightweight config files
        if st.session_state.waypoints_df is None:
            st.session_state.waypoints_df = load_excel_data("waypoints.xlsx")
        
        if st.session_state.access_rates_df is None:
            st.session_state.access_rates_df = load_excel_data("access_rates.xlsx")
        
        # In local development, preload heavy graph files for performance
        if st.session_state.IS_LOCAL_DEV:
            if st.session_state.railway_detailed is None:
                st.session_state.railway_detailed = load_lz4_data("south_africa_railway_detailed.lz4")
            
            if st.session_state.railway_simple is None:
                st.session_state.railway_simple = load_lz4_data("south_africa_railway_simple.lz4")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    # Waypoint routing section
    with col1:
        st.subheader("📍 Waypoint Routing")
        
        waypoint_names = []
        if st.session_state.waypoints_df is not None and 'name' in st.session_state.waypoints_df.columns:
            waypoint_names = st.session_state.waypoints_df['name'].unique().tolist()
        
        def get_waypoint_coordinates(selected_waypoint):
            if selected_waypoint and st.session_state.waypoints_df is not None:
                waypoint_data = st.session_state.waypoints_df[
                    st.session_state.waypoints_df['name'] == selected_waypoint
                ].iloc[0]
                
                if 'latitude' in waypoint_data and 'longitude' in waypoint_data:
                    return f"{waypoint_data['latitude']}, {waypoint_data['longitude']}"
            return ""
        
        for i in range(1, number_of_waypoints + 1):
            if waypoint_names:
                selected_waypoint = st.selectbox(
                    f"Waypoint {i}:",
                    options=[""] + waypoint_names,
                    key=f"waypoint_select_{i}",
                    help=f"Select waypoint {i} for routing"
                )
            else:
                selected_waypoint = st.selectbox(
                    f"Waypoint {i}:",
                    options=["No waypoints available"],
                    key=f"waypoint_select_{i}",
                    disabled=True
                )
            
            coordinates_value = get_waypoint_coordinates(selected_waypoint) if selected_waypoint else ""
            st.text_input(
                f"Coordinates {i} (Lat, Lon):",
                value=coordinates_value,
                key=f"coordinates_input_{i}",
                help="Format: latitude, longitude"
            )
            
            if i < number_of_waypoints:
                st.markdown("")
           
        st.markdown("")
        if st.button("🗺️ Calculate Route", key="calculate_route_btn", use_container_width=True):
            calculate_route()
            
        if st.session_state.route_calculation_result:
            st.markdown(st.session_state.route_calculation_result)    
            
    # Access rates information
    with col2:
        st.subheader("💰 Access Rate Details")
        
        access_categories = []
        selected_category = None
        
        if st.session_state.access_rates_df is not None:
            if 'Category' in st.session_state.access_rates_df.columns:
                access_categories = st.session_state.access_rates_df['Category'].unique().tolist()
                selected_category = st.selectbox(
                    "Select Access Rate Category:",
                    options=access_categories,
                    key="access_rate_select"
                )
            else:
                st.error("'Category' column not found in access rates data")
        else:
            st.error("Access rates data not loaded")
        
        st.checkbox("Electric locomotives:", key="electric_locomotives")
        
        st.markdown("")
        
        if selected_category and st.session_state.access_rates_df is not None:
            rate_data = st.session_state.access_rates_df[
                st.session_state.access_rates_df['Category'] == selected_category
            ].iloc[0]
            
            required_columns = [
                "Tariff per Trainkm (Rand)",
                "Tariff per GTK (cents)",
                "E Rate per GTK (Cents)"
            ]
            
            for col_name in required_columns:
                if col_name in rate_data:
                    st.text_input(
                        col_name,
                        value=str(rate_data[col_name]),
                        key=f"rate_{col_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
                    )
                else:
                    st.error(f"Column '{col_name}' not found in access rates data")
        else:
            for col_name in ["Tariff per Trainkm (Rand)", "Tariff per GTK (cents)", "E-Rate (Cents/GTK)"]:
                st.text_input(
                    col_name,
                    value="",
                    disabled=True,
                    help="Select an access rate category to see values"
                ) 
          
        st.text_input(
            "Trip distance [km]:",            
            key="trip_distance",
            help="Total trip distance in km"
        ) 
        st.text_input(
            "Loaded train mass [ton]:",
            value="1000",
            key="loaded_train_mass",
            help="Empty train mass in ton"
        )
        st.text_input(
            "Empty train mass [ton]:",
            value="500",
            key="empty_train_mass",
            help="Empty train massin ton"
        )
        st.text_input(
            "Annual volume [ton]:",
            value="20000",
            key="annual_volume",
            help="Annual volume in ton"
        )
         
        st.markdown("")
        if st.button("💲 Calculate Rate", key="calculate_rate_btn", use_container_width=True):
            calculate_rate()
            
        if st.session_state.rates_calculation_result:
            st.markdown(st.session_state.rates_calculation_result) 
            
            st.checkbox("Display breakdown of rates calculation results:", value=True, key="display_breakdown")
    
    # Show route distance if calculated
    if st.session_state.route_calculated and st.session_state.route_distance:
        st.info(f"Current route distance: {st.session_state.route_distance/1000:.2f} km")
        
    # Detailed breakdown of rates calculation (dataframe)
    if st.session_state.rates_calculation_result and st.session_state.display_breakdown:
        st.dataframe(st.session_state.rates_calculation_df[0]) 
        st.dataframe(st.session_state.rates_calculation_df[1]) 
        st.dataframe(st.session_state.rates_calculation_df[2])    
    
    # Full-width map section
    st.markdown("---")
    st.subheader("🗺️ Railway Route Map")
    
    # Display cached map HTML
    map_html = get_map_html()
    st.components.v1.html(map_html, height=600)    

if __name__ == "__main__":
    main()