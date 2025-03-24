# main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import requests
import os
import json
import polyline
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI-Powered Dynamic Route Optimization API",
    description="API for optimizing delivery routes using AI algorithms",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Cache for weather and traffic data
weather_cache = {}
traffic_cache = {}
CACHE_EXPIRY = 30 * 60  # 30 minutes

# Models
class Location(BaseModel):
    lat: float
    lng: float
    name: Optional[str] = None
    address: Optional[str] = None
    priority: Optional[int] = 0
    time_window: Optional[List[int]] = None

class OptimizationRequest(BaseModel):
    locations: List[Location]
    consider_traffic: bool = True
    consider_weather: bool = True
    vehicles: int = 1

class RouteResponse(BaseModel):
    route: List[int]
    total_distance: float
    total_duration: float
    route_geometry: str
    stops: List[Dict[str, Any]]

def create_data_model(locations, travel_times=None):
    """Creates the data model for the problem."""
    data = {}
    
    # Convert locations to a list of (latitude, longitude) tuples
    locations_list = [(loc.lat, loc.lng) for loc in locations]
    data['locations'] = locations_list
    
    # Compute distance matrix if travel_times is not provided
    if travel_times is None:
        data['distance_matrix'] = compute_distance_matrix(locations_list)
    else:
        data['distance_matrix'] = travel_times
    
    data['num_vehicles'] = 1
    data['depot'] = 0  # Start and end at the first location
    
    return data

def compute_distance_matrix(locations):
    """Computes the distance matrix between all locations using Mapbox API."""
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    
    # Use Mapbox Matrix API for batch distance calculation
    if n <= 25:  # Mapbox Matrix API limit
        coordinates = ";".join([f"{lng},{lat}" for lat, lng in locations])
        url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coordinates}?access_token={MAPBOX_API_KEY}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Convert durations (seconds) to distance matrix
            for i in range(n):
                for j in range(n):
                    distance_matrix[i][j] = data['durations'][i][j]
        else:
            # Fallback to Euclidean distance if API fails
            for i in range(n):
                for j in range(n):
                    distance_matrix[i][j] = np.sqrt(
                        (locations[i][0] - locations[j][0])**2 +
                        (locations[i][1] - locations[j][1])**2
                    ) * 111000  # Rough conversion to meters
    else:
        # For large problems, split into batches or use Euclidean distance
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = np.sqrt(
                    (locations[i][0] - locations[j][0])**2 +
                    (locations[i][1] - locations[j][1])**2
                ) * 111000  # Rough conversion to meters
    
    return distance_matrix.tolist()

def get_weather_data(lat, lng):
    """Fetches weather data for a given location."""
    cache_key = f"{lat:.4f},{lng:.4f}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in weather_cache and current_time - weather_cache[cache_key]['timestamp'] < CACHE_EXPIRY:
        return weather_cache[cache_key]['data']
    
    # If not in cache or expired, fetch from API
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        weather_data = response.json()
        result = {
            'temperature': weather_data['main']['temp'],
            'weather_condition': weather_data['weather'][0]['main'],
            'wind_speed': weather_data['wind']['speed']
        }
        
        # Update cache
        weather_cache[cache_key] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
    else:
        return {'temperature': 20, 'weather_condition': 'Clear', 'wind_speed': 5}  # Default values

def get_traffic_data(locations):
    """Fetches traffic data for routes between locations."""
    n = len(locations)
    traffic_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            lat1, lng1 = locations[i].lat, locations[i].lng
            lat2, lng2 = locations[j].lat, locations[j].lng
            
            cache_key = f"{lat1:.4f},{lng1:.4f}-{lat2:.4f},{lng2:.4f}"
            current_time = time.time()
            
            # Check cache first
            if cache_key in traffic_cache and current_time - traffic_cache[cache_key]['timestamp'] < CACHE_EXPIRY:
                traffic_matrix[i][j] = traffic_cache[cache_key]['data']
                continue
            
            # If not in cache or expired, fetch from Mapbox API
            url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{lng1},{lat1};{lng2},{lat2}?access_token={MAPBOX_API_KEY}"
            response = requests.get(url)
            
            if response.status_code == 200:
                route_data = response.json()
                if 'routes' in route_data and len(route_data['routes']) > 0:
                    # Get duration with traffic
                    duration = route_data['routes'][0]['duration']
                    traffic_matrix[i][j] = duration
                    
                    # Update cache
                    traffic_cache[cache_key] = {
                        'data': duration,
                        'timestamp': current_time
                    }
                else:
                    # Fallback to Euclidean distance
                    traffic_matrix[i][j] = np.sqrt(
                        (lat1 - lat2)**2 + (lng1 - lng2)**2
                    ) * 111000 / 35  # Rough conversion to seconds (assuming 35 m/s speed)
            else:
                # Fallback to Euclidean distance
                traffic_matrix[i][j] = np.sqrt(
                    (lat1 - lat2)**2 + (lng1 - lng2)**2
                ) * 111000 / 35
    
    return traffic_matrix.tolist()

def solve_vrp(data):
    """Solves the Vehicle Routing Problem."""
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), 
        data['num_vehicles'],
        data['depot']
    )
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add Distance constraint
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3600 * 8,  # vehicle maximum travel distance (8 hours in seconds)
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Return the solution
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Add the depot at the end
        
        total_distance = solution.ObjectiveValue()
        
        return {
            'route': route,
            'total_distance': total_distance
        }
    else:
        return None

def get_route_geometry(locations, route):
    """Get the polyline geometry for the route."""
    # Extract coordinates for the route
    route_coords = []
    for idx in route:
        route_coords.append((locations[idx].lng, locations[idx].lat))
    
    # Get detailed route from Mapbox if there are at least 2 points
    if len(route_coords) > 1:
        coordinates = ";".join([f"{lng},{lat}" for lng, lat in route_coords])
        url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{coordinates}?geometries=polyline&overview=full&access_token={MAPBOX_API_KEY}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                return data['routes'][0]['geometry'], data['routes'][0]['duration']
    
    # Fallback: Return simple linestring connecting points
    return polyline.encode([(lat, lng) for lng, lat in route_coords]), None

def adjust_matrix_for_weather(matrix, locations, weather_factor=0.2):
    """Adjust travel times based on weather conditions."""
    n = len(locations)
    adjusted_matrix = np.array(matrix)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Get weather at midpoint of the route
            mid_lat = (locations[i].lat + locations[j].lat) / 2
            mid_lng = (locations[i].lng + locations[j].lng) / 2
            weather = get_weather_data(mid_lat, mid_lng)
            
            # Apply weather adjustments
            if weather['weather_condition'] in ['Rain', 'Thunderstorm']:
                adjusted_matrix[i][j] *= (1 + weather_factor)
            elif weather['weather_condition'] in ['Snow', 'Sleet']:
                adjusted_matrix[i][j] *= (1 + weather_factor * 2)
            elif weather['weather_condition'] == 'Fog':
                adjusted_matrix[i][j] *= (1 + weather_factor * 0.5)
            
            # Wind speed adjustment
            if weather['wind_speed'] > 10:  # m/s
                adjusted_matrix[i][j] *= (1 + (weather['wind_speed'] - 10) * 0.01)
    
    return adjusted_matrix.tolist()

def cluster_locations(locations, max_clusters=3):
    """Cluster locations to create multiple vehicle routes."""
    if len(locations) <= 3:
        # For very few locations, no need to cluster
        return [[i for i in range(len(locations))]]
    
    # Extract coordinates
    coords = np.array([[loc.lat, loc.lng] for loc in locations])
    
    # Determine optimal number of clusters
    n_clusters = min(max_clusters, len(locations) // 5 + 1)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(coords)
    
    # Group locations by cluster
    cluster_groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(i)
    
    return list(cluster_groups.values())

@app.post("/optimize", response_model=RouteResponse)
async def optimize_route(request: OptimizationRequest):
    try:
        # Get locations
        locations = request.locations
        
        # Create data model
        data = create_data_model(locations)
        
        # Adjust for traffic if requested
        if request.consider_traffic:
            traffic_matrix = get_traffic_data(locations)
            data['distance_matrix'] = traffic_matrix
        
        # Adjust for weather if requested
        if request.consider_weather and request.consider_traffic:
            data['distance_matrix'] = adjust_matrix_for_weather(
                data['distance_matrix'], 
                locations
            )
        
        # Solve the VRP
        result = solve_vrp(data)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to find an optimal route")
        
        # Get route geometry
        geometry, route_duration = get_route_geometry(locations, result['route'])
        
        # Create stops information
        stops = []
        for idx in result['route']:
            loc = locations[idx]
            stops.append({
                'index': idx,
                'name': loc.name or f"Stop {idx}",
                'address': loc.address or "",
                'lat': loc.lat,
                'lng': loc.lng,
                'priority': loc.priority or 0
            })
        
        # Create response
        return {
            'route': result['route'],
            'total_distance': result['total_distance'],
            'total_duration': route_duration or (result['total_distance'] / 35),  # Fallback if API doesn't provide duration
            'route_geometry': geometry,
            'stops': stops
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-vehicle-optimize")
async def multi_vehicle_optimize(request: OptimizationRequest):
    try:
        locations = request.locations
        
        # Cluster locations based on number of vehicles
        clusters = cluster_locations(locations, max_clusters=request.vehicles)
        
        routes = []
        for cluster in clusters:
            # Create subset of locations for this cluster
            cluster_locations = [locations[i] for i in cluster]
            
            # Create data model for this cluster
            data = create_data_model(cluster_locations)
            
            # Adjust for traffic and weather if requested
            if request.consider_traffic:
                traffic_matrix = get_traffic_data(cluster_locations)
                data['distance_matrix'] = traffic_matrix
                
                if request.consider_weather:
                    data['distance_matrix'] = adjust_matrix_for_weather(
                        data['distance_matrix'], 
                        cluster_locations
                    )
            
            # Solve VRP for this cluster
            result = solve_vrp(data)
            
            if result:
                # Map cluster indices back to original indices
                route = [cluster[i] for i in result['route']]
                
                # Get route geometry
                geometry, route_duration = get_route_geometry([locations[i] for i in route], list(range(len(route))))
                
                # Create stops information
                stops = []
                for idx in route:
                    loc = locations[idx]
                    stops.append({
                        'index': idx,
                        'name': loc.name or f"Stop {idx}",
                        'address': loc.address or "",
                        'lat': loc.lat,
                        'lng': loc.lng,
                        'priority': loc.priority or 0
                    })
                
                routes.append({
                    'route': route,
                    'total_distance': result['total_distance'],
                    'total_duration': route_duration or (result['total_distance'] / 35),
                    'route_geometry': geometry,
                    'stops': stops
                })
        
        return {"routes": routes}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def main():
    return {"message": "Welcome to the AI-Powered Dynamic Route Optimization API!"}

    
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
