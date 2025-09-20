import cv2
import numpy as np
import time
from datetime import datetime, timedelta
import threading
import json
from collections import deque
import random
from flask import Flask, render_template, jsonify, request
import math
from sklearn.linear_model import LinearRegression
import uuid
import os

# Add this class before the TrafficAnalyzer class
class TrafficPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.historical_data = []
        
    def add_data_point(self, hour, day_of_week, congestion_level):
        self.historical_data.append({
            'hour': hour,
            'day_of_week': day_of_week,
            'congestion_level': congestion_level
        })
        
    def train(self):
        # Simple implementation - predict congestion based on time of day
        if len(self.historical_data) > 10:
            X = [[data['hour'], data['day_of_week']] for data in self.historical_data]
            y = [data['congestion_level'] for data in self.historical_data]
            self.model.fit(X, y)
            
    def predict_congestion(self, hour, day_of_week):
        if len(self.historical_data) > 10 and hasattr(self.model, 'coef_'):
            return self.model.predict([[hour, day_of_week]])[0]
        return 0
# Add this class before TrafficAnalyzer
class IoTSensorSimulator:
    def __init__(self):
        self.sensor_data = {}
        
    def generate_sensor_data(self):
        # Simulate various IoT sensors
        return {
            'weather': random.choice(['clear', 'rain', 'fog']),
            'road_conditions': random.choice(['dry', 'wet', 'icy']),
            'emergency_vehicle': random.random() < 0.05,  # 5% chance
            'air_quality': random.randint(50, 150),
            'timestamp': datetime.now().isoformat()
        }

class TrafficAnalyzer:
    def __init__(self, video_sources, config_file="traffic_config.json"):
        # Store video sources
        self.video_sources = video_sources
        self.config_file = config_file
        
        # Admin control properties
        self.manual_control = {}  # Track which intersections are manually controlled
        self.emergency_mode = False
        self.signal_timings_backup = self.signal_timings.copy() if hasattr(self, 'signal_timings') else []
        
        # Ensure signal_states is properly initialized
        if not hasattr(self, 'signal_states'):
            self.signal_states = [False] * 4
        if not hasattr(self, 'current_green'):
            self.current_green = 0
        
        # Camera statistics
        self.camera_stats = []

        # Initialize analyzers for each video source
        self.analyzers = []
        for source in video_sources:
            analyzer = SingleVideoAnalyzer(source, config_file)
            self.analyzers.append(analyzer)
            
        # Load configuration
        self.config = self.load_config(config_file)
        self.predictor = TrafficPredictor()
        self.iot_simulator = IoTSensorSimulator()
        
        


        # Data storage (aggregated from all analyzers)
        self.traffic_data = {
            "current_vehicles": 0,
            "total_vehicles": 0,
            "vehicle_types": {"car": 0, "motorbike": 0, "bus": 0, "truck": 0},
            "congestion_level": 0,
            "average_speed": 0
        }
        
        # Historical data for trend analysis
        self.historical_data = deque(maxlen=100)
        
        # Signal control parameters (shared across all intersections)
        self.signal_states = [False] * 16  
        self.signal_timings = [30] * 16   # Default 30 seconds for each direction
        self.current_green = 0
        self.last_signal_change = time.time()
        
        # Thread for real-time processing
        self.processing = False
        self.process_thread = None
        
        # Simulation integration
        self.simulator = None
        self.dashboard_data = deque(maxlen=1000)

        self.analyzer_threads = []
        self.stop_event = threading.Event()

    


    def ensure_signal_states_format(self):
        """Ensure signal states are in the correct format for the dashboard"""
        if not hasattr(self, 'signal_states') or not self.signal_states:
            self.signal_states = [False] * 4
            self.current_green = 0
            return
    
        # If we have 16 signals, convert to 4 for dashboard
        if len(self.signal_states) == 16:
            # For dashboard, use only the first intersection's signals
            # Get the green signals for the first intersection (positions 0-3)
            first_intersection_signals = self.signal_states[0:4]
            self.signal_states = first_intersection_signals
        elif len(self.signal_states) != 4:
            # If unexpected length, default to 4 signals
            self.signal_states = [False] * 4


    # In detect_traffic.py, add this method to TrafficAnalyzer:
    def ensure_signal_states_format(self):
        """Ensure signal states are in the correct format for the dashboard"""
        if not hasattr(self, 'signal_states') or not self.signal_states:
            self.signal_states = [False] * 4
            self.current_green = 0
            return
        
        # If we have 16 signals, convert to 4 for dashboard
        if len(self.signal_states) == 16:
            # For dashboard, use only the first intersection
            self.signal_states = self.signal_states[0:4]
        elif len(self.signal_states) != 4:
            # If format is unexpected, default to 4 signals
            self.signal_states = [False] * 4
    
    # Add stop method to TrafficAnalyzer
    def stop(self):
        """Stop all analyzers"""
        self.processing = False
        for analyzer in self.analyzers:
            if hasattr(analyzer, 'stop'):
                analyzer.stop()

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            "congestion_thresholds": {
                "low": 5,
                "medium": 15,
                "high": 25
            },
            "detection_interval": 2,
            "output_video": True,
            "log_data": True,
            "min_contour_area": 800,  # Increased to reduce false positives
            "max_contour_area": 3000,
            "detection_confidence": 0.7,  # Confidence threshold for vehicle detection
            "max_vehicle_speed": 50,  # pixels per second
            "vehicle_type_thresholds": {
                "motorbike": {"min_width": 30, "max_width": 60, "min_height": 30, "max_height": 60},
                "car": {"min_width": 60, "max_width": 120, "min_height": 40, "max_height": 80},
                "bus": {"min_width": 120, "max_width": 200, "min_height": 80, "max_height": 120},
                "truck": {"min_width": 100, "max_width": 180, "min_height": 70, "max_height": 110}
            },
            "aspect_ratio_thresholds": {
                "motorbike": {"min": 0.8, "max": 1.5},
                "car": {"min": 1.5, "max": 2.5},
                "bus": {"min": 2.0, "max": 3.5},
                "truck": {"min": 1.8, "max": 3.0}
            },
            "dashboard": {
                "host": "0.0.0.0",
                "port": 5000
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default settings.")
        
        return default_config
    
    def setup_detection_zones(self):
        """Setup detection zones for traffic analysis (invisible)"""
        # Define a region of interest (ROI) - typically the road area
        self.roi_points = np.array([
            [int(self.width * 0.1), int(self.height * 0.7)],  # Bottom left
            [int(self.width * 0.1), int(self.height * 0.3)],  # Top left
            [int(self.width * 0.9), int(self.height * 0.3)],  # Top right
            [int(self.width * 0.9), int(self.height * 0.7)]   # Bottom right
        ], np.int32)
        
        # Create mask for ROI (used internally, not displayed)
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_points], 255)
        
        # Detection line (vehicles crossing this line are counted) - invisible
        self.detection_line_y = int(self.height * 0.6)
    
    def setup_video_writer(self):
        """Setup video writer for output"""
        if self.config["output_video"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter("output_annotated.mp4", fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None

    
    def detect_vehicles(self, frame):
        """Detect vehicles using background subtraction and contour analysis with improved filtering"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI mask (invisible to user)
        gray_masked = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
        
        # Apply background subtraction
        fgmask = self.fgbg.apply(gray_masked)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to smooth the mask
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        
        # Threshold to create binary image
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if not (self.config["min_contour_area"] < area < self.config["max_contour_area"]):
                continue
                
            # Filter by aspect ratio (vehicles are typically wider than tall)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio < 0.8 or aspect_ratio > 4.0:  # Unlikely vehicle aspect ratios
                continue
                
            # Filter by solidity (how convex the shape is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.5:  # Vehicles tend to be fairly solid shapes
                continue
                
            # Passed all filters - likely a vehicle
            vehicles.append((x, y, w, h))
                
        return vehicles
    
    def track_vehicles(self, current_vehicles, current_time):
        """Optimized vehicle tracking with better matching and lifetime management"""
        # Extract centroids from current vehicles
        current_centroids = []
        for (x, y, w, h) in current_vehicles:
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy, w, h))
    
        # Clean up old vehicles first (using the separate method)
        self.cleanup_old_vehicles(current_time)
    
        # If no vehicles currently tracked, initialize tracking
        if not self.tracked_vehicles:
            return self.initialize_tracking(current_centroids, current_time)
    
        # Match current centroids with tracked vehicles using optimized approach
        matched_ids = []
        matched_indices = set()
        tracked_ids = list(self.tracked_vehicles.keys())
    
        # Create a list of old centroids for matching
        old_centroids = []
        for vehicle_id in tracked_ids:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            old_centroids.append(vehicle_data['centroid'] + (vehicle_id,))
    
        # Perform matching
        for i, (old_cx, old_cy, vehicle_id) in enumerate(old_centroids):
            vehicle_data = self.tracked_vehicles[vehicle_id]
            best_match_idx = None
            min_distance = float('inf')
        
            for j, (cx, cy, w, h) in enumerate(current_centroids):
                if j in matched_indices:
                    continue
                
                distance = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                max_distance = self.config["max_vehicle_speed"] * (current_time - vehicle_data['last_seen'])
            
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_match_idx = j
        
            if best_match_idx is not None:
                cx, cy, w, h = current_centroids[best_match_idx]
                self.update_tracked_vehicle(vehicle_id, cx, cy, w, h, current_time)
                matched_ids.append(vehicle_id)
                matched_indices.add(best_match_idx)
    
        # Add new vehicles for unmatched centroids
        for i, (cx, cy, w, h) in enumerate(current_centroids):
            if i not in matched_indices:
                self.add_new_vehicle(cx, cy, w, h, current_time)
                matched_ids.append(self.next_vehicle_id - 1)  # ID of the newly added vehicle
    
        return matched_ids

    def initialize_tracking(self, current_centroids, current_time):
        """Initialize tracking for new vehicles"""
        for cx, cy, w, h in current_centroids:
            self.add_new_vehicle(cx, cy, w, h, current_time)
        return list(self.tracked_vehicles.keys())

    def add_new_vehicle(self, cx, cy, w, h, current_time):
        """Add a new vehicle to tracking"""
        self.tracked_vehicles[self.next_vehicle_id] = {
            'centroid': (cx, cy),
            'width': w,
            'height': h,
            'last_seen': current_time,
            'first_seen': current_time,
            'counted': False
        }
        # Initialize path tracking with optimized structure
        self.vehicle_paths[self.next_vehicle_id] = {
            'positions': np.zeros((10, 2), dtype=np.float32),
            'timestamps': np.zeros(10, dtype=np.float64),
            'index': 0,
            'count': 0
        }
        self.update_vehicle_path(self.next_vehicle_id, cx, cy, current_time)
        self.next_vehicle_id += 1

    def update_tracked_vehicle(self, vehicle_id, cx, cy, w, h, current_time):
        """Update an existing tracked vehicle"""
        vehicle_data = self.tracked_vehicles[vehicle_id]
        self.tracked_vehicles[vehicle_id] = {
            'centroid': (cx, cy),
            'width': w,
            'height': h,
            'last_seen': current_time,
            'first_seen': vehicle_data['first_seen'],
            'counted': vehicle_data['counted'] or cy > self.detection_line_y
        }
        # Update path tracking
        self.update_vehicle_path(vehicle_id, cx, cy, current_time)

    def update_vehicle_path(self, vehicle_id, x, y, timestamp):
        """Update vehicle path with simpler storage"""
        if vehicle_id not in self.vehicle_paths:
            self.vehicle_paths[vehicle_id] = []
    
        # Add new point
        path = self.vehicle_paths[vehicle_id]
        idx = path['index']
        path['positions'][idx % 10] = [x, y]
        path['timestamps'][idx % 10] = timestamp
        path['index'] = idx + 1
        path['count'] = min(path['count'] + 1, 10)

    
        # Keep only last 10 points
        if len(self.vehicle_paths[vehicle_id]) > 10:
            self.vehicle_paths[vehicle_id] = self.vehicle_paths[vehicle_id][-10:]

    def cleanup_old_vehicles(self, current_time):
        """Remove vehicles that haven't been seen for a while"""
        vehicles_to_remove = []
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            if current_time - vehicle_data['last_seen'] > self.vehicle_lifetime:
                vehicles_to_remove.append(vehicle_id)
    
        for vehicle_id in vehicles_to_remove:
            if vehicle_id in self.vehicle_paths:
                del self.vehicle_paths[vehicle_id]
            del self.tracked_vehicles[vehicle_id]
    
    def estimate_vehicle_type(self, width, height):
        """Improved vehicle type estimation with better thresholds and aspect ratio"""
        thresholds = self.config["vehicle_type_thresholds"]
        aspect_ratio = width / height if height > 0 else 0
        aspect_thresholds = self.config["aspect_ratio_thresholds"]
        
        # Check if dimensions fit within any category with aspect ratio constraints
        if (thresholds["motorbike"]["min_width"] <= width <= thresholds["motorbike"]["max_width"] and
            thresholds["motorbike"]["min_height"] <= height <= thresholds["motorbike"]["max_height"] and
            aspect_thresholds["motorbike"]["min"] <= aspect_ratio <= aspect_thresholds["motorbike"]["max"]):
            return "motorbike"
        elif (thresholds["car"]["min_width"] <= width <= thresholds["car"]["max_width"] and
              thresholds["car"]["min_height"] <= height <= thresholds["car"]["max_height"] and
              aspect_thresholds["car"]["min"] <= aspect_ratio <= aspect_thresholds["car"]["max"]):
            return "car"
        elif (thresholds["bus"]["min_width"] <= width <= thresholds["bus"]["max_width"] and
              thresholds["bus"]["min_height"] <= height <= thresholds["bus"]["max_height"] and
              aspect_thresholds["bus"]["min"] <= aspect_ratio <= aspect_thresholds["bus"]["max"]):
            return "bus"
        elif (thresholds["truck"]["min_width"] <= width <= thresholds["truck"]["max_width"] and
              thresholds["truck"]["min_height"] <= height <= thresholds["truck"]["max_height"] and
              aspect_thresholds["truck"]["min"] <= aspect_ratio <= aspect_thresholds["truck"]["max"]):
            return "truck"
        else:
            # If no category fits well, use aspect ratio as primary classifier
            if aspect_ratio < 1.2:
                return "motorbike"
            elif 1.2 <= aspect_ratio < 2.0:
                return "car"
            elif 2.0 <= aspect_ratio < 2.8:
                return "truck"
            else:
                return "bus"
            
    def start_processing(self):
            """Professional multi-threading with proper resource management"""
            self.processing = True
            self.stop_event.clear()
        
            print(f"🚦 Starting {len(self.analyzers)} traffic analyzers...")
        
            # Start all analyzers in separate threads with proper error handling
            for i, analyzer in enumerate(self.analyzers):
                thread = threading.Thread(
                    target=self._analyzer_worker,
                    args=(analyzer, i),
                   name=f"TrafficAnalyzer-{i}",
                    daemon=True
                )
                thread.start()
                self.analyzer_threads.append(thread)
                print(f"📹 Started analyzer {i}: {analyzer.video_source}")

    def _analyzer_worker(self, analyzer, analyzer_id):
        """Worker thread for each video analyzer"""
        try:
            analyzer.run(self.stop_event)
        except Exception as e:
            print(f"❌ Analyzer {analyzer_id} crashed: {e}")
            # Implement proper error reporting here
    
    def stop(self):
        """Graceful shutdown of all analyzers"""
        print("🛑 Shutting down traffic analysis system...")
        self.stop_event.set()
        self.processing = False
        
        # Wait for all threads to finish
        for thread in self.analyzer_threads:
            thread.join(timeout=5.0)
        
        print("✅ All analyzers stopped gracefully")
    
    def update_traffic_data(self):
        """Update traffic data from all analyzers"""
        total_vehicles = 0
        total_speed = 0
        speed_count = 0
        vehicle_types = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}

        for analyzer in self.analyzers:
            # Get current stats from each analyzer
            data = analyzer.get_current_stats()
            total_vehicles += data.get("current_vehicles", 0)

            # Only add speed if it's valid
            avg_speed = data.get("average_speed", 0)
            if avg_speed > 0:
                total_speed += avg_speed
                speed_count += 1

            # Aggregate vehicle types
            for vtype in vehicle_types:
                vehicle_types[vtype] += data.get("vehicle_types", {}).get(vtype, 0)

        # Update aggregated data
        self.traffic_data.update({
            "current_vehicles": total_vehicles,
            "vehicle_types": vehicle_types,
            "average_speed": total_speed / speed_count if speed_count > 0 else 0,
            "congestion_level": self.calculate_congestion_level(total_vehicles),
            "timestamp": datetime.now().isoformat()
        })

        # Update total vehicles count
        for analyzer in self.analyzers:
            self.traffic_data["total_vehicles"] += analyzer.traffic_data.get("total_vehicles", 0)

        # Ensure signal states are in correct format
        self.ensure_signal_states_format()
    
    def calculate_congestion_level(self, vehicle_count):
        """Calculate congestion level based on vehicle count"""
        thresholds = self.config["congestion_thresholds"]
        if vehicle_count < thresholds["low"]:
            return 0  # Low congestion
        elif vehicle_count < thresholds["medium"]:
            return 1  # Medium congestion
        else:
            return 2  # High congestion
    
    def optimize_signal_timing(self):
        """Optimize traffic signal timing based on current traffic conditions"""
        congestion_level = self.traffic_data["congestion_level"]
        vehicle_count = self.traffic_data["current_vehicles"]
        
        # Enhanced optimization algorithm with stronger adjustments
        if congestion_level == 2:  # High congestion
            additional_time = min(60, 30 + vehicle_count * 3)  # Up to 60 seconds
        elif congestion_level == 1:  # Medium congestion
            additional_time = 30  # Standard timing
        else:  # Low congestion
            additional_time = max(15, 30 - vehicle_count)  # Minimum 15 seconds
            
        # Check for ambulance in any intersection
        for analyzer in self.analyzers:
            if hasattr(analyzer, 'ambulance_detected') and analyzer.ambulance_detected:
                if (time.time() - analyzer.ambulance_detection_time) < 30:
                    additional_time = 45  # Extended green time for ambulance
                    break
            
        return additional_time
    
    # In detect_traffic.py, update the update_signal_control method:
    def update_signal_control(self):
        """Update traffic signal control based on analysis"""
        # Get optimized timing for current direction
        optimized_time = self.optimize_signal_timing()

        # Update timing for all intersections in the current direction
        for i in range(4):  # For each intersection
            signal_index = i * 4 + self.current_green
            if signal_index < len(self.signal_timings):
                self.signal_timings[signal_index] = optimized_time

        # Simple signal rotation
        current_time = time.time()
        if current_time - self.last_signal_change > optimized_time:
            # Turn off all signals
            for i in range(len(self.signal_states)):
                self.signal_states[i] = False

            # Set new green signals for all intersections
            self.current_green = (self.current_green + 1) % 4
            for i in range(4):  # For each intersection
                signal_index = i * 4 + self.current_green
                if signal_index < len(self.signal_states):
                    self.signal_states[signal_index] = True

            self.last_signal_change = current_time
            print(f"Signal changed to direction {self.current_green} for {optimized_time} seconds")
    
    def estimate_vehicle_speed(self, vehicle_id):
        """Estimate vehicle speed based on path history"""
        if vehicle_id not in self.vehicle_paths:
            return 0

        path = self.vehicle_paths[vehicle_id]
        positions = path['positions']
        timestamps = path['timestamps']
        count = path['count']
        index = path['index']

        if count < 2:
            return 0

        # Get points in correct (time) order (oldest to newest)
        # Ring buffer logic
        points = []
        for k in range(count):
            i = (index - count + k) % 10
            x, y = positions[i]
            t = timestamps[i]
            points.append((x, y, t))

        total_distance = 0
        total_time = 0
        for i in range(1, len(points)):
            x1, y1, t1 = points[i-1]
            x2, y2, t2 = points[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_diff = t2 - t1
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff

        if total_time > 0:
            speed = total_distance / total_time  # pixels per second
            speed_kmh = speed * 0.1 * 3.6       # if 1 pixel = 0.1 meters
            return speed_kmh

        return 0

    
    def process_frame(self, frame, frame_id, current_time):
        """Process a single frame for traffic analysis"""
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Track vehicles
        vehicle_ids = self.track_vehicles(vehicles, current_time)
        
        # Count vehicles by type and check if they've crossed the detection line
        vehicle_types = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0, "unknown": 0}
        current_vehicles_count = 0
        total_speed = 0
        speed_count = 0
        
        for vehicle_id in vehicle_ids:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            
            # Only count vehicles that are currently in the frame
            if vehicle_data['last_seen'] == current_time:
                vehicle_type = self.estimate_vehicle_type(vehicle_data['width'], vehicle_data['height'])
                if vehicle_type in vehicle_types:
                    vehicle_types[vehicle_type] += 1
                current_vehicles_count += 1
                
                # Calculate speed for this vehicle
                speed = self.estimate_vehicle_speed(vehicle_id)
                if speed > 0:
                    total_speed += speed
                    speed_count += 1
                
            # Count vehicle in total if it crossed the detection line and hasn't been counted yet
            if (not vehicle_data['counted'] and 
                vehicle_data['centroid'][1] > self.detection_line_y and
                vehicle_data['last_seen'] == current_time):
                vehicle_data['counted'] = True
                self.traffic_data["total_vehicles"] += 1
        
        # Calculate average speed
        avg_speed = round(total_speed / speed_count, 1) if speed_count > 0 else 0
        
        # Update traffic data
        self.traffic_data.update({
            "current_vehicles": current_vehicles_count,
            "vehicle_types": vehicle_types,
            "congestion_level": self.calculate_congestion_level(current_vehicles_count),
            "average_speed": avg_speed,
            "timestamp": datetime.now().isoformat()
        })
        
        # In process_frame method, after updating traffic_data:
        # Add data to predictor and train
        current_time_obj = datetime.now()
        self.predictor.add_data_point(
            current_time_obj.hour, 
            current_time_obj.weekday(), 
            self.traffic_data['congestion_level']
        )
        self.predictor.train()

        # Store historical data for trend analysis
        self.historical_data.append(self.traffic_data.copy())
        
        # Update dashboard data
        self.update_dashboard_data()
        
        return vehicles, vehicle_ids
    
    def update_dashboard_data(self):
        """Update data for the dashboard"""
        current_data = {
            'timestamp': datetime.now().isoformat(),
            'current_vehicles': self.traffic_data['current_vehicles'],
            'total_vehicles': self.traffic_data['total_vehicles'],
            'vehicle_types': self.traffic_data['vehicle_types'],
            'congestion_level': self.traffic_data['congestion_level'],
            'average_speed': self.traffic_data['average_speed'],
            'signal_states': self.signal_states.copy(),
            'signal_timings': self.signal_timings.copy(),
            'current_green': self.current_green
        }
        
        self.dashboard_data.append(current_data)
        # Keep only last 1000 entries
        if len(self.dashboard_data) > 1000:
            self.dashboard_data = deque(list(self.dashboard_data)[-1000:], maxlen=1000)

    def _generate_overlay(self, base_frame, vehicle_ids):
        """Create a persistent overlay that doesn't blink"""
        overlay = base_frame.copy()
    
        # -----------------------------
        # Bounding boxes - show ALL tracked vehicles, not just current ones
        # -----------------------------
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            # Only show vehicles that have been seen recently (within last 2 seconds)
            if time.time() - vehicle_data['last_seen'] < 2.0:
                cx, cy = vehicle_data['centroid']
                w, h = vehicle_data['width'], vehicle_data['height']
                x, y = cx - w // 2, cy - h // 2

                # Vehicle color based on type
                vehicle_type = self.estimate_vehicle_type(w, h)
                if vehicle_type == "car":
                    color = (0, 255, 0)
                elif vehicle_type == "motorbike":
                    color = (255, 0, 0)
                elif vehicle_type == "bus":
                    color = (0, 165, 255)
                elif vehicle_type == "truck":
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)

                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
                label = f"{vehicle_id}:{vehicle_type}"
                cv2.putText(overlay, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # -----------------------------
        # Dashboard info box (unchanged)
            # -----------------------------
        y_offset = 40
        line_height = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        info_lines = [
            "TRAFFIC MANAGEMENT SYSTEM",
            "",
            "VEHICLES:",
            f"  Current: {self.traffic_data.get('current_vehicles', 0)}",
            f"  Total: {self.traffic_data.get('total_vehicles', 0)}",
            f"  Cars: {self.traffic_data['vehicle_types'].get('car', 0)}  Bikes: {self.traffic_data['vehicle_types'].get('motorbike', 0)}",
            f"  Buses: {self.traffic_data['vehicle_types'].get('bus', 0)}  Trucks: {self.traffic_data['vehicle_types'].get('truck', 0)}",
            "",
            "TRAFFIC CONDITIONS:",
            f"  Congestion: {'LOW' if self.traffic_data.get('congestion_level', 0) == 0 else 'MEDIUM' if self.traffic_data.get('congestion_level', 0) == 1 else 'HIGH'}",
            f"  Avg Speed: {self.traffic_data.get('average_speed', 0)} km/h",
            "",
            "SIGNAL STATUS:"
        ]

        # Add intersection signal info (always show something)
        for i in range(4):
            directions = []
            for j in range(4):
                idx = i * 4 + j
                if idx < len(self.signal_states) and self.signal_states[idx]:
                    directions.append(f"D{j}")
            if directions:
                info_lines.append(f"  Intersection {i}: {', '.join(directions)}")
            else:
                info_lines.append(f"  Intersection {i}: None")

        info_lines.append("")
        info_lines.append(f"Active Direction: {self.current_green}")
        info_lines.append(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

        # -----------------------------
        # Draw info panel background
        # -----------------------------
        max_line_length = max(len(line) for line in info_lines)
        max_width = int(max_line_length * 18) + 60
        total_text_height = y_offset + line_height * len(info_lines) + 20

        # Solid panel (no blinking)
        cv2.rectangle(overlay, (10, 10), (max_width, total_text_height), (40, 40, 40), -1)
        cv2.rectangle(overlay, (10, 10), (20, total_text_height), (0, 150, 255), -1)  # Accent bar
        cv2.rectangle(overlay, (10, 10), (max_width, total_text_height), (0, 150, 255), 2)

        # -----------------------------
        # Draw text lines
        # -----------------------------
        for i, line in enumerate(info_lines):
            pos = (25, y_offset + i * line_height)
            if i == 0:
                cv2.putText(overlay, line, pos, font, 1.0, (0, 255, 255), thickness + 1)
            elif line in ["VEHICLES:", "TRAFFIC CONDITIONS:", "SIGNAL STATUS:"]:
                cv2.putText(overlay, line, pos, font, 0.9, (0, 200, 255), thickness + 1)
            elif "LOW" in line:
                cv2.putText(overlay, line, pos, font, font_scale, (0, 255, 0), thickness)
            elif "MEDIUM" in line:
                cv2.putText(overlay, line, pos, font, font_scale, (0, 165, 255), thickness)
            elif "HIGH" in line:
                cv2.putText(overlay, line, pos, font, font_scale, (0, 0, 255), thickness)
            elif "Active Direction:" in line:
                cv2.putText(overlay, line, pos, font, font_scale, (200, 255, 200), thickness)
            elif "Last Update:" in line:
                cv2.putText(overlay, line, pos, font, font_scale, (200, 200, 255), thickness)
            elif line.startswith("  Intersection"):
                color = (150, 150, 150) if "None" in line else (0, 255, 150)
                cv2.putText(overlay, line, pos, font, font_scale, color, thickness)
            else:
                cv2.putText(overlay, line, pos, font, font_scale, (255, 255, 255), thickness)

        return overlay



    def annotate_frame(self, frame, vehicles, vehicle_ids):
        """Annotate frame with traffic information - WITHOUT the red line and box"""
        annotated_frame = frame.copy()

        # Draw bounding boxes around vehicles ONLY (no ROI or detection line)
        for vehicle_id in vehicle_ids:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            cx, cy = vehicle_data['centroid']
            w, h = vehicle_data['width'], vehicle_data['height']
            x, y = cx - w//2, cy - h//2

            # Only draw vehicles detected in the current frame
            if vehicle_data['last_seen'] == time.time():
                # Choose color based on vehicle type
                vehicle_type = self.estimate_vehicle_type(w, h)
                if vehicle_type == "car":
                    color = (0, 255, 0)  # Green for cars
                elif vehicle_type == "motorbike":
                    color = (255, 0, 0)  # Blue for motorbikes
                elif vehicle_type == "bus":
                    color = (0, 165, 255)  # Orange for buses
                elif vehicle_type == "truck":
                    color = (0, 0, 255)  # Red for trucks
                else:
                    color = (255, 255, 0)  # Cyan for unknown

                # Draw rectangle with thicker lines
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)

                # Draw vehicle ID and type with larger font
                label = f"{vehicle_id}:{vehicle_type}"
                cv2.putText(annotated_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Add traffic information overlay with background for readability
        y_offset = 40
        line_height = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Format information for better display
        info_lines = [
            "TRAFFIC MANAGEMENT SYSTEM",
            "",
            "VEHICLES:",
            f"  Current: {self.traffic_data['current_vehicles']}",
            f"  Total: {self.traffic_data['total_vehicles']}",
            f"  Cars: {self.traffic_data['vehicle_types']['car']}  Bikes: {self.traffic_data['vehicle_types']['motorbike']}",
            f"  Buses: {self.traffic_data['vehicle_types']['bus']}  Trucks: {self.traffic_data['vehicle_types']['truck']}",
            "",
            "TRAFFIC CONDITIONS:",
            f"  Congestion: {'LOW' if self.traffic_data['congestion_level'] == 0 else 'MEDIUM' if self.traffic_data['congestion_level'] == 1 else 'HIGH'}",
            f"  Avg Speed: {self.traffic_data['average_speed']} km/h",
            "",
            "SIGNAL STATUS:"
        ]

        # Add intersection information with better formatting
        for i in range(4):
            directions = []
            for j in range(4):
                idx = i * 4 + j
                if self.signal_states[idx]:
                    directions.append(f"D{j}")
            if directions:
                info_lines.append(f"  Intersection {i}: {', '.join(directions)}")
            else:
                info_lines.append(f"  Intersection {i}: None")

        info_lines.append("")
        info_lines.append(f"Active Direction: {self.current_green}")
        info_lines.append(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

        # Calculate maximum line length for background
        max_line_length = max([len(line) for line in info_lines])
        # Estimate pixel width with more space
        max_width = int(max_line_length * 18) + 60

        # Calculate total text height with some extra padding
        total_text_height = y_offset + line_height * len(info_lines) + 20

        # Draw semi-transparent background for text with rounded corners effect
        overlay = annotated_frame.copy()

        # Draw main background
        cv2.rectangle(overlay, (10, 10), (max_width, total_text_height), (40, 40, 40), -1)

        # Draw accent bar on left
        cv2.rectangle(overlay, (10, 10), (20, total_text_height), (0, 150, 255), -1)

        # Add border
        cv2.rectangle(overlay, (10, 10), (max_width, total_text_height), (0, 150, 255), 2)

        # Apply overlay
        cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)

        # Draw text with improved styling
        for i, line in enumerate(info_lines):
            if i == 0:  # Header line
                color = (0, 255, 255)  # Yellow
                cv2.putText(annotated_frame, line, (30, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness + 1)
            elif line in ["VEHICLES:", "TRAFFIC CONDITIONS:", "SIGNAL STATUS:"]:  # Section headers
                color = (0, 200, 255)  # Light orange
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness + 1)
            elif "LOW" in line:
                color = (0, 255, 0)  # Green for low congestion
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "MEDIUM" in line:
                color = (0, 165, 255)  # Orange for medium congestion
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "HIGH" in line:
                color = (0, 0, 255)  # Red for high congestion
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "Active Direction:" in line:
                color = (200, 255, 200)  # Light green
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "Last Update:" in line:
                color = (200, 200, 255)  # Light purple
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif line.startswith("  Intersection"):
                if "None" in line:
                    color = (150, 150, 150)  # Gray for no active signals
                else:
                    color = (0, 255, 150)  # Bright green for active signals
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif line != "":  # Regular text
                color = (200, 255, 200)  # Light green
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)

        return annotated_frame

    
    # In detect_traffic.py, modify the start_processing method
    def start_processing(self):
        """Start processing all video sources"""
        self.processing = True
    
        # Start all but first analyzer in background threads
        for analyzer in self.analyzers[1:]:
            analyzer.start()
    
        # Run first analyzer in main thread (so 'q' works)
        if self.analyzers:
            print(f"Running main analyzer in foreground, {len(self.analyzers)-1} in background")
            self.analyzers[0].start_processing()  # This should block until 'q' is pressed


    def start(self):
        """Start the traffic analysis system"""
        self.process_thread = threading.Thread(target=self.start_processing)
        self.process_thread.start()
    
    def stop(self):
        """Stop the traffic analysis system"""
        self.processing = False
        if self.process_thread:
            self.process_thread.join()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
    
    # Simulation integration methods
    def connect_to_simulator(self, simulator):
        """Connect to the urban simulator"""
        self.simulator = simulator
        # Also give simulator access to analyzer for bidirectional communication
        if hasattr(simulator, 'connect_to_analyzer'):
            simulator.connect_to_analyzer(self)
        print("Simulator connected with bidirectional data exchange")
        
    def get_traffic_data_for_simulation(self):
        """Get traffic data formatted for the simulator"""
        self.update_traffic_data()  # Make sure data is updated
        return {
            'current_vehicles': self.traffic_data['current_vehicles'],
            'vehicle_types': self.traffic_data['vehicle_types'],
            'congestion_level': self.traffic_data['congestion_level'],
            'average_speed': self.traffic_data['average_speed'],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_signal_states(self, vehicles, current_time):
        """Get signal states for the simulator - returns proper format for dashboard"""
        # Calculate congestion per direction (simplified)
        congestion_by_direction = [0] * 4

        for vehicle in vehicles:
            if vehicle.get('status') != 'waiting':
                continue
        
            # Simplified: assume each vehicle is waiting at an intersection
            direction = hash(str(vehicle.get('id', ''))) % 4
            congestion_by_direction[direction] += 1

        # Set green light for the direction with most congestion
        max_congestion = max(congestion_by_direction) if congestion_by_direction else 0
        if max_congestion > 0:
            green_direction = congestion_by_direction.index(max_congestion)
        else:
            # Default rotation if no congestion
            green_direction = (self.current_green + 1) % 4

        # Create signal states array with 4 values (one per direction)
        signal_states_array = [False] * 4
        signal_states_array[green_direction] = True

        # Also update the internal signal states for consistency
        self.signal_states = signal_states_array
        self.current_green = green_direction
        self.last_signal_change = current_time.timestamp() if hasattr(current_time, 'timestamp') else time.time()

        return signal_states_array  # Return array with 4 values
    
    def get_current_stats(self):
        """Get current aggregated traffic statistics"""
        self.update_traffic_data()
        return self.traffic_data
    
    def get_historical_stats(self, limit=100):
        """Get historical traffic statistics"""
        # This would need to be modified to handle multiple sources
        # For simplicity, return data from the first analyzer
        if self.analyzers:
            return self.analyzers[0].get_historical_stats(limit)
        return []
    
    def manual_signal_control(self, intersection, direction, duration):
        """Manual control of traffic signals"""
        print(f"Manual control: Intersection {intersection}, Direction {direction}, Duration {duration}")
        # In a real implementation, this would update the actual signal control system

    def predict_bottlenecks(self):
        """Predict potential bottlenecks based on current conditions"""
        sensor_data = self.iot_simulator.generate_sensor_data()
        bottlenecks = []
        
        # Simple rules-based prediction
        if self.traffic_data['current_vehicles'] > 20 and sensor_data['weather'] == 'rain':
            bottlenecks.append({
                'location': 'Main Intersection',
                'severity': 'high',
                'expected_delay': '15-20 minutes',
                'recommendation': 'Extend green time by 30%'
            })
        
        if self.traffic_data['current_vehicles'] > 15:
            bottlenecks.append({
                'location': 'Downtown Corridor',
                'severity': 'medium',
                'expected_delay': '5-10 minutes',
                'recommendation': 'Adjust signal phasing'
            })
        
        return {
            'bottlenecks': bottlenecks,
            'sensor_data': sensor_data
        }
    
def connect_to_simulator(self, simulator):
    """Connect to the urban simulator"""
    self.simulator = simulator
    # Also give simulator access to analyzer for bidirectional communication
    if hasattr(simulator, 'connect_to_analyzer'):
        simulator.connect_to_analyzer(self)
    print("Simulator connected with bidirectional data exchange")

    def get_traffic_data_for_simulation(self):
        """Get comprehensive traffic data formatted for the simulator"""
        return {
            'current_vehicles': self.traffic_data['current_vehicles'],
            'vehicle_types': self.traffic_data['vehicle_types'],
            'congestion_level': self.traffic_data['congestion_level'],
            'average_speed': self.traffic_data['average_speed'],
            'signal_states': self.signal_states.copy(),
            'signal_timings': self.signal_timings.copy(),
            'current_green': self.current_green,
            'timestamp': datetime.now().isoformat()
        }

    def update_from_simulation(self, simulation_data):
        """Receive data from simulation to update real-world analysis"""
        if 'recommended_signal_timings' in simulation_data:
            # Update signal timings based on simulation recommendations
            for i, timing in enumerate(simulation_data['recommended_signal_timings']):
                if i < len(self.signal_timings):
                    self.signal_timings[i] = timing
    
        if 'congestion_predictions' in simulation_data:
            # Use simulation predictions to anticipate congestion
            self.simulated_congestion_predictions = simulation_data['congestion_predictions']
        
        print("Received simulation data update")

# detect_traffic.py - Complete SingleVideoAnalyzer class with all methods

class SingleVideoAnalyzer:
    """Class to handle analysis for a single video source"""
    def __init__(self, video_source, config_file):
        self.video_source = video_source
        self.analyzer_id = str(uuid.uuid4())[:8]  # Unique ID for this analyzer

        self.latest_frame = None  # ADD THIS LINE
        self.latest_annotated_frame = None  # Store the latest annotated frame

        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize vehicle detection (using background subtraction)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=50, 
            detectShadows=True
        )
        
        # Data storage
        self.traffic_data = {
            "current_vehicles": 0,
            "total_vehicles": 0,
            "vehicle_types": {"car": 0, "motorbike": 0, "bus": 0, "truck": 0},
            "congestion_level": 0,
            "average_speed": 0
        }
        
        # Historical data for trend analysis
        self.historical_data = deque(maxlen=100)
        
        # Video source
        self.cap = cv2.VideoCapture(video_source)
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        # Video writer setup
        self.setup_video_writer()
                
        # Thread for real-time processing
        self.processing = False
        self.process_thread = None
        
        # Vehicle tracking
        self.tracked_vehicles = {}
        self.next_vehicle_id = 1
        self.vehicle_lifetime = 5.0  # seconds to keep a vehicle in memory after it disappears
        
        # Detection zone (invisible - used for processing only)
        self.setup_detection_zones()
        
        # Frame counter for debugging
        self.frame_count = 0
        
        # For speed estimation
        self.vehicle_paths = {}  # Store path history for speed calculation
        
        # Ambulance detection
        self.ambulance_detected = False
        self.ambulance_detection_time = 0

        self.latest_annotated_frame = None  # Add this line

    def get_current_stats(self):
        """Get current traffic statistics for this camera"""
        return {
            'current_vehicles': self.traffic_data.get('current_vehicles', 0),
            'total_vehicles': self.traffic_data.get('total_vehicles', 0),
            'vehicle_types': self.traffic_data.get('vehicle_types', {}),
            'congestion_level': self.traffic_data.get('congestion_level', 0),
            'average_speed': self.traffic_data.get('average_speed', 0),
            'timestamp': self.traffic_data.get('timestamp', '')
        }

    def start(self):
        """Start the video analysis in a separate thread"""
        self.process_thread = threading.Thread(target=self.run, daemon=True)
        self.process_thread.start()
        print(f"Started video analyzer for: {self.video_source}")

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            "congestion_thresholds": {
                "low": 5,
                "medium": 15,
                "high": 25
            },
            "detection_interval": 2,
            "output_video": True,
            "log_data": True,
            "min_contour_area": 800,  # Increased to reduce false positives
            "max_contour_area": 3000,
            "detection_confidence": 0.7,  # Confidence threshold for vehicle detection
            "max_vehicle_speed": 50,  # pixels per second
            "vehicle_type_thresholds": {
                "motorbike": {"min_width": 30, "max_width": 60, "min_height": 30, "max_height": 60},
                "car": {"min_width": 60, "max_width": 120, "min_height": 40, "max_height": 80},
                "bus": {"min_width": 120, "max_width": 200, "min_height": 80, "max_height": 120},
                "truck": {"min_width": 100, "max_width": 180, "min_height": 70, "max_height": 110}
            },
            "aspect_ratio_thresholds": {
                "motorbike": {"min": 0.8, "max": 1.5},
                "car": {"min": 1.5, "max": 2.5},
                "bus": {"min": 2.0, "max": 3.5},
                "truck": {"min": 1.8, "max": 3.0}
            },
            "dashboard": {
                "host": "0.0.0.0",
                "port": 5000
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default settings.")
        
        return default_config
    
    def setup_detection_zones(self):
        """Setup detection zones for traffic analysis (invisible)"""
        # Define a region of interest (ROI) - typically the road area
        self.roi_points = np.array([
            [int(self.width * 0.1), int(self.height * 0.7)],  # Bottom left
            [int(self.width * 0.1), int(self.height * 0.3)],  # Top left
            [int(self.width * 0.9), int(self.height * 0.3)],  # Top right
            [int(self.width * 0.9), int(self.height * 0.7)]   # Bottom right
        ], np.int32)
        
        # Create mask for ROI (used internally, not displayed)
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_points], 255)
        
        # Detection line (vehicles crossing this line are counted) - invisible
        self.detection_line_y = int(self.height * 0.6)
    
    def setup_video_writer(self):
        """Setup video writer for output"""
        if self.config["output_video"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(f"output_annotated_{self.analyzer_id}.mp4", fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None
    
    
    def detect_vehicles(self, frame):
        """Detect vehicles using background subtraction and contour analysis with improved filtering"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI mask (invisible to user)
        gray_masked = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
        
        # Apply background subtraction
        fgmask = self.fgbg.apply(gray_masked)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to smooth the mask
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        
        # Threshold to create binary image
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if not (self.config["min_contour_area"] < area < self.config["max_contour_area"]):
                continue
                
            # Filter by aspect ratio (vehicles are typically wider than tall)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio < 0.8 or aspect_ratio > 4.0:  # Unlikely vehicle aspect ratios
                continue
                
            # Filter by solidity (how convex the shape is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.5:  # Vehicles tend to be fairly solid shapes
                continue
                
            # Check for ambulance characteristics
            roi = frame[y:y+h, x:x+w]
            if self.is_ambulance(roi):
                self.ambulance_detected = True
                self.ambulance_detection_time = time.time()
                
            # Passed all filters - likely a vehicle
            vehicles.append((x, y, w, h))
                
        return vehicles
    
    def is_ambulance(self, vehicle_roi):
        """Simple ambulance detection based on emergency colors only"""
        if vehicle_roi.size == 0:
            return False

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
    
        # Emergency color detection (red and blue)
        red_mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) | \
                   cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    
        blue_mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([140, 255, 255]))
    
        # Calculate color percentages
        total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
        if total_pixels == 0:
            return False

        red_percentage = cv2.countNonZero(red_mask) / total_pixels
        blue_percentage = cv2.countNonZero(blue_mask) / total_pixels

        # Conservative threshold to avoid false positives
        has_emergency_colors = (red_percentage > 0.12) or (blue_percentage > 0.12)

        return has_emergency_colors
        
    def cleanup_old_vehicles(self, current_time):
        """Remove vehicles that haven't been seen for a while"""
        vehicles_to_remove = []
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            if current_time - vehicle_data['last_seen'] > self.vehicle_lifetime:
                vehicles_to_remove.append(vehicle_id)
    
        for vehicle_id in vehicles_to_remove:
            if vehicle_id in self.vehicle_paths:
                del self.vehicle_paths[vehicle_id]
            del self.tracked_vehicles[vehicle_id]
    
    def initialize_tracking(self, current_centroids, current_time):
        """Initialize tracking for new vehicles"""
        for cx, cy, w, h in current_centroids:
            self.add_new_vehicle(cx, cy, w, h, current_time)
        return list(self.tracked_vehicles.keys())

    def add_new_vehicle(self, cx, cy, w, h, current_time):
        """Add a new vehicle to tracking"""
        self.tracked_vehicles[self.next_vehicle_id] = {
            'centroid': (cx, cy),
            'width': w,
            'height': h,
            'last_seen': current_time,
            'first_seen': current_time,
            'counted': False
        }
        # Initialize path tracking with optimized structure
        self.vehicle_paths[self.next_vehicle_id] = {
            'positions': np.zeros((10, 2), dtype=np.float32),
            'timestamps': np.zeros(10, dtype=np.float64),
            'index': 0,
            'count': 0
        }
        self.update_vehicle_path(self.next_vehicle_id, cx, cy, current_time)
        self.next_vehicle_id += 1

    def update_tracked_vehicle(self, vehicle_id, cx, cy, w, h, current_time):
        """Update an existing tracked vehicle"""
        vehicle_data = self.tracked_vehicles[vehicle_id]
        self.tracked_vehicles[vehicle_id] = {
            'centroid': (cx, cy),
            'width': w,
            'height': h,
            'last_seen': current_time,
            'first_seen': vehicle_data['first_seen'],
            'counted': vehicle_data['counted'] or cy > self.detection_line_y
        }
        # Update path tracking
        self.update_vehicle_path(vehicle_id, cx, cy, current_time)

    def update_vehicle_path(self, vehicle_id, x, y, timestamp):
        """Update vehicle path with simpler storage"""
        if vehicle_id not in self.vehicle_paths:
            self.vehicle_paths[vehicle_id] = {
                'positions': np.zeros((10, 2), dtype=np.float32),
                'timestamps': np.zeros(10, dtype=np.float64),
                'index': 0,
                'count': 0
            }
    
        # Add new point
        path = self.vehicle_paths[vehicle_id]
        idx = path['index']
        path['positions'][idx % 10] = [x, y]
        path['timestamps'][idx % 10] = timestamp
        path['index'] = idx + 1
        path['count'] = min(path['count'] + 1, 10)
    
    def track_vehicles(self, current_vehicles, current_time):
        """Optimized vehicle tracking with better matching and lifetime management"""
        # Extract centroids from current vehicles
        current_centroids = []
        for (x, y, w, h) in current_vehicles:
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy, w, h))
    
        # Clean up old vehicles first
        self.cleanup_old_vehicles(current_time)
    
        # If no vehicles currently tracked, initialize tracking
        if not self.tracked_vehicles:
            return self.initialize_tracking(current_centroids, current_time)
    
        # Match current centroids with tracked vehicles using optimized approach
        matched_ids = []
        matched_indices = set()
        tracked_ids = list(self.tracked_vehicles.keys())
    
        # Create a list of old centroids for matching
        old_centroids = []
        for vehicle_id in tracked_ids:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            old_centroids.append(vehicle_data['centroid'] + (vehicle_id,))
    
        # Perform matching
        for i, (old_cx, old_cy, vehicle_id) in enumerate(old_centroids):
            vehicle_data = self.tracked_vehicles[vehicle_id]
            best_match_idx = None
            min_distance = float('inf')
        
            for j, (cx, cy, w, h) in enumerate(current_centroids):
                if j in matched_indices:
                    continue
                
                distance = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                max_distance = self.config["max_vehicle_speed"] * (current_time - vehicle_data['last_seen'])
            
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_match_idx = j
        
            if best_match_idx is not None:
                cx, cy, w, h = current_centroids[best_match_idx]
                self.update_tracked_vehicle(vehicle_id, cx, cy, w, h, current_time)
                matched_ids.append(vehicle_id)
                matched_indices.add(best_match_idx)
    
        # Add new vehicles for unmatched centroids
        for i, (cx, cy, w, h) in enumerate(current_centroids):
            if i not in matched_indices:
                self.add_new_vehicle(cx, cy, w, h, current_time)
                matched_ids.append(self.next_vehicle_id - 1)  # ID of the newly added vehicle
    
        return matched_ids

    def estimate_vehicle_type(self, width, height):
        """Improved vehicle type estimation with better thresholds and aspect ratio"""
        thresholds = self.config["vehicle_type_thresholds"]
        aspect_ratio = width / height if height > 0 else 0
        aspect_thresholds = self.config["aspect_ratio_thresholds"]
        
        # Check if dimensions fit within any category with aspect ratio constraints
        if (thresholds["motorbike"]["min_width"] <= width <= thresholds["motorbike"]["max_width"] and
            thresholds["motorbike"]["min_height"] <= height <= thresholds["motorbike"]["max_height"] and
            aspect_thresholds["motorbike"]["min"] <= aspect_ratio <= aspect_thresholds["motorbike"]["max"]):
            return "motorbike"
        elif (thresholds["car"]["min_width"] <= width <= thresholds["car"]["max_width"] and
              thresholds["car"]["min_height"] <= height <= thresholds["car"]["max_height"] and
              aspect_thresholds["car"]["min"] <= aspect_ratio <= aspect_thresholds["car"]["max"]):
            return "car"
        elif (thresholds["bus"]["min_width"] <= width <= thresholds["bus"]["max_width"] and
              thresholds["bus"]["min_height"] <= height <= thresholds["bus"]["max_height"] and
              aspect_thresholds["bus"]["min"] <= aspect_ratio <= aspect_thresholds["bus"]["max"]):
            return "bus"
        elif (thresholds["truck"]["min_width"] <= width <= thresholds["truck"]["max_width"] and
              thresholds["truck"]["min_height"] <= height <= thresholds["truck"]["max_height"] and
              aspect_thresholds["truck"]["min"] <= aspect_ratio <= aspect_thresholds["truck"]["max"]):
            return "truck"
        else:
            # If no category fits well, use aspect ratio as primary classifier
            if aspect_ratio < 1.2:
                return "motorbike"
            elif 1.2 <= aspect_ratio < 2.0:
                return "car"
            elif 2.0 <= aspect_ratio < 2.8:
                return "truck"
            else:
                return "bus"
    
    def calculate_congestion_level(self, vehicle_count):
        """Calculate congestion level based on vehicle count"""
        thresholds = self.config["congestion_thresholds"]
        if vehicle_count < thresholds["low"]:
            return 0  # Low congestion
        elif vehicle_count < thresholds["medium"]:
            return 1  # Medium congestion
        else:
            return 2  # High congestion
    
    def estimate_vehicle_speed(self, vehicle_id):
        """Estimate vehicle speed based on path history"""
        if vehicle_id not in self.vehicle_paths:
            return 0

        path = self.vehicle_paths[vehicle_id]
        positions = path['positions']
        timestamps = path['timestamps']
        count = path['count']
        index = path['index']

        if count < 2:
            return 0

        # Get points in correct (time) order (oldest to newest)
        # Ring buffer logic
        points = []
        for k in range(count):
            i = (index - count + k) % 10
            x, y = positions[i]
            t = timestamps[i]
            points.append((x, y, t))

        total_distance = 0
        total_time = 0
        for i in range(1, len(points)):
            x1, y1, t1 = points[i-1]
            x2, y2, t2 = points[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_diff = t2 - t1
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff

        if total_time > 0:
            speed = total_distance / total_time  # pixels per second
            speed_kmh = speed * 0.1 * 3.6       # if 1 pixel = 0.1 meters
            return speed_kmh

        return 0
    
    def process_frame(self, frame, frame_id, current_time):
        """Process a single frame for traffic analysis"""
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Track vehicles
        vehicle_ids = self.track_vehicles(vehicles, current_time)
        
        # Count vehicles by type and check if they've crossed the detection line
        vehicle_types = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0, "unknown": 0}
        current_vehicles_count = 0
        total_speed = 0
        speed_count = 0
        
        for vehicle_id in vehicle_ids:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            
            # Only count vehicles that are currently in the frame
            if vehicle_data['last_seen'] == current_time:
                vehicle_type = self.estimate_vehicle_type(vehicle_data['width'], vehicle_data['height'])
                if vehicle_type in vehicle_types:
                    vehicle_types[vehicle_type] += 1
                current_vehicles_count += 1
                
                # Calculate speed for this vehicle
                speed = self.estimate_vehicle_speed(vehicle_id)
                if speed > 0:
                    total_speed += speed
                    speed_count += 1
                
            # Count vehicle in total if it crossed the detection line and hasn't been counted yet
            if (not vehicle_data['counted'] and 
                vehicle_data['centroid'][1] > self.detection_line_y and
                vehicle_data['last_seen'] == current_time):
                vehicle_data['counted'] = True
                self.traffic_data["total_vehicles"] += 1
        
        # Calculate average speed
        avg_speed = round(total_speed / speed_count, 1) if speed_count > 0 else 0
        
        # Update traffic data
        self.traffic_data.update({
            "current_vehicles": current_vehicles_count,
            "vehicle_types": vehicle_types,
            "congestion_level": self.calculate_congestion_level(current_vehicles_count),
            "average_speed": avg_speed,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store historical data for trend analysis
        self.historical_data.append(self.traffic_data.copy())
        
        return vehicles, vehicle_ids
    
    def annotate_frame(self, frame, vehicles, vehicle_ids):
        """Annotate frame with traffic information - WITHOUT the red line and box"""
        annotated_frame = frame.copy()
    
        # Draw bounding boxes around vehicles ONLY (no ROI or detection line)
        for vehicle_id in vehicle_ids:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            cx, cy = vehicle_data['centroid']
            w, h = vehicle_data['width'], vehicle_data['height']
            x, y = cx - w//2, cy - h//2

            # Only draw vehicles detected in the current frame
            if vehicle_data['last_seen'] == time.time():
                # Choose color based on vehicle type
                vehicle_type = self.estimate_vehicle_type(w, h)
                if vehicle_type == "car":
                    color = (0, 255, 0)  # Green for cars
                elif vehicle_type == "motorbike":
                    color = (255, 0, 0)  # Blue for motorbikes
                elif vehicle_type == "bus":
                    color = (0, 165, 255)  # Orange for buses
                elif vehicle_type == "truck":
                    color = (0, 0, 255)  # Red for trucks
                else:
                    color = (255, 255, 0)  # Cyan for unknown

                # Draw rectangle with thicker lines
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)

                # Draw vehicle ID and type with larger font
                label = f"{vehicle_id}:{vehicle_type}"
                cv2.putText(annotated_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Add traffic information overlay with background for readability
        y_offset = 40
        line_height = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Format information for better display
        info_lines = [
            "TRAFFIC MANAGEMENT SYSTEM",
            "",
            "VEHICLES:",
            f"  Current: {self.traffic_data['current_vehicles']}",
            f"  Total: {self.traffic_data['total_vehicles']}",
            f"  Cars: {self.traffic_data['vehicle_types']['car']}  Bikes: {self.traffic_data['vehicle_types']['motorbike']}",
            f"  Buses: {self.traffic_data['vehicle_types']['bus']}  Trucks: {self.traffic_data['vehicle_types']['truck']}",
            "",
            "TRAFFIC CONDITIONS:",
            f"  Congestion: {'LOW' if self.traffic_data['congestion_level'] == 0 else 'MEDIUM' if self.traffic_data['congestion_level'] == 1 else 'HIGH'}",
            f"  Avg Speed: {self.traffic_data['average_speed']} km/h",
            "",
            f"Ambulance Detected: {'YES' if self.ambulance_detected else 'NO'}",
            "",
            f"Camera ID: {self.analyzer_id}",
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        ]

        # Calculate maximum line length for background
        max_line_length = max([len(line) for line in info_lines])
        # Estimate pixel width with more space
        max_width = int(max_line_length * 18) + 60

        # Calculate total text height with some extra padding
        total_text_height = y_offset + line_height * len(info_lines) + 20

        # Draw semi-transparent background for text with rounded corners effect
        overlay = annotated_frame.copy()

        # Draw main background
        cv2.rectangle(overlay, (10, 10), (max_width, total_text_height), (40, 40, 40), -1)

        # Draw accent bar on left
        cv2.rectangle(overlay, (10, 10), (20, total_text_height), (0, 150, 255), -1)

        # Add border
        cv2.rectangle(overlay, (10, 10), (max_width, total_text_height), (0, 150, 255), 2)

        # Apply overlay
        cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)

        # Draw text with improved styling
        for i, line in enumerate(info_lines):
            if i == 0:  # Header line
                color = (0, 255, 255)  # Yellow
                cv2.putText(annotated_frame, line, (30, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness + 1)
            elif "Ambulance Detected: YES" in line:
                color = (0, 0, 255)  # Red for ambulance alert
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness + 1)
            elif line in ["VEHICLES:", "TRAFFIC CONDITIONS:"]:  # Section headers
                color = (0, 200, 255)  # Light orange
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness + 1)
            elif "LOW" in line:
                color = (0, 255, 0)  # Green for low congestion
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "MEDIUM" in line:
                color = (0, 165, 255)  # Orange for medium congestion
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "HIGH" in line:
                color = (0, 0, 255)  # Red for high congestion
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "Camera ID:" in line:
                color = (200, 200, 255)  # Light purple
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif "Last Update:" in line:
                color = (200, 200, 255)  # Light purple
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
            elif line != "":  # Regular text
                color = (200, 255, 200)  # Light green
                cv2.putText(annotated_frame, line, (25, y_offset + i * line_height), 
                           font, font_scale, color, thickness)
                
        self.latest_annotated_frame = annotated_frame  # Store the latest frame
        return annotated_frame

    # In detect_traffic.py, update the run method in SingleVideoAnalyzer class:
    # In detect_traffic.py, update the SingleVideoAnalyzer run method:
    def run(self):
        """Main processing loop"""
        self.processing = True
        frame_id = 0
    
        # Create window for display
        self.window_name = f"TrafficCam - {os.path.basename(str(self.video_source))}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 900, 600)

        try:
            while self.processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Video ended or error reading frame: {self.video_source}")
                    # If it's a file, try to reopen it
                    if isinstance(self.video_source, str):
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.video_source)
                        time.sleep(1)
                        continue
                    else:
                        break
                        
                current_time = time.time()

                # Process frame at the specified interval
                if frame_id % max(1, self.config.get("detection_interval", 1)) == 0:
                    vehicles, vehicle_ids = self.process_frame(frame, frame_id, current_time)
                    annotated_frame = self.annotate_frame(frame, vehicles, vehicle_ids)

                    # Display the annotated frame
                    cv2.imshow(self.window_name, annotated_frame)
                else:
                    # Just display the frame without processing
                    cv2.imshow(self.window_name, frame)

                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                    
                frame_id += 1

        except Exception as e:
            print(f"Error in video processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            cv2.destroyWindow(self.window_name)

    
    def start_processing(self):
        """Start processing video frames"""
        self.processing = True
        frame_id = 0

        # Create a larger window
        window_name = f"Traffic Management System - {self.analyzer_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)

        while self.processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
        
            frame_id += 1
            current_time = time.time()
    
            # Use this for debugging (process every frame):
            if self.config["detection_interval"] > 1 and frame_id % self.config["detection_interval"] != 0:
                continue
    
            # Process frame
            vehicles, vehicle_ids = self.process_frame(frame, frame_id, current_time)
    
            # Annotate frame
            annotated_frame = self.annotate_frame(frame, vehicles, vehicle_ids)

            # Write output video
            if self.out:
                self.out.write(annotated_frame)

            # Log data
            self.log_data(frame_id)

            # Display video in resizable window
            cv2.imshow(window_name, annotated_frame)

            # Allow window to be resized by user
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.processing = False  # Add this line to stop processing
                break
                        
        self.cleanup()
    
    def start(self):
        """Start the traffic analysis system"""
        self.process_thread = threading.Thread(target=self.start_processing)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop(self):
        """Stop the traffic analysis system"""
        self.processing = False
        if self.process_thread:
            self.process_thread.join()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
    

    
    def get_historical_stats(self, limit=100):
        """Get historical traffic statistics"""
        data_list = list(self.historical_data)
        return data_list[-limit:] if data_list else []
    

    

class UrbanTrafficSimulator:
    def __init__(self, road_network, traffic_patterns):
        self.road_network = road_network  # Dictionary of intersections and roads
        self.traffic_patterns = traffic_patterns  # Time-based traffic patterns
        self.vehicles = []
        self.simulation_time = 0
        self.commute_times = []
        self.base_commute_time = 0

    def run(self, stop_event):
        """Professional video processing with proper resource management"""
        self.processing = True
        frame_id = 0
        
        window_name = f"TrafficCam - {os.path.basename(str(self.video_source))}"
        
        try:
            while self.processing and not stop_event.is_set() and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                vehicles, vehicle_ids = self.process_frame(frame, frame_id, time.time())
                annotated_frame = self.annotate_frame(frame, vehicles, vehicle_ids)
                self.latest_annotated_frame = annotated_frame
                
                # Display
                cv2.imshow(window_name, annotated_frame)
                
                # Professional key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or stop_event.is_set():
                    break
                    
                frame_id += 1
                
        except Exception as e:
            print(f"Video processing error: {e}")
        finally:
            self.cleanup()
            cv2.destroyWindow(window_name)

    def run_simulation(self, duration_hours=24):
        """Run the simulation with real data exchange with traffic analyzer"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        current_time = start_time
    
        print("Starting simulation with live traffic data integration...")
    
        while current_time < end_time:
            # Get real traffic data from analyzer
            live_data = self.get_live_traffic_data()
        
            if live_data:
                # Adjust simulation based on real traffic conditions
                self.adjust_simulation_parameters(live_data)
        
            # Generate vehicles based on current conditions
            density_multiplier = 1.0
            if live_data and 'congestion_level' in live_data:
                # Increase vehicle generation during high congestion
                density_multiplier = 1.0 + (live_data['congestion_level'] * 0.3)
        
            self.generate_vehicles(current_time, density_multiplier)
        
            # Get signal states (either from analyzer or use simulation logic)
            if live_data and 'signal_states' in live_data:
                signal_states = live_data['signal_states']
            else:
                signal_states = self.generate_signal_states()
        
            # Update vehicle positions
            self.update_vehicle_positions(signal_states, 1)
        
            # Send simulation results back to analyzer periodically
            if int(current_time.timestamp()) % 30 == 0:  # Every 30 seconds
                simulation_results = {
                    'recommended_signal_timings': self.optimize_signal_timings(),
                    'congestion_predictions': self.predict_future_congestion(),
                    'simulation_time': current_time
                }
                self.send_to_analyzer(simulation_results)
        
            # Update simulation time
            self.simulation_time += 1
            current_time += timedelta(seconds=1)
        
            # Progress reporting
            if self.simulation_time % 3600 == 0:
                hour = current_time.hour
                reduction = self.calculate_commute_reduction()
                print(f"Simulation hour {hour}: {reduction:.2f}% reduction in commute time")
    
        final_reduction = self.calculate_commute_reduction()
        print(f"Final simulation result: {final_reduction:.2f}% reduction in average commute time")
    
        return final_reduction

    def optimize_signal_timings(self):
        """Calculate optimized signal timings based on simulation"""
        # Simple optimization based on simulated traffic
        base_time = 30
        optimized_timings = []
    
        # This would be more sophisticated in a real implementation
        for i in range(16):  # For all 16 signals
            optimized_timings.append(base_time + random.randint(-10, 10))
    
        return optimized_timings

    def predict_future_congestion(self):
        """Predict future congestion patterns based on simulation"""
        # Simple prediction logic
        predictions = {}
        for i in range(4):  # For next 4 hours
            future_time = (datetime.now().hour + i) % 24
            # Simple prediction based on time of day patterns
            congestion = self.traffic_patterns.get(future_time, 1.0) * 0.8
            predictions[future_time] = min(2, int(congestion))  # Scale to 0-2 congestion level
    
        return predictions
    
    def generate_vehicles(self, time_of_day, density_multiplier=1.0):
        """Generate vehicles based on time of day and traffic patterns"""
        hour = time_of_day.hour
        pattern = self.traffic_patterns.get(hour, 1.0)
        
        num_vehicles = int(pattern * density_multiplier * 10)  # Reduced for performance
        
        for _ in range(num_vehicles):
            origin = random.choice(list(self.road_network['intersections'].keys()))
            destination = random.choice([
                node for node in self.road_network['intersections'].keys() 
                if node != origin
            ])
            
            vehicle = {
                'id': len(self.vehicles) + 1,
                'origin': origin,
                'destination': destination,
                'start_time': time_of_day,
                'current_position': origin,
                'path': self.calculate_path(origin, destination),
                'path_index': 0,
                'status': 'waiting',  # waiting, moving, arrived
                'wait_time': 0,
                'travel_time': 0
            }
            self.vehicles.append(vehicle)
    
    def calculate_path(self, origin, destination):
        """Calculate the shortest path using Dijkstra's algorithm"""
        # Simplified pathfinding implementation
        intersections = self.road_network['intersections']
        roads = self.road_network['roads']
        
        # This would be replaced with a proper pathfinding algorithm
        # For simplicity, we'll return a random path
        path = [origin]
        current = origin
        
        while current != destination:
            connections = roads.get(current, {})
            if not connections:
                break
            current = random.choice(list(connections.keys()))
            path.append(current)
            
        return path
    
    def update_vehicle_positions(self, signal_states, time_step):
        """Update vehicle positions based on signal states"""
        for vehicle in self.vehicles:
            if vehicle['status'] == 'arrived':
                continue
                
            current_node = vehicle['current_position']
            path = vehicle['path']
            current_index = vehicle['path_index']
            
            if current_index >= len(path) - 1:
                vehicle['status'] = 'arrived'
                vehicle['travel_time'] = self.simulation_time
                self.commute_times.append(vehicle['travel_time'])
                continue
                
            next_node = path[current_index + 1]
            
            # Check if the signal allows movement in this direction
            signal_key = f"{current_node}_{next_node}"
            if signal_states.get(signal_key, False):  # Green light
                vehicle['current_position'] = next_node
                vehicle['path_index'] += 1
                vehicle['status'] = 'moving'
                vehicle['wait_time'] = 0
            else:  # Red light
                vehicle['status'] = 'waiting'
                vehicle['wait_time'] += time_step
                
    def calculate_commute_reduction(self):
        """Calculate the percentage reduction in average commute time"""
        if not self.commute_times or self.base_commute_time == 0:
            return 0
            
        current_avg = sum(self.commute_times) / len(self.commute_times)
        reduction = ((self.base_commute_time - current_avg) / self.base_commute_time) * 100
        return reduction
    
    def run_baseline(self, start_time, duration_hours):
        """Run simulation with fixed signal timing to establish baseline"""
        baseline_times = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        sim_time = 0
        
        # Fixed signal timing (30 seconds each direction)
        fixed_signals = {}
        signal_cycle = 120  # 2-minute cycle for 4 directions
        
        while current_time < end_time:
            # Generate vehicles
            self.generate_vehicles(current_time)
            
            # Fixed signal pattern
            signal_phase = (sim_time % signal_cycle) // 30
            signal_states = {
                f"intersection_{i}_direction_{j}": (j == signal_phase) 
                for i in range(4) for j in range(4)
            }
            
            # Update vehicle positions
            self.update_vehicle_positions(signal_states, 1)
            
            # Update time
            sim_time += 1
            current_time += timedelta(seconds=1)
            
            # Collect commute times for arrived vehicles
            for vehicle in self.vehicles:
                if vehicle['status'] == 'arrived' and vehicle['travel_time'] > 0:
                    baseline_times.append(vehicle['travel_time'])
                    vehicle['travel_time'] = 0  # Mark as recorded
        
        return sum(baseline_times) / len(baseline_times) if baseline_times else 0

    def connect_to_analyzer(self, analyzer):
        """Connect to the traffic analyzer for bidirectional data exchange"""
        self.traffic_analyzer = analyzer
        print("Traffic analyzer connected to simulator")

    def get_live_traffic_data(self):
        """Get real-time traffic data from the analyzer"""
        if self.traffic_analyzer:
            return self.traffic_analyzer.get_traffic_data_for_simulation()
        return None

    def send_to_analyzer(self, data):
        """Send simulation results back to the analyzer"""
        if self.traffic_analyzer:
            self.traffic_analyzer.update_from_simulation(data)

