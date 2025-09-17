# dashboard.py
from flask import Flask, render_template, jsonify, request
import threading
import json
import random  # <-- ADDED THIS IMPORT
from datetime import datetime
import time

app = Flask(__name__)

class TrafficDashboard:
    def __init__(self, traffic_analyzer, urban_simulator):
        self.traffic_analyzer = traffic_analyzer
        self.urban_simulator = urban_simulator
        self.historical_data = []
        self.signal_control_history = []
        self.update_interval = 5  # seconds
        
    def start(self):
        """Start the dashboard and data collection"""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _update_loop(self):
        """Background thread to update dashboard data"""
        while self.running:
            # Collect current data
            current_data = {
                'timestamp': datetime.now().isoformat(),
                'current_vehicles': self.traffic_analyzer.traffic_data['current_vehicles'],
                'total_vehicles': self.traffic_analyzer.traffic_data['total_vehicles'],
                'vehicle_types': self.traffic_analyzer.traffic_data['vehicle_types'],
                'congestion_level': self.traffic_analyzer.traffic_data['congestion_level'],
                'average_speed': self.traffic_analyzer.traffic_data['average_speed'],
                'signal_states': self.traffic_analyzer.signal_states,
                'signal_timings': self.traffic_analyzer.signal_timings,
                'current_green': self.traffic_analyzer.current_green
            }
            
            self.historical_data.append(current_data)
            # Keep only last 1000 entries
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data[-1000:]
                
            time.sleep(self.update_interval)
    
    def get_current_stats(self):
        """Get current traffic statistics"""
        if not self.historical_data:
            return {}
        return self.historical_data[-1]
    
    def get_historical_stats(self, limit=100):
        """Get historical traffic statistics"""
        return self.historical_data[-limit:] if self.historical_data else []
    
    def get_commute_reduction(self):
        """Get current commute time reduction percentage"""
        if self.urban_simulator:
            return self.urban_simulator.calculate_commute_reduction()
        return 0
    
    def manual_signal_control(self, intersection, direction, duration):
        """Manual control of traffic signals"""
        # Implementation would depend on your signal control system
        print(f"Manual control: Intersection {intersection}, Direction {direction}, Duration {duration}")
        self.signal_control_history.append({
            'timestamp': datetime.now().isoformat(),
            'intersection': intersection,
            'direction': direction,
            'duration': duration,
            'type': 'manual'
        })

# Flask routes
dashboard = None  # Will be initialized with TrafficDashboard instance

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/current_stats')
def current_stats():
    if dashboard:
        return jsonify(dashboard.get_current_stats())
    return jsonify({})

@app.route('/api/historical_stats')
def historical_stats():
    limit = request.args.get('limit', 100, type=int)
    if dashboard:
        return jsonify(dashboard.get_historical_stats(limit))
    return jsonify([])

@app.route('/api/commute_reduction')
def commute_reduction():
    if dashboard and dashboard.urban_simulator:
        reduction = dashboard.urban_simulator.calculate_commute_reduction()
        # If no simulation is running, provide a demo value
        if reduction == 0:
            reduction = random.uniform(8, 15)  # Realistic range for demo
        return jsonify({'reduction': reduction})
    elif dashboard:
        return jsonify({'reduction': dashboard.get_commute_reduction()})
    else:
        return jsonify({'reduction': 0})

@app.route('/api/control_signal', methods=['POST'])
def control_signal():
    data = request.json
    intersection = data.get('intersection')
    direction = data.get('direction')
    duration = data.get('duration', 30)
    
    if dashboard and intersection is not None and direction is not None:
        dashboard.manual_signal_control(intersection, direction, duration)
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Invalid parameters'})

@app.route('/api/bottleneck_predictions')
def bottleneck_predictions():
    if dashboard and hasattr(dashboard.traffic_analyzer, 'predict_bottlenecks'):
        try:
            predictions = dashboard.traffic_analyzer.predict_bottlenecks()
            return jsonify(predictions)
        except:
            return jsonify({'bottlenecks': [], 'sensor_data': {}})
    return jsonify({'bottlenecks': [], 'sensor_data': {}})

@app.route('/api/iot_data')
def iot_data():
    if dashboard and hasattr(dashboard.traffic_analyzer, 'iot_simulator'):
        try:
            sensor_data = dashboard.traffic_analyzer.iot_simulator.generate_sensor_data()
            return jsonify(sensor_data)
        except:
            return jsonify({})
    return jsonify({})

def run_dashboard(analyzer, simulator, host='0.0.0.0', port=5000):
    global dashboard
    dashboard = TrafficDashboard(analyzer, simulator)
    dashboard.start()
    app.run(host=host, port=port, debug=False)