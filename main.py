# main.py
from detect_traffic import TrafficAnalyzer
from urban_simulator import UrbanTrafficSimulator
from dashboard import run_dashboard
import threading
import os

def main():
    # Initialize traffic analyzer
    video_source = "traffic_sample.mp4"
    if not os.path.exists(video_source):
        print(f"Warning: {video_source} not found. Using built-in camera instead.")
        video_source = 0  # Use webcam
    
    analyzer = TrafficAnalyzer(video_source=video_source)
    
    # Define a simple road network
    road_network = {
        'intersections': {
            'intersection_0': {'x': 100, 'y': 100},
            'intersection_1': {'x': 300, 'y': 100},
            'intersection_2': {'x': 300, 'y': 300},
            'intersection_3': {'x': 100, 'y': 300}
        },
        'roads': {
            'intersection_0': {'intersection_1': 200, 'intersection_3': 200},
            'intersection_1': {'intersection_0': 200, 'intersection_2': 200},
            'intersection_2': {'intersection_1': 200, 'intersection_3': 200},
            'intersection_3': {'intersection_2': 200, 'intersection_0': 200}
        }
    }
    
    # Define traffic patterns by hour (0-23)
    traffic_patterns = {
        0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.3,
        6: 0.5, 7: 0.8, 8: 1.0, 9: 0.9, 10: 0.7, 11: 0.8,
        12: 0.9, 13: 0.8, 14: 0.7, 15: 0.8, 16: 1.0, 17: 1.2,
        18: 1.0, 19: 0.8, 20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3
    }
    
    # Initialize urban simulator
    simulator = UrbanTrafficSimulator(road_network, traffic_patterns)
    
    # Connect analyzer to simulator with bidirectional data exchange
    analyzer.connect_to_simulator(simulator)
    simulator.connect_to_analyzer(analyzer)  # Add bidirectional connection
    
    # Start traffic analysis in a separate thread
    analysis_thread = threading.Thread(target=analyzer.start)
    analysis_thread.daemon = True
    analysis_thread.start()

    # Start the simulation in a separate thread
    simulation_thread = threading.Thread(target=simulator.run_simulation, args=(6,))  # 6-hour simulation
    simulation_thread.daemon = True
    simulation_thread.start()

    # Start the dashboard
    print("Starting dashboard on http://localhost:5000")
    print("Starting 6-hour traffic simulation with live data exchange...")
    run_dashboard(analyzer, simulator)

if __name__ == "__main__":
    main()