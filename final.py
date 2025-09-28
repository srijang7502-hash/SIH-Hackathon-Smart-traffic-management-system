#!/usr/bin/env python3
"""
final_junction_sim_with_heatmap.py

Simulates a 4-way traffic junction with advanced features and real-time heatmap generation.
- Implements YOLOv8 for vehicle detection.
- Features a dynamic 2x2 live-footage dashboard.
- Uses a dynamic multiplier to estimate traffic queue length.
- Generates and updates a Folium traffic heatmap after each light cycle.
"""
import cv2
import numpy as np
import time
import folium
from folium.plugins import HeatMap
import random
from ultralytics import YOLO

# ------------------- HELPER FUNCTIONS -------------------

def estimate_direction(cx, lane_width):
    """Divide road width into 3 sections: left / straight / right."""
    if cx < lane_width:
        return "left"
    elif cx < 2 * lane_width:
        return "straight"
    else:
        return "right"

def calculate_dynamic_multiplier(visible_vehicles):
    """Calculates a multiplier to estimate the real queue length."""
    if visible_vehicles < 5:
        return 1.2
    elif visible_vehicles < 15:
        return 1.8
    else:
        return 2.5

def get_live_frames(captures):
    """Reads a frame from each capture, looping the video if it has ended."""
    frames = {}
    for side, cap in captures.items():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frames[side] = frame
    return frames

# ------------------- HEATMAP GENERATION -------------------

def generate_traffic_heatmap(vehicle_counts, junction_coords):
    """
    Generates and saves a Folium heatmap based on current vehicle counts.
    """
    # Bounding box for synthetic traffic
    bbox = (17.965, 17.975, 79.590, 79.600)
    lat_min, lat_max, lon_min, lon_max = bbox
    
    heat_points = []
    
    # 1. Add real points from our junction counts
    for side, count in vehicle_counts.items():
        coords = junction_coords[side]
        for _ in range(count):
            # Add jitter to spread points around the junction
            lat_jitter = coords[0] + random.uniform(-0.0001, 0.0001)
            lon_jitter = coords[1] + random.uniform(-0.0001, 0.0001)
            heat_points.append([lat_jitter, lon_jitter, 1.0]) # Weight of 1 for real cars

    # 2. Add synthetic points for surrounding area to make the map look busy
    for _ in range(150): # Add 150 random background traffic points
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        weight = random.uniform(0.2, 0.6) # Lower weight for synthetic traffic
        heat_points.append([lat, lon, weight])

    # 3. Create the map
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles="CartoDB positron")

    # 4. Add the heatmap layer
    HeatMap(heat_points, radius=20, blur=15).add_to(m)

    # 5. Save to file
    output_file = "traffic_heatmap.html"
    m.save(output_file)
    print(f"----- HEATMAP UPDATED: Refreshe '{output_file}' in your browser. -----")

# ------------------- CORE LOGIC & VISUALIZATION -------------------
# ... (All other classes and functions like TrafficLightLogic, get_vehicle_counts_from_snapshot, create_dynamic_dashboard remain the same) ...
class TrafficLightLogic:
    def __init__(self, min_green=8, yellow=3, time_per_vehicle=1.5, max_green=90):
        self.min_green, self.yellow, self.time_per_vehicle, self.max_green = min_green, yellow, time_per_vehicle, max_green
    def calculate_green_duration(self, lane_counts):
        total_visible_vehicles = sum(lane_counts.values())
        multiplier = calculate_dynamic_multiplier(total_visible_vehicles)
        estimated_queue = int(total_visible_vehicles * multiplier)
        print(f"INFO: Visible vehicles: {total_visible_vehicles}, Using multiplier: {multiplier}, Estimated queue: {estimated_queue}")
        estimated_green_time = self.min_green + (estimated_queue * self.time_per_vehicle)
        return max(self.min_green, min(estimated_green_time, self.max_green))

def get_vehicle_counts_from_snapshot(image, model):
    if image is None: return {"left": 0, "straight": 0, "right": 0}
    frame_h, frame_w, _ = image.shape
    lane_width = frame_w // 3
    lane_counts = {"left": 0, "straight": 0, "right": 0}
    vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
    results = model(image, verbose=False, conf=0.25)
    if results:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                class_id = int(box.cls.cpu().item())
                if model.names.get(class_id, "") in vehicle_classes:
                    x1, _, x2, _ = box.xyxy.cpu().numpy().flatten()
                    lane_counts[estimate_direction((x1 + x2) / 2, lane_width)] += 1
    return lane_counts

def create_dynamic_dashboard(frames, light_states, active_side_name, timers):
    dash_h, dash_w = 720, 1280
    dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
    quad_h, quad_w = dash_h // 2, dash_w // 2
    light_colors = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}
    positions = {"North": (0, 0), "East": (0, quad_w), "South": (quad_h, 0), "West": (quad_h, quad_w)}
    for side, frame in frames.items():
        if frame is None: continue
        resized_frame = cv2.resize(frame, (quad_w, quad_h))
        y, x = positions[side]
        dashboard[y:y+quad_h, x:x+quad_w] = resized_frame
        border_color = light_colors[light_states[side]]
        border_thickness = 10 if side == active_side_name else 4
        cv2.rectangle(dashboard, (x, y), (x+quad_w, y+quad_h), border_color, border_thickness)
        text = f"{side}: {light_states[side]}"
        cv2.putText(dashboard, text, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
        cv2.putText(dashboard, text, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    timer_text = f"Active: {active_side_name} | State: {timers['state']} for {timers['duration']:.1f}s"
    cv2.putText(dashboard, timer_text, (20, dash_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
    cv2.putText(dashboard, timer_text, (20, dash_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return dashboard

# ------------------- MAIN SIMULATION -------------------

def run_4_way_junction_simulation():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8s.pt")
    print("Model loaded.")
    
    logic_controller = TrafficLightLogic(max_green=90)

    # --- COORDINATES FOR THE 4-WAY JUNCTION ---
    # Centered around Hanamkonda Bus Station area for a realistic location
    junction_center = (17.9700, 79.5950)
    offset = 0.0008
    junction_coords = {
        "North": (junction_center[0] + offset, junction_center[1]),
        "South": (junction_center[0] - offset, junction_center[1]),
        "East":  (junction_center[0], junction_center[1] + offset),
        "West":  (junction_center[0], junction_center[1] - offset),
    }

    side_order = ["North", "East", "South", "West"]
    junction_sides = {side: {"video_path": f"videos/traffic_jam_{i+1}.mp4"} for i, side in enumerate(side_order)}
    
    captures = {}
    for side, data in junction_sides.items():
        cap = cv2.VideoCapture(data['video_path'])
        if not cap.isOpened():
            print(f"ERROR: Cannot open video for {side}. Exiting.")
            return
        captures[side] = cap
        
    latest_vehicle_counts = {side: 0 for side in side_order}
    
    # --- Initial Scan & First Heatmap ---
    print("Performing initial scan for the first heatmap...")
    initial_frames = get_live_frames(captures)
    for side in side_order:
        counts = get_vehicle_counts_from_snapshot(initial_frames[side], model)
        latest_vehicle_counts[side] = sum(counts.values())
    generate_traffic_heatmap(latest_vehicle_counts, junction_coords)
    
    current_side_idx = 0
    print("\n--- Starting 4-Way Junction Simulation ---")
    print("Press 'q' on the dashboard window to quit.")

    try:
        while True:
            current_side_name = side_order[current_side_idx]
            all_frames = get_live_frames(captures)
            snapshot_frame = all_frames[current_side_name]
            
            print(f"\n----- Evaluating: {current_side_name} Side -----")
            lane_counts = get_vehicle_counts_from_snapshot(snapshot_frame, model)
            latest_vehicle_counts[current_side_name] = sum(lane_counts.values()) # Update count
            
            green_duration = logic_controller.calculate_green_duration(lane_counts)
            yellow_duration = logic_controller.yellow
            
            light_states = {side: "RED" for side in side_order}
            
            # --- GREEN PHASE ---
            light_states[current_side_name] = "GREEN"
            print(f"ACTION: {current_side_name} signal is GREEN for {green_duration:.1f}s.")
            start_time = time.time()
            while time.time() - start_time < green_duration:
                timers = {"state": "GREEN", "duration": green_duration - (time.time() - start_time)}
                live_frames = get_live_frames(captures)
                dashboard = create_dynamic_dashboard(live_frames, light_states, current_side_name, timers)
                cv2.imshow("Junction Dashboard", dashboard)
                if cv2.waitKey(1) & 0xFF == ord('q'): raise SystemExit

            # --- YELLOW PHASE ---
            light_states[current_side_name] = "YELLOW"
            print(f"ACTION: {current_side_name} signal is YELLOW for {yellow_duration:.1f}s.")
            start_time = time.time()
            while time.time() - start_time < yellow_duration:
                timers = {"state": "YELLOW", "duration": yellow_duration - (time.time() - start_time)}
                live_frames = get_live_frames(captures)
                dashboard = create_dynamic_dashboard(live_frames, light_states, current_side_name, timers)
                cv2.imshow("Junction Dashboard", dashboard)
                if cv2.waitKey(1) & 0xFF == ord('q'): raise SystemExit

            # --- UPDATE HEATMAP at the end of the cycle ---
            generate_traffic_heatmap(latest_vehicle_counts, junction_coords)
            
            current_side_idx = (current_side_idx + 1) % len(side_order)
            
    except (SystemExit, KeyboardInterrupt):
        print("\nSimulation stopped by user.")
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    run_4_way_junction_simulation()