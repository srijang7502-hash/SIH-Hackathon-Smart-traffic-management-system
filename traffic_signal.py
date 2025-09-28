#!/usr/bin/env python3
"""
final_junction_sim.py

Simulates a 4-way traffic junction with advanced features.
- Implements YOLOv8 for vehicle detection.
- Features a dynamic 2x2 live-footage dashboard showing all four approaches.
- Uses a dynamic multiplier to better estimate traffic queue length.
- Includes corrected video looping to prevent black screens.
"""
import cv2
import numpy as np
import time
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
    """
    Calculates a multiplier to estimate the real queue length.
    The more cars visible, the higher the chance of heavy, occluded traffic.
    """
    if visible_vehicles < 5:
        return 1.2  # Small queue, less likely to be occluded
    elif visible_vehicles < 15:
        return 1.8  # Medium queue, moderate chance of occlusion
    else:
        return 2.5  # Heavy traffic, assume significant occlusion

def get_live_frames(captures):
    """
    Reads a frame from each capture object, looping the video if it has ended.
    This function centralizes the looping logic to prevent black screens.
    """
    frames = {}
    for side, cap in captures.items():
        ret, frame = cap.read()
        if not ret:
            # If the video ends, reset it to the beginning (loop)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frames[side] = frame
    return frames
    
# ------------------- CORE LOGIC -------------------

class TrafficLightLogic:
    """
    Calculates green light duration using a dynamic multiplier for queue estimation.
    """
    def __init__(self, min_green=20, yellow=3, time_per_vehicle=1.5, max_green=120):
        self.min_green = min_green
        self.yellow = yellow
        self.time_per_vehicle = time_per_vehicle
        self.max_green = max_green

    def calculate_green_duration(self, lane_counts):
        """Calculates green duration based on the dynamically estimated queue."""
        total_visible_vehicles = sum(lane_counts.values())
        
        # Use the dynamic multiplier for a smarter estimate
        multiplier = calculate_dynamic_multiplier(total_visible_vehicles)
        estimated_queue = int(total_visible_vehicles * multiplier)
        
        print(f"INFO: Visible vehicles: {total_visible_vehicles}, "
              f"Using multiplier: {multiplier}, "
              f"Estimated queue: {estimated_queue}")
        
        estimated_green_time = self.min_green + (estimated_queue * self.time_per_vehicle)
        green_duration = max(self.min_green, min(estimated_green_time, self.max_green))
        
        return green_duration

def get_vehicle_counts_from_snapshot(image, model):
    """Takes an image, runs YOLO, and returns vehicle counts per lane."""
    if image is None:
        return {"left": 0, "straight": 0, "right": 0}
        
    frame_h, frame_w, _ = image.shape
    lane_width = frame_w // 3
    lane_counts = {"left": 0, "straight": 0, "right": 0}
    
    vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
    model_names = model.names

    results = model(image, verbose=False, conf=0.25)
    
    if results:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                class_id = int(box.cls.cpu().item())
                class_name = model_names.get(class_id, "")
                
                if class_name in vehicle_classes:
                    x1, _, x2, _ = box.xyxy.cpu().numpy().flatten()
                    cx = (x1 + x2) / 2
                    direction = estimate_direction(cx, lane_width)
                    lane_counts[direction] += 1

    return lane_counts

# ------------------- VISUALIZATION -------------------

def create_dynamic_dashboard(frames, light_states, active_side_name, timers):
    """Creates a 2x2 grid dashboard with live feeds and status."""
    dash_h, dash_w = 720, 1280
    dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
    
    quad_h, quad_w = dash_h // 2, dash_w // 2
    
    light_colors = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}

    positions = {
        "North": (0, 0), "East": (0, quad_w),
        "South": (quad_h, 0), "West": (quad_h, quad_w)
    }

    for side, frame in frames.items():
        if frame is None: continue
        
        resized_frame = cv2.resize(frame, (quad_w, quad_h))
        y, x = positions[side]
        dashboard[y:y+quad_h, x:x+quad_w] = resized_frame

        border_color = light_colors[light_states[side]]
        border_thickness = 10 if side == active_side_name else 4
        cv2.rectangle(dashboard, (x, y), (x+quad_w, y+quad_h), border_color, border_thickness)
        
        text = f"{side}: {light_states[side]}"
        cv2.putText(dashboard, text, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6) # Black outline
        cv2.putText(dashboard, text, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2) # White text

    timer_text = f"Active: {active_side_name} | State: {timers['state']} for {timers['duration']:.1f}s"
    cv2.putText(dashboard, timer_text, (20, dash_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
    cv2.putText(dashboard, timer_text, (20, dash_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    return dashboard

# ------------------- MAIN SIMULATION -------------------

def run_4_way_junction_simulation():
    """Main function to run the traffic junction simulation with corrected video looping."""
    
    print("Loading YOLOv8 model... (Using yolov8s.pt for speed)")
    model = YOLO("yolov8s.pt")
    print("Model loaded.")
    
    logic_controller = TrafficLightLogic()

    side_order = ["North", "East", "South", "West"]
    junction_sides = {
        "North": {"video_path": "videos/traffic_jam_1.mp4"},
        "East":  {"video_path": "videos/traffic_jam_2.mp4"},
        "South": {"video_path": "videos/traffic_jam_3.mp4"},
        "West":  {"video_path": "videos/traffic_jam_4.mp4"},
    }
    
    captures = {}
    for side, data in junction_sides.items():
        cap = cv2.VideoCapture(data['video_path'])
        if not cap.isOpened():
            print(f"ERROR: Cannot open video for {side} at {data['video_path']}. Exiting.")
            return
        captures[side] = cap
        
    current_side_idx = 0
    print("\n--- Starting 4-Way Junction Simulation ---")
    print("Press 'q' on the dashboard window to quit.")

    try:
        while True:
            current_side_name = side_order[current_side_idx]
            
            # Use the helper function to get frames, ensuring videos are looped
            all_frames = get_live_frames(captures)
            snapshot_frame = all_frames[current_side_name]
            
            print(f"\n----- Evaluating: {current_side_name} Side -----")
            
            lane_counts = get_vehicle_counts_from_snapshot(snapshot_frame, model)
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