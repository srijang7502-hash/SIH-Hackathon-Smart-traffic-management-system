#!/usr/bin/env python3
"""
main.py - Headless FastAPI Backend for Smart Traffic Simulation (Corrected Looping)

This script provides a backend-only API for the 4-way junction simulation.
It serves a video stream, a heatmap, and a JSON status endpoint for a custom frontend.
This version includes the corrected video looping logic to prevent black screens.
"""
import cv2
import numpy as np
import time
import folium
from folium.plugins import HeatMap
import random
import asyncio
from ultralytics import YOLO
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiofiles

# --- Global State & Configuration ---

class SimulationState:
    def __init__(self):
        self.is_running = False
        self.dashboard_frame = None
        self.light_states = {}
        self.timers = {}
        self.latest_vehicle_counts = {}
        self.active_side_name = ""
        self.lock = asyncio.Lock()

state = SimulationState()

# --- Helper Functions & Core Logic ---

def estimate_direction(cx, lane_width):
    if cx < lane_width: return "left"
    elif cx < 2 * lane_width: return "straight"
    else: return "right"

def calculate_dynamic_multiplier(visible_vehicles):
    if visible_vehicles < 5: return 1.2
    elif visible_vehicles < 15: return 1.8
    else: return 2.5

# ADDED: Centralized function to read and loop video frames correctly.
def get_live_frames(captures):
    """
    Reads a frame from each capture object, looping the video if it has ended.
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

class TrafficLightLogic:
    def __init__(self, min_green=8, yellow=3, time_per_vehicle=1.5, max_green=90):
        self.min_green, self.yellow, self.time_per_vehicle, self.max_green = min_green, yellow, time_per_vehicle, max_green

    def calculate_green_duration(self, lane_counts):
        total_visible_vehicles = sum(lane_counts.values())
        multiplier = calculate_dynamic_multiplier(total_visible_vehicles)
        estimated_queue = int(total_visible_vehicles * multiplier)
        print(f"INFO: Visible: {total_visible_vehicles}, Multiplier: {multiplier}, Estimated Queue: {estimated_queue}")
        estimated_green_time = self.min_green + (estimated_queue * self.time_per_vehicle)
        return max(self.min_green, min(estimated_green_time, self.max_green))

def analyze_snapshot(image, model):
    if image is None: return {"counts": {"left": 0, "straight": 0, "right": 0}, "detections": []}
    frame_h, frame_w, _ = image.shape
    lane_width = frame_w // 3
    lane_counts = {"left": 0, "straight": 0, "right": 0}
    detections = []
    vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
    results = model(image, verbose=False, conf=0.35)
    if results:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                class_id = int(box.cls.cpu().item())
                class_name = model.names.get(class_id, "")
                if class_name in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                    detections.append((x1, y1, x2, y2, class_name))
                    cx = (x1 + x2) / 2
                    lane_counts[estimate_direction(cx, lane_width)] += 1
    return {"counts": lane_counts, "detections": detections}

async def generate_traffic_heatmap(vehicle_counts, junction_coords):
    bbox = (17.965, 17.975, 79.590, 79.600)
    lat_min, lat_max, lon_min, lon_max = bbox
    heat_points = []
    for side, count in vehicle_counts.items():
        coords = junction_coords[side]
        for _ in range(count):
            lat_jitter = coords[0] + random.uniform(-0.0001, 0.0001)
            lon_jitter = coords[1] + random.uniform(-0.0001, 0.0001)
            heat_points.append([lat_jitter, lon_jitter, 1.0])
    for _ in range(150):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        heat_points.append([lat, lon, random.uniform(0.2, 0.6)])
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles="CartoDB positron")
    HeatMap(heat_points, radius=20, blur=15).add_to(m)
    output_file = "traffic_heatmap.html"
    m.save(output_file)
    print(f"----- HEATMAP UPDATED: '{output_file}' -----")

def create_dynamic_dashboard(frames, light_states, active_side_name, timers, detections_for_green_lane=None):
    dash_h, dash_w = 720, 1280
    dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
    quad_h, quad_w = dash_h // 2, dash_w // 2
    light_colors = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}
    positions = {"North": (0, 0), "East": (0, quad_w), "South": (quad_h, 0), "West": (quad_h, quad_w)}
    for side, frame in frames.items():
        if frame is None: continue
        if side == active_side_name and light_states.get(side) == "GREEN" and detections_for_green_lane:
            for x1, y1, x2, y2, class_name in detections_for_green_lane:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        resized_frame = cv2.resize(frame, (quad_w, quad_h))
        y, x = positions[side]
        dashboard[y:y+quad_h, x:x+quad_w] = resized_frame
        border_color = light_colors.get(light_states.get(side, "RED"), (128,128,128))
        border_thickness = 10 if side == active_side_name else 4
        cv2.rectangle(dashboard, (x, y), (x+quad_w, y+quad_h), border_color, border_thickness)
        text = f"{side}: {light_states.get(side, 'N/A')}"
        cv2.putText(dashboard, text, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
        cv2.putText(dashboard, text, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    timer_text = f"Active: {active_side_name} | State: {timers.get('state', 'N/A')} for {timers.get('duration', 0):.1f}s"
    cv2.putText(dashboard, timer_text, (20, dash_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
    cv2.putText(dashboard, timer_text, (20, dash_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return dashboard

# --- Background Simulation Task ---

async def simulation_loop():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8s.pt")
    print("Model loaded.")
    logic_controller = TrafficLightLogic(max_green=90)
    junction_center = (17.9700, 79.5950)
    offset = 0.0008
    junction_coords = {
        "North": (junction_center[0] + offset, junction_center[1]), "South": (junction_center[0] - offset, junction_center[1]),
        "East":  (junction_center[0], junction_center[1] + offset), "West":  (junction_center[0], junction_center[1] - offset),
    }
    side_order = ["North", "East", "South", "West"]
    junction_sides = {side: {"video_path": f"videos/traffic_jam_{i+1}.mp4"} for i, side in enumerate(side_order)}
    captures = {side: cv2.VideoCapture(data['video_path']) for side, data in junction_sides.items()}
    
    # Initial scan
    initial_frames = get_live_frames(captures)
    for side in side_order:
        analysis = analyze_snapshot(initial_frames[side], model)
        async with state.lock:
            state.latest_vehicle_counts[side] = sum(analysis["counts"].values())
    await generate_traffic_heatmap(state.latest_vehicle_counts, junction_coords)

    current_side_idx = 0
    while state.is_running:
        current_side_name = side_order[current_side_idx]
        
        # CHANGED: Use the helper function to get the snapshot frame and ensure looping
        all_frames = get_live_frames(captures)
        snapshot_frame = all_frames[current_side_name]

        analysis = analyze_snapshot(snapshot_frame, model)
        lane_counts, detections = analysis["counts"], analysis["detections"]
        
        async with state.lock:
            state.latest_vehicle_counts[current_side_name] = sum(lane_counts.values())
            state.active_side_name = current_side_name

        green_duration = logic_controller.calculate_green_duration(lane_counts)
        yellow_duration = logic_controller.yellow
        
        # --- GREEN PHASE ---
        start_time = time.time()
        while time.time() - start_time < green_duration and state.is_running:
            # CHANGED: Use helper function inside the loop
            live_frames = get_live_frames(captures)
            async with state.lock:
                state.light_states = {side: "RED" for side in side_order}
                state.light_states[current_side_name] = "GREEN"
                state.timers = {"state": "GREEN", "duration": green_duration - (time.time() - start_time)}
                state.dashboard_frame = create_dynamic_dashboard(live_frames, state.light_states, current_side_name, state.timers, detections)
            await asyncio.sleep(0.03)

        # --- YELLOW PHASE ---
        start_time = time.time()
        while time.time() - start_time < yellow_duration and state.is_running:
            # CHANGED: Use helper function inside the loop
            live_frames = get_live_frames(captures)
            async with state.lock:
                state.light_states[current_side_name] = "YELLOW"
                state.timers = {"state": "YELLOW", "duration": yellow_duration - (time.time() - start_time)}
                state.dashboard_frame = create_dynamic_dashboard(live_frames, state.light_states, current_side_name, state.timers)
            await asyncio.sleep(0.03)
        
        if state.is_running:
            await generate_traffic_heatmap(state.latest_vehicle_counts, junction_coords)
            current_side_idx = (current_side_idx + 1) % len(side_order)

    for cap in captures.values():
        cap.release()
    print("Simulation stopped and resources released.")

# --- FastAPI Application ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.post("/start-simulation")
async def start_simulation():
    if not state.is_running:
        state.is_running = True
        asyncio.create_task(simulation_loop())
        return JSONResponse(content={"message": "Simulation started successfully."})
    return JSONResponse(content={"message": "Simulation is already running."}, status_code=400)

@app.post("/stop-simulation")
async def stop_simulation():
    if state.is_running:
        state.is_running = False
        return JSONResponse(content={"message": "Simulation stopping."})
    return JSONResponse(content={"message": "Simulation is not running."}, status_code=400)

@app.get("/simulation-status")
async def get_status():
    if not state.is_running: return JSONResponse(content={"status": "not_running"}, status_code=404)
    async with state.lock:
        return JSONResponse(content={
            "status": "running", "active_side": state.active_side_name, "lights": state.light_states,
            "timers": state.timers, "vehicle_counts": state.latest_vehicle_counts
        })

@app.get("/heatmap", response_class=HTMLResponse)
async def get_heatmap():
    try:
        async with aiofiles.open("traffic_heatmap.html", mode="r") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Heatmap not generated yet. Start the simulation.</h1>", status_code=404)

async def video_stream_generator():
    while True:
        async with state.lock:
            frame = state.dashboard_frame
        if frame is None or not state.is_running:
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Simulation Stopped. POST to /start-simulation", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(1)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.03)

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)