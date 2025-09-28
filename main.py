#!/usr/bin/env python3
"""
main.py - FastAPI Backend for Smart Traffic Simulation

This script creates a complete API backend for the 4-way junction simulation.
- Endpoints to start/stop the simulation.
- Streams the 2x2 dashboard as a video feed.
- Serves the Folium heatmap, which is updated live.
- Provides a JSON endpoint for real-time traffic light status.
- Adds YOLO bounding boxes to the video feed for the currently GREEN light.
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
import uvicorn
import aiofiles

# --- Global State & Configuration ---

# A simple class to hold the simulation's state so it can be accessed by API endpoints
class SimulationState:
    def __init__(self):
        self.is_running = False
        self.dashboard_frame = None
        self.light_states = {}
        self.timers = {}
        self.latest_vehicle_counts = {}
        self.active_side_name = ""
        self.lock = asyncio.Lock()

# Instantiate global state
state = SimulationState()

# --- Helper Functions & Core Logic (Adapted for Async) ---
# Most functions remain the same, but the simulation loop is now async

def estimate_direction(cx, lane_width):
    if cx < lane_width: return "left"
    elif cx < 2 * lane_width: return "straight"
    else: return "right"

def calculate_dynamic_multiplier(visible_vehicles):
    if visible_vehicles < 5: return 1.2
    elif visible_vehicles < 15: return 1.8
    else: return 2.5

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
    """Modified to return both counts and raw detection boxes."""
    if image is None: return {"counts": {"left": 0, "straight": 0, "right": 0}, "detections": []}
    
    frame_h, frame_w, _ = image.shape
    lane_width = frame_w // 3
    lane_counts = {"left": 0, "straight": 0, "right": 0}
    detections = []
    
    vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
    results = model(image, verbose=False, conf=0.35) # Slightly higher confidence for drawing
    
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
        
        # ** NEW: Draw bounding boxes if this is the active green lane **
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
    for side in side_order:
        ret, frame = captures[side].read()
        if ret:
            analysis = analyze_snapshot(frame, model)
            async with state.lock:
                state.latest_vehicle_counts[side] = sum(analysis["counts"].values())
    await generate_traffic_heatmap(state.latest_vehicle_counts, junction_coords)

    current_side_idx = 0
    while state.is_running:
        current_side_name = side_order[current_side_idx]
        
        # Frame reading and analysis
        ret, snapshot_frame = captures[current_side_name].read()
        if not ret:
            captures[current_side_name].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, snapshot_frame = captures[current_side_name].read()
        
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
            live_frames = {s: c.read()[1] for s, c in captures.items()}
            async with state.lock:
                state.light_states = {side: "RED" for side in side_order}
                state.light_states[current_side_name] = "GREEN"
                state.timers = {"state": "GREEN", "duration": green_duration - (time.time() - start_time)}
                state.dashboard_frame = create_dynamic_dashboard(live_frames, state.light_states, current_side_name, state.timers, detections)
            await asyncio.sleep(0.03) # Approx 30 FPS

        # --- YELLOW PHASE ---
        start_time = time.time()
        while time.time() - start_time < yellow_duration and state.is_running:
            live_frames = {s: c.read()[1] for s, c in captures.items()}
            async with state.lock:
                state.light_states[current_side_name] = "YELLOW"
                state.timers = {"state": "YELLOW", "duration": yellow_duration - (time.time() - start_time)}
                state.dashboard_frame = create_dynamic_dashboard(live_frames, state.light_states, current_side_name, state.timers) # No boxes on yellow
            await asyncio.sleep(0.03)

        await generate_traffic_heatmap(state.latest_vehicle_counts, junction_coords)
        current_side_idx = (current_side_idx + 1) % len(side_order)

    # Cleanup
    for cap in captures.values():
        cap.release()
    print("Simulation stopped and resources released.")

# --- FastAPI Application ---

app = FastAPI()

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
    if not state.is_running:
        return JSONResponse(content={"status": "not_running"}, status_code=404)
    async with state.lock:
        return JSONResponse(content={
            "status": "running",
            "active_side": state.active_side_name,
            "lights": state.light_states,
            "timers": state.timers,
            "vehicle_counts": state.latest_vehicle_counts
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
            # If simulation is not running, stream a placeholder
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Simulation Stopped. Press 'Start Simulation'.", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(1)
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.03) # Match FPS of simulation loop

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Example Frontend Page ---
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Traffic Control Dashboard</title>
        <style>
            body { font-family: sans-serif; background-color: #1a1a1a; color: #e0e0e0; margin: 0; display: flex; flex-direction: column; align-items: center; }
            h1 { color: #4CAF50; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; width: 100%; padding: 20px; box-sizing: border-box; }
            .card { background-color: #2c2c2c; border-radius: 8px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
            .video-container { width: 1280px; height: 720px; }
            .map-container { width: 600px; height: 500px; }
            .status-container { width: 600px; }
            img { width: 100%; height: 100%; border-radius: 8px; }
            iframe { width: 100%; height: 100%; border: none; border-radius: 8px; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 5px; }
            button:hover { background-color: #45a049; }
            #status-box { background-color: #1a1a1a; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; }
        </style>
    </head>
    <body>
        <h1>Smart Traffic Control Dashboard</h1>
        <div class="controls card">
            <button onclick="startSim()">Start Simulation</button>
            <button onclick="stopSim()">Stop Simulation</button>
        </div>
        <div class="container">
            <div class="video-container card">
                <img id="videoFeed" src="/video-feed" alt="Live Video Feed">
            </div>
            <div class="right-panel">
                <div class="map-container card">
                    <iframe id="heatmapFrame" src="/heatmap"></iframe>
                </div>
                <div class="status-container card">
                    <h2>Live Status</h2>
                    <div id="status-box">Fetching status...</div>
                </div>
            </div>
        </div>

        <script>
            function startSim() { fetch('/start-simulation', { method: 'POST' }); }
            function stopSim() { fetch('/stop-simulation', { method: 'POST' }); }

            // Update status and heatmap periodically
            setInterval(() => {
                fetch('/simulation-status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status-box').textContent = JSON.stringify(data, null, 2);
                    })
                    .catch(err => {
                        document.getElementById('status-box').textContent = "Simulation not running or error fetching status.";
                    });
                
                // Force the iframe to reload
                const iframe = document.getElementById('heatmapFrame');
                iframe.src = iframe.src;

            }, 2000); // Update every 2 seconds
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)