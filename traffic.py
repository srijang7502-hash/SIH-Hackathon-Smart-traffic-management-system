#!/usr/bin/env python3
"""
junction_3lane_simulation_demo.py
Simulates a 3-lane junction for a single side with realistic traffic light cycle
including GREEN -> YELLOW -> RED and dynamic green duration based on estimated vehicles.
"""

import cv2
import numpy as np
import json
from collections import defaultdict, deque
from ultralytics import YOLO

# ------------------- TRAFFIC LIGHT CONTROLLER -------------------
class SingleSideTrafficLight:
    """
    Simulates a realistic traffic signal cycle for one side of a junction.
    Each lane: GREEN -> YELLOW -> RED
    Dynamic green based on number of vehicles
    """
    def __init__(self, lanes, min_green=5, yellow=3, time_per_vehicle=2.0, max_green_dict=None):
        self.lanes = lanes
        self.min_green = min_green
        self.yellow = yellow
        self.time_per_vehicle = time_per_vehicle
        self.max_green_dict = max_green_dict or {lane: 40 for lane in lanes}
        
        self.current_idx = 0
        self.state = "GREEN"  # GREEN, YELLOW, RED
        self.timer = 0
        self.green_duration = min_green
        self.active_vehicles = {lane:set() for lane in lanes}
        self.red_duration = {lane:0 for lane in lanes}  # will calculate based on other lanes

    def update_active_vehicles(self, lane_vehicle_ids):
        for lane in self.lanes:
            # Estimate hidden vehicles by multiplying visible by 1.5
            visible_count = len(lane_vehicle_ids.get(lane,set()))
            estimated_count = int(visible_count * 1.5)
            self.active_vehicles[lane] = estimated_count

    def step(self):
        current_lane = self.lanes[self.current_idx]
        self.timer += 1

        # Dynamic green based on vehicles
        queue_len = self.active_vehicles[current_lane]
        estimated_green = self.min_green + queue_len * self.time_per_vehicle
        lane_max = self.max_green_dict.get(current_lane, 40)
        self.green_duration = min(max(estimated_green, self.min_green), lane_max)

        # Total red for this lane = sum of other lanes' green + yellow + clearance
        total_red = sum([self.max_green_dict.get(l, 40) + self.yellow for i,l in enumerate(self.lanes) if l != current_lane])
        self.red_duration[current_lane] = total_red

        # Traffic light cycle
        if self.state == "GREEN":
            if self.timer >= self.green_duration:
                self.state = "YELLOW"
                self.timer = 0
        elif self.state == "YELLOW":
            if self.timer >= self.yellow:
                self.state = "RED"
                self.timer = 0
        elif self.state == "RED":
            if self.timer >= self.red_duration[current_lane]:
                # Move to next lane
                self.current_idx = (self.current_idx + 1) % len(self.lanes)
                self.state = "GREEN"
                self.timer = 0

        return current_lane, self.state, round(self.timer,2), round(self.green_duration,2)

# ------------------- HELPERS -------------------
def centroid_of_box(box_xyxy):
    x1,y1,x2,y2 = map(int, box_xyxy)
    return int((x1+x2)/2), int((y1+y2)/2)

def estimate_direction(cx, lane_width):
    """Divide road width into 3 sections: left / straight / right"""
    if cx < lane_width:
        return "left"
    elif cx < 2*lane_width:
        return "straight"
    else:
        return "right"

# ------------------- MAIN SIMULATION -------------------
def simulate_3lane_demo(video_path, model_path="yolov8s.pt", display=True, out_video="demo_out.mp4"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video, fourcc, fps, (frame_w, frame_h))

    lane_width = frame_w // 3
    lane_names = ["left", "straight", "right"]

    # Tracking for JSON output
    track_hist = defaultdict(lambda: deque(maxlen=30))
    track_first_seen = {}
    track_last_seen = {}
    track_class = {}
    track_counted = set()
    turn_decisions = []
    lane_log = []
    vehicle_classes = ["bicycle","car","motorcycle","bus","truck"]
    model_names = getattr(model,"names",{0:"person",1:"bicycle",2:"car",3:"motorcycle",5:"bus",7:"truck"})
    frame_idx = 0

    # Traffic light controller
    controller = SingleSideTrafficLight(
        lanes=lane_names,
        min_green=5,
        yellow=3,
        time_per_vehicle=2.0,
        max_green_dict={"left":60,"straight":90,"right":30}
    )

    # YOLO tracking stream
    stream = model.track(source=video_path, tracker="bytetrack.yaml", stream=True, show=False, conf=0.15, imgsz=1280)
    for res in stream:
        frame = res.orig_img.copy()
        frame_idx += 1

        lane_vehicle_ids = {lane:set() for lane in lane_names}
        lane_counts = {lane:0 for lane in lane_names}

        if hasattr(res,"boxes") and res.boxes is not None:
            for b in res.boxes:
                cls_id = int(b.cls.cpu().item())
                cls_name = model_names.get(cls_id,str(cls_id))
                if cls_name=="person" or cls_name not in vehicle_classes: continue
                tid = int(b.id.cpu().item()) if hasattr(b,"id") and b.id is not None else None
                if tid is None: continue

                cx,cy = centroid_of_box(b.xyxy.cpu().numpy().reshape(-1))
                direction = estimate_direction(cx, lane_width)
                lane_vehicle_ids[direction].add(tid)
                lane_counts[direction] += 1

                # Track for JSON
                track_hist[tid].append((cx,cy,frame_idx))
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                    track_class[tid] = cls_name
                track_last_seen[tid] = frame_idx

        # Finalize vehicle turns
        gap_threshold = max(15,int(fps*0.5))
        for tid,last in list(track_last_seen.items()):
            if frame_idx - last > gap_threshold and tid not in track_counted:
                hist = track_hist.get(tid)
                if hist and len(hist)>=5:
                    ex,ey,_ = hist[-1]
                    lane_decision = estimate_direction(ex, lane_width)
                else:
                    lane_decision = "unknown"
                turn_decisions.append({
                    "id":tid,"class":track_class.get(tid,"unknown"),
                    "start_frame":track_first_seen.get(tid),
                    "end_frame":track_last_seen.get(tid),
                    "start_xy": (track_hist[tid][0][0],track_hist[tid][0][1]),
                    "end_xy": (track_hist[tid][-1][0],track_hist[tid][-1][1]),
                    "direction": lane_decision})
                track_counted.add(tid)

        # Update traffic light
        controller.update_active_vehicles(lane_vehicle_ids)
        active_lane, signal_state, elapsed, green_duration = controller.step()

        # Record JSON log
        lane_log.append({
            "frame": frame_idx,
            **lane_counts,
            "signal_lane": active_lane,
            "signal_state": signal_state,
            "green_duration": green_duration
        })

        # Display annotated video
        if display:
            colors = {"GREEN":(0,255,0),"YELLOW":(0,255,255),"RED":(0,0,255)}
            for i,lane in enumerate(lane_names):
                x1 = i*lane_width
                x2 = (i+1)*lane_width
                lane_color = colors[signal_state] if lane==active_lane else colors["RED"]
                cv2.rectangle(frame,(x1,0),(x2,frame_h),lane_color,2)
                cv2.putText(frame,lane,(x1+10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,lane_color,2)
            cv2.putText(frame,f"Signal:{active_lane} State:{signal_state} Timer:{elapsed}s Green:{green_duration}s",
                        (10,frame_h-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.imshow("3-Lane Junction Demo", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

        out_writer.write(frame)

    out_writer.release()
    cap.release()
    if display: cv2.destroyAllWindows()

    return json.dumps(turn_decisions, indent=2), json.dumps(lane_log, indent=2)

# ------------------- RUN -------------------
if __name__=="__main__":
    video_file = "videos/traffic_jam_1.mp4"  # replace with your video
    vehicle_turns_json, lane_counts_json = simulate_3lane_demo(video_file, display=True)

    with open("vehicle_turns.json","w") as f: f.write(vehicle_turns_json)
    with open("lane_counts.json","w") as f: f.write(lane_counts_json)
    print("Demo simulation complete! JSON outputs saved.")
