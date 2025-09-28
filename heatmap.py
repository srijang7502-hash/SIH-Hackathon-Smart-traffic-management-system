#!/usr/bin/env python3
"""
traffic_heatmap_demo_fixed.py
Generates a traffic density heatmap using real vehicle counts for a demo junction
and synthetic traffic for surrounding streets.
All points now include a weight for Folium compatibility.
"""

import json
import folium
from folium.plugins import HeatMap
import random

# ------------------- CONFIG -------------------
# Junction coordinates (lat, lon) - replace with your junction's coordinates
junction_coords = [
    (28.6139, 77.2090),   # left lane
    (28.61395, 77.2090),  # straight lane
    (28.6140, 77.2090)    # right lane
]

# Surrounding area bounding box (lat_min, lat_max, lon_min, lon_max)
bbox = (28.6125, 28.6155, 77.2080, 77.2105)

# Maximum synthetic traffic points
synthetic_points = 200

# Heatmap parameters
heatmap_radius = 25
heatmap_blur = 15

# ------------------- LOAD REAL JUNCTION DATA -------------------
with open("lane_counts.json", "r") as f:
    lane_data = json.load(f)

# For demo, take the **last frame** counts
last_frame = lane_data[-1]
real_counts = [
    last_frame.get("left", 0),
    last_frame.get("straight", 0),
    last_frame.get("right", 0)
]

# ------------------- GENERATE HEAT POINTS -------------------
heat_points = []

# Real junction points
for coords, count in zip(junction_coords, real_counts):
    for _ in range(count):
        # Slight jitter to avoid exact overlap
        lat_jitter = coords[0] + random.uniform(-0.00005, 0.00005)
        lon_jitter = coords[1] + random.uniform(-0.00005, 0.00005)
        heat_points.append([lat_jitter, lon_jitter, 1.0])  # weight = 1.0

# Synthetic traffic for surrounding area
lat_min, lat_max, lon_min, lon_max = bbox
for _ in range(synthetic_points):
    lat = random.uniform(lat_min, lat_max)
    lon = random.uniform(lon_min, lon_max)
    weight = random.uniform(0.2, 0.8)
    heat_points.append([lat, lon, weight])

# ------------------- CREATE MAP -------------------
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="OpenStreetMap")

# Add HeatMap
HeatMap(heat_points, radius=heatmap_radius, blur=heatmap_blur).add_to(m)

# Save map
output_file = "traffic_heatmap_fixed.html"
m.save(output_file)
print(f"Traffic heatmap generated: {output_file}")
