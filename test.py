
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from ultralytics import YOLO
import time
import csv

model = YOLO("yolov8n.pt")
cap1 = cv2.VideoCapture("cctv1.mp4")
cap2 = cv2.VideoCapture("cctv2.mp4")

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video files.")
    exit()

frame_width = int(cap1.get(3))
frame_height = int(cap1.get(4))
out = cv2.VideoWriter("output_ui.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width * 2, frame_height))

time_limit = 800
signal_timer = time.time()

plt.ion()
fig = plt.figure(figsize=(14, 13))
gs = gridspec.GridSpec(5, 3, height_ratios=[4, 1.5, 0.2, 2, 1.5])  # Added a padding row
ax_video = fig.add_subplot(gs[0, :])
ax_table = fig.add_subplot(gs[1, :])
ax_pad = fig.add_subplot(gs[2, :])
ax_pad.axis('off')  # empty padding
ax_signal = fig.add_subplot(gs[3, 0])
ax_brightness1 = fig.add_subplot(gs[3, 1])
ax_brightness2 = fig.add_subplot(gs[3, 2])
ax_analytics = fig.add_subplot(gs[4, :])

vehicle_history_1 = []
vehicle_history_2 = []
frame_counter = 0

csv_file = open("traffic_data.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Road1_Vehicles", "Road2_Vehicles", "Road1_Pedestrians", "Road2_Pedestrians"])

brightness1 = 30
brightness2 = 30

def process_frame(results, frame):
    vehicle_classes = [2, 3, 5, 7]
    person_class = 0
    vehicle_count = 0
    person_count = 0
    for r in results:
        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cls = int(r.boxes.cls[i])
            if cls in vehicle_classes:
                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vehicle: Green box
            elif cls == person_class:
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Person: Blue box

    return frame, vehicle_count, person_count

def draw_traffic_signals(ax, signal1, signal2, blink):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("Modulated Signal Display")

    for i, signal in enumerate([signal1, signal2]):
        rect = patches.Rectangle((1 + i*5, 4), 3, 5, linewidth=2, edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        ax.text(2.5 + i*5, 9.5, f"Road {i+1}", ha='center', fontsize=10, weight='bold')
        colors = ['red', 'yellow', 'green']
        for j, color in enumerate(colors):
            cy = 8 - j*1.5
            fill = color if signal == color.upper() else 'lightgray'
            circle = plt.Circle((2.5 + i*5, cy), 0.5, color=fill, ec='black', linewidth=2)
            ax.add_patch(circle)
            if signal == color.upper() and blink:
                ax.add_patch(plt.Circle((2.5 + i*5, cy), 0.7, fill=False, ec='orange', linestyle='--', linewidth=2))

# Signal & brightness logic
def get_signals(vc1, vc2, emg1, emg2):
    orig1, orig2 = ("GREEN", "RED") if int(time.time()) % 20 < 10 else ("RED", "GREEN")
    
    # Emergency vehicle has highest priority
    if emg1 and not emg2:
        return "GREEN", "RED"
    elif emg2 and not emg1:
        return "RED", "GREEN"
    elif emg1 and emg2:
        return "YELLOW", "YELLOW"  # Both detected - set caution
    elif orig1 == "RED" and vc2 == 0:
        return "YELLOW", "YELLOW"
    elif orig2 == "RED" and vc1 == 0:
        return "YELLOW", "YELLOW"
    
    return orig1, orig2


def draw_brightness_box(ax, brightness, label="Brightness"):
    ax.clear()
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(label)
    box = patches.Rectangle((0.1, 0.3), 0.8, 0.4, edgecolor='black', facecolor='white')
    ax.add_patch(box)
    ax.text(0.5, 0.5, f"{brightness}%", ha='center', va='center', fontsize=20, color='black')

# Signal & brightness logic
def get_signals(vc1, vc2, emg1, emg2):
    orig1, orig2 = ("GREEN", "RED") if int(time.time()) % 20 < 10 else ("RED", "GREEN")
    
    # Emergency vehicle has highest priority
    if emg1 and not emg2:
        return "GREEN", "RED"
    elif emg2 and not emg1:
        return "RED", "GREEN"
    elif emg1 and emg2:
        return "YELLOW", "YELLOW"  # Both detected - set caution
    elif orig1 == "RED" and vc2 == 0:
        return "YELLOW", "YELLOW"
    elif orig2 == "RED" and vc1 == 0:
        return "YELLOW", "YELLOW"
    
    return orig1, orig2


def smooth_brightness_transition(current, target, step=10):
    if current < target:
        current += step
        if current > target:
            current = target
    elif current > target:
        current -= step
        if current < target:
            current = target
    return current

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    results1 = model(frame1)
    results2 = model(frame2)
    frame1, vc1, pc1 = process_frame(results1, frame1)
    frame2, vc2, pc2 = process_frame(results2, frame2)

    elapsed = time.time() - signal_timer
    if elapsed >= time_limit:
        signal_timer = time.time()
        elapsed = 0

    if elapsed < time_limit / 2:
        original_signal1, original_signal2 = "GREEN", "RED"
    else:
        original_signal1, original_signal2 = "RED", "GREEN"

    if original_signal1 == "RED" and vc2 == 0:
        mod_signal1 = mod_signal2 = "YELLOW"
    elif original_signal2 == "RED" and vc1 == 0:
        mod_signal1 = mod_signal2 = "YELLOW"
    else:
        mod_signal1, mod_signal2 = original_signal1, original_signal2

    target_brightness1 = 100 if (vc1 + pc1) > 0 else 30
    target_brightness2 = 100 if (vc2 + pc2) > 0 else 30
    brightness1 = smooth_brightness_transition(brightness1, target_brightness1)
    brightness2 = smooth_brightness_transition(brightness2, target_brightness2)

    combined_frame = np.hstack((frame1, frame2))
    out.write(combined_frame)

    vehicle_history_1.append(vc1)
    vehicle_history_2.append(vc2)
    csv_writer.writerow([frame_counter, vc1, vc2, pc1, pc2])
    frame_counter += 1

    ax_video.clear()
    ax_video.imshow(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
    ax_video.axis('off')
    ax_video.set_title("Camera Feeds")

    ax_table.clear()
    ax_table.axis('off')
    columns = ['Road', 'Orig. Signal', 'Mod. Signal', 'Brightness', 'Vehicles', 'Pedestrians', 'Timer']
    data = [
        ["1", original_signal1, mod_signal1, f"{brightness1}%", vc1, pc1, f"{int(time_limit - elapsed)}s"],
        ["2", original_signal2, mod_signal2, f"{brightness2}%", vc2, pc2, f"{int(time_limit - elapsed)}s"]
    ]
    table = ax_table.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.scale(1, 2)
    table.set_fontsize(12)

    draw_traffic_signals(ax_signal, mod_signal1, mod_signal2, blink=True)
    draw_brightness_box(ax_brightness1, brightness1, "Streetlight 1 Brightness")
    draw_brightness_box(ax_brightness2, brightness2, "Streetlight 2 Brightness")

    ax_analytics.clear()
    ax_analytics.plot(vehicle_history_1, label='Road 1 Vehicles', color='blue')
    ax_analytics.plot(vehicle_history_2, label='Road 2 Vehicles', color='orange')
    ax_analytics.set_title("Traffic Flow Analytics")
    ax_analytics.set_ylabel("Vehicle Count")
    ax_analytics.set_xlabel("Frame")
    ax_analytics.legend()
    ax_analytics.grid(True)

    plt.pause(0.001)

cap1.release()
cap2.release()
csv_file.close()
out.release()
plt.ioff()
plt.close()
