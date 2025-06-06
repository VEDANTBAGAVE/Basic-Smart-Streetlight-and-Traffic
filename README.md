# 🚦 Smart Traffic & Streetlight Control System using YOLOv8

This project is a Smart Traffic and Streetlight Control System that uses **YOLOv8 object detection** to analyze traffic footage from two cameras, count vehicles and pedestrians, and modulate traffic signals and streetlight brightness accordingly.

It visualizes:

- Real-time video feeds with detection boxes.
- Signal states (original & modulated).
- Pedestrian and vehicle counts.
- Streetlight brightness modulation.
- Traffic analytics graph.

## 📁 Project Structure

```bash
.
├── cctv1.mp4 # First traffic camera input
├── cctv2.mp4 # Second traffic camera input
├── output_ui.avi # (Generated) Output video with side-by-side visuals
├── traffic_data.csv # (Generated) CSV log of traffic data per frame
├── yolov8n.pt # YOLOv8 nano weights file
├── main.py # Main script
└── README.md # Project documentation
```

---

## 🛠️ Requirements

- Python 3.8+
- `ultralytics` for YOLOv8
- OpenCV
- Matplotlib
- NumPy

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VEDANTBAGAVE/smart-traffic-system.git
cd smart-traffic-system
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

Create a requirements.txt file with:

```bash
ultralytics
opencv-python
matplotlib
numpy

```

## 📥 YOLOv8 Weights

Download the YOLOv8 Nano weights (yolov8n.pt) from the Ultralytics Model Zoo or using:

```bash
from ultralytics import YOLO
YOLO("yolov8n.pt")
```

Ensure yolov8n.pt is placed in the project root directory.

## 📽️ How to Run

Make sure cctv1.mp4 and cctv2.mp4 are present (you can replace them with your own camera recordings).

Run the script:

```bash
python main.py
```

## 💡 Features

🚘 Vehicle and pedestrian detection using YOLOv8

📊 Real-time analytics with Matplotlib

🚦 Signal modulation logic (Emergency vehicles, no-traffic scenario)

💡 Adaptive streetlight brightness

📈 Logs traffic data into a CSV file (traffic_data.csv)

🎥 Combines both video feeds into a side-by-side rendered output (output_ui.avi)

Performance can be enhanced by switching to more powerful YOLO models like yolov8s.pt or yolov8m.pt.

## 📊 Output

output_ui.avi: Visualization combining both camera feeds with detection overlays

traffic_data.csv: Logs frame-wise count of vehicles and pedestrians per road

Live UI showing:

Video feed

Signal state

Brightness box

Traffic flow graph
