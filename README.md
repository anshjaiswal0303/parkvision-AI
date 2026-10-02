# ParkVision AI 🚗

ParkVision AI is a smart parking detection system built using **Python, OpenCV, and FastAPI**.

The system analyzes parking-lot video frames and automatically detects whether individual parking spaces are **Occupied** or **Available**.

## ✨ Features

- 🚗 Real-time parking space detection
- 🟢 Available / 🔴 Occupied slot identification
- 📊 Live occupancy statistics
- 🎥 OpenCV-based video processing
- 🅿️ Individual parking slot monitoring
- 📈 Parking analytics dashboard
- ⚡ FastAPI backend
- 🌐 Responsive web dashboard
- 🔄 Automatic live data updates

## 🖥️ Dashboard

The dashboard provides:

- Total parking spaces
- Available spaces
- Occupied spaces
- Occupancy percentage
- Live computer vision feed
- Recent parking activity

## 🅿️ Parking Slot Detection

Each parking space is individually monitored.

Example:

| Slot | Status |
|------|--------|
| P1 | Occupied |
| P2 | Occupied |
| P3 | Available |
| P4 | Occupied |
| P5 | Occupied |
| P6 | Occupied |
| P7 | Available |
| P8 | Occupied |
| P9 | Available |
| P10 | Occupied |

**Total:** 10  
**Occupied:** 7  
**Available:** 3  
**Occupancy:** 70%

## 🧠 How It Works

1. A parking-lot video is processed using OpenCV.
2. Each parking space is defined using coordinates.
3. The current video frame is compared with a clean parking-lot background.
4. Pixel differences are analyzed to detect changes.
5. Each parking space is classified as occupied or available.
6. FastAPI provides the processed data and video stream.
7. The web dashboard displays the results in real time.

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **FastAPI**
- **Uvicorn**
- **HTML**
- **CSS**
- **JavaScript**

## 📁 Project Structure

```text
ParkVision AI/
│
├── backend/
│   └── main.py
│
├── data/
│   ├── images/
│   │   ├── parking_lot.png
│   │   └── parking_clean.png
│   │
│   ├── videos/
│   │   └── parking_video.mp4
│   │
│   └── parking_spaces.json
│
├── detection/
│   ├── mark_spaces.py
│   └── parking_detector.py
│
├── static/
│   └── index.html
│
├── templates/
│
├── test_opencv.py
├── requirements.txt
├── .gitignore
└── README.md