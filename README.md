# 🚗 Surround View Autonomous Parking System

An autonomous parking system using **Surround View Moniter** (bird's eye view) and perform YOLO-based parking spot detection at Gazebo simulation.

---

## 📌 Overview

This project implements a full autonomous parking pipeline using a surround view camera system. Parking spots are detected using YOLOv8 and a pose estimation model, and the vehicle autonomously navigates into the target spot.

**Key highlights:**
- YOLOv8 pose estimation for parking spot recognition
- Modular ROS2 node architecture (`camera_perception_pkg`, `decision_making_pkg`, `serial_communication_pkg`)
- Supports at **Gazebo simulation**
- Includes calibration tools and data analysis utilities

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Middleware | ROS 2 Humble |
| Simulation | Gazebo |
| Vision / AI | YOLOv8 (Ultralytics 8.2.69), OpenCV |
| Language | Python 3, Shell, CMake |
| Communication | pyserial (serial to motor controller) |
| Other | pynput, transformers, HuggingFace Hub |


---

## ⚙️ Installation

### Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yunss01/surround_view_parking.git
   cd surround_view_parking
   ```

2. Run the installation script:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

   This will install:
   - Python dependencies: `opencv-python`, `pyserial`, `ultralytics`, `pynput`, `transformers`, `huggingface_hub`
   - ROS 2 Gazebo packages: `gazebo-ros-pkgs`, `xacro`, `gazebo-ros`, `gazebo-msgs`, `gazebo-plugins`
   - Gazebo simulation model assets

3. Build the ROS 2 workspace:
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source ./install/local_setup.bash
   ```

---

## 🚀 Usage

```bash
ros2 launch simulation_pkg mission_sim.launch.py
```

---

## 🤖 Pre-trained Models

| File | Description |
|---|---|
| `best_pose_rev03.pt` | YOLOv8 pose estimation model for parking spot orientation |

---

## 🔬 Data Analysis & Utilities

| Folder | Description |
|---|---|
| `DL model/` | Scripts for training and evaluating deep learning models |
| `data_analyze/` | Data exploration and visualization tools |
| `image stdev extract/` | Utilities for extracting image standard deviation (used for data quality filtering) |
| `npy/` | Preprocessed NumPy arrays for model training/testing |

---

## 📦 ROS2 Package Architecture

```
surround_view_parking
├── camera_perception_pkg
│   ├── image_publisher_node          → Reads and publishes raw fisheye camera feeds
│   ├── surroundview_stitching_node   → Undistorts, warps, and stitches 4 cameras into BEV
│   ├── yolov8_node                   → Runs YOLO detection/pose on the stitched image
│   └── parking_spot_extractor_node   → Extracts valid parking slot candidates
│
├── decision_making_pkg
│   ├── parking_planner_node          → Selects target slot and computes parking path
│   └── motion_planner_node           → Generates velocity/steering commands
│
├── serial_communication_pkg
│   └── serial_sender_node            → Transmits control commands to vehicle via serial
│
└── simulation_pkg
    └── launch/                        → Launch files (mission_sim.launch.py)
     
```

---
