# Computer Vision Robotics Simulation

A lightweight computer vision + robotics simulation that demonstrates how to make control decisions based on camera frames. This project simulates a camera feed and uses a pretrained CNN to make simple control decisions (move forward, stop, turn) without requiring actual robot hardware.

## Overview

This project consists of:
- **Model Module** (`src/model.py`): A lightweight CNN using MobileNetV2 for binary classification (object detected vs. no object)
- **Simulation Script** (`src/run_camera_sim.py`): Simulates a camera feed and makes control decisions based on model predictions
- **Control Logic**: Simple decision-making that can be extended to real robot commands

## Project Structure

```
cv-robotics-sim/
├── data/              # Place your test images or videos here
├── src/
│   ├── model.py      # CNN model definition and inference functions
│   └── run_camera_sim.py  # Main simulation script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Test Data

You have two options for providing input frames:

#### Option A: Use Images
Place test images (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the `data/` directory:

```bash
mkdir -p data
# Copy some test images into data/
cp /path/to/your/images/*.jpg data/
```

The script will automatically process all images in this directory.

#### Option B: Use a Video File
Provide a video file path when running the script:

```bash
python src/run_camera_sim.py --video path/to/your/video.mp4
```

## Running the Simulation

### Basic Usage

Process images from the `data/` directory:
```bash
python src/run_camera_sim.py
```

Process a video file:
```bash
python src/run_camera_sim.py --video path/to/video.mp4
```

### With Visual Display

Show frames with overlaid control decisions:
```bash
python src/run_camera_sim.py --display
```

Controls while displaying:
- **Space**: Pause/continue
- **Q**: Quit simulation

### Advanced Options

```bash
# Adjust confidence threshold (default: 0.5)
python src/run_camera_sim.py --threshold 0.7

# Limit number of frames processed
python src/run_camera_sim.py --max-frames 10

# Use GPU if available
python src/run_camera_sim.py --device cuda

# Adjust frame delay when displaying (milliseconds)
python src/run_camera_sim.py --display --delay 200

# Use a custom trained model
python src/run_camera_sim.py --model-path models/checkpoint.pth
```

## Control Decision Logic

The model makes binary predictions:
- **Class 0**: `no_object` → Robot action: **STOP**
- **Class 1**: `object_detected` → Robot action: **MOVE FORWARD**
- **Low Confidence**: → Robot action: **TURN** (cautious mode)

Decision threshold is configurable via `--threshold` (default: 0.5).

## Mapping to Real Hardware

This simulation demonstrates the control loop that would run on a real robot. Here's how to adapt it:

### 1. Replace Camera Simulation with Real Camera

**Current (simulation):**
```python
for frame in simulate_camera_from_images('data'):
    # Process frame
```

**Real robot (example):**
```python
import cv2
cap = cv2.VideoCapture(0)  # USB camera index
while True:
    ret, frame = cap.read()
    if ret:
        # Process frame (same as simulation)
```

Or using ROS:
```python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def image_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    # Process frame (same as simulation)
    
rospy.Subscriber('/camera/image_raw', Image, image_callback)
```

### 2. Replace Control Commands with Real Robot Commands

**Current (simulation):**
```python
def send_robot_command(decision):
    print(f"Robot Command: {decision['action']}")
```

**Real robot - ROS example:**
```python
import rospy
from geometry_msgs.msg import Twist

cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

def send_robot_command(decision):
    cmd = Twist()
    if decision['action'] == 'move_forward':
        cmd.linear.x = 0.5
        cmd.angular.z = 0.0
    elif decision['action'] == 'stop':
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
    elif decision['action'] == 'turn':
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5
    cmd_vel_pub.publish(cmd)
```

**Real robot - Arduino/Serial example:**
```python
import serial
ser = serial.Serial('/dev/ttyUSB0', 9600)

def send_robot_command(decision):
    action_code = decision['action_code']
    ser.write(f"{action_code}\n".encode())
```

**Real robot - HTTP API example:**
```python
import requests

def send_robot_command(decision):
    url = "http://robot-api.local/control"
    payload = {'action': decision['action']}
    requests.post(url, json=payload)
```

### 3. Integration Points

Key integration points are marked in the code with comments:
- **Camera input**: Replace `simulate_camera_from_images()` or `simulate_camera_from_video()`
- **Robot commands**: Replace `send_robot_command()` function
- **Control loop**: Modify `main()` function for continuous operation

## Model Details

The project uses a **MobileNetV2** architecture pretrained on ImageNet, with a custom binary classification head:
- Input: 224x224 RGB images
- Output: 2 classes (no_object, object_detected)
- Framework: PyTorch

To train a custom model for your specific use case:
1. Prepare a labeled dataset
2. Fine-tune the model (modify the last layer)
3. Save the checkpoint
4. Use `--model-path` to load your trained model

## Performance

- **Inference speed**: ~50-100ms per frame on CPU (depends on hardware)
- **Memory**: ~20-50MB model size
- **Compatible**: CPU-only (no GPU required)

## Limitations & Future Improvements

- Current model uses generic pretrained weights (not fine-tuned for specific objects)
- Binary classification is simplified (real robots may need more nuanced decisions)
- No temporal information (doesn't consider previous frames)
- No safety checks or emergency stops (critical for real robots)

**Suggested improvements:**
- Fine-tune on domain-specific data
- Add temporal smoothing (consider previous N frames)
- Implement more sophisticated control policies
- Add safety limits and emergency stop mechanisms
- Integrate with SLAM or navigation stacks for autonomous operation

## License

This is an educational/prototyping project. Feel free to modify and extend for your needs.

