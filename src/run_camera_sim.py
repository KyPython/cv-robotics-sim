"""
Camera Simulation for Robotics Control.

This script simulates a camera feed and makes control decisions based on
computer vision model predictions. In a real robot, this would be replaced
with actual camera streams and robot control commands.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model, run_inference, get_class_names


def simulate_camera_from_images(data_dir='data', extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Simulate camera feed by reading images from a directory.
    
    Args:
        data_dir: Directory containing images
        extensions: Tuple of valid image extensions
    
    Yields:
        Tuple of (frame, frame_path) for each image
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Data directory '{data_dir}' does not exist. Creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"Please add some images to {data_path.absolute()} and run again.")
        return
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {data_dir}. Please add some images and try again.")
        return
    
    image_files.sort()
    print(f"Found {len(image_files)} images in {data_dir}")
    
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            yield frame, str(img_path)
        else:
            print(f"Warning: Could not read {img_path}")


def simulate_camera_from_video(video_path):
    """
    Simulate camera feed by reading frames from a video file.
    
    Args:
        video_path: Path to video file
    
    Yields:
        Frame (numpy array) from the video
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        yield frame
    
    cap.release()
    print(f"Processed {frame_count} frames from video")


def make_control_decision(prediction_result, threshold=0.5):
    """
    Convert model prediction into a control decision.
    
    In a real robot, this would send commands to motor controllers,
    ROS topics, or microcontroller interfaces.
    
    Args:
        prediction_result: Dictionary from run_inference()
        threshold: Confidence threshold for making decisions
    
    Returns:
        Dictionary with control decision information
    """
    prediction = prediction_result['prediction']
    confidence = prediction_result['confidence']
    probabilities = prediction_result['probabilities']
    
    class_names = get_class_names()
    predicted_class = class_names[prediction]
    
    # Simple control logic:
    # - If object_detected (class 1) with high confidence -> move forward
    # - Otherwise -> stop or turn
    if prediction == 1 and confidence >= threshold:
        action = "move_forward"
        action_code = 1
    elif prediction == 0 and confidence >= threshold:
        action = "stop"
        action_code = 0
    else:
        # Low confidence -> cautious turn
        action = "turn"
        action_code = 2
    
    return {
        'action': action,
        'action_code': action_code,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }


def send_robot_command(decision):
    """
    Placeholder for sending commands to a real robot.
    
    In a real implementation, this would:
    - Send ROS messages to /cmd_vel topic (for ROS-based robots)
    - Send serial commands to Arduino/microcontroller
    - Call robot control API endpoints
    - Publish to MQTT topics
    
    Args:
        decision: Dictionary with control decision from make_control_decision()
    """
    action = decision['action']
    action_code = decision['action_code']
    
    # ROS EXAMPLE (commented out - would require ROS installation):
    # from geometry_msgs.msg import Twist
    # cmd = Twist()
    # if action_code == 1:  # move_forward
    #     cmd.linear.x = 0.5
    # elif action_code == 0:  # stop
    #     cmd.linear.x = 0.0
    # elif action_code == 2:  # turn
    #     cmd.angular.z = 0.5
    # cmd_vel_pub.publish(cmd)
    
    # Arduino/Serial EXAMPLE (commented out):
    # import serial
    # ser = serial.Serial('/dev/ttyUSB0', 9600)
    # ser.write(f"{action_code}\n".encode())
    
    # For simulation, just print
    print(f"  → Robot Command: {action.upper()} (code: {action_code})")


def overlay_text_on_frame(frame, decision, frame_info=None):
    """
    Overlay control decision text on the frame for visualization.
    
    Args:
        frame: OpenCV frame (numpy array)
        decision: Dictionary from make_control_decision()
        frame_info: Optional frame information (e.g., frame number, path)
    
    Returns:
        Frame with overlaid text
    """
    display_frame = frame.copy()
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Determine color based on action
    if decision['action'] == "move_forward":
        color = (0, 255, 0)  # Green
    elif decision['action'] == "stop":
        color = (0, 0, 255)  # Red
    else:
        color = (0, 165, 255)  # Orange
    
    # Prepare text lines
    lines = [
        f"Action: {decision['action'].upper()}",
        f"Class: {decision['predicted_class']}",
        f"Confidence: {decision['confidence']:.2%}"
    ]
    
    if frame_info:
        lines.insert(0, frame_info)
    
    # Draw semi-transparent background
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 30 + len(lines) * 25), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
    
    # Draw text
    y_offset = 30
    for line in lines:
        cv2.putText(display_frame, line, (15, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
    
    return display_frame


def main(args):
    """
    Main simulation loop.
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("Computer Vision Robotics Simulation")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path=args.model_path, 
                      device=args.device, 
                      pretrained=args.pretrained)
    print(f"Model loaded on device: {args.device}")
    
    # Determine camera source
    if args.video:
        print(f"\nSimulating camera from video: {args.video}")
        frame_source = simulate_camera_from_video(args.video)
        frame_info_template = None
    else:
        print(f"\nSimulating camera from images in: {args.data_dir}")
        frame_source = simulate_camera_from_images(args.data_dir)
        frame_info_template = "Frame: {}"
    
    # Main control loop
    # In a real robot, this would be an infinite loop reading from camera
    frame_count = 0
    
    print("\nStarting simulation loop...")
    print("-" * 60)
    
    for frame_data in frame_source:
        if args.video:
            frame = frame_data
            frame_count += 1
            frame_info = f"Frame #{frame_count}"
        else:
            frame, frame_path = frame_data
            frame_count += 1
            frame_name = Path(frame_path).name
            frame_info = f"Frame #{frame_count}: {frame_name}"
        
        # Run inference
        result = run_inference(model, frame, device=args.device)
        
        # Make control decision
        decision = make_control_decision(result, threshold=args.threshold)
        
        # Print decision
        print(f"\n[{frame_info}]")
        print(f"  Prediction: {decision['predicted_class']} "
              f"(confidence: {decision['confidence']:.2%})")
        
        # Send robot command (placeholder in simulation)
        send_robot_command(decision)
        
        # Display frame if requested
        if args.display:
            display_frame = overlay_text_on_frame(frame, decision, frame_info)
            cv2.imshow('Robotics Simulation', display_frame)
            
            # Press 'q' to quit, space to continue
            key = cv2.waitKey(args.delay) & 0xFF
            if key == ord('q'):
                print("\nSimulation stopped by user.")
                break
            elif key == ord(' '):
                # Pause on spacebar
                cv2.waitKey(0)
        
        # Limit frames if specified
        if args.max_frames and frame_count >= args.max_frames:
            print(f"\nReached maximum frame limit ({args.max_frames})")
            break
    
    if args.display:
        cv2.destroyAllWindows()
    
    print("-" * 60)
    print(f"\nSimulation complete. Processed {frame_count} frame(s).")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulate camera feed and make robotics control decisions'
    )
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing input images (default: data)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file (overrides --data-dir)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model checkpoint (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on (default: cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for decisions (default: 0.5)')
    parser.add_argument('--display', action='store_true',
                       help='Display frames with OpenCV window')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in milliseconds (default: 100)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: unlimited)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Use randomly initialized weights instead of pretrained')
    parser.set_defaults(pretrained=True)
    
    args = parser.parse_args()
    
    main(args)

