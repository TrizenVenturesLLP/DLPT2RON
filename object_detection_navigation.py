from ultralytics import YOLO
import cv2
import numpy as np
from playsound import playsound
import threading

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8n for lightweight, faster inference

def play_sound(direction):
    """Play a sound alert for the given direction."""
    sounds = {
        "left": "left_instruction.mp3",
        "right": "right_instruction.mp3",
        "center": "center_instruction.mp3"
    }
    if direction in sounds:
        threading.Thread(target=playsound, args=(sounds[direction],)).start()

def detect_obstacles():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    print("Camera opened successfully. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from camera.")
            break

        frame_height, frame_width, _ = frame.shape

        # Define regions for left, center, and right
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        # Perform detection
        results = model(frame)

        # Annotate frame with detections
        annotated_frame = frame.copy()
        instructions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                object_center_x = (x1 + x2) // 2

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Obstacle ({x1},{y1})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Determine region and give instructions
                if object_center_x < left_boundary:
                    instructions.append("Obstacle on the left, move to the center or right.")
                    play_sound("left")
                elif object_center_x > right_boundary:
                    instructions.append("Obstacle on the right, move to the center or left.")
                    play_sound("right")
                else:
                    instructions.append("Obstacle in the center, avoid or move left/right.")
                    play_sound("center")

        # Draw region divisions on the frame
        cv2.line(annotated_frame, (left_boundary, 0), (left_boundary, frame_height), (255, 0, 0), 2)
        cv2.line(annotated_frame, (right_boundary, 0), (right_boundary, frame_height), (255, 0, 0), 2)

        # Display instructions on the terminal
        if instructions:
            print("\n".join(set(instructions)))

        # Show the frame with detections
        cv2.imshow("Object Detection for Navigation", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    detect_obstacles()
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from pydub import AudioSegment
# from pydub.playback import play
# import threading
# import time
# import os
#
# # Ensure FFmpeg is set up correctly
# if not os.system("ffmpeg -version"):
#     print("FFmpeg is available.")
# else:
#     raise RuntimeError("FFmpeg is not installed or not in PATH. Please install and configure it.")
#
# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use yolov8n for lightweight, faster inference
#
# # Load audio files
# try:
#     audio_files = {
#         "left": AudioSegment.from_file("left_instruction.mp3"),
#         "center": AudioSegment.from_file("center_instruction.mp3"),
#         "right": AudioSegment.from_file("right_instruction.mp3")
#     }
# except FileNotFoundError as e:
#     raise FileNotFoundError(f"Audio file missing: {e}")
#
# # Cooldown dictionary to avoid repeated alerts
# cooldown = {"left": 0, "center": 0, "right": 0}
# COOLDOWN_TIME = 3  # seconds
#
#
# def play_sound(direction):
#     """Play a sound alert for the given direction."""
#     current_time = time.time()
#     if current_time - cooldown[direction] > COOLDOWN_TIME:
#         cooldown[direction] = current_time
#         threading.Thread(target=play, args=(audio_files[direction],)).start()
#
#
# def detect_obstacles():
#     cap = cv2.VideoCapture(0)  # Open webcam
#
#     if not cap.isOpened():
#         print("Error: Camera not found!")
#         return
#
#     print("Camera opened successfully. Press 'q' to exit.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to fetch frame from camera.")
#             break
#
#         frame_height, frame_width, _ = frame.shape
#
#         # Define regions for left, center, and right
#         left_boundary = frame_width // 3
#         right_boundary = 2 * frame_width // 3
#
#         # Perform detection
#         results = model(frame)
#
#         # Annotate frame with detections
#         annotated_frame = frame.copy()
#         instructions = []
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 object_center_x = (x1 + x2) // 2
#
#                 # Draw bounding box and label
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label = f"Obstacle ({x1},{y1})"
#                 cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#                 # Determine region and give instructions
#                 if object_center_x < left_boundary:
#                     instructions.append("Obstacle on the left, move to the center or right.")
#                     play_sound("left")
#                 elif object_center_x > right_boundary:
#                     instructions.append("Obstacle on the right, move to the center or left.")
#                     play_sound("right")
#                 else:
#                     instructions.append("Obstacle in the center, avoid or move left/right.")
#                     play_sound("center")
#
#         # Draw region divisions on the frame
#         cv2.line(annotated_frame, (left_boundary, 0), (left_boundary, frame_height), (255, 0, 0), 2)
#         cv2.line(annotated_frame, (right_boundary, 0), (right_boundary, frame_height), (255, 0, 0), 2)
#
#         # Display instructions on the terminal
#         if instructions:
#             print("\n".join(set(instructions)))
#
#         # Show the frame with detections
#         cv2.imshow("Object Detection for Navigation", annotated_frame)
#
#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting...")
#             break
#
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Camera closed.")
#
#
# if __name__ == "__main__":
#     detect_obstacles()
