import cv2
import numpy as np
import os
import time
import pyttsx3
from threading import Thread
from queue import Queue

#import translation_app  # Import the translation_app module

# Threshold to detect object
thres = 0.45
nms_threshold = 0.2

# Approximate focal length of the camera (adjust this value based on your camera)
focal_length = 615

# Load class names
classFile = 'src\dataset\coco.names'
if not os.path.exists(classFile):
    raise FileNotFoundError(f"{classFile} does not exist. Ensure the file is in the working directory.")

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load average sizes
average_sizes_file = 'src/dataset/average_sizes.txt'
if not os.path.exists(average_sizes_file):
    raise FileNotFoundError(f"{average_sizes_file} does not exist. Create the file with object sizes.")

average_sizes = {}
with open(average_sizes_file, 'rt') as f:
    for line in f:
        obj, size = line.strip().split(',')
        average_sizes[obj.strip()] = float(size.strip())

# Load model files
configPath = 'assests/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'assests/models/frozen_inference_graph.pb'

if not os.path.exists(configPath) or not os.path.exists(weightsPath):
    raise FileNotFoundError("Model config or weights file is missing. Check the paths.")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Queue and Thread for Voice Feedback
queue = Queue()

def speak(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 235)
    engine.setProperty('volume', 1.0)
    while True:
        if not q.empty():
            label, distance, position = q.get()
            rounded_distance = round(distance * 2) / 2
            rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
            engine.say(f"{label.upper()} is {rounded_distance_str} meters to your {position}")
            engine.runAndWait()
            with queue.mutex:
                queue.queue.clear()
        else:
            time.sleep(0.1)
            
t = Thread(target=speak, args=(queue,))
t.start()

# Calculate distance
def calculate_distance(object_width, real_width):
    return (real_width * focal_length) / (object_width + 1e-6)

# Get position of the object
def get_position(frame_width, box):
    if box[0] < frame_width // 3:
        return "LEFT"
    elif box[0] < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"

# Open webcam
cap = cv2.VideoCapture("src/videos/test_video2.mp4")
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break

    frame_width = img.shape[1]
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

            # Get class label
            class_id = classIds[i][0] if isinstance(classIds[i], np.ndarray) else classIds[i]
            label = classNames[class_id - 1].lower()

            # Check if the object has a known average size
            if label in average_sizes:
                distance = calculate_distance(w, average_sizes[label])
                position = get_position(frame_width, (x, y, x + w, y + h))

                text = f"{label.upper()} - {distance:.2f} m"
                queue.put((label, distance, position))  # Send to voice feedback
            else:
                text = f"{label.upper()} (size unknown)"

            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection with Distance and Voice Guide", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()


















