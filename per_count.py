######with voice included 
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define COCO class for 'person'
PERSON_CLASS_ID = 1  # Class ID for 'person' in COCO dataset

# Function to announce the total count
def announce_count(count):
    if count > 0:
        message = f"There are {count} person{'s' if count > 1 else ''} detected."
    else:
        message = "No persons detected."
    engine.say(message)
    engine.runAndWait()

# Function to detect and display persons with labels and counts
def detect_persons(frame, prev_count):
    # Convert the frame to tensor
    transform = transforms.ToTensor()
    frame_tensor = transform(frame).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        predictions = model(frame_tensor)[0]

    # Extract bounding boxes, labels, and scores
    boxes = predictions['boxes'].numpy()
    labels = predictions['labels'].numpy()
    scores = predictions['scores'].numpy()

    # Filter for persons with a confidence threshold
    person_count = 0
    for i, label in enumerate(labels):
        if label == PERSON_CLASS_ID and scores[i] > 0.6:  # Detect only persons
            person_count += 1
            box = boxes[i].astype(int)
            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Add label above the box
            label_text = f"Person {person_count}"
            cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display total person count on the frame
    cv2.putText(frame, f"Total Persons: {person_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Announce the count only if it changes
    if person_count != prev_count:
        announce_count(person_count)

    return frame, person_count

# Access the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("src/videos/test_video2.mp4")

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

prev_count = -1  # Initialize the previous count as -1 to ensure the first count is announced

# Real-time video loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect persons and display labels and counts
    frame, prev_count = detect_persons(frame, prev_count)

    # Display the frame
    cv2.imshow("Person Detector", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
