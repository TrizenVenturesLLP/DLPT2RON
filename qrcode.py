import cv2
import json

# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Initialize the QRCodeDetector
detector = cv2.QRCodeDetector()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Failed to grab frame")
        break

    # Detect and decode QR code in the frame
    retval, decoded_info, points = detector.detectAndDecode(frame)

    if retval:
        # Display the decoded content on the frame
        print("Decoded Data:", decoded_info)

        # Convert points to integers (as cv2.line expects int coordinates)
        if points is not None:
            points = points[0].astype(int)
            for i in range(len(points)):
                cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 5)

    # Display the resulting frame
    cv2.imshow("QR Code Scanner", frame)

    # Break the loop if the user presses the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value for 'ESC'
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
