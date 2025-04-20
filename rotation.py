import cv2
import numpy as np
import math

def correct_alignment_handwritten(frame):
    """Attempts to correct horizontal alignment based on visual cues for handwritten text."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -60 < angle < 60:
                angles.append(angle)

    if angles:
        weighted_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if -60 < angle < 60:
                weighted_angles.extend([angle] * int(length / 10))

        if weighted_angles:
            median_angle = np.median(weighted_angles)
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            return rotated_frame
        else:
            return frame
    else:
        return frame

# Capture video from the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

capture_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    aligned_frame = correct_alignment_handwritten(frame)

    cv2.imshow("Webcam Feed (Aligned - Handwritten)", aligned_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        capture_count += 1
        filename = f"captured_image_{capture_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image captured and saved as {filename}")

cap.release()
cv2.destroyAllWindows()