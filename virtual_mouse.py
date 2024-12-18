import cv2
import pyautogui
import time
import numpy as np
from Hand_Tracking_Module_ import handDetector

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam dimensions
frame_width, frame_height = 640, 480

# Thresholds
LEFT_CLICK_THRESHOLD = 30
RIGHT_CLICK_THRESHOLD = 30
SMOOTHING = 7  # Adjust for smoother movement
CLICK_COOLDOWN = 1  # Time in seconds to prevent multiple clicks

# Initialize click timers
last_left_click = 0
last_right_click = 0

def map_coordinates(x, y):
    """Map webcam coordinates to inverted screen coordinates."""
    x_screen = np.interp(x, [0, frame_width], [screen_width, 0])  # Invert X-axis
    y_screen = np.interp(y, [0, frame_height], [screen_height, 0])  # Invert Y-axis
    return x_screen, y_screen

def detect_click(distance, threshold, last_click_time, action):
    """Detect a click gesture and trigger the action if in cooldown."""
    current_time = time.time()
    if distance < threshold and (current_time - last_click_time) > CLICK_COOLDOWN:
        action()
        return current_time
    return last_click_time

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    detector = handDetector(detectionCon=0.8)
    prev_x, prev_y = 0, 0
    global last_left_click, last_right_click

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect hands and landmarks
        frame = detector.findHands(frame)
        lmList = detector.findPositon(frame)

        if lmList:
            # Fingertip positions
            index_tip = lmList[8][1:]  # Index finger tip
            middle_tip = lmList[12][1:]  # Middle finger tip
            thumb_tip = lmList[4][1:]  # Thumb tip

            # Map coordinates to inverted screen space
            curr_x, curr_y = map_coordinates(index_tip[0], index_tip[1])

            # Smooth cursor movement
            smoothed_x = prev_x + (curr_x - prev_x) / SMOOTHING
            smoothed_y = prev_y + (curr_y - prev_y) / SMOOTHING

            # Move the cursor
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y

            # Gesture: Left Click (Index & Thumb Close)
            distance_thumb_index = ((thumb_tip[0] - index_tip[0]) ** 2 +
                                     (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
            last_left_click = detect_click(
                distance_thumb_index, LEFT_CLICK_THRESHOLD, last_left_click, pyautogui.click
            )
            if distance_thumb_index < LEFT_CLICK_THRESHOLD:
                cv2.putText(frame, "Left Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gesture: Right Click (Index & Middle Close)
            distance_index_middle = ((index_tip[0] - middle_tip[0]) ** 2 +
                                      (index_tip[1] - middle_tip[1]) ** 2) ** 0.5
            last_right_click = detect_click(
                distance_index_middle, RIGHT_CLICK_THRESHOLD, last_right_click, pyautogui.rightClick
            )
            if distance_index_middle < RIGHT_CLICK_THRESHOLD:
                cv2.putText(frame, "Right Click", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display FPS
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(frame, f"FPS: {fps}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show video feed
        cv2.imshow("Virtual Mouse (Inverted)", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
