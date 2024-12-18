import cv2
import pyautogui
import time
from Hand_Tracking_Module_ import handDetector

# Thresholds and parameters
SWIPE_THRESHOLD = 100  # Minimum horizontal swipe distance
PAUSE_PLAY_THRESHOLD = 30  # Minimum vertical hand raise distance
COOLDOWN_TIME = 0.5  # Cooldown between gestures in seconds

# Initialize timers
last_action_time = 0

def detect_gestures(lmList, current_time, last_action_time, hand_raised):
    """Detect gestures for media control."""
    global SWIPE_THRESHOLD, PAUSE_PLAY_THRESHOLD, COOLDOWN_TIME
    action = None

    # Positions of hand landmarks
    wrist = lmList[0][1:]  # Wrist coordinates
    index_tip = lmList[8][1:]  # Index finger tip

    # Detect horizontal swipes
    dx = index_tip[0] - wrist[0]  # Horizontal movement
    dy = index_tip[1] - wrist[1]  # Vertical movement

    # Gesture: Swipe Right (Next Track)
    if dx > SWIPE_THRESHOLD and current_time - last_action_time > COOLDOWN_TIME:
        pyautogui.hotkey('ctrl', 'right')  # Next track
        action = "Next Track"
        last_action_time = current_time

    # Gesture: Swipe Left (Previous Track)
    elif dx < -SWIPE_THRESHOLD and current_time - last_action_time > COOLDOWN_TIME:
        pyautogui.hotkey('ctrl', 'left')  # Previous track
        action = "Previous Track"
        last_action_time = current_time

    # Gesture: Raise Hand (Pause/Play)
    elif dy < -PAUSE_PLAY_THRESHOLD and not hand_raised and current_time - last_action_time > COOLDOWN_TIME:
        pyautogui.press('space')  # Pause/Play
        action = "Pause/Play"
        hand_raised = True
        last_action_time = current_time

    # Reset hand raised state
    elif dy > 0:
        hand_raised = False

    return action, last_action_time, hand_raised

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)  # Frame width
    cap.set(4, 480)  # Frame height

    detector = handDetector(detectionCon=0.8)
    global last_action_time
    hand_raised = False

    prev_time = 0  # For FPS calculation

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect hands
        frame = detector.findHands(frame)
        lmList = detector.findPositon(frame)

        current_time = time.time()

        action = None
        if lmList:
            # Detect gestures
            action, last_action_time, hand_raised = detect_gestures(lmList, current_time, last_action_time, hand_raised)

        # FPS Calculation
        fps = int(1 / (current_time - prev_time)) if prev_time else 0
        prev_time = current_time

        # Display gesture feedback
        if action:
            cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display cooldown feedback
        elif current_time - last_action_time < COOLDOWN_TIME:
            remaining_time = COOLDOWN_TIME - (current_time - last_action_time)
            cv2.putText(frame, f"Cooldown: {remaining_time:.2f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show video feed
        cv2.imshow("Gesture-Controlled Media Player", frame)

        # Quit program
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exiting Program")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
