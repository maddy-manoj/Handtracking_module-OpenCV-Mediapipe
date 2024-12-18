
import cv2
import mediapipe as mp
import time

class handDetector():
    ################ Initialization ####################
    def __init__(self, mode = False, maxHands = 2, modelComp = 1,
                    detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # these are the formalities for Hand Landmarks module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.modelComp, self.detectionCon,
                                        self.trackCon)  # hands get detected
        self.mpDraw = mp.solutions.drawing_utils  # to draw the points

    ####### Detection Part ############
    def findHands(self, frame):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLms,
                                           self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPositon(self, frame, handNo=0):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # enumerate returns both id and landmarks
                # print(id)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
        return lmList

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()
    while True:
        # read the frame from camera source
        success, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPositon(frame)

        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            break

if __name__ == "__main__":
    main()