import cv2
import time
import numpy as np
import mediapipe as mp
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Constants
wCam, hCam = 640, 480
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
vol = 0
volBar = 300
volPer = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

with mp_hand.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        lmList = []
        xList = []
        yList = []
        bbox = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                myHands = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHands.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(image, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (255, 0, 255), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)
                print(length)

                # Hand Range 10-200
                # Volume Range -95 - 0

                vol = np.interp(length, [10, 180], [minVol, maxVol])
                volBar = np.interp(length, [10, 180], [400, 150])
                volPer = np.interp(length, [10, 180], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)

                if length < 20:
                    cv2.circle(image, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (0, 255, 0), cv2.FILLED)

                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        cv2.rectangle(image, (50, 150), (85, 400), (255, 255, 0), 3)
        cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(image, f'FPS: {int(fps)}', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Webcam", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
