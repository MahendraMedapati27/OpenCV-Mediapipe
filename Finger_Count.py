import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

tipids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.7, max_num_hands=1, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmlist = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                myHands = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHands.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
                fingers = []
                if len(lmlist) != 0:
                    if lmlist[tipids[0]][1] > lmlist[tipids[0]-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    for id in range (1,5):
                        if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    total_count = fingers.count(1)
                    
                cv2.putText(image, f'No of Fingers : {total_count}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                
                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        cv2.imshow("Webcam", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()