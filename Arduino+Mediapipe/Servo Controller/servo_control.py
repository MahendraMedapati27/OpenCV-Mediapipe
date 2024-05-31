import cv2
import serial
import mediapipe as mp
import math
import  numpy as np
import time
import pyfirmata

#--------------

board  = pyfirmata.Arduino('COM7')
servo = board.get_pin('d:9:s')
  

mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands  = mp.solutions.hands



# Webcam Setup
wCam, hCam = 640, 480
cam  = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)  as hands:

  while cam.isOpened():
    success, image = cam.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

    # multi_hand_landmarks method for Finding postion of Hand  landmarks      
    lmList = []
    if results.multi_hand_landmarks:
      myHand  = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])          

    # Assigning variables for  Thumb and Index finger position
    if len(lmList) != 0:
      x1, y1 = lmList[4][1],  lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]

      # Marking  Thumb and Index finger
      cv2.circle(image, (x1,y1),15,(255,255,255))  
      cv2.circle(image, (x2,y2),15,(255,255,255))   
      cv2.line(image,(x1,y1),(x2,y2),(255,0,0),3)
      length = math.hypot(x2-x1,y2-y1)
      if length < 50:
        cv2.line(image,(x1,y1),(x2,y2),(0,0,0),3)

      
      Dis = np.interp(length, [20, 220], [0, 180])
      Distance=  (round(Dis))
      #print(Posgripper)
      converted_Posgripper = str(Distance)
      cv2.putText(image, f'Servo Angle : {str(Distance)}', (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1,  (255, 0, 0), 3)
      #cv2.line(image, 320, 320, (0,0,0), 2)
      Servopos=(Distance)
      print (Servopos)
      servo.write(Servopos) 

    
    cv2.imshow('Servo Control',  image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cam.release()