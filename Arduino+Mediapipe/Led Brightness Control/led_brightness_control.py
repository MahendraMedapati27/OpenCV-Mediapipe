import cv2
import mediapipe as mp
import numpy as np
import math
import serial

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize serial connection (adjust port according to your system)
comport = 'COM6'
try:
    ser = serial.Serial(comport, baudrate=9600, timeout=1)  # Change to your serial port and baudrate
except serial.SerialException as e:
    print("Error opening serial port:", e)
    exit()

# Function to map distance to brightness
def map_value(value, left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range
    return int(right_min + (value_scaled * right_span))

# Function to calculate LED brightness
def calculate_brightness(distance):
    # Map the distance to LED brightness (adjust range according to your hand gesture)
    brightness = map_value(distance, 20, 300, 0, 255)
    return brightness

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)
    lmList = []

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            myHands = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate the distance between two points (x1, y1) and (x2, y2)
            distance = math.hypot(x2 - x1, y2 - y1)

            # Calculate the LED brightness
            brightness = calculate_brightness(distance)
            
            # Send the brightness value to Arduino
            try:
                ser.write(f'{brightness}\n'.encode())
            except serial.SerialTimeoutException:
                print("Serial write timeout occurred")

    # Display the image
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
