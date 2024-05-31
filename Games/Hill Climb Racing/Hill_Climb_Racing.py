import cv2
import mediapipe as mp
import time
from directkeys import right_pressed, left_pressed
from directkeys import PressKey, ReleaseKey  

# Constants for key presses
BREAK_KEY = left_pressed
ACCELERATOR_KEY = right_pressed

# Delay to switch to camera view
time.sleep(2.0)

# Set to keep track of currently pressed keys
current_key_pressed = set()

# MediaPipe setup
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Tip IDs for fingertips
tip_ids = [4, 8, 12, 16, 20]

# Start video capture
video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = video.read()
        if not ret:
            continue
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and get hand landmarks
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        lm_list = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                
                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3)
                )

        fingers = []
        if len(lm_list) != 0:
            # Thumb
            if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Fingers
            for id in range(1, 5):
                if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            total_fingers = fingers.count(1)
            key_pressed = None
            
            if total_fingers == 0:
                cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                PressKey(BREAK_KEY)
                key_pressed = BREAK_KEY
            elif total_fingers == 5:
                cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, "GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                PressKey(ACCELERATOR_KEY)
                key_pressed = ACCELERATOR_KEY
            
            if key_pressed:
                current_key_pressed.add(key_pressed)
            else:
                for key in current_key_pressed:
                    ReleaseKey(key)
                current_key_pressed.clear()
                
            if len(current_key_pressed) > 1:
                for key in current_key_pressed:
                    if key != key_pressed:
                        ReleaseKey(key)
                current_key_pressed = {key_pressed}
        
        cv2.imshow("Frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
