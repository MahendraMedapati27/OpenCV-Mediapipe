import cv2 
import mediapipe as mp
import os

class HandPoseDetector:
    def __init__(self):
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.video = cv2.VideoCapture(0)

    def detect_hand_pose(self, image=None, draw=True):
        """
        Detects hands in the input image, returns landmarks and bounding boxes.
        :param image: The input image. If None, capture from the video feed.
        :param draw: Boolean flag to draw landmarks and bounding boxes on the image.
        :return: A tuple containing the landmark list, bounding box, and the image with drawings.
        """
        hands_info = []
        
        if image is None:
            ret, image = self.video.read()
            if not ret:
                return hands_info

            # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        results = self.hands.process(self.image)

        # Draw hand landmarks on the image and get landmarks list
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hand.HAND_CONNECTIONS, 
                                                   landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

                # Get the landmarks list and bounding box for the detected hand
                lmlist = []
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')
                for id, lm in enumerate(hand_landmarks.landmark):
                    height, width, _ = self.image.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    lmlist.append((cx, cy))
                    x_min, y_min = min(x_min, cx), min(y_min, cy)
                    x_max, y_max = max(x_max, cx), max(y_max, cy)

                bbox = (x_min, y_min, x_max, y_max)
                hands_info.append((lmlist, bbox))

                if draw:
                    # Draw the bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return hands_info, image

    def run(self):
        """
        Capture video from the webcam and detect hand poses.
        """
        while True:
            hands_info, image = self.detect_hand_pose()
            for lmlist, bbox in hands_info:
                print(f'Landmarks: {lmlist}')
                print(f'Bounding Box: {bbox}')

            cv2.imshow("Hand Pose Detection", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()
