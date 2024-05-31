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

    def get_langdmark_list(self, hand_landmarks):
        """
        Returns a list of landmarks in the format (x, y) for the detected hand.
        :param hand_landmarks: The landmarks of the detected hand.
        :return: A list of (x, y) tuples representing the landmarks.
        """
        lmlist = []
        for id, lm in enumerate(hand_landmarks.landmark):
            # Convert normalized coordinates to pixel values
            height, width, _ = self.image.shape
            cx, cy = int(lm.x * width), int(lm.y * height)
            lmlist.append((cx, cy))
        return lmlist

    def detect_hand_pose(self):
        while True:
            ret, image = self.video.read()
            if not ret:
                break

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # Convert the BGR image to RGB
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image to detect hands
            results = self.hands.process(self.image)

            # Draw hand landmarks on the image and get landmarks list
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hand.HAND_CONNECTIONS, 
                                                   landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                    
                    # Get the landmarks list for the detected hand
                    lmlist = self.get_landmark_list(hand_landmarks)
                    print(lmlist)  # Print the landmarks list to the console (or you can process it further)

            # Display the image with hand landmarks
            cv2.imshow("Hand Pose Detection", image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    detector = HandPoseDetector()
    detector.detect_hand_pose()


