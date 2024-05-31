import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def get_hand_label(landmarks):
        wrist = landmarks[0]
        thumb_base = landmarks[1]
        if thumb_base.x < wrist.x:
            return "Right Hand"
        else:
            return "Left Hand"

    @staticmethod
    def calculate_bounding_box(landmarks, image_shape):
        h, w, _ = image_shape
        min_x = min([landmark.x for landmark in landmarks]) * w
        min_y = min([landmark.y for landmark in landmarks]) * h
        max_x = max([landmark.x for landmark in landmarks]) * w
        max_y = max([landmark.y for landmark in landmarks]) * h
        return int(min_x), int(min_y), int(max_x), int(max_y)

    def process_frame(self, frame):
        # Flip the frame horizontally for a later selfie-view display
        image = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                # Determine if the hand is right or left
                label = self.get_hand_label(hand_landmarks.landmark)

                # Calculate the bounding box of the hand
                min_x, min_y, max_x, max_y = self.calculate_bounding_box(hand_landmarks.landmark, image.shape)

                # Draw the bounding box
                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

                # Display the label on top of the bounding box
                cv2.putText(image, label, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return image

    def run(self):
        # Start video capture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            image = self.process_frame(frame)

            # Display the frame
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Create a HandTracker object and run it if the script is run directly
if __name__ == "__main__":
    hand_tracker = HandTracker()
    hand_tracker.run()
