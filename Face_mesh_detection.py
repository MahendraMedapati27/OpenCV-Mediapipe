import os
import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self, num_faces=1):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.num_faces = num_faces
        self.video = cv2.VideoCapture(0)

    def detect_face_mesh(self):
        with self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=self.num_faces, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while True:
                ret, image = self.video.read()
                if not ret:
                    break
                
                # Convert the BGR image to RGB before processing
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image to detect face mesh
                image_rgb = np.copy(image_rgb)  # Create a writable copy of the image
                result = face_mesh.process(image_rgb)
                
                # Draw the face mesh annotations on the image
                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                        )
                
                # Display the result
                cv2.imshow("Face Mesh", image)
                
                # Break the loop when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture object and close all OpenCV windows
            self.video.release()
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    num_faces_to_detect = 2  # Change this value to the desired number of faces to detect
    detector = FaceMeshDetector(num_faces=num_faces_to_detect)
    detector.detect_face_mesh()

