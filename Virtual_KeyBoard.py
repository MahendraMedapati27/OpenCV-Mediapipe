import cv2
import HandMarkDetection2
import time
import math

# Initialize the video capture and set the resolution
video = cv2.VideoCapture(0)
video.set(3, 1280)  # Width
video.set(4, 720)   # Height

# Initialize the hand pose detector
detector = HandMarkDetection2.HandPoseDetector()

class Button:
    def __init__(self, pos, text, size=[40, 40]):
        self.pos = pos
        self.text = text
        self.size = size
        self.hovered = False  # Track if the button is hovered

    def draw(self, image):
        # Draw the rectangle (button)
        color = (0, 255, 0) if self.hovered else (255, 0, 255)
        cv2.rectangle(image, tuple(self.pos), (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), color, cv2.FILLED)
        # Calculate the position to place the text so it's centered in the rectangle
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_PLAIN, 2, 3)[0]
        text_x = self.pos[0] + (self.size[0] - text_size[0]) // 2
        text_y = self.pos[1] + (self.size[1] + text_size[1]) // 2
        cv2.putText(image, self.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    def is_over(self, point):
        # Check if a point is over the button
        if self.pos[0] < point[0] < self.pos[0] + self.size[0] and self.pos[1] < point[1] < self.pos[1] + self.size[1]:
            return True
        return False

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Create buttons for a simple keyboard layout
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
]

buttons = []
start_x, start_y = 30, 30
gap = 10
button_size = [40, 40]

for row_index, row in enumerate(keys):
    for col_index, key in enumerate(row):
        pos_x = start_x + col_index * (button_size[0] + gap)
        pos_y = start_y + row_index * (button_size[1] + gap)
        buttons.append(Button([pos_x, pos_y], key, button_size))

# Create a button for the backward arrow
backward_arrow = Button([start_x + len(keys[0]) * (button_size[0] + gap) + 50, start_y], "<-", [70, 40])

notepad = []  # List to store clicked buttons
prev_tip_distance = 0  # Distance between index finger tips in the previous frame
click_threshold = 30  # Threshold distance to register a click

while True:
    ret, image = video.read()  # Read a frame from the video
    if not ret:
        break  # Break the loop if no frame is captured

    # Detect hand pose
    hand_info, image = detector.detect_hand_pose(image)

    # Check if any hands are detected
    if hand_info:
        # Unpack the first detected hand's landmarks and bounding box
        lmlist, bbox = hand_info[0]
        # Get the positions of the index finger tip (landmark 8)
        index_tip = lmlist[8]

        # Draw a rectangle around the index finger tip for debugging
        cv2.rectangle(image, (index_tip[0] - 10, index_tip[1] - 10), (index_tip[0] + 10, index_tip[1] + 10), (0, 255, 0), cv2.FILLED)

        # Check if the index finger tip is over any button
        for button in buttons:
            if button.is_over(index_tip):
                button.hovered = True
                if button.text not in notepad:
                    notepad.append(button.text)
            else:
                button.hovered = False

        # Check if the backward arrow is clicked
        if backward_arrow.is_over(index_tip):
            # Remove the last clicked button from the notepad
            if notepad:
                notepad.pop()

    # Draw all buttons on the image
    for button in buttons:
        button.draw(image)

    # Draw the backward arrow button
    backward_arrow.draw(image)

    # Draw rectangular box as background for notepad
    notepad_box_start_x = start_x
    notepad_box_start_y = start_y + len(keys) * (button_size[1] + gap) + 30
    notepad_box_end_x = start_x + len(keys[0]) * (button_size[0] + gap) + 80
    notepad_box_end_y = notepad_box_start_y + 60
    cv2.rectangle(image, (notepad_box_start_x, notepad_box_start_y), (notepad_box_end_x, notepad_box_end_y), (255, 255, 255), cv2.FILLED)

    # Display the notepad at the bottom of the screen
    notepad_text = " ".join(notepad)
    cv2.putText(image, notepad_text, (notepad_box_start_x + 10, notepad_box_start_y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    # Display the resulting image
    cv2.imshow("Hand Pose Detection", image)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
