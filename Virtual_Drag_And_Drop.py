import cv2
import HandMarkDetection2
import math
import numpy as np

video = cv2.VideoCapture(0)
video.set(3, 1280)
video.set(4, 720)
detector = HandMarkDetection2.HandPoseDetector()

class DragRec():
    def __init__(self, posCenter, size=[70, 70], colorRec=(255, 0, 255)):
        self.posCenter = posCenter
        self.size = size
        self.colorRec = colorRec
        
    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.colorRec = (0, 255, 255)  # Change color when cursor is within the rectangle
            self.posCenter = cursor
        else:
            self.colorRec = (255, 0, 255)  # Revert color when cursor is outside the rectangle


rectList = []
for x in range(8):
    rectList.append(DragRec(posCenter=[x*75+65, 65]))

while True:
    ret, image = video.read()
    if not ret:
        break  # Break if the video frame is not read properly

    hand_info, image = detector.detect_hand_pose(image)
    
    def get_lmlist(info):
        for lmlist, _ in info:
            return lmlist
        return []  # Return empty list if no landmarks found

    lmlist = get_lmlist(hand_info)
    if lmlist and len(lmlist) > 12:
        x1, y1 = lmlist[8][0], lmlist[8][1]
        x2, y2 = lmlist[12][0], lmlist[12][1]
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance < 35:
            cursor = lmlist[8]
            for rect in rectList:
                rect.update(cursor)

    # Create a new image for drawing rectangles with transparency
    imgNew = np.zeros_like(image, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        colorRec = rect.colorRec  # Assign the rectangle's color
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorRec, cv2.FILLED)

    # Blend the new image with the original image
    alpha = 0.4  # Adjust transparency level as needed
    out = cv2.addWeighted(image, 1-alpha, imgNew, alpha, 0)

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
