import cv2
import os
import glob

PADDING = 50

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for file in glob.glob("images/*"):
        img  = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            x1 = x-PADDING
            y1 = y-PADDING
            x2 = x+w+PADDING
            y2 = y+h+PADDING

            height, width, channels = img.shape
            # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
            part_image = img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            key = cv2.waitKey(100)
            cv2.imwrite(file + '_face', part_image)
            #cv2.imshow('face', part_image)
