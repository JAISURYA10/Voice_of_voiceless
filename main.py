import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/Nice"
counter = 0


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 225
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            hCal = imgSize
            wCal = int(w * k)
            wGap = math.ceil((imgSize - wCal)/2)
            imgResize = cv2.resize(imgCrop, (wCal, hCal))
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            wCal = imgSize
            hCal = int(h * k)
            hGap = math.ceil((imgSize - hCal)/2)
            imgResize = cv2.resize(imgCrop, (wCal, hCal))
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow("Imagecrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

