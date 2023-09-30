from flask import Flask, render_template, Response
import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import pygame
import math

pygame.mixer.init()

audio_files = {
    "Bye": "Bye.mp3",
    "Fine": "Fine.mp3",
    "Hi": "Hi.mp3",
    "Nice": "Nice.mp3",
    "Thank_you": "Thank_you.mp3"
}

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
Classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["Bye", "Fine", "Hi", "Nice", "Thank_you"]

app = Flask(__name__)

def generate_frames():
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
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
                wGap = math.ceil((imgSize - wCal) / 2)
                imgResize = cv2.resize(imgCrop, (wCal, hCal))
                imgWhite[:, wGap:wGap + wCal] = imgResize
                prediction, index = Classifier.getPrediction(imgWhite, draw=False)

            else:
                k = imgSize / w
                wCal = imgSize
                hCal = int(h * k)
                hGap = math.ceil((imgSize - hCal) / 2)
                imgResize = cv2.resize(imgCrop, (wCal, hCal))
                imgWhite[hGap:hGap + hCal, :] = imgResize
                prediction, index = Classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset + 90, y - offset - 50), (x - offset + 150, y - offset + 50 - 50),
                          (255, 0, 0), cv2.FILLED)

            text = labels[index]
            textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
            textX = x - offset + 90 + (30 - textSize[0] // 2)
            textY = y - 30
            cv2.putText(imgOutput, text, (textX, textY), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 4)

            if text in audio_files:
                pygame.mixer.music.load(audio_files[text])
                pygame.mixer.music.play()

            _, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
