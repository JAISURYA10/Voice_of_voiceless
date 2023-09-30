import streamlit as st
import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import pygame
import math

pygame.mixer.init()

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

st.title("Hand Gesture Recognition with Streamlit")

uploaded_file = st.file_uploader("voice/sample.mp4", type=["mp4", "avi"])

if uploaded_file is not None:
    cap = cv2.VideoCapture("voice/sample.mp4")
    detector = HandDetector(maxHands=1)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            text = "Name"  
            audio_files = {
                "Hi": "hello.mp3",
                "Name": "name.mp3",
                "Nice": "nice.mp3"
            }

            if text in audio_files:
                pygame.mixer.music.load(audio_files[text])
                pygame.mixer.music.play()

        st.image(img, channels="BGR")

    cap.release()

st.sidebar.markdown("### Instructions")
st.sidebar.write("1. Upload a video file (MP4 or AVI format).")
st.sidebar.write("2. The application will process the video frame by frame.")
st.sidebar.write("3. Hand gestures will be recognized and audio feedback will be provided.")