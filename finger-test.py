#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project    :mediapipe-try 
@File       :finger-test.py
@IDE        :PyCharm 
@Author     :Echo_Lin
@Date       :2026/2/27 10:44
@Description: 
'''
import cv2
import os
import sys
import time
from pathlib import Path

import mediapipe as mp

GESTURE = ["none", "thumb_up"]


def is_thumb_up(landmarks):
    # 规则：拇指(5->4)张开距离大于 base*0.3，且其余四指都不超过base
    p0_x = landmarks[0].x
    p0_y = landmarks[0].y
    p5_x = landmarks[5].x
    p5_y = landmarks[5].y
    distance_0_5 = pow(p0_x - p5_x, 2) + pow(p0_y - p5_y, 2)
    base = distance_0_5 / 0.6

    p4_x = landmarks[4].x
    p4_y = landmarks[4].y
    distance_5_4 = pow(p5_x - p4_x, 2) + pow(p5_y - p4_y, 2)

    p8_x = landmarks[8].x
    p8_y = landmarks[8].y
    distance_0_8 = pow(p0_x - p8_x, 2) + pow(p0_y - p8_y, 2)

    p12_x = landmarks[12].x
    p12_y = landmarks[12].y
    distance_0_12 = pow(p0_x - p12_x, 2) + pow(p0_y - p12_y, 2)

    p16_x = landmarks[16].x
    p16_y = landmarks[16].y
    distance_0_16 = pow(p0_x - p16_x, 2) + pow(p0_y - p16_y, 2)

    p20_x = landmarks[20].x
    p20_y = landmarks[20].y
    distance_0_20 = pow(p0_x - p20_x, 2) + pow(p0_y - p20_y, 2)

    thumb_extended = distance_5_4 > base * 0.3
    other_folded = (
        distance_0_8 <= base and
        distance_0_12 <= base and
        distance_0_16 <= base and
        distance_0_20 <= base
    )
    return thumb_extended and other_folded


def run_with_legacy_solutions():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("video/test.mp4")

    try:
        while True:
            thumb_up_detected = False
            ret, frame = cap.read()
            if not ret:
                break

            # 因为摄像头是镜像的，所以将摄像头水平翻转
            # 不是镜像的可以不翻转
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_thumb_up(hand_landmarks.landmark):
                        thumb_up_detected = True
                    # 关键点可视化
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )
            label = GESTURE[1 if thumb_up_detected else 0]
            cv2.putText(frame, label, (50, 50), 0, 1.3, (0, 0, 255), 3)
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


def run_with_tasks_api():
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    model_download_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    default_model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
    model_path = Path(os.environ.get("MP_HAND_MODEL", default_model_path))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Cannot find hand model: {model_path}\n"
            f"Download model: {model_download_url}\n"
            f"Example: wget -O \"{default_model_path}\" \"{model_download_url}\"\n"
            "Then place `hand_landmarker.task` next to this script,\n"
            "or set environment variable MP_HAND_MODEL to the model path."
        )

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.75,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    mp_drawing = vision.drawing_utils
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("video/test.mp4")
    last_timestamp_ms = -1
    try:
        while True:
            thumb_up_detected = False
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.perf_counter() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    if is_thumb_up(hand_landmarks):
                        thumb_up_detected = True
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hand_connections,
                    )
            label = GESTURE[1 if thumb_up_detected else 0]
            cv2.putText(frame, label, (50, 50), 0, 1.3, (0, 0, 255), 3)
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        if hasattr(mp, "solutions"):
            run_with_legacy_solutions()
        else:
            run_with_tasks_api()
    except Exception as exc:
        print(f"[finger-test] {exc}", file=sys.stderr)
        sys.exit(1)
