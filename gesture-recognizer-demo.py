#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""MediaPipe GestureRecognizer demo.

Examples:
  1) Image mode
     python gesture-recognizer-demo.py --image image/test1.png --no-display

  2) Video file mode
     python gesture-recognizer-demo.py --source test.mp4

  3) Webcam mode
     python gesture-recognizer-demo.py --source 0
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


def parse_source(source_text: str):
    if source_text.isdigit():
        return int(source_text)
    return source_text


def build_recognizer(model_path: Path, running_mode, num_hands: int, score_threshold: float):
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.components.processors.classifier_options import (
        ClassifierOptions,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=ClassifierOptions(
            score_threshold=score_threshold,
        ),
    )
    return vision.GestureRecognizer.create_from_options(options)


def hand_label(result, hand_index: int) -> str:
    handed = "Unknown"
    gesture = "None"
    score = 0.0

    if hand_index < len(result.handedness) and result.handedness[hand_index]:
        top_handed = result.handedness[hand_index][0]
        handed = top_handed.category_name or handed

    if hand_index < len(result.gestures) and result.gestures[hand_index]:
        top_gesture = result.gestures[hand_index][0]
        gesture = top_gesture.category_name or gesture
        score = top_gesture.score or 0.0

    return f"{handed}: {gesture} ({score:.2f})"


def draw_result(frame, result, drawing_utils, hand_connections):
    labels = []
    for i, landmarks in enumerate(result.hand_landmarks):
        label = hand_label(result, i)
        labels.append(label)
        drawing_utils.draw_landmarks(frame, landmarks, hand_connections)
        cv2.putText(
            frame,
            label,
            (20, 40 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    return labels


def run_image_mode(args):
    from mediapipe.tasks.python import vision

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    recognizer = build_recognizer(
        model_path=Path(args.model),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=args.num_hands,
        score_threshold=args.score_threshold,
    )
    drawing_utils = vision.drawing_utils
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = recognizer.recognize(mp_image)

        labels = draw_result(frame, result, drawing_utils, hand_connections)
        if labels:
            print("Detected gestures:")
            for text in labels:
                print(f"  - {text}")
        else:
            print("No hand/gesture detected.")

        if args.output:
            output_path = Path(args.output)
            cv2.imwrite(str(output_path), frame)
            print(f"Saved output image: {output_path}")

        if not args.no_display:
            cv2.imshow("GestureRecognizer Demo", frame)
            cv2.waitKey(0)
    finally:
        recognizer.close()
        cv2.destroyAllWindows()


def run_video_mode(args):
    from mediapipe.tasks.python import vision

    source = parse_source(args.source)
    recognizer = build_recognizer(
        model_path=Path(args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        score_threshold=args.score_threshold,
    )
    drawing_utils = vision.drawing_utils
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        recognizer.close()
        raise RuntimeError(f"Cannot open source: {source}")

    last_timestamp_ms = -1
    last_labels = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.perf_counter() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            labels = draw_result(frame, result, drawing_utils, hand_connections)

            if labels != last_labels:
                printable = ", ".join(labels) if labels else "None"
                print(f"frame={frame_index}: {printable}")
                last_labels = labels

            if not args.no_display:
                cv2.imshow("GestureRecognizer Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        recognizer.close()
        cv2.destroyAllWindows()


def build_parser():
    default_model = Path(__file__).resolve().parent / "gesture_recognizer.task"
    default_source = Path(__file__).resolve().parent / "test.mp4"
    source_value = str(default_source) if default_source.exists() else "0"

    parser = argparse.ArgumentParser(description="MediaPipe GestureRecognizer demo")
    parser.add_argument(
        "--model",
        default=str(default_model),
        help="Path to gesture_recognizer.task",
    )
    parser.add_argument(
        "--source",
        default=source_value,
        help="Video source for video mode: camera index like 0, or video file path",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Run in image mode with a single image path",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output image path (image mode only)",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Max number of hands",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Min score for canned gesture classifier",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow window (useful in headless env)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Cannot find model: {model_path}\n"
            f"Download from: {MODEL_DOWNLOAD_URL}\n"
            f"Example:\n"
            f"  wget -O \"{model_path}\" \"{MODEL_DOWNLOAD_URL}\""
        )

    if args.image:
        run_image_mode(args)
    else:
        run_video_mode(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[gesture-demo] {exc}", file=sys.stderr)
        sys.exit(1)
