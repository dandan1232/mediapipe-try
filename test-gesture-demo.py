#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Recognize gestures from local video file (default: video/test.mp4).

Example:
  python3 test-gesture-demo.py --source video/test.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp


def build_recognizer(model_path: Path):
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def build_label(result, hand_index: int) -> str:
    handed = "Unknown"
    gesture = "None"
    score = 0.0

    if hand_index < len(result.handedness) and result.handedness[hand_index]:
        handed = result.handedness[hand_index][0].category_name or handed
    if hand_index < len(result.gestures) and result.gestures[hand_index]:
        top = result.gestures[hand_index][0]
        gesture = top.category_name or gesture
        score = top.score or 0.0

    return f"{handed}: {gesture} ({score:.2f})"


def run_video(video_path: Path, model_path: Path, no_display: bool):
    from mediapipe.tasks.python import vision

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    recognizer = build_recognizer(model_path)
    drawing_utils = vision.drawing_utils
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(frame_idx * 1000.0 / fps)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            y = 35
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                drawing_utils.draw_landmarks(frame, hand_landmarks, hand_connections)
                label = build_label(result, i)
                cv2.putText(
                    frame,
                    label,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                y += 35

            if not no_display:
                cv2.imshow("Gesture Video Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        recognizer.close()
        cv2.destroyAllWindows()


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Gesture recognizer on local video")
    parser.add_argument(
        "--source",
        default=str(root / "video" / "test.mp4"),
        help="Path to local video file",
    )
    parser.add_argument(
        "--model",
        default=str(root / "gesture_recognizer.task"),
        help="Path to gesture_recognizer.task",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2 window (for headless environment)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_video(
        video_path=Path(args.source),
        model_path=Path(args.model),
        no_display=args.no_display,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[test-gesture-demo] {exc}", file=sys.stderr)
        sys.exit(1)
