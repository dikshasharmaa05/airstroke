from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


Point = Tuple[int, int]


@dataclass(frozen=True)
class HandState:
    """Screen-space hand state derived from MediaPipe landmarks."""

    landmarks: list[Point]
    index_tip: Point
    fingers_open: dict[str, bool]
    gesture: str

    @property
    def is_draw_gesture(self) -> bool:
        return self.gesture == "draw"

    @property
    def is_pause_gesture(self) -> bool:
        return self.gesture == "pause"

    @property
    def is_fist(self) -> bool:
        return self.gesture == "fist"


class HandTracker:
    """Thin wrapper around MediaPipe Hands with app-specific gesture labels."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

    TIP_IDS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }
    PIP_IDS = {
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        self._ensure_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._last_timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[Optional[HandState], np.ndarray]:
        """Return the primary hand state and an annotated frame."""

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._last_timestamp_ms += 1
        result = self._landmarker.detect_for_video(image, self._last_timestamp_ms)

        if not result.hand_landmarks:
            return None, frame_bgr

        hand_landmarks = result.hand_landmarks[0]
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in hand_landmarks
        ]
        self._draw_landmarks(frame_bgr, points)
        fingers_open = self._classify_fingers(points)
        gesture = self._classify_gesture(fingers_open)

        return (
            HandState(
                landmarks=points,
                index_tip=points[self.TIP_IDS["index"]],
                fingers_open=fingers_open,
                gesture=gesture,
            ),
            frame_bgr,
        )

    def close(self) -> None:
        self._landmarker.close()

    def _ensure_model(self) -> None:
        if self.MODEL_PATH.exists():
            return

        print("Downloading MediaPipe hand landmark model...")
        urlretrieve(self.MODEL_URL, self.MODEL_PATH)

    def _draw_landmarks(self, frame: np.ndarray, points: list[Point]) -> None:
        for connection in self._connections:
            cv2.line(
                frame,
                points[connection.start],
                points[connection.end],
                (70, 220, 255),
                2,
                cv2.LINE_AA,
            )

        for point in points:
            cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _classify_fingers(self, points: list[Point]) -> dict[str, bool]:
        fingers_open = {
            finger: points[self.TIP_IDS[finger]][1] < points[pip_id][1] - 12
            for finger, pip_id in self.PIP_IDS.items()
        }

        # Thumb direction varies with handedness and mirroring, so use distance
        # from the palm: an extended thumb sits farther from the wrist than its IP joint.
        wrist = np.array(points[0])
        thumb_tip = np.array(points[self.TIP_IDS["thumb"]])
        thumb_ip = np.array(points[3])
        fingers_open["thumb"] = (
            np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_ip - wrist) + 18
        )
        return fingers_open

    def _classify_gesture(self, fingers_open: dict[str, bool]) -> str:
        long_fingers = ("index", "middle", "ring", "pinky")

        if all(fingers_open[finger] for finger in long_fingers):
            return "pause"

        if not any(fingers_open[finger] for finger in long_fingers):
            return "fist"

        if fingers_open["index"] and not any(fingers_open[finger] for finger in ("middle", "ring", "pinky")):
            return "draw"

        return "idle"
