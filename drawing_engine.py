from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Color = Tuple[int, int, int]  # BGR


class DrawingEngine:
    """Maintains a transparent drawing layer and renders it with a neon glow."""

    def __init__(self, frame_shape: tuple[int, int, int], thickness: int = 5) -> None:
        height, width = frame_shape[:2]
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.thickness = thickness
        self.previous_point: Optional[Point] = None
        self.previous_midpoint: Optional[Point] = None

    def reset(self) -> None:
        self.canvas.fill(0)
        self.previous_point = None
        self.previous_midpoint = None

    def resize_if_needed(self, frame_shape: tuple[int, int, int]) -> None:
        height, width = frame_shape[:2]
        if self.canvas.shape[:2] == (height, width):
            return

        resized = np.zeros((height, width, 3), dtype=np.uint8)
        old_height, old_width = self.canvas.shape[:2]
        copy_height = min(height, old_height)
        copy_width = min(width, old_width)
        resized[:copy_height, :copy_width] = self.canvas[:copy_height, :copy_width]
        self.canvas = resized
        self.previous_point = None
        self.previous_midpoint = None

    def set_thickness(self, thickness: int) -> None:
        self.thickness = max(1, min(40, thickness))

    def start_or_continue_stroke(self, point: Point, color: Color) -> None:
        if self.previous_point is None:
            self.previous_point = point
            self.previous_midpoint = point
            cv2.circle(self.canvas, point, self.thickness // 2, color, -1)
            return

        midpoint = self._midpoint(self.previous_point, point)
        self._draw_quadratic_segment(
            self.previous_midpoint or self.previous_point,
            self.previous_point,
            midpoint,
            color,
        )
        self.previous_point = point
        self.previous_midpoint = midpoint

    def end_stroke(self) -> None:
        self.previous_point = None
        self.previous_midpoint = None

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Composite strokes and a blurred glow over the webcam frame."""

        self.resize_if_needed(frame.shape)
        stroke_mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, stroke_mask = cv2.threshold(stroke_mask, 1, 255, cv2.THRESH_BINARY)

        glow = cv2.GaussianBlur(self.canvas, (0, 0), sigmaX=8, sigmaY=8)
        glow = cv2.addWeighted(glow, 1.25, self.canvas, 0.35, 0)

        output = cv2.addWeighted(frame, 1.0, glow, 0.75, 0)
        crisp_strokes = cv2.addWeighted(output, 0.35, self.canvas, 0.95, 0)
        output[stroke_mask > 0] = crisp_strokes[stroke_mask > 0]
        return output

    def _draw_quadratic_segment(self, start: Point, control: Point, end: Point, color: Color) -> None:
        points = self._quadratic_points(start, control, end)
        cv2.polylines(
            self.canvas,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=color,
            thickness=self.thickness,
            lineType=cv2.LINE_AA,
        )
        radius = max(1, self.thickness // 2)
        cv2.circle(self.canvas, start, radius, color, -1, cv2.LINE_AA)
        cv2.circle(self.canvas, end, radius, color, -1, cv2.LINE_AA)

    def _quadratic_points(self, start: Point, control: Point, end: Point) -> list[Point]:
        distance = int(
            np.hypot(end[0] - start[0], end[1] - start[1])
            + np.hypot(control[0] - start[0], control[1] - start[1])
        )
        steps = max(8, distance // 3)
        curve: list[Point] = []
        for t in np.linspace(0.0, 1.0, steps):
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
            curve.append((int(x), int(y)))
        return curve

    def _midpoint(self, first: Point, second: Point) -> Point:
        return ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
