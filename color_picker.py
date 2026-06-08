from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Color = Tuple[int, int, int]  # BGR


@dataclass(frozen=True)
class ColorSelection:
    color: Color
    source: str


class ColorPickerUI:
    """OpenCV-rendered floating color mixer inspired by mobile color tools."""

    PRESET_COLORS: tuple[Color, ...] = (
        (12, 232, 232),   # yellow
        (30, 188, 188),   # lime
        (35, 178, 245),   # orange
        (0, 160, 255),    # amber
        (120, 210, 105),  # green
        (190, 200, 65),   # teal
        (175, 135, 65),   # sky
        (230, 120, 80),   # blue
        (205, 95, 105),   # indigo
        (210, 75, 160),   # purple
        (165, 65, 200),   # pink
        (70, 80, 210),    # coral
    )

    def __init__(self, panel_width: int = 330) -> None:
        self.panel_width = panel_width
        self.active_color: Color = (210, 75, 160)
        self._wheel_outer_radius = 120
        self._wheel_inner_radius = 58
        self._panel_cache_key: tuple[int, int, int, int] | None = None
        self._panel_base: np.ndarray | None = None
        self._panel_mask: np.ndarray | None = None

    def render(self, frame: np.ndarray, fingertip: Optional[Point] = None) -> np.ndarray:
        height, width = frame.shape[:2]
        layout = self._layout(width, height)
        frame = self._apply_panel_base(frame, layout)
        self._draw_active_bubbles(frame, layout["active_center"])
        self._draw_active_wheel_marker(frame, layout["wheel_center"], fingertip)
        self._draw_swatch_selection(frame, layout["swatch_origin"], layout["swatch_size"], fingertip)
        self._draw_close_button(frame, layout, fingertip)
        return frame

    def update_from_fingertip(
        self,
        fingertip: Optional[Point],
        frame_shape: tuple[int, int, int],
    ) -> Optional[ColorSelection]:
        if fingertip is None:
            return None

        height, width = frame_shape[:2]
        layout = self._layout(width, height)

        wheel_color = self._color_from_wheel(fingertip, layout["wheel_center"])
        if wheel_color is not None:
            self.active_color = wheel_color
            return ColorSelection(wheel_color, "wheel")

        swatch_color = self._color_from_swatches(
            fingertip,
            layout["swatch_origin"],
            layout["swatch_size"],
        )
        if swatch_color is not None:
            self.active_color = swatch_color
            return ColorSelection(swatch_color, "swatch")

        return None

    def is_close_hovered(self, fingertip: Optional[Point], frame_shape: tuple[int, int, int]) -> bool:
        if fingertip is None:
            return False

        height, width = frame_shape[:2]
        layout = self._layout(width, height)
        close_center = layout["close_center"]
        return np.hypot(fingertip[0] - close_center[0], fingertip[1] - close_center[1]) <= 28

    def draw_cursor(self, frame: np.ndarray, point: Optional[Point]) -> np.ndarray:
        if point is None:
            return frame

        glow_layer = frame.copy()
        cv2.circle(glow_layer, point, 17, self.active_color, -1, cv2.LINE_AA)
        frame = cv2.addWeighted(glow_layer, 0.38, frame, 0.62, 0)
        cv2.circle(frame, point, 7, self.active_color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 10, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _layout(self, frame_width: int, frame_height: int) -> dict[str, object]:
        panel_width = min(self.panel_width, max(285, frame_width // 3))
        panel_height = min(frame_height - 28, 560)
        x0, y0 = 18, 18
        x1, y1 = x0 + panel_width, y0 + panel_height
        wheel_radius = min(120, max(88, (panel_width - 76) // 2))
        self._wheel_outer_radius = wheel_radius
        self._wheel_inner_radius = max(46, int(wheel_radius * 0.48))

        return {
            "panel": (x0, y0, x1, y1),
            "active_center": (x0 + 60, y0 + 158),
            "wheel_center": (x0 + panel_width // 2, y0 + 260),
            "swatch_origin": (x0 + 42, y0 + 405),
            "swatch_size": max(28, min(39, (panel_width - 104) // 6)),
            "close_center": (x1 - 35, y0 + 34),
        }

    def _apply_panel_base(self, frame: np.ndarray, layout: dict[str, object]) -> np.ndarray:
        height, width = frame.shape[:2]
        panel = layout["panel"]
        cache_key = (width, height, self._wheel_outer_radius, self._wheel_inner_radius)
        if self._panel_cache_key != cache_key or self._panel_base is None or self._panel_mask is None:
            self._panel_cache_key = cache_key
            self._panel_base, self._panel_mask = self._build_panel_base(width, height, layout)

        alpha = (self._panel_mask.astype(np.float32) / 255.0 * 0.88)[..., None]
        frame = (frame.astype(np.float32) * (1.0 - alpha) + self._panel_base.astype(np.float32) * alpha).astype(np.uint8)
        x0, y0, x1, y1 = panel
        cv2.rectangle(frame, (x0 + 22, y1 - 1), (x1 - 22, y1 + 1), (235, 235, 238), -1)
        return frame

    def _build_panel_base(self, frame_width: int, frame_height: int, layout: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        base = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        x0, y0, x1, y1 = layout["panel"]

        shadow_mask = np.zeros_like(mask)
        self._rounded_rect_mask(shadow_mask, (x0 + 5, y0 + 6), (x1 + 5, y1 + 6), 28, 120)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), 8)
        shadow = np.zeros_like(base)
        shadow[:] = (90, 90, 95)
        shadow_alpha = (shadow_mask.astype(np.float32) / 255.0 * 0.16)[..., None]
        base[:] = (base.astype(np.float32) * (1.0 - shadow_alpha) + shadow.astype(np.float32) * shadow_alpha).astype(np.uint8)

        self._rounded_rect_mask(mask, (x0, y0), (x1, y1), 28, 255)
        card = np.zeros_like(base)
        card[:] = (250, 250, 252)
        card_alpha = (mask.astype(np.float32) / 255.0)[..., None]
        base[:] = (base.astype(np.float32) * (1.0 - card_alpha) + card.astype(np.float32) * card_alpha).astype(np.uint8)

        self._draw_header(base, layout)
        self._draw_color_wheel_static(base, layout["wheel_center"])
        self._draw_swatches_static(base, layout["swatch_origin"], layout["swatch_size"])
        self._draw_footer_hints(base, layout)
        cv2.rectangle(base, (x0 + 22, y1 - 2), (x1 - 22, y1), (232, 232, 236), -1)
        return base, mask

    def _draw_header(self, frame: np.ndarray, layout: dict[str, object]) -> None:
        x0, y0, x1, _ = layout["panel"]
        cv2.putText(frame, "Color", (x0 + 130, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 45, 48), 1, cv2.LINE_AA)

        cv2.putText(frame, "PICKER", (x0 + 88, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (175, 175, 178), 1, cv2.LINE_AA)
        cv2.putText(frame, "MIXER", (x0 + 190, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (95, 145, 85), 1, cv2.LINE_AA)
        cv2.line(frame, (x0 + 190, y0 + 100), (x0 + 242, y0 + 100), (95, 175, 90), 2, cv2.LINE_AA)

    def _draw_active_bubbles(self, frame: np.ndarray, center: Point) -> None:
        companion = self._wheel_color(210)
        cv2.circle(frame, (center[0] + 54, center[1] - 30), 24, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (center[0] + 54, center[1] - 30), 22, self.active_color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 24, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 22, companion, -1, cv2.LINE_AA)

    def _draw_color_wheel_static(self, frame: np.ndarray, center: Point) -> None:
        outer = self._wheel_outer_radius
        inner = self._wheel_inner_radius

        shadow = frame.copy()
        cv2.circle(shadow, (center[0] + 3, center[1] + 7), outer + 3, (180, 180, 185), -1, cv2.LINE_AA)
        cv2.circle(shadow, (center[0] + 3, center[1] + 7), inner - 3, (246, 246, 248), -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(shadow, 0.18, frame, 0.82, 0)

        for y in range(center[1] - outer, center[1] + outer + 1):
            for x in range(center[0] - outer, center[0] + outer + 1):
                if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
                    continue
                dx = x - center[0]
                dy = y - center[1]
                radius = np.hypot(dx, dy)
                if inner <= radius <= outer:
                    hue = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                    saturation = 0.58 + 0.32 * ((radius - inner) / max(1, outer - inner))
                    value = 0.95
                    frame[y, x] = self._wheel_color(hue, saturation, value)

        cv2.circle(frame, center, outer, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, center, inner, (246, 246, 248), -1, cv2.LINE_AA)
        cv2.circle(frame, center, inner, (235, 235, 238), 1, cv2.LINE_AA)

    def _draw_active_wheel_marker(self, frame: np.ndarray, center: Point, fingertip: Optional[Point]) -> None:
        marker = self._active_marker_position(center)
        cv2.circle(frame, marker, 28, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, marker, 24, self.active_color, -1, cv2.LINE_AA)
        cv2.circle(frame, marker, 25, (238, 238, 242), 2, cv2.LINE_AA)

        if fingertip is not None and self._color_from_wheel(fingertip, center) is not None:
            cv2.circle(frame, fingertip, 13, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_swatches_static(self, frame: np.ndarray, origin: Point, size: int) -> None:
        x0, y0 = origin
        gap_x = 14
        gap_y = 18
        for index, color in enumerate(self.PRESET_COLORS):
            row, col = divmod(index, 6)
            center = (x0 + col * (size + gap_x) + size // 2, y0 + row * (size + gap_y) + size // 2)
            cv2.circle(frame, center, size // 2 + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center, size // 2, color, -1, cv2.LINE_AA)
        return

    def _draw_swatch_selection(self, frame: np.ndarray, origin: Point, size: int, fingertip: Optional[Point]) -> None:
        x0, y0 = origin
        gap_x = 14
        gap_y = 18
        for index, color in enumerate(self.PRESET_COLORS):
            row, col = divmod(index, 6)
            center = (x0 + col * (size + gap_x) + size // 2, y0 + row * (size + gap_y) + size // 2)
            if self._colors_close(color, self.active_color):
                cv2.circle(frame, center, size // 2 + 5, (80, 160, 170), 2, cv2.LINE_AA)

        if fingertip is not None and self._color_from_swatches(fingertip, origin, size) is not None:
            cv2.circle(frame, fingertip, 12, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_close_button(self, frame: np.ndarray, layout: dict[str, object], fingertip: Optional[Point]) -> None:
        close_center = layout["close_center"]
        hovered = fingertip is not None and np.hypot(fingertip[0] - close_center[0], fingertip[1] - close_center[1]) <= 28
        fill = (215, 245, 248) if hovered else (248, 252, 253)
        border = (70, 170, 178) if hovered else (205, 225, 230)
        cv2.circle(frame, close_center, 18, fill, -1, cv2.LINE_AA)
        cv2.circle(frame, close_center, 18, border, 2, cv2.LINE_AA)
        cv2.line(frame, (close_center[0] - 7, close_center[1] - 7), (close_center[0] + 7, close_center[1] + 7), (80, 145, 150), 2, cv2.LINE_AA)
        cv2.line(frame, (close_center[0] + 7, close_center[1] - 7), (close_center[0] - 7, close_center[1] + 7), (80, 145, 150), 2, cv2.LINE_AA)

    def _draw_footer_hints(self, frame: np.ndarray, layout: dict[str, object]) -> None:
        x0, _, _, y1 = layout["panel"]
        cv2.putText(frame, "Hover colors to select  |  X: close", (x0 + 28, y1 - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (85, 85, 90), 1, cv2.LINE_AA)
        cv2.putText(frame, "Index: draw  |  Fist 2s: clear", (x0 + 28, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (85, 85, 90), 1, cv2.LINE_AA)

    def _color_from_wheel(self, point: Point, center: Point) -> Optional[Color]:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        radius = np.hypot(dx, dy)
        if not self._wheel_inner_radius - 18 <= radius <= self._wheel_outer_radius + 18:
            return None

        hue = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        clamped_radius = min(self._wheel_outer_radius, max(self._wheel_inner_radius, radius))
        saturation = 0.58 + 0.32 * ((clamped_radius - self._wheel_inner_radius) / max(1, self._wheel_outer_radius - self._wheel_inner_radius))
        return self._wheel_color(hue, saturation, 0.95)

    def _color_from_swatches(self, point: Point, origin: Point, size: int) -> Optional[Color]:
        x0, y0 = origin
        gap_x = 14
        gap_y = 18
        for index, color in enumerate(self.PRESET_COLORS):
            row, col = divmod(index, 6)
            center = (x0 + col * (size + gap_x) + size // 2, y0 + row * (size + gap_y) + size // 2)
            if np.hypot(point[0] - center[0], point[1] - center[1]) <= size // 2 + 16:
                return color
        return None

    def _active_marker_position(self, center: Point) -> Point:
        blue, green, red = [value / 255 for value in self.active_color]
        hue, _, _ = colorsys.rgb_to_hsv(red, green, blue)
        angle = hue * 2 * np.pi
        radius = (self._wheel_inner_radius + self._wheel_outer_radius) // 2
        return (int(center[0] + np.cos(angle) * radius), int(center[1] + np.sin(angle) * radius))

    def _wheel_color(self, hue: float, saturation: float = 0.72, value: float = 0.95) -> Color:
        red, green, blue = colorsys.hsv_to_rgb((hue % 360) / 360, saturation, value)
        return (int(blue * 255), int(green * 255), int(red * 255))

    def _colors_close(self, first: Color, second: Color) -> bool:
        return sum(abs(first[i] - second[i]) for i in range(3)) < 36

    def _rounded_rect_mask(self, mask: np.ndarray, top_left: Point, bottom_right: Point, radius: int, value: int) -> None:
        x0, y0 = top_left
        x1, y1 = bottom_right
        cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), value, -1)
        cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), value, -1)
        cv2.circle(mask, (x0 + radius, y0 + radius), radius, value, -1, cv2.LINE_AA)
        cv2.circle(mask, (x1 - radius, y0 + radius), radius, value, -1, cv2.LINE_AA)
        cv2.circle(mask, (x0 + radius, y1 - radius), radius, value, -1, cv2.LINE_AA)
        cv2.circle(mask, (x1 - radius, y1 - radius), radius, value, -1, cv2.LINE_AA)
