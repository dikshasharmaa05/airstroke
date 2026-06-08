from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from color_picker import ColorPickerUI
from drawing_engine import DrawingEngine
from hand_tracking import HandTracker


WINDOW_NAME = "AI Air Drawing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time AI air drawing with MediaPipe hand tracking.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--thickness", type=int, default=5, help="Initial stroke thickness in pixels.")
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    for backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if capture.isOpened():
            ok, _ = read_camera_frame(capture, attempts=20)
            if ok:
                return capture
        capture.release()

    raise RuntimeError(f"Could not open webcam at index {camera_index}.")


def read_camera_frame(capture: cv2.VideoCapture, attempts: int = 30):
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok:
            return ok, frame
        time.sleep(0.08)
    return False, None


def smooth_cursor(raw_point: tuple[int, int] | None, previous_point: tuple[int, int] | None) -> tuple[int, int] | None:
    if raw_point is None:
        return None
    if previous_point is None:
        return raw_point

    distance = np.hypot(raw_point[0] - previous_point[0], raw_point[1] - previous_point[1])
    alpha = 0.22 if distance < 45 else 0.42 if distance < 110 else 0.68
    x = int(previous_point[0] + (raw_point[0] - previous_point[0]) * alpha)
    y = int(previous_point[1] + (raw_point[1] - previous_point[1]) * alpha)
    return (x, y)


def draw_status(frame, gesture: str, thickness: int, clear_progress: float) -> None:
    height, width = frame.shape[:2]
    status = f"Gesture: {gesture.upper()} | Thickness: {thickness}px"
    cv2.putText(
        frame,
        status,
        (280, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if clear_progress > 0:
        bar_width = 220
        filled = int(bar_width * min(1.0, clear_progress))
        x, y = width - bar_width - 32, height - 44
        cv2.rectangle(frame, (x, y), (x + bar_width, y + 14), (55, 55, 65), -1)
        cv2.rectangle(frame, (x, y), (x + filled, y + 14), (0, 220, 255), -1)
        cv2.putText(
            frame,
            "Hold fist to clear",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def handle_key(key: int, engine: DrawingEngine) -> bool:
    if key in (27, ord("q")):
        return False
    if key in (ord("+"), ord("=")):
        engine.set_thickness(engine.thickness + 1)
    elif key in (ord("-"), ord("_")):
        engine.set_thickness(engine.thickness - 1)
    elif key == ord("]"):
        engine.set_thickness(engine.thickness + 2)
    elif key == ord("["):
        engine.set_thickness(engine.thickness - 2)
    elif key == ord("c"):
        engine.reset()
    return True


def main() -> None:
    args = parse_args()
    capture = open_camera(args.camera, args.width, args.height)
    tracker = HandTracker()
    picker = ColorPickerUI()

    ok, frame = read_camera_frame(capture)
    if not ok:
        raise RuntimeError("Could not read from webcam.")

    frame = cv2.flip(frame, 1)
    engine = DrawingEngine(frame.shape, thickness=args.thickness)
    fist_started_at: float | None = None
    close_started_at: float | None = None
    draw_grace_until = 0.0
    smoothed_fingertip: tuple[int, int] | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = read_camera_frame(capture, attempts=3)
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            hand_state, annotated_frame = tracker.process(frame)
            fingertip = hand_state.index_tip if hand_state else None
            smoothed_fingertip = smooth_cursor(fingertip, smoothed_fingertip)
            gesture = hand_state.gesture if hand_state else "none"

            selection = picker.update_from_fingertip(fingertip, annotated_frame.shape)
            selecting_color = selection is not None

            now = time.monotonic()
            close_hovered = picker.is_close_hovered(fingertip, annotated_frame.shape)
            if close_hovered:
                if close_started_at is None:
                    close_started_at = now
                if now - close_started_at >= 0.75:
                    break
            else:
                close_started_at = None

            clear_progress = 0.0
            if hand_state and hand_state.is_fist:
                if fist_started_at is None:
                    fist_started_at = now
                clear_progress = (now - fist_started_at) / 2.0
                engine.end_stroke()
                if clear_progress >= 1.0:
                    engine.reset()
                    fist_started_at = None
                    clear_progress = 0.0
            else:
                fist_started_at = None

            if hand_state and hand_state.is_draw_gesture:
                draw_grace_until = now + 0.18

            can_continue_stroke = hand_state is not None and now <= draw_grace_until
            if can_continue_stroke and not selecting_color:
                engine.start_or_continue_stroke(smoothed_fingertip or hand_state.index_tip, picker.active_color)
            else:
                engine.end_stroke()

            output = engine.render(annotated_frame)
            output = picker.render(output, smoothed_fingertip)
            output = picker.draw_cursor(output, smoothed_fingertip)
            draw_status(output, gesture, engine.thickness, clear_progress)

            cv2.imshow(WINDOW_NAME, output)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKeyEx(1) & 0xFF
            if not handle_key(key, engine):
                break
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
