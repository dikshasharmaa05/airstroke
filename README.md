# AI Air Drawing

Real-time webcam drawing with Python, OpenCV, MediaPipe Hands, and NumPy.

## Features

- Tracks the index fingertip as a drawing cursor.
- Draw gesture: raise only the index finger.
- Pause gesture: open all fingers.
- Clear gesture: hold a closed fist for 2 seconds.
- Floating semi-transparent color panel with a donut color wheel and 10 swatches.
- Neon glow strokes rendered over the live webcam feed.
- Adjustable stroke thickness with keyboard shortcuts.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On first run, the MediaPipe hand landmark model (`hand_landmarker.task`) downloads automatically.

## Run

```bash
python main.py
```

Optional camera settings:

```bash
python main.py --camera 0 --width 1280 --height 720 --thickness 5
```

## Controls

- `q` or `Esc`: quit
- `+` / `-`: adjust thickness by 1 px
- `[` / `]`: adjust thickness by 2 px
- `c`: clear canvas immediately

Hover your index fingertip over the color wheel or swatches to change the active color.
