# PyMOL Gesture Control

Control PyMOL molecular visualization with hand gestures via your webcam using MediaPipe hand tracking.


<p align="center">
  <video src="assets/demo1-s.mp4" alt="Demo" width="600" controls></video>
</p>

## Gestures

| Gesture                 | Action                                               |
| ----------------------- | ---------------------------------------------------- |
| **Thumb + Index tap**   | Zoom out                                             |
| **Thumb + Middle tap**  | Zoom in                                              |
| **Thumb + Index hold**  | Rotate (x/y)                                         |
| **Middle + Index hold** | Pan (x/y)                                            |
| **Fist hold**           | Lock — freezes all controls. Release fist to unlock. |

## Prerequisites

- Python 3.9+
- A webcam
- PyMOL with XML-RPC server enabled (port 9123)

### Enabling PyMOL XML-RPC

Start PyMOL with the RPC server:

```bash
pymol -R
```

## Installation

```bash
git clone https://github.com/yourname/pymol-gesture-control.git
cd pymol-gesture-control
pip install -r requirements.txt
```

Download the [MediaPipe Hand Landmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and place it in `models/`:

```bash
mkdir -p models
curl -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Usage

1. Start PyMOL with RPC enabled (`pymol -R`)
2. Load a molecule in PyMOL (e.g. `fetch 1ubq`)
3. Run the gesture controller:

```bash
python main.py
```

Press **ESC** in the camera window to quit.

## Configuration

Key parameters at the top of `main.py`:

| Parameter       | Default                 | Description                                |
| --------------- | ----------------------- | ------------------------------------------ |
| `PYMOL_RPC_URL` | `http://localhost:9123` | PyMOL RPC address                          |
| `CAM_INDEX`     | `0`                     | Webcam index                               |
| `ROT_GAIN`      | `360.0`                 | Rotation sensitivity                       |
| `PAN_GAIN`      | `350.0`                 | Pan sensitivity                            |
| `ZOOM_STEP`     | `18.0`                  | Zoom distance per tap                      |
| `EMA_ALPHA`     | `0.7`                   | Smoothing factor (higher = more smoothing) |

## License

MIT