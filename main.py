import cv2
import math
import time
import numpy as np
import xmlrpc.client as xmlrpclib

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# Config
# =========================
PYMOL_RPC_URL = "http://localhost:9123"
MODEL_PATH = "models/hand_landmarker.task"

# Pinch thresholds (normalized coords)
THUMB_INDEX_PINCH_ON = 0.045
THUMB_INDEX_PINCH_OFF = 0.055
THUMB_MIDDLE_PINCH_ON = 0.050
THUMB_MIDDLE_PINCH_OFF = 0.060

# Middle-Index touch thresholds (for pan gesture)
# Set tighter than thumb pinches to avoid false triggers when fingers are just extended
MID_INDEX_PINCH_ON = 0.030
MID_INDEX_PINCH_OFF = 0.040

# Tap detection
TAP_MAX_DURATION = 0.35
TAP_MAX_MOVEMENT = 0.03

# Fist detection: all fingertips must be below their PIP joints,
# and thumb tip must be close to palm center
FIST_HOLD_TIME = 0.3  # seconds fist must be held to activate lock

# Gains
ROT_GAIN = 360.0
PAN_GAIN = 350.0
ZOOM_STEP = 18.0

EMA_ALPHA = 0.7
CAM_INDEX = 0


# =========================
# Utils
# =========================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ema(prev, cur, alpha=EMA_ALPHA):
    if prev is None:
        return cur
    return (alpha * prev[0] + (1 - alpha) * cur[0],
            alpha * prev[1] + (1 - alpha) * cur[1])


def is_fist(lm):
    """Detect a closed fist: all four fingers curled (tip below PIP)
    and thumb tip tucked near index MCP."""
    index_curled = lm[8].y > lm[6].y
    middle_curled = lm[12].y > lm[10].y
    ring_curled = lm[16].y > lm[14].y
    pinky_curled = lm[20].y > lm[18].y

    # Thumb tip should be close to index finger base (MCP, landmark 5)
    thumb_tucked = dist((lm[4].x, lm[4].y), (lm[5].x, lm[5].y)) < 0.08

    return index_curled and middle_curled and ring_curled and pinky_curled and thumb_tucked


class PinchDetector:
    """Tracks a single pinch pair with hysteresis and distinguishes tap vs hold."""

    def __init__(self, on_thresh, off_thresh):
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.active = False
        self.start_time = None
        self.start_pos = None
        self.max_travel = 0.0

    def update(self, distance, midpoint):
        """Returns event string: 'tap', 'hold_end', or None."""
        event = None

        if not self.active and distance < self.on_thresh:
            self.active = True
            self.start_time = time.time()
            self.start_pos = midpoint
            self.max_travel = 0.0

        elif self.active and distance > self.off_thresh:
            self.active = False
            duration = time.time() - self.start_time if self.start_time else 999
            if duration < TAP_MAX_DURATION and self.max_travel < TAP_MAX_MOVEMENT:
                event = "tap"
            else:
                event = "hold_end"
            self.start_time = None
            self.start_pos = None
            self.max_travel = 0.0

        elif self.active and self.start_pos is not None:
            travel = dist(midpoint, self.start_pos)
            self.max_travel = max(self.max_travel, travel)

        return event

    @property
    def is_hold(self):
        """True when pinch has been held long enough / moved enough to count as hold."""
        if not self.active or self.start_time is None:
            return False
        duration = time.time() - self.start_time
        return duration >= TAP_MAX_DURATION or self.max_travel >= TAP_MAX_MOVEMENT

    def reset(self):
        self.active = False
        self.start_time = None
        self.start_pos = None
        self.max_travel = 0.0


# =========================
# Main
# =========================
def main():
    cmd = xmlrpclib.ServerProxy(PYMOL_RPC_URL, allow_none=True)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}. Try 1/2 or check permissions.")

    # Pinch detectors
    thumb_index_pinch = PinchDetector(THUMB_INDEX_PINCH_ON, THUMB_INDEX_PINCH_OFF)
    thumb_middle_pinch = PinchDetector(THUMB_MIDDLE_PINCH_ON, THUMB_MIDDLE_PINCH_OFF)
    mid_index_pinch = PinchDetector(MID_INDEX_PINCH_ON, MID_INDEX_PINCH_OFF)

    # State
    index_tip_s = None
    mid_tip_s = None
    prev_index_tip = None
    prev_pan_tip = None
    rotating = False
    panning = False

    # Fist lock state
    locked = False
    fist_start_time = None  # when we first saw the fist

    # Cooldown for zoom taps
    last_zoom_time = 0.0
    ZOOM_COOLDOWN = 0.3

    t0 = time.time()
    status_text = "idle"
    status_expire = 0.0

    def set_status(text, duration=0.8):
        nonlocal status_text, status_expire
        status_text = text
        status_expire = time.time() + duration

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int((time.time() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            now = time.time()

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                # --- Fist toggle logic (works regardless of lock state) ---
                fist_detected = is_fist(lm)

                if fist_detected:
                    if fist_start_time is None:
                        fist_start_time = now
                    elif (now - fist_start_time) >= FIST_HOLD_TIME and not locked:
                        locked = True
                        # Reset all active gestures
                        thumb_index_pinch.reset()
                        thumb_middle_pinch.reset()
                        mid_index_pinch.reset()
                        rotating = False
                        panning = False
                        prev_index_tip = None
                        prev_pan_tip = None
                        set_status("LOCKED (fist)", duration=999)
                else:
                    # Fist released
                    if locked and fist_start_time is not None:
                        # Unlock when fist is released after locking
                        locked = False
                        set_status("UNLOCKED", duration=1.0)
                    fist_start_time = None

                # --- If locked, skip all gesture processing ---
                if locked:
                    # Draw landmarks in red tint to indicate lock
                    for i, l in enumerate(lm):
                        cx, cy = int(l.x * w), int(l.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 200), -1)

                    # Show lock status
                    cv2.putText(frame, "LOCKED (fist)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("PyMOL Gesture Control (ESC)", frame)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                    continue

                # --- Normal gesture processing ---
                thumb_tip = (lm[4].x, lm[4].y)
                index_tip = (lm[8].x, lm[8].y)
                middle_tip = (lm[12].x, lm[12].y)

                index_tip_s = ema(index_tip_s, index_tip)

                # Smoothed midpoint of index+middle for pan tracking
                pan_point = ((index_tip[0] + middle_tip[0]) / 2,
                             (index_tip[1] + middle_tip[1]) / 2)
                mid_tip_s = ema(mid_tip_s, pan_point)

                ti_dist = dist(thumb_tip, index_tip)
                tm_dist = dist(thumb_tip, middle_tip)
                mi_dist = dist(middle_tip, index_tip)

                ti_mid = ((thumb_tip[0] + index_tip[0]) / 2,
                          (thumb_tip[1] + index_tip[1]) / 2)
                tm_mid = ((thumb_tip[0] + middle_tip[0]) / 2,
                          (thumb_tip[1] + middle_tip[1]) / 2)
                mi_mid = ((middle_tip[0] + index_tip[0]) / 2,
                          (middle_tip[1] + index_tip[1]) / 2)

                # Update all pinch detectors
                ti_event = thumb_index_pinch.update(ti_dist, ti_mid)
                tm_event = thumb_middle_pinch.update(tm_dist, tm_mid)
                mi_event = mid_index_pinch.update(mi_dist, mi_mid)

                # --- Thumb + Index TAP -> Zoom Out ---
                if ti_event == "tap" and (now - last_zoom_time) > ZOOM_COOLDOWN:
                    if not thumb_middle_pinch.active and not mid_index_pinch.active:
                        cmd.move("z", ZOOM_STEP)
                        last_zoom_time = now
                        set_status("ZOOM OUT (index tap)")

                # --- Thumb + Middle TAP -> Zoom In ---
                if tm_event == "tap" and (now - last_zoom_time) > ZOOM_COOLDOWN:
                    if not thumb_index_pinch.active and not mid_index_pinch.active:
                        cmd.move("z", -ZOOM_STEP)
                        last_zoom_time = now
                        set_status("ZOOM IN (middle tap)")

                # --- Thumb + Index HOLD -> Rotate ---
                if (thumb_index_pinch.is_hold
                        and not thumb_middle_pinch.active
                        and not mid_index_pinch.active):
                    if not rotating:
                        rotating = True
                        prev_index_tip = index_tip_s
                        set_status("ROTATE", duration=999)

                    if prev_index_tip is not None:
                        dx = index_tip_s[0] - prev_index_tip[0]
                        dy = index_tip_s[1] - prev_index_tip[1]
                        cmd.turn("y", -dx * ROT_GAIN)
                        cmd.turn("x", -dy * ROT_GAIN)
                    prev_index_tip = index_tip_s
                else:
                    if rotating:
                        rotating = False
                        prev_index_tip = None
                        set_status("idle")

                # --- Middle + Index HOLD -> Pan (move x/y) ---
                if (mid_index_pinch.is_hold
                        and not thumb_index_pinch.active
                        and not thumb_middle_pinch.active):
                    if not panning:
                        panning = True
                        prev_pan_tip = mid_tip_s
                        set_status("PAN", duration=999)

                    if prev_pan_tip is not None:
                        dx = mid_tip_s[0] - prev_pan_tip[0]
                        dy = mid_tip_s[1] - prev_pan_tip[1]
                        cmd.move("x", dx * PAN_GAIN)
                        cmd.move("y", -dy * PAN_GAIN)
                    prev_pan_tip = mid_tip_s
                else:
                    if panning:
                        panning = False
                        prev_pan_tip = None
                        set_status("idle")

                # --- Draw landmarks ---
                for i, l in enumerate(lm):
                    cx, cy = int(l.x * w), int(l.y * h)
                    color = (0, 255, 0)
                    if i == 4:
                        color = (0, 0, 255)   # thumb tip: red
                    elif i == 8:
                        color = (255, 0, 0)   # index tip: blue
                    elif i == 12:
                        color = (255, 255, 0) # middle tip: cyan
                    cv2.circle(frame, (cx, cy), 4, color, -1)

                # Draw pinch lines
                if thumb_index_pinch.active:
                    cv2.line(frame,
                             (int(thumb_tip[0] * w), int(thumb_tip[1] * h)),
                             (int(index_tip[0] * w), int(index_tip[1] * h)),
                             (0, 0, 255), 2)
                if thumb_middle_pinch.active:
                    cv2.line(frame,
                             (int(thumb_tip[0] * w), int(thumb_tip[1] * h)),
                             (int(middle_tip[0] * w), int(middle_tip[1] * h)),
                             (0, 255, 255), 2)
                if mid_index_pinch.active:
                    cv2.line(frame,
                             (int(middle_tip[0] * w), int(middle_tip[1] * h)),
                             (int(index_tip[0] * w), int(index_tip[1] * h)),
                             (255, 0, 255), 2)

            else:
                # Lost hand - reset everything
                thumb_index_pinch.reset()
                thumb_middle_pinch.reset()
                mid_index_pinch.reset()
                rotating = False
                panning = False
                index_tip_s = None
                mid_tip_s = None
                prev_index_tip = None
                prev_pan_tip = None
                fist_start_time = None
                # Keep locked state across hand loss so accidental drop doesn't unlock
                if not locked and now > status_expire:
                    status_text = "no hand"

            # HUD
            if locked:
                display = "LOCKED (fist)"
            elif now < status_expire or status_text in ("no hand", "idle"):
                display = status_text
            else:
                display = "idle"

            hud_color = (0, 0, 255) if locked else (0, 255, 0)
            cv2.putText(frame, display, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)

            cv2.imshow("PyMOL Gesture Control (ESC)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()