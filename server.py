#!/usr/bin/env python3
"""
Ball Falling Game - Python backend
Detects pink sticky notes via webcam and broadcasts their screen-space
positions over WebSocket to the browser game.

Usage:
    uv run server.py              # auto-picks camera
    uv run server.py --camera 1   # use camera index 1
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
import threading
import time
import os
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0  # overridden by --camera arg
WEBSOCKET_HOST    = "localhost"
WEBSOCKET_PORT    = 8765
DETECTION_FPS     = 15
CALIBRATION_FILE  = "calibration.npz"
COLOR_CONFIG_FILE = "color_config.json"

# Default yellow sticky-note HSV ranges (OpenCV: H 0-180, S/V 0-255)
DEFAULT_LOWER_1 = np.array([18,  80,  80], dtype=np.uint8)
DEFAULT_UPPER_1 = np.array([35, 255, 255], dtype=np.uint8)
DEFAULT_LOWER_2 = np.array([18,  80,  80], dtype=np.uint8)
DEFAULT_UPPER_2 = np.array([35, 255, 255], dtype=np.uint8)

PINK_LOWER_1 = DEFAULT_LOWER_1.copy()
PINK_UPPER_1 = DEFAULT_UPPER_1.copy()
PINK_LOWER_2 = DEFAULT_LOWER_2.copy()
PINK_UPPER_2 = DEFAULT_UPPER_2.copy()

CANVAS_W = 1920
CANVAS_H = 1080

MIN_CONTOUR_AREA = 400     # px² in 1920x1080 rectified space
MAX_CONTOUR_AREA = 120_000 # px²

DEBUG_MODE = False

def load_color_config() -> None:
    global PINK_LOWER_1, PINK_UPPER_1, PINK_LOWER_2, PINK_UPPER_2
    if os.path.exists(COLOR_CONFIG_FILE):
        try:
            with open(COLOR_CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                PINK_LOWER_1 = np.array(cfg["lower_1"], dtype=np.uint8)
                PINK_UPPER_1 = np.array(cfg["upper_1"], dtype=np.uint8)
                PINK_LOWER_2 = np.array(cfg["lower_2"], dtype=np.uint8)
                PINK_UPPER_2 = np.array(cfg["upper_2"], dtype=np.uint8)
                print(f"[camera] Loaded custom color config from {COLOR_CONFIG_FILE}")
        except Exception as e:
            print(f"[camera] Error reading {COLOR_CONFIG_FILE}: {e}")

def save_color_config(l1, u1, l2, u2) -> None:
    try:
        cfg = {
            "lower_1": l1.tolist(),
            "upper_1": u1.tolist(),
            "lower_2": l2.tolist(),
            "upper_2": u2.tolist(),
        }
        with open(COLOR_CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[camera] Saved new color config to {COLOR_CONFIG_FILE}")
    except Exception as e:
        print(f"[camera] Error saving {COLOR_CONFIG_FILE}: {e}")

def reset_default_color() -> None:
    global PINK_LOWER_1, PINK_UPPER_1, PINK_LOWER_2, PINK_UPPER_2
    PINK_LOWER_1 = DEFAULT_LOWER_1.copy()
    PINK_UPPER_1 = DEFAULT_UPPER_1.copy()
    PINK_LOWER_2 = DEFAULT_LOWER_2.copy()
    PINK_UPPER_2 = DEFAULT_UPPER_2.copy()
    if os.path.exists(COLOR_CONFIG_FILE):
        try:
            os.remove(COLOR_CONFIG_FILE)
            print(f"[camera] Reset to default yellow; removed {COLOR_CONFIG_FILE}")
        except Exception as e:
            print(f"[camera] Error removing {COLOR_CONFIG_FILE}: {e}")

# ── Shared state (GIL-safe for simple list replacement) ───────────────────────
detected_notes: list[dict] = []
homography: np.ndarray | None = None
latest_status_msg: str = "Click on sticky note in camera view to calibrate color"
status_msg_expiry: float = 0.0


# ── Detection (Warp-First Architecture) ────────────────────────────────────────

def detect_pink_notes(frame: np.ndarray, H: np.ndarray | None) -> tuple[list[dict], np.ndarray]:
    """
    Warp camera frame through homography H into rectified 1920x1080 screen space first,
    eliminating all outside room clutter and keystone angle distortion.
    Detects sticky notes directly in screen coordinates.
    Returns (notes, proc_frame).
    """
    if H is not None:
        proc_frame = cv2.warpPerspective(frame, H, (CANVAS_W, CANVAS_H))
    else:
        proc_frame = cv2.resize(frame, (CANVAS_W, CANVAS_H))

    hsv  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, PINK_LOWER_1, PINK_UPPER_1),
        cv2.inRange(hsv, PINK_LOWER_2, PINK_UPPER_2),
    )

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    notes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA):
            continue

        rect = cv2.minAreaRect(cnt)
        center, (rw, rh), angle = rect

        notes.append({
            "x":      float(center[0]),
            "y":      float(center[1]),
            "width":  float(max(rw, rh)),
            "height": float(min(rw, rh)),
            "angle":  float(angle),
        })

    return notes, proc_frame


# ── Camera loop (runs in background thread) ───────────────────────────────────

current_frame: np.ndarray | None = None

def sample_color_at_point(frame: np.ndarray, x: int, y: int) -> None:
    global PINK_LOWER_1, PINK_UPPER_1, PINK_LOWER_2, PINK_UPPER_2
    global latest_status_msg, status_msg_expiry
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    # Sample a 7x7 patch around click
    y1, y2 = max(0, y - 3), min(h, y + 4)
    x1, x2 = max(0, x - 3), min(w, x + 4)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return

    h_med = int(np.median(patch[:, :, 0]))
    s_med = int(np.median(patch[:, :, 1]))
    v_med = int(np.median(patch[:, :, 2]))

    H_TOL = 14
    S_MIN = int(max(35, s_med - 75))
    V_MIN = int(max(35, v_med - 75))

    if h_med - H_TOL < 0:
        l1 = np.array([0, S_MIN, V_MIN], dtype=np.uint8)
        u1 = np.array([h_med + H_TOL, 255, 255], dtype=np.uint8)
        l2 = np.array([180 + (h_med - H_TOL), S_MIN, V_MIN], dtype=np.uint8)
        u2 = np.array([180, 255, 255], dtype=np.uint8)
    elif h_med + H_TOL > 180:
        l1 = np.array([h_med - H_TOL, S_MIN, V_MIN], dtype=np.uint8)
        u1 = np.array([180, 255, 255], dtype=np.uint8)
        l2 = np.array([0, S_MIN, V_MIN], dtype=np.uint8)
        u2 = np.array([(h_med + H_TOL) - 180, 255, 255], dtype=np.uint8)
    else:
        l1 = np.array([h_med - H_TOL, S_MIN, V_MIN], dtype=np.uint8)
        u1 = np.array([h_med + H_TOL, 255, 255], dtype=np.uint8)
        l2 = l1.copy()
        u2 = u1.copy()

    PINK_LOWER_1, PINK_UPPER_1 = l1, u1
    PINK_LOWER_2, PINK_UPPER_2 = l2, u2
    save_color_config(l1, u1, l2, u2)
    latest_status_msg = f"Color calibrated! HSV: ({h_med}, {s_med}, {v_med})"
    status_msg_expiry = time.monotonic() + 4.0
    print(f"[camera] {latest_status_msg}")

def on_debug_mouse_click(event, x, y, flags, param) -> None:
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        sx = int(x * (current_frame.shape[1] / 960.0))
        sy = int(y * (current_frame.shape[0] / 540.0))
        sample_color_at_point(current_frame, sx, sy)

def camera_loop() -> None:
    global detected_notes, homography, current_frame
    global latest_status_msg, status_msg_expiry

    load_color_config()

    camera_index = CAMERA_INDEX
    if os.path.exists(CALIBRATION_FILE):
        data       = np.load(CALIBRATION_FILE)
        homography = data["H"]
        if "camera_index" in data:
            camera_index = int(data["camera_index"])
        print(f"[camera] Loaded calibration from {CALIBRATION_FILE} (camera {camera_index})")
        print("[camera] Perspective Rectification active — warping camera frame to 1920x1080 play area!")
    else:
        print("[camera] WARNING: no calibration file found — "
              "positions may not align with the projection.")
        print("[camera] Run  uv run calibrate.py  first.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera {camera_index}")
        return

    interval = 1.0 / DETECTION_FPS
    print(f"[camera] Capturing at {DETECTION_FPS} fps (camera {camera_index})")

    show_debug = DEBUG_MODE
    if show_debug:
        cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug", 960, 540)
        cv2.setMouseCallback("Debug", on_debug_mouse_click)
        print("[camera] Debug window open — CLICK any sticky note to sample its color!")
        print("[camera] Press R to reset default yellow, Q to close debug window.")

    while True:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if ret:
            detected_notes, proc_frame = detect_pink_notes(frame, homography)
            current_frame = proc_frame

            if show_debug:
                hsv  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.bitwise_or(
                    cv2.inRange(hsv, PINK_LOWER_1, PINK_UPPER_1),
                    cv2.inRange(hsv, PINK_LOWER_2, PINK_UPPER_2),
                )
                debug = proc_frame.copy()
                # Highlight detected mask in cyan
                debug[mask > 0] = (255, 255, 0)

                # Draw notes bounding boxes directly on rectified frame
                for note in detected_notes:
                    cx, cy = note["x"], note["y"]
                    w, h = note["width"], note["height"]
                    ang = note["angle"]
                    rect = ((cx, cy), (w, h), ang)
                    box = cv2.boxPoints(rect).astype(np.int32)
                    cv2.drawContours(debug, [box], 0, (0, 255, 0), 3)
                    cv2.circle(debug, (int(cx), int(cy)), 6, (0, 0, 255), -1)

                # Top banner
                cv2.rectangle(debug, (0, 0), (debug.shape[1], 48), (20, 20, 20), -1)
                if time.monotonic() < status_msg_expiry:
                    banner_text = latest_status_msg
                    banner_color = (0, 255, 0)
                else:
                    mode_str = "WARPED (Rectified)" if homography is not None else "RAW (Uncalibrated)"
                    banner_text = f"[{mode_str}] Notes: {len(detected_notes)} | Click note to pick color | R=Reset | Q=Quit"
                    banner_color = (0, 220, 255)

                cv2.putText(debug, banner_text, (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

                debug_scaled = cv2.resize(debug, (960, 540))
                cv2.imshow("Debug", debug_scaled)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    show_debug = False
                    cv2.destroyAllWindows()
                elif key == ord('r'):
                    reset_default_color()
                    latest_status_msg = "Reset to default yellow note color"
                    status_msg_expiry = time.monotonic() + 3.0

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))

    cap.release()


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def ws_handler(websocket) -> None:
    addr = websocket.remote_address
    print(f"[ws] Browser connected: {addr}")
    try:
        while True:
            await websocket.send(json.dumps({"notes": detected_notes}))
            await asyncio.sleep(1.0 / DETECTION_FPS)
    except websockets.exceptions.ConnectionClosed:
        print(f"[ws] Browser disconnected: {addr}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (omit to use calibrate.py's choice or default 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Show live camera window with detection overlay")
    args = parser.parse_args()
    if args.camera is not None:
        global CAMERA_INDEX
        CAMERA_INDEX = args.camera  # overrides calibration file
    global DEBUG_MODE
    DEBUG_MODE = args.debug

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    print(f"[ws] Listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    print("[ws] Open index.html in your browser, press F for fullscreen.")

    async with websockets.serve(ws_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
