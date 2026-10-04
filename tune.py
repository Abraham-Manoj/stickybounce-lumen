#!/usr/bin/env python3
"""
Detection tuning tool — run this standalone to check if pink notes are detected.
Shows live camera feed with cyan highlight where pink is detected.
Press Q to quit.

Usage:
    uv run tune.py
    uv run tune.py --camera 1
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys

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
MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 60_000

latest_status = "Click on sticky note to pick color | R=Reset | Q=Quit"
status_expiry = 0.0

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
                print(f"Loaded {COLOR_CONFIG_FILE}")
        except Exception as e:
            print(f"Error loading {COLOR_CONFIG_FILE}: {e}")

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
        print(f"Saved {COLOR_CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving {COLOR_CONFIG_FILE}: {e}")

def reset_default_color() -> None:
    global PINK_LOWER_1, PINK_UPPER_1, PINK_LOWER_2, PINK_UPPER_2
    PINK_LOWER_1 = DEFAULT_LOWER_1.copy()
    PINK_UPPER_1 = DEFAULT_UPPER_1.copy()
    PINK_LOWER_2 = DEFAULT_LOWER_2.copy()
    PINK_UPPER_2 = DEFAULT_UPPER_2.copy()
    if os.path.exists(COLOR_CONFIG_FILE):
        try:
            os.remove(COLOR_CONFIG_FILE)
            print("Reset to default yellow")
        except Exception:
            pass

current_frame = None

def sample_color_at_point(frame, x, y):
    global PINK_LOWER_1, PINK_UPPER_1, PINK_LOWER_2, PINK_UPPER_2
    global latest_status, status_expiry
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
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
    import time
    latest_status = f"Color sampled! HSV: ({h_med}, {s_med}, {v_med})"
    status_expiry = time.monotonic() + 4.0
    print(latest_status)

def on_mouse_click(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        sample_color_at_point(current_frame, x, y)


def list_cameras(max_test: int = 6) -> list[int]:
    found = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()

    if args.camera is not None:
        camera_index = args.camera
    else:
        indices = list_cameras()
        if not indices:
            print("No cameras found.")
            sys.exit(1)
        camera_index = indices[-1]  # last = most recently added (iPhone)
        print(f"Using camera {camera_index}  (all found: {indices})")

    load_color_config()

    CALIBRATION_FILE = "calibration.npz"
    homography = None
    if os.path.exists(CALIBRATION_FILE):
        try:
            cal = np.load(CALIBRATION_FILE)
            homography = cal["H"]
            print(f"Loaded perspective calibration from {CALIBRATION_FILE}")
        except Exception:
            pass

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera_index}")
        sys.exit(1)

    win_name = "Detection Tuning"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)
    cv2.setMouseCallback(win_name, on_mouse_click)
    print("Click on any sticky note to sample its color!")
    print("Press R to reset default yellow, Q to quit.\n")

    import time
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if homography is not None:
            proc_frame = cv2.warpPerspective(frame, homography, (1920, 1080))
        else:
            proc_frame = cv2.resize(frame, (1920, 1080))

        global current_frame
        current_frame = proc_frame

        hsv  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, PINK_LOWER_1, PINK_UPPER_1),
            cv2.inRange(hsv, PINK_LOWER_2, PINK_UPPER_2),
        )

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = proc_frame.copy()
        display[mask > 0] = (255, 255, 0)  # cyan highlight

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 400 < area < 120_000:
                count += 1
                rect = cv2.minAreaRect(cnt)
                box  = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(display, [box], 0, (0, 255, 0), 3)

        cv2.rectangle(display, (0, 0), (display.shape[1], 48), (20, 20, 20), -1)
        if time.monotonic() < status_expiry:
            banner_text = latest_status
            banner_color = (0, 255, 0)
        else:
            mode_tag = "RECTIFIED (Warped)" if homography is not None else "RAW (Uncalibrated)"
            banner_text = f"[{mode_tag}] Detected: {count} | Click note to pick color | R=Reset | Q=Quit"
            banner_color = (0, 220, 255)

        cv2.putText(display, banner_text, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

        display_scaled = cv2.resize(display, (960, 540))
        cv2.imshow(win_name, display_scaled)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_default_color()
            latest_status = "Reset to default yellow note color"
            status_expiry = time.monotonic() + 3.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
