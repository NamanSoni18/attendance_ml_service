from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "face_scans"
HISTORY_PATH = ROOT / "data" / "scan_history.json"

ANGLES = ["front", "left", "right", "up", "down"]


def ensure_storage():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def show_instruction(window_name: str, message: str):
    cv2.putText(
        frame := np.zeros((280, 640, 3), dtype=np.uint8),
        message,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, frame)


def capture_variations(uid: str):
    ensure_storage()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Please check your camera connection.")

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8")) or []

    for angle in ANGLES:
        print(f"\n[ANGLE] {angle.upper()}\nLook at the camera and press 'c' to capture.")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read camera frame.")

            cv2.putText(
                frame,
                f"UID: {uid} | Angle: {angle}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press 'c' to capture | 'q' to quit",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Face Capture", frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("c"), ord("C")):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_name = f"{uid}_{angle}_{timestamp}.jpg"
                file_path = DATA_DIR / file_name
                cv2.imwrite(str(file_path), frame)

                history.append(
                    {
                        "scan_id": f"{uid}-{angle}-{timestamp}",
                        "uid": uid,
                        "angle": angle,
                        "image_path": str(file_path),
                        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "score": None,
                        "threshold": 0.68,
                        "result": "pending",
                    }
                )
                HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
                print(f"[SAVED] {file_path}")
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("[EXIT] Capture cancelled.")
                return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[COMPLETE] Saved {len(history)} scans to {HISTORY_PATH}")


if __name__ == "__main__":
    import numpy as np

    user_id = input("Enter UID or name to tag the scans: ").strip() or "student_01"
    capture_variations(user_id)
