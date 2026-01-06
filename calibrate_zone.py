import os
import json
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "").strip()

# click order: BL, BR, TR, TL
CLICK_ORDER = ["BL", "BR", "TR", "TL"]

pts = []
img = None

def on_mouse(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append([int(x), int(y)])
            print(f"Point {len(pts)}/4 ({CLICK_ORDER[len(pts)-1]}): {pts[-1]}")
        else:
            print("Already have 4 points. Press R to reset or S to save.")

def draw_overlay(frame):
    d = frame.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(d, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(d, f"{CLICK_ORDER[i]}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(pts) == 4:
        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(d, [poly], True, (0, 255, 0), 2)

    cv2.putText(d, "Click 4 points: BL, BR, TR, TL | R-reset | S-save | Q-quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
    return d

def main():
    global img, pts
    if not VIDEO_SOURCE:
        raise SystemExit("VIDEO_SOURCE is empty in .env")

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("Failed to open VIDEO_SOURCE")

    # grab a fresh frame
    for _ in range(10):
        cap.grab()
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise SystemExit("Failed to read a frame")

    win = "Calibrate Zone"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = draw_overlay(img)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("r"), ord("R")):
            pts = []
            print("Reset points.")
        if k in (ord("s"), ord("S")):
            if len(pts) != 4:
                print("Need exactly 4 points.")
                continue
            out = json.dumps(pts, ensure_ascii=False)
            print("\nZONE_POLYGON_JSON=" + out + "\n")
            with open("zone_polygon.json", "w", encoding="utf-8") as f:
                f.write(out)
            print("Saved to zone_polygon.json")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
