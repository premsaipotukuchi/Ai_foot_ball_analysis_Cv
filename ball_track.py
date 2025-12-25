from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ==== SETTINGS ====
VIDEO_PATH = "inputvideos/08fd33_4.mp4"   # your input video
OUTPUT_PATH = "outputvideos/08fd33_4_ball_tracked.mp4"
MODEL_PATH = "yolov8x.pt"                 # use yolov8n for low load, or your custom ball model
PROCESS_EVERY_N_FRAMES = 1                # set to 2 or 3 to make it even lighter
BALL_CLASS_ID = 32                        # COCO "sports ball"
# ===================


def main():
    # Load model
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sure output folder exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    trajectory = []   # list of (cx, cy) points
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # To reduce load, optionally skip frames
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            # still draw previous trajectory and just write frame
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)
            out.write(frame)
            frame_idx += 1
            continue

        # Run YOLO on this frame (small imgsz for speed)
        results = model(frame, imgsz=640, conf=0.25, verbose=False)[0]

        ball_box = None
        best_conf = 0.0

        # Find the best "sports ball" detection
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != BALL_CLASS_ID:
                continue

            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                ball_box = box.xyxy[0].cpu().numpy()

        if ball_box is not None:
            x1, y1, x2, y2 = ball_box.astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            trajectory.append((cx, cy))

            # Draw box & center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Draw trajectory line
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Saved ball-tracked video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
