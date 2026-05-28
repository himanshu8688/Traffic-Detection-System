import cv2
from ultralytics import YOLO
import math

model = YOLO("yolov8n.pt")

tracker = {}
next_id = 0
counted_ids = set()

def get_center(x1, y1, x2, y2):
    return int((x1+x2)/2), int((y1+y2)/2)

def detect_vehicles(video_path, output_path):
    global tracker, next_id, counted_ids

    tracker = {}
    counted_ids = set()
    next_id = 0

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    line_y = height // 2

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    total_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        # Detection
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in ["car", "truck", "bus", "motorbike"]:
                    cx, cy = get_center(x1, y1, x2, y2)
                    detections.append((cx, cy, x1, y1, x2, y2))

        # Tracking
        new_tracker = {}

        for cx, cy, x1, y1, x2, y2 in detections:
            matched = False

            for obj_id, (px, py, _, _, _, _) in tracker.items():
                if math.hypot(cx-px, cy-py) < 50:
                    new_tracker[obj_id] = (cx, cy, x1, y1, x2, y2)
                    matched = True

                    # Counting
                    if py < line_y and cy >= line_y:
                        if obj_id not in counted_ids:
                            total_count += 1
                            counted_ids.add(obj_id)

                    break

            if not matched:
                new_tracker[next_id] = (cx, cy, x1, y1, x2, y2)
                next_id += 1

        tracker = new_tracker

        # Density
        if total_count < 5:
            density = "LOW"
            color = (0,255,0)
        elif total_count < 15:
            density = "MEDIUM"
            color = (0,165,255)
        else:
            density = "HIGH"
            color = (0,0,255)

        # Draw
        for obj_id, (cx, cy, x1, y1, x2, y2) in tracker.items():
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

        cv2.line(frame, (0,line_y), (width,line_y), (0,255,255), 2)

        cv2.putText(frame, f"Count: {total_count}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.putText(frame, f"Density: {density}", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        out.write(frame)

    cap.release()
    out.release()