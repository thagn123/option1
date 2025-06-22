import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import defaultdict

VIDEO_PATH = 'Road_1.mp4'
OUTPUT_VIDEO = 'result_reid.mp4'
OUTPUT_LOG = 'vehicle_count_log.txt'
FPS = 20
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorbike']
model = YOLO('yolov8m.pt')
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=20,
    past_detections_length=50
)

vehicle_counter = defaultdict(set)

cap = cv2.VideoCapture(VIDEO_PATH)
w, h = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))
y_line = int(h * 0.5)

log_file = open(OUTPUT_LOG, 'w')
log_file.write("Vehicle Counting Log\n")
log_file.write("===================\n")

def yolo_to_detections(yolo_results):
    detections = []
    for box in yolo_results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in VEHICLE_CLASSES and box.conf[0] > 0.4:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append(Detection(points=np.array([cx, cy]), scores=np.array([box.conf[0]]), data=label))
    return detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.4)[0]
    detections = yolo_to_detections(results)
    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        label = obj.last_detection.data
        track_id = obj.id

        if track_id not in vehicle_counter[label]:
            prev_y = obj.past_detections[-2].points[0][1] if len(obj.past_detections) >= 2 else cy
            if prev_y < y_line <= cy:
                vehicle_counter[label].add(track_id)
                log_file.write(f"{label} ID {track_id} crossed at Y={int(cy)}\n")

        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{label}-{track_id}", (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for past in obj.past_detections:
            px, py = past.points[0]
            cv2.circle(frame, (int(px), int(py)), 2, (255, 0, 0), -1)

    cv2.line(frame, (0, y_line), (w, y_line), (0, 0, 255), 2)

    y_offset = 30
    for idx, (label, ids) in enumerate(vehicle_counter.items()):
        text = f"{label}: {len(ids)}"
        cv2.putText(frame, text, (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Traffic ReID Analysis", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
