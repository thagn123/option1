import cv2
import numpy as np
import math
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import defaultdict

VIDEO_PATH      = 'Road_2.mp4'
OUTPUT_VIDEO    = 'result_grid_cell_direction.mp4'
OUTPUT_LOG      = 'vehicle_cell_direction_log.txt'
FPS             = 20
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorbike']

# Grid config
ROWS, COLS = 3, 3
CALIB_FRAMES = 10  # số frame đầu để tính hướng đúng mỗi ô

model = YOLO('yolov8m.pt')
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=20,
    past_detections_length=50
)


cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

row_h = h // ROWS
col_w = w // COLS


log_file = open(OUTPUT_LOG, 'w', encoding='utf-8')
log_file.write("Vehicle Cell Direction Log\n===========================\n")


def yolo_to_detections(results):
    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label in VEHICLE_CLASSES and box.conf[0] > 0.4:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dets.append(
                Detection(points=np.array([cx, cy]), scores=np.array([box.conf[0]]), data=(label, cx, cy))
            )
    return dets


def get_direction_from_angle(angle):
    if 337.5 <= angle or angle < 22.5: return 'east'
    if 22.5 <= angle < 67.5:        return 'northeast'
    if 67.5 <= angle < 112.5:       return 'north'
    if 112.5 <= angle < 157.5:      return 'northwest'
    if 157.5 <= angle < 202.5:      return 'west'
    if 202.5 <= angle < 247.5:      return 'southwest'
    if 247.5 <= angle < 292.5:      return 'south'
    return 'southeast'


grid_counts = {(r, c): defaultdict(int) for r in range(ROWS) for c in range(COLS)}
grid_dir    = {cell: None for cell in grid_counts}
frame_idx   = 0
# Lưu ô cuối cùng của mỗi object
tid_last_cell = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    res      = model(frame, verbose=False, conf=0.4)[0]
    dets     = yolo_to_detections(res)
    tracked  = tracker.update(detections=dets)

    # Temp đếm hướng mỗi ô trong frame hiện tại
    temp_counts = defaultdict(lambda: defaultdict(int))
    # Lưu hướng và ô cho mỗi object
    obj_info    = {}

    # Xác định hướng và ô của object
    for obj in tracked:
        cx, cy = obj.estimate[0]
        label, ox, oy = obj.last_detection.data
        tid = obj.id

        # Tính ô, clamp
        row = min(max(int(cy // row_h), 0), ROWS - 1)
        col = min(max(int(cx // col_w), 0), COLS - 1)
        cell = (row, col)

        # Tính hướng nếu có lịch sử
        if len(obj.past_detections) >= 2:
            px, py = obj.past_detections[-2].points[0]
            dx, dy = cx - px, cy - py
            if abs(dx) + abs(dy) > 0:
                ang = math.degrees(math.atan2(-dy, dx)) % 360
                dname = get_direction_from_angle(ang)
                obj_info[tid] = (dname, cell)
                if frame_idx <= CALIB_FRAMES:
                    grid_counts[cell][dname] += 1
                else:
                    temp_counts[cell][dname] += 1

        # Vẽ object
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"{label}-{tid} C{cell}",
                    (int(cx), int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

        for cell, counts in grid_counts.items():
            if counts:
                grid_dir[cell] = max(counts, key=counts.get)

    for i in range(1, ROWS): cv2.line(frame, (0, i*row_h), (w, i*row_h), (100,100,100), 1)
    for j in range(1, COLS): cv2.line(frame, (j*col_w, 0), (j*col_w, h), (100,100,100), 1)

    for (r, c), d in grid_dir.items():
        txt = d if d else ('Calibrating' if frame_idx <= CALIB_FRAMES else 'Unknown')
        cv2.putText(frame, f"Cell {r,c}: {txt}",
                    (10, 20 + (r*COLS + c)*15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0) if d else (200, 200, 200), 1)

    for tid, (dname, cell) in obj_info.items():
        last = tid_last_cell.get(tid)
        # Khi object chuyển ô hoặc lần đầu
        if last != cell and grid_dir[cell]:
            correct = (dname == grid_dir[cell])
            # Tô xanh/đỏ
            for obj in tracked:
                if obj.id == tid:
                    cx, cy = obj.estimate[0]
                    color = (0,255,0) if correct else (0,0,255)
                    cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)
            if not correct:
                log_file.write(f"Wrong entry: ID {tid} Dir {dname} Cell {cell}\n")
        tid_last_cell[tid] = cell

    out.write(frame)
    cv2.imshow('Cell Dir Analysis', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release(); out.release(); log_file.close(); cv2.destroyAllWindows()
