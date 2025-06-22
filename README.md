# option1
# Vehicle Tracking with Direction & Wrong-Way Detection


## File cấu trúc
* Input: Video `Option1.mp4`
* Output: Video `result_direction.mp4`
* (Tuỳ chọn) Log file: `vehicle_count_log.txt`

---
```
.
├── main.py                  # Code chính theo dõi và phân tích hướng
├── Detect_object.py          # (tuỳ chọn) File hỗ trợ phát hiện
├── requirements.txt          # Danh sách thư viện
├── Option1.mp4               # Video đầu vào
├── result_direction.mp4      # Video đầu ra
├── yolov8m.pt                # Model YOLOv8
├── .gitignore                # Bỏ qua file không cần thiết khi upload git
```

##  Yêu cầu

```
opencv-python
ultralytics
norfair
numpy
```

Hoặc cài từ file `requirements.txt`:

```
pip install -r requirements.txt
```

##  Phát hiện đối tượng và Theo dõi đối tượng và quản lý ID

* Phát hiện đối tượng phương tiện (ví dụ: YOLO, v.v.)
* Phân loại loại phương tiện (ví dụ: ô tô con, xe buýt, xe tải, v.v.)
* Tính toán số lượng phương tiện và số lượng người đi bộ trên đường
* Theo dõi đa đối tượng
* Gán và duy trì ID duy nhất cho mỗi đối tượng
* Nhận diện lại (Re-ID) các phương tiện bị che khuất hoặc trùng lặp


### 💻 Cách chạy

```
python main.py
```
## Phân tích hướng di chuyển và đi ngược chiều
* Xác định hướng dựa trên sự thay đổi tọa độ giữa các khung hình
* Phân tích vào/ra khu vực dựa trên vùng định trước (Line/ROI)
* Hiển thị quỹ đạo di chuyển của phương tiện
* Phát hiện và đánh dấu xe đi ngược chiều
### Ý tưởng
* Chia khung hình thành nhiều ô nhỏ  bằng cách kẻ các line ngang và dọc, phân vùng đều khung hình.
* Mỗi phương tiện sẽ được xác định thuộc ô nào dựa trên tọa độ bounding box (thường là tọa độ tâm).
* Xác định hướng di chuyển của phương tiện bằng cách so sánh sự thay đổi tọa độ giữa các frame.
* Đếm số lượng phương tiện di chuyển theo từng hướng trong mỗi frame.
* Nếu có ≥ 2 phương tiện di chuyển cùng chiều → xác định đó là chiều đúng cho khung hình đó.
* So sánh hướng của từng phương tiện với chiều đúng → phát hiện phương tiện đi ngược chiều.
### Nhược điểm tiềm ẩn
1️⃣ Chiều đúng dễ thay đổi theo từng frame
* Vì chiều đúng xác định dựa vào số lượng phương tiện trong frame hiện tại, nên nếu số lượng phương tiện ít hoặc không đồng nhất giữa các frame, chiều đúng có thể dao động liên tục → gây nhầm lẫn khi xác định sai chiều.

2️⃣ Không xét đến ngữ cảnh tuyến đường
*  Hướng đúng không phụ thuộc vào cơ sở hạ tầng (ví dụ: làn đường hợp pháp) mà chỉ phụ thuộc vào số lượng xe trong frame → dễ sai nếu xe đông nhưng đi sai.

3️⃣ Với video có nhiều lane hoặc giao lộ phức tạp
*  Việc chia grid đều không phân biệt được từng lane thực tế → khó xác định chính xác hướng đúng trên từng lane.
### Dùng atan2 để lấy góc di chuyển
```
import math
angle = math.degrees(math.atan2(-delta_y, delta_x))  # đổi trục Y vì ảnh ngược
angle = (angle + 360) % 360  # chuẩn hóa 0-360
```
 * nguyên tắc chia 360° thành 8 quãng 45° mỗi quãng (tương đương 8 hướng chính) như sau sẽ bao phủ toàn bộ vòng tròn mà không bỏ sót:
```
Hướng	Khoảng góc (độ)
East (→)	[337.5°, 360°) ∪ [0°, 22.5°)
Northeast (↗)	[22.5°, 67.5°)
North (↑)	[67.5°, 112.5°)
Northwest (↖)	[112.5°, 157.5°)
West (←)	[157.5°, 202.5°)
Southwest (↙)	[202.5°, 247.5°)
South (↓)	[247.5°, 292.5°)
Southeast (↘)	[292.5°, 337.5°)
```
* (0, 255, 0) — xanh lá — vẽ cho xe đi đúng chiều
* (0, 0, 255) — đỏ — vẽ cho xe đi sai chiều
```
color = (0, 0, 255) if is_wrong else (0, 255, 0)
cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
```
