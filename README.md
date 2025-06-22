# option1
# Vehicle Tracking with Direction & Wrong-Way Detection

Hệ thống sử dụng YOLOv8 kết hợp Norfair để theo dõi phương tiện giao thông, xác định hướng di chuyển và phát hiện xe đi ngược chiều.

---

## 🔑 Tính năng chính

* Theo dõi đa phương tiện với ID duy nhất
* Đếm phương tiện khi qua line xác định
* Xác định hướng di chuyển (↑ đi lên, ↓ đi xuống)
* Phát hiện và đánh dấu phương tiện đi ngược chiều (màu đỏ)
* Hiển thị quỹ đạo di chuyển
* Ghi log kết quả vào file `.txt` (nếu cần mở rộng)

---

## 🚀 Yêu cầu

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

---

## 💻 Cách chạy

```
python main.py
```

* Input: Video `Option1.mp4`
* Output: Video `result_direction.mp4`
* (Tuỳ chọn) Log file: `vehicle_count_log.txt`

---

## 📂 File cấu trúc

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

