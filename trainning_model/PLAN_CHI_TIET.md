# Kế Hoạch Chi Tiết: Kiến Trúc Training Offline + Matching Online (TF-IDF + KNN)

## 1. Mục tiêu
- Tách rời quá trình huấn luyện và phục vụ dự đoán để tăng hiệu năng runtime.
- Huấn luyện model theo batch/offline từ `Job_dataset.csv`.
- Khi chạy online, chỉ load artifacts lên RAM và trả kết quả Top K nhanh theo request.

## 2. Phạm vi dữ liệu và đầu vào/đầu ra
### 2.1. Nguồn dữ liệu
- File nguồn: `Job_dataset.csv`
- Cột chính sử dụng:
- `Job_ID`
- `Job_Name`
- `Job_Requirements`
- (Tùy chọn) `User_ID`, `User_Skills` cho test nội bộ.

### 2.2. Đầu vào online
- API nhận payload gồm:
- `user_skills` (chuỗi kỹ năng, ưu tiên dùng trực tiếp)
- `top_k` (mặc định 5)

### 2.3. Đầu ra online
- Danh sách Top K jobs gần nhất theo cosine distance/similarity.
- Mỗi item gồm:
- `job_id`
- `job_name`
- `match_score` (chuẩn hóa về khoảng 0..1 để dễ hiểu)

## 3. Kiến trúc tổng thể
### 3.1. Offline (Training Pipeline)
1. Đọc `Job_dataset.csv`.
2. Làm sạch dữ liệu:
- Giữ các cột cần thiết.
- Xử lý thiếu dữ liệu: `Job_Requirements = ""` nếu null.
3. Lọc `Unique Jobs`:
- Khuyến nghị ưu tiên `drop_duplicates(subset=["Job_ID"])`.
- Nếu `Job_ID` không ổn định, fallback theo `Job_Name + Job_Requirements`.
4. Train Model 1 (Translator):
- `TfidfVectorizer.fit()` trên toàn bộ `Job_Requirements`.
- Sinh ma trận `job_vectors` bằng `transform()`.
5. Train Model 2 (Matcher):
- Khởi tạo `NearestNeighbors(metric="cosine", algorithm="brute")`.
- `knn.fit(job_vectors)`.
6. Export artifacts:
- `tfidf.pkl`
- `knn_model.pkl`
- `jobs_info.pkl` (dataframe/list map index -> metadata job)

### 3.2. Online (Backend Serving)
1. Backend startup:
- Load `tfidf.pkl`, `knn_model.pkl`, `jobs_info.pkl` vào RAM.
2. Mỗi API request:
- Nhận `user_skills`.
- `tfidf.transform([user_skills])` -> `user_vector`.
- `knn_model.kneighbors(user_vector, n_neighbors=top_k)`.
3. Hậu xử lý:
- Map index jobs từ `jobs_info.pkl`.
- Đổi distance cosine thành score: `score = 1 - distance`.
- Trả JSON kết quả.

## 4. Thiết kế file và thư mục đề xuất
- `trainning_model/`
- `main.py` (FastAPI online serving)
- `train_model.py` (script training offline)
- `artifacts/`
- `tfidf.pkl`
- `knn_model.pkl`
- `jobs_info.pkl`
- `requirements.txt`
- `PLAN_CHI_TIET.md`

## 5. Thiết kế chi tiết pipeline offline
### 5.1. Bước 1: Data ingestion
- Đọc CSV bằng pandas.
- Validate cột bắt buộc: `Job_ID`, `Job_Name`, `Job_Requirements`.
- Nếu thiếu cột, fail sớm với log rõ ràng.

### 5.2. Bước 2: Data cleaning & dedup
- Chuẩn hóa text:
- lower-case
- trim spaces
- thay nhiều khoảng trắng thành 1 khoảng trắng
- Lọc duplicate.
- Thống kê số lượng:
- số dòng ban đầu
- số dòng sau dedup
- tỉ lệ giảm trùng lặp

### 5.3. Bước 3: Train TF-IDF
- Cấu hình khuyến nghị ban đầu:
- `ngram_range=(1,2)`
- `min_df=1` (hoặc 2 nếu dữ liệu lớn)
- `max_features=5000` (tùy kích thước dữ liệu)
- Fit trên `Job_Requirements` đã clean.

### 5.4. Bước 4: Train KNN matcher
- Model: `NearestNeighbors`
- Cấu hình:
- `n_neighbors` không cố định khi train (set khi query)
- `metric="cosine"`
- `algorithm="brute"` phù hợp sparse matrix
- Fit trực tiếp trên ma trận TF-IDF jobs.

### 5.5. Bước 5: Export artifacts
- Dùng `pickle` hoặc `joblib`.
- `jobs_info.pkl` cần chứa tối thiểu:
- `Job_ID`
- `Job_Name`
- `Job_Requirements` (để debug/giải thích)
- Lưu thêm metadata training:
- timestamp
- số jobs
- version pipeline

## 6. Thiết kế chi tiết backend online
### 6.1. Startup sequence
- Load đủ 3 artifacts.
- Kiểm tra tính đồng bộ:
- số hàng `jobs_info` phải khớp số vector mà `knn_model` đã fit.
- Nếu mismatch, chặn service startup.

### 6.2. API contract đề xuất
- Endpoint: `POST /api/recommend`
- Request:
```json
{
  "user_skills": "python, fastapi, sql, machine learning",
  "top_k": 5
}
```
- Response:
```json
[
  {
    "job_id": 101,
    "job_name": "Backend Python Engineer",
    "match_score": 0.8123
  }
]
```

### 6.3. Logic request
1. Validate `user_skills` không rỗng.
2. Validate `top_k` trong ngưỡng hợp lệ (ví dụ: 1..20).
3. Vector hóa user skills bằng TF-IDF đã load.
4. Query KNN để lấy distances + indices.
5. Convert score = `1 - distance`.
6. Trả kết quả đã sort giảm dần theo `match_score`.

### 6.4. Error handling
- `400`: input không hợp lệ.
- `500`: lỗi transform/model.
- `503`: artifacts chưa load hoặc load lỗi.

## 7. Kế hoạch triển khai theo giai đoạn
### Giai đoạn A: Chuẩn hóa training offline
- Tách logic training ra `train_model.py`.
- Sinh 3 file artifacts vào `artifacts/`.
- In log thống kê sau mỗi lần train.

### Giai đoạn B: Chuyển backend sang chế độ load artifacts
- Loại bỏ việc fit model trực tiếp trong `main.py`.
- Chỉ load artifacts khi startup.
- Cập nhật endpoint dùng KNN thay cho tính cosine toàn bộ trực tiếp.

### Giai đoạn C: Kiểm thử và xác nhận chất lượng
- Unit test cho:
- data cleaning
- dedup
- artifact loading
- inference flow
- Integration test API với bộ mẫu.

## 8. Tiêu chí nghiệm thu (Definition of Done)
- Có thể chạy training offline độc lập và sinh đủ 3 artifacts.
- Backend khởi động thành công khi có artifacts hợp lệ.
- API `POST /api/recommend` trả Top K jobs trong thời gian mục tiêu.
- Đảm bảo kết quả nhất quán giữa các lần gọi cùng input.

## 9. Chỉ số vận hành khuyến nghị
- Latency API P50/P95.
- Tỉ lệ request lỗi.
- Thời gian load artifacts khi startup.
- Số lượng jobs trong artifacts theo từng version.

## 10. Rủi ro và cách giảm thiểu
- Dữ liệu nhiễu/trùng lặp nhiều -> tăng sai lệch gợi ý:
- Bổ sung bước clean mạnh hơn và quy tắc dedup rõ ràng.
- OOV (từ ngoài từ điển TF-IDF):
- Cập nhật train định kỳ theo dữ liệu mới.
- Artifact mismatch giữa các phiên bản:
- Dùng cùng `version_id` cho bộ 3 file khi xuất và khi load.

## 11. Lịch chạy đề xuất
- Training offline định kỳ: mỗi ngày/tuần tùy tần suất cập nhật data.
- Triển khai artifacts theo phiên bản (rolling update).
- Có thể fallback về phiên bản artifacts trước đó nếu phát sinh lỗi.

## 12. Checklist thực thi nhanh
1. Tạo `train_model.py` và tách pipeline offline.
2. Tạo thư mục `artifacts/`.
3. Huấn luyện và export `tfidf.pkl`, `knn_model.pkl`, `jobs_info.pkl`.
4. Refactor `main.py` để load artifacts khi startup.
5. Cập nhật endpoint recommend dùng `kneighbors`.
6. Viết test và benchmark latency cơ bản.
7. Chốt release notes + version artifacts.

---
Plan này bám sát kiến trúc bạn mô tả: Train offline để tạo bộ dịch + bộ khớp, backend online chỉ load model lên RAM và truy vấn Top K theo thời gian thực.
