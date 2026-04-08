from pathlib import Path
import pickle
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Nạp models vào RAM 1 lần khi service khởi động.
    load_artifacts()
    yield


# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Job Recommendation AI Service",
    description="Microservice gợi ý công việc bằng kiến trúc offline training + online matching",
    version="1.0.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
KNN_PATH = ARTIFACTS_DIR / "knn_model.pkl"
JOBS_INFO_PATH = ARTIFACTS_DIR / "jobs_info.pkl"

# Các biến global giữ model đã load để phục vụ realtime.
vectorizer = None
knn_model = None
df_jobs = None


def load_artifacts() -> None:
    global vectorizer, knn_model, df_jobs

    # Kiểm tra đủ 3 file artifacts trước khi phục vụ API.
    missing = [
        str(path) for path in [TFIDF_PATH, KNN_PATH, JOBS_INFO_PATH] if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Thiếu artifacts. Hãy chạy script train offline trước: "
            + ", ".join(missing)
        )

    # Load bộ dịch TF-IDF.
    with TFIDF_PATH.open("rb") as f:
        vectorizer = pickle.load(f)

    # Load bộ khớp KNN.
    with KNN_PATH.open("rb") as f:
        knn_model = pickle.load(f)

    # Load metadata job để map index -> thông tin trả về.
    with JOBS_INFO_PATH.open("rb") as f:
        jobs_obj = pickle.load(f)

    if isinstance(jobs_obj, pd.DataFrame):
        df_jobs = jobs_obj
    else:
        df_jobs = pd.DataFrame(jobs_obj)

    required_columns = {"Job_ID", "Job_Name", "Job_Requirements"}
    if not required_columns.issubset(df_jobs.columns):
        raise RuntimeError(
            "jobs_info.pkl thiếu cột bắt buộc: Job_ID, Job_Name, Job_Requirements"
        )

    if knn_model.n_samples_fit_ != len(df_jobs):
        raise RuntimeError(
            "Artifacts không đồng bộ: số mẫu KNN khác số jobs trong jobs_info"
        )


# 3. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ĐẦU VÀO VÀ ĐẦU RA (Pydantic Models)
class UserRequest(BaseModel):
    # Chuỗi kỹ năng đầu vào từ client.
    user_skills: str = Field(..., min_length=1)
    # Giới hạn top_k để tránh request quá nặng.
    top_k: int = Field(default=5, ge=1, le=20)

class JobResponse(BaseModel):
    job_id: int
    job_name: str
    match_score: float


# 4. TẠO API ENDPOINT XỬ LÝ GỢI Ý
@app.post("/api/recommend", response_model=list[JobResponse])
def get_job_recommendations(request: UserRequest):
    if vectorizer is None or knn_model is None or df_jobs is None:
        raise HTTPException(status_code=503, detail="Model artifacts chưa sẵn sàng")

    skills_to_check = request.user_skills.strip()
    if not skills_to_check:
        raise HTTPException(status_code=400, detail="user_skills không được để trống")

    try:
        # Dịch kỹ năng người dùng sang vector cùng không gian với jobs.
        user_vector = vectorizer.transform([skills_to_check])
        n_neighbors = min(request.top_k, len(df_jobs))

        # Truy vấn Top K job gần nhất theo cosine distance.
        distances, indices = knn_model.kneighbors(user_vector, n_neighbors=n_neighbors)

        recommendations = []
        for distance, idx in zip(distances[0], indices[0]):
            # Chuyển distance -> score để dễ đọc (cao hơn là phù hợp hơn).
            score = round(float(1 - distance), 4)
            if score > 0:
                recommendations.append(
                    JobResponse(
                        job_id=int(df_jobs.iloc[idx]["Job_ID"]),
                        job_name=str(df_jobs.iloc[idx]["Job_Name"]),
                        match_score=score
                    )
                )

        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống AI: {str(e)}") from e


# Endpoint kiểm tra trạng thái
@app.get("/health")
def health_check():
    return {
        "status": "AI Server is running",
        "artifacts_loaded": vectorizer is not None and knn_model is not None and df_jobs is not None,
        "total_unique_jobs_loaded": len(df_jobs) if df_jobs is not None else 0,
    }


if __name__ == "__main__":
    # Cho phep chay truc tiep bang: python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)