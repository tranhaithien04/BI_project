from __future__ import annotations

import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"
KNN_PATH = ARTIFACTS_DIR / "knn_model.pkl"
JOBS_INFO_PATH = ARTIFACTS_DIR / "jobs_info.pkl"
META_PATH = ARTIFACTS_DIR / "training_meta.pkl"

REQUIRED_COLUMNS = ["Job_ID", "Job_Name", "Job_Requirements"]


def resolve_data_path() -> Path:
    # Hỗ trợ cả tên file chuẩn và tên file đang có trong repo.
    candidates = [
        BASE_DIR / "Job_dataset.csv",
        BASE_DIR / "Job Datsset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Khong tim thay file du lieu. Da thu: "
        + ", ".join(str(path) for path in candidates)
    )


def clean_text(value: object) -> str:
    # Chuẩn hóa text đầu vào để TF-IDF học ổn định hơn.
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def validate_columns(df: pd.DataFrame) -> None:
    # Fail sớm nếu dữ liệu thiếu cột bắt buộc cho pipeline.
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong CSV: {missing}")


def load_and_prepare_data(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    # Đọc CSV và tiền xử lý toàn bộ dữ liệu jobs.
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    validate_columns(df_raw)

    before_count = len(df_raw)

    # Chỉ giữ cột cần thiết để train.
    df_jobs = df_raw[REQUIRED_COLUMNS].copy()
    df_jobs["Job_Requirements"] = df_jobs["Job_Requirements"].apply(clean_text)
    df_jobs["Job_Name"] = df_jobs["Job_Name"].fillna("").astype(str).str.strip()

    # Ưu tiên dedup theo Job_ID để tránh train lặp theo cùng vị trí công việc.
    df_jobs = df_jobs.drop_duplicates(subset=["Job_ID"]).reset_index(drop=True)

    after_count = len(df_jobs)
    dedup_ratio = 0.0
    if before_count > 0:
        dedup_ratio = round((before_count - after_count) / before_count, 4)

    # Stats dùng để theo dõi chất lượng dữ liệu qua mỗi lần train.
    stats = {
        "rows_before": before_count,
        "rows_after": after_count,
        "dedup_ratio": dedup_ratio,
    }
    return df_jobs, stats


def train_models(df_jobs: pd.DataFrame) -> tuple[TfidfVectorizer, NearestNeighbors]:
    # Model 1: vectorizer biến text yêu cầu công việc thành vector sparse.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    job_vectors = vectorizer.fit_transform(df_jobs["Job_Requirements"])

    # Model 2: KNN học không gian vector để truy vấn láng giềng gần nhất.
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_model.fit(job_vectors)

    return vectorizer, knn_model


def save_artifacts(
    vectorizer: TfidfVectorizer,
    knn_model: NearestNeighbors,
    df_jobs: pd.DataFrame,
    stats: dict,
) -> None:
    # Tạo thư mục artifacts nếu chưa tồn tại.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with TFIDF_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)

    with KNN_PATH.open("wb") as f:
        pickle.dump(knn_model, f)

    with JOBS_INFO_PATH.open("wb") as f:
        pickle.dump(df_jobs, f)

    # Metadata giúp truy vết phiên bản và thống kê training.
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v1",
        "num_jobs": len(df_jobs),
        "stats": stats,
    }
    with META_PATH.open("wb") as f:
        pickle.dump(metadata, f)


def main() -> None:
    # Orchestrator cho luồng train offline end-to-end.
    print("[INFO] Bat dau training offline...")
    data_path = resolve_data_path()
    print(f"[INFO] Su dung du lieu: {data_path}")
    df_jobs, stats = load_and_prepare_data(data_path)
    print(
        "[INFO] So dong truoc dedup: {rows_before}, sau dedup: {rows_after}, ty le giam: {dedup_ratio}".format(
            **stats
        )
    )

    vectorizer, knn_model = train_models(df_jobs)
    save_artifacts(vectorizer, knn_model, df_jobs, stats)

    print("[INFO] Training hoan tat. Da xuat artifacts:")
    print(f"  - {TFIDF_PATH}")
    print(f"  - {KNN_PATH}")
    print(f"  - {JOBS_INFO_PATH}")
    print(f"  - {META_PATH}")


if __name__ == "__main__":
    main()
