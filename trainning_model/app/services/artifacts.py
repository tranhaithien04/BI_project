from __future__ import annotations

import pickle
import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from ..config import JOBS_INFO_PATH, KNN_PATH, TFIDF_PATH

vectorizer: Any | None = None
knn_model: Any | None = None
df_jobs: pd.DataFrame | None = None
artifact_error: str | None = None


def _skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9#]+", "", value.lower())


_SKILL_ALIASES_BY_KEY = {
    _skill_key("py"): "python",
    _skill_key("python3"): "python",
    _skill_key("fast api"): "fastapi",
    _skill_key("js"): "javascript",
    _skill_key("reactjs"): "react",
    _skill_key("react.js"): "react",
    _skill_key("nodejs"): "node.js",
    _skill_key("node.js"): "node.js",
    _skill_key("ts"): "typescript",
    _skill_key("postgres"): "postgresql",
    _skill_key("postgre"): "postgresql",
    _skill_key("mssql"): "sql server",
    _skill_key("sqlserver"): "sql server",
    _skill_key("asp net"): "asp.net",
    _skill_key("dotnet"): ".net",
    _skill_key("golang"): "go",
}


def _normalize_skill_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[+&]", " ", normalized)
    normalized = re.sub(r"[^\w#./ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _canonicalize_skill(value: str) -> str:
    normalized = _normalize_skill_text(value)
    if not normalized:
        return ""
    return _SKILL_ALIASES_BY_KEY.get(_skill_key(normalized), normalized)


def _split_skills(value: str) -> list[str]:
    raw_parts = re.split(r"[,;|/\n]+", value)
    result: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        canonical = _canonicalize_skill(part)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)

    return result


def _pair_similarity(candidate_skill: str, required_skill: str) -> float:
    if candidate_skill == required_skill:
        return 1.0
    if candidate_skill in required_skill or required_skill in candidate_skill:
        return 0.92

    candidate_tokens = set(candidate_skill.split())
    required_tokens = set(required_skill.split())
    jaccard = 0.0
    if candidate_tokens and required_tokens:
        union = candidate_tokens | required_tokens
        intersection = candidate_tokens & required_tokens
        jaccard = len(intersection) / len(union)

    sequence = SequenceMatcher(None, candidate_skill, required_skill).ratio()
    return max(jaccard, sequence)


def _skill_coverage_score(user_skill_items: list[str], required_skill_items: list[str]) -> float:
    if not user_skill_items or not required_skill_items:
        return 0.0

    total = 0.0
    for required_skill in required_skill_items:
        best_score = max(
            (_pair_similarity(user_skill, required_skill) for user_skill in user_skill_items),
            default=0.0,
        )
        if best_score < 0.5:
            best_score = 0.0
        total += best_score

    return total / len(required_skill_items)


def _cosine_similarity_from_text(left_text: str, right_text: str) -> float:
    if vectorizer is None:
        return 0.0

    left_vector = vectorizer.transform([left_text])
    right_vector = vectorizer.transform([right_text])

    numerator = float(left_vector.multiply(right_vector).sum())
    left_norm = float(left_vector.multiply(left_vector).sum()) ** 0.5
    right_norm = float(right_vector.multiply(right_vector).sum()) ** 0.5

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def set_artifact_error(message: str | None) -> None:
    global artifact_error
    artifact_error = message


def get_artifact_error() -> str | None:
    return artifact_error


def get_loaded_jobs_count() -> int:
    return len(df_jobs) if df_jobs is not None else 0


def load_artifacts() -> None:
    global vectorizer, knn_model, df_jobs

    missing = [
        str(path) for path in [TFIDF_PATH, KNN_PATH, JOBS_INFO_PATH] if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Thieu artifacts. Hay chay train_model.py truoc: " + ", ".join(missing)
        )

    with TFIDF_PATH.open("rb") as tfidf_file:
        vectorizer = pickle.load(tfidf_file)

    with KNN_PATH.open("rb") as knn_file:
        knn_model = pickle.load(knn_file)

    with JOBS_INFO_PATH.open("rb") as jobs_file:
        jobs_obj = pickle.load(jobs_file)

    if isinstance(jobs_obj, pd.DataFrame):
        df_jobs = jobs_obj
    else:
        df_jobs = pd.DataFrame(jobs_obj)

    required_columns = {"Job_ID", "Job_Name", "Job_Requirements"}
    if not required_columns.issubset(df_jobs.columns):
        raise RuntimeError("jobs_info.pkl thieu cot: Job_ID, Job_Name, Job_Requirements")

    if hasattr(knn_model, "n_samples_fit_") and knn_model.n_samples_fit_ != len(df_jobs):
        raise RuntimeError("Artifacts khong dong bo giua KNN va jobs_info")

    set_artifact_error(None)


def recommend_jobs(user_skills: str, top_k: int = 5) -> list[dict[str, Any]]:
    if artifact_error is not None or vectorizer is None or knn_model is None or df_jobs is None:
        raise RuntimeError(artifact_error or "Model artifacts chua san sang")

    cleaned_skills = user_skills.strip().lower()
    if not cleaned_skills:
        raise ValueError("user_skills khong duoc de trong")

    user_vector = vectorizer.transform([cleaned_skills])
    n_neighbors = min(max(1, top_k), len(df_jobs))
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=n_neighbors)

    results: list[dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        score = round(float(1 - distance), 4)
        job = df_jobs.iloc[int(idx)]
        results.append(
            {
                "job_id": int(job["Job_ID"]),
                "job_name": str(job["Job_Name"]),
                "match_score": max(0.0, score),
                "job_requirements": str(job["Job_Requirements"]),
            }
        )

    return sorted(results, key=lambda item: item["match_score"], reverse=True)


def score_job_match(user_skills: str, job_requirements: str) -> float | None:
    if artifact_error is not None or vectorizer is None:
        return None

    cleaned_skills = user_skills.strip().lower()
    cleaned_requirements = job_requirements.strip().lower()
    if not cleaned_skills or not cleaned_requirements:
        return None

    user_skill_items = _split_skills(cleaned_skills)
    required_skill_items = _split_skills(cleaned_requirements)

    coverage_score = _skill_coverage_score(user_skill_items, required_skill_items)
    cosine_score = _cosine_similarity_from_text(cleaned_skills, cleaned_requirements)

    # Khi danh sach skills khop gan nhu tuyet doi, uu tien tra ve 100%.
    if coverage_score >= 0.999:
        return 1.0

    # Van giu cosine score tu model de xep hang cac case chua khop hoan toan.
    if coverage_score >= 0.85:
        final_score = 0.9 * coverage_score + 0.1 * cosine_score
    else:
        final_score = 0.65 * coverage_score + 0.35 * cosine_score

    return round(max(0.0, min(1.0, final_score)), 4)
