"""Các hàm tìm kiếm vector similarity — hỗ trợ 3 phương pháp: Cosine, L2, Manhattan."""

import os
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.models import LeafImage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def _resolve_path(file_name: str) -> str:
    """Trả về đường dẫn thực tế của ảnh trên disk."""
    return os.path.join(DATA_DIR, file_name)


def _vec_to_str(vec: list[float]) -> str:
    """Chuyển list float thành chuỗi pgvector '[x1,x2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _rows_to_dicts(rows) -> list[dict]:
    return [
        {
            "image_id": r.image_id,
            "file_name": r.file_name,
            "distance": float(r.distance),
            "path": _resolve_path(r.file_name),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Phương pháp 1: Cosine Distance  (pgvector <=>)
# ---------------------------------------------------------------------------

def search_cosine(query_vec: list[float], top_k: int, db: Session) -> list[dict]:
    """
    Tìm top_k ảnh gần nhất theo Cosine distance.
    Dùng toán tử pgvector <=> — hỗ trợ HNSW/IVFFlat index.

    Returns:
        list[dict] sắp xếp theo distance tăng dần (0 = giống nhất).
    """
    sql = text("""
        SELECT image_id, file_name,
               feature_vector <=> CAST(:vec AS vector) AS distance
        FROM leaf_images
        ORDER BY distance
        LIMIT :k
    """)
    rows = db.execute(sql, {"vec": _vec_to_str(query_vec), "k": top_k}).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Phương pháp 2: L2 / Euclidean Distance  (pgvector <->)
# ---------------------------------------------------------------------------

def search_l2(query_vec: list[float], top_k: int, db: Session) -> list[dict]:
    """
    Tìm top_k ảnh gần nhất theo Euclidean (L2) distance.
    Dùng toán tử pgvector <-> — hỗ trợ HNSW/IVFFlat index.

    Returns:
        list[dict] sắp xếp theo distance tăng dần.
    """
    sql = text("""
        SELECT image_id, file_name,
               feature_vector <-> CAST(:vec AS vector) AS distance
        FROM leaf_images
        ORDER BY distance
        LIMIT :k
    """)
    rows = db.execute(sql, {"vec": _vec_to_str(query_vec), "k": top_k}).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Phương pháp 3: Manhattan / L1 Distance  (tính trong Python)
# ---------------------------------------------------------------------------

def search_l1(query_vec: list[float], top_k: int, db: Session) -> list[dict]:
    """
    Tìm top_k ảnh gần nhất theo Manhattan (L1) distance.
    Fetch toàn bộ vectors về Python rồi tính thủ công
    (pgvector chưa hỗ trợ L1 natively).

    Returns:
        list[dict] sắp xếp theo distance tăng dần.
    """
    records = db.query(
        LeafImage.image_id,
        LeafImage.file_name,
        LeafImage.feature_vector,
    ).all()

    q = np.array(query_vec, dtype=np.float64)

    results = []
    for r in records:
        v = np.array(list(r.feature_vector), dtype=np.float64)
        dist = float(np.sum(np.abs(q - v)))
        results.append({
            "image_id": r.image_id,
            "file_name": r.file_name,
            "distance": dist,
            "path": _resolve_path(r.file_name),
        })

    results.sort(key=lambda x: x["distance"])
    return results[:top_k]
