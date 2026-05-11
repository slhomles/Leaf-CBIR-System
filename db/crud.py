"""Tìm kiếm ảnh tương đồng — Cosine distance trên 4 vector + tổ hợp trọng số.

Theo spec: Total = W_shape·D_shape + W_color·D_color + W_texture·D_texture + W_venation·D_venation
với D_x là cosine distance pgvector (toán tử <=>) trên cột tương ứng.
"""

from __future__ import annotations

import json
import os

from sqlalchemy import text
from sqlalchemy.orm import Session

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

_BALANCED_PATH = os.path.join(PROJECT_ROOT, "data", "balanced_weights.json")
_FALLBACK_WEIGHTS: dict[str, float] = {
    "shape": 0.25,
    "color": 0.25,
    "texture": 0.25,
    "venation": 0.25,
}


def _load_default_weights() -> dict[str, float]:
    try:
        with open(_BALANCED_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(data[k]) for k in ("shape", "color", "texture", "venation")}
    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError, TypeError):
        return dict(_FALLBACK_WEIGHTS)


DEFAULT_WEIGHTS: dict[str, float] = _load_default_weights()


def _resolve_path(file_name: str) -> str:
    return os.path.join(DATA_DIR, file_name)


def _vec_to_str(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def search_weighted_cosine(
    query_vectors: dict[str, list[float]],
    top_k: int,
    db: Session,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Tìm top_k ảnh có distance tổng hợp nhỏ nhất.

    Args:
        query_vectors: dict với keys {"shape","color","texture","venation"}
                       — đã được Z-score normalize.
        top_k: số kết quả trả về.
        db: SQLAlchemy session.
        weights: trọng số 4 nhóm (mặc định 0.25 đều nhau).

    Returns:
        list[dict] sắp xếp theo total ASC. Mỗi item gồm:
          image_id, file_name, path, distance (= total),
          và breakdown từng thành phần shape_dist / color_dist / ...
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    sql = text(
        """
        SELECT
            image_id,
            file_name,
            (shape_vector    <=> CAST(:v_shape    AS vector)) AS d_shape,
            (color_vector    <=> CAST(:v_color    AS vector)) AS d_color,
            (texture_vector  <=> CAST(:v_texture  AS vector)) AS d_texture,
            (venation_vector <=> CAST(:v_venation AS vector)) AS d_venation,
            (
                :w_shape    * (shape_vector    <=> CAST(:v_shape    AS vector)) +
                :w_color    * (color_vector    <=> CAST(:v_color    AS vector)) +
                :w_texture  * (texture_vector  <=> CAST(:v_texture  AS vector)) +
                :w_venation * (venation_vector <=> CAST(:v_venation AS vector))
            ) AS total
        FROM leaf_images
        ORDER BY total ASC
        LIMIT :k
        """
    )

    rows = db.execute(
        sql,
        {
            "v_shape":    _vec_to_str(query_vectors["shape"]),
            "v_color":    _vec_to_str(query_vectors["color"]),
            "v_texture":  _vec_to_str(query_vectors["texture"]),
            "v_venation": _vec_to_str(query_vectors["venation"]),
            "w_shape":    w["shape"],
            "w_color":    w["color"],
            "w_texture":  w["texture"],
            "w_venation": w["venation"],
            "k":          top_k,
        },
    ).fetchall()

    return [
        {
            "image_id": r.image_id,
            "file_name": r.file_name,
            "path": _resolve_path(r.file_name),
            "distance": float(r.total),
            "shape_dist": float(r.d_shape),
            "color_dist": float(r.d_color),
            "texture_dist": float(r.d_texture),
            "venation_dist": float(r.d_venation),
        }
        for r in rows
    ]
