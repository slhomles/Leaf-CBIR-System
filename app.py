"""
Streamlit UI — Leaf Image Similarity Search
Tìm top-5 ảnh lá giống nhất theo Cosine distance tổ hợp trọng số 4 nhóm đặc trưng.

Chạy:
    streamlit run app.py
"""

import os
import sys
import tempfile

import cv2
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from features.preprocess import preprocess
from features.pipeline import extract_all
from features.zscore import normalize_all
from db.database import SessionLocal
from db.crud import search_weighted_cosine, DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Cấu hình trang
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Leaf Search",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Leaf Similarity Search")
st.caption(
    "Upload một ảnh lá → tìm 5 ảnh giống nhất theo Cosine distance "
    "tổ hợp trọng số trên 4 nhóm: Shape (10) · Color (72) · Texture (38) · Venation (5)"
)

# ---------------------------------------------------------------------------
# Sidebar: trọng số 4 nhóm
# ---------------------------------------------------------------------------

st.sidebar.header("Trọng số tổ hợp")
st.sidebar.caption("Tổng các trọng số sẽ được tự chuẩn hóa về 1.")

w_shape    = st.sidebar.slider("Shape",    0.0, 1.0, DEFAULT_WEIGHTS["shape"],    0.05)
w_color    = st.sidebar.slider("Color",    0.0, 1.0, DEFAULT_WEIGHTS["color"],    0.05)
w_texture  = st.sidebar.slider("Texture",  0.0, 1.0, DEFAULT_WEIGHTS["texture"],  0.05)
w_venation = st.sidebar.slider("Venation", 0.0, 1.0, DEFAULT_WEIGHTS["venation"], 0.05)

w_sum = w_shape + w_color + w_texture + w_venation
if w_sum <= 0:
    st.sidebar.error("Tổng trọng số phải > 0. Đang dùng mặc định 0.25 đều nhau.")
    weights = DEFAULT_WEIGHTS
else:
    weights = {
        "shape":    w_shape    / w_sum,
        "color":    w_color    / w_sum,
        "texture":  w_texture  / w_sum,
        "venation": w_venation / w_sum,
    }
    st.sidebar.caption(
        f"Sau chuẩn hóa: shape={weights['shape']:.2f}, color={weights['color']:.2f}, "
        f"texture={weights['texture']:.2f}, venation={weights['venation']:.2f}"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_results(results: list[dict]):
    """Hiển thị top-5 kết quả theo dạng lưới 5 cột với breakdown distance."""
    cols = st.columns(5)
    for col, r in zip(cols, results):
        with col:
            if os.path.exists(r["path"]):
                st.image(r["path"], use_container_width=True)
            else:
                st.warning("Không tìm thấy ảnh")
            st.caption(
                f"**{r['file_name']}**\n\n"
                f"total = `{r['distance']:.4f}`\n\n"
                f"shape={r['shape_dist']:.3f} · color={r['color_dist']:.3f}\n\n"
                f"texture={r['texture_dist']:.3f} · vein={r['venation_dist']:.3f}"
            )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Chọn ảnh lá (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with col_preview:
    if uploaded:
        st.image(uploaded, caption="Ảnh gốc", width=220)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

if uploaded and st.button("🔍 Tìm kiếm", type="primary"):
    suffix = os.path.splitext(uploaded.name)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Đang tiền xử lý ảnh..."):
            preprocessed_bgr = preprocess(tmp_path)
            cv2.imwrite(tmp_path, preprocessed_bgr)

        preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
        st.image(
            preprocessed_rgb,
            caption="Ảnh sau tiền xử lý (nền đen — dùng để tìm kiếm)",
            width=220,
        )

        with st.spinner("Đang trích xuất đặc trưng (4 nhóm)..."):
            raw_vectors = extract_all(tmp_path)
            norm_vectors = normalize_all(raw_vectors)

        db = SessionLocal()
        st.divider()

        with st.spinner("Đang tìm Cosine weighted..."):
            results = search_weighted_cosine(
                norm_vectors, top_k=5, db=db, weights=weights
            )

        st.markdown(
            "<span style='color:#2196F3;font-size:18px;font-weight:bold'>"
            "● Cosine Distance (weighted)</span>",
            unsafe_allow_html=True,
        )
        show_results(results)
        db.close()

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Lỗi: {e}")
    finally:
        os.unlink(tmp_path)
