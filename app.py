"""
Streamlit UI — Leaf Image Similarity Search
Tìm top 5 ảnh lá giống nhất theo 3 phương pháp: Cosine, L2, Manhattan.

Chạy:
    streamlit run app.py
"""

import os
import sys
import tempfile

import cv2
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from features.preprocess import preprocess_image
from features.pipeline import extract_all
from features.zscore import normalize
from db.database import SessionLocal
from db import crud

# ---------------------------------------------------------------------------
# Cấu hình trang
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Leaf Search",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Leaf Similarity Search")
st.caption("Upload một ảnh lá → tìm 5 ảnh giống nhất theo 3 phương pháp đo khoảng cách")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METHOD_INFO = {
    "Cosine Distance": {
        "desc": "Đo góc giữa 2 vector — ít nhạy cảm với scale",
        "func": crud.search_cosine,
        "color": "#2196F3",
    },
    "L2 / Euclidean": {
        "desc": "Khoảng cách Euclid — hiệu quả sau Z-score normalize",
        "func": crud.search_l2,
        "color": "#4CAF50",
    },
    "Manhattan L1": {
        "desc": "Tổng |aᵢ - bᵢ| — robust hơn với outlier",
        "func": crud.search_l1,
        "color": "#FF9800",
    },
}


def show_results(results: list[dict]):
    """Hiển thị top-5 kết quả theo dạng lưới 5 cột."""
    cols = st.columns(5)
    for col, r in zip(cols, results):
        with col:
            if os.path.exists(r["path"]):
                st.image(r["path"], use_container_width=True)
            else:
                st.warning("Không tìm thấy ảnh")
            st.caption(f"**{r['file_name']}**\n\ndist = `{r['distance']:.4f}`")


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

if uploaded:
    if st.button("🔍 Tìm kiếm", type="primary"):
        suffix = os.path.splitext(uploaded.name)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            # Tiền xử lý: xóa nền + Bounding Box Crop
            with st.spinner("Đang xử lý Bounding Box Crop..."):
                preprocessed_path = preprocess_image(tmp_path)
                preprocessed_bgr = cv2.imread(preprocessed_path)
                # Đổi đường dẫn để extract_all lấy đúng file
                tmp_path = preprocessed_path

            # Hiển thị ảnh sau preprocessing để user kiểm tra
            preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
            st.image(preprocessed_rgb, caption="Ảnh sau tiền xử lý (nền đen — dùng để tìm kiếm)", width=220)

            with st.spinner("Đang trích xuất đặc trưng..."):
                raw_vec = extract_all(tmp_path)
                norm_vec = normalize(raw_vec)

            db = SessionLocal()

            st.divider()

            for method_name, info in METHOD_INFO.items():
                label_col, _ = st.columns([3, 7])
                with label_col:
                    st.markdown(
                        f"<span style='color:{info['color']};font-size:18px;font-weight:bold'>"
                        f"● {method_name}</span>",
                        unsafe_allow_html=True,
                    )
                st.caption(info["desc"])

                with st.spinner(f"Đang tìm với {method_name}..."):
                    results = info["func"](norm_vec, top_k=5, db=db)

                show_results(results)
                st.divider()

            db.close()

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Lỗi: {e}")
        finally:
            os.unlink(tmp_path)
