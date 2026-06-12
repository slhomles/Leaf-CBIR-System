"""
Thực hiện Grid Search Optimization để tìm trọng số tốt nhất cho mAP@5.
Tạo các tổ hợp [w_shape, w_color, w_texture, w_venation] tổng = 1.0, bước nhảy 0.05.
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm
import random

# Fix encoding console on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage

# Bảng mã ID các loài của Flavia (bao gồm cả các biến thể mở rộng đến 3621)
FLAVIA_RANGES = [
    (1001, 1059), (1060, 1122), (1123, 1194), (1195, 1267), (1268, 1323),
    (1324, 1385), (1386, 1437), (1438, 1496), (1497, 1555), (1556, 1616),
    (1617, 1675), (1676, 1733), (1734, 1792), (1793, 1873), (1874, 1932),
    (1933, 1993), (1994, 2050), (2051, 2113), (2114, 2165), (2166, 2230),
    (2231, 2290), (2291, 2346), (2347, 2423), (2424, 2485), (2486, 2546),
    (2547, 2612), (2613, 2675), (2676, 2734), (2735, 2789), (2790, 2849),
    (2850, 2909), (2910, 2969), (2970, 3029), (3030, 3089), (3090, 3149),
    (3150, 3209), (3210, 3269), (3270, 3329), (3330, 3389), (3390, 3449),
    (3450, 3509), (3510, 3569), (3570, 3621)
]

def get_label(image_id: int) -> int:
    """Xác định ID loài dựa trên danh mục ID của Flavia."""
    for i, (start, end) in enumerate(FLAVIA_RANGES):
        if start <= image_id <= end:
            return i
    return -1

def compute_ap(retrieved_labels: np.ndarray, true_label: int, top_k: int) -> float:
    """Tính Average Precision @ K cho một truy vấn đơn lẻ."""
    hits = 0
    score_sum = 0.0
    for i, label in enumerate(retrieved_labels[:top_k]):
        if label == true_label:
            hits += 1
            score_sum += hits / (i + 1)
            
    if hits == 0:
        return 0.0
    return score_sum / top_k

def generate_weights(step=0.05):
    """Sinh ra tất cả các hoán vị [w_shape, w_color, w_texture, w_venation] có tổng = 1.0"""
    steps = int(round(1.0 / step))
    weights = []
    # Dùng vòng lặp vét cạn 4 tầng
    for s in range(steps + 1):
        for c in range(steps + 1 - s):
            for t in range(steps + 1 - s - c):
                v = steps - s - c - t
                weights.append((s * step, c * step, t * step, v * step))
    return weights

def run_grid_search():
    # 1. Đọc dữ liệu
    print("1. Đang đọc dữ liệu từ Database vào RAM để tăng tốc...")
    db = SessionLocal()
    records = db.query(
        LeafImage.image_id,
        LeafImage.shape_vector,
        LeafImage.color_vector,
        LeafImage.texture_vector,
        LeafImage.venation_vector
    ).all()
    db.close()
    
    if not records:
        print("Database trống!")
        return

    # Trích xuất nhãn và ID
    image_ids = np.array([r.image_id for r in records])
    labels = np.array([get_label(idx) for idx in image_ids])
    
    # Cảnh báo nếu có ảnh không xác định được loài
    unknowns = np.sum(labels == -1)
    if unknowns > 0:
        print(f"Cảnh báo: Có {unknowns} ảnh không khớp với bất kỳ loài Flavia nào.")
    
    # Chuyển đổi vector sang numpy matrix
    M_shape = np.array([list(r.shape_vector) for r in records], dtype=np.float32)
    M_color = np.array([list(r.color_vector) for r in records], dtype=np.float32)
    M_texture = np.array([list(r.texture_vector) for r in records], dtype=np.float32)
    M_venation = np.array([list(r.venation_vector) for r in records], dtype=np.float32)

    # 2. Lập tập kiểm thử (Validation Set) - Lấy ngẫu nhiên 50 ảnh
    random.seed(42)  # Cố định seed để kết quả ổn định qua nhiều lần chạy
    valid_indices = random.sample(range(len(image_ids)), 50)
    query_labels = labels[valid_indices]
    
    print(f"2. Đã tạo tập Validation Set gồm 50 queries.")
    
    # 3. Tiền tính toán ma trận khoảng cách Cosine
    # Cosine Distance = 1 - Cosine Similarity
    def normalize_l2(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return M / norms
        
    print("3. Đang tiền tính toán các ma trận khoảng cách (Cosine Distance)...")
    N_shape = normalize_l2(M_shape)
    N_color = normalize_l2(M_color)
    N_texture = normalize_l2(M_texture)
    N_venation = normalize_l2(M_venation)
    
    # Lọc ra các vector truy vấn (N_query x Dimension)
    Q_shape = N_shape[valid_indices]
    Q_color = N_color[valid_indices]
    Q_texture = N_texture[valid_indices]
    Q_venation = N_venation[valid_indices]
    
    # Tính ma trận Cosine distance: (N_query x N_database)
    D_shape = 1.0 - np.dot(Q_shape, N_shape.T)
    D_color = 1.0 - np.dot(Q_color, N_color.T)
    D_texture = 1.0 - np.dot(Q_texture, N_texture.T)
    D_venation = 1.0 - np.dot(Q_venation, N_venation.T)
    
    # 4. Sinh không gian nghiệm
    weights_space = generate_weights(step=0.05)
    print(f"\n4. Bắt đầu Grid Search: Tổng số tổ hợp cần duyệt là {len(weights_space)}")
    
    best_map = -1.0
    best_weights = None
    
    # Quét không gian nghiệm
    for w in tqdm(weights_space, desc="Evaluating"):
        ws, wc, wt, wv = w
        
        # Tính khoảng cách tổng hợp (Weighted Distance) - kích thước: (50 x N_database)
        D_total = ws * D_shape + wc * D_color + wt * D_texture + wv * D_venation
        
        # Tìm top 5 kết quả nhỏ nhất cho từng query
        top_k = 5
        # np.argsort để lấy index của ảnh có khoảng cách nhỏ nhất
        top_k_indices = np.argsort(D_total, axis=1)[:, :top_k]
        
        # Tính mAP@5
        ap_sum = 0.0
        for i in range(len(valid_indices)):
            retrieved_labels = labels[top_k_indices[i]]
            ap_sum += compute_ap(retrieved_labels, query_labels[i], top_k)
            
        mAP = ap_sum / len(valid_indices)
        
        if mAP > best_map:
            best_map = mAP
            best_weights = w
            
    # Chốt nghiệm
    print("\n" + "="*50)
    print(f"🥇 HOAN TAT! TIM THAY TO HOP TOI UU NHAT:")
    print(f"   mAP@5 dat: {best_map:.4f}")
    print(f"   w_shape    = {best_weights[0]:.2f}")
    print(f"   w_color    = {best_weights[1]:.2f}")
    print(f"   w_texture  = {best_weights[2]:.2f}")
    print(f"   w_venation = {best_weights[3]:.2f}")
    print("="*50)
    
    # Lưu vào JSON
    out_path = os.path.join(PROJECT_ROOT, "data", "grid_search_weights.json")
    payload = {
        "shape": round(best_weights[0], 2),
        "color": round(best_weights[1], 2),
        "texture": round(best_weights[2], 2),
        "venation": round(best_weights[3], 2),
        "_meta": {
            "method": "Grid Search Optimization",
            "metric": "mAP@5",
            "best_score": round(best_map, 4),
            "validation_queries": 50,
            "step": 0.05
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"\nDa luu trong so vao: {out_path}")
    print("Ban co the thay doi file nay thanh balanced_weights.json de cap nhat UI")

if __name__ == "__main__":
    run_grid_search()
