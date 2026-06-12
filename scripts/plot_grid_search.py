import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage
from scripts.grid_search_weights import FLAVIA_RANGES, get_label, compute_ap

def plot_peak():
    print("Đang đọc dữ liệu từ Database...")
    db = SessionLocal()
    records = db.query(
        LeafImage.image_id,
        LeafImage.shape_vector,
        LeafImage.color_vector,
        LeafImage.texture_vector,
        LeafImage.venation_vector
    ).all()
    db.close()
    
    image_ids = np.array([r.image_id for r in records])
    labels = np.array([get_label(idx) for idx in image_ids])
    
    M_shape = np.array([list(r.shape_vector) for r in records], dtype=np.float32)
    M_color = np.array([list(r.color_vector) for r in records], dtype=np.float32)
    M_texture = np.array([list(r.texture_vector) for r in records], dtype=np.float32)
    M_venation = np.array([list(r.venation_vector) for r in records], dtype=np.float32)

    random.seed(42)
    valid_indices = random.sample(range(len(image_ids)), 50)
    query_labels = labels[valid_indices]
    
    def normalize_l2(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return M / norms
        
    N_shape = normalize_l2(M_shape)
    N_color = normalize_l2(M_color)
    N_texture = normalize_l2(M_texture)
    N_venation = normalize_l2(M_venation)
    
    Q_shape = N_shape[valid_indices]
    Q_color = N_color[valid_indices]
    Q_texture = N_texture[valid_indices]
    Q_venation = N_venation[valid_indices]
    
    D_shape = 1.0 - np.dot(Q_shape, N_shape.T)
    D_color = 1.0 - np.dot(Q_color, N_color.T)
    D_texture = 1.0 - np.dot(Q_texture, N_texture.T)
    D_venation = 1.0 - np.dot(Q_venation, N_venation.T)
    
    # Cố định w_color = 0.15 và w_venation = 0.25 (theo tổ hợp tốt nhất)
    # Thay đổi w_shape từ 0 đến 0.60, w_texture = 0.60 - w_shape
    w_color_fixed = 0.15
    w_venation_fixed = 0.25
    
    w_shapes = np.linspace(0.0, 0.60, 50)
    mAPs = []
    
    for ws in tqdm(w_shapes, desc="Calculating curve"):
        wt = 0.60 - ws
        D_total = ws * D_shape + w_color_fixed * D_color + wt * D_texture + w_venation_fixed * D_venation
        
        top_k = 5
        top_k_indices = np.argsort(D_total, axis=1)[:, :top_k]
        
        ap_sum = 0.0
        for i in range(len(valid_indices)):
            retrieved_labels = labels[top_k_indices[i]]
            ap_sum += compute_ap(retrieved_labels, query_labels[i], top_k)
            
        mAP = ap_sum / len(valid_indices)
        mAPs.append(mAP)
        
    # Tạo biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(w_shapes, mAPs, marker='o', linestyle='-', color='b', markersize=6)
    
    # Tìm điểm cực đại
    max_idx = np.argmax(mAPs)
    max_ws = w_shapes[max_idx]
    max_map = mAPs[max_idx]
    
    plt.scatter(max_ws, max_map, color='red', s=150, zorder=5, label=f'Đỉnh cực đại (w_shape={max_ws:.2f}, mAP@5={max_map:.4f})')
    plt.annotate(f' Tối ưu\n (0.35, {max_map:.4f})', xy=(max_ws, max_map), xytext=(max_ws+0.02, max_map-0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=12, fontweight='bold')

    plt.title('Biểu diễn cực đại mAP@5 theo trọng số hình dạng (w_shape)\n(Cố định w_color=0.15, w_venation=0.25, w_texture=0.60-w_shape)', fontsize=14)
    plt.xlabel('Trọng số Hình dạng (w_shape)', fontsize=12)
    plt.ylabel('Điểm mAP@5', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    out_path = os.path.join(PROJECT_ROOT, "reports", "map_peak_visualization.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nĐã lưu biểu đồ thành công tại: {out_path}")

if __name__ == "__main__":
    plot_peak()
