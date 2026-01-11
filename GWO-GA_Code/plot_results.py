import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    print("Đang đọc dữ liệu từ file .txt do C tạo ra...")
    
    # 1. Đọc dữ liệu
    try:
        nodes = np.loadtxt('data_nodes.txt')     # Nút mạng
        chs = np.loadtxt('data_chs.txt')         # Trưởng cụm
        history = np.loadtxt('data_history.txt') # Lịch sử hội tụ
    except OSError:
        print("LỖI: Không tìm thấy file .txt! Hãy chạy code C (main.c) trước.")
        return

    num_nodes = len(nodes)
    num_chs = len(chs)

    # 2. Xử lý phân cụm (Gán màu)
    # Tính lại xem nút nào thuộc cụm nào để tô màu cho khớp
    labels = []
    for i in range(num_nodes):
        # Tính khoảng cách từ nút i đến tất cả CH
        dists = np.linalg.norm(nodes[i] - chs, axis=1)
        # Gán nhãn cho CH gần nhất
        labels.append(np.argmin(dists))
    labels = np.array(labels)

    # 3. Cấu hình biểu đồ
    plt.figure(figsize=(14, 6))

    # --- BIỂU ĐỒ 1: KẾT QUẢ PHÂN CỤM ---
    plt.subplot(1, 2, 1)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    
    # Vẽ từng cụm
    for k in range(num_chs):
        cluster_mask = (labels == k)
        cluster_nodes = nodes[cluster_mask]
        
        # Lấy màu (nếu số cụm > số màu thì xoay vòng)
        c = colors[k % len(colors)]
        
        # Vẽ nút thường
        plt.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], c=c, s=50, alpha=0.6, edgecolors='none')
        
        # Vẽ đường nối từ nút về CH (để nhìn cho chuyên nghiệp)
        for node in cluster_nodes:
            plt.plot([node[0], chs[k, 0]], [node[1], chs[k, 1]], c=c, alpha=0.15, linewidth=1)

    # Vẽ Trưởng cụm (Cluster Heads)
    plt.scatter(chs[:, 0], chs[:, 1], c='black', marker='*', s=250, edgecolors='white', linewidth=1.5, label='Cluster Heads')
    
    plt.title(f'Kết quả Phân cụm WSN (Code C + Python Plot)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- BIỂU ĐỒ 2: TỐC ĐỘ HỘI TỤ ---
    plt.subplot(1, 2, 2)
    # Cột 0 là vòng lặp, Cột 1 là Fitness
    # Nếu file history chỉ có 1 cột fitness thì dùng history trực tiếp
    if history.ndim > 1: 
        x_axis = history[:, 0]
        y_axis = history[:, 1]
    else:
        x_axis = np.arange(len(history))
        y_axis = history

    plt.plot(x_axis, y_axis, 'r-', linewidth=2.5)
    plt.title('Tốc độ hội tụ (Convergence Curve)')
    plt.xlabel('Vòng lặp (Iteration)')
    plt.ylabel('Tổng khoảng cách (Fitness)')
    plt.grid(True)

    # 4. Lưu và Hiển thị
    plt.tight_layout()
    output_filename = 'Hybrid_GWO_GA_Result_C_Viz.png'
    plt.savefig(output_filename, dpi=300) # Xuất ảnh chất lượng cao
    print(f"Xong! Đã lưu ảnh kết quả vào file: {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_results()