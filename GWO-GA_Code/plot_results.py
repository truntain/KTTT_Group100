import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
NUM_NODES = 100         # Số nút cảm biến
NUM_CLUSTERS = 5        # Số cụm (Cluster Heads)
MAX_ITER = 50           # Số vòng lặp tối đa
NUM_WOLVES = 20         # Số lượng sói
AREA_SIZE = 100.0       # Kích thước vùng mạng (100x100)
MUTATION_RATE = 0.1     # Tỉ lệ đột biến
DIM = NUM_CLUSTERS * 2  # Số chiều (5 cụm * 2 tọa độ x,y)

# ==========================================
# 2. HÀM HỖ TRỢ VÀ FITNESS FUNCTION
# ==========================================

def init_nodes():
    """Khởi tạo tọa độ ngẫu nhiên cho các nút cảm biến"""
    # Tạo ma trận (NUM_NODES, 2) với giá trị từ 0 đến AREA_SIZE
    return np.random.uniform(0, AREA_SIZE, (NUM_NODES, 2))

def calculate_fitness(position, nodes):
    """
    Tính tổng khoảng cách từ các nút về CH gần nhất.
    position: Vector (DIM,) chứa tọa độ của các CH
    nodes: Ma trận (NUM_NODES, 2) chứa tọa độ các nút
    """
    total_dist = 0
    
    # Reshape vector vị trí thành ma trận (NUM_CLUSTERS, 2) để dễ tính toán
    chs = position.reshape(NUM_CLUSTERS, 2)
    
    # Tính khoảng cách từ mỗi nút đến tất cả các CH
    # Dùng broadcasting của numpy để tính nhanh
    # nodes[:, np.newaxis, :] shape là (100, 1, 2)
    # chs[np.newaxis, :, :] shape là (1, 5, 2)
    # diff shape là (100, 5, 2)
    diff = nodes[:, np.newaxis, :] - chs[np.newaxis, :, :]
    
    # Khoảng cách Euclide: sqrt(dx^2 + dy^2)
    dists = np.sqrt(np.sum(diff**2, axis=2)) # Shape (100, 5)
    
    # Tìm khoảng cách nhỏ nhất cho mỗi nút (min theo trục CH)
    min_dists = np.min(dists, axis=1)
    
    # Tổng các khoảng cách nhỏ nhất
    total_dist = np.sum(min_dists)
    
    return total_dist

# ==========================================
# 3. THUẬT TOÁN CHÍNH: HYBRID GWO-GA
# ==========================================

def run_hybrid_GWO_GA(nodes):
    print(">>> Bắt đầu chạy Hybrid GWO-GA (Python version)...")
    
    # 1. Khởi tạo quần thể sói
    # Ma trận (NUM_WOLVES, DIM)
    population = np.random.uniform(0, AREA_SIZE, (NUM_WOLVES, DIM))
    
    # Tính fitness ban đầu
    fitness = np.array([calculate_fitness(ind, nodes) for ind in population])
    
    # Sắp xếp tìm Alpha, Beta, Delta
    sorted_indices = np.argsort(fitness)
    alpha_pos = population[sorted_indices[0]].copy()
    alpha_score = fitness[sorted_indices[0]]
    
    beta_pos = population[sorted_indices[1]].copy()
    delta_pos = population[sorted_indices[2]].copy()
    
    history = [] # Lưu lịch sử hội tụ
    
    # 2. Vòng lặp chính
    for t in range(MAX_ITER):
        history.append(alpha_score)
        
        if (t + 1) % 10 == 0:
            print(f"Vòng lặp {t+1}: Best Fitness = {alpha_score:.4f}")
            
        a = 2.0 - t * (2.0 / MAX_ITER) # Tham số a giảm dần từ 2 xuống 0
        half_pop = NUM_WOLVES // 2
        
        # --- GIAI ĐOẠN 1: GWO (Top 50% sói tốt nhất) ---
        # Cập nhật vị trí dựa trên Alpha, Beta, Delta
        for i in range(half_pop):
            # Lấy vị trí sói hiện tại (theo index đã sort hoặc không, ở đây dùng index gốc của pop)
            # Lưu ý: Code C không sort lại mảng population mà chỉ tìm alpha/beta/delta.
            # Nhưng để khớp logic "Top 50%", ta nên sort population trước.
            # Tuy nhiên, code C gốc duyệt i từ 0->half_pop trên mảng population đã qsort.
            # Nên ta sẽ sort population theo fitness trước khi vào vòng lặp update.
            
            # Sort population và fitness
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Thực hiện update cho nửa đầu (Top 50%)
            for d in range(DIM):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # Alpha
                A1 = 2*a*r1 - a
                C1 = 2*r2
                D_alpha = abs(C1 * alpha_pos[d] - population[i, d])
                X1 = alpha_pos[d] - A1 * D_alpha
                
                # Beta
                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2*a*r1 - a
                C2 = 2*r2
                D_beta = abs(C2 * beta_pos[d] - population[i, d])
                X2 = beta_pos[d] - A2 * D_beta
                
                # Delta
                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2*a*r1 - a
                C3 = 2*r2
                D_delta = abs(C3 * delta_pos[d] - population[i, d])
                X3 = delta_pos[d] - A3 * D_delta
                
                # Cập nhật vị trí
                population[i, d] = (X1 + X2 + X3) / 3.0
        
        # --- GIAI ĐOẠN 2: GA (Bottom 50% sói yếu hơn) ---
        for i in range(half_pop, NUM_WOLVES):
            for d in range(DIM):
                # Lai ghép (Crossover) giữa Alpha và Beta
                w_cross = np.random.rand()
                child_gene = w_cross * alpha_pos[d] + (1.0 - w_cross) * beta_pos[d]
                
                # Đột biến (Mutation)
                if np.random.rand() < MUTATION_RATE:
                    child_gene = np.random.uniform(0, AREA_SIZE)
                
                population[i, d] = child_gene

        # Ràng buộc biên (Boundary Check) cho toàn bộ quần thể
        population = np.clip(population, 0, AREA_SIZE)
        
        # Cập nhật lại Fitness và Lãnh đạo
        fitness = np.array([calculate_fitness(ind, nodes) for ind in population])
        
        # Tìm Alpha, Beta, Delta mới
        sorted_indices = np.argsort(fitness)
        
        # Cập nhật global best nếu tìm thấy tốt hơn
        if fitness[sorted_indices[0]] < alpha_score:
            alpha_score = fitness[sorted_indices[0]]
            alpha_pos = population[sorted_indices[0]].copy()
            
        # Cập nhật Beta, Delta cục bộ của vòng lặp này
        beta_pos = population[sorted_indices[1]].copy()
        delta_pos = population[sorted_indices[2]].copy()
        
    return alpha_pos, history

# ... (Các phần trên giữ nguyên) ...

# ==========================================
# 4. VẼ BIỂU ĐỒ & LƯU ẢNH
# ==========================================

def plot_results(nodes, best_pos, history):
    print("Đang xử lý và vẽ biểu đồ...")
    
    # 1. Tách tọa độ CH từ best_pos
    chs = best_pos.reshape(NUM_CLUSTERS, 2)
    
    # 2. Phân cụm (Gán nhãn cho nút về CH gần nhất)
    diff = nodes[:, np.newaxis, :] - chs[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    labels = np.argmin(dists, axis=1)
    
    # 3. Khởi tạo hình ảnh (1 hình chứa 2 biểu đồ con)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- BIỂU ĐỒ 1: KẾT QUẢ PHÂN CỤM WSN ---
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    
    for k in range(NUM_CLUSTERS):
        cluster_nodes = nodes[labels == k]
        c = colors[k % len(colors)]
        
        # Vẽ các nút cảm biến (Sensor Nodes)
        ax1.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], c=c, s=50, alpha=0.6, label=f'Cluster {k+1}')
        
        # Vẽ đường nối từ Node về Cluster Head
        for node in cluster_nodes:
            ax1.plot([node[0], chs[k, 0]], [node[1], chs[k, 1]], c=c, alpha=0.1, linewidth=1)
            
    # Vẽ các Trưởng cụm (Cluster Heads)
    ax1.scatter(chs[:, 0], chs[:, 1], c='black', marker='*', s=300, edgecolors='yellow', linewidth=1.5, label='Cluster Heads')
    
    ax1.set_title(f'WSN Clustering Result (Pop={NUM_WOLVES}, Iter={MAX_ITER})')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.set_xlim(0, AREA_SIZE)
    ax1.set_ylim(0, AREA_SIZE)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # --- BIỂU ĐỒ 2: TỐC ĐỘ HỘI TỤ ---
    ax2.plot(range(1, MAX_ITER + 1), history, 'r-o', linewidth=2, markersize=4)
    ax2.set_title('Convergence Curve (Hybrid GWO-GA)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Distance (Fitness)')
    ax2.grid(True)
    
    # 4. Lưu ảnh
    plt.tight_layout()
    output_filename = 'Hybrid_GWO_GA_WSN_Result.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f">>> Đã lưu ảnh kết quả tại: {output_filename}")
    plt.show()

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    np.random.seed(42) # Giữ cố định seed để bài báo cáo nhất quán
    
    # 1. Khởi tạo
    nodes = init_nodes()
    
    # 2. Chạy tối ưu
    best_solution, convergence_history = run_hybrid_GWO_GA(nodes)
    
    # 3. Vẽ và Lưu
    plot_results(nodes, best_solution, convergence_history)