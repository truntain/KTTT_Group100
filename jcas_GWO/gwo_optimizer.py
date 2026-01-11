import numpy as np

class GWO_Optimizer:
    def __init__(self, fitness_func, dim, pop_size=30, max_iter=100, lower_bound=-1, upper_bound=1):
        self.fitness_func = fitness_func
        self.dim = dim # Số chiều bài toán (2*N)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lower_bound
        self.ub = upper_bound

    def optimize(self):
        # 1. Khởi tạo quần thể sói (Positions)
        # Mỗi hàng là một con sói
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Khởi tạo Alpha, Beta, Delta
        alpha_pos = np.zeros(self.dim)
        alpha_score = -float('inf') # Chúng ta đang tìm Maximize Fitness
        
        beta_pos = np.zeros(self.dim)
        beta_score = -float('inf')
        
        delta_pos = np.zeros(self.dim)
        delta_score = -float('inf')
        
        history = [] # Lưu lịch sử hội tụ

        # Vòng lặp chính
        for l in range(0, self.max_iter):
            # Tính a giảm dần từ 2 xuống 0
            a = 2 - l * ((2) / self.max_iter) 
            
            # Đánh giá fitness cho từng con sói
            for i in range(self.pop_size):
                # Tính hàm mục tiêu
                fitness = self.fitness_func(positions[i, :])
                
                # Cập nhật Alpha, Beta, Delta (Tìm MAX)
                if fitness > alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = positions[i, :].copy()
                elif fitness > beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = positions[i, :].copy()
                elif fitness > delta_score:
                    delta_score = fitness
                    delta_pos = positions[i, :].copy()
            
            history.append(alpha_score)
            
            # Cập nhật vị trí các con sói (Bao vây con mồi)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    # Tính toán dựa trên Alpha
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    # Tính toán dựa trên Beta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    # Tính toán dựa trên Delta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    # Cập nhật vị trí mới: trung bình cộng
                    positions[i, j] = (X1 + X2 + X3) / 3
            
            print(f"Iteration {l+1}/{self.max_iter}, Best Fitness: {alpha_score:.4f}")

        return alpha_pos, history