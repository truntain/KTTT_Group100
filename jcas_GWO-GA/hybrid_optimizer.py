import numpy as np

class Hybrid_GWO_GA_Optimizer:
    def __init__(self, fitness_func, dim, pop_size, max_iter, lower_bound, upper_bound, mutation_rate=0.1):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lower_bound
        self.ub = upper_bound
        self.mutation_rate = mutation_rate
        
        # Khởi tạo quần thể
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        
        # Đánh giá fitness ban đầu
        for i in range(self.pop_size):
            self.fitness[i] = self.fitness_func(self.population[i])
            
    def optimize(self):
        # Sắp xếp để tìm Alpha, Beta, Delta
        sorted_indices = np.argsort(self.fitness)[::-1] # Fitness bài này là Score (càng cao càng tốt)
        # Lưu ý: Ở bài WSN là khoảng cách (càng nhỏ càng tốt), còn bài JCAS là Gain (càng lớn càng tốt).
        # Nên ta dùng [::-1] để sort giảm dần (Max problem).
        
        alpha_pos = self.population[sorted_indices[0]].copy()
        alpha_score = self.fitness[sorted_indices[0]]
        
        beta_pos = self.population[sorted_indices[1]].copy()
        delta_pos = self.population[sorted_indices[2]].copy()
        
        history = []
        
        print(">>> Bắt đầu chạy Hybrid GWO-GA cho JCAS...")
        
        for t in range(self.max_iter):
            history.append(alpha_score)
            
            a = 2.0 - t * (2.0 / self.max_iter) # Hệ số a giảm dần
            
            # Sắp xếp lại quần thể theo Fitness giảm dần (người giỏi nhất đứng đầu)
            sorted_idx = np.argsort(self.fitness)[::-1]
            self.population = self.population[sorted_idx]
            self.fitness = self.fitness[sorted_idx]
            
            # Chia đôi quần thể
            half_pop = self.pop_size // 2
            
            # === GIAI ĐOẠN 1: GWO (Top 50% Tốt nhất) ===
            for i in range(half_pop):
                for d in range(self.dim):
                    r1 = np.random.rand(); r2 = np.random.rand()
                    A1 = 2*a*r1 - a; C1 = 2*r2
                    D_alpha = abs(C1 * alpha_pos[d] - self.population[i, d])
                    X1 = alpha_pos[d] - A1 * D_alpha
                    
                    r1 = np.random.rand(); r2 = np.random.rand()
                    A2 = 2*a*r1 - a; C2 = 2*r2
                    D_beta = abs(C2 * beta_pos[d] - self.population[i, d])
                    X2 = beta_pos[d] - A2 * D_beta
                    
                    r1 = np.random.rand(); r2 = np.random.rand()
                    A3 = 2*a*r1 - a; C3 = 2*r2
                    D_delta = abs(C3 * delta_pos[d] - self.population[i, d])
                    X3 = delta_pos[d] - A3 * D_delta
                    
                    self.population[i, d] = (X1 + X2 + X3) / 3.0
            
            # === GIAI ĐOẠN 2: GA (Bottom 50% Yếu hơn) ===
            for i in range(half_pop, self.pop_size):
                # Lai ghép (Crossover) giữa Alpha và Beta
                w_cross = np.random.rand() # Trọng số lai ghép
                child = w_cross * alpha_pos + (1.0 - w_cross) * beta_pos
                
                # Đột biến (Mutation) trên từng gen
                for d in range(self.dim):
                    if np.random.rand() < self.mutation_rate:
                        # Đột biến ngẫu nhiên trong vùng tìm kiếm
                        child[d] = np.random.uniform(self.lb, self.ub)
                
                # Thay thế cá thể yếu bằng con mới sinh ra
                self.population[i] = child
            
            # Ràng buộc biên (Boundary Check)
            self.population = np.clip(self.population, self.lb, self.ub)
            
            # Cập nhật Fitness
            for i in range(self.pop_size):
                self.fitness[i] = self.fitness_func(self.population[i])
            
            # Cập nhật Alpha, Beta, Delta toàn cục
            # Tìm max fitness
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > alpha_score:
                alpha_score = self.fitness[current_best_idx]
                alpha_pos = self.population[current_best_idx].copy()
            
            # Cập nhật lại Beta, Delta (xét trên toàn quần thể mới)
            sorted_indices_new = np.argsort(self.fitness)[::-1]
            beta_pos = self.population[sorted_indices_new[1]].copy()
            delta_pos = self.population[sorted_indices_new[2]].copy()
            
            if (t+1) % 10 == 0:
                print(f"Iter {t+1}: Best Fitness = {alpha_score:.4f}")
                
        return alpha_pos, history