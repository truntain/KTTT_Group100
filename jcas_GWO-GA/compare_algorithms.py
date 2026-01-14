import numpy as np
import matplotlib.pyplot as plt
from jcas_model import JCAS_System
from hybrid_optimizer import Hybrid_GWO_GA_Optimizer

# --- CẤU HÌNH ---
N = 64
USER_ANGLE = -15.0
TARGET_ANGLE = 30.0
POP_SIZE = 30
MAX_ITER = 100
ALPHA_WEIGHT = 0.5
LAMBDA_INT = 0.5

# Khởi tạo hệ thống
jcas = JCAS_System(num_antennas=N)

# --- HÀM FITNESS (Dùng chung) ---
def fitness_function(wolf_position):
    w_real = wolf_position[0:N]
    w_imag = wolf_position[N:]
    w_complex = w_real + 1j * w_imag
    w_complex = w_complex.reshape(-1, 1)
    w_complex = w_complex / np.linalg.norm(w_complex)
    
    gain_comm = 10 * np.log10(jcas.calculate_beampattern(w_complex, np.array([USER_ANGLE]))[0] + 1e-12)
    gain_sense = 10 * np.log10(jcas.calculate_beampattern(w_complex, np.array([TARGET_ANGLE]))[0] + 1e-12)
    
    scan_angles = np.linspace(-90, 90, 181)
    mask = np.ones(len(scan_angles), dtype=bool)
    mask[(scan_angles > USER_ANGLE-5) & (scan_angles < USER_ANGLE+5)] = False
    mask[(scan_angles > TARGET_ANGLE-5) & (scan_angles < TARGET_ANGLE+5)] = False
    max_sll = 10 * np.log10(np.max(jcas.calculate_beampattern(w_complex, scan_angles[mask])) + 1e-12)
    
    score = (ALPHA_WEIGHT * gain_comm) + ((1 - ALPHA_WEIGHT) * gain_sense) - (LAMBDA_INT * max_sll)
    return score

# --- ĐỊNH NGHĨA LẠI GWO THƯỜNG (Để chạy so sánh tại đây) ---
class Standard_GWO_Optimizer:
    def __init__(self, fitness_func, dim, pop_size, max_iter, lb, ub):
        self.fitness_func = fitness_func
        self.dim = dim; self.pop_size = pop_size; self.max_iter = max_iter
        self.lb = lb; self.ub = ub
        self.population = np.random.uniform(lb, ub, (pop_size, dim))
        self.fitness = np.zeros(pop_size)
        
    def optimize(self):
        for i in range(self.pop_size): self.fitness[i] = self.fitness_func(self.population[i])
        sorted_idx = np.argsort(self.fitness)[::-1]
        alpha = self.population[sorted_idx[0]].copy(); alpha_score = self.fitness[sorted_idx[0]]
        beta = self.population[sorted_idx[1]].copy()
        delta = self.population[sorted_idx[2]].copy()
        history = []
        
        for t in range(self.max_iter):
            history.append(alpha_score)
            a = 2.0 - t * (2.0 / self.max_iter)
            for i in range(self.pop_size):
                for d in range(self.dim):
                    r1=np.random.rand(); r2=np.random.rand(); A1=2*a*r1-a; C1=2*r2
                    D_alpha=abs(C1*alpha[d]-self.population[i,d]); X1=alpha[d]-A1*D_alpha
                    r1=np.random.rand(); r2=np.random.rand(); A2=2*a*r1-a; C2=2*r2
                    D_beta=abs(C2*beta[d]-self.population[i,d]); X2=beta[d]-A2*D_beta
                    r1=np.random.rand(); r2=np.random.rand(); A3=2*a*r1-a; C3=2*r2
                    D_delta=abs(C3*delta[d]-self.population[i,d]); X3=delta[d]-A3*D_delta
                    self.population[i,d]=(X1+X2+X3)/3
            
            self.population = np.clip(self.population, self.lb, self.ub)
            for i in range(self.pop_size): self.fitness[i] = self.fitness_func(self.population[i])
            sorted_idx = np.argsort(self.fitness)[::-1]
            if self.fitness[sorted_idx[0]] > alpha_score:
                alpha_score = self.fitness[sorted_idx[0]]
                alpha = self.population[sorted_idx[0]].copy()
                beta = self.population[sorted_idx[1]].copy()
                delta = self.population[sorted_idx[2]].copy()
        return history

# --- CHẠY SO SÁNH ---
print("Đang chạy Standard GWO...")
gwo_opt = Standard_GWO_Optimizer(fitness_function, 2*N, POP_SIZE, MAX_ITER, -1, 1)
hist_gwo = gwo_opt.optimize()

print("Đang chạy Hybrid GWO-GA...")
hybrid_opt = Hybrid_GWO_GA_Optimizer(fitness_function, 2*N, POP_SIZE, MAX_ITER, -1, 1, mutation_rate=0.1)
_, hist_hybrid = hybrid_opt.optimize()

# --- VẼ VÀ LƯU ẢNH ---
plt.figure(figsize=(10, 6))
plt.plot(hist_gwo, 'k--', linewidth=1.5, label='Standard GWO')
plt.plot(hist_hybrid, 'b-o', linewidth=2, markersize=4, label='Hybrid GWO-GA')
plt.title('Comparison of Convergence Speed')
plt.xlabel('Iteration')
plt.ylabel('Fitness Score')
plt.legend()
plt.grid(True)
plt.savefig('comparison_convergence.png', dpi=300)
print("Đã lưu ảnh: comparison_convergence.png")
plt.show()