import numpy as np
import matplotlib.pyplot as plt
from jcas_model import JCAS_System
from hybrid_optimizer import Hybrid_GWO_GA_Optimizer

# --- CẤU HÌNH (GIỮ NGUYÊN ĐỂ SO SÁNH) ---
N = 64
USER_ANGLE = -15.0
TARGET_ANGLE = 30.0
POP_SIZE = 30
MAX_ITER = 100
ALPHA_WEIGHT = 0.5
LAMBDA_INT = 0.5

jcas = JCAS_System(num_antennas=N)

# --- HAM FITNESS (GIỮ NGUYÊN) ---
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

# --- CHẠY TỐI ƯU HYBRID ---
optimizer = Hybrid_GWO_GA_Optimizer(
    fitness_func=fitness_function,
    dim=2*N,
    pop_size=POP_SIZE,
    max_iter=MAX_ITER,
    lower_bound=-1,
    upper_bound=1,
    mutation_rate=0.1 # Tỷ lệ đột biến
)

best_pos, history = optimizer.optimize()

# --- VẼ KẾT QUẢ SO SÁNH ---
w_real = best_pos[0:N]
w_imag = best_pos[N:]
w_opt = w_real + 1j * w_imag
w_opt = w_opt.reshape(-1, 1)
w_opt = w_opt / np.linalg.norm(w_opt)

theta_plot = np.linspace(-90, 90, 720)
beampattern_db = 10 * np.log10(jcas.calculate_beampattern(w_opt, theta_plot) + 1e-12)
beampattern_norm = beampattern_db - np.max(beampattern_db)

plt.figure(figsize=(10, 6))
# Vẽ kết quả Hybrid
plt.plot(theta_plot, beampattern_norm, 'b-', linewidth=2, label='Hybrid GWO-GA')
# Vẽ đường tham chiếu User/Target
plt.axvline(x=USER_ANGLE, color='g', linestyle='--', label='User')
plt.axvline(x=TARGET_ANGLE, color='r', linestyle='--', label='Target')
# Vẽ đường giới hạn nhiễu mong muốn (Ví dụ -15dB)
plt.axhline(y=-15, color='orange', linestyle=':', label='Desired SLL limit')

plt.title(f'JCAS Beamforming with Hybrid GWO-GA (N={N})')
plt.xlabel('Angle (deg)')
plt.ylabel('Normalized Gain (dB)')
plt.ylim([-50, 0])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('jcas_hybrid_result.png', dpi=300)
plt.show()

# Vẽ hội tụ
plt.figure(figsize=(8, 4))
plt.plot(history, 'b-o', markersize=3)
plt.title('Hybrid GWO-GA Convergence')
plt.xlabel('Iteration')
plt.ylabel('Fitness Score')
plt.grid(True)
plt.savefig('jcas_hybrid_convergence.png', dpi=300)
plt.show()