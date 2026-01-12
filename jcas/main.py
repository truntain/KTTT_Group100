import numpy as np
import matplotlib.pyplot as plt
from jcas_model import JCAS_System
from ils_optimizer import ILS_Optimizer

# ==========================================
# 1. CẤU HÌNH (Giống hệt GWO để so sánh)
# ==========================================
N = 64
USER_ANGLE = -15.0
TARGET_ANGLE = 30.0

# Khởi tạo hệ thống
jcas = JCAS_System(num_antennas=N)

# ==========================================
# 2. CHẠY THUẬT TOÁN GỐC (ILS)
# ==========================================
print(f"Đang chạy thuật toán gốc ILS cho N={N}...")
optimizer = ILS_Optimizer(jcas, USER_ANGLE, TARGET_ANGLE, N)

# ILS hội tụ rất nhanh, chỉ cần khoảng 20 vòng lặp
w_opt_ils, error_history = optimizer.optimize(max_iter=50)

# ==========================================
# 3. VẼ VÀ LƯU KẾT QUẢ
# ==========================================
theta_plot = np.linspace(-90, 90, 720)
beampattern = jcas.calculate_beampattern(w_opt_ils, theta_plot)
beampattern_db = 10 * np.log10(beampattern + 1e-12)
beampattern_db_norm = beampattern_db - np.max(beampattern_db)

# --- HÌNH 1: ĐỒ THỊ BÚP SÓNG (BEAMPATTERN) ---
plt.figure(figsize=(10, 6))
plt.plot(theta_plot, beampattern_db_norm, 'k-', linewidth=2, label='ILS (Original) Benchmark')
plt.axvline(x=USER_ANGLE, color='g', linestyle='--', label='User Direction')
plt.axvline(x=TARGET_ANGLE, color='r', linestyle='--', label='Target Direction')

plt.title(f'JCAS Benchmark: Iterative Least Squares (N={N})')
plt.xlabel('Angle (deg)')
plt.ylabel('Normalized Gain (dB)')
plt.ylim([-60, 0])
plt.xlim([-90, 90])
plt.legend()
plt.grid(True, alpha=0.3)

# Lưu ảnh Beampattern
plt.savefig('jcas_benchmark_beampattern.png', dpi=300)
print("Đã lưu ảnh 1: jcas_benchmark_beampattern.png")
plt.show()

# --- HÌNH 2: TỐC ĐỘ HỘI TỤ (CONVERGENCE) ---
plt.figure(figsize=(8, 5))
plt.plot(error_history, 'k-o', linewidth=1.5, markersize=4)
plt.title('Convergence Speed: ILS Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Least Squares Error (Cost Function)')
plt.grid(True, linestyle='--', alpha=0.7)

# Lưu ảnh Hội tụ
plt.savefig('jcas_benchmark_convergence.png', dpi=300)
print("Đã lưu ảnh 2: jcas_benchmark_convergence.png")
plt.show()