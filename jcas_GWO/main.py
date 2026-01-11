import numpy as np
import matplotlib.pyplot as plt
from jcas_model import JCAS_System
from gwo_optimizer import GWO_Optimizer

# ==========================================
# 1. CẤU HÌNH THAM SỐ (Theo báo cáo của bạn)
# ==========================================
N = 64                  # Số phần tử ăng-ten
USER_ANGLE = -15.0      # Góc người dùng (độ)
TARGET_ANGLE = 30.0     # Góc mục tiêu Radar (độ)
POP_SIZE = 30           # Số lượng sói
MAX_ITER = 100          # Số vòng lặp
ALPHA_WEIGHT = 0.5      # Trọng số cân bằng (alpha trong công thức)

# Khởi tạo hệ thống
jcas = JCAS_System(num_antennas=N)

# ==========================================
# 2. ĐỊNH NGHĨA HÀM MỤC TIÊU (FITNESS FUNCTION)
# ==========================================
def fitness_function(wolf_position):
    """
    Ánh xạ vị trí sói (vector thực 2N) thành trọng số phức và tính điểm thích nghi.
    Công thức: F = alpha * Gain_Comm + (1-alpha) * Gain_Sensing - lambda * Interference
    """
    # 1. Giải mã (Decode): Chuyển vector thực 2N thành vector phức N
    # Nửa đầu là phần thực, nửa sau là phần ảo
    w_real = wolf_position[0:N]
    w_imag = wolf_position[N:]
    w_complex = w_real + 1j * w_imag
    w_complex = w_complex.reshape(-1, 1) # Vector cột (N, 1)
    
    # Chuẩn hóa công suất (Norm = 1 hoặc theo P_max)
    w_complex = w_complex / np.linalg.norm(w_complex)
    
    # 2. Tính toán các thành phần Gain
    # Gain tại hướng User (Communication)
    gain_comm_arr = jcas.calculate_beampattern(w_complex, np.array([USER_ANGLE]))
    gain_comm = 10 * np.log10(gain_comm_arr[0] + 1e-12) # Đổi sang dB
    
    # Gain tại hướng Target (Sensing)
    gain_sense_arr = jcas.calculate_beampattern(w_complex, np.array([TARGET_ANGLE]))
    gain_sense = 10 * np.log10(gain_sense_arr[0] + 1e-12) # Đổi sang dB
    
    # 3. Tính toán nhiễu (Sidelobe Level - SLL)
    # Quét sơ bộ các góc để tìm búp sóng phụ (tránh vùng main lobes)
    # Vùng loại trừ: xung quanh User (+-5 độ) và Target (+-5 độ)
    scan_angles = np.linspace(-90, 90, 181)
    mask = np.ones(len(scan_angles), dtype=bool)
    mask[(scan_angles > USER_ANGLE-5) & (scan_angles < USER_ANGLE+5)] = False
    mask[(scan_angles > TARGET_ANGLE-5) & (scan_angles < TARGET_ANGLE+5)] = False
    
    sidelobe_angles = scan_angles[mask]
    sidelobe_gains = jcas.calculate_beampattern(w_complex, sidelobe_angles)
    max_sll = 10 * np.log10(np.max(sidelobe_gains) + 1e-12)
    
    # 4. Tổng hợp hàm mục tiêu
    # Chúng ta muốn Gain Comm và Sense cao, SLL thấp (max_sll càng nhỏ càng tốt)
    # Hàm Fitness cần Maximize.
    # Logic: Tăng Gain Comm/Sense, Phạt nếu SLL cao.
    
    # Trọng số lambda cho nhiễu
    lambda_int = 0.5
    
    # Lưu ý: Các giá trị dB thường âm hoặc dương tùy chuẩn hóa. 
    # Để GWO hoạt động tốt, nên giữ các giá trị này dương hoặc cùng tỷ lệ.
    # Ở đây dùng trực tiếp dB để tối ưu.
    
    score = (ALPHA_WEIGHT * gain_comm) + \
            ((1 - ALPHA_WEIGHT) * gain_sense) - \
            (lambda_int * max_sll)
            
    return score

# ==========================================
# 3. CHẠY TỐI ƯU HÓA GWO
# ==========================================
print("Bắt đầu tối ưu hóa JCAS với thuật toán GWO...")
print(f"Cấu hình: N={N}, User tại {USER_ANGLE} deg, Target tại {TARGET_ANGLE} deg")

# Số chiều tìm kiếm = 2 * N (thực + ảo)
optimizer = GWO_Optimizer(fitness_func=fitness_function, 
                          dim=2*N, 
                          pop_size=POP_SIZE, 
                          max_iter=MAX_ITER,
                          lower_bound=-1, 
                          upper_bound=1)

best_position, convergence_curve = optimizer.optimize()

# ==========================================
# 4. HIỂN THỊ KẾT QUẢ VÀ VẼ ĐỒ THỊ
# ==========================================
# Lấy trọng số tối ưu cuối cùng
w_real_opt = best_position[0:N]
w_imag_opt = best_position[N:]
w_opt = w_real_opt + 1j * w_imag_opt
w_opt = w_opt.reshape(-1, 1)
w_opt = w_opt / np.linalg.norm(w_opt) # Chuẩn hóa lại lần cuối

# Tính toán Beampattern đầy đủ (-90 đến 90 độ)
theta_plot = np.linspace(-90, 90, 720) # Độ phân giải cao để vẽ đẹp
beampattern = jcas.calculate_beampattern(w_opt, theta_plot)
beampattern_db = 10 * np.log10(beampattern + 1e-12)
# Chuẩn hóa về 0dB đỉnh cao nhất để dễ nhìn (Normalized Beampattern)
beampattern_db_norm = beampattern_db - np.max(beampattern_db)

# Vẽ hình
plt.figure(figsize=(10, 6))
plt.plot(theta_plot, beampattern_db_norm, linewidth=2, label='GWO Optimized Beam')

# Đánh dấu vị trí User và Target
plt.axvline(x=USER_ANGLE, color='g', linestyle='--', label='User Direction (Comm)')
plt.axvline(x=TARGET_ANGLE, color='r', linestyle='--', label='Target Direction (Sensing)')

# Trang trí đồ thị
plt.title(f'JCAS Beampattern Optimized by GWO (N={N})')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Normalized Magnitude (dB)')
plt.ylim([-50, 0]) # Giới hạn trục Y để nhìn rõ búp sóng phụ
plt.xlim([-90, 90])
plt.legend()
plt.grid(True, alpha=0.3)

# Lưu ảnh để bạn chèn vào báo cáo
plt.savefig('jcas_gwo_result.png', dpi=300)
print("Đã hoàn tất! Kết quả được lưu tại 'jcas_gwo_result.png'")
plt.show()

# Vẽ đồ thị hội tụ (Convergence)
plt.figure(figsize=(8, 4))
plt.plot(convergence_curve)
plt.title('GWO Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.grid(True)
plt.show()