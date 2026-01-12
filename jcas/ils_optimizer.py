import numpy as np

class ILS_Optimizer:
    """
    Iterative Least Squares (ILS) Optimizer
    Thuật toán gốc dựa trên Toán học (Đại số tuyến tính) để so sánh với GWO.
    """
    def __init__(self, jcas_system, user_angle, target_angle, num_antennas):
        self.jcas = jcas_system
        self.user_angle = user_angle
        self.target_angle = target_angle
        self.N = num_antennas

    def optimize(self, max_iter=20):
        """
        Thực hiện tối ưu hóa bằng phương pháp Lặp Bình Phương Tối Thiểu.
        Mục tiêu: Tìm trọng số w sao cho Búp sóng thực tế (A*w) khớp với Búp sóng mong muốn (d).
        """
        # 1. Định nghĩa các điểm mẫu (góc) để tối ưu
        # Ta lấy mẫu dày đặc để ép búp sóng phụ xuống
        sample_angles = np.linspace(-90, 90, 181) 
        
        # 2. Tạo vector mong muốn (Desired Pattern - d)
        # Tại hướng User và Target: Mong muốn biên độ = 1
        # Tại các hướng khác (Sidelobe): Mong muốn biên độ = 0
        desired_magnitude = np.zeros(len(sample_angles))
        
        # Tìm chỉ số (index) gần đúng nhất của góc User và Target
        idx_user = np.abs(sample_angles - self.user_angle).argmin()
        idx_target = np.abs(sample_angles - self.target_angle).argmin()
        
        # Đặt biên độ mong muốn (Main lobes)
        # Mở rộng nhẹ vùng đỉnh (+- 1 độ) để búp sóng không quá nhọn
        desired_magnitude[max(0, idx_user-1):min(len(sample_angles), idx_user+2)] = 1.0
        desired_magnitude[max(0, idx_target-1):min(len(sample_angles), idx_target+2)] = 1.0
        
        # 3. Tính Ma trận lái (Steering Matrix) cho tất cả các góc mẫu
        # Shape: (N, 181)
        A = self.jcas.steering_vector(sample_angles)
        
        # Tính ma trận giả nghịch đảo (Moore-Penrose Pseudo-inverse)
        # Công thức LS: w = pinv(A) * d
        # A_pinv shape: (181, N)
        A_pinv = np.linalg.pinv(A)
        
        # 4. Vòng lặp ILS (Iterative Least Squares)
        # Vì ta chỉ quan tâm biên độ |A*w| khớp với |d|, còn pha có thể tự do.
        # Thuật toán sẽ lặp để chỉnh pha sao cho tối ưu nhất.
        
        # Khởi tạo pha ban đầu ngẫu nhiên cho vector mong muốn
        current_phase = np.exp(1j * np.random.rand(len(sample_angles)) * 2 * np.pi)
        
        history = []
        
        for i in range(max_iter):
            # Vector mục tiêu phức (Biên độ mong muốn + Pha hiện tại)
            y = desired_magnitude * current_phase
            
            # Bước 1: Tìm w tối ưu cho y hiện tại (Least Squares Step)
            # w = A_dagger * y
            # y shape (181,), A_pinv shape (N, 181) -> w shape (N,)
            # Cần transpose y thành cột hoặc xử lý chiều đúng
            w_opt = np.matmul(A_pinv.T, y) # Lưu ý: A_pinv của numpy là (N, M) nếu A là (M, N). 
            # Check lại: A (N, 181). pinv(A) là (181, N).
            # Công thức đúng: A.T * w = y (trong miền mẫu). 
            # Thực tế: Pattern = w.H * A. => Pattern.T = A.H * w.
            # Hệ phương trình: A^H * w = y
            # => w = pinv(A^H) * y
            
            # Sửa lại tính toán cho khớp chiều vật lý:
            # Pattern P = w^H * A. Ta muốn P ~ y.
            # Lấy liên hợp: P^H = A^H * w.
            # Đặt B = A^H (Shape: 181, N). Ta giải B * w = y^H (hoặc y vì y là 1 chiều).
            B = A.conj().T
            w_opt, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
            
            # Chuẩn hóa công suất w
            w_opt = w_opt / np.linalg.norm(w_opt)
            
            # Bước 2: Cập nhật pha (Update Phase)
            # Tính búp sóng thực tế thu được
            pattern_actual = np.matmul(B, w_opt)
            
            # Giữ nguyên biên độ mong muốn, chỉ lấy pha của búp sóng thực tế cập nhật lại
            current_phase = np.exp(1j * np.angle(pattern_actual))
            
            # Tính lỗi (Error/Fitness) để so sánh
            error = np.linalg.norm(np.abs(pattern_actual) - desired_magnitude)
            history.append(error)
            
        return w_opt, history