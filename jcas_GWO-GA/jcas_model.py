import numpy as np

class JCAS_System:
    def __init__(self, num_antennas, frequency=28e9, spacing_ratio=0.5):
        self.N = num_antennas
        self.fc = frequency
        self.lam = 3e8 / frequency
        self.d = spacing_ratio * self.lam  # Khoảng cách d = lambda/2

    def steering_vector(self, theta_deg):
        """
        Tạo vector lái (Steering Vector) cho mảng ULA.
        Tương ứng với file generateSteeringVector.m
        """
        theta_rad = np.deg2rad(theta_deg)
        k = 2 * np.pi / self.lam
        # Vector chỉ số anten: [0, 1, ..., N-1]
        n = np.arange(self.N).reshape(-1, 1)
        # a(theta) = exp(j * k * d * n * sin(theta))
        # Lưu ý: Code gốc của bạn dùng positive phase trong generateSteeringVector.m
        sv = np.exp(1j * k * self.d * n * np.sin(theta_rad))
        return sv

    def calculate_beampattern(self, weights, theta_range):
        """
        Tính toán đồ thị bức xạ (Beampattern)
        weights: Vector trọng số w (complex)
        theta_range: Dải góc cần quét
        """
        # Đảm bảo dimensions khớp nhau để nhân ma trận
        # a_matrix shape: (N, số lượng góc)
        a_matrix = self.steering_vector(theta_range)
        
        # Array Factor: AF = w^H * a
        # weights.conj().T shape (1, N)
        af = np.matmul(weights.conj().T, a_matrix)
        
        # Power Gain (Magnitude squared)
        power_gain = np.abs(af)**2
        return power_gain.flatten()