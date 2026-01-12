import numpy as np

class JCAS_System:
    def __init__(self, num_antennas, frequency=28e9, spacing_ratio=0.5):
        self.N = num_antennas
        self.fc = frequency
        self.lam = 3e8 / frequency
        self.d = spacing_ratio * self.lam

    def steering_vector(self, theta_deg):
        """Tạo vector lái (Steering Vector)"""
        if isinstance(theta_deg, (float, int)):
            theta_deg = np.array([theta_deg])
            
        theta_rad = np.deg2rad(theta_deg)
        k = 2 * np.pi / self.lam
        n = np.arange(self.N).reshape(-1, 1)
        # a(theta) shape: (N, số lượng góc)
        sv = np.exp(1j * k * self.d * n * np.sin(theta_rad))
        return sv

    def calculate_beampattern(self, weights, theta_range):
        """Tính đồ thị bức xạ"""
        a_matrix = self.steering_vector(theta_range)
        # AF = w^H * a
        af = np.matmul(weights.conj().T, a_matrix)
        power_gain = np.abs(af)**2
        return power_gain.flatten()