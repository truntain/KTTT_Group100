# Phần III: Ứng dụng GWO trong Tối ưu hóa JCAS

Module này thực hiện mô phỏng hệ thống **Joint Communication and Sensing (JCAS)** sử dụng thuật toán Bầy sói xám (GWO) để tối ưu hóa trọng số ăng-ten.

## 1. Bài toán
Tìm kiếm cấu hình Beamforming vector $\mathbf{W}$ để thỏa mãn hàm mục tiêu:
$$F(\mathbf{W}) = \alpha \cdot \Gamma_{comm} + (1 - \alpha) \cdot \Gamma_{sensing} - \lambda \cdot P_{interference}$$

* **Input:** Số lượng ăng-ten ($N=64$), Góc người dùng ($-15^\circ$), Góc mục tiêu ($30^\circ$).
* **Output:** Đồ thị búp sóng (Beampattern) và đường cong hội tụ.

## 2. Hướng dẫn chạy chương trình

### Yêu cầu cài đặt
```bash
pip install numpy matplotlib