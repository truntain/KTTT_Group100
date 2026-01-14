# JCAS Benchmark: Iterative Least Squares (ILS)

Module này chứa mã nguồn thuật toán gốc (Baseline) sử dụng phương pháp **Iterative Least Squares (ILS)** để so sánh hiệu năng với các thuật toán tối ưu bầy đàn (GWO, Hybrid GWO-GA).

## 1. Mục đích
* Cung cấp một chuẩn so sánh (Benchmark) về tốc độ hội tụ và hình dạng búp sóng.
* ILS được biết đến với tốc độ hội tụ rất nhanh nhưng dễ bị kẹt tại cực trị địa phương nếu điểm khởi tạo không tốt.

## 2. Cấu trúc File
* `main.py`: Chạy mô phỏng ILS và vẽ biểu đồ Beampattern/Hội tụ.
* `ils_optimizer.py`: Triển khai thuật toán tối ưu ILS.
* `jcas_model.py`: Các hàm tính toán vật lý của hệ thống ăng-ten.

## 3. Cách chạy
```bash
python main.py