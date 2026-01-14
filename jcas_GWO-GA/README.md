# Ứng dụng Hybrid GWO-GA cho JCAS

Module này triển khai thuật toán lai ghép **GWO-GA (Grey Wolf Optimizer - Genetic Algorithm)** để giải quyết bài toán tối ưu hóa đa mục tiêu trong hệ thống JCAS (Joint Communication and Sensing).

## 1. Tổng quan
Thuật toán tận dụng khả năng khai thác (exploitation) của GWO và khả năng khám phá (exploration) thông qua cơ chế lai ghép/đột biến của GA để tối ưu hóa trọng số Beamforming.

* **Input:** Số lượng ăng-ten (N=64), góc người dùng, góc mục tiêu.
* **Output:** Vector trọng số tối ưu giúp tối đa hóa SINR cho thông tin liên lạc và độ lợi Radar, đồng thời giảm thiểu nhiễu (SLL).

## 2. Cấu trúc File
* `main.py`: Chương trình chính chạy mô phỏng đơn lẻ.
* `hybrid_optimizer.py`: Class chứa logic thuật toán lai (Khởi tạo quần thể -> Săn mồi GWO -> Lai ghép & Đột biến GA).
* `compare_algorithms.py`: Script so sánh hiệu năng giữa GWO thường và Hybrid GWO-GA.
* `jcas_model.py`: Mô hình hệ thống JCAS (Steering vector, Beampattern).

## 3. Kết quả
Thuật toán lai giúp cân bằng tốt hơn giữa tốc độ hội tụ và chất lượng nghiệm so với GWO truyền thống, đặc biệt trong không gian tìm kiếm phức tạp của bài toán đa mục tiêu.