# Nhóm 100 - Nghiên cứu và Ứng dụng Thuật toán Tối ưu cho JCAS

Chào mừng đến với kho lưu trữ mã nguồn của Nhóm 100. Dự án này tập trung vào nghiên cứu thuật toán **Grey Wolf Optimizer (GWO)**, phát triển biến thể lai ghép **Hybrid GWO-GA** và ứng dụng thực tiễn vào bài toán tối ưu hóa búp sóng đa chức năng trong mạng 6G (JCAS - Joint Communication and Sensing).

---

## 📂 Cấu trúc Dự án

Dự án được tổ chức thành 4 module chính, đi từ nghiên cứu lý thuyết, tạo chuẩn so sánh đến ứng dụng nâng cao:

### 1. [GWO-GA Core Research](./GWO-GA_Code)
* **Loại hình:** Nghiên cứu cơ bản (Legacy).
* **Mô tả:** Mã nguồn triển khai cốt lõi của giải thuật lai ghép trên nền tảng Python thuần túy.
* **Mục đích:** Kiểm chứng lý thuyết lai ghép, phân tích hành vi của các tác tử (Sói/Nhiễm sắc thể) và khả năng thoát khỏi cực trị địa phương trên các hàm toán học trước khi áp dụng vào vật lý.
* **Dữ liệu:** Chứa các file log (`data_*.txt`) và script vẽ đồ thị phân tích.

### 2. [JCAS Benchmark (ILS)](./jcas)
* **Loại hình:** Chuẩn so sánh (Baseline).
* **Thuật toán:** Iterative Least Squares (ILS).
* **Mô tả:** Module này tạo ra kết quả chuẩn dựa trên phương pháp giải tích toán học truyền thống.
* **Vai trò:** Cung cấp "mốc vàng" (Gold Standard) về tốc độ hội tụ và dạng búp sóng để đánh giá hiệu năng của các thuật toán Meta-heuristic (GWO, Hybrid).

### 3. [JCAS Standard Optimization](./jcas_GWO)
* **Loại hình:** Ứng dụng GWO nguyên bản.
* **Mô tả:** Áp dụng thuật toán **Standard GWO** để giải quyết bài toán tối ưu hóa trọng số ăng-ten JCAS.
* **Hạn chế:** Mặc dù tạo được hướng chính tốt, thuật toán thường bị kẹt tại các cực trị địa phương, dẫn đến mức búp sóng phụ (Sidelobe Level - SLL) còn cao, gây nhiễu cho hệ thống.

### 4. [JCAS Hybrid GWO-GA (Proposed)](./jcas_GWO-GA)
* **Loại hình:** Giải pháp đề xuất (Advanced).
* **Mô tả:** Triển khai biến thể **Hybrid GWO-GA** tối ưu cho bài toán JCAS.
* **Cơ chế hoạt động:**
    * **Phân tầng:** Top 50% cá thể tốt nhất thực hiện cơ chế săn mồi GWO (Khai thác - Exploitation).
    * **Lai ghép:** Bottom 50% cá thể yếu được thay thế bằng con lai từ nhóm tốt (GA Crossover).
    * **Đột biến:** Áp dụng đột biến ngẫu nhiên (Mutation) để duy trì sự đa dạng quần thể (Khám phá - Exploration).
* **Tính năng:** Tự động vẽ đồ thị so sánh trực tiếp với Benchmark và GWO thường.

---

## 📊 Tóm tắt Kết quả Nghiên cứu

Dưới đây là bảng so sánh hiệu năng giữa thuật toán đề xuất và các phương pháp khác:

| Thuật toán | Định hướng (Main Lobe) | Khử nhiễu (SLL) | Khả năng thoát bẫy | Đánh giá |
| :--- | :---: | :---: | :--- | :--- |
| **Standard GWO** | Tốt | Trung bình (-10dB) | Kém (Dễ bão hòa sớm) | Phù hợp bài toán đơn giản |
| **Hybrid GWO-GA** | **Tốt** | **Tốt (-15dB)** | **Cao (Nhờ đột biến)** | **Cân bằng tốt nhất** |
| **Benchmark (ILS)**| Lý tưởng | Rất tốt (<-20dB)| N/A (Giải tích) | Tốc độ nhanh nhưng khó mở rộng |

> **Kết luận:** Hybrid GWO-GA giúp giảm mức búp sóng phụ (SLL) thêm **3-5 dB** so với GWO truyền thống, tiệm cận gần hơn với giới hạn lý thuyết của ILS mà không cần tính toán ma trận nghịch đảo phức tạp.

---

## 🚀 Hướng dẫn Cài đặt & Chạy

### 1. Yêu cầu môi trường
Dự án sử dụng **Python 3.8+** và các thư viện tính toán khoa học:

```bash
pip install numpy matplotlib