# GWO-GA Hybrid Core (Python Version)

Thư mục này chứa mã nguồn triển khai và thử nghiệm thuật toán lai ghép **Grey Wolf Optimizer (GWO)** và **Genetic Algorithm (GA)** sử dụng hoàn toàn ngôn ngữ **Python**.

## Mô tả
Module này tập trung vào việc nghiên cứu cốt lõi của giải thuật lai (Hybrid Algorithm) và trực quan hóa kết quả chạy nghiệm.

## Thành phần chính
* **Data Logs (`data_*.txt`)**: Các file dữ liệu lịch sử hội tụ và vị trí các tác tử (wolves/nodes) được ghi lại từ quá trình chạy thuật toán.
* **Visualization (`plot_results.py`)**: Script Python dùng để vẽ đồ thị và phân tích kết quả từ các file dữ liệu.

## Điểm nổi bật
* **Mục tiêu:** Kiểm chứng lý thuyết lai ghép, cải thiện khả năng thoát khỏi cực trị địa phương của GWO truyền thống bằng cơ chế lai ghép và đột biến của GA.