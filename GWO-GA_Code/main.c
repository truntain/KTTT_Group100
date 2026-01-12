/*
 * PROJECT: PHÂN CỤM WSN TỐI ƯU BẰNG HYBRID GWO-GA
 * Ngôn ngữ: C (Standard C99)
 * Mô tả: Chạy thuật toán tối ưu và xuất dữ liệu ra file .txt
 * Sau khi chạy xong file này, hãy chạy 'plot_results.py' để xem biểu đồ.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- CẤU HÌNH HỆ THỐNG ---
#define NUM_NODES 100       // Số nút cảm biến
#define NUM_CLUSTERS 5      // Số cụm (Cluster Heads)
#define MAX_ITER 50         // Số vòng lặp tối đa
#define NUM_WOLVES 20       // Số lượng sói
#define AREA_SIZE 100.0     // Kích thước vùng mạng (100x100)
#define MUTATION_RATE 0.1   // Tỉ lệ đột biến
#define DIM (NUM_CLUSTERS * 2) // Số chiều bài toán (5 cụm * 2 tọa độ x,y)

// Cấu trúc dữ liệu
typedef struct {
    double x, y;
} Point;

typedef struct {
    double position[DIM]; // Vector vị trí (x1, y1, x2, y2...)
    double fitness;       // Giá trị thích nghi
} Wolf;

// Biến toàn cục
Point nodes[NUM_NODES];
Wolf population[NUM_WOLVES];
Wolf alpha, beta, delta;
double convergence_history[MAX_ITER];

// --- HÀM HỖ TRỢ ---

// Sinh số thực ngẫu nhiên trong khoảng [0, 1]
double rand_01() {
    return (double)rand() / (double)RAND_MAX;
}

// Sinh số thực ngẫu nhiên trong khoảng [min, max]
double rand_range(double min, double max) {
    return min + rand_01() * (max - min);
}

// Khởi tạo mạng WSN (Các nút cảm biến)
void init_nodes() {
    for (int i = 0; i < NUM_NODES; i++) {
        nodes[i].x = rand_range(0, AREA_SIZE);
        nodes[i].y = rand_range(0, AREA_SIZE);
    }
}

// Tính khoảng cách Euclide giữa 2 điểm
double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// --- HÀM MỤC TIÊU (FITNESS FUNCTION) ---
// Tính tổng khoảng cách từ các nút thường về CH gần nhất
double calculate_fitness(double *pos) {
    double total_dist = 0;
    
    // Duyệt qua từng nút cảm biến
    for (int i = 0; i < NUM_NODES; i++) {
        double min_d = 1e9; // Khoảng cách nhỏ nhất tìm được
        
        // Duyệt qua các Cluster Heads (CH) trong gen của sói
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            double ch_x = pos[2*k];
            double ch_y = pos[2*k + 1];
            double d = distance(nodes[i].x, nodes[i].y, ch_x, ch_y);
            if (d < min_d) min_d = d;
        }
        total_dist += min_d;
    }
    return total_dist;
}

// Hàm so sánh dùng cho qsort (Sắp xếp sói theo fitness tăng dần)
int compare_wolves(const void *a, const void *b) {
    Wolf *w1 = (Wolf *)a;
    Wolf *w2 = (Wolf *)b;
    if (w1->fitness < w2->fitness) return -1;
    if (w1->fitness > w2->fitness) return 1;
    return 0;
}

// --- THUẬT TOÁN CHÍNH HYBRID GWO-GA ---
void run_hybrid_GWO_GA() {
    // 1. Khởi tạo quần thể ngẫu nhiên
    for (int i = 0; i < NUM_WOLVES; i++) {
        for (int d = 0; d < DIM; d++) {
            population[i].position[d] = rand_range(0, AREA_SIZE);
        }
        population[i].fitness = calculate_fitness(population[i].position);
    }

    // Sắp xếp ban đầu để tìm Alpha, Beta, Delta
    qsort(population, NUM_WOLVES, sizeof(Wolf), compare_wolves);
    alpha = population[0];
    beta = population[1];
    delta = population[2];

    printf(">>> Bat dau chay Hybrid GWO-GA...\n");

    // 2. Vòng lặp chính
    for (int t = 0; t < MAX_ITER; t++) {
        // Lưu lịch sử để vẽ biểu đồ
        convergence_history[t] = alpha.fitness;
        
        if ((t+1) % 10 == 0) {
            printf("Vong lap %d: Best Fitness = %.4f\n", t+1, alpha.fitness);
        }

        double a = 2.0 - (double)t * (2.0 / MAX_ITER); // Tham số a giảm dần
        int half_pop = NUM_WOLVES / 2;

        // --- GIAI ĐOẠN 1: SĂN MỒI THEO GWO (Áp dụng cho Top 50% sói tốt nhất) ---
        for (int i = 0; i < half_pop; i++) {
            for (int d = 0; d < DIM; d++) {
                double r1, r2;
                
                // Tính khoảng cách tới Alpha
                r1 = rand_01(); r2 = rand_01();
                double A1 = 2*a*r1 - a; double C1 = 2*r2;
                double D_alpha = fabs(C1*alpha.position[d] - population[i].position[d]);
                double X1 = alpha.position[d] - A1*D_alpha;

                // Tính khoảng cách tới Beta
                r1 = rand_01(); r2 = rand_01();
                double A2 = 2*a*r1 - a; double C2 = 2*r2;
                double D_beta = fabs(C2*beta.position[d] - population[i].position[d]);
                double X2 = beta.position[d] - A2*D_beta;

                // Tính khoảng cách tới Delta
                r1 = rand_01(); r2 = rand_01();
                double A3 = 2*a*r1 - a; double C3 = 2*r2;
                double D_delta = fabs(C3*delta.position[d] - population[i].position[d]);
                double X3 = delta.position[d] - A3*D_delta;

                // Cập nhật vị trí trung bình
                population[i].position[d] = (X1 + X2 + X3) / 3.0;
                
                // Ràng buộc biên (Boundary Check)
                if(population[i].position[d] < 0) population[i].position[d] = 0;
                if(population[i].position[d] > AREA_SIZE) population[i].position[d] = AREA_SIZE;
            }
        }

        // --- GIAI ĐOẠN 2: TIẾN HÓA GA (Thay thế Bottom 50% sói yếu) ---
        for (int i = half_pop; i < NUM_WOLVES; i++) {
            for (int d = 0; d < DIM; d++) {
                // LAI GHÉP (Crossover): Con lai giữa Alpha và Beta
                // Công thức: Child = w * Alpha + (1-w) * Beta
                double w_cross = rand_01(); 
                double child_gene = w_cross * alpha.position[d] + (1.0 - w_cross) * beta.position[d];

                // ĐỘT BIẾN (Mutation): Thay đổi ngẫu nhiên vị trí để thoát bẫy
                if (rand_01() < MUTATION_RATE) {
                    child_gene = rand_range(0, AREA_SIZE);
                }
                
                population[i].position[d] = child_gene;
                
                // Ràng buộc biên
                if(population[i].position[d] < 0) population[i].position[d] = 0;
                if(population[i].position[d] > AREA_SIZE) population[i].position[d] = AREA_SIZE;
            }
        }

        // --- CẬP NHẬT LẠI FITNESS VÀ LÃNH ĐẠO ---
        for (int i = 0; i < NUM_WOLVES; i++) {
            population[i].fitness = calculate_fitness(population[i].position);
        }
        
        // Sắp xếp lại để tìm Alpha mới
        qsort(population, NUM_WOLVES, sizeof(Wolf), compare_wolves);
        alpha = population[0];
        beta = population[1];
        delta = population[2];
    }
}

// --- XUẤT KẾT QUẢ RA FILE (Để Python đọc) ---
void save_results() {
    printf("\nDang xuat du lieu ra file .txt ...\n");

    // 1. Lưu tọa độ các nút cảm biến (nodes)
    FILE *f_nodes = fopen("data_nodes.txt", "w");
    if (f_nodes == NULL) { printf("Loi mo file data_nodes.txt\n"); return; }
    for(int i=0; i<NUM_NODES; i++) {
        fprintf(f_nodes, "%.4f %.4f\n", nodes[i].x, nodes[i].y);
    }
    fclose(f_nodes);

    // 2. Lưu tọa độ Cluster Heads tối ưu (Kết quả cuối cùng từ Alpha)
    FILE *f_chs = fopen("data_chs.txt", "w");
    if (f_chs == NULL) { printf("Loi mo file data_chs.txt\n"); return; }
    for(int k=0; k<NUM_CLUSTERS; k++) {
        fprintf(f_chs, "%.4f %.4f\n", alpha.position[2*k], alpha.position[2*k+1]);
    }
    fclose(f_chs);

    // 3. Lưu lịch sử hội tụ (Vòng lặp - Fitness)
    FILE *f_hist = fopen("data_history.txt", "w");
    if (f_hist == NULL) { printf("Loi mo file data_history.txt\n"); return; }
    for(int t=0; t<MAX_ITER; t++) {
        fprintf(f_hist, "%d %.4f\n", t+1, convergence_history[t]);
    }
    fclose(f_hist);
    
    printf("HOAN THANH! Da tao 3 file:\n - data_nodes.txt\n - data_chs.txt\n - data_history.txt\n");
    printf("Hay chay file Python 'plot_results.py' de ve do thi.\n");
}

int main() {
    srand(42); // Cố định seed để kết quả giống nhau mỗi lần chạy
    init_nodes();
    run_hybrid_GWO_GA();
    save_results();
    return 0;
}