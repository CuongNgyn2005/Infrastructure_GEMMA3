#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm> // Cho std::max
#include "kernel_forward.h"

// ══════════════════════════════════════════════════════════
//  Quantize 1 row theo Q8_0
// ══════════════════════════════════════════════════════════
void quantize_row_q8_0(const float* x_row, uint16_t* d_row, int8_t* qs_row, int row_len) {
    int nb = row_len / QK8_0;
    for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float v = fabsf(x_row[b * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d  = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        d_row[b] = f32_to_f16(d);
        for (int j = 0; j < QK8_0; ++j)
            qs_row[b * QK8_0 + j] = (int8_t)roundf(x_row[b * QK8_0 + j] * id);
    }
}

// ══════════════════════════════════════════════════════════
//  Golden reference — layout (N, K)
// ══════════════════════════════════════════════════════════
void matmul_cpu_golden(const float* A,
                       const uint16_t* B_d,
                       const int8_t* B_qs,
                       float* C,
                       int M, int K, int N)
{
    int nb_k = K / QK8_0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float scale = f16_to_f32(B_d[n * nb_k + k / QK8_0]);
                sum += A[m * K + k] * (float)(B_qs[n * K + k]) * scale;
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    const int M = 16, K = 128, N = 64; // Test size
    printf("=== kernel_forward testbench ===\n");
    printf("M=%d K=%d N=%d | B layout: (%d rows x %d cols)\n\n", M, K, N, N, K);

    // ── Kiểm tra conversion f16/f32 ──
    uint16_t h = f32_to_f16(0.5f);
    float    v = f16_to_f32(h);
    if (h != 0x3800U || v < 0.499f || v > 0.501f) {
        printf("FAIL: f16/f32 conversion broken!\n"); return 1;
    }

    int nb_k = K / QK8_0;

    // ── Bơm kích thước mảng an toàn để chống sập Co-Sim (Crash do wrapper đọc vượt giới hạn) ──
    int min_depth = 65536;
    std::vector<float>    A_host  (std::max(M * K, min_depth));
    std::vector<float>    B_float (std::max(N * K, min_depth));
    std::vector<uint16_t> B_d     (std::max(N * nb_k, min_depth));
    std::vector<int8_t>   B_qs    (std::max(N * K, min_depth));
    std::vector<float>    C_cpu   (std::max(M * N, min_depth), 0.0f);
    std::vector<float>    C_fpga  (std::max(M * N, min_depth), 0.0f);

    // ── Khởi tạo ngẫu nhiên ──
    srand(123);
    for (int i = 0; i < M * K; ++i) A_host[i] = (float)rand()/(float)RAND_MAX - 0.5f;
    for (int i = 0; i < N * K; ++i) B_float[i] = (float)rand()/(float)RAND_MAX - 0.5f;

    // ── Quantize B ──
    for (int n = 0; n < N; ++n) {
        quantize_row_q8_0(&B_float[n * K], &B_d[n * nb_k], &B_qs[n * K], K);
    }

    // Làm bẩn mảng kết quả FPGA để chắc chắn kernel chạy đúng (ghi đè hết rác)
    for (int i = 0; i < std::max(M * N, min_depth); ++i) {
        C_fpga[i] = -9999.0f;
    }

    // ── Chạy ──
    matmul_cpu_golden(A_host.data(), B_d.data(), B_qs.data(), C_cpu.data(),  M, K, N);
    kernel_forward   (A_host.data(), B_d.data(), B_qs.data(), C_fpga.data(), M, K, N);

    // ── So sánh kết quả ──
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_cpu[i] - C_fpga[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-2f || std::isnan(C_fpga[i])) { // Cảnh báo ngay nếu sinh ra NaN
            if (errors < 10)
                printf("[MISMATCH] [%d,%d] cpu=%.5f fpga=%.5f diff=%.6f\n",
                       i/N, i%N, C_cpu[i], C_fpga[i], diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("\n*** TESTBENCH THIEN THANH CONG! CHUA LANH BENH NaN & STRIDE ***\n");
    } else {
        printf("\n*** TEST FAILED voi %d loi! ***\n", errors);
    }

    return (errors == 0) ? 0 : 1;
}
