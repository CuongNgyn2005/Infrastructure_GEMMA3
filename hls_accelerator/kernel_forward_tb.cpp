#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include "kernel_forward.h"

// ══════════════════════════════════════════════════════════
//  Quantize 1 row (length = row_len) theo Q8_0
//  Dùng để quantize từng hàng n của B (mỗi hàng dài K)
//  Output:
//    d_row [row_len/32]  — scale fp16
//    qs_row[row_len]     — quant int8
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
//  Golden reference — layout MỚI (N, K):
//    B_d [n * (K/32) + k/32]
//    B_qs[n * K      + k   ]
//
//  C[m][n] = sum_k  A[m][k] * B_qs[n][k] * scale[n][k/32]
// ══════════════════════════════════════════════════════════
void matmul_cpu_golden(const float*    A,
                       const uint16_t* B_d,
                       const int8_t*   B_qs,
                       float*          C,
                       int M, int K, int N)
{
    int nb_k = K / QK8_0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float scale = f16_to_f32(B_d[n * nb_k + k / QK8_0]);
                sum += A[m * K + k] * (float)(B_qs[n * K + k]) * scale;
            }
            C[m * N + n] = sum;
        }
}

// ══════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════
int main() {
    const int M = 16, K = 64, N = 64;
    printf("=== kernel_forward testbench ===\n");
    printf("M=%d K=%d N=%d | B layout: (%d rows x %d cols)\n\n", M, K, N, N, K);

    // ── Kiểm tra f16/f32 ──
    uint16_t h = f32_to_f16(0.5f);
    float    v = f16_to_f32(h);
    printf("[f16] f32_to_f16(0.5)  = 0x%04X  (expect 0x3800)\n", h);
    printf("[f16] f16_to_f32(above)= %.6f  (expect 0.500000)\n\n", v);
    if (h != 0x3800U || v < 0.499f || v > 0.501f) {
        printf("FAIL: f16/f32 conversion broken!\n"); return 1;
    }

    int nb_k = K / QK8_0;   // = 2

    // ── Alloc ──
    std::vector<float>    A_host  (M * K);
    std::vector<float>    B_float (N * K);   // FIX: N×K (không phải K×N)
    std::vector<uint16_t> B_d     (N * nb_k); // FIX: N×(K/32)
    std::vector<int8_t>   B_qs    (N * K);    // FIX: N×K
    std::vector<float>    C_cpu   (M * N, 0.0f);
    std::vector<float>    C_fpga  (M * N, 0.0f);

    // ── Dữ liệu ngẫu nhiên ──
    srand(123);
    for (auto& x : A_host)  x = (float)rand()/(float)RAND_MAX - 0.5f;
    for (auto& x : B_float) x = (float)rand()/(float)RAND_MAX - 0.5f;

    // ── Quantize B: mỗi hàng n dài K ──
    for (int n = 0; n < N; ++n)
        quantize_row_q8_0(
            &B_float[n * K],    // float row n, length K
            &B_d    [n * nb_k], // scale output, K/32 values
            &B_qs   [n * K],    // quant output, K values
            K
        );

    // ── Chạy golden ──
    matmul_cpu_golden(A_host.data(), B_d.data(), B_qs.data(), C_cpu.data(),  M, K, N);

    // ── Chạy kernel HLS ──
    kernel_forward   (A_host.data(), B_d.data(), B_qs.data(), C_fpga.data(), M, K, N);

    // ── So sánh ──
    int   errors   = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_cpu[i] - C_fpga[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
        if (diff > 1e-2f) {
            if (errors < 10)
                printf("[MISMATCH] [%d,%d] cpu=%.5f fpga=%.5f diff=%.6f\n",
                       i/N, i%N, C_cpu[i], C_fpga[i], diff);
            errors++;
        }
    }

    printf("\n--- Kết quả ---\n");
    printf("Max diff : %.6f\n", max_diff);
    printf("Avg diff : %.6f\n", sum_diff / (M * N));
    printf("Errors   : %d / %d\n", errors, M * N);

    printf("\n[Mẫu C_cpu ] "); for(int i=0;i<6;i++) printf("%8.4f ", C_cpu[i]);
    printf("\n[Mẫu C_fpga] "); for(int i=0;i<6;i++) printf("%8.4f ", C_fpga[i]);
    printf("\n");

    if (errors == 0)
        printf("\n*** TEST PASSED! ***\n");
    else
        printf("\n*** TEST FAILED! %d errors ***\n", errors);

    return (errors == 0) ? 0 : 1;
}
