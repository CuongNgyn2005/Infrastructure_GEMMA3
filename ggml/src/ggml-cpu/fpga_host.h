#pragma once
#ifdef __cplusplus
extern "C" {
#endif

int  fpga_init(void);
void fpga_cleanup(void);

// Hàm LOW-LEVEL (dùng nội bộ)
int fpga_run_matmul(
    const float*    A,
    const uint16_t* B_d,
    const int8_t*   B_qs,
    float*          C,
    int M, int K, int N,
    int ith
);

// Hàm HIGH-LEVEL — được gọi từ ggml-cpu.c
// Nhận ggml_tensor*, tự bóc tách data, gọi fpga_run_matmul
// src0 = weight (Q8_0), src1 = activation (F32), dst = output
struct ggml_tensor;
int fpga_try_matmul(
    struct ggml_tensor * src0,   // weight B (Q8_0)
    struct ggml_tensor * src1,   // activation A (F32)
    struct ggml_tensor * dst,    // output C (F32)
    int ith                      // thread id
);

#ifdef __cplusplus
}
#endif