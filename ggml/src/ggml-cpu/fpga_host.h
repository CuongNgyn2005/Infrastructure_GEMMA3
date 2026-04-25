#pragma once
#include <stdint.h>
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
void fpga_set_context(int layer_id, int seq_pos, int is_attn);
int fpga_try_matmul_extended(
    const struct ggml_tensor * src0,   // weights (Q8_0)
    const struct ggml_tensor * src1,   // activations (F32)
    const struct ggml_tensor * dst,    // output (F32)
    int ith,                     // thread id
    int layer_id,                // which layer (0-31)
    int seq_pos,                 // sequence position (for K-V cache)
    int is_attention             // 0=matmul, 1=attention with URAM cache
);
// Hàm HIGH-LEVEL — được gọi từ ggml-cpu.c
// Nhận ggml_tensor*, tự bóc tách data, gọi fpga_run_matmul
// src0 = weight (Q8_0), src1 = activation (F32), dst = output
struct ggml_tensor;
int fpga_try_matmul(
   const struct ggml_tensor * src0,   // weight B (Q8_0)
   const struct ggml_tensor * src1,   // activation A (F32)
   const struct ggml_tensor * dst,    // output C (F32)
    int ith                      // thread id
);

#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C"
#endif
void fpga_reset_kv_cache(void);