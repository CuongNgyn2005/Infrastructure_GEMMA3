#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

int  fpga_init(void);
void fpga_cleanup(void);

// Legacy low-level API kept for link compatibility. The current DATN_RTL
// bitstream uses the ggml tensor hook below.
int fpga_run_matmul(
    const float *    A,
    const uint16_t * B_d,
    const int8_t *   B_qs,
    float *          C,
    int M,
    int K,
    int N,
    int ith);

void fpga_set_context(int layer_id, int seq_pos, int is_attn);
int  fpga_get_sequence_position(void);
void fpga_advance_sequence_position(int n_tokens);

// High-level hook called from ggml-cpu.c.
// src0 = Q8_0 weights, src1 = F32 activations, dst = F32 output.
int fpga_try_matmul(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ith);

int fpga_try_matmul_extended(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ith,
    int layer_id,
    int seq_pos,
    int is_attention);

void fpga_reset_kv_cache(void);

#ifdef __cplusplus
}
#endif
