#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

// Decode-only FPGA service counters. The transfer duration contains H2IP DMA,
// IP2H DMA, and the required host read of the output buffer; it excludes
// graph construction, input preparation, host accumulation, and sampling.
struct fpga_perf_decode_data {
    int64_t decode_tokens;
    int64_t decode_wall_us;
    int64_t fpga_matmuls;
    int64_t vpu_runs;
    int64_t ip_compute_us;
    int64_t h2ip_dma_us;
    int64_t output_transfer_us;
    int64_t preparation_us;
    int64_t direct_weight_pack_us;
    int64_t scale_pack_us;
    int64_t zdma_descriptors;
    uint64_t zdma_bytes;
    int64_t preload_dma_us;
    int64_t preload_overlap_jobs;
    int64_t run_fpga_gemvs;
    int64_t run_q8_unavailable_cpu_fallbacks;
    int64_t run_rejects;
    int64_t run_stream_drops;
    int64_t run_stream_errors;
    int64_t residency_slots_used;
    int64_t residency_slots_total;
    int64_t residency_hits;
    int64_t residency_misses;
};

// Return values from fpga_try_matmul[_extended]().  CPU shadow is an explicit
// contract mode: the VPU result is checked but GGML retains ownership of dst.
// It is not a hardware-unavailable CPU fallback.
enum fpga_matmul_route {
    FPGA_MATMUL_NOT_HANDLED          = 0,
    FPGA_MATMUL_FPGA_DST             = 1,
    FPGA_MATMUL_CONTRACT_CPU_SHADOW  = 2,
};

int  fpga_init(void);
void fpga_cleanup(void);

// Side-effect-free environment queries used before FPGA initialization.
int fpga_source_audit_only_requested(void);
int fpga_contract_check_requested(void);

// C0 is allowed to map and drive ZDMA/VPU only after llama-model-loader has
// validated every model tensor.  This is a process-local handshake; it never
// accesses board hardware or mutates model data.
void fpga_mark_model_tensor_validation_passed(void);
int  fpga_model_tensor_validation_passed(void);


void fpga_set_context(int layer_id, int seq_pos, int is_attn);
int  fpga_get_sequence_position(void);
void fpga_advance_sequence_position(int n_tokens);
int  fpga_perf_decode_get(struct fpga_perf_decode_data * data);

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
