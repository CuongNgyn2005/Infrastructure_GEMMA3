#!/usr/bin/env python3
"""Exercise the production P2 scheduler with RAM-only hardware stubs."""

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()


def extract_definition(signature: str) -> str:
    """Extract a balanced C++ definition, skipping forward declarations."""
    search = 0
    while True:
        start = SOURCE.find(signature, search)
        if start < 0:
            raise AssertionError(f"missing definition: {signature}")
        body_start = SOURCE.find("{", start)
        semicolon = SOURCE.find(";", start, body_start)
        if semicolon < 0:
            depth = 0
            for position in range(body_start, len(SOURCE)):
                character = SOURCE[position]
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        return SOURCE[start : position + 1]
            raise AssertionError(f"unterminated definition: {signature}")
        search = body_start + 1


scheduler = extract_definition("static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(")
release = extract_definition("static bool fpga_release_pl_scaled_q8_tile_job(")
accumulate = extract_definition("static bool fpga_accumulate_pl_scaled_q8_tile_job(")
residency_align = extract_definition("static bool fpga_p2_residency_align_up(")
residency_end = extract_definition("static bool fpga_p2_residency_range_end(")

# Keep the test tied to the production handoff order.  Runtime assertions below
# execute the extracted scheduler; these guards only make an accidental source
# extraction of an older/partial scheduler fail early.
assert "fpga_p2_dma_pipeline_consume(lookahead, prepared" in scheduler
assert "fpga_p2_dma_pipeline_finalize(lookahead)" in scheduler
running = scheduler[scheduler.index("                if (running) {") :]
assert running.index("fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals, false)") < running.index(
    "fpga_p2_dma_pipeline_finalize(lookahead)"
)


harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "fpga_q8_layout.h"

struct ggml_tensor { void * data = nullptr; };
struct block_q8_0_t {};
struct fpga_weight_cache_entry_t {};
struct fpga_prompt_weight_staging_t {};

constexpr int VPU_BLOCK_BEATS = 2;
constexpr int VPU_RESULT_PACK_LANES = 4;
constexpr int FPGA_SLOT_FREE = 0;
constexpr int FPGA_SLOT_COMPUTING = 4;
constexpr int FPGA_SLOT_DMA_DRAINING = 6;
constexpr int FPGA_SLOT_HOST_CONSUMING = 7;
constexpr uint32_t ACT_BASE = 0x00010000U, ACT_END = 0x00020000U;
constexpr uint32_t WEIGHT_BASE = 0x00100000U, WEIGHT_END = 0x00200000U;
constexpr uint32_t RESULT_BASE = 0x00200000U, RESULT_END = 0x00210000U;
constexpr uint32_t SPU_OUT_BASE = 0x00340000U, SPU_OUT_END = 0x00380000U;
constexpr uint32_t SPU_PARAM_BASE = 0x00380000U, SPU_PARAM_END = 0x003C0000U;
constexpr uint32_t P2_PIPELINE_WEIGHT_SLOT_BYTES = 512U * 1024U;
constexpr uint32_t P2_PIPELINE_SCALE_SLOT_BYTES = 64U * 1024U;
constexpr uint32_t P2_PIPELINE_WEIGHT_SLOT_COUNT = 2U;
constexpr uint32_t P2_PIPELINE_SCALE_SLOT_COUNT = 2U;
constexpr size_t P2_PIPELINE_PL_SCALE_BYTES = 64U * 1024U;
constexpr size_t WEIGHT_CACHE_ALIGN = 4096U;
constexpr uint32_t WEIGHT_CACHE_BASE = 0x01000000U;
constexpr uint32_t P2_WEIGHT_RESIDENCY_END = 0x03000000U;
constexpr uint32_t P2_WEIGHT_RESIDENCY_NO_SLOT = UINT32_MAX;
constexpr uint32_t P2_WEIGHT_RESIDENCY_LAYOUT_V2 = 2U;
constexpr uint64_t DDR_BASE_PHYS = 0U, LMM_BASE_PHYS = 0U;
constexpr size_t DDR_REGION_SIZE = 0x10000000U;
constexpr size_t FPGA_P2_PACK_PARALLEL_MIN_BYTES = 256U * 1024U;

struct fpga_stage_totals_t {
    long long prep_us = 0, dma_act_us = 0, dma_weight_us = 0, dma_scale_us = 0;
    long long dma_result_us = 0, ip_compute_us = 0, host_result_us = 0, host_accum_us = 0;
    long long weight_cache_hits = 0, weight_cache_misses = 0, weight_cache_lookup_us = 0;
    long long weight_cache_crc_us = 0, weight_pack_us = 0;
    long long prep_weight_select_us = 0, prep_direct_weight_pack_us = 0;
    long long prep_scale_pack_us = 0, prep_act_pack_us = 0;
    long long activation_scale_fp16_overflows = 0;
    long long scheduler_prepare_overlap_us = 0, scheduler_prepare_late_us = 0;
    long long scheduler_prepare_headroom_us = 0;
    long long scheduler_output_to_launch_us = 0, scheduler_retire_to_launch_us = 0;
    long long scheduler_handoffs = 0, scheduler_prepare_late_jobs = 0;
    long long result_overlap_jobs = 0;
    long long result_overlap_host_us = 0;
    long long bank_h2ip_us[2] = {}, bank_compute_us[2] = {}, bank_ip2host_us[2] = {};
    long long bank_host_read_us[2] = {}, bank_jobs[2] = {};
    long long first_ip_launch_mono_us = 0, last_ip_output_ready_mono_us = 0;
    size_t activation_bytes = 0, weight_bytes = 0, scale_bytes = 0, result_bytes = 0;
    long long vpu_runs = 0;
};

struct fpga_tile_job_t {
    int bank = 0;
    uint32_t job_id = 0;
    unsigned long long matmul_call_id = 0;
    int graph_seq = 0, layer_id = 0;
    int64_t shape_k = 0, shape_n = 0, shape_m = 0;
    bool cpu_shadow_dst = false, pingpong_scheduler = false;
    const char * tensor_name = nullptr;
    uint32_t tile_id = 0, tensor_id = 0;
    int64_t row0 = 0, k_block0 = 0, col = 0;
    int rows = 0, group_blocks = 0, group_beats = 0;
    size_t act_bytes = 0, weight_bytes = 0, scale_bytes = 0;
    size_t spu_result_bytes = 0, result_bytes = 0;
    uint32_t result_values = 0, result_words = 0, scale_words = 0;
    uint32_t weight_src_off = WEIGHT_BASE, scale_src_off = SPU_PARAM_BASE;
    bool weight_cache_hit = false, p2_residency_hit = false;
    uint32_t p2_residency_slot = P2_WEIGHT_RESIDENCY_NO_SLOT;
    uint64_t p2_residency_epoch = 0;
    uint32_t p2_residency_seal = 0;
    const block_q8_0_t * act_group = nullptr;
    const ggml_tensor * src0 = nullptr;
    const fpga_weight_cache_entry_t * weight_cache = nullptr;
    std::vector<int32_t> partial;
    std::vector<float> weight_scales;
    long long result_clear_us = 0, dma_act_us = 0, dma_weight_us = 0, dma_scale_us = 0;
    long long dma_result_us = 0, ip_start_us = 0, measured_compute_us = 0, ip_compute_us = 0;
    long long host_result_us = 0;
    bool output_dma_ready = false, hardware_released = false, result_consumed = false;
    long long event_prep_begin_us = 0, event_prep_done_us = 0, event_submit_begin_us = 0;
    long long event_input_transfer_begin_us = 0, event_launch_us = 0, event_vpu_done_us = 0;
    long long event_spu_finality_us = 0, event_retire_us = 0;
    long long handoff_prev_output_ready_us = 0, handoff_prev_retire_us = 0;
    uint32_t vpu_status = 0, spu_stream_count_before = 0, spu_stream_done_before = 0;
    uint32_t spu_stream_out_before = 0, spu_stream_drop_before = 0, spu_stream_error_before = 0;
    bool weight_bank_reused = false;
};

struct fpga_p2_dma_pipeline_lookahead_t {
    bool valid = false, worker_submitted = false, consumed = false;
    uint64_t generation = 0, source_epoch = 0;
    size_t pair_count = 0, helper_pair_begin = 0, main_pair_next = 0;
    size_t words_per_pair = 0, expected_words = 0, expected_scale_entries = 0;
    size_t callback_bytes = 0, main_written_words = 0, main_scale_entries = 0;
    long long helper_service_us = 0, residual_us = 0, callback_us = 0;
    fpga_stage_totals_t * totals = nullptr;
    const void * weight_data_base = nullptr;
    volatile uint32_t * weight_words = nullptr;
    volatile uint32_t * scale_words = nullptr;
    fpga_tile_job_t job = {};
};

using fpga_dma_poll_hook_fn = bool (*)(void *, long long);
struct fpga_dma_poll_hook_t { fpga_dma_poll_hook_fn fn = nullptr; void * context = nullptr; };

int g_vpu_max_rows = 3, g_vpu_max_beats = 128;
int g_packed_q8_max_blocks = 2, g_packed_q8_result_words = 2;
int configured_max_rows = 3, configured_max_blocks = 2;
bool configured_descriptor = true;
int max_blocks = 2, test_blocks = 0, test_columns = 0, test_rows = 0;
bool g_p1_sched_summary_enabled = true;
bool g_p2_result_overlap_enabled = false, g_vpu_descriptor_supported = true;
bool g_pingpong_timing_enabled = false;
bool g_p2_pack_dma_pipeline_enabled = false, pipeline_resident = false;
bool g_p2_weight_residency_enabled = false;
long long g_p2_weight_residency_budget_mb = 1;
uint32_t g_p2_residency_next_off = 0;
size_t g_p2_residency_next_slot = 0;
uint64_t g_p2_weight_residency_epoch = 1;
std::vector<int> g_p2_resident_tiles(128);
long long g_p2_pack_dma_pipeline_eligible_jobs = 0;
long long g_p2_pack_dma_pipeline_consumed_jobs = 0;
long long g_p2_pack_dma_pipeline_rejected_jobs = 0;
unsigned long long g_active_matmul_call_id = 0;
int g_active_matmul_graph_seq = 0, g_active_matmul_layer_id = 0;
int64_t g_active_matmul_shape_k = 0, g_active_matmul_shape_n = 0, g_active_matmul_shape_m = 0;
bool g_active_matmul_cpu_shadow = false, g_active_matmul_pingpong = true;
const char * g_active_matmul_tensor_name = nullptr;
struct { long long pingpong_pairs = 0; } g_p1_sched_summary;

static int packed_q8_group_blocks_for_rows(int rows, int remaining_blocks) {
    const int beat_limited_blocks = std::max(1, g_vpu_max_beats / VPU_BLOCK_BEATS);
    const int result_limited_blocks =
        std::max(1, (g_packed_q8_result_words * VPU_RESULT_PACK_LANES) / std::max(1, rows));
    int blocks = std::min(g_packed_q8_max_blocks, beat_limited_blocks);
    blocks = std::min(blocks, result_limited_blocks);
    blocks = std::min(blocks, remaining_blocks);
    return std::max(1, blocks);
}

using Key = std::tuple<long long, long long, long long, long long>;

void LOGE(const char *, ...) {}
void LOGSTAGE(const char *, ...) {}
void fpga_log_line(bool, const char *, bool, const char *, ...) {}
template<class... Args> void fpga_log_line(Args...) {}
template<class... Args> void p2_trace_first_tile(Args...) {}
template<class... Args> void p2_event_trace(Args...) {}
const char * p2_bank_label(int) { return "test"; }
bool should_log_detail_run(unsigned) { return false; }
long long now_us() { return 100; }
long long p2_event_now_us() { return 100; }
uint32_t fpga_next_job_id() { static uint32_t id = 1000; return ++id; }
uint32_t fpga_tensor_id_from_ptr(const ggml_tensor * tensor) { return (uint32_t) (uintptr_t) tensor; }
void fpga_fatal(const char *, ...) { assert(false); }
void mmio_fence() {}

long long operations = 0, fail_at = 0;
bool step() { return ++operations != fail_at; }
bool dma_success = true;
int active = -1, loads = 0, activations = 0, scales = 0, launches = 0, reused = 0;
long long pending_result = -1;
int consumed_jobs = 0, overlap_reads = 0, release_writes = 0;
int cross_prepares = 0, expected_crossings = 0;
std::vector<Key> prepared_order, consumed_order;
std::vector<int64_t> pipeline_admitted_k, pipeline_consumed_k, pipeline_activation_k;
std::vector<int64_t> pipeline_admitted_rows, pipeline_consumed_rows;
const block_q8_0_t * expected_pipeline_act_base = nullptr;
int pipeline_finalized = 0;
fpga_tile_job_t returned;
long long logged_jobs = -1;
unsigned long long logged_bytes = 0;
std::vector<float> expected, stored;
std::vector<int> reads, stores;

Key banks[2];
bool valid[2] = {};
Key key(const fpga_tile_job_t & j) { return {j.row0, j.rows, j.k_block0, j.group_blocks}; }

size_t allocation_bytes(int rows, int blocks) {
    const size_t weights = ((size_t) rows + 1U) / 2U * (size_t) blocks * 64U;
    const size_t scales_bytes = (size_t) rows * (size_t) blocks * 2U;
    return ((weights + 4095U) / 4096U + (scales_bytes + 4095U) / 4096U) * 4096U;
}
bool expect_cross(int rows) {
    const size_t available = P2_WEIGHT_RESIDENCY_END - g_p2_residency_next_off;
    return test_columns == 1 && g_p2_result_overlap_enabled && g_vpu_descriptor_supported &&
           (!g_p2_weight_residency_enabled || g_p2_residency_next_slot == 128 ||
            available < allocation_bytes(rows, std::min(max_blocks, test_blocks)));
}

bool fpga_dma_copy(uint64_t, uint64_t, size_t, const char *) { ++loads; return dma_success; }
bool fpga_dma_copy(uint64_t, uint64_t, size_t, const char *, const fpga_dma_poll_hook_t *) {
    ++loads;
    return dma_success;
}
void vpu_select_banks(int bank, int) { assert(bank != active); }

bool fpga_p2_dma_pipeline_geometry(size_t rows, size_t groups, int beats, size_t * weight_bytes,
                                    size_t * scale_bytes, size_t * pairs, size_t * words_per_pair,
                                    size_t * scale_entries) {
    if (!weight_bytes || !scale_bytes || !pairs || !words_per_pair || !scale_entries || rows == 0 || groups == 0 ||
        beats != (int) groups * VPU_BLOCK_BEATS || rows > (size_t) g_vpu_max_rows) return false;
    *pairs = (rows + 1U) / 2U;
    *words_per_pair = (size_t) beats * 8U;
    *scale_entries = rows * groups;
    *weight_bytes = *pairs * *words_per_pair * sizeof(uint32_t);
    *scale_bytes = ((*scale_entries + VPU_RESULT_PACK_LANES - 1U) / VPU_RESULT_PACK_LANES) * 16U;
    return *pairs > 0 && *weight_bytes > 0 && *scale_bytes > 0;
}
bool fpga_p2_dma_pipeline_slot_ranges(uint32_t, uint32_t, size_t, size_t) { return true; }
void fpga_p2_dma_pipeline_invalidate(fpga_p2_dma_pipeline_lookahead_t * lookahead) {
    if (lookahead) { lookahead->valid = false; lookahead->worker_submitted = false; lookahead->consumed = false; }
}

bool fpga_p2_residency_contains(const ggml_tensor *, int64_t, int, int64_t, int, int, size_t) {
    return pipeline_resident;
}

bool fpga_p2_dma_pipeline_admit(fpga_p2_dma_pipeline_lookahead_t & lookahead, const ggml_tensor * src0,
                                const void * weight_data_base, const block_q8_0_t * act_group, int64_t row0,
                                int rows, int64_t k_block0, int groups, int64_t col, uint32_t tile_id, int bank,
                                fpga_stage_totals_t * totals, uint32_t weight_src_off, uint32_t scale_src_off) {
    size_t weight_bytes = 0, scale_bytes = 0, pairs = 0, words = 0, entries = 0;
    if (!fpga_p2_dma_pipeline_geometry((size_t) rows, (size_t) groups, groups * VPU_BLOCK_BEATS,
                                        &weight_bytes, &scale_bytes, &pairs, &words, &entries)) return false;
    lookahead = {};
    lookahead.valid = true;
    lookahead.source_epoch = g_p2_weight_residency_epoch;
    lookahead.pair_count = pairs;
    lookahead.helper_pair_begin = pairs / 2U;
    lookahead.main_pair_next = lookahead.helper_pair_begin;
    lookahead.words_per_pair = words;
    lookahead.expected_words = pairs * words;
    lookahead.expected_scale_entries = entries;
    const size_t helper_rows = (size_t) rows - lookahead.helper_pair_begin * 2U;
    lookahead.main_written_words = lookahead.helper_pair_begin * words;
    lookahead.main_scale_entries = entries - helper_rows * (size_t) groups;
    lookahead.totals = totals;
    lookahead.weight_data_base = weight_data_base;
    lookahead.job = {};
    lookahead.job.bank = bank & 1;
    lookahead.job.job_id = tile_id + 1U;
    lookahead.job.tile_id = tile_id;
    lookahead.job.tensor_id = fpga_tensor_id_from_ptr(src0);
    lookahead.job.row0 = row0;
    lookahead.job.rows = rows;
    lookahead.job.k_block0 = k_block0;
    lookahead.job.group_blocks = groups;
    lookahead.job.group_beats = groups * VPU_BLOCK_BEATS;
    lookahead.job.col = col;
    lookahead.job.act_group = act_group;
    lookahead.job.src0 = src0;
    lookahead.job.act_bytes = (size_t) lookahead.job.group_beats * 16U;
    lookahead.job.weight_bytes = weight_bytes;
    lookahead.job.scale_bytes = scale_bytes;
    lookahead.job.spu_result_bytes = (size_t) rows * 16U;
    lookahead.job.result_values = (uint32_t) entries;
    lookahead.job.result_words = (uint32_t) ((entries + 3U) / 4U);
    lookahead.job.scale_words = lookahead.job.result_words;
    lookahead.job.weight_src_off = weight_src_off;
    lookahead.job.scale_src_off = scale_src_off;
    lookahead.job.event_prep_begin_us = 1;
    ++g_p2_pack_dma_pipeline_eligible_jobs;
    pipeline_admitted_k.push_back(k_block0);
    pipeline_admitted_rows.push_back(row0);
    return true;
}

bool fpga_p2_dma_pipeline_consume(fpga_p2_dma_pipeline_lookahead_t & lookahead, fpga_tile_job_t & job,
                                  fpga_stage_totals_t * totals, const ggml_tensor * src0, const void * weights,
                                  int64_t row0, int rows, int64_t k_block0, int groups, int64_t col,
                                  uint32_t tile_id) {
    if (!lookahead.valid || lookahead.consumed || lookahead.worker_submitted || lookahead.totals != totals ||
        lookahead.source_epoch != g_p2_weight_residency_epoch || lookahead.job.src0 != src0 ||
        lookahead.weight_data_base != weights || lookahead.job.row0 != row0 || lookahead.job.rows != rows ||
        lookahead.job.k_block0 != k_block0 || lookahead.job.group_blocks != groups || lookahead.job.col != col ||
        lookahead.job.tile_id != tile_id || lookahead.job.bank != (int) (tile_id & 1U)) return false;
    job = lookahead.job;
    lookahead.consumed = true;
    lookahead.valid = false;
    ++g_p2_pack_dma_pipeline_consumed_jobs;
    pipeline_consumed_k.push_back(k_block0);
    pipeline_consumed_rows.push_back(row0);
    assert(job.act_group != nullptr);
    return true;
}
bool fpga_p2_dma_pipeline_prepare_activation(fpga_tile_job_t & job, fpga_stage_totals_t *) {
    assert(active == -1);
    assert(expected_pipeline_act_base != nullptr);
    assert(job.act_group == expected_pipeline_act_base + job.k_block0);
    pipeline_activation_k.push_back(job.k_block0);
    job.event_prep_begin_us = 1;
    job.event_prep_done_us = 2;
    return true;
}
bool fpga_p2_dma_pipeline_finalize(fpga_p2_dma_pipeline_lookahead_t & lookahead) {
    if (!lookahead.valid || lookahead.consumed) return false;
    ++pipeline_finalized;
    lookahead.worker_submitted = false;
    return true;
}
bool fpga_p2_dma_pipeline_poll(void *, long long) { return true; }
struct fpga_p2_dma_pipeline_scope_guard {
    fpga_p2_dma_pipeline_lookahead_t * lookahead;
    ~fpga_p2_dma_pipeline_scope_guard() {
        if (lookahead && (lookahead->valid || lookahead->worker_submitted)) fpga_p2_dma_pipeline_invalidate(lookahead);
    }
};

void store_dst_value(const ggml_tensor *, int64_t row, int64_t col, float value) {
    assert(row >= 0 && row < test_rows && col >= 0 && col < test_columns);
    const size_t index = (size_t) col * (size_t) test_rows + (size_t) row;
    assert(++stores[index] == 1 && reads[index] == test_blocks);
    assert(std::memcmp(&value, &expected[index], sizeof(value)) == 0);
    stored[index] = value;
}
int64_t raw_result(int64_t row, int64_t col, int64_t block) {
    const int group = (int) block / max_blocks;
    const int64_t base = group % 3 == 0 ? (1LL << 50) : group % 3 == 2 ? -(1LL << 50) : 0;
    return base + ((row * 100 + col * 10 + block + 1) << 16);
}
int64_t ddr_read_spu_q16_row(uint32_t off, uint16_t * row_id) {
    assert(pending_result == returned.job_id);
    const unsigned row = (off - SPU_OUT_BASE) / 16U;
    assert(row < (unsigned) returned.rows);
    *row_id = (uint16_t) row;
    if (!step()) *row_id = 0xffffU;
    if (row == 0 && active != returned.bank && active != -1) ++overlap_reads;
    const size_t index = (size_t) returned.col * (size_t) test_rows + (size_t) returned.row0 + row;
    assert(reads[index] == returned.k_block0 && stores[index] == 0);
    reads[index] += returned.group_blocks;
    const int64_t value = raw_result(returned.row0 + row, returned.col, returned.k_block0);
    if (row + 1U == (unsigned) returned.rows) {
        pending_result = -1;
        ++consumed_jobs;
        consumed_order.push_back(key(returned));
    }
    return value;
}
'''

# The production residency arithmetic is useful in the legacy cache/tail cases.
harness += residency_align + "\n" + residency_end

# Keep the existing old scheduler checks: these snippets are the reviewed
# production guards around weight DMA, while all register/DDR operations remain
# RAM-only stubs.
weight_transfer = SOURCE[
    SOURCE.index("    if (!job.weight_bank_reused &&", SOURCE.index("static bool fpga_submit_q8_tile_job(")) :
    SOURCE.index("\n    const long long event_dma_weight_done", SOURCE.index("static bool fpga_submit_q8_tile_job("))
]
harness += r'''
bool weight_transfer(fpga_tile_job_t & job) {
    const fpga_dma_poll_hook_t * poll_hook = nullptr;
''' + weight_transfer + r'''
    return true;
}
bool transfer(fpga_tile_job_t & job) {
    assert(active != job.bank);
    ++activations;
    const int before = loads;
    assert(weight_transfer(job));
    if (job.weight_bank_reused) {
        assert(loads == before && valid[job.bank] && banks[job.bank] == key(job));
    } else {
        assert(loads == before + 1);
        banks[job.bank] = key(job);
        valid[job.bank] = true;
    }
    return true;
}
bool fpga_prepare_q8_tile_job(fpga_tile_job_t & job, const ggml_tensor * src0, const void * weight_data_base,
                              const block_q8_0_t * act_group, int64_t row0, int rows, int64_t ib0, int groups,
                              int64_t col, uint32_t, const fpga_weight_cache_entry_t *, uint32_t id, int bank,
                              fpga_stage_totals_t *, fpga_prompt_weight_staging_t *, uint32_t weight_stage_off,
                              uint32_t scale_stage_off) {
    if (!step()) return false;
    assert(bank != active);
    if (row0 > 0 && ib0 == 0 && col == 0) {
        assert((active != -1) == expect_cross(rows));
        cross_prepares += active != -1;
    }
    if (active != -1) assert(job.result_consumed || job.job_id == 0);
    job = {};
    job.bank = bank;
    job.job_id = id + 1U;
    job.tile_id = id;
    job.row0 = row0;
    job.rows = rows;
    job.k_block0 = ib0;
    job.group_blocks = groups;
    job.group_beats = groups * VPU_BLOCK_BEATS;
    job.col = col;
    job.act_group = act_group;
    job.src0 = src0;
    job.weight_src_off = weight_stage_off;
    job.scale_src_off = scale_stage_off;
    job.act_bytes = (size_t) job.group_beats * 16U;
    job.weight_bytes = g_p2_pack_dma_pipeline_enabled ?
        ((size_t) (rows + 1) / 2U) * (size_t) job.group_beats * 8U * sizeof(uint32_t) : 128U;
    job.scale_bytes = g_p2_pack_dma_pipeline_enabled ?
        (((size_t) rows * (size_t) groups + 3U) / 4U) * 16U : 128U;
    job.spu_result_bytes = (size_t) rows * 16U;
    job.weight_cache = nullptr;
    job.p2_residency_hit = pipeline_resident;
    job.p2_residency_slot = pipeline_resident ? 0U : P2_WEIGHT_RESIDENCY_NO_SLOT;
    job.p2_residency_epoch = g_p2_weight_residency_epoch;
    prepared_order.push_back(key(job));
    (void) weight_data_base;
    return true;
}
bool fpga_wait_and_drain_q8_tile_job(fpga_tile_job_t & job, fpga_stage_totals_t *, const char *, int,
                                     int64_t, int64_t, int64_t, int) {
    if (!step()) return false;
    assert(active == job.bank);
    assert(pending_result == -1);
    pending_result = (int) job.job_id;
    returned = job;
    returned.output_dma_ready = true;
    job.output_dma_ready = true;
    job.event_launch_us = 1;
    job.event_spu_finality_us = 2;
    job.event_retire_us = 3;
    return true;
}
bool fpga_write_post_spu_descriptor(const fpga_tile_job_t & job, int, int, unsigned, const char *, bool, bool force) {
    if (!step()) return false;
    assert(active == job.bank);
    assert(job.output_dma_ready);
    if (!job.result_consumed) assert(force);
    ++release_writes;
    active = -1;
    return true;
}
bool fpga_submit_q8_tile_job(fpga_tile_job_t & job, fpga_stage_totals_t *, const char *, int,
                             int64_t, int64_t, int64_t, int, const fpga_dma_poll_hook_t * poll_hook) {
    if (!step()) return false;
    assert(active == -1);
    assert(transfer(job));
    if (poll_hook && poll_hook->fn) assert(poll_hook->fn(poll_hook->context, 0));
    assert(valid[job.bank] && banks[job.bank] == key(job));
    ++scales;
    ++launches;
    reused += job.weight_bank_reused ? 1 : 0;
    active = job.bank;
    return true;
}
void fpga_log_prompt_weight_reuse(const char *, int64_t, long long jobs, uint64_t bytes) {
    logged_jobs = jobs;
    logged_bytes = bytes;
}
'''

harness += release + "\n" + accumulate + "\n" + scheduler
harness += r'''

bool run_legacy(int columns, bool overlap, int failure, int rows = 7, int blocks = 5, int cache = 0) {
    active = -1; valid[0] = valid[1] = false;
    loads = activations = scales = launches = reused = 0;
    operations = 0; fail_at = failure; logged_jobs = -1; logged_bytes = 0;
    pending_result = -1; consumed_jobs = overlap_reads = release_writes = 0;
    cross_prepares = expected_crossings = 0;
    prepared_order.clear(); consumed_order.clear();
    pipeline_admitted_k.clear(); pipeline_consumed_k.clear(); pipeline_activation_k.clear();
    pipeline_admitted_rows.clear(); pipeline_consumed_rows.clear(); pipeline_finalized = 0;
    pipeline_resident = false;
    test_columns = columns; test_rows = rows; test_blocks = blocks;
    g_vpu_max_rows = configured_max_rows; max_blocks = configured_max_blocks;
    g_packed_q8_max_blocks = configured_max_blocks;
    g_packed_q8_result_words = std::max(1, (g_vpu_max_rows * g_packed_q8_max_blocks + VPU_RESULT_PACK_LANES - 1) /
                                              VPU_RESULT_PACK_LANES);
    g_p2_pack_dma_pipeline_enabled = false;
    g_p2_result_overlap_enabled = overlap;
    g_vpu_descriptor_supported = configured_descriptor;
    g_p2_weight_residency_enabled = cache != 0;
    g_p2_weight_residency_budget_mb = 32;
    g_p2_residency_next_slot = cache == 3 ? 128 : 0;
    size_t remaining = 1024U * 1024U;
    const int first_rows = std::min(rows, g_vpu_max_rows);
    if (cache == 2) remaining = 0;
    if (cache == 4) remaining = allocation_bytes(rows % g_vpu_max_rows ? rows % g_vpu_max_rows : first_rows,
                                                   std::min(max_blocks, blocks));
    if (cache == 5 || cache == 6) remaining = allocation_bytes(first_rows, std::min(max_blocks, blocks)) - (cache == 6);
    g_p2_residency_next_off = P2_WEIGHT_RESIDENCY_END - (uint32_t) remaining;
    expected.assign((size_t) rows * (size_t) columns, 0.0f);
    stored.assign((size_t) rows * (size_t) columns, std::numeric_limits<float>::quiet_NaN());
    reads.assign((size_t) rows * (size_t) columns, 0);
    stores.assign((size_t) rows * (size_t) columns, 0);
    for (int col = 0; col < columns; ++col) for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; block += max_blocks)
            expected[(size_t) col * (size_t) rows + (size_t) row] += float(raw_result(row, col, block)) / 65536.0f;
    for (int row = g_vpu_max_rows; row < rows; row += g_vpu_max_rows)
        expected_crossings += expect_cross(std::min(g_vpu_max_rows, rows - row));
    ggml_tensor tensor = {};
    std::vector<block_q8_0_t> acts((size_t) columns * (size_t) blocks);
    std::vector<float> values, scales_local;
    fpga_stage_totals_t totals = {};
    const bool ok = fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
        &tensor, &tensor, acts, scales_local, nullptr, &totals, "test", 0, blocks * 32, rows, columns, blocks, values);
    if (ok) {
        const int row_groups = (rows + g_vpu_max_rows - 1) / g_vpu_max_rows;
        const int weight_tiles = row_groups * ((blocks + max_blocks - 1) / max_blocks);
        assert(active == -1 && pending_result == -1 && consumed_jobs == launches && release_writes == launches);
        assert(cross_prepares == expected_crossings && prepared_order == consumed_order);
        for (int count : stores) assert(count == 1);
        assert(overlap_reads == (overlap && g_vpu_descriptor_supported ? launches - row_groups + expected_crossings : 0));
        assert(totals.result_overlap_jobs == overlap_reads);
        assert(loads == weight_tiles * std::min(columns, 2));
        assert(launches == weight_tiles * columns && scales == launches && activations == launches);
        assert(reused == weight_tiles * std::max(columns - 2, 0));
        if (columns > 1) {
            assert(logged_jobs == reused && logged_bytes == (unsigned long long) reused * 128U);
        } else assert(logged_jobs == -1);
    } else {
        assert(logged_jobs == -1);
    }
    return ok;
}

bool run_pipeline_case(int rows, int blocks, bool enabled, bool resident) {
    active = -1; valid[0] = valid[1] = false;
    loads = activations = scales = launches = reused = 0;
    operations = 0; fail_at = 0; pending_result = -1; consumed_jobs = 0;
    overlap_reads = release_writes = 0; prepared_order.clear(); consumed_order.clear();
    pipeline_admitted_k.clear(); pipeline_consumed_k.clear(); pipeline_activation_k.clear();
    pipeline_admitted_rows.clear(); pipeline_consumed_rows.clear(); pipeline_finalized = 0;
    pipeline_resident = resident;
    test_columns = 1; test_rows = rows; test_blocks = blocks;
    g_vpu_max_rows = 256; max_blocks = 64;
    g_p2_pack_dma_pipeline_enabled = enabled;
    g_p2_result_overlap_enabled = true;
    g_vpu_descriptor_supported = true;
    g_p2_weight_residency_enabled = resident;
    g_p2_residency_next_slot = 0;
    g_p2_residency_next_off = WEIGHT_CACHE_BASE;
    g_p2_weight_residency_budget_mb = 32;
    g_p2_pack_dma_pipeline_eligible_jobs = 0;
    g_p2_pack_dma_pipeline_consumed_jobs = 0;
    expected.assign((size_t) rows, 0.0f);
    stored.assign((size_t) rows, std::numeric_limits<float>::quiet_NaN());
    reads.assign((size_t) rows, 0);
    stores.assign((size_t) rows, 0);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; block += max_blocks)
            expected[(size_t) row] += float(raw_result(row, 0, block)) / 65536.0f;
    ggml_tensor tensor = {};
    std::vector<block_q8_0_t> acts((size_t) blocks);
    expected_pipeline_act_base = acts.data();
    std::vector<float> values, act_scales;
    fpga_stage_totals_t totals = {};
    const bool ok = fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
        &tensor, &tensor, acts, act_scales, nullptr, &totals, "pipeline", 0, blocks * 32, rows, 1, blocks, values);
    assert(ok);
    assert(active == -1 && pending_result == -1 && consumed_jobs == launches && release_writes == launches);
    for (int count : stores) assert(count == 1);
    if (!enabled || resident) {
        assert(g_p2_pack_dma_pipeline_eligible_jobs == 0);
        assert(g_p2_pack_dma_pipeline_consumed_jobs == 0);
    } else {
        assert(g_p2_pack_dma_pipeline_eligible_jobs == (long long) pipeline_admitted_k.size());
        assert(g_p2_pack_dma_pipeline_consumed_jobs == (long long) pipeline_consumed_k.size());
        assert(pipeline_finalized == (int) pipeline_admitted_k.size());
    }
    return true;
}

int main() {
    fpga_tile_job_t job = {};
    dma_success = false;
    assert(!weight_transfer(job));
    job.weight_bank_reused = true;
    assert(weight_transfer(job));
    job.weight_bank_reused = false;
    assert(!weight_transfer(job));
    dma_success = true;
    int cases = 0;
    configured_max_rows = 3;
    configured_max_blocks = 2;
    for (bool descriptors : {false, true}) {
        configured_descriptor = descriptors;
        for (bool overlap : {false, true}) for (int columns : {1, 2, 3, 4, 5, 46})
        for (int blocks : {1, 2, 5})
        for (int cache : {0, 1, 2, 3, 4, 5, 6}) {
            assert(run_legacy(columns, overlap, 0, 7, blocks, cache)); ++cases;
            const int count = (int) operations;
            for (int fail = 1; fail <= count; ++fail) {
                assert(!run_legacy(columns, overlap, fail, 7, blocks, cache)); ++cases;
            }
            assert(run_legacy(columns, overlap, 0, 7, blocks, cache)); ++cases;
        }
    }

    // Retain the production Gemma geometry check from the original test:
    // 85 jobs/layer and 78 cross-row opportunities with descriptor overlap.
    configured_max_rows = 256;
    configured_max_blocks = 64;
    configured_descriptor = true;
    for (bool overlap : {false, true}) {
        int jobs = 0, handoffs = 0;
        for (const auto & shape : {std::pair<int, int>{1024, 36}, {256, 36}, {256, 36}, {1152, 32},
                                  {6912, 36}, {6912, 36}, {1152, 216}}) {
            assert(run_legacy(1, overlap, 0, shape.first, shape.second, 2));
            jobs += launches;
            handoffs += overlap_reads;
        }
        assert(jobs * 26 == 2210 && handoffs * 26 == (overlap ? 2028 : 0));
        ++cases;
    }

    // Boundary sweep for full rows, odd rows, small tails, and residency
    // capacity decisions.  Pipeline is disabled here so this remains the
    // original scheduler/cache regression.
    for (int rows : {1, 255, 256, 257, 512, 513, 769})
        for (int blocks : {1, 36, 64, 65, 216})
            for (int cache : {0, 1, 2, 3, 4, 5, 6}) {
                assert(run_legacy(1, true, 0, rows, blocks, cache));
                ++cases;
            }

    // Production scheduler cases with the opt-in P2 pipeline enabled.  These
    // assertions observe real scheduler admission/consume calls, not a tile
    // model or a copied loop.
    assert(run_pipeline_case(512, 1152 / 32, true, false));
    assert(!pipeline_consumed_rows.empty());
    assert(std::find(pipeline_consumed_rows.begin(), pipeline_consumed_rows.end(), 256) != pipeline_consumed_rows.end());
    assert(run_pipeline_case(256, 6912 / 32, true, false));
    assert(std::find(pipeline_consumed_k.begin(), pipeline_consumed_k.end(), 64) != pipeline_consumed_k.end());
    assert(std::find(pipeline_consumed_k.begin(), pipeline_consumed_k.end(), 128) != pipeline_consumed_k.end());
    assert(pipeline_activation_k == pipeline_consumed_k);
    assert(run_pipeline_case(256, 1152 / 32, false, false));
    assert(run_pipeline_case(256, 1152 / 32, true, true));
    assert(run_pipeline_case(257, 6912 / 32, true, false));
    for (size_t i = 0; i < pipeline_admitted_rows.size(); ++i) assert(pipeline_admitted_rows[i] < 256);

    std::printf("PASS: %d legacy scheduler cases; production pipeline rows/K transitions, disabled/resident/small-tail admission\n", cases);
}
'''

with tempfile.TemporaryDirectory(prefix="fpga-bank-reuse-") as directory:
    cpp = Path(directory) / "test.cpp"
    exe = Path(directory) / "test"
    cpp.write_text(harness)
    subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-Wall", "-Wextra",
        "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I" + str(ROOT / "ggml/src/ggml-cpu"),
        str(cpp), str(ROOT / "ggml/src/ggml-cpu/fpga_q8_layout.cpp"), "-o", str(exe)
    ], check=True, timeout=90)
    subprocess.run([str(exe)], check=True, timeout=90)
