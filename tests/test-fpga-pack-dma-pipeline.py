#!/usr/bin/env python3
"""RAM-only checks for the production P2 pack/DMA pipeline."""

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "ggml/src/ggml-cpu"
SOURCE = (CPU / "fpga_host.cpp").read_text()


def extract_definition(signature: str) -> str:
    """Extract one balanced definition, skipping a preceding forward declaration."""
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


PACK_BATCH = extract_definition("static bool fpga_p2_dma_pipeline_pack_batch(")
POLL = extract_definition("static bool fpga_p2_dma_pipeline_poll(")
INVALIDATE = extract_definition("static void fpga_p2_dma_pipeline_invalidate(")
SLOT_RANGES = extract_definition("static bool fpga_p2_dma_pipeline_slot_ranges(")
CONSUME = extract_definition("static bool fpga_p2_dma_pipeline_consume(")
FINALIZE = extract_definition("static bool fpga_p2_dma_pipeline_finalize(")
WAIT_DISABLED = extract_definition("static bool zdma_wait_channel_disabled(")
WAIT_COMPLETE = extract_definition("static bool zdma_wait_transfer_complete(")
SCHEDULER = extract_definition("static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(")

# Supplementary source guards: the C++ harness below executes the extracted
# production bodies, while these checks catch accidental extraction drift.
assert "fpga_p2_dma_pipeline_consume(lookahead, prepared" in SCHEDULER
assert "fpga_p2_dma_pipeline_finalize(lookahead)" in SCHEDULER
assert SCHEDULER.index("fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals, false)") < SCHEDULER.index(
    "fpga_p2_dma_pipeline_finalize(lookahead)"
)
assert "dma_pipeline_host_dma_includes_callback=1" in SOURCE
assert "const size_t scale_capacity_entries = lookahead.job.scale_bytes / sizeof(uint32_t);" in SOURCE


harness = r'''
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "ggml.h"
#include "quants.h"
#include "fpga_q8_pack.h"

static constexpr int VPU_BLOCK_BEATS = 2;
static constexpr int VPU_RESULT_PACK_LANES = 4;
static constexpr size_t P2_PIPELINE_PACK_BATCH_PAIRS = 8;
static constexpr uint32_t ACT_BASE = 0x00010000U, ACT_END = 0x00020000U;
static constexpr uint32_t WEIGHT_BASE = 0x00100000U, WEIGHT_END = 0x00200000U;
static constexpr uint32_t RESULT_BASE = 0x00200000U, RESULT_END = 0x00210000U;
static constexpr uint32_t SPU_OUT_BASE = 0x00340000U, SPU_OUT_END = 0x00380000U;
static constexpr uint32_t SPU_PARAM_BASE = 0x00380000U, SPU_PARAM_END = 0x003C0000U;
static constexpr uint32_t P2_PIPELINE_WEIGHT_SLOT_BYTES = 512U * 1024U;
static constexpr uint32_t P2_PIPELINE_SCALE_SLOT_BYTES = 64U * 1024U;
static constexpr uint32_t P2_PIPELINE_WEIGHT_SLOT_COUNT = 2U;
static constexpr uint32_t P2_PIPELINE_SCALE_SLOT_COUNT = 2U;
static constexpr size_t P2_PIPELINE_PL_SCALE_BYTES = 64U * 1024U;
static constexpr uint32_t ZDMA_STATUS_STATE_MASK = 0x00000003U;
static constexpr uint32_t ZDMA_CTRL2_EN = 0x00000001U;
static constexpr uint32_t ZDMA_ISR_DMA_DONE = 0x00000400U;
static constexpr uint32_t ZDMA_ISR_ERROR_MASK = 0x00000BF9U;

struct fpga_tile_job_t {
    int bank = 0;
    uint32_t job_id = 0, tile_id = 0;
    int rows = 0, group_blocks = 0, group_beats = 0;
    int64_t row0 = 0, k_block0 = 0, col = 0;
    const block_q8_0 * act_group = nullptr;
    const struct ggml_tensor * src0 = nullptr;
    size_t act_bytes = 0, weight_bytes = 0, scale_bytes = 0;
    size_t spu_result_bytes = 0;
    uint32_t result_values = 0, result_words = 0, scale_words = 0;
    uint32_t weight_src_off = WEIGHT_BASE, scale_src_off = SPU_PARAM_BASE;
    long long event_prep_begin_us = 0;
};

struct fpga_stage_totals_t {
    long long prep_us = 0;
    long long prep_direct_weight_pack_us = 0;
};

typedef struct {
    bool                  valid = false;
    bool                  worker_submitted = false;
    bool                  consumed = false;
    uint64_t              generation = 0;
    uint64_t              source_epoch = 0;
    size_t                pair_count = 0;
    size_t                helper_pair_begin = 0;
    size_t                main_pair_next = 0;
    size_t                words_per_pair = 0;
    size_t                expected_words = 0;
    size_t                expected_scale_entries = 0;
    size_t                callback_bytes = 0;
    size_t                main_written_words = 0;
    size_t                main_scale_entries = 0;
    long long             helper_service_us = 0;
    long long             residual_us = 0;
    long long             callback_us = 0;
    fpga_stage_totals_t * totals = nullptr;
    const void *          weight_data_base = nullptr;
    volatile uint32_t *   weight_words = nullptr;
    volatile uint32_t *   scale_words = nullptr;
    fpga_tile_job_t       job = {};
} fpga_p2_dma_pipeline_lookahead_t;

typedef bool (*fpga_dma_poll_hook_fn)(void *, long long);
typedef struct { fpga_dma_poll_hook_fn fn = nullptr; void * context = nullptr; } fpga_dma_poll_hook_t;
typedef struct {
    uint32_t status = 0;
    uint32_t isr = 0;
    uint32_t ctrl2 = 0;
    long long polls = 0;
    bool saw_enabled = false;
} zdma_completion_info_t;

struct dma_ctrl {
    uint32_t ZDMA_ERR_CTRL;
    uint32_t dmy0[63];
    uint32_t ZDMA_CH_ISR;
    uint32_t ZDMA_CH_IMR;
    uint32_t ZDMA_CH_IEN;
    uint32_t ZDMA_CH_IDS;
    uint32_t ZDMA_CH_CTRL0;
    uint32_t ZDMA_CH_CTRL1;
    uint32_t ZDMA_CH_FCI;
    uint32_t ZDMA_CH_STATUS;
    uint32_t ZDMA_CH_DATA_ATTR;
    uint32_t ZDMA_CH_DSCR_ATTR;
    uint32_t ZDMA_CH_SRC_DSCR_WORD0;
    uint32_t ZDMA_CH_SRC_DSCR_WORD1;
    uint32_t ZDMA_CH_SRC_DSCR_WORD2;
    uint32_t ZDMA_CH_SRC_DSCR_WORD3;
    uint32_t ZDMA_CH_DST_DSCR_WORD0;
    uint32_t ZDMA_CH_DST_DSCR_WORD1;
    uint32_t ZDMA_CH_DST_DSCR_WORD2;
    uint32_t ZDMA_CH_DST_DSCR_WORD3;
    uint32_t ZDMA_CH_WR_ONLY_WORD0;
    uint32_t ZDMA_CH_WR_ONLY_WORD1;
    uint32_t ZDMA_CH_WR_ONLY_WORD2;
    uint32_t ZDMA_CH_WR_ONLY_WORD3;
    uint32_t ZDMA_CH_SRC_START_LSB;
    uint32_t ZDMA_CH_SRC_START_MSB;
    uint32_t ZDMA_CH_DST_START_LSB;
    uint32_t ZDMA_CH_DST_START_MSB;
    uint32_t ZDMA_CH_SRC_CUR_PYLD_LSB;
    uint32_t ZDMA_CH_SRC_CUR_PYLD_MSB;
    uint32_t ZDMA_CH_DST_CUR_PYLD_LSB;
    uint32_t ZDMA_CH_DST_CUR_PYLD_MSB;
    uint32_t ZDMA_CH_SRC_CUR_DSCR_LSB;
    uint32_t ZDMA_CH_SRC_CUR_DSCR_MSB;
    uint32_t ZDMA_CH_DST_CUR_DSCR_LSB;
    uint32_t ZDMA_CH_DST_CUR_DSCR_MSB;
    uint32_t ZDMA_CH_TOTAL_BYTE;
    uint32_t ZDMA_CH_RATE_CTRL;
    uint32_t ZDMA_CH_IRQ_SRC_ACCT;
    uint32_t ZDMA_CH_IRQ_DST_ACCT;
    uint32_t dmy2[26];
    uint32_t ZDMA_CH_CTRL2;
};

static dma_ctrl g_regs = {};
static volatile dma_ctrl * g_dma = &g_regs;
static void * g_dma_map_base = &g_regs;
static long long g_dma_timeout_us = 100;
static long long g_clock = 0;
static int g_fatal_count = 0, g_log_errors = 0, g_dump_count = 0, g_fences = 0;
static int g_helper_waits = 0;
static bool g_worker_ok = true;
static uint64_t g_p2_weight_residency_epoch = 9;
static long long g_p2_pack_dma_pipeline_residual_bytes = 0;
static long long g_p2_pack_dma_pipeline_helper_us = 0;
static long long g_p2_pack_dma_pipeline_residual_us = 0;
static uint64_t g_p2_pack_dma_pipeline_callback_bytes = 0;
static long long g_p2_pack_dma_pipeline_callback_us = 0;
static long long g_p2_pack_dma_pipeline_consumed_jobs = 0;

static bool dma_is_mapped() { return g_dma != nullptr && g_dma_map_base != nullptr; }
static long long now_us() { return ++g_clock; }
static void mmio_fence() { ++g_fences; }
static int sched_yield() { return 0; }
static void fpga_fatal(const char *, ...) { ++g_fatal_count; }
static void zdma_dump(const char *) { ++g_dump_count; }
static void zdma_format_error_mask(uint32_t, char * out, size_t out_size) {
    if (out && out_size) out[0] = 0;
}
static void test_log_error(const char *, ...) { ++g_log_errors; }
#define LOGE(...) do { test_log_error(__VA_ARGS__); } while (0)

static bool range_fits(uint32_t off, size_t bytes, uint32_t begin, uint32_t end) {
    return off >= begin && off <= end && bytes <= (size_t) end - off;
}
'''

harness += SLOT_RANGES
harness += r'''

static fpga_p2_dma_pipeline_lookahead_t * g_worker_context = nullptr;
static bool g_worker_pack_suffix = true;

static bool fpga_p2_pack_worker_wait(uint64_t generation, size_t * words, long long * service_us,
                                     long long * wait_us, size_t * scale_entries = nullptr) {
    assert(generation != 0);
    ++g_helper_waits;
    *words = 0;
    *service_us = 7;
    *wait_us = 2;
    if (scale_entries) *scale_entries = 0;
    if (!g_worker_ok) return false;
    if (g_worker_context && g_worker_pack_suffix) {
        auto & lookahead = *g_worker_context;
        size_t written_words = 0, written_scales = 0;
        if (!fpga_pack_direct_weight_scale_pair_range(
                lookahead.weight_words, lookahead.scale_words, lookahead.job.src0, lookahead.weight_data_base,
                lookahead.job.act_group, lookahead.job.row0, lookahead.job.k_block0, lookahead.job.rows,
                lookahead.job.group_blocks, lookahead.job.group_beats, lookahead.helper_pair_begin,
                lookahead.pair_count, true, &written_words, &written_scales)) return false;
        *words = written_words;
        if (scale_entries) *scale_entries = written_scales;
    }
    return true;
}
'''
harness += INVALIDATE + "\n" + PACK_BATCH + "\n" + POLL + "\n" + CONSUME + "\n" + FINALIZE + "\n" + WAIT_DISABLED + "\n" + WAIT_COMPLETE
harness += r'''

static void fill_source(std::vector<block_q8_0> & weights, std::vector<block_q8_0> & activations, unsigned seed) {
    // Keep the harness independent of ggml's conversion library: these are
    // the exact half-precision encodings for the deterministic test scales.
    static constexpr uint16_t weight_scales[] = {
        0x3400, 0x3d00, 0x4080, 0x4280, 0x4440, 0x4540, 0x4640, 0x4740,
    };
    static constexpr uint16_t activation_scales[] = { 0x3800, 0x3e00, 0x4100, 0x4300 };
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i].d = weight_scales[(seed + i) & 7U];
        for (int lane = 0; lane < QK8_0; ++lane)
            weights[i].qs[lane] = static_cast<int8_t>((seed + i * 13U + (size_t) lane * 7U) & 0xFFU);
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i].d = activation_scales[(seed + i) & 3U];
        for (int lane = 0; lane < QK8_0; ++lane)
            activations[i].qs[lane] = static_cast<int8_t>((seed + i * 17U + (size_t) lane * 3U) & 0xFFU);
    }
}

struct PackedTile {
    size_t weight_words;
    size_t scale_entries;
    std::vector<uint64_t> weight_storage;
    std::vector<uint64_t> scale_storage;
    volatile uint32_t * weight = nullptr;
    volatile uint32_t * scale = nullptr;

    void bind() {
        weight = reinterpret_cast<volatile uint32_t *>(weight_storage.data() + 2U);
        scale = reinterpret_cast<volatile uint32_t *>(scale_storage.data() + 2U);
    }
};

static PackedTile make_tile(size_t weight_words, size_t scale_entries) {
    PackedTile tile = {
        weight_words,
        scale_entries,
        std::vector<uint64_t>(weight_words / 2U + 8U, UINT64_C(0xabababababababab)),
        std::vector<uint64_t>((scale_entries + 3U) / 2U + 8U, UINT64_C(0xabababababababab)),
        nullptr,
        nullptr,
    };
    tile.bind();
    return tile;
}

static void assert_canaries(const PackedTile & tile, size_t scale_words) {
    const uint64_t canary = UINT64_C(0xabababababababab);
    assert(tile.weight_storage[0] == canary && tile.weight_storage[1] == canary && tile.weight_storage.back() == canary);
    assert(tile.scale_storage[0] == canary && tile.scale_storage[1] == canary && tile.scale_storage.back() == canary);
    for (size_t i = tile.scale_entries; i < scale_words * 4U; ++i)
        assert(tile.scale[i] == UINT32_C(0xabababab));
}

static fpga_p2_dma_pipeline_lookahead_t prepare_lookahead(
        ggml_tensor & tensor, const std::vector<block_q8_0> & weights, const std::vector<block_q8_0> & activations,
        PackedTile & tile, int rows, int group_blocks, size_t helper_pair_begin, fpga_stage_totals_t & totals) {
    const size_t pair_count = ((size_t) rows + 1U) / 2U;
    const size_t words_per_pair = (size_t) (group_blocks * VPU_BLOCK_BEATS) * 8U;
    const size_t entries = (size_t) rows * (size_t) group_blocks;
    fpga_p2_dma_pipeline_lookahead_t lookahead = {};
    lookahead.valid = true;
    lookahead.worker_submitted = true;
    lookahead.generation = 1;
    lookahead.source_epoch = g_p2_weight_residency_epoch;
    lookahead.pair_count = pair_count;
    lookahead.helper_pair_begin = helper_pair_begin;
    lookahead.main_pair_next = 0;
    lookahead.words_per_pair = words_per_pair;
    lookahead.expected_words = pair_count * words_per_pair;
    lookahead.expected_scale_entries = entries;
    lookahead.totals = &totals;
    lookahead.weight_data_base = weights.data();
    lookahead.weight_words = tile.weight;
    lookahead.scale_words = tile.scale;
    lookahead.job = {};
    lookahead.job.bank = 1;
    lookahead.job.tile_id = 7;
    lookahead.job.rows = rows;
    lookahead.job.group_blocks = group_blocks;
    lookahead.job.group_beats = group_blocks * VPU_BLOCK_BEATS;
    lookahead.job.row0 = 1;
    lookahead.job.k_block0 = 1;
    lookahead.job.col = 2;
    lookahead.job.act_group = activations.data();
    lookahead.job.src0 = &tensor;
    lookahead.job.weight_bytes = pair_count * words_per_pair * sizeof(uint32_t);
    lookahead.job.scale_bytes = ((entries + 3U) / 4U) * 16U;
    lookahead.job.weight_src_off = WEIGHT_BASE + P2_PIPELINE_WEIGHT_SLOT_BYTES;
    lookahead.job.scale_src_off = SPU_PARAM_BASE + P2_PIPELINE_SCALE_SLOT_BYTES;
    lookahead.job.event_prep_begin_us = 3;
    return lookahead;
}

static void pack_all_in_bounded_batches(int rows, int group_blocks, unsigned seed, PackedTile & result) {
    const size_t pairs = ((size_t) rows + 1U) / 2U;
    const size_t words_per_pair = (size_t) (group_blocks * VPU_BLOCK_BEATS) * 8U;
    const size_t entries = (size_t) rows * (size_t) group_blocks;
    std::vector<block_q8_0> weights(((size_t) rows + 2U) * (size_t) group_blocks);
    std::vector<block_q8_0> activations((size_t) group_blocks);
    fill_source(weights, activations, seed);
    ggml_tensor tensor = {};
    tensor.data = weights.data();
    tensor.nb[1] = (size_t) group_blocks * sizeof(block_q8_0);
    size_t written_words = 0, written_scales = 0;
    assert(fpga_pack_direct_weight_scale_pair_range(
        result.weight, result.scale, &tensor, weights.data(), activations.data(), 1, 1, rows, group_blocks,
        group_blocks * VPU_BLOCK_BEATS, 0, pairs, true, &written_words, &written_scales));
    assert(written_words == pairs * words_per_pair && written_scales == entries);
}

static void test_two_slots_and_tails() {
    for (const auto & shape : {std::pair<int, int>{3, 3}, {5, 4}, {257, 36}}) {
        const int rows = shape.first;
        const int group_blocks = shape.second;
        const size_t pairs = ((size_t) rows + 1U) / 2U;
        const size_t words_per_pair = (size_t) (group_blocks * VPU_BLOCK_BEATS) * 8U;
        const size_t entries = (size_t) rows * (size_t) group_blocks;
        PackedTile slot0 = make_tile(pairs * words_per_pair, entries);
        PackedTile slot1 = make_tile(pairs * words_per_pair, entries);
        pack_all_in_bounded_batches(rows, group_blocks, 11U, slot0);
        const std::vector<uint64_t> slot0_after = slot0.weight_storage;
        const std::vector<uint64_t> slot0_scale_after = slot0.scale_storage;
        pack_all_in_bounded_batches(rows, group_blocks, 97U, slot1);
        assert_canaries(slot0, (entries + 3U) / 4U);
        assert_canaries(slot1, (entries + 3U) / 4U);
        assert(slot0.weight_storage == slot0_after && slot0.scale_storage == slot0_scale_after);
        assert(slot1.weight_storage != slot0_after || slot1.scale_storage != slot0_scale_after);
    }
}

static void test_real_poll_and_cleanup() {
    constexpr int rows = 33, group_blocks = 4;
    const size_t pairs = ((size_t) rows + 1U) / 2U;
    const size_t words_per_pair = (size_t) (group_blocks * VPU_BLOCK_BEATS) * 8U;
    const size_t entries = (size_t) rows * (size_t) group_blocks;
    std::vector<block_q8_0> weights(((size_t) rows + 2U) * (size_t) group_blocks);
    std::vector<block_q8_0> activations((size_t) group_blocks);
    fill_source(weights, activations, 23U);
    ggml_tensor tensor = {};
    tensor.data = weights.data();
    tensor.nb[1] = (size_t) group_blocks * sizeof(block_q8_0);
    PackedTile tile = make_tile(pairs * words_per_pair, entries);
    fpga_stage_totals_t totals;
    auto lookahead = prepare_lookahead(tensor, weights, activations, tile, rows, group_blocks, pairs / 2U, totals);
    lookahead.worker_submitted = false;
    lookahead.generation = 7;
    g_p2_pack_dma_pipeline_callback_bytes = 0;
    g_p2_pack_dma_pipeline_callback_us = 0;
    while (lookahead.main_pair_next < lookahead.helper_pair_begin)
        assert(fpga_p2_dma_pipeline_poll(&lookahead, 1));
    assert(lookahead.main_pair_next == lookahead.helper_pair_begin);
    assert(lookahead.callback_bytes > 0 && g_p2_pack_dma_pipeline_callback_bytes == lookahead.callback_bytes);

    lookahead.valid = true;
    lookahead.main_pair_next = 0;
    lookahead.main_written_words = 0;
    lookahead.main_scale_entries = 0;
    lookahead.scale_words = nullptr;
    assert(!fpga_p2_dma_pipeline_poll(&lookahead, 2));
    assert(!lookahead.valid);

    lookahead = prepare_lookahead(tensor, weights, activations, tile, rows, group_blocks, pairs / 2U, totals);
    g_worker_context = &lookahead;
    const int waits_before = g_helper_waits;
    fpga_p2_dma_pipeline_invalidate(&lookahead);
    assert(g_helper_waits == waits_before + 1 && !lookahead.valid && !lookahead.worker_submitted);
    fpga_p2_dma_pipeline_invalidate(&lookahead);
    assert(g_helper_waits == waits_before + 1);

    lookahead = prepare_lookahead(tensor, weights, activations, tile, rows, group_blocks, pairs / 2U, totals);
    g_worker_context = &lookahead;
    g_worker_ok = false;
    fpga_p2_dma_pipeline_invalidate(&lookahead);
    assert(g_fatal_count == 1 && !lookahead.worker_submitted);
    g_worker_ok = true;
    g_worker_context = nullptr;
}

static void test_real_finalize_and_consume() {
    constexpr int rows = 3, group_blocks = 3;
    const size_t pairs = ((size_t) rows + 1U) / 2U;
    const size_t words_per_pair = (size_t) (group_blocks * VPU_BLOCK_BEATS) * 8U;
    const size_t entries = (size_t) rows * (size_t) group_blocks;
    const size_t capacity_entries = ((entries + 3U) / 4U) * 4U;
    std::vector<block_q8_0> weights(((size_t) rows + 2U) * (size_t) group_blocks);
    std::vector<block_q8_0> activations((size_t) group_blocks);
    fill_source(weights, activations, 77U);
    ggml_tensor tensor = {};
    tensor.data = weights.data();
    tensor.nb[1] = (size_t) group_blocks * sizeof(block_q8_0);
    PackedTile result = make_tile(pairs * words_per_pair, entries);
    PackedTile expected = make_tile(pairs * words_per_pair, entries);
    size_t expected_words = 0, expected_scales = 0;
    assert(fpga_pack_direct_weight_scale_pair_range(
        expected.weight, expected.scale, &tensor, weights.data(), activations.data(), 1, 1, rows, group_blocks,
        group_blocks * VPU_BLOCK_BEATS, 0, pairs, true, &expected_words, &expected_scales));
    assert(expected_words == pairs * words_per_pair && expected_scales == entries);
    fpga_stage_totals_t totals;
    auto lookahead = prepare_lookahead(tensor, weights, activations, result, rows, group_blocks, pairs / 2U, totals);
    lookahead.scale_words[entries] = UINT32_C(0xDEADBEEF);
    lookahead.scale_words[entries + 1U] = UINT32_C(0xDEADBEEF);
    lookahead.scale_words[entries + 2U] = UINT32_C(0xDEADBEEF);
    g_worker_context = &lookahead;
    g_worker_pack_suffix = true;
    g_p2_pack_dma_pipeline_residual_bytes = 0;
    assert(fpga_p2_dma_pipeline_finalize(lookahead));
    assert(!lookahead.worker_submitted && lookahead.valid);
    assert(lookahead.main_pair_next == lookahead.helper_pair_begin);
    assert(lookahead.main_written_words == lookahead.helper_pair_begin * words_per_pair);
    assert(lookahead.main_scale_entries == entries - group_blocks);
    for (size_t i = 0; i < expected_words; ++i) assert(result.weight[i] == expected.weight[i]);
    for (size_t i = 0; i < entries; ++i) assert(result.scale[i] == expected.scale[i]);
    for (size_t i = entries; i < capacity_entries; ++i) assert(result.scale[i] == 0U);
    assert(result.weight_storage[0] == UINT64_C(0xabababababababab) && result.weight_storage.back() == UINT64_C(0xabababababababab));
    assert(result.scale_storage[0] == UINT64_C(0xabababababababab) && result.scale_storage.back() == UINT64_C(0xabababababababab));
    assert(g_p2_pack_dma_pipeline_residual_bytes ==
           (long long) (lookahead.main_written_words * sizeof(uint32_t)));

    fpga_tile_job_t consumed = {};
    assert(fpga_p2_dma_pipeline_consume(lookahead, consumed, &totals, &tensor, weights.data(), 1, rows, 1,
                                         group_blocks, 2, 7));
    assert(!lookahead.valid && lookahead.consumed);
    assert(consumed.row0 == 1 && consumed.k_block0 == 1 && consumed.col == 2 && consumed.tile_id == 7);
    assert(consumed.weight_src_off == WEIGHT_BASE + P2_PIPELINE_WEIGHT_SLOT_BYTES);
    assert(consumed.scale_src_off == SPU_PARAM_BASE + P2_PIPELINE_SCALE_SLOT_BYTES);

    // Source identity and residency epoch are mandatory consume keys.
    auto bad = prepare_lookahead(tensor, weights, activations, result, rows, group_blocks, pairs / 2U, totals);
    bad.worker_submitted = false;
    bad.main_pair_next = bad.helper_pair_begin;
    bad.main_written_words = bad.helper_pair_begin * words_per_pair;
    bad.main_scale_entries = entries - group_blocks;
    ggml_tensor other = {};
    assert(!fpga_p2_dma_pipeline_consume(bad, consumed, &totals, &other, weights.data(), 1, rows, 1,
                                         group_blocks, 2, 7));
    assert(bad.valid);
    const uint64_t epoch = g_p2_weight_residency_epoch;
    g_p2_weight_residency_epoch = epoch + 1U;
    assert(!fpga_p2_dma_pipeline_consume(bad, consumed, &totals, &tensor, weights.data(), 1, rows, 1,
                                         group_blocks, 2, 7));
    g_p2_weight_residency_epoch = epoch;
}

static void reset_dma() {
    std::memset(&g_regs, 0, sizeof(g_regs));
    g_clock = 0;
    g_dma_timeout_us = 100;
    g_log_errors = g_dump_count = 0;
}
static int g_hook_calls = 0;
static bool complete_hook(void *, long long) {
    ++g_hook_calls;
    g_dma->ZDMA_CH_CTRL2 = 0;
    g_dma->ZDMA_CH_ISR = ZDMA_ISR_DMA_DONE;
    return true;
}
static bool failing_hook_quiesced(void *, long long) {
    ++g_hook_calls;
    g_dma->ZDMA_CH_CTRL2 = 0;
    return false;
}

static void test_zdma_wait_order() {
    zdma_completion_info_t info = {};
    reset_dma();
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_EN;
    fpga_dma_poll_hook_t hook{complete_hook, nullptr};
    g_hook_calls = 0;
    assert(zdma_wait_transfer_complete("done", &info, &hook, 1));
    assert(info.saw_enabled && g_hook_calls == 1 && info.isr == ZDMA_ISR_DMA_DONE && info.ctrl2 == 0);

    reset_dma();
    g_dma->ZDMA_CH_ISR = ZDMA_ISR_DMA_DONE;
    g_hook_calls = 0;
    assert(zdma_wait_transfer_complete("immediate", &info, &hook, 1));
    assert(g_hook_calls == 0 && !info.saw_enabled);

    reset_dma();
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_EN;
    g_dma->ZDMA_CH_ISR = 1U;
    g_hook_calls = 0;
    assert(!zdma_wait_transfer_complete("error", &info, &hook, 1));
    assert(g_hook_calls == 0 && g_log_errors == 1);

    reset_dma();
    g_dma_timeout_us = 2;
    // Anchor the fake clock beyond launch_us + timeout before entering the
    // completion loop so the hook cannot complete an already-expired wait.
    g_clock = 4;
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_EN;
    g_hook_calls = 0;
    assert(!zdma_wait_transfer_complete("timeout", &info, &hook, 1));
    assert(g_hook_calls == 0 && g_dump_count == 1);

    reset_dma();
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_EN;
    g_hook_calls = 0;
    fpga_dma_poll_hook_t fail{failing_hook_quiesced, nullptr};
    assert(!zdma_wait_transfer_complete("callback", &info, &fail, 1));
    assert(g_hook_calls == 1 && g_log_errors == 1 && (g_dma->ZDMA_CH_CTRL2 & ZDMA_CTRL2_EN) == 0U);

    reset_dma();
    g_dma_timeout_us = 2;
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_EN;
    g_hook_calls = 0;
    assert(!zdma_wait_channel_disabled("channel-stuck", "test"));
    assert(g_hook_calls == 0 && g_log_errors == 1 && g_dump_count == 1 &&
           (g_dma->ZDMA_CH_CTRL2 & ZDMA_CTRL2_EN) != 0U);
}

int main() {
    test_two_slots_and_tails();
    test_real_poll_and_cleanup();
    test_real_finalize_and_consume();
    test_zdma_wait_order();
    assert(g_fatal_count == 1);
    puts("PASS: real pack/poll/finalize/consume; scale padding canaries; source identity/epoch; helper cleanup; ZDMA DONE/error/timeout/callback/quiescence ordering");
}
'''

with tempfile.TemporaryDirectory(prefix="fpga-pack-dma-pipeline-") as directory:
    test_cpp = Path(directory) / "pipeline.cpp"
    binary = Path(directory) / "pipeline"
    test_cpp.write_text(harness)
    command = [
        os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
        "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I" + str(CPU),
        "-I" + str(ROOT / "ggml/include"), "-I" + str(ROOT / "ggml/src"), str(test_cpp),
        str(CPU / "fpga_q8_pack.cpp"), "-o", str(binary),
    ]
    subprocess.run(command, check=True, timeout=90)
    subprocess.run([str(binary)], check=True, timeout=90)
