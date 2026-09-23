#!/usr/bin/env python3
"""Run the production row/column scheduler with simulated banks; no hardware access."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()
start = source.index("static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(")
scheduler = source[start:source.index("static bool fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_single_bank(", start)]
fields = set(re.findall(r"(?:prepared\.|running->)(\w+)", scheduler))
fields.update(["bank", "row0", "rows", "col", "k_block0", "group_blocks", "weight_bytes", "job_id"])
fields.add("weight_src_off")
release_start = source.index("static bool fpga_release_pl_scaled_q8_tile_job(")
helpers = source[release_start:source.index("// The P2 contract is a tile-level", release_start)]
fields.update(re.findall(r"job\.(\w+)", helpers))
fields.add("output_dma_ready")
totals = set(re.findall(r"totals->(\w+)", scheduler))
totals.update(re.findall(r"totals->(\w+)", helpers))
harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <tuple>
#include "fpga_q8_layout.h"
struct ggml_tensor { void * data; };
struct block_q8_0_t {};
struct fpga_weight_cache_entry_t {};
struct fpga_prompt_weight_staging_t {};
constexpr int VPU_BLOCK_BEATS = 2;
int g_vpu_max_rows = 3;
int max_blocks = 2, test_blocks, test_columns, test_rows;
bool g_p2_input_preload_enabled, g_p1_sched_summary_enabled = true;
bool g_p2_result_overlap_enabled, g_vpu_descriptor_supported = true;
bool g_pingpong_timing_enabled = false;
constexpr unsigned ACT_END=16, WEIGHT_END=32, SPU_OUT_BASE=64, SPU_OUT_END=128, SPU_PARAM_BASE=128;
constexpr int FPGA_SLOT_FREE=0;
constexpr uint32_t WEIGHT_CACHE_BASE = 0x100000, P2_WEIGHT_RESIDENCY_END = 0x200000;
constexpr uint32_t DDR_REGION_SIZE = 0x400000;
constexpr size_t WEIGHT_CACHE_ALIGN = 4096;
bool g_p2_weight_residency_enabled;
long long g_p2_weight_residency_budget_mb = 1;
uint32_t g_p2_residency_next_off;
size_t g_p2_residency_next_slot;
std::vector<int> g_p2_resident_tiles(128);
std::vector<float> expected, stored;
std::vector<int> reads, stores;
void LOGE(const char *, ...) {}
template<class... Args> void fpga_log_line(Args...) {}
const char *p2_bank_label(int) { return "test"; }
struct { int pingpong_pairs, serial_submit_after_no_preload; } g_p1_sched_summary;
bool should_log_detail_run(unsigned) { return false; }
void LOGSTAGE(const char *, ...) {}
long long now_us() { return 100; }
int packed_q8_group_blocks_for_rows(int, int remaining) { return std::min(max_blocks, remaining); }
int64_t raw_result(int64_t row, int64_t col, int64_t block) {
    // Large opposite signs make changing the order of float additions visible.
    const int group = int(block) / max_blocks;
    const int64_t base = group % 3 == 0 ? (1LL << 50) : group % 3 == 2 ? -(1LL << 50) : 0;
    return base + ((row * 100 + col * 10 + block + 1) << 16);
}
void store_dst_value(const ggml_tensor *, int64_t row, int64_t col, float value) {
    assert(row >= 0 && row < test_rows && col >= 0 && col < test_columns);
    const size_t index = col * test_rows + row;
    assert(++stores[index] == 1 && reads[index] == test_blocks);
    assert(std::memcmp(&value, &expected[index], sizeof(value)) == 0);
    stored[index] = value;
}
'''
for name in ("fpga_p2_residency_align_up", "fpga_p2_residency_range_end"):
    begin = source.index(f"static bool {name}(")
    harness += source[begin:source.index("\n}\n", begin) + 3]
fields.discard("tensor_name")
harness += "struct fpga_tile_job_t { const char *tensor_name = nullptr; " + " ".join(f"long long {f} = 0;" for f in sorted(fields)) + " };\n"
bank_fields = {f for f in totals if f.startswith("bank_")}
harness += "struct fpga_stage_totals_t { " + " ".join(f"long long {f}{'[2]' if f in bank_fields else ''} = {{}};" for f in sorted(totals)) + " };\n"
harness += r'''
using Key = std::tuple<long long, long long, long long, long long>;
Key banks[2];
bool valid[2];
int active = -1, loads, activations, scales, launches, reused, operations, fail_at;
int pending_result = -1, consumed_jobs, overlap_reads, release_writes;
int cross_prepares, expected_crossings;
std::vector<Key> prepared_order, consumed_order;
fpga_tile_job_t returned;
long long logged_jobs;
unsigned long long logged_bytes;
bool step() { return ++operations != fail_at; }
Key key(const fpga_tile_job_t & j) { return {j.row0, j.rows, j.k_block0, j.group_blocks}; }
size_t allocation_bytes(int rows, int blocks) {
    // Independent payload arithmetic, including odd-row companion padding.
    const size_t weights = ((rows + 1) / 2) * blocks * 64;
    const size_t scales = rows * blocks * 2;
    return ((weights + 4095) / 4096 + (scales + 4095) / 4096) * 4096;
}
bool expect_cross(int rows) {
    const size_t available = P2_WEIGHT_RESIDENCY_END - g_p2_residency_next_off;
    return test_columns == 1 && g_p2_result_overlap_enabled && g_vpu_descriptor_supported &&
           (!g_p2_weight_residency_enabled || g_p2_residency_next_slot == 128 ||
            available < allocation_bytes(rows, std::min(max_blocks, test_blocks)));
}
constexpr uint64_t DDR_BASE_PHYS = 0, LMM_BASE_PHYS = 0, WEIGHT_BASE = 0;
bool dma_success = true;
bool fpga_dma_copy(uint64_t, uint64_t, size_t, const char *) { ++loads; return dma_success; }
void vpu_select_banks(int bank, int) { assert(bank != active); }
long long p2_event_now_us() { return 0; }
template<class... Args> void p2_trace_first_tile(Args...) {}
template<class... Args> void p2_event_trace(Args...) {}
'''
# Exercise the production DMA guards as well as the production scheduler.
start = source.index('    if (!job.weight_bank_reused &&')
preload = source[start:source.index('\n    if (g_p1_sched_summary_enabled)', start)]
start = source.index('    const long long dma_weight0', source.index('static bool fpga_submit_q8_tile_job('))
submit = source[start:source.index('    const long long dma_scale0', start)]
harness += 'bool weight_transfer(fpga_tile_job_t & job, bool preload) {\nif (preload) {\n'
harness += 'auto poison = [](const char *) { return false; };\n' + preload
harness += '\n} else {\n' + submit + '\n(void) dma_weight0; (void) dma_weight1;\n}\nreturn true;\n}\n'
harness += r'''
bool transfer(fpga_tile_job_t & j) {
    assert(active != j.bank);
    ++activations;
    const int before = loads;
    assert(weight_transfer(j, active != -1));
    if (j.weight_bank_reused) {
        assert(loads == before && valid[j.bank] && banks[j.bank] == key(j));
    } else {
        assert(loads == before + 1); banks[j.bank] = key(j); valid[j.bank] = true;
    }
    return true;
}
bool fpga_prepare_q8_tile_job(fpga_tile_job_t & j, const ggml_tensor *, void *,
    const block_q8_0_t *, int64_t row0, int rows, int64_t ib0, int groups, int64_t col,
    unsigned, const fpga_weight_cache_entry_t *, unsigned id, int bank,
    fpga_stage_totals_t *, fpga_prompt_weight_staging_t *) {
    if (!step()) return false;
    assert(bank != active);
    if (row0 > 0 && ib0 == 0 && col == 0) {
        assert((active != -1) == expect_cross(rows));
        cross_prepares += active != -1;
    }
    if (active != -1) assert(j.result_consumed || j.job_id == 0);
    j = {}; j.bank = bank; j.row0 = row0; j.rows = rows; j.k_block0 = ib0;
    j.group_blocks = groups; j.col = col; j.weight_bytes = 128; j.job_id = id + 1;
    prepared_order.push_back(key(j));
    return true;
}
bool fpga_preload_q8_tile_inputs(fpga_tile_job_t & j, const fpga_tile_job_t & running,
                               fpga_stage_totals_t *) {
    if (!step()) return false;
    assert(active == running.bank);
    // Exercise both successful preload and its legal terminal-job skip.
    if (j.job_id % 3) { transfer(j); j.input_preloaded = true; }
    return true;
}
template<class... Args> bool fpga_wait_and_drain_q8_tile_job(fpga_tile_job_t & j, Args...) {
    if (!step()) return false;
    assert(active == j.bank);
    // Any premature next output DMA would destroy A's unread DDR data.
    assert(pending_result == -1);
    pending_result = j.job_id; returned = j; j.output_dma_ready = true;
    return true;
}
bool fpga_write_post_spu_descriptor(fpga_tile_job_t & j, int, int, unsigned, const char *, bool, bool force) {
    if (!step()) return false;
    assert(active == j.bank); // Catch stale A descriptor writes after B starts.
    assert(j.output_dma_ready);
    if (!j.result_consumed) assert(force);
    ++release_writes; active = -1; return true;
}
int64_t ddr_read_spu_q16_row(unsigned off, uint16_t *row_id) {
    assert(pending_result == returned.job_id);
    const unsigned row = (off - SPU_OUT_BASE) / 16;
    assert(row < (unsigned) returned.rows);
    *row_id = row;
    if (!step()) *row_id = 0xffff; // Inject a real row validation failure.
    if (row == 0 && active != returned.bank && active != -1) ++overlap_reads;
    const size_t index = returned.col * test_rows + returned.row0 + row;
    assert(reads[index] == returned.k_block0 && stores[index] == 0);
    reads[index] += returned.group_blocks;
    const int64_t value = raw_result(returned.row0 + row, returned.col, returned.k_block0);
    if (row + 1 == (unsigned) returned.rows) {
        pending_result = -1; ++consumed_jobs; consumed_order.push_back(key(returned));
    }
    return value;
}
template<class... Args> bool fpga_submit_q8_tile_job(fpga_tile_job_t & j, Args...) {
    if (!step()) return false;
    assert(active == -1);
    if (!j.input_preloaded) transfer(j);
    assert(valid[j.bank] && banks[j.bank] == key(j));
    ++scales; ++launches; reused += j.weight_bank_reused != 0; active = j.bank;
    return true;
}
void fpga_log_prompt_weight_reuse(const char *, long long, long long jobs,
                                unsigned long long bytes) {
    logged_jobs = jobs; logged_bytes = bytes;
}
'''
harness += helpers
harness += scheduler
harness += r'''
bool run(int columns, bool preload, bool overlap, int failure,
         int rows = 7, int blocks = 5, int cache = 0) {
    active = -1; valid[0] = valid[1] = false;
    loads = activations = scales = launches = reused = operations = 0;
    fail_at = failure; logged_jobs = -1; logged_bytes = 0;
    pending_result = -1; consumed_jobs = overlap_reads = release_writes = 0;
    cross_prepares = expected_crossings = 0;
    prepared_order.clear(); consumed_order.clear();
    test_columns = columns; test_rows = rows; test_blocks = blocks;
    g_p2_input_preload_enabled = preload;
    g_p2_result_overlap_enabled = overlap;
    g_p2_weight_residency_enabled = cache != 0;
    g_p2_residency_next_slot = cache == 3 ? 128 : 0;
    size_t remaining = 1024 * 1024;
    const int first_rows = std::min(rows, g_vpu_max_rows);
    if (cache == 2) remaining = 0;
    if (cache == 4) remaining = allocation_bytes(rows % g_vpu_max_rows ? rows % g_vpu_max_rows : first_rows,
                                                std::min(max_blocks, blocks));
    if (cache == 5 || cache == 6) remaining = allocation_bytes(first_rows, std::min(max_blocks, blocks)) - (cache == 6);
    g_p2_residency_next_off = P2_WEIGHT_RESIDENCY_END - remaining;
    expected.assign(rows * columns, 0.0f);
    stored.assign(rows * columns, std::numeric_limits<float>::quiet_NaN());
    reads.assign(rows * columns, 0); stores.assign(rows * columns, 0);
    for (int col = 0; col < columns; ++col) for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; block += max_blocks) {
            expected[col * rows + row] += float(raw_result(row, col, block)) / 65536.0f;
        }
    }
    for (int row = g_vpu_max_rows; row < rows; row += g_vpu_max_rows) {
        expected_crossings += expect_cross(std::min(g_vpu_max_rows, rows - row));
    }
    ggml_tensor t = {};
    std::vector<block_q8_0_t> act(columns * blocks);
    std::vector<float> values;
    fpga_stage_totals_t totals = {};
    const bool ok = fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
        &t, &t, act, values, nullptr, &totals, "test", 0, blocks * 32, rows, columns, blocks, values);
    if (ok) {
        const int row_groups = (rows + g_vpu_max_rows - 1) / g_vpu_max_rows;
        const int weight_tiles = row_groups * ((blocks + max_blocks - 1) / max_blocks);
        assert(active == -1);
        assert(pending_result == -1 && consumed_jobs == launches && release_writes == launches);
        assert(cross_prepares == expected_crossings && prepared_order == consumed_order);
        for (int count : stores) assert(count == 1);
        assert(overlap_reads == (overlap && g_vpu_descriptor_supported ? launches - row_groups + expected_crossings : 0));
        assert(totals.result_overlap_jobs == overlap_reads);
        assert(loads == weight_tiles * std::min(columns, 2));
        assert(launches == weight_tiles * columns && scales == launches && activations == launches);
        assert(reused == weight_tiles * std::max(columns - 2, 0));
        if (columns > 1) {
            assert(logged_jobs == reused && logged_bytes == (unsigned long long) reused * 128);
        } else assert(logged_jobs == -1);
    } else assert(logged_jobs == -1);
    return ok;
}
int main() {
    for (bool preload : {false, true}) {
        fpga_tile_job_t j = {};
        dma_success = false;
        assert(!weight_transfer(j, preload));
        j.weight_bank_reused = true;
        assert(weight_transfer(j, preload));
        j.weight_bank_reused = false; j.input_preloaded = true;
        if (!preload) assert(weight_transfer(j, false));
        dma_success = true;
    }
    int cases = 0;
    for (bool descriptors : {false, true}) for (bool overlap : {false, true})
    for (int columns : {1, 2, 3, 4, 5, 46}) for (bool preload : {false, true})
    for (int blocks : {1, 2, 5}) for (int cache : {0, 1, 2, 3, 4, 5, 6}) {
        g_vpu_descriptor_supported = descriptors;
        assert(run(columns, preload, overlap, 0, 7, blocks, cache)); ++cases;
        const int count = operations;
        for (int fail = 1; fail <= count; ++fail) {
            assert(!run(columns, preload, overlap, fail, 7, blocks, cache)); ++cases;
        }
        assert(run(columns, preload, overlap, 0, 7, blocks, cache)); ++cases;
    }
    // Actual Gemma shapes: 85 jobs/layer, 15 old vs 78 cross-row handoffs.
    g_vpu_max_rows = 256; max_blocks = 64; g_vpu_descriptor_supported = true;
    for (bool preload : {false, true}) for (bool overlap : {false, true}) {
        int jobs = 0, handoffs = 0;
        for (auto shape : {std::pair<int,int>{1024,36}, {256,36}, {256,36}, {1152,32},
                           {6912,36}, {6912,36}, {1152,216}}) {
            assert(run(1, preload, overlap, 0, shape.first, shape.second, 2)); ++cases;
            jobs += launches; handoffs += overlap_reads;
        }
        assert(jobs * 26 == 2210 && handoffs * 26 == (overlap ? 2028 : 0));
    }
    // Capacity may forbid crossing into a small tail after allowing full rows.
    for (int rows : {1, 255, 256, 257, 512, 513, 769}) for (int blocks : {1, 36, 64, 65, 216})
    for (int cache : {0, 1, 2, 3, 4, 5, 6}) {
        assert(run(1, true, true, 0, rows, blocks, cache)); ++cases;
    }
    printf("PASS: %d scheduler cases; cross-row preparation, bit-exact ordered sums, cache capacity/tails, prompt reuse, failures/retries; Gemma 2210 jobs and 2028 overlap opportunities\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-bank-reuse-") as directory:
    cpp = Path(directory) / "test.cpp"
    exe = Path(directory) / "test"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-Wall", "-Wextra",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    "-I" + str(root / "ggml/src/ggml-cpu"), str(cpp),
                    str(root / "ggml/src/ggml-cpu/fpga_q8_layout.cpp"), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
