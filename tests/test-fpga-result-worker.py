#!/usr/bin/env python3
"""Exercise the production result worker/exit guard with RAM, never board IO."""
import os
import re
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()
begin = source.index("class fpga_result_worker_t {")
end = source.index("static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(", begin)
worker = source[begin:end]
scheduler = source[end:source.index("static bool fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_single_bank(", end)]
reader_begin = source.index("static int64_t ddr_read_spu_q16_row(")
reader = source[reader_begin:source.index("\n}\n", reader_begin) + 3]
harness = r'''
#include <pthread.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
static constexpr uint32_t SPU_OUT_BASE = 0;
alignas(16) static uint8_t memory[256 * 16];
static std::atomic<bool> gate{false}, entered{false};
static bool slow_reads = false;
static int errors = 0;
#define LOGE(...) (++errors)
[[noreturn]] static void fpga_fatal(const char *, ...) { throw std::runtime_error("fatal"); }
static void fpga_p2_boundary_marker(const char *, ...) {}
static long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint8_t * ddr_ptr(uint32_t off, size_t bytes) {
    assert(off <= sizeof(memory) && bytes <= sizeof(memory) - off);
    entered.store(true);
    while (gate.load()) std::this_thread::yield();
    if (slow_reads) std::this_thread::sleep_for(std::chrono::microseconds(25));
    return memory + off;
}
struct fpga_stage_totals_t {
    long long host_result_us = 0, bank_host_read_us[2] = {};
    long long result_overlap_jobs = 0, result_overlap_host_us = 0;
};
'''
harness += reader + worker
common = harness
harness += r'''
static int64_t value(int row, int tile) {
    const int64_t edge[] = {INT64_MIN, INT64_MAX, -65537, 0, 65537};
    return row < 5 ? edge[row] : ((int64_t) row * 1234567 - tile * 7654321);
}
static void fill(int rows, int tile) {
    memset(memory, 0xA5, sizeof(memory));
    for (int row = 0; row < rows; ++row) {
        memory[row * 16] = (uint8_t) row;
        memory[row * 16 + 1] = (uint8_t) (row >> 8);
        const int64_t v = value(row, tile);
        memcpy(memory + row * 16 + 2, &v, 8);
    }
}
static void wait_entered() {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!entered.load()) {
        assert(std::chrono::steady_clock::now() < limit);
        std::this_thread::yield();
    }
}
int main() {
    fpga_stage_totals_t totals;
    for (int rows : {1, 3, 255, 256}) {
        std::vector<float> actual(rows * 3, 0), expected(rows * 3, 0);
        for (int tile = 0; tile < 40; ++tile) {
            const int col = tile % 3;
            fill(rows, tile);
            fpga_result_pending_t pending{&totals};
            assert(g_result_worker.submit(rows, actual.data() + col * rows));
            pending.pending = true;
            pending.bank = tile & 1;
            pending.job_id = tile + 1;
            // Expected additions use exactly the serial row/K order.
            for (int row = 0; row < rows; ++row)
                expected[col * rows + row] += (float) value(row, tile) / 65536.0f;
            assert(pending.finish()); // required before overwrite or resize
            assert(memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
        }
    }
    assert(totals.result_overlap_jobs == 160);
    assert(totals.host_result_us == totals.result_overlap_host_us);
    assert(totals.host_result_us == totals.bank_host_read_us[0] + totals.bank_host_read_us[1]);
    // Force a delayed consumer: producer work can progress, but a second
    // task is rejected and collection cannot return before the read finishes.
    fill(1, 0);
    float accum = 0;
    gate.store(true); entered.store(false);
    assert(g_result_worker.submit(1, &accum));
    wait_entered();
    assert(!g_result_worker.submit(1, &accum));
    std::atomic<bool> collected{false};
    std::thread collector([&] {
        long long elapsed; int bad;
        assert(g_result_worker.collect(elapsed, bad));
        collected.store(true);
    });
    assert(!collected.load());
    gate.store(false); collector.join();
    assert(collected.load());
    // Guard on an early return completes the outstanding read before the
    // caller destroys or reuses accumulation storage.
    fill(1, 0); accum = 0;
    [&] {
        fpga_result_pending_t pending{nullptr};
        assert(g_result_worker.submit(1, &accum));
        pending.pending = true;
        return;
    }();
    assert(accum == (float) INT64_MIN / 65536.0f);
    // Bad row IDs remain errors, never repaired/substituted results.
    fill(3, 0); memory[16] = 9;
    float bad_accum[3] = {};
    fpga_result_pending_t bad{&totals};
    assert(g_result_worker.submit(3, bad_accum)); bad.pending = true;
    assert(!bad.finish()); assert(errors == 1);
    assert(bad_accum[1] == 0 && bad_accum[2] == 0);
    // Shutdown must drain outstanding work; restarting is supported.
    fill(1, 0); accum = 0;
    assert(g_result_worker.submit(1, &accum));
    g_result_worker.shutdown();
    assert(accum == (float) INT64_MIN / 65536.0f);
    assert(g_result_worker.submit(1, &accum));
    long long elapsed; int bad_row;
    assert(g_result_worker.collect(elapsed, bad_row));
    g_result_worker.shutdown();
    puts("PASS: ordered accumulation, tails, columns, signed extremes, delayed reader, overwrite gate, early return, failure, shutdown/restart");
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-result-worker-") as directory:
    cpp = Path(directory) / "worker.cpp"
    binary = Path(directory) / "worker"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2",
                    "-Wall", "-Wextra", "-Werror", "-pthread",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    str(cpp), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)

    # Compile the real scheduler too. Only hardware/packing endpoints are
    # mocked; row transitions, reuse of slots, worker dispatch and joins are
    # production code. Delayed reads make missing joins visible numerically.
    fields = set(re.findall(r"(?:running->|prepared\.)(\w+)", scheduler))
    fields.update(["rows", "row0", "col", "bank", "job_id", "weight_bytes", "k_block0"])
    extra_totals = set(re.findall(r"totals->(\w+)", scheduler)) - {
        "result_overlap_jobs", "result_overlap_host_us"}
    integration = common.replace(
        "long long host_result_us = 0, bank_host_read_us[2] = {};",
        "long long host_result_us = 0, bank_host_read_us[2] = {};\n" +
        "\n".join(f"long long {field} = 0;" for field in sorted(extra_totals)))
    integration += "\nstruct fpga_tile_job_t {\n" + "\n".join(
        f"long long {field} = 0;" for field in sorted(fields)) + "\n};\n"
    integration += r'''
#include <algorithm>
#include <climits>
#include <limits>
static void mock_log(const char *, ...) {}
#define LOGSTAGE(...) mock_log(__VA_ARGS__)
struct ggml_tensor { void * data; int64_t n; };
struct block_q8_0_t {};
struct fpga_weight_cache_entry_t {};
struct fpga_prompt_weight_staging_t {};
static bool g_p2_result_overlap_enabled = true, g_vpu_descriptor_supported = true;
static bool g_p2_weight_residency_enabled = false, g_p1_sched_summary_enabled = true;
static bool g_p2_input_preload_enabled = true;
static size_t g_p2_residency_next_slot = 0;
static std::vector<int> g_p2_resident_tiles;
static long long g_p2_weight_residency_budget_mb = 0;
static uint32_t g_p2_residency_next_off = 0;
static constexpr int g_vpu_max_rows = 3, VPU_BLOCK_BEATS = 2;
static constexpr uint32_t ACT_END = 0, WEIGHT_END = 0, SPU_OUT_END = 0, SPU_PARAM_BASE = 0;
static constexpr uint32_t WEIGHT_CACHE_BASE = 0, WEIGHT_CACHE_ALIGN = 16, P2_WEIGHT_RESIDENCY_END = 0;
static struct { long long pingpong_pairs = 0, serial_submit_after_no_preload = 0; } g_p1_sched_summary;
static int fail_prepare = -1, fail_drain = -1, fail_submit = -1, corrupt_drain = -1;
static bool should_log_detail_run(uint32_t) { return false; }
static int packed_q8_group_blocks_for_rows(int, int remaining) { return std::min(2, remaining); }
static size_t weight_window_bytes_for_rows(int rows, int beats) { return rows * beats * 16; }
static bool fpga_p2_residency_align_up(size_t n, size_t, size_t * out) { *out = n; return true; }
static bool fpga_p2_residency_range_end(uint32_t, size_t, uint32_t, uint32_t *) { return false; }
static void store_dst_value(const ggml_tensor * dst, int64_t row, int64_t col, float v) {
    static_cast<float *>(dst->data)[col * dst->n + row] = v;
}
static bool fpga_prepare_q8_tile_job(fpga_tile_job_t & j, const ggml_tensor *, const void *,
        const block_q8_0_t *, int64_t row0, int rows, int64_t block, int, int64_t col,
        uint32_t, const fpga_weight_cache_entry_t *, uint32_t tile, int bank,
        fpga_stage_totals_t *, fpga_prompt_weight_staging_t *) {
    j = {}; j.row0 = row0; j.rows = rows; j.col = col; j.k_block0 = block;
    j.job_id = tile + 1; j.bank = bank; j.weight_bytes = 16;
    return (int) tile != fail_prepare;
}
static bool fpga_preload_q8_tile_inputs(fpga_tile_job_t &, const fpga_tile_job_t &, fpga_stage_totals_t *) { return true; }
static bool fpga_submit_q8_tile_job(fpga_tile_job_t & j, fpga_stage_totals_t *, const char *, int,
        int64_t, int64_t, int64_t, int) { return j.job_id != fail_submit; }
static int64_t result_value(const fpga_tile_job_t & j, int row) {
    return (j.row0 + row + 1) * (j.col + 1) * (j.k_block0 + 1) * 65537;
}
static bool fpga_wait_and_drain_q8_tile_job(fpga_tile_job_t & j, fpga_stage_totals_t *, const char *, int,
        int64_t, int64_t, int64_t, int) {
    if (j.job_id == fail_drain) return false;
    for (int row = 0; row < j.rows; ++row) {
        memory[row * 16] = j.job_id == corrupt_drain ? 99 : (uint8_t) row;
        memory[row * 16 + 1] = 0;
        const int64_t v = result_value(j, row);
        memcpy(memory + row * 16 + 2, &v, 8);
    }
    return true;
}
static bool fpga_release_pl_scaled_q8_tile_job(fpga_tile_job_t &, bool = false) { return true; }
static bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t & j, std::vector<float> & accum,
        fpga_stage_totals_t *, bool = true) {
    for (int row = 0; row < j.rows; ++row) {
        uint16_t id;
        const int64_t v = ddr_read_spu_q16_row(row * 16, &id);
        if (id != row) return false;
        accum[j.col * j.rows + row] += (float) v / 65536.0f;
    }
    return true;
}
static void fpga_log_prompt_weight_reuse(const char *, int64_t, long long, uint64_t) {}
'''
    integration += scheduler
    integration += r'''
int main() {
    slow_reads = true;
    for (bool overlap : {false, true}) {
        g_p2_result_overlap_enabled = overlap;
        for (int n : {1, 3, 4, 7}) for (int m : {1, 2, 5}) for (int nb : {1, 3, 5}) {
            std::vector<float> output(n * m, -99), accum, scales;
            std::vector<block_q8_0_t> acts(m * nb);
            ggml_tensor tensor{output.data(), n}; fpga_stage_totals_t totals;
            assert(fpga_hw_q8_0_matmul_dma_to_ip_pipelined(&tensor, &tensor, acts, scales,
                nullptr, &totals, "test", 0, nb * 32, n, m, nb, accum));
            for (int col = 0; col < m; ++col) for (int row = 0; row < n; ++row) {
                float expected = 0;
                for (int block = 0; block < nb; block += 2)
                    expected += (float) ((row + 1LL) * (col + 1) * (block + 1) * 65537) / 65536.0f;
                assert(output[col * n + row] == expected);
            }
        }
    }
    // Errors while a reader may be pending must leave no outstanding writes
    // when the scheduler returns; subsequent calls reuse the same worker.
    for (int which = 0; which < 4; ++which) {
        fail_prepare = fail_drain = fail_submit = corrupt_drain = -1;
        if (which == 0) fail_prepare = 2;
        if (which == 1) fail_drain = 2;
        if (which == 2) fail_submit = 3;
        if (which == 3) corrupt_drain = 1;
        std::vector<float> output(7, -99), accum, scales;
        std::vector<block_q8_0_t> acts(5);
        ggml_tensor tensor{output.data(), 7}; fpga_stage_totals_t totals;
        assert(!fpga_hw_q8_0_matmul_dma_to_ip_pipelined(&tensor, &tensor, acts, scales,
            nullptr, &totals, "test", 0, 160, 7, 1, 5, accum));
        const auto snapshot = accum;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        assert(snapshot == accum);
    }
    g_result_worker.shutdown();
    puts("PASS: production scheduler, serial/parallel equivalence, prompt columns, row carry/tails, single/final tiles, prepare/drain/submit/result failures");
}
'''
    cpp.write_text(integration)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2",
                    "-Wall", "-Wextra", "-Werror", "-pthread",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    str(cpp), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)
