#!/usr/bin/env python3
"""Run the production column scheduler with simulated banks; no hardware access."""
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
totals = set(re.findall(r"totals->(\w+)", scheduler))
harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <tuple>
struct ggml_tensor { void * data; };
struct block_q8_0_t {};
struct fpga_weight_cache_entry_t {};
struct fpga_prompt_weight_staging_t {};
constexpr int VPU_BLOCK_BEATS = 4;
int g_vpu_max_rows = 3;
bool g_p2_input_preload_enabled, g_p1_sched_summary_enabled = true;
struct { int pingpong_pairs, serial_submit_after_no_preload; } g_p1_sched_summary;
bool should_log_detail_run(unsigned) { return false; }
void LOGSTAGE(const char *, ...) {}
long long now_us() { return 100; }
int packed_q8_group_blocks_for_rows(int, int remaining) { return std::min(2, remaining); }
void store_dst_value(const ggml_tensor *, int64_t, int64_t, float) {}
'''
harness += "struct fpga_tile_job_t { " + " ".join(f"long long {f} = 0;" for f in sorted(fields)) + " };\n"
harness += "struct fpga_stage_totals_t { " + " ".join(f"long long {f} = 0;" for f in sorted(totals)) + " };\n"
harness += r'''
using Key = std::tuple<long long, long long, long long, long long>;
Key banks[2];
bool valid[2];
int active = -1, loads, activations, scales, launches, reused, operations, fail_at;
long long logged_jobs;
unsigned long long logged_bytes;
bool step() { return ++operations != fail_at; }
Key key(const fpga_tile_job_t & j) { return {j.row0, j.rows, j.k_block0, j.group_blocks}; }
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
    j = {}; j.bank = bank; j.row0 = row0; j.rows = rows; j.k_block0 = ib0;
    j.group_blocks = groups; j.col = col; j.weight_bytes = 128; j.job_id = id + 1;
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
    assert(active == j.bank); return true;
}
bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t & j, std::vector<float> &,
                                         fpga_stage_totals_t *) {
    if (!step()) return false;
    assert(active == j.bank); active = -1; return true;
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
harness += scheduler
harness += r'''
bool run(int columns, bool preload, int failure) {
    active = -1; valid[0] = valid[1] = false;
    loads = activations = scales = launches = reused = operations = 0;
    fail_at = failure; logged_jobs = -1; logged_bytes = 0;
    g_p2_input_preload_enabled = preload;
    ggml_tensor t = {};
    std::vector<block_q8_0_t> act(columns * 5);
    std::vector<float> values;
    fpga_stage_totals_t totals = {};
    const bool ok = fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
        &t, &t, act, values, nullptr, &totals, "test", 0, 160, 7, columns, 5, values);
    if (ok) {
        // 3 row tiles x 3 K tiles, including partial final tiles.
        assert(active == -1);
        assert(loads == 9 * std::min(columns, 2));
        assert(launches == 9 * columns && scales == launches && activations == launches);
        assert(reused == 9 * std::max(columns - 2, 0));
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
    for (int columns : {1, 2, 3, 4, 5, 46}) for (bool preload : {false, true}) {
        assert(run(columns, preload, 0)); ++cases;
        const int count = operations;
        for (int fail = 1; fail <= count; ++fail) {
            assert(!run(columns, preload, fail)); ++cases;
        }
        assert(run(columns, preload, 0)); ++cases;
    }
    printf("PASS: %d scheduler cases, both preload modes, every failure boundary, fresh retries\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-bank-reuse-") as directory:
    cpp = Path(directory) / "test.cpp"
    exe = Path(directory) / "test"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-Wall", "-Wextra",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
