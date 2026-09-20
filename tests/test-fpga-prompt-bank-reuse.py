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
#include <vector>
#include <tuple>
struct ggml_tensor { void * data; };
struct block_q8_0_t {};
struct fpga_weight_cache_entry_t {};
struct fpga_prompt_weight_staging_t {};
constexpr int VPU_BLOCK_BEATS = 4;
int g_vpu_max_rows = 3;
bool g_p2_input_preload_enabled, g_p1_sched_summary_enabled = true;
bool g_p2_result_overlap_enabled, g_vpu_descriptor_supported = true;
bool g_pingpong_timing_enabled = false;
constexpr unsigned ACT_END=16, WEIGHT_END=32, SPU_OUT_BASE=64, SPU_OUT_END=128, SPU_PARAM_BASE=128;
constexpr int FPGA_SLOT_FREE=0;
void LOGE(const char *, ...) {}
template<class... Args> void fpga_log_line(Args...) {}
const char *p2_bank_label(int) { return "test"; }
struct { int pingpong_pairs, serial_submit_after_no_preload; } g_p1_sched_summary;
bool should_log_detail_run(unsigned) { return false; }
void LOGSTAGE(const char *, ...) {}
long long now_us() { return 100; }
int packed_q8_group_blocks_for_rows(int, int remaining) { return std::min(2, remaining); }
void store_dst_value(const ggml_tensor *, int64_t row, int64_t col, float value) {
    // Three K groups start at 0, 2, 4. The mocked raw result preserves
    // distinct row/column identities, so reordered or skipped sums fail.
    assert(value == float(3 * (row * 100 + col * 10) + 9));
}
'''
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
fpga_tile_job_t returned;
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
    const int64_t value = (returned.row0 + row) * 100 + returned.col * 10 + returned.k_block0 + 1;
    if (row + 1 == (unsigned) returned.rows) { pending_result = -1; ++consumed_jobs; }
    return value * 65536;
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
bool run(int columns, bool preload, bool overlap, int failure) {
    active = -1; valid[0] = valid[1] = false;
    loads = activations = scales = launches = reused = operations = 0;
    fail_at = failure; logged_jobs = -1; logged_bytes = 0;
    pending_result = -1; consumed_jobs = overlap_reads = release_writes = 0;
    g_p2_input_preload_enabled = preload;
    g_p2_result_overlap_enabled = overlap;
    ggml_tensor t = {};
    std::vector<block_q8_0_t> act(columns * 5);
    std::vector<float> values;
    fpga_stage_totals_t totals = {};
    const bool ok = fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
        &t, &t, act, values, nullptr, &totals, "test", 0, 160, 7, columns, 5, values);
    if (ok) {
        // 3 row tiles x 3 K tiles, including partial final tiles.
        assert(active == -1);
        assert(pending_result == -1 && consumed_jobs == launches && release_writes == launches);
        assert(overlap_reads == (overlap && g_vpu_descriptor_supported ? launches - 3 : 0));
        assert(totals.result_overlap_jobs == overlap_reads);
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
    for (bool descriptors : {false, true}) for (bool overlap : {false, true})
    for (int columns : {1, 2, 3, 4, 5, 46}) for (bool preload : {false, true}) {
        g_vpu_descriptor_supported = descriptors;
        assert(run(columns, preload, overlap, 0)); ++cases;
        const int count = operations;
        for (int fail = 1; fail <= count; ++fail) {
            assert(!run(columns, preload, overlap, fail)); ++cases;
        }
        assert(run(columns, preload, overlap, 0)); ++cases;
    }
    printf("PASS: %d scheduler cases; overlap/preload/descriptor modes, no unread-result overwrite or stale release, row failures, fresh retries\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-bank-reuse-") as directory:
    cpp = Path(directory) / "test.cpp"
    exe = Path(directory) / "test"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-Wall", "-Wextra",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
