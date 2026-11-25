#!/usr/bin/env python3
"""RAM-only TASK G helper splitting, prefetch ownership and poll-hook checks."""

import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()


def definition(signature):
    start = SOURCE.index(signature)
    brace = SOURCE.index("{", start)
    depth = 0
    for pos in range(brace, len(SOURCE)):
        depth += (SOURCE[pos] == "{") - (SOURCE[pos] == "}")
        if depth == 0:
            return SOURCE[start:pos + 1]
    raise AssertionError(signature)


split = definition("static int fpga_p2_pack_helper_split(")
scale_entries = definition("static size_t fpga_p2_pack_helper_scale_entries(")
promote = definition("static bool fpga_p2_weight_prefetch_promote(")
dma = definition("static bool fpga_p2_weight_prefetch_dma(")
configure = definition("static void fpga_host_offload_configure(")
submit = definition("static bool fpga_submit_q8_tile_job(")
scheduler = definition("static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(")
ctx_end = SOURCE.index("} fpga_p2_prefetch_read_ctx_t;")
ctx = SOURCE[SOURCE.rfind("typedef struct {", 0, ctx_end):ctx_end + len("} fpga_p2_prefetch_read_ctx_t;")]

# The pre-readback failure branch also restores the bank; rindex selects the
# normal post-DMA restoration. Runtime cases below cover both restoration paths.
assert dma.index("vpu_select_banks(job.bank, run_bank)") < dma.index("vpu_rd32(REG_BANK)") < dma.index('"PREFETCH_W"') < dma.rindex("vpu_select_banks(run_bank, run_bank)")
weight_call = submit.index('job.weight_bytes, "WEIGHT")')
assert submit.rfind("if (weight_dma_needed &&", 0, weight_call) >= 0
assert "const bool weight_dma_needed = !job.weight_bank_reused && !job.weight_prefetched;" in submit
for forbidden in ("CTRL_CLEAR_DONE", "configure_vpu", "vpu_write_tile_descriptor", "SPU_PARAM_BASE", "SPU_SCRATCH_BASE"):
    assert forbidden not in dma
assert scheduler.count("if (ready.job.group_beats == prepared.group_beats)") == 2
assert "poll_hook" not in scheduler and "poll_hook" not in submit
assert "const bool weight_prefetch = pipeline_route && overlap_results;" in scheduler
assert "const bool pipeline_route = g_p2_pack_dma_pipeline_enabled && m == 1;" in scheduler
assert "if (!weight_prefetch || !pipeline_prepared) prepared.weight_prefetched = false;" in scheduler
# Host I/O policy is fixed: configure() reads only the cycle-phase switch.
assert configure.count('"FPGA_') == 1 and 'flag("FPGA_P2_CYCLE_PHASES")' in configure
assert "static constexpr int P2_PACK_HELPERS   = 3;" in SOURCE
assert "static constexpr int P2_HOOK_READ_ROWS = 32;" in SOURCE
assert "fpga_p2_pack_helper_split(pair_count, P2_PACK_HELPERS, begin, end)" in SOURCE
assert "const fpga_dma_poll_hook_t hook = { fpga_p2_prefetch_read_poll_chunked, read_ctx };" in dma

harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct fpga_tile_job_t {
    int bank = 0;
    uint32_t job_id = 1, tile_id = 1;
    uint32_t weight_src_off = 0x100000, scale_src_off = 0x310000;
    size_t weight_bytes = 256, scale_bytes = 16;
    bool weight_prefetched = false, weight_bank_reused = false;
};
struct fpga_stage_totals_t {};
struct fpga_p2_dma_pipeline_lookahead_t {
    bool valid = false, consumed = false;
    bool worker_submitted = false, worker1_submitted = false, worker2_submitted = false;
    fpga_tile_job_t job;
};
struct fpga_dma_poll_hook_t { bool (*fn)(void *, long long); void * context; };
struct { bool created = false; } g_p2_pack_workers[3];
static constexpr int P2_PACK_HELPERS = 3, P2_HOOK_READ_ROWS = 32;
static constexpr bool k_fpga_pair_mem_ops = false;
long long g_p2_pack_offload_jobs, g_p2_pack_offload_helper_us[3], g_p2_pack_offload_finalize_wait_us;
uint64_t g_p2_pack_offload_helper_bytes[3];
long long g_p2_weight_prefetch_jobs, g_p2_weight_prefetch_consumed;
uint64_t g_p2_weight_prefetch_bytes;
long long g_p2_weight_prefetch_dma_us, g_p2_weight_prefetch_skip_geometry;
long long g_p2_weight_prefetch_restarts, g_p2_weight_prefetch_hook_reads, g_p2_weight_prefetch_late_reads;
bool g_p2_cycle_phases_enabled;
long long g_p2_weight_prefetch_partial_reads, g_p2_cycle_count, g_zdma_lean_setup_desc;
enum { P2CP_COUNT = 10 };
long long g_p2_cycle_us[P2CP_COUNT];
template<class... T> void LOGI(T...) {}
template<class... T> void LOGE(T...) {}
template<class... T> void fpga_fatal(const char * msg, T...) { throw std::runtime_error(msg); }
int pair_store_calls = 0;
void fpga_q8_pack_set_pair_stores(bool enabled) { assert(enabled == k_fpga_pair_mem_ops); ++pair_store_calls; }
constexpr uint64_t DDR_BASE_PHYS = 0x800000000ULL, LMM_BASE_PHYS = 0xa0000000ULL;
constexpr uint32_t WEIGHT_BASE = 0x100000U, REG_BANK = 0x100U;
int reads = 0, dma_calls = 0, bank_reads = 0, finalize_calls = 0;
bool read_ok = true, dma_ok = true, finalize_ok = true, ranges_ok = true, invoke_hook = true;
int mismatch = 0;
uint32_t bank_reg = 0;
std::vector<std::string> events;
long long now_us() { static long long clock = 100; return ++clock; }
void vpu_select_banks(int wr, int rd) { bank_reg = (wr & 1) | ((rd & 1) << 1); events.push_back("select" + std::to_string(bank_reg)); }
uint32_t vpu_rd32(uint32_t reg) {
    assert(reg == REG_BANK);
    events.push_back("readback");
    return ++bank_reads == mismatch ? bank_reg ^ 1U : bank_reg;
}
bool fpga_p2_dma_pipeline_slot_ranges(uint32_t, uint32_t, size_t, size_t) { return ranges_ok; }
bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t &, std::vector<float> &, fpga_stage_totals_t *, bool release) {
    assert(!release); ++reads; events.push_back("consume"); return read_ok;
}
bool fpga_p2_dma_pipeline_finalize(fpga_p2_dma_pipeline_lookahead_t & record) {
    ++finalize_calls;
    if (finalize_ok) record.worker_submitted = record.worker1_submitted = record.worker2_submitted = false;
    return finalize_ok;
}
bool fpga_p2_prefetch_read_poll_chunked(void *, long long);
bool fpga_dma_copy(uint64_t src, uint64_t dst, size_t bytes, const char * tag, const fpga_dma_poll_hook_t * hook) {
    ++dma_calls; events.push_back("dma");
    assert(src == DDR_BASE_PHYS + 0x100000 && dst == LMM_BASE_PHYS + WEIGHT_BASE && bytes == 256);
    assert(std::strcmp(tag, "PREFETCH_W") == 0 && bank_reg == 1);
    if (hook && invoke_hook) {
        for (int i = 0; i < 5; ++i) if (!hook->fn(hook->context, i)) return false;
    }
    return dma_ok;
}
'''
consume_once = r'''
bool fpga_p2_prefetch_read_poll_chunked(void * context, long long) {
    auto * const ctx = static_cast<fpga_p2_prefetch_read_ctx_t *>(context);
    if (!ctx || !ctx->job || !ctx->accum) return false;
    if (ctx->done) return ctx->ok;
    ctx->done = true;
    ctx->ok = fpga_accumulate_pl_scaled_q8_tile_job(*ctx->job, *ctx->accum, ctx->totals, false);
    if (ctx->ok) ++g_p2_weight_prefetch_hook_reads;
    return ctx->ok;
}
'''
harness += ctx + "\n" + consume_once + split + "\n" + scale_entries + "\n" + promote + "\n" + dma + "\n" + configure
harness += r'''
void reset_mocks() {
    reads = dma_calls = bank_reads = finalize_calls = mismatch = 0;
    read_ok = dma_ok = finalize_ok = ranges_ok = invoke_hook = true;
    bank_reg = 0; events.clear();
    g_p2_weight_prefetch_jobs = g_p2_weight_prefetch_bytes = g_p2_weight_prefetch_hook_reads = 0;
    g_p2_weight_prefetch_dma_us = 0;
}
void reject_config(const char * expected) {
    bool rejected = false;
    try { fpga_host_offload_configure(); }
    catch (const std::runtime_error & e) { rejected = true; assert(std::strcmp(e.what(), expected) == 0); }
    assert(rejected);
}
int main() {
    int cases = 0;
    for (size_t p = 2; p <= 300; ++p) for (size_t rows : {2*p-1, 2*p})
    for (int helpers : {2,3}) for (size_t groups : {1U,36U,64U}) {
        size_t b[3], e[3];
        int h = fpga_p2_pack_helper_split(p, helpers, b, e);
        assert(h == ((helpers == 3 && p >= 3) ? 3 : 2));
        assert(b[0] == 0 && e[h-1] == p);
        size_t total = 0;
        for (int i = 0; i < h; ++i) {
            assert(b[i] < e[i] && (i == 0 || b[i] == e[i-1]));
            total += fpga_p2_pack_helper_scale_entries(b[i], e[i], rows, groups);
        }
        assert(total == rows * groups);
        if (helpers == 2) assert(b[1] == p/2);
        if (h == 2) assert(b[2] == p && e[2] == p);
        ++cases;
    }
    fpga_tile_job_t prior, running;
    std::vector<float> accum(1);
    for (int fault = 0; fault < 12; ++fault) {
        reset_mocks();
        fpga_p2_dma_pipeline_lookahead_t ready;
        ready.valid = true;
        ready.job.bank = 1;
        fpga_p2_prefetch_read_ctx_t ctx{&prior, &accum, nullptr, false, true};
        if (fault == 1) mismatch = 1;
        if (fault == 2) mismatch = 2;
        if (fault == 3) dma_ok = false;
        if (fault == 4) read_ok = false;
        if (fault == 5) ready.job.bank = 0;
        if (fault == 6) ready.worker2_submitted = true;
        if (fault == 7) ready.job.weight_bank_reused = true;
        if (fault == 8) ready.consumed = true;
        if (fault == 9) ranges_ok = false;
        if (fault == 10) invoke_hook = false; // DMA already complete before polling hook.
        if (fault == 11) ctx.job = nullptr; // Initial job has no preceding result.
        const bool ok = fpga_p2_weight_prefetch_dma(ready, running, &ctx);
        const bool expected = fault == 0 || fault >= 10;
        assert(ok == expected && ready.job.weight_prefetched == expected);
        assert(bank_reg == 0); // Every attempted DMA restores the running bank.
        assert(g_p2_weight_prefetch_jobs == int(expected));
        assert(g_p2_weight_prefetch_bytes == (expected ? 256U : 0U));
        assert(dma_calls == ((fault == 1 || (fault >= 5 && fault <= 9)) ? 0 : 1));
        if (fault == 0) {
            assert((events == std::vector<std::string>{"select1","readback","dma","consume","select0","readback"}));
            assert(reads == 1 && ctx.done && g_p2_weight_prefetch_hook_reads == 1);
        }
        if (fault >= 10) assert(reads == 0 && !ctx.done);
    }
    reset_mocks();
    fpga_p2_dma_pipeline_lookahead_t lookahead, ready;
    lookahead.valid = true; lookahead.worker2_submitted = true; lookahead.job.weight_prefetched = true;
    assert(fpga_p2_weight_prefetch_promote(lookahead, ready));
    assert(!lookahead.valid && !ready.worker2_submitted && !ready.job.weight_prefetched);
    assert(!fpga_p2_weight_prefetch_promote(lookahead, ready));
    lookahead.valid = true; assert(!fpga_p2_weight_prefetch_promote(lookahead, ready));
    ready.valid = false; finalize_ok = false;
    assert(!fpga_p2_weight_prefetch_promote(lookahead, ready) && lookahead.valid && !ready.valid);

    // Retired switches are inert: garbage values must neither fail nor change the policy.
    for (const char * name : {"FPGA_P2_PACK_OFFLOAD","FPGA_P2_WEIGHT_PREFETCH","FPGA_P2_PACK_HELPERS",
                              "FPGA_PACK_STP","FPGA_RESULT_LDP","FPGA_ZDMA_LEAN_SETUP","FPGA_P2_HOOK_READ_ROWS",
                              "FPGA_UNCACHED_WRITE_PROBE"}) setenv(name, "garbage", 1);
    unsetenv("FPGA_P2_CYCLE_PHASES");
    g_p2_weight_prefetch_jobs = g_p2_weight_prefetch_consumed = 5;
    fpga_host_offload_configure();
    assert(pair_store_calls == 1 && !g_p2_cycle_phases_enabled);
    assert(!g_p2_weight_prefetch_jobs && !g_p2_weight_prefetch_consumed && !g_p2_weight_prefetch_bytes);
    setenv("FPGA_P2_CYCLE_PHASES", "1", 1); fpga_host_offload_configure(); assert(g_p2_cycle_phases_enabled);
    setenv("FPGA_P2_CYCLE_PHASES", "0", 1); fpga_host_offload_configure(); assert(!g_p2_cycle_phases_enabled);
    setenv("FPGA_P2_CYCLE_PHASES", "yes", 1);
    reject_config("%s must be 0 or 1");
    unsetenv("FPGA_P2_CYCLE_PHASES");
    g_p2_pack_workers[2].created = true;
    reject_config("Host I/O policy cannot change while pack helpers exist");
    std::printf("PASS test-fpga-weight-prefetch (%d split cases; hook once/failure; bank restoration; promote; fixed config)\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-weight-prefetch-") as tmp:
    tmp = Path(tmp)
    source = tmp / "test.cpp"
    source.write_text(harness)
    binary = tmp / "test"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
