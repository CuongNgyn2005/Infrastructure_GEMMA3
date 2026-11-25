#!/usr/bin/env python3
"""RAM-only checks: 32-row poll-hook result read, bit-identical accumulation, lean ZDMA setup source order."""

import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cpu/fpga_host.cpp").read_text(encoding="utf-8")


def definition(signature):
    start = SOURCE.index(signature)
    brace = SOURCE.index("{", start)
    depth = 0
    for pos in range(brace, len(SOURCE)):
        depth += (SOURCE[pos] == "{") - (SOURCE[pos] == "}")
        if depth == 0:
            return SOURCE[start:pos + 1]
    raise AssertionError(signature)


ctx_end = SOURCE.index("} fpga_p2_prefetch_read_ctx_t;")
ctx = SOURCE[SOURCE.rfind("typedef struct {", 0, ctx_end):ctx_end + len("} fpga_p2_prefetch_read_ctx_t;")]
chunked = definition("static bool fpga_p2_prefetch_read_poll_chunked(")
finish = definition("static bool fpga_p2_prefetch_read_finish(")
read_rows = definition("static bool fpga_p2_result_read_rows(")
accumulate = definition("static bool fpga_accumulate_pl_scaled_q8_tile_job(")
dma_copy_one = definition("static bool fpga_dma_copy_one(")
configure = definition("static void fpga_host_offload_configure(")
scheduler = definition("static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(")

# ---- source-order assertions -------------------------------------------------
a_check = accumulate.index("fpga_p2_result_read_check(")
a_rows = accumulate.index("fpga_p2_result_read_rows(job, accum, 0, job.rows)")
a_pub = accumulate.index("fpga_p2_result_read_publish(")
assert a_check < a_rows < a_pub

start = dma_copy_one.index("g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_START;")
for needle in ('zdma_wait_channel_disabled(tag, "before_descriptor")', "zdma_total_byte_clear()",
               "fpga_p2_ddr_sync(src_ddr_off", "zdma_clear_isr_for_new_transfer(tag)"):
    assert 0 <= dma_copy_one.index(needle) < start, needle
word3 = dma_copy_one.index("g_dma->ZDMA_CH_DST_DSCR_WORD3 = 0U;")
fence = dma_copy_one.index("mmio_fence();", word3)
assert word3 < fence < start
# No diagnostic MMIO between the last descriptor word and START.
assert "g_dma->" not in dma_copy_one[word3 + len("g_dma->ZDMA_CH_DST_DSCR_WORD3 = 0U;"):start]
assert "vpu_rd32(" not in dma_copy_one[:start]
# Lean setup must not read post-DMA diagnostics on the success path.
success = dma_copy_one[dma_copy_one.index("const long long t1"):]
assert "ZDMA_CH_TOTAL_BYTE" not in success and "vpu_rd32(" not in success
# The failure path keeps its full diagnostic reads and now prints them.
fail_branch = dma_copy_one[dma_copy_one.index("if (!completed) {"):dma_copy_one.index("const long long t1")]
for needle in ("g_dma->ZDMA_CH_TOTAL_BYTE", "vpu_rd32(REG_STATUS)", "vpu_rd32(REG_PROGRESS)",
               "total_bytes=0x%08x", "total_bytes_after_transfer, post_vpu_status, post_vpu_progress"):
    assert needle in fail_branch, needle

# The chunk size is fixed; the retired switches are no longer read.
assert "static constexpr int P2_HOOK_READ_ROWS = 32;" in SOURCE
for retired in ("FPGA_P2_HOOK_READ_ROWS", "FPGA_ZDMA_LEAN_SETUP", "FPGA_P2_WEIGHT_PREFETCH"):
    assert retired not in SOURCE, retired

# Scheduler: the three counters, chunk finish and phase marks exist only on the weight-prefetch path.
assert "fpga_p2_prefetch_read_finish(read_ctx)" in scheduler
assert "++g_p2_weight_prefetch_partial_reads;" in scheduler
assert "++g_p2_weight_prefetch_late_reads;" in scheduler
assert "const long long cp_top = (g_p2_cycle_phases_enabled && weight_prefetch) ? now_us() : 0;" in scheduler
marks = ["P2CP_PRE", "P2CP_DRAIN", "P2CP_RELEASE", "P2CP_ACT", "P2CP_SUBMIT", "P2CP_ADMIT",
         "P2CP_PREFETCH", "P2CP_TAIL", "P2CP_STORE"]
positions = [scheduler.index(f"p2_cycle_mark({m}, &cp_last);") for m in marks]
assert positions == sorted(positions), positions
assert scheduler.index("p2_cycle_mark(P2CP_STORE, &cp_last);") < scheduler.index("g_p2_cycle_us[P2CP_TOTAL] +=")

# ---- harness 1: mocked check/rows/publish ------------------------------------
PRELUDE = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
struct fpga_stage_totals_t {};
long long g_p2_weight_prefetch_hook_reads = 0;
static constexpr int P2_HOOK_READ_ROWS = 32;
template<class... T> void LOGE(T...) {}
long long now_us() { static long long c = 0; return ++c; }
long long p2_event_now_us() { return 7; }
'''

HARNESS_MOCK = PRELUDE + r'''
struct fpga_tile_job_t { int rows = 0; };
''' + ctx + r'''
struct Range { int b, e; };
std::vector<Range> ranges;
int rows_calls = 0, publish_calls = 0, fail_rows_call = -1;
bool check_ok = true, publish_release = true, publish_ok = true;
long long publish_read_us = -1;
bool fpga_p2_result_read_check(const fpga_tile_job_t &, const std::vector<float> &, bool) { return check_ok; }
bool fpga_p2_result_read_rows(fpga_tile_job_t &, std::vector<float> &, int b, int e) {
    if (rows_calls++ == fail_rows_call) return false;
    ranges.push_back({b, e});
    return true;
}
bool fpga_p2_result_read_publish(fpga_tile_job_t &, fpga_stage_totals_t *, long long us, bool release) {
    ++publish_calls; publish_release = release; publish_read_us = us; return publish_ok;
}
''' + chunked + "\n" + finish + r'''
void reset() {
    ranges.clear(); rows_calls = publish_calls = 0; fail_rows_call = -1;
    check_ok = publish_ok = true; publish_release = true; publish_read_us = -1;
    g_p2_weight_prefetch_hook_reads = 0;
}
int main() {
    int cases = 0;
    fpga_tile_job_t job;
    std::vector<float> accum(1);
    constexpr int chunk = P2_HOOK_READ_ROWS;
    for (int rows : {1, 15, 31, 32, 33, 100, 255, 256}) {
        const int full = (rows + chunk - 1) / chunk;
        for (int k = 0; k <= full + 2; ++k) {
            reset(); job.rows = rows;
            fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
            for (int i = 0; i < k; ++i) assert(fpga_p2_prefetch_read_poll_chunked(&ctx, i));
            const bool done_in_hook = ctx.done;
            assert(done_in_hook == (k >= full));
            assert(ctx.started == (k >= 1));
            assert(g_p2_weight_prefetch_hook_reads == (done_in_hook ? 1 : 0));
            if (ctx.started && !ctx.done) {
                assert(fpga_p2_prefetch_read_finish(ctx));
                assert(ctx.done && ctx.ok);
                assert(g_p2_weight_prefetch_hook_reads == 0);   // finish is a partial read, not a hook read
            }
            if (!ctx.started) {
                assert(ranges.empty() && publish_calls == 0);   // late-read path: nothing consumed
            } else {
                int next = 0;
                for (const Range & r : ranges) { assert(r.b == next && r.e > r.b && r.e <= rows); next = r.e; }
                assert(next == rows);                            // every row once, ascending
                assert(publish_calls == 1 && !publish_release);
                for (size_t i = 0; i + 1 < ranges.size(); ++i) assert(ranges[i].e - ranges[i].b == chunk);
                assert(publish_read_us > 0);
            }
            if (ctx.done) {
                const int calls = rows_calls;
                assert(fpga_p2_prefetch_read_poll_chunked(&ctx, 99) == ctx.ok && rows_calls == calls);
            }
            ++cases;
        }
    }
    // Failure: rows fails on the second chunk.
    reset(); job.rows = 100; fail_rows_call = 1;
    { fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
      assert(fpga_p2_prefetch_read_poll_chunked(&ctx, 0));
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 1));
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 2));
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 3));
      assert(publish_calls == 0 && ctx.done && !ctx.ok && g_p2_weight_prefetch_hook_reads == 0); }
    // Failure: check fails on the first call.
    reset(); check_ok = false;
    { fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 0));
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 1));
      assert(rows_calls == 0 && publish_calls == 0); }
    // Failure: publish fails at the last chunk, and in finish.
    reset(); publish_ok = false; job.rows = 32;
    { fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 0));
      assert(publish_calls == 1 && g_p2_weight_prefetch_hook_reads == 0); }
    reset(); publish_ok = false; job.rows = 100;
    { fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
      assert(fpga_p2_prefetch_read_poll_chunked(&ctx, 0));
      assert(!fpga_p2_prefetch_read_finish(ctx)); assert(publish_calls == 1); }
    // finish rows failure.
    reset(); job.rows = 100; fail_rows_call = 1;
    { fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
      assert(fpga_p2_prefetch_read_poll_chunked(&ctx, 0));
      assert(!fpga_p2_prefetch_read_finish(ctx)); assert(publish_calls == 0 && ctx.done && !ctx.ok); }
    // Precondition failures.
    reset(); job.rows = 100;
    { fpga_p2_prefetch_read_ctx_t ctx{&job, &accum, nullptr, false, true};
      assert(!fpga_p2_prefetch_read_finish(ctx)); }                  // !started
    { fpga_p2_prefetch_read_ctx_t ctx{nullptr, &accum, nullptr, false, true};
      assert(!fpga_p2_prefetch_read_poll_chunked(&ctx, 0)); assert(!fpga_p2_prefetch_read_poll_chunked(nullptr, 0)); }
    std::printf("cases=%d\n", cases);
    return 0;
}
'''

# ---- harness 2: real row reader, bit-identical comparison --------------------
HARNESS_REAL = PRELUDE + r'''
struct fpga_tile_job_t { int rows = 0; int col = 0; uint32_t job_id = 1, tile_id = 1; int bank = 0; };
constexpr uint32_t SPU_OUT_BASE = 0x2000U;
alignas(16) static uint8_t g_spu_out[256 * 16];
void * ddr_ptr(uint32_t, size_t) { return g_spu_out; }
int64_t ddr_read_spu_q16_row(uint32_t, uint16_t * row_id) { *row_id = 0xffffU; return 0; }
''' + read_rows + r'''
int main() {
    std::mt19937_64 rng(0x5eed1234ULL);
    for (int rows : {1, 15, 16, 17, 100, 255, 256}) {
        for (int bad_row : {-1, rows / 2}) {
            for (int r = 0; r < 256; ++r) {
                const int64_t q = (int64_t) rng() >> (rng() % 24);
                uint64_t low = (uint64_t) (uint16_t) r | (((uint64_t) q & 0xFFFFFFFFFFFFULL) << 16);
                uint64_t high = ((uint64_t) q >> 48) & 0xFFFFULL;
                if (r == bad_row) low = (low & ~0xFFFFULL) | 0x7777ULL;   // corrupt row_id
                std::memcpy(g_spu_out + r * 16, &low, 8);
                std::memcpy(g_spu_out + r * 16 + 8, &high, 8);
            }
            fpga_tile_job_t job; job.rows = rows; job.col = 1;
            std::vector<float> init(2 * (size_t) rows);
            for (float & f : init) f = (float) ((int) (rng() % 2001) - 1000) / 7.0f;
            std::vector<float> full = init;
            const bool full_ok = fpga_p2_result_read_rows(job, full, 0, rows);
            assert(full_ok == (bad_row < 0));
            for (int chunk : {P2_HOOK_READ_ROWS}) {
                std::vector<float> got = init;
                bool ok = true;
                for (int b = 0; b < rows && ok; b += chunk) ok = fpga_p2_result_read_rows(job, got, b, std::min(rows, b + chunk));
                assert(ok == full_ok);
                if (full_ok) assert(std::memcmp(full.data(), got.data(), full.size() * sizeof(float)) == 0);
            }
            // Column 0 must be untouched.
            assert(std::memcmp(init.data(), full.data(), (size_t) rows * sizeof(float)) == 0);
        }
    }
    std::printf("bit-identical rows ok\n");
    return 0;
}
'''


def build_and_run(name, source):
    with tempfile.TemporaryDirectory(prefix=f"fpga-{name}-") as tmp:
        tmp = Path(tmp)
        src = tmp / "test.cpp"
        src.write_text(source)
        binary = tmp / "test"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(src), "-o", str(binary)], check=True)
        out = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.strip()
        return out


mock_out = build_and_run("hook-chunk-mock", HARNESS_MOCK)
real_out = build_and_run("hook-chunk-real", HARNESS_REAL)
print(f"PASS test-fpga-hook-chunk-read ({mock_out}; {real_out}; lean setup source order; scheduler marks)")
