#!/usr/bin/env python3
"""Compare the frozen host formatter with production snapshot/reporting code; RAM-only."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "ggml/src/ggml-cpu"
HOST = (CPU / "fpga_host.cpp").read_text()
LOGGER = (CPU / "fpga_log.cpp").read_text()

# Actual pre-relocation slice from host SHA256 A3CA70D33E738DF5BA46D26C43555D46F1C53FAE722084691492AC78186AA5B2.
# Preserve this reference, including its strings and math; do not regenerate it from the new formatter.
OLD_FORMATTER = r'''    if (g_token_timing_enabled) {
        fpga_log_line(
            true, "TIMING", force_flush,
            "graph_seq=%d next_graph_seq=%d ubatch_tokens=%lld scope=%s reason=%s matmuls=%lld vpu_runs=%lld "
            "graph_wall_ms=%.3f device_first_start_to_last_output_ready_ms=%.3f fpga_matmul_wall_sum_ms=%.3f "
            "host_to_ip_dma_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f "
            "ping_h2ip_ms=%.3f pong_h2ip_ms=%.3f ping_jobs=%lld pong_jobs=%lld "
            "ip_compute_sum_ms=%.3f ping_compute_ms=%.3f pong_compute_ms=%.3f "
            "ip_to_host_dma_ms=%.3f ping_ip2host_ms=%.3f pong_ip2host_ms=%.3f "
            "host_result_read_ms=%.3f ping_host_read_ms=%.3f pong_host_read_ms=%.3f token_read_ms=%.3f "
            "prep_ms=%.3f host_accum_ms=%.3f act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu "
            "measurement_scope=host_ddr_to_ip_ip_compute_ip_to_host_ddr internal_pl_interconnect=not_observable_from_host",
            g_token_timing.graph_seq, next_graph_seq, (long long) ubatch_tokens, scope, reason ? reason : "unknown",
            g_token_timing.matmuls, g_token_timing.vpu_runs, (double) token_wall_us / 1000.0,
            (double) device_span_us / 1000.0, (double) g_token_timing.matmul_wall_us / 1000.0,
            (double) h2ip_dma_us / 1000.0, (double) g_token_timing.act_dma_us / 1000.0,
            (double) g_token_timing.weight_dma_us / 1000.0, (double) g_token_timing.scale_dma_us / 1000.0,
            (double) g_token_timing.bank_h2ip_us[0] / 1000.0,
            (double) g_token_timing.bank_h2ip_us[1] / 1000.0, g_token_timing.bank_jobs[0],
            g_token_timing.bank_jobs[1], fpga_compute_report_us(g_token_timing.ip_compute_us) / 1000.0,
            fpga_compute_report_us(g_token_timing.bank_compute_us[0]) / 1000.0,
            fpga_compute_report_us(g_token_timing.bank_compute_us[1]) / 1000.0,
            (double) g_token_timing.ip2host_dma_us / 1000.0,
            (double) g_token_timing.bank_ip2host_us[0] / 1000.0,
            (double) g_token_timing.bank_ip2host_us[1] / 1000.0,
            (double) g_token_timing.host_read_us / 1000.0,
            (double) g_token_timing.bank_host_read_us[0] / 1000.0,
            (double) g_token_timing.bank_host_read_us[1] / 1000.0, (double) token_read_us / 1000.0,
            (double) g_token_timing.prep_us / 1000.0, (double) g_token_timing.host_accum_us / 1000.0,
            g_token_timing.activation_bytes, g_token_timing.weight_bytes, g_token_timing.scale_bytes,
            g_token_timing.result_bytes);
    }

    if (g_bottleneck_summary_enabled) {
        const long long prep_known_us = g_token_timing.prep_weight_select_us +
                                        g_token_timing.prep_direct_weight_pack_us +
                                        g_token_timing.prep_scale_pack_us + g_token_timing.prep_act_pack_us;
        const long long prep_other_us = std::max(0LL, g_token_timing.prep_us - prep_known_us);
        const long long outside_matmul_us = std::max(0LL, token_wall_us - g_token_timing.matmul_wall_us);
        const double device_noncompute_us = fpga_compute_report_us(std::max(0LL, device_span_us - g_token_timing.ip_compute_us));
        const double compute_util_pct = !g_compute_watch_enabled ? std::numeric_limits<double>::quiet_NaN() :
                                        device_span_us > 0 ?
                                            100.0 * (double) g_token_timing.ip_compute_us / (double) device_span_us :
                                            0.0;
        fpga_log_line(
            true, "BREAKDOWN", force_flush,
            "graph_seq=%d next_graph_seq=%d scope=%s graph_wall_ms=%.3f matmul_wall_ms=%.3f "
            "outside_matmul_ms=%.3f device_span_ms=%.3f ip_compute_ms=%.3f device_noncompute_ms=%.3f "
            "compute_util_pct=%.2f prep_total_ms=%.3f prep_weight_select_ms=%.3f "
            "prep_direct_weight_pack_ms=%.3f prep_scale_pack_ms=%.3f prep_act_pack_ms=%.3f prep_other_ms=%.3f "
            "handoffs=%lld prep_overlap_ms=%.3f prep_late_jobs=%lld prep_late_ms=%.3f prep_headroom_ms=%.3f "
            "output_ready_to_next_launch_ms=%.3f retire_to_next_launch_ms=%.3f",
            g_token_timing.graph_seq, next_graph_seq, scope, (double) token_wall_us / 1000.0,
            (double) g_token_timing.matmul_wall_us / 1000.0, (double) outside_matmul_us / 1000.0,
            (double) device_span_us / 1000.0, fpga_compute_report_us(g_token_timing.ip_compute_us) / 1000.0,
            device_noncompute_us / 1000.0, compute_util_pct, (double) g_token_timing.prep_us / 1000.0,
            (double) g_token_timing.prep_weight_select_us / 1000.0,
            (double) g_token_timing.prep_direct_weight_pack_us / 1000.0,
            (double) g_token_timing.prep_scale_pack_us / 1000.0,
            (double) g_token_timing.prep_act_pack_us / 1000.0, (double) prep_other_us / 1000.0,
            g_token_timing.scheduler_handoffs, (double) g_token_timing.scheduler_prepare_overlap_us / 1000.0,
            g_token_timing.scheduler_prepare_late_jobs, (double) g_token_timing.scheduler_prepare_late_us / 1000.0,
            (double) g_token_timing.scheduler_prepare_headroom_us / 1000.0,
            (double) g_token_timing.scheduler_output_to_launch_us / 1000.0,
            (double) g_token_timing.scheduler_retire_to_launch_us / 1000.0);

        if (sampled_detail) {
            fpga_log_line(
                true, "GEMV_TIME_BY_LAYER_TYPE", force_flush,
                "graph_seq=%d scope=%s "
                "attention_matmuls=%lld attention_tile_jobs=%lld attention_wall_ms=%.3f attention_host_prepare_ms=%.3f attention_fpga_compute_ms=%.3f attention_dma_ms=%.3f "
                "ffn_gate_matmuls=%lld ffn_gate_tile_jobs=%lld ffn_gate_wall_ms=%.3f ffn_gate_host_prepare_ms=%.3f ffn_gate_fpga_compute_ms=%.3f ffn_gate_dma_ms=%.3f "
                "ffn_up_matmuls=%lld ffn_up_tile_jobs=%lld ffn_up_wall_ms=%.3f ffn_up_host_prepare_ms=%.3f ffn_up_fpga_compute_ms=%.3f ffn_up_dma_ms=%.3f "
                "ffn_down_matmuls=%lld ffn_down_tile_jobs=%lld ffn_down_wall_ms=%.3f ffn_down_host_prepare_ms=%.3f ffn_down_fpga_compute_ms=%.3f ffn_down_dma_ms=%.3f "
                "other_matmuls=%lld other_tile_jobs=%lld other_wall_ms=%.3f other_host_prepare_ms=%.3f other_fpga_compute_ms=%.3f other_dma_ms=%.3f",
                g_token_timing.graph_seq, scope,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_ATTN], g_token_timing.category_runs[FPGA_BOTTLENECK_ATTN],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                fpga_compute_report_us(g_token_timing.category_compute_us[FPGA_BOTTLENECK_ATTN]) / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_FFN_GATE], g_token_timing.category_runs[FPGA_BOTTLENECK_FFN_GATE],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                fpga_compute_report_us(g_token_timing.category_compute_us[FPGA_BOTTLENECK_FFN_GATE]) / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_FFN_UP], g_token_timing.category_runs[FPGA_BOTTLENECK_FFN_UP],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                fpga_compute_report_us(g_token_timing.category_compute_us[FPGA_BOTTLENECK_FFN_UP]) / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_FFN_DOWN], g_token_timing.category_runs[FPGA_BOTTLENECK_FFN_DOWN],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                fpga_compute_report_us(g_token_timing.category_compute_us[FPGA_BOTTLENECK_FFN_DOWN]) / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_OTHER], g_token_timing.category_runs[FPGA_BOTTLENECK_OTHER],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_OTHER] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_OTHER] / 1000.0,
                fpga_compute_report_us(g_token_timing.category_compute_us[FPGA_BOTTLENECK_OTHER]) / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_OTHER] / 1000.0);

            const double zdma_avg_us = g_token_timing.zdma_descriptors > 0 ?
                                           (double) g_token_timing.zdma_elapsed_us / (double) g_token_timing.zdma_descriptors :
                                           0.0;
            fpga_log_line(
                true, "ZDMA", force_flush,
                "graph_seq=%d scope=%s descriptors=%lld bytes=%zu elapsed_ms=%.3f avg_descriptor_us=%.3f polls=%lld "
                "zero_poll_descriptors=%lld saw_enabled_descriptors=%lld act_desc=%lld weight_desc=%lld scale_desc=%lld "
                "result_desc=%lld other_desc=%lld max_descriptor_bytes=%zu",
                g_token_timing.graph_seq, scope, g_token_timing.zdma_descriptors, g_token_timing.zdma_bytes,
                (double) g_token_timing.zdma_elapsed_us / 1000.0, zdma_avg_us, g_token_timing.zdma_polls,
                g_token_timing.zdma_zero_poll_descriptors, g_token_timing.zdma_saw_enabled_descriptors,
                g_token_timing.zdma_act_descriptors, g_token_timing.zdma_weight_descriptors,
                g_token_timing.zdma_scale_descriptors, g_token_timing.zdma_result_descriptors,
                g_token_timing.zdma_other_descriptors, g_zdma_max_transfer_bytes);
        }
    }
'''
assert hashlib.sha256(OLD_FORMATTER.replace("\n", "\r\n").encode()).hexdigest() == (
    "c4a500419b7f92d327fc0dff6ce257965090959236afdc88fa1c309174d5422b"
)


def definition(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


emit_start = HOST.index("static bool fpga_token_timing_emit(int next_graph_seq,",
                        HOST.index("static void fpga_token_timing_reset(void) {"))
snapshot_start = HOST.index("    if (g_token_timing_enabled || g_bottleneck_summary_enabled) {", emit_start)
snapshot_end = HOST.index("    fpga_token_timing_reset();", snapshot_start)
SNAPSHOT = HOST[snapshot_start:snapshot_end]
assert HOST[emit_start:snapshot_start].rstrip().endswith("g_summary_detail_after_error = false;")
assert SNAPSHOT.count("fpga_log_token_timing(report);") == 1
for forbidden in ("now_us(", "monotonic_now_us(", "new ", "malloc(", "vpu_rd", "vpu_wr", "pthread_"):
    assert forbidden not in SNAPSHOT
struct_end = HOST.index("} fpga_token_timing_t;") + len("} fpga_token_timing_t;")
struct_start = HOST.rfind("typedef struct {", 0, struct_end)
TOKEN_TYPE = HOST[struct_start:struct_end]
CATEGORY_TYPE = HOST[HOST.index("enum fpga_bottleneck_category_t : int {"):
                     HOST.index("};", HOST.index("enum fpga_bottleneck_category_t : int {")) + 2]
fields = re.findall(r"^\s*(bool|int|int64_t|long long|size_t)\s+(\w+)(\[[^\]]+\])?;", TOKEN_TYPE, re.M)
assert fields
harness = r'''
#include "fpga_log.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static FILE * sink = nullptr;
static int sink_accesses = 0;
static int snapshot_calls = 0;
static std::vector<long> flush_positions;
FILE * fpga_log_fp() { ++sink_accesses; assert(sink); return sink; }
static int record_flush(FILE * fp) {
    flush_positions.push_back(std::ftell(fp));
    return std::fflush(fp);
}
'''
for name in ("g_flush_every", "g_pending_lines"):
    harness += re.search(r"^int " + name + r"\s*=.*;", LOGGER, re.M).group() + "\n"
harness += "#define fflush record_flush\n"
harness += definition(LOGGER, "void fpga_log_set_flush_every(") + "\n"
harness += definition(LOGGER, "void fpga_log_finish_line(") + "\n"
harness += definition(LOGGER, "void fpga_log_vline(") + "\n#undef fflush\n"
harness += definition(LOGGER, "void fpga_log_token_line(") + "\n"
harness += definition(LOGGER, "void fpga_log_token_timing(") + "\n"
harness += CATEGORY_TYPE + "\n" + TOKEN_TYPE + r'''
static fpga_token_timing_t g_token_timing = {};
static bool g_token_timing_enabled, g_bottleneck_summary_enabled, g_compute_watch_enabled;
static bool g_summary_detail_after_error = false;
static size_t g_zdma_max_transfer_bytes = 0;
'''
harness += definition(HOST, "static void fpga_log_line(bool enabled, const char * tag, bool force_flush, const char * fmt, ...) {") + "\n"
harness += definition(HOST, "static double fpga_compute_report_us(long long us) {") + "\n"
parameters = """int next_graph_seq, long long ubatch_tokens, const char * scope, const char * reason,
                          long long token_wall_us, long long h2ip_dma_us, long long token_read_us,
                          long long device_span_us, bool sampled_detail, bool force_flush"""
harness += "static void legacy_format(" + parameters + ") {\n" + OLD_FORMATTER + "}\n"
harness += r'''
static void observed_token_report(const fpga_token_timing_log_t & data) {
    ++snapshot_calls;
    fpga_log_token_timing(data);
}
#define fpga_log_token_timing observed_token_report
'''
harness += "static void current_format(" + parameters + ") {\n" + SNAPSHOT + "}\n"
harness += "#undef fpga_log_token_timing\nstatic void fill_token(int mode) {\n    g_token_timing = {};\n"
for index, (kind, name, array) in enumerate(fields, 1):
    if array:
        count = "5" if "CATEGORY" in array else "2"
        harness += (f"    for (int i = 0; i < {count}; ++i) g_token_timing.{name}[i] = "
                    f"mode == 0 ? 0 : (mode == 2 ? (1LL << 40) : 0) + {index} * 1009LL + i * 313LL;\n")
    else:
        value = "mode != 0" if kind == "bool" else f"mode == 0 ? 0 : {index} * 1009LL"
        if kind in ("long long", "int64_t", "size_t"):
            value = f"mode == 0 ? 0 : (mode == 2 ? (1LL << 40) : 0) + {index} * 1009LL"
        harness += f"    g_token_timing.{name} = static_cast<{kind}>({value});\n"
harness += r'''
    if (mode == 1) {
        g_token_timing.prep_us = 900001;
        g_token_timing.matmul_wall_us = 1100003;
        g_token_timing.ip_compute_us = 31007;
    }
    if (mode == 3) {
        g_token_timing.prep_us = 1;
        g_token_timing.matmul_wall_us = 900000;
        g_token_timing.ip_compute_us = 50000;
        g_token_timing.zdma_descriptors = 0;
    }
}
struct Capture {
    std::string bytes;
    std::vector<long> flushes;
    int pending;
    int accesses;
};
static Capture capture(bool legacy, int mode, bool sampled, bool forced, int interval, int pending) {
    sink = std::tmpfile();
    assert(sink);
    fpga_log_set_flush_every(interval);
    g_pending_lines = pending;
    flush_positions.clear();
    sink_accesses = snapshot_calls = 0;
    g_summary_detail_after_error = true;
    const long long wall = mode == 0 ? 0 : mode == 2 ? (1LL << 42) + 123 : mode == 3 ? 1 : 1234567;
    const long long span = mode == 0 || mode == 3 ? 0 : mode == 2 ? (1LL << 41) + 789 : 456789;
    const char * reason = mode == 0 ? nullptr : mode == 1 ? "sequence_advance" : "";
    const char * scope = mode == 0 ? "incomplete" : mode == 3 ? "prefill_or_ubatch" : "decode_token";
    const long long tokens = mode == 0 ? 0 : mode == 3 ? 19 : 1;
    const long long input = g_token_timing.act_dma_us + g_token_timing.weight_dma_us + g_token_timing.scale_dma_us;
    const long long output = g_token_timing.ip2host_dma_us + g_token_timing.host_read_us;
    if (legacy) legacy_format(-17, tokens, scope, reason, wall, input, output, span, sampled, forced);
    else current_format(-17, tokens, scope, reason, wall, input, output, span, sampled, forced);
    assert(g_summary_detail_after_error);
    assert(snapshot_calls == (legacy ? 0 : int(g_token_timing_enabled || g_bottleneck_summary_enabled)));
    const long size = std::ftell(sink);
    assert(size >= 0);
    Capture result = {std::string(static_cast<size_t>(size), char(0)), flush_positions, g_pending_lines, sink_accesses};
    std::rewind(sink);
    if (size) assert(std::fread(&result.bytes[0], 1, result.bytes.size(), sink) == result.bytes.size());
    assert(std::fclose(sink) == 0);
    sink = nullptr;
    return result;
}
int main() {
    static_assert(sizeof(size_t) >= 8, "large-byte-count coverage requires a 64-bit host");
    int cases = 0;
    for (int mode = 0; mode < 4; ++mode) {
        fill_token(mode);
        g_zdma_max_transfer_bytes = mode == 2 ? (size_t(1) << 35) + 16 : 262144;
        for (bool timing : {false, true}) for (bool breakdown : {false, true})
        for (bool watch : {false, true}) for (bool sampled : {false, true})
        for (bool forced : {false, true}) for (int interval : {1, 3, 256})
        for (int pending : {0, interval - 1}) {
            g_token_timing_enabled = timing;
            g_bottleneck_summary_enabled = breakdown;
            g_compute_watch_enabled = watch;
            const Capture expected = capture(true, mode, sampled, forced, interval, pending);
            const Capture actual = capture(false, mode, sampled, forced, interval, pending);
            assert(actual.bytes == expected.bytes);
            assert(actual.flushes == expected.flushes && actual.pending == expected.pending);
            assert(actual.accesses == expected.accesses);
            const size_t lines = size_t(timing) + size_t(breakdown) * (sampled ? 3U : 1U);
            assert(size_t(std::count(actual.bytes.begin(), actual.bytes.end(), '\n')) == lines);
            if (!timing && !breakdown) assert(actual.bytes.empty() && actual.accesses == 0);
            if (forced || interval == 1) assert(actual.flushes.size() == lines);
            if (timing && mode == 0) assert(actual.bytes.find("reason=unknown") != std::string::npos);
            if (!watch && lines) assert(actual.bytes.find("nan") != std::string::npos);
            if (breakdown && mode == 3) {
                assert(actual.bytes.find("outside_matmul_ms=0.000") != std::string::npos);
                assert(actual.bytes.find("prep_other_ms=0.000") != std::string::npos);
            }
            ++cases;
        }
    }
    std::printf("PASS: %d token report cases; frozen old output versus production snapshot/formatter; "
                "gate combinations, byte order/precision, NaNs, large counters, flush sequence and pending lines\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-token-log-") as directory:
    source = Path(directory) / "token_log.cpp"
    binary = Path(directory) / ("token_log.exe" if os.name == "nt" else "token_log")
    source.write_text(harness)
    sanitizer = [] if os.name == "nt" else ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    *sanitizer, "-I" + str(CPU), str(source), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=60)
