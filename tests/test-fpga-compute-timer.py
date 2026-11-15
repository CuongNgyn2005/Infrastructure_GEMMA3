#!/usr/bin/env python3
"""RAM-only watcher policy, lifecycle, production drain and reporting checks."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()
start = source.index("static bool g_compute_watch_enabled = true;")
end = source.index("static bool fpga_submit_q8_tile_job(", start)
helper = source[start:end]
# Guard the integration boundary: arm before START, notify after START.
submit_end = source.index("static bool fpga_wait_and_drain_q8_tile_job(", end)
submit = source[end:submit_end]
launch = submit.index("job.ip_start_us = now_us();")
assert submit.index("fpga_compute_watch_arm(job)") < launch
capture = submit.index("fpga_compute_watch_launch(compute_start_us)", launch)
between = submit[launch:capture]
assert "vpu_wr32(REG_CTRL, CTRL_START);" in between
assert "LOG" not in between and "fpga_dma" not in between
assert "job.ip_compute_us = job.measured_compute_us;" in source
launch_body = submit[submit.index("    if (!fpga_compute_watch_arm(job))"):submit.index("    job.event_launch_us =")]
drain = source[submit_end:source.index("static bool fpga_release_pl_scaled_q8_tile_job(", submit_end)]
env_helpers = source[source.index("static bool env_flag_enabled("):source.index("static bool should_log_detail_run(")]
perf = source[source.index('extern "C" int fpga_perf_decode_get('):source.index('extern "C" void fpga_advance_sequence_position(')]
sampling = (root / "src/llama-sampling.cpp").read_text()
summary = sampling[sampling.index("void llama_perf_sampler_print("):sampling.index("void llama_perf_sampler_reset(")]
init = source[source.index("int fpga_init(void)"):source.index("void fpga_cleanup(void)")]
assert init.index("if (g_fpga_init_complete)") < init.index("fpga_compute_watch_configure()")
assert init.index("fpga_compute_watch_configure()") < init.index("map_weight_path_bench_vpu_ddr()")
cleanup = source[source.index("void fpga_cleanup(void)"):source.index('extern "C" int fpga_perf_decode_get(')]
assert cleanup.index("fpga_compute_watch_shutdown()") < cleanup.index("munmap(")
assert "g_fpga_init_complete = false;" in cleanup

harness = r'''
#include <cassert>
#include <cstdint>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <initializer_list>
#include <pthread.h>
#include <sched.h>
#include "fpga_host.h"
struct fpga_tile_job_t {
    long long ip_start_us, measured_compute_us, ip_compute_us, dma_result_us;
    long long event_launch_us, event_vpu_done_us, event_spu_finality_us;
    unsigned job_id, tile_id, vpu_status;
    int bank, rows, group_blocks, group_beats, graph_seq;
    unsigned spu_stream_count_before, spu_stream_done_before, spu_stream_out_before;
    unsigned spu_stream_drop_before, spu_stream_error_before;
    size_t spu_result_bytes;
    bool output_dma_ready, hardware_released, result_consumed;
};
struct fpga_stage_totals_t {
    long long last_ip_output_ready_mono_us, dma_result_us, ip_compute_us;
    long long bank_compute_us[2], bank_ip2host_us[2];
};
std::atomic<long long> clock_us{0};
std::atomic<bool> hardware_done{true};
long long g_ip_timeout_us = 5000000;
int failure, calls, starts, creates;
bool fail_create = false, final_visible = false;
std::string order;
const char * watch_env = nullptr;
const char * test_getenv(const char * name) {
    assert(std::strcmp(name, "FPGA_COMPUTE_WATCH") == 0);
    return watch_env;
}
int test_pthread_create(pthread_t * thread, const pthread_attr_t * attr,
                        void * (*fn)(void *), void * arg) {
    ++creates;
    return fail_create ? 1 : pthread_create(thread, attr, fn, arg);
}
#define getenv test_getenv
#define pthread_create test_pthread_create
void LOGE(const char *, ...) __attribute__((format(gnu_printf, 1, 2)));
void LOGE(const char *, ...) {}
void LOGINIT(const char *, ...) __attribute__((format(gnu_printf, 1, 2)));
void LOGINIT(const char *, ...) {}
void LOGIP(const char *, ...) __attribute__((format(gnu_printf, 1, 2)));
void LOGIP(const char *, ...) {}
void fpga_log_line(bool, const char *, bool, const char *, ...) __attribute__((format(gnu_printf, 4, 5)));
void fpga_log_line(bool, const char *, bool, const char *, ...) {}
long long monotonic_now_us() { return clock_us.load(); }
long long now_us() { return clock_us.load(); }
long long p2_event_now_us() { return clock_us.load(); }
bool wait_vpu_done(uint32_t * status) {
    while (!hardware_done.load()) { sched_yield(); }
    ++calls; order += 'V';
    clock_us += 20;
    *status = 1;
    return failure != 1;
}
bool wait_spu_stream_outputs(const fpga_tile_job_t &) {
    ++calls; order += 'S';
    clock_us += 7;
    final_visible = failure != 2;
    return failure != 2;
}
constexpr unsigned REG_CTRL = 0, CTRL_START = 1, REG_PROGRESS = 2;
constexpr unsigned REG_SPU_STREAM_DROP = 3, REG_SPU_STREAM_ERROR = 4, REG_SPU_STREAM_OUT = 5;
constexpr unsigned FPGA_SLOT_FREE = 0, FPGA_SLOT_DMA_DRAINING = 1, FPGA_SLOT_HOST_CONSUMING = 2;
constexpr unsigned SPU_OUT_BASE = 0x340000;
bool g_pingpong_timing_enabled = true, g_ip_timing_enabled = true;
long long g_pl_scale_stream_drops = 0, g_pl_scale_stream_errors = 0;
void vpu_wr32(unsigned reg, unsigned val) { assert(reg == REG_CTRL && val == CTRL_START); ++starts; }
unsigned vpu_rd32(unsigned) { return 1; }
void mmio_fence() {}
void p2_trace_first_tile(const fpga_tile_job_t &, const char *, const char *) {}
void p2_event_trace(const fpga_tile_job_t &, const char *, long long, const char *, long long) {}
const char * p2_bank_label(int) { return "test"; }
bool should_log_detail_run(long long) { return true; }
bool vpu_verify_done_job(const fpga_tile_job_t &, unsigned) { order += 'J'; return failure != 3; }
bool fpga_write_post_spu_descriptor(const fpga_tile_job_t &, unsigned, unsigned state,
                                    unsigned, const char *, bool) {
    assert(final_visible);
    order += state == FPGA_SLOT_DMA_DRAINING ? 'D' : 'H';
    return failure != (state == FPGA_SLOT_DMA_DRAINING ? 4 : 6);
}
void vpu_select_banks(int, int) { order += 'B'; }
bool fpga_dma_read_from_ip(unsigned addr, size_t, const char *) {
    assert(final_visible && addr == SPU_OUT_BASE);
    order += 'R'; clock_us += 10; return failure != 5;
}
fpga_perf_decode_data g_fpga_perf_decode{};
bool g_spu_q8_scale_stream_supported = true;
int g_contract_check_limit = 0;
long long g_fpga_count = 0, g_reject_count = 0, g_p2_resident_tile_count = 0;
long long g_p2_residency_hits = 0, g_p2_residency_misses = 0;
std::array<int, 2> g_p2_resident_tiles{};
std::atomic<long long> g_q8_unavailable_cpu_fallback_calls{0};
struct llama_sampler;
'''
harness += env_helpers + helper
harness += "static bool submit_start(fpga_tile_job_t & job) {\n" + launch_body + "return true;\n}\n"
harness += drain + perf + summary
harness += r'''
int main() {
    assert(fpga_compute_watch_configure() && g_compute_watch_enabled); // unset default
    for (const char * value : {"", "1", "true", "TRUE", "yes", "YES", "on", "ON"}) {
        watch_env = value;
        assert(fpga_compute_watch_configure() && g_compute_watch_enabled);
    }
    for (const char * value : {"0", "false", "FALSE", "no", "NO", "off", "OFF"}) {
        watch_env = value;
        assert(fpga_compute_watch_configure() && !g_compute_watch_enabled);
    }
    for (const char * value : {"2", "-1", "falsee", "True", " 0", "0 ", "1junk"}) {
        watch_env = value;
        assert(!fpga_compute_watch_configure() && !g_compute_watch_enabled);
    }
    watch_env = nullptr;
    assert(fpga_compute_watch_configure() && g_compute_watch_enabled);
    fail_create = true;
    fpga_tile_job_t create_failure{};
    assert(!submit_start(create_failure) && starts == 0 && !g_compute_watch_created);
    fail_create = false;
    for (int iteration = 0; iteration < 500; ++iteration) {
        const long long preparation = iteration * 1000;
        clock_us = 100 + preparation;
        fpga_tile_job_t job{};
        job.job_id = iteration + 1;
        failure = 0; calls = 0;
        hardware_done = false;
        assert(fpga_compute_watch_arm(job));
        fpga_compute_watch_launch(clock_us.load());
        // Main can continue CPU work while the watcher awaits hardware.
        assert(g_compute_watch_phase.load() == 2);
        hardware_done = true;
        while (g_compute_watch_phase.load() != 3) { sched_yield(); }
        clock_us += 500000; // main processes results long after completion
        assert(fpga_compute_watch_collect(job));
        assert(calls == 2);
        assert(job.measured_compute_us == 27); // includes SPU tail
    }
    for (int fail : {1, 2}) {
        clock_us = 100; failure = fail; calls = 0;
        fpga_tile_job_t job{};
        job.job_id = 42;
        job.measured_compute_us = 999;
        assert(fpga_compute_watch_arm(job));
        fpga_compute_watch_launch(100);
        assert(!fpga_compute_watch_collect(job));
        assert(job.measured_compute_us == 0); // no stale successful sample
        assert(calls == fail);
    }
    failure = 0; clock_us = 100;
    fpga_tile_job_t invalid{};
    invalid.ip_start_us = 127;
    assert(!fpga_measure_q8_compute_completion(invalid));
    assert(invalid.measured_compute_us == 0);
    fpga_compute_watch_shutdown();
    // Restart and cancel an armed job before START; shutdown must not hang.
    assert(fpga_compute_watch_arm(invalid));
    fpga_compute_watch_shutdown();
    // Shutdown also joins a launched job before mappings could be removed.
    clock_us = 100;
    assert(fpga_compute_watch_arm(invalid));
    fpga_compute_watch_launch(100);
    fpga_compute_watch_shutdown();

    // Repeated ON/OFF lifetimes exercise the actual launch and drain bodies.
    for (int lifetime = 0; lifetime < 20; ++lifetime) {
        const bool enabled = lifetime % 2 != 0;
        watch_env = enabled ? "1" : "0";
        assert(fpga_compute_watch_configure());
        const int before_creates = creates;
        clock_us = 100; failure = 0; calls = 0; starts = 0; final_visible = false; order.clear();
        fpga_tile_job_t job{};
        job.job_id = lifetime + 1;
        job.measured_compute_us = 999; // stale slot timing must not leak in OFF
        hardware_done = false;
        assert(submit_start(job) && starts == 1);
        hardware_done = true;
        if (enabled) {
            while (g_compute_watch_phase.load() != 3) { sched_yield(); }
            clock_us += 500000;
        } else {
            assert(!g_compute_watch_created && g_compute_watch_phase.load() == 0);
            assert(creates == before_creates && calls == 0);
        }
        fpga_stage_totals_t totals{};
        assert(fpga_wait_and_drain_q8_tile_job(job, &totals, "test", 0, 32, 1, 1, 1));
        assert(order == (enabled ? "VSVJSDBRH" : "VJSDBRH"));
        assert(calls == (enabled ? 4 : 2) && job.output_dma_ready);
        assert(job.ip_compute_us == (enabled ? 27 : 0));
        assert(totals.ip_compute_us == job.ip_compute_us && totals.dma_result_us == 10);
        assert(enabled ? fpga_compute_report_us(job.ip_compute_us) == 27 :
                         std::isnan(fpga_compute_report_us(job.ip_compute_us)));
        fpga_compute_watch_shutdown();
        fpga_compute_watch_shutdown(); // idempotent, OFF never creates a thread
        assert(!g_compute_watch_created && !g_compute_watch_pending && g_compute_watch_phase.load() == 0);
    }
    watch_env = "0";
    assert(fpga_compute_watch_configure());
    for (int fail = 1; fail <= 6; ++fail) {
        failure = fail; calls = 0; order.clear(); final_visible = false;
        fpga_tile_job_t job{};
        assert(!fpga_wait_and_drain_q8_tile_job(job, nullptr, "test", 0, 32, 1, 1, 1));
        assert(!job.output_dma_ready && !g_compute_watch_created);
        if (fail <= 4) assert(order.find('R') == std::string::npos);
    }
    for (int ownership = 0; ownership < 3; ++ownership) {
        fpga_tile_job_t job{};
        job.output_dma_ready = ownership == 0;
        job.hardware_released = ownership == 1;
        job.result_consumed = ownership == 2;
        calls = 0;
        assert(!fpga_wait_and_drain_q8_tile_job(job, nullptr, "test", 0, 32, 1, 1, 1));
        assert(calls == 0);
    }
    fpga_perf_decode_data data{};
    assert(!fpga_perf_decode_get(nullptr));
    assert(!fpga_perf_decode_get(&data)); // no decode data
    g_fpga_perf_decode.decode_tokens = 2;
    g_fpga_perf_decode.decode_wall_us = 2000000;
    g_fpga_perf_decode.ip_compute_us = 100000; // must stay unavailable in OFF even if nonzero
    g_fpga_perf_decode.h2ip_dma_us = 200000;
    g_fpga_perf_decode.ip2host_dma_us = 10000;
    assert(fpga_perf_decode_get(&data) && !data.ip_compute_timing_valid && data.ip_compute_us == -1);
    puts("SUMMARY_OFF");
    llama_perf_sampler_print(nullptr);
    watch_env = "1";
    assert(fpga_compute_watch_configure());
    assert(fpga_perf_decode_get(&data) && data.ip_compute_timing_valid && data.ip_compute_us == 100000);
    puts("SUMMARY_ON");
    llama_perf_sampler_print(nullptr);
    g_contract_check_limit = 1;
    assert(fpga_perf_decode_get(&data) && !data.ip_compute_timing_valid);
    g_contract_check_limit = 0; g_spu_q8_scale_stream_supported = false;
    assert(fpga_perf_decode_get(&data) && !data.ip_compute_timing_valid);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    cpp = Path(tmp) / "timer.cpp"
    exe = Path(tmp) / ("timer.exe" if os.name == "nt" else "timer")
    cpp.write_text(harness)
    sanitizer = [] if os.name == "nt" else ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    *sanitizer, "-pthread", "-DUSE_FPGA", "-I" + str(root / "ggml/src/ggml-cpu"),
                    str(cpp), "-o", str(exe)], check=True, timeout=60)
    output = subprocess.run([str(exe)], check=True, timeout=30, capture_output=True, text=True).stdout
    off, on = output.split("SUMMARY_OFF\n")[1].split("SUMMARY_ON\n")
    assert off.count("not measured") == 2 and "20.00 tokens/s" not in off
    assert "20.00 tokens/s" in on and "6.45 tokens/s" in on and "not measured" not in on
print("PASS: boolean/default/invalid policy; 500 asynchronous jobs; 20 ON/OFF lifetimes; "
      "production launch/drain, finality/ownership/error paths; unavailable timing and summary guards")
