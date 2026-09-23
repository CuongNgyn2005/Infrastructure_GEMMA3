#!/usr/bin/env python3
"""Exercise the production timing helper with a deterministic completion clock."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()
start = source.index("static bool fpga_measure_q8_compute_completion(")
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
assert "FPGA_IP_COMPUTE_TIMING" not in source

harness = r'''
#include <cassert>
#include <cstdint>
#include <atomic>
#include <pthread.h>
#include <sched.h>
struct fpga_tile_job_t {
    long long ip_start_us, measured_compute_us;
    unsigned job_id;
    int bank, rows, group_blocks;
    unsigned spu_stream_count_before, spu_stream_done_before, spu_stream_out_before;
    unsigned spu_stream_drop_before, spu_stream_error_before;
};
std::atomic<long long> clock_us{0};
std::atomic<bool> hardware_done{true};
long long g_ip_timeout_us = 5000000;
int failure, calls;
void LOGE(const char *, ...) {}
long long monotonic_now_us() { return clock_us.load(); }
bool wait_vpu_done(uint32_t * status) {
    while (!hardware_done.load()) { sched_yield(); }
    ++calls;
    clock_us += 20;
    *status = 1;
    return failure != 1;
}
bool wait_spu_stream_outputs(const fpga_tile_job_t &) {
    ++calls;
    clock_us += 7;
    return failure != 2;
}
'''
harness += helper
harness += r'''
int main() {
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
}
'''
harness = "#include <initializer_list>\n" + harness
with tempfile.TemporaryDirectory() as tmp:
    cpp = Path(tmp) / "timer.cpp"
    exe = Path(tmp) / "timer"
    cpp.write_text(harness)
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-pthread", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
print("PASS: 500 asynchronous jobs, delayed collection, SPU tail, failures, restart/shutdown")
