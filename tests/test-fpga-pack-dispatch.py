#!/usr/bin/env python3
"""Linux RAM-only regression of the production P2 worker; never maps hardware.

Extract the static worker implementation rather than maintain a second copy.
Link the real Q8 packer. Device fences are replaced by a RAM atomic fence;
this test provides no evidence about device memory, DMA, or board performance.
"""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
cpu = root / "ggml/src/ggml-cpu"
source = (cpu / "fpga_host.cpp").read_text()


def section(begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


harness = r'''
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <pthread.h>
#include "ggml.h"
#include "quants.h"
#include "fpga_q8_pack.h"
static int g_p2_pack_workers_requested = 2;
static long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void mmio_fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
'''
harness += section("enum fpga_pack_detail_field {", "// Caller-owned counters")
harness += "static std::array<long long, PACK_DETAIL_COUNT> g_pack_detail = {};\n"
task_end = source.index("} fpga_p2_pack_worker_task_t;")
task_start = source.rfind("typedef struct {", 0, task_end)
harness += source[task_start:source.index("static long long         g_p2_residency_avoided_cpu_pack_bytes", task_end)]
harness += r'''
// Force scheduling opportunities between publication and notification.
static int checked_signal(pthread_cond_t * cv) {
    std::this_thread::yield();
    return pthread_cond_signal(cv);
}
#define pthread_cond_signal checked_signal
'''
harness += section("static void * fpga_p2_pack_worker_main(void *) {", "static bool checked_size_add(")
harness += r'''
#undef pthread_cond_signal
int main() {
    constexpr int rows = 5, blocks = 3, beats = blocks * 2;
    constexpr size_t pairs = (rows + 1) / 2, words_per_pair = beats * 8;
    constexpr size_t words = pairs * words_per_pair;
    std::vector<block_q8_0> input(rows * blocks);
    ggml_tensor tensor = {};
    tensor.data = input.data();
    tensor.nb[1] = blocks * sizeof(block_q8_0);
    std::vector<uint64_t> actual(words / 2 + 2), expected(words / 2 + 2);
    auto dst = reinterpret_cast<uint32_t *>(actual.data());
    auto ref = reinterpret_cast<uint32_t *>(expected.data());
    fpga_p2_pack_worker_task_t task = {
        &tensor, input.data(), 0, 0, rows, blocks, beats,
        1, pairs, dst, (pairs - 1) * words_per_pair, 1
    };
    uint64_t generation = 0;
    assert(!fpga_p2_pack_worker_next_generation(nullptr));
    assert(!fpga_p2_pack_worker_next_generation(&generation));
    assert(!fpga_p2_pack_worker_submit(task));
    for (int cycle = 0; cycle < 3; ++cycle) {
        assert(fpga_p2_pack_worker_start());
        assert(!fpga_p2_pack_worker_start());
        task.generation = 0;
        assert(!fpga_p2_pack_worker_submit(task));
        // Admission rejection checks are isolated while the helper is idle.
        pthread_mutex_lock(&g_p2_pack_worker_mutex);
        g_p2_pack_worker_busy = true;
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        task.generation = 1;
        assert(!fpga_p2_pack_worker_submit(task));
        assert(!fpga_p2_pack_worker_next_generation(&generation));
        assert(!fpga_p2_pack_worker_stop());
        pthread_mutex_lock(&g_p2_pack_worker_mutex);
        g_p2_pack_worker_busy = false;
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        for (int i = 0; i < 2000; ++i) {
            for (size_t j = 0; j < input.size(); ++j)
                for (int k = 0; k < 32; ++k)
                    input[j].qs[k] = static_cast<int8_t>((i + j * 31 + k) % 256 - 128);
            std::fill(actual.begin(), actual.end(), UINT64_C(0xabababababababab));
            expected = actual;
            size_t ref_words = 0, main_words = 0, helper_words = 0;
            long long service = 0, wait = 0;
            assert(fpga_pack_direct_weight_pair_range(ref, &tensor, input.data(), 0, 0,
                rows, blocks, beats, 0, pairs, false, &ref_words));
            assert(fpga_p2_pack_worker_next_generation(&generation));
            task.generation = generation;
            if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(20));
            assert(fpga_p2_pack_worker_submit(task));
            assert(fpga_pack_direct_weight_pair_range(dst, &tensor, input.data(), 0, 0,
                rows, blocks, beats, 0, 1, true, &main_words));
            mmio_fence();
            if (i % 3 == 1) std::this_thread::sleep_for(std::chrono::microseconds(20));
            assert(fpga_p2_pack_worker_wait(generation, &helper_words, &service, &wait));
            assert(main_words + helper_words == ref_words);
            assert(actual == expected); // Includes odd-row padding and tail canary.
            if (generation > 1)
                assert(!fpga_p2_pack_worker_wait(generation - 1, &helper_words, &service, &wait));
        }
        // A failed range must publish a failed completion, never hang or pass.
        assert(fpga_p2_pack_worker_next_generation(&generation));
        task.generation = generation;
        task.group_beats = 0;
        assert(fpga_p2_pack_worker_submit(task));
        size_t written = 0;
        long long service = 0, wait = 0;
        assert(!fpga_p2_pack_worker_wait(generation, &written, &service, &wait));
        task.group_beats = beats;
        assert(fpga_p2_pack_worker_stop());
        assert(fpga_p2_pack_worker_stop());
        assert(!fpga_p2_pack_worker_submit(task));
    }
    assert(fpga_p2_pack_worker_start());
    g_p2_pack_worker_next_generation = UINT64_MAX;
    assert(!fpga_p2_pack_worker_next_generation(&generation));
    assert(fpga_p2_pack_worker_stop());
    puts("PASS: 6000 byte-exact parallel jobs; wake races, stale/failed completion, admission, overflow, restart");
}
'''
# Structural guard: delayed notification is the intended production change.
submit = section("static bool fpga_p2_pack_worker_submit(", "static bool fpga_p2_pack_worker_wait(")
assert submit.index("const long long unlocked") < submit.index("pthread_cond_signal")
with tempfile.TemporaryDirectory(prefix="fpga-dispatch-") as directory:
    test = Path(directory) / "dispatch.cpp"
    binary = Path(directory) / "dispatch"
    test.write_text(harness)
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-pthread",
               "-Wall", "-Wextra", "-Werror", "-I" + str(cpu),
               "-I" + str(root / "ggml/include"), "-I" + str(root / "ggml/src"),
               str(test), str(cpu / "fpga_q8_pack.cpp"), "-o", str(binary)]
    subprocess.run(command, check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=60)
