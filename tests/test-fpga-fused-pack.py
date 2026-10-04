#!/usr/bin/env python3
"""RAM-only regression for the direct P2 fused WEIGHT+SCALE packer.

The test compiles the production fused kernel and production persistent worker
extraction. It never maps hardware, touches UIO/ZDMA, or claims board
performance. Scale values are compared as raw FP16 bits so signed zero, NaN,
and infinity payloads are preserved.
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
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <thread>
#include <vector>

constexpr int VPU_BLOCK_BEATS = 2;
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
harness = harness.replace("#include <array>\n", "#include <array>\n#include <atomic>\n")
harness += section("enum fpga_pack_detail_field {", "// Caller-owned counters")
harness += "static std::array<long long, PACK_DETAIL_COUNT> g_pack_detail = {};\n"
task_end = source.index("} fpga_p2_pack_worker_task_t;")
task_start = source.rfind("typedef struct {", 0, task_end)
harness += source[task_start:source.index("static long long         g_p2_residency_avoided_cpu_pack_bytes", task_end)]
harness += section("static bool fpga_copy_weight_pair_range(", "static bool checked_size_add(")

harness += r'''
static uint16_t raw_d(const block_q8_0 & block) {
    uint16_t bits = 0;
    std::memcpy(&bits, &block.d, sizeof(bits));
    return bits;
}

static void set_raw_d(block_q8_0 & block, uint16_t bits) {
    std::memcpy(&block.d, &bits, sizeof(bits));
}

static void fill_q8(block_q8_0 & block, uint16_t scale_bits, int seed) {
    set_raw_d(block, scale_bits);
    for (int i = 0; i < 32; ++i) {
        block.qs[i] = static_cast<int8_t>((seed + i * 13) % 256 - 128);
    }
}

static void expected_scales(const std::vector<block_q8_0> & weights,
                            size_t stride,
                            int row0,
                            int k_block0,
                            int rows,
                            int group_blocks,
                            const std::vector<block_q8_0> & activation,
                            std::vector<uint32_t> & output) {
    output.assign(static_cast<size_t>(rows) * static_cast<size_t>(group_blocks), 0U);
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const auto & weight = weights[static_cast<size_t>(row0 + row) * stride +
                                          static_cast<size_t>(k_block0 + gb)];
            output[static_cast<size_t>(row) * static_cast<size_t>(group_blocks) + static_cast<size_t>(gb)] =
                static_cast<uint32_t>(raw_d(activation[static_cast<size_t>(gb)])) |
                (static_cast<uint32_t>(raw_d(weight)) << 16U);
        }
    }
}

static void check_canary(const std::vector<uint64_t> & storage, size_t first_word, size_t live_words) {
    constexpr uint32_t canary = UINT32_C(0xabababab);
    const auto * words = reinterpret_cast<const uint32_t *>(storage.data());
    const size_t word_count = storage.size() * 2U;
    assert(first_word >= 1U);
    assert(first_word + live_words < word_count);
    assert(words[first_word - 1U] == canary);
    assert(words[first_word + live_words] == canary);
}

static void test_fused_ranges() {
    constexpr int source_rows = 260;
    constexpr int source_blocks = 65;
    constexpr size_t source_stride = 69;
    std::vector<block_q8_0> weights(static_cast<size_t>(source_rows) * source_stride);
    for (int row = 0; row < source_rows; ++row) {
        for (int gb = 0; gb < source_blocks; ++gb) {
            uint16_t bits = static_cast<uint16_t>((row * 37 + gb * 19) & 0x7bff);
            if (row == 2 && gb == 1) bits = UINT16_C(0x8000);
            if (row == 4 && gb == 3) bits = UINT16_C(0x7e01);
            if (row == 6 && gb == 5) bits = UINT16_C(0x7c00);
            fill_q8(weights[static_cast<size_t>(row) * source_stride + static_cast<size_t>(gb)], bits,
                    row * 97 + gb * 11);
        }
        for (int gb = source_blocks; gb < static_cast<int>(source_stride); ++gb) {
            fill_q8(weights[static_cast<size_t>(row) * source_stride + static_cast<size_t>(gb)],
                    UINT16_C(0xdead), 3);
        }
    }

    ggml_tensor tensor = {};
    tensor.data = weights.data();
    tensor.nb[1] = source_stride * sizeof(block_q8_0);
    std::vector<block_q8_0> activation(source_blocks);
    for (size_t gb = 0; gb < activation.size(); ++gb) {
        const uint16_t bits = gb == 0U ? UINT16_C(0x8000) :
                              (gb == 1U ? UINT16_C(0x7e01) :
                               (gb == 2U ? UINT16_C(0x7c00) : static_cast<uint16_t>(0x1100U + gb)));
        fill_q8(activation[gb], bits, static_cast<int>(gb * 7U));
    }

    const std::array<int, 8> row_cases = {1, 2, 3, 5, 7, 255, 256, 257};
    const std::array<int, 8> block_cases = {1, 2, 3, 4, 7, 32, 63, 64};
    for (int rows : row_cases) {
        for (int group_blocks : block_cases) {
            constexpr int row0 = 1;
            constexpr int k_block0 = 1;
            const int group_beats = group_blocks * VPU_BLOCK_BEATS;
            const size_t pairs = (static_cast<size_t>(rows) + 1U) / 2U;
            const size_t words_per_pair = static_cast<size_t>(group_beats) * 8U;
            const size_t weight_words = pairs * words_per_pair;
            const size_t scale_entries = static_cast<size_t>(rows) * static_cast<size_t>(group_blocks);
            const size_t scale_words = (scale_entries + 3U) / 4U;

            std::vector<uint64_t> expected_weight_storage(weight_words / 2U + 4U,
                                                           UINT64_C(0xabababababababab));
            std::vector<uint64_t> actual_weight_storage(weight_words / 2U + 4U,
                                                         UINT64_C(0xabababababababab));
            auto * expected_weight = reinterpret_cast<uint32_t *>(expected_weight_storage.data() + 1U);
            auto * actual_weight = reinterpret_cast<uint32_t *>(actual_weight_storage.data() + 1U);
            size_t expected_written = 0U;
            assert(fpga_pack_direct_weight_pair_range(
                expected_weight, &tensor, weights.data(), row0, k_block0, rows, group_blocks, group_beats,
                0U, pairs, false, &expected_written));
            assert(expected_written == weight_words);

            std::vector<uint32_t> expected_scale;
            expected_scales(weights, source_stride, row0, k_block0, rows, group_blocks, activation, expected_scale);

            for (size_t scale_offset_words : {0U, 1U}) {
                std::vector<uint64_t> actual_scale_storage((scale_entries + 1U) / 2U + 8U,
                                                            UINT64_C(0xabababababababab));
                auto * actual_scale = reinterpret_cast<uint32_t *>(actual_scale_storage.data() + 2U) +
                                      scale_offset_words;
                std::fill(actual_weight_storage.begin(), actual_weight_storage.end(),
                          UINT64_C(0xabababababababab));
                const std::array<size_t, 4> splits = {0U, 1U, pairs / 2U, pairs};
                for (size_t split : splits) {
                    std::fill(actual_weight_storage.begin(), actual_weight_storage.end(),
                              UINT64_C(0xabababababababab));
                    std::fill(actual_scale_storage.begin(), actual_scale_storage.end(),
                              UINT64_C(0xabababababababab));
                    size_t left_weight = 0U, left_scale = 0U;
                    size_t right_weight = 0U, right_scale = 0U;
                    assert(fpga_pack_direct_weight_scale_pair_range(
                        actual_weight, actual_scale, &tensor, weights.data(), activation.data(), row0, k_block0,
                        rows, group_blocks, group_beats, 0U, split, true, &left_weight, &left_scale));
                    assert(fpga_pack_direct_weight_scale_pair_range(
                        actual_weight, actual_scale, &tensor, weights.data(), activation.data(), row0, k_block0,
                        rows, group_blocks, group_beats, split, pairs, true, &right_weight, &right_scale));
                    assert(left_weight + right_weight == weight_words);
                    assert(left_scale + right_scale == scale_entries);
                    for (size_t i = 0; i < weight_words; ++i) assert(actual_weight[i] == expected_weight[i]);
                    for (size_t i = 0; i < scale_entries; ++i) assert(actual_scale[i] == expected_scale[i]);
                    // The production caller clears final 128-bit padding after
                    // both producers join; the fused kernel must leave it alone.
                    for (size_t i = scale_entries; i < scale_words * 4U; ++i) {
                        assert(actual_scale[i] == UINT32_C(0xabababab));
                    }
                    check_canary(actual_weight_storage, 2U, weight_words);
                    check_canary(actual_scale_storage, 4U + scale_offset_words, scale_entries);
                }
            }
        }
    }
}

static void test_invalid_prewrite_rejection() {
    constexpr int rows = 3, group_blocks = 3, group_beats = group_blocks * VPU_BLOCK_BEATS;
    std::vector<block_q8_0> weights(static_cast<size_t>(rows) * group_blocks);
    std::vector<block_q8_0> activation(group_blocks);
    ggml_tensor tensor = {};
    tensor.data = weights.data();
    tensor.nb[1] = group_blocks * sizeof(block_q8_0);
    std::vector<uint64_t> weight_storage(128U, UINT64_C(0xabababababababab));
    std::vector<uint64_t> scale_storage(128U, UINT64_C(0xabababababababab));
    auto * weight_dst = reinterpret_cast<uint32_t *>(weight_storage.data() + 2U);
    auto * scale_dst = reinterpret_cast<uint32_t *>(scale_storage.data() + 2U);
    auto assert_unchanged = [&]() {
        for (uint64_t value : weight_storage) assert(value == UINT64_C(0xabababababababab));
        for (uint64_t value : scale_storage) assert(value == UINT64_C(0xabababababababab));
    };
    size_t weight_count = 0U, scale_count = 0U;
    assert(!fpga_pack_direct_weight_scale_pair_range(
        weight_dst, scale_dst, &tensor, weights.data(), activation.data(), 0, 0, 0, group_blocks,
        group_beats, 0, 1, true, &weight_count, &scale_count));
    assert_unchanged();
    assert(!fpga_pack_direct_weight_scale_pair_range(
        weight_dst, scale_dst, &tensor, weights.data(), nullptr, 0, 0, rows, group_blocks,
        group_beats, 0, 1, true, &weight_count, &scale_count));
    assert_unchanged();
    assert(!fpga_pack_direct_weight_scale_pair_range(
        weight_dst, scale_dst, &tensor, weights.data(), activation.data(), 0, 0, rows, group_blocks,
        0, 0, 1, true, &weight_count, &scale_count));
    assert_unchanged();
    assert(!fpga_pack_direct_weight_scale_pair_range(
        weight_dst, scale_dst, &tensor, weights.data(), activation.data(), 0, 0, rows, group_blocks,
        group_beats, 0, 99, true, &weight_count, &scale_count));
    assert_unchanged();
    assert(!fpga_pack_direct_weight_scale_pair_range(
        weight_dst, scale_dst, &tensor, weights.data(), activation.data(), -1, 0, rows, group_blocks,
        group_beats, 0, 1, true, &weight_count, &scale_count));
    assert_unchanged();
    assert(!fpga_pack_direct_weight_scale_pair_range(
        weight_dst, scale_dst, &tensor, weights.data(), activation.data(), 0, -1, rows, group_blocks,
        group_beats, 0, 1, true, &weight_count, &scale_count));
    assert_unchanged();
    auto * misaligned_weight = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(weight_dst) + 4U);
    assert(!fpga_pack_direct_weight_scale_pair_range(
        misaligned_weight, scale_dst, &tensor, weights.data(), activation.data(), 0, 0, rows, group_blocks,
        group_beats, 0, 1, true, &weight_count, &scale_count));
    assert_unchanged();
}

static void test_fused_worker() {
    constexpr int rows = 5, group_blocks = 3, group_beats = group_blocks * VPU_BLOCK_BEATS;
    constexpr size_t pairs = (rows + 1U) / 2U;
    constexpr size_t split = 1U;
    constexpr size_t words_per_pair = static_cast<size_t>(group_beats) * 8U;
    const size_t weight_words = pairs * words_per_pair;
    const size_t scale_entries = static_cast<size_t>(rows) * group_blocks;
    std::vector<block_q8_0> weights(static_cast<size_t>(rows) * group_blocks);
    std::vector<block_q8_0> activation(group_blocks);
    for (size_t i = 0; i < weights.size(); ++i) fill_q8(weights[i], static_cast<uint16_t>(0x1200U + i), i + 1);
    for (size_t i = 0; i < activation.size(); ++i) fill_q8(activation[i], static_cast<uint16_t>(0x2200U + i), i + 9);
    ggml_tensor tensor = {};
    tensor.data = weights.data();
    tensor.nb[1] = group_blocks * sizeof(block_q8_0);
    std::vector<uint64_t> expected_weight_storage(weight_words / 2U + 3U,
                                                   UINT64_C(0xabababababababab));
    std::vector<uint64_t> expected_scale_storage((scale_entries + 3U) / 2U + 3U,
                                                  UINT64_C(0xabababababababab));
    auto * expected_weight = reinterpret_cast<uint32_t *>(expected_weight_storage.data() + 1U);
    auto * expected_scale = reinterpret_cast<uint32_t *>(expected_scale_storage.data() + 1U);
    size_t expected_weight_count = 0U, expected_scale_count = 0U;
    assert(fpga_pack_direct_weight_scale_pair_range(
        expected_weight, expected_scale, &tensor, weights.data(), activation.data(), 0, 0, rows, group_blocks,
        group_beats, 0, pairs, true, &expected_weight_count, &expected_scale_count));
    assert(expected_weight_count == weight_words && expected_scale_count == scale_entries);

    std::vector<uint64_t> actual_weight_storage(weight_words / 2U + 3U,
                                                 UINT64_C(0xabababababababab));
    std::vector<uint64_t> actual_scale_storage((scale_entries + 3U) / 2U + 3U,
                                                UINT64_C(0xabababababababab));
    auto * actual_weight = reinterpret_cast<uint32_t *>(actual_weight_storage.data() + 1U);
    auto * actual_scale = reinterpret_cast<uint32_t *>(actual_scale_storage.data() + 1U);

    assert(fpga_p2_pack_worker_start());
    uint64_t generation = 0U;
    assert(fpga_p2_pack_worker_next_generation(&generation));
    fpga_p2_pack_worker_task_t task = {
        &tensor, weights.data(), 0, 0, rows, group_blocks, group_beats,
        split, pairs, actual_weight, (pairs - split) * words_per_pair, generation,
    };
    task.activation_data_base = activation.data();
    task.scale_words = actual_scale;
    task.expected_scale_entries = static_cast<size_t>(rows - 2) * group_blocks;
    task.fused_scale = true;
    assert(fpga_p2_pack_worker_submit(task));
    size_t main_weight_count = 0U, main_scale_count = 0U;
    assert(fpga_pack_direct_weight_scale_pair_range(
        actual_weight, actual_scale, &tensor, weights.data(), activation.data(), 0, 0, rows, group_blocks,
        group_beats, 0, split, true, &main_weight_count, &main_scale_count));
    mmio_fence();
    size_t helper_weight_count = 0U, helper_scale_count = 0U;
    long long service = 0, wait = 0;
    assert(fpga_p2_pack_worker_wait(generation, &helper_weight_count, &service, &wait, &helper_scale_count));
    assert(main_weight_count + helper_weight_count == weight_words);
    assert(main_scale_count + helper_scale_count == scale_entries);
    for (size_t i = 0; i < weight_words; ++i) assert(actual_weight[i] == expected_weight[i]);
    for (size_t i = 0; i < scale_entries; ++i) assert(actual_scale[i] == expected_scale[i]);

    const size_t expected_helper_scale_entries = static_cast<size_t>(rows - 2) * group_blocks;
    for (int iteration = 0; iteration < 64; ++iteration) {
        for (size_t gb = 0; gb < activation.size(); ++gb) {
            set_raw_d(activation[gb], static_cast<uint16_t>(0x3000U + iteration * 17U + gb));
        }
        std::fill(actual_weight_storage.begin(), actual_weight_storage.end(),
                  UINT64_C(0xabababababababab));
        std::fill(actual_scale_storage.begin(), actual_scale_storage.end(),
                  UINT64_C(0xabababababababab));
        std::fill(expected_weight_storage.begin(), expected_weight_storage.end(),
                  UINT64_C(0xabababababababab));
        std::fill(expected_scale_storage.begin(), expected_scale_storage.end(),
                  UINT64_C(0xabababababababab));
        expected_weight_count = 0U;
        expected_scale_count = 0U;
        assert(fpga_pack_direct_weight_scale_pair_range(
            expected_weight, expected_scale, &tensor, weights.data(), activation.data(), 0, 0, rows,
            group_blocks, group_beats, 0, pairs, true, &expected_weight_count, &expected_scale_count));
        assert(expected_weight_count == weight_words && expected_scale_count == scale_entries);
        assert(fpga_p2_pack_worker_next_generation(&generation));
        task.generation = generation;
        task.expected_scale_entries = expected_helper_scale_entries;
        assert(fpga_p2_pack_worker_submit(task));
        assert(fpga_pack_direct_weight_scale_pair_range(
            actual_weight, actual_scale, &tensor, weights.data(), activation.data(), 0, 0, rows, group_blocks,
            group_beats, 0, split, true, &main_weight_count, &main_scale_count));
        mmio_fence();
        assert(fpga_p2_pack_worker_wait(generation, &helper_weight_count, &service, &wait, &helper_scale_count));
        assert(main_weight_count + helper_weight_count == weight_words);
        assert(main_scale_count + helper_scale_count == scale_entries);
        for (size_t i = 0; i < weight_words; ++i) assert(actual_weight[i] == expected_weight[i]);
        for (size_t i = 0; i < scale_entries; ++i) assert(actual_scale[i] == expected_scale[i]);
    }

    assert(fpga_p2_pack_worker_next_generation(&generation));
    task.generation = generation;
    task.expected_scale_entries++;
    assert(fpga_p2_pack_worker_submit(task));
    assert(!fpga_p2_pack_worker_wait(generation, &helper_weight_count, &service, &wait, &helper_scale_count));
    task.expected_scale_entries--;
    assert(fpga_p2_pack_worker_stop());
    assert(fpga_p2_pack_worker_stop());
}

int main() {
    test_fused_ranges();
    test_invalid_prewrite_rejection();
    test_fused_worker();
    puts("PASS: fused direct WEIGHT+SCALE byte equivalence; raw FP16 bits; strided source; split ranges; odd rows; "
         "aligned/4-byte scale destinations; guards; pre-write rejection; worker counts and failure");
}
'''

with tempfile.TemporaryDirectory(prefix="fpga-fused-pack-") as directory:
    test = Path(directory) / "fused_pack.cpp"
    binary = Path(directory) / "fused_pack"
    test.write_text(harness)
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-pthread",
               "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined", "-fno-sanitize-recover=all",
               "-I" + str(cpu), "-I" + str(root / "ggml/include"), "-I" + str(root / "ggml/src"),
               str(test), str(cpu / "fpga_q8_pack.cpp"), "-o", str(binary)]
    subprocess.run(command, check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=60)
