#!/usr/bin/env python3
"""RAM-only equivalence checks for optional AArch64 paired weight stores."""

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "ggml/src/ggml-cpu"
HARNESS = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml.h"
#include "quants.h"
#include "fpga_q8_pack.h"

struct alignas(16) Beat { uint32_t word[4]; };
static_assert(sizeof(Beat) == 16, "weight beat layout");
static const uint16_t bits[] = {0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 1, 0x3c00};

static void check(int rows, int groups, int kind) {
    const size_t pairs = (static_cast<size_t>(rows) + 1) / 2;
    const size_t word_count = pairs * groups * 16;
    const size_t entries = static_cast<size_t>(rows) * groups;
    const int stride = groups + 2;
    std::vector<block_q8_0> weights(static_cast<size_t>(rows + 2) * stride);
    std::vector<block_q8_0> act(groups);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i].d = bits[i % 7];
        for (int lane = 0; lane < 32; ++lane)
            weights[i].qs[lane] = static_cast<int8_t>(static_cast<int>((i * 29 + lane * 17) % 256) - 128);
    }
    for (int i = 0; i < groups; ++i) act[i].d = bits[(i + 2) % 7];
    ggml_tensor tensor = {};
    tensor.nb[1] = static_cast<size_t>(stride) * sizeof(block_q8_0);
    tensor.data = weights.data();
    std::vector<Beat> first(word_count / 4 + 1), second(first.size());
    std::memset(first.data(), 0xa5, first.size() * sizeof(Beat));
    std::memset(second.data(), 0xa5, second.size() * sizeof(Beat));
    std::vector<uint64_t> scale_first((entries * 4 + 7) / 8 + 2, UINT64_C(0xa5a5a5a5a5a5a5a5));
    std::vector<uint64_t> scale_second = scale_first;
    size_t counts[2] = {}, scale_counts[2] = {};
    for (int pass = 0; pass < 2; ++pass) {
        fpga_q8_pack_set_pair_stores(pass != 0);
#if defined(__aarch64__)
        assert(fpga_q8_pack_pair_stores_enabled() == (pass != 0));
#else
        assert(!fpga_q8_pack_pair_stores_enabled());
#endif
        auto * weight = reinterpret_cast<uint32_t *>(pass ? second.data() : first.data());
        auto * scale = pass ? scale_second.data() : scale_first.data();
        assert((reinterpret_cast<uintptr_t>(weight) & 15U) == 0U);
        bool ok;
        if (kind == 0) {
            ok = fpga_pack_direct_weight_pair_range(weight, &tensor, weights.data(), 1, 1,
                    rows, groups, groups * 2, 0, pairs, true, &counts[pass]);
        } else if (kind == 1) {
            ok = fpga_pack_direct_weight_scale_pair_range(weight, reinterpret_cast<uint32_t *>(scale),
                    &tensor, weights.data(), act.data(), 1, 1, rows, groups, groups * 2,
                    0, pairs, true, &counts[pass], &scale_counts[pass]);
        } else {
            ok = fpga_pack_direct_weight_p3scale_pair_range(weight, reinterpret_cast<uint16_t *>(scale),
                    &tensor, weights.data(), 1, 1, rows, groups, groups * 2,
                    0, pairs, true, &counts[pass], &scale_counts[pass]);
        }
        assert(ok && counts[pass] == word_count);
        if (kind != 0) assert(scale_counts[pass] == entries);
    }
    assert(std::memcmp(first.data(), second.data(), first.size() * sizeof(Beat)) == 0);
    assert(scale_first == scale_second);
    for (uint32_t guard : second.back().word) assert(guard == 0xa5a5a5a5U);
#if defined(__aarch64__)
    // Aligned to 8, deliberately not aligned to 16: no silent scalar fallback.
    auto * misaligned = reinterpret_cast<uint32_t *>(second.data()) + 2;
    size_t count = 0, scales = 0;
    assert(!fpga_pack_direct_weight_pair_range(misaligned, &tensor, weights.data(), 1, 1,
                rows, groups, groups * 2, 0, pairs, true, &count));
    assert(!fpga_pack_direct_weight_scale_pair_range(misaligned,
                reinterpret_cast<uint32_t *>(scale_second.data()), &tensor, weights.data(), act.data(),
                1, 1, rows, groups, groups * 2, 0, pairs, true, &count, &scales));
    assert(!fpga_pack_direct_weight_p3scale_pair_range(misaligned,
                reinterpret_cast<uint16_t *>(scale_second.data()), &tensor, weights.data(),
                1, 1, rows, groups, groups * 2, 0, pairs, true, &count, &scales));
    assert(std::memcmp(first.data(), second.data(), first.size() * sizeof(Beat)) == 0);
    assert(scale_first == scale_second);
#endif
    fpga_q8_pack_set_pair_stores(false);
}

int main() {
    int cases = 0;
    for (int rows : {3, 4, 255, 256}) for (int groups : {1, 36, 64})
        for (int kind = 0; kind < 3; ++kind) { check(rows, groups, kind); ++cases; }
    assert(cases == 36 && !fpga_q8_pack_pair_stores_enabled());
#if defined(__aarch64__)
    std::printf("PASS test-fpga-pack-stp (%d comparisons, AArch64 STP and alignment rejection)\n", cases);
#else
    std::printf("PASS test-fpga-pack-stp (%d comparisons, non-AArch64 setter remains disabled)\n", cases);
#endif
}
'''

with tempfile.TemporaryDirectory(prefix="fpga-pack-stp-") as directory:
    source = Path(directory) / "pack_stp.cpp"
    binary = Path(directory) / ("pack_stp.exe" if os.name == "nt" else "pack_stp")
    source.write_text(HARNESS, encoding="utf-8")
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(CPU), "-I" + str(ROOT / "ggml/include"), "-I" + str(ROOT / "ggml/src"),
                    str(source), str(CPU / "fpga_q8_pack.cpp"), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=60)
