#!/usr/bin/env python3
"""Compile the actual SPU reader against RAM; no FPGA mappings or transfers."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()
start = source.index("static int64_t ddr_read_spu_q16_row(")
reader = source[start:source.index("\n}\n", start) + 3]
harness = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <initializer_list>
alignas(16) static uint8_t memory[64];
static unsigned markers = 0;
[[noreturn]] static void fpga_fatal(const char *, ...) { throw std::runtime_error("reject"); }
static uint8_t * ddr_ptr(uint32_t off, size_t bytes) {
    if (off > sizeof(memory) || bytes > sizeof(memory) - off) fpga_fatal("range");
    return memory + off;
}
static void fpga_p2_boundary_marker(const char *, ...) { ++markers; }
'''
harness += reader
harness += r'''
int main() {
    unsigned cases = 0;
    uint64_t random = 12345;
    for (unsigned i = 0; i < 10000; ++i) {
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        const uint64_t edges[] = {0, 1, UINT64_MAX, UINT64_C(0x8000000000000000),
                                 UINT64_C(0x7fffffffffffffff), UINT64_C(0xffffffffffff0000)};
        const uint64_t bits = i < 6 ? edges[i] : random;
        const uint16_t id = static_cast<uint16_t>(i * 37);
        int64_t expected;
        memcpy(&expected, &bits, sizeof(expected));
        for (uint32_t off : {0U, 4U, 48U}) {
            memset(memory, static_cast<int>(i & 255), sizeof(memory));
            memory[off] = static_cast<uint8_t>(id);
            memory[off + 1] = static_cast<uint8_t>(id >> 8);
            for (unsigned b = 0; b < 8; ++b)
                memory[off + 2 + b] = static_cast<uint8_t>(bits >> (b * 8));
            // Remaining bytes deliberately retain arbitrary padding.
            for (bool trace : {false, true}) {
                markers = 0;
                uint16_t actual_id = 0;
                assert(ddr_read_spu_q16_row(off, &actual_id, trace) == expected);
                assert(actual_id == id);
                assert(markers == (trace ? 6U : 0U));
                assert(ddr_read_spu_q16_row(off, nullptr, trace) == expected);
                ++cases;
            }
        }
    }
    for (uint32_t off : {1U, 2U, 3U, 52U, UINT32_MAX}) {
        bool rejected = false;
        try { (void) ddr_read_spu_q16_row(off, nullptr); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
    }
    printf("PASS: %u reader cases; signed extremes, padding, aligned/scalar/trace paths, null ID, bounds rejection\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-reader-") as directory:
    cpp = Path(directory) / "reader.cpp"
    binary = Path(directory) / "reader"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2",
                    "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fno-sanitize-recover=all", str(cpp), "-o", str(binary)],
                   check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)
