#!/usr/bin/env python3
"""RAM-only comparison of the production scale emission block; no device access."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cpu/fpga_host.cpp").read_text()
begin = source.index("    volatile uint32_t * const scale_words = ddr_checked_u32_ptr(SPU_PARAM_BASE, job.scale_bytes);")
block = source[begin:source.index("    mmio_fence();", begin)]
begin = source.index("static inline uint32_t fpga_p2_pack_scale_entry(")
entry = source[begin:source.index("\n}\n", begin) + 3]
harness = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
constexpr unsigned SPU_PARAM_BASE = 0, VPU_RESULT_PACK_LANES = 4;
struct block_q8_0_t { uint16_t d; };
struct Tensor { int stride; };
struct Shape { size_t entries, words; };
struct Job { size_t scale_bytes; bool p2_residency_hit; unsigned p2_residency_slot, job_id, tile_id; };
struct Resident { std::vector<uint16_t> scale_bits; size_t scale_count; };
static std::vector<Resident> g_p2_resident_tiles;
static volatile uint32_t * output;
static bool poisoned;
static volatile uint32_t * ddr_checked_u32_ptr(unsigned, size_t) { return output; }
static bool fpga_p2_residency_host_metadata_shape_valid(const Resident & r) {
    return r.scale_count == r.scale_bits.size();
}
static void fpga_p2_residency_poison_slot(unsigned, const char *) { poisoned = true; }
static void LOGE(const char *, ...) {}
static const block_q8_0_t * weight_block_from_base(const Tensor * t, const void * base, int row, int block) {
    return static_cast<const block_q8_0_t *>(base) + row * t->stride + block;
}
'''
harness += entry
harness += '''
static bool emit(Job job, Shape scale_shape, int rows, int group_blocks,
                 const block_q8_0_t * act_group, const Tensor * src0,
                 const void * weight_data_base, int row0, int k_block0) {
'''
harness += block + "\nreturn true;\n}\n"
harness += r'''
int main() {
    unsigned cases = 0;
    for (int rows : {1, 2, 3, 7, 255, 256}) {
        for (int blocks : {1, 2, 3, 7, 31, 32, 63, 64}) {
            const size_t entries = rows * blocks, words = (entries + 3) / 4;
            Tensor tensor{blocks + 3};
            std::vector<block_q8_0_t> weights((rows + 2) * tensor.stride), acts(blocks);
            for (size_t i = 0; i < weights.size(); ++i) weights[i].d = static_cast<uint16_t>(i * 7919);
            for (int i = 0; i < blocks; ++i) acts[i].d = static_cast<uint16_t>(0xffff - i * 313);
            // Include infinities, NaNs and signed zeros as raw bits, not arithmetic.
            weights[tensor.stride + 1].d = 0x7c00;
            acts[0].d = 0x8000;
            g_p2_resident_tiles = {Resident{std::vector<uint16_t>(entries), entries}};
            std::vector<uint32_t> expected(words * 4, 0);
            for (int row = 0; row < rows; ++row) for (int gb = 0; gb < blocks; ++gb) {
                const auto w = weights[(row + 1) * tensor.stride + gb + 1].d;
                g_p2_resident_tiles[0].scale_bits[row * blocks + gb] = w;
                expected[row * blocks + gb] = static_cast<uint32_t>(acts[gb].d) | (static_cast<uint32_t>(w) << 16);
            }
            for (bool resident : {false, true}) for (unsigned shift : {0U, 4U}) {
                std::vector<uint64_t> storage(words * 2 + 4, UINT64_C(0xabababababababab));
                output = reinterpret_cast<uint32_t *>(storage.data()) + 2 + shift / 4;
                Job job{words * 16, resident, 0, 1, 1};
                assert(emit(job, {entries, words}, rows, blocks, acts.data(), &tensor, weights.data(), 1, 1));
                for (size_t i = 0; i < expected.size(); ++i) assert(output[i] == expected[i]);
                assert(output[-1] == 0xababababU && output[expected.size()] == 0xababababU);
                ++cases;
            }
            std::vector<uint64_t> storage(words * 2 + 2, UINT64_C(0xabababababababab));
            output = reinterpret_cast<uint32_t *>(storage.data());
            g_p2_resident_tiles[0].scale_count++;
            poisoned = false;
            assert(!emit({words * 16, true, 0, 1, 1}, {entries, words}, rows, blocks,
                         acts.data(), &tensor, weights.data(), 1, 1));
            assert(poisoned);
            for (auto value : storage) assert(value == UINT64_C(0xabababababababab));
        }
    }
    printf("PASS: %u scale layouts; both sources, odd/even rows/blocks, alignment, padding, canaries, pre-write rejection\n", cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-scales-") as directory:
    cpp = Path(directory) / "scales.cpp"
    binary = Path(directory) / "scales"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra",
                    "-Werror", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    str(cpp), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)
