#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

// Emit the canonical VPU2 pair-major Q8_0 weight layout for one contiguous
// range of even/odd row pairs. The caller owns mapping, synchronization, and
// worker scheduling; this module owns only the Q8 byte transformation.
bool fpga_pack_direct_weight_pair_range(
    volatile uint32_t *       dst_words,
    const struct ggml_tensor * src0,
    const void *               weight_data_base,
    int64_t                    row0,
    int64_t                    k_block0,
    int                        rows,
    int                        group_blocks,
    int                        group_beats,
    size_t                     pair_begin,
    size_t                     pair_end,
    size_t *                   written_words);
