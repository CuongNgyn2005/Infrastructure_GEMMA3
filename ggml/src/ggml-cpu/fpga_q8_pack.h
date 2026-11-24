#pragma once

#include <cstddef>
#include <cstdint>

#include "quants.h"

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
    bool                       wide_stores,
    size_t *                   written_words);

// Emit the canonical VPU2 pair-major Q8_0 weight layout and the matching
// row-major P2 scale entries while visiting each source block once.  The
// activation data points to a contiguous group of block_q8_0 values; its
// scale field is read as raw FP16 bits.  The caller owns mapping, range
// admission, synchronization, and worker scheduling.
bool fpga_pack_direct_weight_scale_pair_range(
    volatile uint32_t *       weight_dst_words,
    volatile uint32_t *       scale_dst_words,
    const struct ggml_tensor * src0,
    const void *               weight_data_base,
    const void *               activation_data_base,
    int64_t                    row0,
    int64_t                    k_block0,
    int                        rows,
    int                        group_blocks,
    int                        group_beats,
    size_t                     pair_begin,
    size_t                     pair_end,
    bool                       wide_stores,
    size_t *                   written_weight_words,
    size_t *                   written_scale_entries);

// Raw FP16 P3 PARAM stores: no destination reads and no stores wider than 64 bits.
void fpga_p3_store_scale_row(volatile uint16_t * dst, const uint16_t * src, int count);

bool fpga_pack_direct_weight_p3scale_pair_range(
    volatile uint32_t *       weight_dst_words,
    volatile uint16_t *       scale_dst_halves,
    const struct ggml_tensor * src0,
    const void *               weight_data_base,
    int64_t                    row0,
    int64_t                    k_block0,
    int                        rows,
    int                        group_blocks,
    int                        group_beats,
    size_t                     pair_begin,
    size_t                     pair_end,
    bool                       wide_stores,
    size_t *                   written_weight_words,
    size_t *                   written_scale_entries);

// Pack raw activation FP16 halves into little-endian SCRATCH words, zeroing padding.
bool fpga_p3_pack_scratch_words(const block_q8_0 * act, int group_blocks,
                                uint32_t * out_words, size_t word_count);
