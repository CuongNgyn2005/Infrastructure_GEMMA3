#include "fpga_q8_pack.h"

#include "ggml.h"
#include "quants.h"

#include <limits>

namespace {

constexpr int kNumLanes   = 16;
constexpr int kQk8        = 32;
constexpr int kBlockBeats = kQk8 / kNumLanes;

using block_q8_0_t = block_q8_0;

static inline uint32_t pack_i8x4_le(const int8_t * lanes) {
    return (uint32_t) (uint8_t) lanes[0] |
           ((uint32_t) (uint8_t) lanes[1] << 8U) |
           ((uint32_t) (uint8_t) lanes[2] << 16U) |
           ((uint32_t) (uint8_t) lanes[3] << 24U);
}

static inline uint64_t pack_i8x8_le(const int8_t * lanes) {
    return (uint64_t) pack_i8x4_le(lanes) | ((uint64_t) pack_i8x4_le(lanes + 4) << 32U);
}

static inline void store_i8x16_words(volatile uint32_t * dst, const int8_t * lanes, bool wide_stores) {
    if (wide_stores) {
        volatile uint64_t * const dst_u64 = reinterpret_cast<volatile uint64_t *>(dst);
        dst_u64[0] = pack_i8x8_le(lanes + 0);
        dst_u64[1] = pack_i8x8_le(lanes + 8);
        return;
    }
    dst[0] = pack_i8x4_le(lanes + 0);
    dst[1] = pack_i8x4_le(lanes + 4);
    dst[2] = pack_i8x4_le(lanes + 8);
    dst[3] = pack_i8x4_le(lanes + 12);
}

static inline const block_q8_0_t * weight_block_from_base(
    const struct ggml_tensor * src0,
    const void *               data_base,
    int64_t                    row,
    int64_t                    block) {
    const char * row_base = (const char *) data_base + row * src0->nb[1];
    return (const block_q8_0_t *) row_base + block;
}

} // namespace

bool fpga_pack_direct_weight_pair_range(
    volatile uint32_t *        dst_words,
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
    size_t *                   written_words) {
    if (!dst_words || (wide_stores && ((uintptr_t) dst_words & (alignof(uint64_t) - 1U)) != 0U) || !src0 ||
        !weight_data_base || !written_words || rows <= 0 || group_blocks <= 0 ||
        group_beats != group_blocks * kBlockBeats || pair_begin > pair_end ||
        pair_end > ((size_t) rows + 1U) / 2U) {
        return false;
    }

    const size_t words_per_pair = (size_t) group_beats * 8U;
    if (words_per_pair == 0U || pair_begin > std::numeric_limits<size_t>::max() / words_per_pair ||
        pair_end - pair_begin > std::numeric_limits<size_t>::max() / words_per_pair) {
        return false;
    }

    static const int8_t zero_i8x16[kNumLanes] = {};

    volatile uint32_t * out   = dst_words + pair_begin * words_per_pair;
    size_t              words = 0U;

    for (size_t pair = pair_begin; pair < pair_end; ++pair) {
        const int even_row = (int) (pair * 2U);
        const int odd_row  = even_row + 1;

        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * even_wb =
                weight_block_from_base(src0, weight_data_base, row0 + even_row, k_block0 + gb);
            const block_q8_0_t * odd_wb = odd_row < rows ?
                                                  weight_block_from_base(
                                                      src0, weight_data_base, row0 + odd_row, k_block0 + gb) :
                                                  nullptr;

            for (int beat = 0; beat < kBlockBeats; ++beat) {
                store_i8x16_words(out, even_wb->qs + beat * kNumLanes, wide_stores);
                out += 4;
                words += 4U;

                store_i8x16_words(out, odd_wb ? odd_wb->qs + beat * kNumLanes : zero_i8x16, wide_stores);
                out += 4;
                words += 4U;
            }
        }
    }

    *written_words = words;
    return words == (pair_end - pair_begin) * words_per_pair;
}
