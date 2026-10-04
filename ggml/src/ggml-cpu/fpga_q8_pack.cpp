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

static inline void store_scale_pair_words(volatile uint32_t * dst, uint32_t first, uint32_t second) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    *reinterpret_cast<volatile uint64_t *>(dst) =
        static_cast<uint64_t>(first) | (static_cast<uint64_t>(second) << 32U);
#else
    dst[0] = first;
    dst[1] = second;
#endif
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

bool fpga_pack_direct_weight_scale_pair_range(
    volatile uint32_t *        weight_dst_words,
    volatile uint32_t *        scale_dst_words,
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
    size_t *                   written_scale_entries) {
    if (!weight_dst_words || !scale_dst_words ||
        (wide_stores && ((uintptr_t) weight_dst_words & (alignof(uint64_t) - 1U)) != 0U) ||
        ((uintptr_t) scale_dst_words & (alignof(uint32_t) - 1U)) != 0U) {
        return false;
    }
    if (!src0 || !weight_data_base || !activation_data_base || !written_weight_words ||
        !written_scale_entries || rows <= 0 || group_blocks <= 0 ||
        group_blocks > std::numeric_limits<int>::max() / kBlockBeats ||
        group_beats != group_blocks * kBlockBeats || pair_begin > pair_end ||
        pair_end > ((size_t) rows + 1U) / 2U || row0 < 0 ||
        row0 > std::numeric_limits<int64_t>::max() - (int64_t) rows || k_block0 < 0 ||
        k_block0 > std::numeric_limits<int64_t>::max() - (int64_t) group_blocks) {
        return false;
    }

    const size_t rows_size         = (size_t) rows;
    const size_t group_blocks_size = (size_t) group_blocks;
    if (rows_size > std::numeric_limits<size_t>::max() / group_blocks_size) {
        return false;
    }
    const size_t total_scale_entries = rows_size * group_blocks_size;
    const size_t words_per_pair      = (size_t) group_beats * 8U;
    if (words_per_pair == 0U || pair_begin > std::numeric_limits<size_t>::max() / words_per_pair ||
        pair_end - pair_begin > std::numeric_limits<size_t>::max() / words_per_pair) {
        return false;
    }
    if (pair_begin == pair_end) {
        *written_weight_words = 0U;
        *written_scale_entries = 0U;
        return true;
    }

    const size_t first_row       = pair_begin * 2U;
    const size_t pair_last_row   = pair_end * 2U;
    const size_t last_row        = pair_last_row < rows_size ? pair_last_row : rows_size;
    if (first_row > last_row || first_row > total_scale_entries ||
        (last_row - first_row) > std::numeric_limits<size_t>::max() / group_blocks_size) {
        return false;
    }
    const size_t expected_scale_entries = (last_row - first_row) * group_blocks_size;

    static const int8_t zero_i8x16[kNumLanes] = {};
    const block_q8_0_t * const activation_blocks =
        static_cast<const block_q8_0_t *>(activation_data_base);
    volatile uint32_t * out = weight_dst_words + pair_begin * words_per_pair;
    size_t weight_words = 0U;

    for (size_t pair = pair_begin; pair < pair_end; ++pair) {
        const int even_row = (int) (pair * 2U);
        const int odd_row  = even_row + 1;
        const bool have_odd_row = odd_row < rows;
        uint32_t even_scale_pending = 0U;
        uint32_t odd_scale_pending  = 0U;
        bool even_scale_is_pending  = false;
        bool odd_scale_is_pending   = false;

        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * even_wb =
                weight_block_from_base(src0, weight_data_base, row0 + even_row, k_block0 + gb);
            const block_q8_0_t * odd_wb = have_odd_row ?
                                                  weight_block_from_base(
                                                      src0, weight_data_base, row0 + odd_row, k_block0 + gb) :
                                                  nullptr;

            for (int beat = 0; beat < kBlockBeats; ++beat) {
                store_i8x16_words(out, even_wb->qs + beat * kNumLanes, wide_stores);
                out += 4;
                weight_words += 4U;

                store_i8x16_words(out, odd_wb ? odd_wb->qs + beat * kNumLanes : zero_i8x16, wide_stores);
                out += 4;
                weight_words += 4U;
            }

            const uint32_t even_scale = (uint32_t) (uint16_t) activation_blocks[gb].d |
                                        ((uint32_t) (uint16_t) even_wb->d << 16U);
            volatile uint32_t * const even_scale_dst =
                scale_dst_words + (size_t) even_row * group_blocks_size + (size_t) gb;
            if (even_scale_is_pending) {
                store_scale_pair_words(even_scale_dst - 1, even_scale_pending, even_scale);
                even_scale_is_pending = false;
            } else if (wide_stores && gb + 1 < group_blocks &&
                       ((uintptr_t) even_scale_dst & (alignof(uint64_t) - 1U)) == 0U) {
                even_scale_pending = even_scale;
                even_scale_is_pending = true;
            } else {
                *even_scale_dst = even_scale;
            }

            if (have_odd_row) {
                const uint32_t odd_scale = (uint32_t) (uint16_t) activation_blocks[gb].d |
                                           ((uint32_t) (uint16_t) odd_wb->d << 16U);
                volatile uint32_t * const odd_scale_dst =
                    scale_dst_words + (size_t) odd_row * group_blocks_size + (size_t) gb;
                if (odd_scale_is_pending) {
                    store_scale_pair_words(odd_scale_dst - 1, odd_scale_pending, odd_scale);
                    odd_scale_is_pending = false;
                } else if (wide_stores && gb + 1 < group_blocks &&
                           ((uintptr_t) odd_scale_dst & (alignof(uint64_t) - 1U)) == 0U) {
                    odd_scale_pending = odd_scale;
                    odd_scale_is_pending = true;
                } else {
                    *odd_scale_dst = odd_scale;
                }
            }
        }

        // A pending value is created only when the next group block exists, so
        // the following iteration must consume it as an aligned pair.
        if (even_scale_is_pending || odd_scale_is_pending) {
            return false;
        }
    }

    *written_weight_words = weight_words;
    *written_scale_entries = expected_scale_entries;
    return weight_words == (pair_end - pair_begin) * words_per_pair;
}
