#include "fpga_q8_layout.h"

#include <limits>

bool fpga_weight_layout_payload_words(int rows, int group_beats, size_t * words) {
    if (!words || rows <= 0 || group_beats <= 0) {
        return false;
    }

    const size_t logical_rows = (size_t) rows;
    if (logical_rows == std::numeric_limits<size_t>::max()) {
        return false;
    }

    const size_t payload_rows = (logical_rows + 1U) & ~size_t(1U);
    const size_t beats        = (size_t) group_beats;
    if (payload_rows == 0U || payload_rows > std::numeric_limits<size_t>::max() / beats) {
        return false;
    }

    *words = payload_rows * beats;
    return true;
}

bool fpga_weight_layout_payload_bytes(int rows, int group_beats, size_t * bytes) {
    size_t words = 0U;
    if (!bytes || !fpga_weight_layout_payload_words(rows, group_beats, &words) ||
        words > std::numeric_limits<size_t>::max() / 16U) {
        return false;
    }

    *bytes = words * 16U;
    return true;
}

bool fpga_weight_layout_word_index(
    int      rows,
    int      group_beats,
    int      physical_row,
    int      beat,
    size_t * index) {
    size_t payload_words = 0U;
    if (!index || physical_row < 0 || beat < 0 || beat >= group_beats ||
        !fpga_weight_layout_payload_words(rows, group_beats, &payload_words)) {
        return false;
    }

    const size_t payload_rows = ((size_t) rows + 1U) & ~size_t(1U);
    if ((size_t) physical_row >= payload_rows) {
        return false;
    }

    const size_t row_pair = (size_t) physical_row >> 1U;
    const size_t candidate =
        (row_pair * (size_t) group_beats + (size_t) beat) * 2U + ((size_t) physical_row & 1U);
    if (candidate >= payload_words) {
        return false;
    }

    *index = candidate;
    return true;
}

bool fpga_weight_layout_word_offset(
    uint32_t   base,
    int        rows,
    int        group_beats,
    int        physical_row,
    int        beat,
    uint32_t * off) {
    size_t index = 0U;
    if (!off || !fpga_weight_layout_word_index(rows, group_beats, physical_row, beat, &index) ||
        index > std::numeric_limits<size_t>::max() / 16U) {
        return false;
    }

    const size_t byte_offset = index * 16U;
    if (byte_offset > UINT32_MAX || (uint64_t) base + (uint64_t) byte_offset > UINT32_MAX) {
        return false;
    }

    *off = base + (uint32_t) byte_offset;
    return true;
}

size_t weight_window_bytes_for_rows(int rows, int active_beats) {
    size_t bytes = 0U;
    return fpga_weight_layout_payload_bytes(rows, active_beats, &bytes) ? bytes : 0U;
}
