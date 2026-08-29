#pragma once

#include <cstddef>
#include <cstdint>

// Canonical Protocol-2/VPU2 pair-interleaved weight-layout arithmetic.
bool fpga_weight_layout_payload_words(int rows, int group_beats, size_t * words);
bool fpga_weight_layout_payload_bytes(int rows, int group_beats, size_t * bytes);

bool fpga_weight_layout_word_index(
    int      rows,
    int      group_beats,
    int      physical_row,
    int      beat,
    size_t * index);

bool fpga_weight_layout_word_offset(
    uint32_t   base,
    int        rows,
    int        group_beats,
    int        physical_row,
    int        beat,
    uint32_t * off);

size_t weight_window_bytes_for_rows(int rows, int active_beats);
