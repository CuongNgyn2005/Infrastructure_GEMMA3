#pragma once
#include <stdint.h>
#include <string.h>

#define QK8_0 32

inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)((h >> 15) & 1U);
    uint32_t e = (uint32_t)((h >> 10) & 0x1FU);
    uint32_t m = (uint32_t)(h & 0x3FFU);
    uint32_t b;
    if      (e == 0U)  b = s << 31;
    else if (e == 31U) b = (s << 31) | 0x7F800000U | (m << 13);
    else               b = (s << 31) | ((e + 112U) << 23) | (m << 13);
    float r; memcpy(&r, &b, 4); return r;
}

inline uint16_t f32_to_f16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t s = (b >> 31) & 1U;
    int32_t  e = (int32_t)((b >> 23) & 0xFFU) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3FFU;
    if (e <= 0)  return (uint16_t)(s << 15);
    if (e >= 31) return (uint16_t)((s << 15) | 0x7C00U);
    return (uint16_t)((s << 15) | ((uint32_t)e << 10) | m);
}

extern "C" {
void kernel_forward(
    const float*    A,
    const uint16_t* B_d,
    const int8_t*   B_qs,
    float*          C,
    int M, int K, int N
);
}
