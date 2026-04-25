#include "kernel_forward.h"
#include <stdint.h>
#include <math.h>

#define TILE_M 16
#define TILE_K 64
#define TILE_N 64

// ══════════════════════════════════════════════════════════
//  KV-Cache — kích thước phù hợp với ZCU104 (96 URAM)
//
//  ZCU104 resources:
//    URAM: 96 blocks × 288KB = 27.6MB
//    BRAM: 624 blocks × 36Kb = 2.7MB
//
//  Gemma-3-1B: 26 layers, head_dim=256, 4 heads
//  Để fit vào URAM: giảm MAX_CTX xuống 64 (decode window)
//
//  k_cache[26][64][64] int8  = 106,496B ≈ 7 URAM ✅
//  v_cache[26][64][64] int8  = 106,496B ≈ 7 URAM ✅
//  k_scale[26][64]     float = 6,656B   → LUTRAM ✅
//  v_scale[26][64]     float = 6,656B   → LUTRAM ✅
//  Tổng URAM: ~14 block << 96 available
// ══════════════════════════════════════════════════════════
#define MAX_LAYERS 26
#define MAX_CTX    64
#define HEAD_DIM   64

// KV-Cache arrays — sẽ được map vào URAM
static int8_t k_cache[MAX_LAYERS][MAX_CTX][HEAD_DIM];
static int8_t v_cache[MAX_LAYERS][MAX_CTX][HEAD_DIM];

// Scale arrays — nhỏ, dùng LUTRAM hoặc BRAM
static float  k_scale[MAX_LAYERS][MAX_CTX];
static float  v_scale[MAX_LAYERS][MAX_CTX];

// f16→f32 inline cho HLS
static float f16_safe(uint16_t h) {
    #pragma HLS INLINE
    uint32_t s = (uint32_t)((h >> 15) & 1U);
    uint32_t e = (uint32_t)((h >> 10) & 0x1FU);
    uint32_t m = (uint32_t)(h & 0x3FFU);
    uint32_t b;
    if      (e == 0U)  b = s << 31;
    else if (e == 31U) b = (s << 31) | 0x7F800000U | (m << 13);
    else               b = (s << 31) | ((e + 112U) << 23) | (m << 13);
    union { uint32_t i; float f; } u; u.i = b; return u.f;
}

extern "C" {
void kernel_forward(
    const float*    A,
    const uint16_t* B_d,
    const int8_t*   B_qs,
    float*          C,
    int M, int K, int N,
    int layer_id,
    int pos,
    int is_attn)
{
    // ── AXI-MM interfaces ────────────────────────────────────
    #pragma HLS INTERFACE m_axi port=A    offset=slave bundle=gmem0 \
        depth=65536 max_read_burst_length=16 max_widen_bitwidth=128
    #pragma HLS INTERFACE m_axi port=B_d  offset=slave bundle=gmem1 \
        depth=65536 max_read_burst_length=16 max_widen_bitwidth=64
    #pragma HLS INTERFACE m_axi port=B_qs offset=slave bundle=gmem2 \
        depth=65536 max_read_burst_length=16 max_widen_bitwidth=128
    #pragma HLS INTERFACE m_axi port=C    offset=slave bundle=gmem3 \
        depth=65536 max_write_burst_length=16 max_widen_bitwidth=128

    #pragma HLS INTERFACE s_axilite port=A        bundle=control
    #pragma HLS INTERFACE s_axilite port=B_d      bundle=control
    #pragma HLS INTERFACE s_axilite port=B_qs     bundle=control
    #pragma HLS INTERFACE s_axilite port=C        bundle=control
    #pragma HLS INTERFACE s_axilite port=M        bundle=control
    #pragma HLS INTERFACE s_axilite port=K        bundle=control
    #pragma HLS INTERFACE s_axilite port=N        bundle=control
    #pragma HLS INTERFACE s_axilite port=layer_id bundle=control
    #pragma HLS INTERFACE s_axilite port=pos      bundle=control
    #pragma HLS INTERFACE s_axilite port=is_attn  bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // ── Map KV-Cache vào URAM ────────────────────────────────
    // Mỗi array ~104KB → vừa với 1 URAM block (288KB)
    #pragma HLS BIND_STORAGE variable=k_cache type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=v_cache type=ram_2p impl=uram
    // Scale arrays nhỏ → để HLS tự chọn (LUTRAM)
    #pragma HLS BIND_STORAGE variable=k_scale type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=v_scale type=ram_2p impl=lutram

    // ════════════════════════════════════════════════════════
    //  NHÁNH 1: MATMUL (is_attn == 0)
    //  Dùng cho: prefill, MLP (gate/up/down projection)
    // ════════════════════════════════════════════════════════
    if (is_attn == 0) {

        float    A_tile   [TILE_M][TILE_K];
        uint16_t B_d_tile [TILE_N][TILE_K / QK8_0];
        int8_t   B_qs_tile[TILE_N][TILE_K];
        float    C_tile   [TILE_M][TILE_N];

        #pragma HLS ARRAY_PARTITION variable=A_tile    cyclic factor=16 dim=2
        #pragma HLS ARRAY_PARTITION variable=B_d_tile  cyclic factor=2  dim=2
        #pragma HLS ARRAY_PARTITION variable=B_qs_tile cyclic factor=16 dim=2
        #pragma HLS ARRAY_PARTITION variable=C_tile    cyclic factor=16 dim=2

        // Row buffer để ghi C burst liên tục (tránh stride bug)
        float C_row[TILE_N];
        #pragma HLS ARRAY_PARTITION variable=C_row complete dim=1

        for (int m0 = 0; m0 < M; m0 += TILE_M) {
            for (int n0 = 0; n0 < N; n0 += TILE_N) {

                init_C:
                for (int ti = 0; ti < TILE_M; ++ti)
                    for (int tj = 0; tj < TILE_N; ++tj) {
                        #pragma HLS PIPELINE II=1
                        C_tile[ti][tj] = 0.0f;
                    }

                for (int k0 = 0; k0 < K; k0 += TILE_K) {

                    // Load A — A[m*K + k]
                    load_A:
                    for (int tm = 0; tm < TILE_M; ++tm)
                        for (int tk = 0; tk < TILE_K; ++tk) {
                            #pragma HLS PIPELINE II=1
                            int gr = m0+tm, gc = k0+tk;
                            A_tile[tm][tk] = (gr<M && gc<K) ? A[gr*K+gc] : 0.0f;
                        }

                    // Load B_qs — N×K layout: B_qs[n*K + k]
                    load_Bqs:
                    for (int tn = 0; tn < TILE_N; ++tn)
                        for (int tk = 0; tk < TILE_K; ++tk) {
                            #pragma HLS PIPELINE II=1
                            int gn = n0+tn, gk = k0+tk;
                            B_qs_tile[tn][tk] = (gn<N && gk<K) ?
                                                B_qs[gn*K+gk] : (int8_t)0;
                        }

                    // Load B_d — N×K layout: B_d[n*(K/32) + k/32]
                    load_Bd:
                    for (int tn = 0; tn < TILE_N; ++tn)
                        for (int t_blk = 0; t_blk < (TILE_K/QK8_0); ++t_blk) {
                            #pragma HLS PIPELINE II=1
                            int gn = n0+tn, gkb = k0/QK8_0+t_blk;
                            B_d_tile[tn][t_blk] = (gn<N) ?
                                                  B_d[gn*(K/QK8_0)+gkb] : (uint16_t)0;
                        }

                    // Compute
                    compute:
                    for (int tm = 0; tm < TILE_M; ++tm)
                        for (int tn = 0; tn < TILE_N; ++tn) {
                            #pragma HLS PIPELINE II=1
                            float sum = 0.0f;
                            for (int tk = 0; tk < TILE_K; ++tk) {
                                #pragma HLS UNROLL factor=16
                                float scale = f16_safe(B_d_tile[tn][tk/QK8_0]);
                                float b_val = (float)(B_qs_tile[tn][tk]) * scale;
                                sum += A_tile[tm][tk] * b_val;
                            }
                            C_tile[tm][tn] += sum;
                        }

                } // k0

                // Store C qua row buffer (burst write, không stride)
                store_C:
                for (int tm = 0; tm < TILE_M; ++tm) {
                    int gr = m0+tm;
                    if (gr >= M) continue;

                    copy_row:
                    for (int tn = 0; tn < TILE_N; ++tn) {
                        #pragma HLS PIPELINE II=1
                        C_row[tn] = C_tile[tm][tn];
                    }
                    write_row:
                    for (int tn = 0; tn < TILE_N; ++tn) {
                        #pragma HLS PIPELINE II=1
                        int gc = n0+tn;
                        if (gc < N) C[gr*N+gc] = C_row[tn];
                    }
                }

            } // n0
        } // m0

    } // is_attn == 0

    // ════════════════════════════════════════════════════════
    //  NHÁNH 2: ATTENTION với KV-Cache trên URAM (is_attn == 1)
    //
    //  Giao thức truyền dữ liệu từ host:
    //    A[0..HEAD_DIM-1]           → Query vector (float32)
    //    B_qs[0..HEAD_DIM-1]        → Key mới (int8)
    //    B_qs[HEAD_DIM..2*HEAD-1]   → Value mới (int8)
    //    B_d[0] → Key scale (fp16)
    //    B_d[1] → Value scale (fp16)
    //    layer_id, pos              → vị trí cache
    // ════════════════════════════════════════════════════════
    else {

        // Clamp pos để tránh out-of-bounds
        int safe_pos = (pos < MAX_CTX) ? pos : (MAX_CTX - 1);
        int safe_lid = (layer_id < MAX_LAYERS) ? layer_id : (MAX_LAYERS - 1);

        float Q[HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=Q complete dim=1

        float scores[MAX_CTX];

        // 1. Lưu K, V mới vào URAM + load Q
        float k_s = f16_safe(B_d[0]);
        float v_s = f16_safe(B_d[1]);
        k_scale[safe_lid][safe_pos] = k_s;
        v_scale[safe_lid][safe_pos] = v_s;

        load_qkv:
        for (int d = 0; d < HEAD_DIM; d++) {
            #pragma HLS PIPELINE II=1
            Q[d] = A[d];
            k_cache[safe_lid][safe_pos][d] = B_qs[d];
            v_cache[safe_lid][safe_pos][d] = B_qs[HEAD_DIM + d];
        }

        // 2. Q × K^T → scores
        float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
        float max_score = -1e20f;

        calc_qk:
        for (int p = 0; p <= safe_pos; p++) {
            float sum = 0.0f;
            float pks = k_scale[safe_lid][p];
            for (int d = 0; d < HEAD_DIM; d++) {
                #pragma HLS PIPELINE II=1
                float kv = (float)(k_cache[safe_lid][p][d]) * pks;
                sum += Q[d] * kv;
            }
            float sc = sum * inv_sqrt;
            scores[p] = sc;
            if (sc > max_score) max_score = sc;
        }

        // 3. Softmax
        float sum_exp = 0.0f;
        softmax_exp:
        for (int p = 0; p <= safe_pos; p++) {
            #pragma HLS PIPELINE II=1
            float ex = expf(scores[p] - max_score);
            scores[p] = ex;
            sum_exp += ex;
        }
        float inv_sum = (sum_exp > 1e-10f) ? 1.0f / sum_exp : 0.0f;
        softmax_norm:
        for (int p = 0; p <= safe_pos; p++) {
            #pragma HLS PIPELINE II=1
            scores[p] *= inv_sum;
        }

        // 4. Score × V → C
        calc_sv:
        for (int d = 0; d < HEAD_DIM; d++) {
            float out = 0.0f;
            for (int p = 0; p <= safe_pos; p++) {
                #pragma HLS PIPELINE II=1
                float pvs = v_scale[safe_lid][p];
                float vv  = (float)(v_cache[safe_lid][p][d]) * pvs;
                out += scores[p] * vv;
            }
            C[d] = out;
        }

    } // is_attn == 1
}
} // extern "C"
