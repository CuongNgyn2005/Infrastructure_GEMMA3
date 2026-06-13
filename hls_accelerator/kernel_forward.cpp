#include "kernel_forward.h"
#include <stdint.h>
#include <math.h>

#define TILE_M 16
#define TILE_K 64
#define TILE_N 64

// ══════════════════════════════════════════════════════════
//  KV-Cache — HIỆU NĂNG CAO DÀNH CHO ZCU104
// ══════════════════════════════════════════════════════════
#define MAX_LAYERS 26
#define MAX_CTX    64     // <--- ĐÃ FIX: Giảm xuống 64 để nằm gọn trong 624 BRAM của ZCU104
#define MAX_HEAD_DIM 256  // <--- Bắt buộc giữ 256 cho Gemma-1B

static int8_t k_cache[MAX_LAYERS][MAX_CTX][MAX_HEAD_DIM];
static int8_t v_cache[MAX_LAYERS][MAX_CTX][MAX_HEAD_DIM];
static float  k_scale[MAX_LAYERS][MAX_CTX];
static float  v_scale[MAX_LAYERS][MAX_CTX];

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
    const float* A, const uint16_t* B_d, const int8_t* B_qs, float* C,
    int M, int K, int N, int layer_id, int pos, int is_attn)
{
    #pragma HLS INTERFACE m_axi port=A    offset=slave bundle=gmem0 depth=65536 max_read_burst_length=16 max_widen_bitwidth=128
    #pragma HLS INTERFACE m_axi port=B_d  offset=slave bundle=gmem1 depth=65536 max_read_burst_length=16 max_widen_bitwidth=64
    #pragma HLS INTERFACE m_axi port=B_qs offset=slave bundle=gmem2 depth=65536 max_read_burst_length=16 max_widen_bitwidth=128
    #pragma HLS INTERFACE m_axi port=C    offset=slave bundle=gmem3 depth=65536 max_write_burst_length=16 max_widen_bitwidth=128

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

    #pragma HLS BIND_STORAGE variable=k_cache type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=v_cache type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=k_scale type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=v_scale type=ram_2p impl=lutram

    // TỐI ƯU CÂN BẰNG TÀI NGUYÊN (factor=8 để tránh tốn BRAM)
    #pragma HLS ARRAY_PARTITION variable=k_cache cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=v_cache cyclic factor=8 dim=3

    if (is_attn == 0) {
        float    A_tile   [TILE_M][TILE_K];
        uint16_t B_d_tile [TILE_N][TILE_K / QK8_0];
        int8_t   B_qs_tile[TILE_N][TILE_K];
        float    C_tile   [TILE_M][TILE_N];

        #pragma HLS ARRAY_PARTITION variable=A_tile    cyclic factor=8 dim=2
        #pragma HLS ARRAY_PARTITION variable=B_d_tile  cyclic factor=2 dim=2
        #pragma HLS ARRAY_PARTITION variable=B_qs_tile cyclic factor=8 dim=2
        #pragma HLS ARRAY_PARTITION variable=C_tile    cyclic factor=8 dim=2

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
                    load_A:
                    for (int tm = 0; tm < TILE_M; ++tm)
                        for (int tk = 0; tk < TILE_K; ++tk) {
                            #pragma HLS PIPELINE II=1
                            int gr = m0+tm, gc = k0+tk;
                            A_tile[tm][tk] = (gr<M && gc<K) ? A[gr*K+gc] : 0.0f;
                        }

                    load_Bqs:
                    for (int tn = 0; tn < TILE_N; ++tn)
                        for (int tk = 0; tk < TILE_K; ++tk) {
                            #pragma HLS PIPELINE II=1
                            int gn = n0+tn, gk = k0+tk;
                            B_qs_tile[tn][tk] = (gn<N && gk<K) ? B_qs[gn*K+gk] : (int8_t)0;
                        }

                    load_Bd:
                    for (int tn = 0; tn < TILE_N; ++tn)
                        for (int t_blk = 0; t_blk < (TILE_K/QK8_0); ++t_blk) {
                            #pragma HLS PIPELINE II=1
                            int gn = n0+tn, gkb = k0/QK8_0+t_blk;
                            B_d_tile[tn][t_blk] = (gn<N) ? B_d[gn*(K/QK8_0)+gkb] : (uint16_t)0;
                        }

                    compute:
                    for (int tm = 0; tm < TILE_M; ++tm)
                        for (int tn = 0; tn < TILE_N; ++tn) {
                            #pragma HLS PIPELINE II=1
                            float sum = 0.0f;
                            for (int tk = 0; tk < TILE_K; ++tk) {
                                #pragma HLS UNROLL factor=8
                                float scale = f16_safe(B_d_tile[tn][tk/QK8_0]);
                                float b_val = (float)(B_qs_tile[tn][tk]) * scale;
                                sum += A_tile[tm][tk] * b_val;
                            }
                            C_tile[tm][tn] += sum;
                        }
                }

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
            }
        }
    }
    else {
        int head_dim = K;
        if (head_dim > MAX_HEAD_DIM) head_dim = MAX_HEAD_DIM;

        int safe_pos = (pos < MAX_CTX) ? pos : (MAX_CTX - 1);
        int safe_lid = (layer_id < MAX_LAYERS) ? layer_id : (MAX_LAYERS - 1);

        float Q[MAX_HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=Q cyclic factor=8 dim=1
        float scores[MAX_CTX];

        float k_s = f16_safe(B_d[0]);
        float v_s = f16_safe(B_d[1]);
        k_scale[safe_lid][safe_pos] = k_s;
        v_scale[safe_lid][safe_pos] = v_s;

        load_qkv:
        for (int d = 0; d < MAX_HEAD_DIM; d++) {
            #pragma HLS PIPELINE II=1
            if (d < head_dim) {
                Q[d] = A[d];
                k_cache[safe_lid][safe_pos][d] = B_qs[d];
                v_cache[safe_lid][safe_pos][d] = B_qs[head_dim + d];
            }
        }

        // Xóa dòng này: float inv_sqrt = 1.0f / sqrtf((float)head_dim);
		float max_score = -1e20f;

		calc_qk:
		for (int p = 0; p <= safe_pos; p++) {
			float sum = 0.0f;
			float pks = k_scale[safe_lid][p];
			for (int d = 0; d < MAX_HEAD_DIM; d++) {
				#pragma HLS PIPELINE II=1
				#pragma HLS UNROLL factor=8
				if (d < head_dim) {
					float kv = (float)(k_cache[safe_lid][p][d]) * pks;
					sum += Q[d] * kv; // KHÔNG NHÂN VỚI inv_sqrt NỮA!
				}
			}

			// Soft-Capping cực chuẩn của Gemma (Giới hạn ở +- 50.0)
			float capped_sc = 50.0f * tanhf(sum * 0.02f);

			scores[p] = capped_sc;
			if (capped_sc > max_score) max_score = capped_sc;
		}

		// --- Softmax (Chỉ xử lý đến safe_pos để tránh rác tương lai) ---
		float sum_exp = 0.0f;
		softmax_exp:
		for (int p = 0; p <= safe_pos; p++) {
			#pragma HLS PIPELINE II=1
			float ex = expf(scores[p] - max_score); // Trick chống Overflow
			scores[p] = ex;
			sum_exp += ex;
		}

		float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f; // Tránh chia cho 0
		softmax_norm:
		for (int p = 0; p <= safe_pos; p++) {
			#pragma HLS PIPELINE II=1
			scores[p] *= inv_sum;
		}

        float temp_out[MAX_HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=temp_out cyclic factor=8 dim=1

        init_temp:
        for (int d = 0; d < MAX_HEAD_DIM; d++) {
            #pragma HLS UNROLL factor=8
            temp_out[d] = 0.0f;
        }

        calc_sv:
        for (int p = 0; p <= safe_pos; p++) {
            float sc = scores[p];
            float pvs = v_scale[safe_lid][p];
            for (int d = 0; d < MAX_HEAD_DIM; d++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=8
                if (d < head_dim) {
                    float vv = (float)(v_cache[safe_lid][p][d]) * pvs;
                    temp_out[d] += sc * vv;
                }
            }
        }

        write_c:
        for (int d = 0; d < MAX_HEAD_DIM; d++) {
            #pragma HLS PIPELINE II=1
            if (d < head_dim) {
                C[d] = temp_out[d];
            }
        }
    }
}
}
