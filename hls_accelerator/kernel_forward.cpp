#include "kernel_forward.h"
#include <stdint.h>
#include <math.h> // cho hàm expf (Softmax)

#define TILE_M 16
#define TILE_K 64
#define TILE_N 64

// ══════════════════════════════════════════════════════════
// THÔNG SỐ KIẾN TRÚC K-V CACHE (Tùy chỉnh theo mô hình của bạn)
// ══════════════════════════════════════════════════════════
#define MAX_LAYERS 32
#define MAX_CTX    512
#define HEAD_DIM   64

// LƯU TRỮ VĨNH VIỄN TRÊN LÕI FPGA BẰNG ULTRARAM (URAM)
// K-V Cache định dạng INT8 + Mảng lưu Scale (để dequantize)
static int8_t k_cache[MAX_LAYERS][MAX_CTX][HEAD_DIM];
static float  k_scale[MAX_LAYERS][MAX_CTX];

static int8_t v_cache[MAX_LAYERS][MAX_CTX][HEAD_DIM];
static float  v_scale[MAX_LAYERS][MAX_CTX];

extern "C" {

void kernel_forward(
    const float* A,
    const uint16_t* B_d,
    const int8_t* B_qs,
    float* C,
    int M, int K, int N,
    
    // --- 3 BIẾN ĐIỀU KHIỂN MỚI CHO LỘ TRÌNH 3 ---
    int layer_id,
    int pos,
    int is_attn)
{
    #pragma HLS INTERFACE m_axi     port=A    offset=slave bundle=gmem0 depth=4096 max_read_burst_length=16  num_read_outstanding=16
    #pragma HLS INTERFACE m_axi     port=B_d  offset=slave bundle=gmem1 depth=4096 max_read_burst_length=16  num_read_outstanding=16
    #pragma HLS INTERFACE m_axi     port=B_qs offset=slave bundle=gmem2 depth=4096 max_read_burst_length=16  num_read_outstanding=16
    #pragma HLS INTERFACE m_axi     port=C    offset=slave bundle=gmem3 depth=4096 max_write_burst_length=16 num_write_outstanding=16
    
    #pragma HLS INTERFACE s_axilite port=A    bundle=control
    #pragma HLS INTERFACE s_axilite port=B_d  bundle=control
    #pragma HLS INTERFACE s_axilite port=B_qs bundle=control
    #pragma HLS INTERFACE s_axilite port=C    bundle=control
    #pragma HLS INTERFACE s_axilite port=M    bundle=control
    #pragma HLS INTERFACE s_axilite port=K    bundle=control
    #pragma HLS INTERFACE s_axilite port=N    bundle=control
    
    // Đăng ký 3 thanh ghi mới
    #pragma HLS INTERFACE s_axilite port=layer_id bundle=control
    #pragma HLS INTERFACE s_axilite port=pos      bundle=control
    #pragma HLS INTERFACE s_axilite port=is_attn  bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // Ép Vivado tổng hợp KV Cache bằng URAM (UltraRAM)
    #pragma HLS BIND_STORAGE variable=k_cache type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=v_cache type=ram_2p impl=uram

    // ===================================================================
    // NHÁNH 1: NHÂN MA TRẬN THÔNG THƯỜNG (A * B_qs) - DÙNG CHO PREFILL & MLP
    // ===================================================================
    if (is_attn == 0) {
        float    A_tile   [TILE_M][TILE_K];
        uint16_t B_d_tile [TILE_K][TILE_N / QK8_0];
        int8_t   B_qs_tile[TILE_K][TILE_N];
        float    C_tile   [TILE_M][TILE_N];

        #pragma HLS ARRAY_PARTITION variable=A_tile    cyclic factor=16 dim=2
        #pragma HLS ARRAY_PARTITION variable=B_d_tile  cyclic factor=8  dim=2
        #pragma HLS ARRAY_PARTITION variable=B_qs_tile cyclic factor=16 dim=2
        #pragma HLS ARRAY_PARTITION variable=C_tile    cyclic factor=16 dim=2

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
                            int gr = m0 + tm, gc = k0 + tk;
                            A_tile[tm][tk] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                        }

                    load_B:
                    for (int tk = 0; tk < TILE_K; ++tk)
                        for (int tn = 0; tn < TILE_N; ++tn) {
                            #pragma HLS PIPELINE II=1
                            int gr = k0 + tk, gc = n0 + tn;
                            if (gr < K && gc < N) {
                                B_qs_tile[tk][tn] = B_qs[gr * N + gc];
                                if (tn % QK8_0 == 0)
                                    B_d_tile[tk][tn / QK8_0] =
                                        B_d[gr * (N / QK8_0) + gc / QK8_0];
                            } else {
                                B_qs_tile[tk][tn] = 0;
                            }
                        }

                    compute:
                    for (int tm = 0; tm < TILE_M; ++tm)
                        for (int tk = 0; tk < TILE_K; ++tk) {
                            #pragma HLS PIPELINE II=1
                            float a_val = A_tile[tm][tk];
                            for (int tn = 0; tn < TILE_N; ++tn) {
                                #pragma HLS UNROLL factor=16
                                float scale     = f16_to_f32(B_d_tile[tk][tn / QK8_0]);
                                float decoded_b = (float)(B_qs_tile[tk][tn]) * scale;
                                C_tile[tm][tn] += a_val * decoded_b;
                            }
                        }

                } // k0

                store_C:
                for (int tm = 0; tm < TILE_M; ++tm)
                    for (int tn = 0; tn < TILE_N; ++tn) {
                        #pragma HLS PIPELINE II=1
                        int gr = m0 + tm, gc = n0 + tn;
                        if (gr < M && gc < N)
                            C[gr * N + gc] = C_tile[tm][tn];
                    }

            } // n0
        } // m0
    }
    
    // ===================================================================
    // NHÁNH 2: SELF-ATTENTION SIÊU TỐC (DÙNG K-V CACHE NỘI BỘ TRÊN URAM)
    // ===================================================================
    else {
        // Giả sử CPU đã nén Q, K, V của token HIỆN TẠI thành 1 cục truyền qua mảng A
        // A[0 .. HEAD_DIM-1]         -> Chứa Q (Query)
        // B_qs[0 .. HEAD_DIM-1]      -> Chứa K (Key int8)   (Tận dụng cổng B_qs truyền data)
        // B_qs[HEAD_DIM .. 2*HEAD-1] -> Chứa V (Value int8) (Tận dụng cổng B_qs truyền data)
        // B_d[0] -> K_scale, B_d[1] -> V_scale
        
        float Q[HEAD_DIM];
        float scores[MAX_CTX];
        
        // 1. Load Q và Lưu K, V mới vào Bộ nhớ URAM
        float k_s = f16_to_f32(B_d[0]);
        float v_s = f16_to_f32(B_d[1]);
        k_scale[layer_id][pos] = k_s;
        v_scale[layer_id][pos] = v_s;

        load_q_and_cache_kv:
        for (int d = 0; d < HEAD_DIM; d++) {
            #pragma HLS PIPELINE II=1
            Q[d] = A[d];
            k_cache[layer_id][pos][d] = B_qs[d];
            v_cache[layer_id][pos][d] = B_qs[HEAD_DIM + d];
        }

        // 2. Tính Q * K^T (Quét toàn bộ quá khứ từ token 0 đến pos) - KHÔNG ĐỤNG TỚI DDR
        float max_score = -1e20f;
        calc_q_k:
        for (int past = 0; past <= pos; past++) {
            float sum = 0.0f;
            float past_k_scale = k_scale[layer_id][past];
            for (int d = 0; d < HEAD_DIM; d++) {
                #pragma HLS PIPELINE II=1
                float k_val = (float)(k_cache[layer_id][past][d]) * past_k_scale;
                sum += Q[d] * k_val;
            }
            float scaled_sum = sum / sqrtf((float)HEAD_DIM);
            scores[past] = scaled_sum;
            if (scaled_sum > max_score) max_score = scaled_sum;
        }

        // 3. Softmax
        float sum_exp = 0.0f;
        calc_softmax_exp:
        for (int past = 0; past <= pos; past++) {
            #pragma HLS PIPELINE II=1
            scores[past] = expf(scores[past] - max_score);
            sum_exp += scores[past];
        }
        calc_softmax_div:
        for (int past = 0; past <= pos; past++) {
            #pragma HLS PIPELINE II=1
            scores[past] /= sum_exp;
        }

        // 4. Tính Score * V -> Ghi thẳng ra C
        calc_score_v:
        for (int d = 0; d < HEAD_DIM; d++) {
            float out_val = 0.0f;
            for (int past = 0; past <= pos; past++) {
                #pragma HLS PIPELINE II=1
                float past_v_scale = v_scale[layer_id][past];
                float v_val = (float)(v_cache[layer_id][past][d]) * past_v_scale;
                out_val += scores[past] * v_val;
            }
            C[d] = out_val;
        }
    }
}
} // extern "C"