#include "kernel_forward.h"
#include <stdint.h>

// ══════════════════════════════════════════════════════════
//  FIX: Layout B phải khớp với ggml Q8_0
//
//  ggml lưu weight src0 shape (K, N) với ne[0]=K, ne[1]=N:
//    - Trong memory: N hàng × K cột (row-major)
//    - Mỗi block Q8_0 cover QK8_0=32 phần tử THEO CHIỀU K
//    - block[n * (K/32) + k/32] = scale cho W[n][k..k+31]
//    - B_qs[n * K + k]           (N, K) layout
//    - B_d [n * (K/32) + k/32]   (N, K/32) layout
//
//  Phép tính đúng:
//    C[m][n] = sum_k  A[m][k] * W[n][k]   (ggml: src1 × src0^T)
//
//  Kernel cũ SAI vì đọc B_qs[k*N+n] và B_d[k*(N/32)+n/32]
//  → transpose so với thực tế → output rác
// ══════════════════════════════════════════════════════════

#define TILE_M  16
#define TILE_K  64
#define TILE_N  64
#define QK8_0   32   // số phần tử mỗi block Q8_0

extern "C" {

void kernel_forward(
    const float*    A,      // (M, K)   — activation  row-major
    const uint16_t* B_d,    // (N, K/32) — scales fp16 row-major
    const int8_t*   B_qs,   // (N, K)   — quants      row-major
    float*          C,      // (M, N)   — output      row-major
    int M, int K, int N)
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
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // ── Local tiles ──────────────────────────────────────
    float    A_tile   [TILE_M][TILE_K];
    // FIX: B_d_tile[TILE_N][TILE_K/QK8_0]
    //   - dim1 = TILE_N  (N index trong tile)
    //   - dim2 = TILE_K/QK8_0 = 2 (block index trong K direction)
    uint16_t B_d_tile [TILE_N][TILE_K / QK8_0];
    int8_t   B_qs_tile[TILE_N][TILE_K];
    float    C_tile   [TILE_M][TILE_N];

    #pragma HLS ARRAY_PARTITION variable=A_tile    cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=B_d_tile  cyclic factor=2  dim=2
    #pragma HLS ARRAY_PARTITION variable=B_qs_tile cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile    cyclic factor=16 dim=2

    // ── Vòng lặp tiling ──────────────────────────────────
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        for (int n0 = 0; n0 < N; n0 += TILE_N) {

            // ── Init C tile về 0 ──
            init_C:
            for (int ti = 0; ti < TILE_M; ++ti)
                for (int tj = 0; tj < TILE_N; ++tj) {
                    #pragma HLS PIPELINE II=1
                    C_tile[ti][tj] = 0.0f;
                }

            for (int k0 = 0; k0 < K; k0 += TILE_K) {

                // ── Load A tile: A[m0..m0+TILE_M][k0..k0+TILE_K] ──
                load_A:
                for (int tm = 0; tm < TILE_M; ++tm)
                    for (int tk = 0; tk < TILE_K; ++tk) {
                        #pragma HLS PIPELINE II=1
                        int gr = m0 + tm;
                        int gc = k0 + tk;
                        A_tile[tm][tk] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                    }

                // ── Load B tile: B[(n0..n0+TILE_N)][k0..k0+TILE_K] ──
                // FIX: đổi thứ tự loop — outer=tn (N), inner=tk (K)
                // B_qs layout: B_qs[n * K + k]       → (N,K) row-major
                // B_d  layout: B_d [n * (K/32) + k/32] → (N, K/32) row-major
                load_B:
                for (int tn = 0; tn < TILE_N; ++tn)
                    for (int tk = 0; tk < TILE_K; ++tk) {
                        #pragma HLS PIPELINE II=1
                        int gn = n0 + tn;   // global N index
                        int gk = k0 + tk;   // global K index
                        if (gn < N && gk < K) {
                            // Quant value: B_qs[n*K + k]
                            B_qs_tile[tn][tk] = B_qs[gn * K + gk];
                            // Scale: B_d[n*(K/32) + k/32], load 1 lần mỗi block
                            if (tk % QK8_0 == 0)
                                B_d_tile[tn][tk / QK8_0] =
                                    B_d[gn * (K / QK8_0) + gk / QK8_0];
                        } else {
                            B_qs_tile[tn][tk] = 0;
                        }
                    }

                // ── Compute: C[m][n] += A[m][k] * (B_qs[n][k] * scale[n][k/32]) ──
                compute:
                for (int tm = 0; tm < TILE_M; ++tm)
                    for (int tk = 0; tk < TILE_K; ++tk) {
                        #pragma HLS PIPELINE II=1
                        float a_val = A_tile[tm][tk];
                        for (int tn = 0; tn < TILE_N; ++tn) {
                            #pragma HLS UNROLL factor=16
                            // FIX: scale index = B_d_tile[tn][tk/QK8_0]
                            float scale     = f16_to_f32(B_d_tile[tn][tk / QK8_0]);
                            float decoded_b = (float)(B_qs_tile[tn][tk]) * scale;
                            C_tile[tm][tn] += a_val * decoded_b;
                        }
                    }

            } // k0

            // ── Store C tile ──
            store_C:
            for (int tm = 0; tm < TILE_M; ++tm)
                for (int tn = 0; tn < TILE_N; ++tn) {
                    #pragma HLS PIPELINE II=1
                    int gr = m0 + tm;
                    int gc = n0 + tn;
                    if (gr < M && gc < N)
                        C[gr * N + gc] = C_tile[tm][tn];
                }

        } // n0
    } // m0
}

} // extern "C"
