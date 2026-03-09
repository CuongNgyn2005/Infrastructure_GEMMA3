#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Khởi tạo một lần khi app start
int  fpga_init(void);
void fpga_cleanup(void);

// Gọi từ ggml_compute_forward_mul_mat
// Trả về 1 nếu thành công, 0 nếu fallback CPU
int fpga_run_matmul(
    const float*    A,      // activation matrix (float32)
    const uint16_t* B_d,    // weight scales (Q8_0 block scale)
    const int8_t*   B_qs,   // weight quants (Q8_0 block data)
    float*          C,      // output matrix
    int M, int K, int N,
    int ith                 // thread id — chỉ chạy nếu ith==0
);

#ifdef __cplusplus
}
#endif