#include "fpga_host.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include "ggml.h"
#include "time.h"
#define QK8_0 32
typedef struct {
    uint16_t d;          // scale (fp16 dưới dạng raw bits)
    int8_t   qs[QK8_0]; // quantized values
} block_q8_0_t;


// ── Địa chỉ từ Vivado Address Editor ──────────────────
#define CTRL_PHYS    0x80000000ULL   // s_axi_control base
#define CTRL_SIZE    0x10000         // 64KB

// ── Register offsets từ csynth.rpt ────────────────────
#define REG_CTRL     0x00
#define REG_A_LO     0x10
#define REG_A_HI     0x14
#define REG_BD_LO    0x1C
#define REG_BD_HI    0x20
#define REG_BQS_LO   0x28
#define REG_BQS_HI   0x2C
#define REG_C_LO     0x34
#define REG_C_HI     0x38
#define REG_M        0x40
#define REG_K        0x48
#define REG_N        0x50

// ── DDR buffer cho FPGA DMA ───────────────────────────
// Dùng vùng an toàn trong Low DDR (tránh Linux heap)
#define BUF_A_PHYS   0x70000000ULL   // 256MB offset
#define BUF_BD_PHYS  0x74000000ULL
#define BUF_BQS_PHYS 0x78000000ULL
#define BUF_C_PHYS   0x7C000000ULL
#define BUF_SIZE     0x4000000       // 64MB mỗi buffer

static int           g_mem_fd  = -1;
static volatile uint32_t* g_ctrl = NULL;
static void*         g_buf_A   = NULL;
static void*         g_buf_Bd  = NULL;
static void*         g_buf_Bqs = NULL;
static void*         g_buf_C   = NULL;
static pthread_mutex_t g_mutex  = PTHREAD_MUTEX_INITIALIZER;

static void wr32(uint32_t off, uint32_t val) {
    g_ctrl[off/4] = val;
}
static uint32_t rd32(uint32_t off) {
    return g_ctrl[off/4];
}

int fpga_init(void) {
    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) { perror("open /dev/mem"); return -1; }

    g_ctrl = (volatile uint32_t*)mmap(
        NULL, CTRL_SIZE, PROT_READ|PROT_WRITE,
        MAP_SHARED, g_mem_fd, CTRL_PHYS);

    g_buf_A   = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE,
                     MAP_SHARED, g_mem_fd, BUF_A_PHYS);
    g_buf_Bd  = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE,
                     MAP_SHARED, g_mem_fd, BUF_BD_PHYS);
    g_buf_Bqs = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE,
                     MAP_SHARED, g_mem_fd, BUF_BQS_PHYS);
    g_buf_C   = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE,
                     MAP_SHARED, g_mem_fd, BUF_C_PHYS);

    if (g_ctrl == MAP_FAILED || g_buf_A == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return -1;
    }
    printf("[FPGA] init OK, ctrl@0x%llx\n", CTRL_PHYS);
    return 0;
}

void fpga_cleanup(void) {
    if (g_ctrl)   munmap((void*)g_ctrl, CTRL_SIZE);
    if (g_buf_A)  munmap(g_buf_A, BUF_SIZE);
    if (g_buf_Bd) munmap(g_buf_Bd, BUF_SIZE);
    if (g_buf_Bqs)munmap(g_buf_Bqs, BUF_SIZE);
    if (g_buf_C)  munmap(g_buf_C, BUF_SIZE);
    if (g_mem_fd >= 0) close(g_mem_fd);
}

int fpga_run_matmul(
    const float* A, const uint16_t* B_d, const int8_t* B_qs,
    float* C, int M, int K, int N, int ith)
{
    if (ith != 0) return 1;  // chỉ thread 0
     if (!g_ctrl || g_ctrl == MAP_FAILED) return 0;
    pthread_mutex_lock(&g_mutex);

    size_t sz_A   = (size_t)M * K * sizeof(float);
    size_t sz_Bd  = (size_t)(K/32) * N * sizeof(uint16_t);
    size_t sz_Bqs = (size_t)K * N * sizeof(int8_t);
    size_t sz_C   = (size_t)M * N * sizeof(float);

    // Copy dữ liệu vào DDR buffer
    memcpy(g_buf_A,   A,   sz_A);
    memcpy(g_buf_Bd,  B_d, sz_Bd);
    memcpy(g_buf_Bqs, B_qs,sz_Bqs);

    // Ghi địa chỉ vật lý vào thanh ghi
    wr32(REG_A_LO,   BUF_A_PHYS & 0xFFFFFFFF);
    wr32(REG_A_HI,   BUF_A_PHYS >> 32);
    wr32(REG_BD_LO,  BUF_BD_PHYS & 0xFFFFFFFF);
    wr32(REG_BD_HI,  BUF_BD_PHYS >> 32);
    wr32(REG_BQS_LO, BUF_BQS_PHYS & 0xFFFFFFFF);
    wr32(REG_BQS_HI, BUF_BQS_PHYS >> 32);
    wr32(REG_C_LO,   BUF_C_PHYS & 0xFFFFFFFF);
    wr32(REG_C_HI,   BUF_C_PHYS >> 32);
    wr32(REG_M, M);
    wr32(REG_K, K);
    wr32(REG_N, N);

    // Kích hoạt kernel
    wr32(REG_CTRL, 0x1);  // AP_START

    // Chờ AP_DONE (bit1=1)
    struct timespec t0, t1;
clock_gettime(CLOCK_MONOTONIC, &t0);
while (!(rd32(REG_CTRL) & 0x2)) {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000
                    + (t1.tv_nsec - t0.tv_nsec) / 1000000;
    if (elapsed_ms > 5000) {
        fprintf(stderr, "[FPGA] TIMEOUT waiting AP_DONE! Fallback CPU.\n");
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }
}

    // Đọc kết quả về
    memcpy(C, g_buf_C, sz_C);

    pthread_mutex_unlock(&g_mutex);
    return 1;
}
int fpga_try_matmul(
    struct ggml_tensor * src0,  // weight (Q8_0) — ma trận B
    struct ggml_tensor * src1,  // activation (F32) — ma trận A
    struct ggml_tensor * dst,   // output (F32) — ma trận C
    int ith)
{
    // ── Bước 1: Chỉ xử lý Q8_0 × F32 → F32 ──────────────────
    if (src0->type != GGML_TYPE_Q8_0) return 0;  // fallback CPU
    if (src1->type != GGML_TYPE_F32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;

    // ── Bước 2: Lấy kích thước ───────────────────────────────
    // src0 shape: [K, N] (weight, Q8_0)
    // src1 shape: [K, M] (activation, F32, transposed khi mul_mat)
    // dst  shape: [N, M]
    const int64_t K = src0->ne[0];  // inner dim
    const int64_t N = src0->ne[1];  // số hàng weight
    const int64_t M = src1->ne[1];  // batch / seq_len

    // Giới hạn kích thước FPGA hỗ trợ (TILE_M=16, TILE_K=64, TILE_N=64)
    if (K > 4096 || N > 4096 || M > 512) return 0;  // fallback CPU

    // ── Bước 3: Bóc tách B_d và B_qs từ block_q8_0 ──────────
    const int num_blocks = (K * N) / QK8_0;
    const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;

    // Alloc tạm trên stack/heap để repack
    static uint16_t B_d_buf [4096 * 4096 / QK8_0];  // scales
    static int8_t   B_qs_buf[4096 * 4096];            // quants

    for (int i = 0; i < num_blocks; i++) {
        B_d_buf[i] = blocks[i].d;
        memcpy(&B_qs_buf[i * QK8_0], blocks[i].qs, QK8_0);
    }

    // ── Bước 4: Gọi fpga_run_matmul ──────────────────────────
    const float* A = (const float*)src1->data;
    float*       C = (float*)dst->data;

    return fpga_run_matmul(A, B_d_buf, B_qs_buf, C,
                           (int)M, (int)K, (int)N, ith);
}