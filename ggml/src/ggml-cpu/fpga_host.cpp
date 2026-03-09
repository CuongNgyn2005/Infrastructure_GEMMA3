#include "fpga_host.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include "ggml.h"

// ── Q8_0 block layout (khớp với ggml internal) ────────
#define QK8_0 32
typedef struct {
    uint16_t d;           // scale (fp16 raw bits)
    int8_t   qs[QK8_0];  // quantized values
} block_q8_0_t;

// ── Địa chỉ từ Vivado Address Editor ──────────────────
#define CTRL_PHYS    0x80000000ULL
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

// ── DDR buffer — vùng an toàn trong Low DDR ───────────
// 4 vùng × 64MB = 256MB tổng, nằm ở upper 512MB (0x70000000~0x7FFFFFFF)
// Linux kernel thường dùng 0x00000000~0x6FFFFFFF cho heap/stack
#define BUF_A_PHYS   0x70000000ULL
#define BUF_BD_PHYS  0x74000000ULL
#define BUF_BQS_PHYS 0x78000000ULL
#define BUF_C_PHYS   0x7C000000ULL
#define BUF_SIZE     0x4000000       // 64MB mỗi buffer

// ── Giới hạn kích thước FPGA kernel ──────────────────
// Khớp với TILE_M=16, TILE_K=64, TILE_N=64 trong HLS
#define FPGA_MAX_K   4096
#define FPGA_MAX_N   4096
#define FPGA_MAX_M   512

static int                g_mem_fd = -1;
static volatile uint32_t* g_ctrl   = NULL;
static void*              g_buf_A   = NULL;
static void*              g_buf_Bd  = NULL;
static void*              g_buf_Bqs = NULL;
static void*              g_buf_C   = NULL;
static pthread_mutex_t    g_mutex   = PTHREAD_MUTEX_INITIALIZER;

// ── Repack buffer — heap alloc khi init, tránh 17MB BSS ──
// FIX: static buffer bị ghi đồng thời bởi 4 threads trước mutex
// → dùng heap, chỉ alloc 1 lần, protected bởi mutex
static uint16_t* g_B_d_buf  = NULL;  // max K*N/32 entries
static int8_t*   g_B_qs_buf = NULL;  // max K*N entries

static void wr32(uint32_t off, uint32_t val) { g_ctrl[off/4] = val; }
static uint32_t rd32(uint32_t off)           { return g_ctrl[off/4]; }

// ─────────────────────────────────────────────────────────
int fpga_init(void) {
    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) { perror("[FPGA] open /dev/mem"); return -1; }

    g_ctrl = (volatile uint32_t*)mmap(
        NULL, CTRL_SIZE, PROT_READ|PROT_WRITE,
        MAP_SHARED, g_mem_fd, CTRL_PHYS);

    g_buf_A   = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_A_PHYS);
    g_buf_Bd  = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_BD_PHYS);
    g_buf_Bqs = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_BQS_PHYS);
    g_buf_C   = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_C_PHYS);

    if (g_ctrl   == MAP_FAILED || g_buf_A   == MAP_FAILED ||
        g_buf_Bd == MAP_FAILED || g_buf_Bqs == MAP_FAILED ||
        g_buf_C  == MAP_FAILED) {
        fprintf(stderr, "[FPGA] mmap failed\n");
        return -1;
    }

    // Alloc repack buffers trên heap (thay vì 17MB static BSS)
    // Max size: K=4096, N=4096 → 4096*4096/32 scales + 4096*4096 quants
    g_B_d_buf  = (uint16_t*)malloc((FPGA_MAX_K * FPGA_MAX_N / QK8_0) * sizeof(uint16_t));
    g_B_qs_buf = (int8_t*)  malloc( FPGA_MAX_K * FPGA_MAX_N          * sizeof(int8_t));
    if (!g_B_d_buf || !g_B_qs_buf) {
        fprintf(stderr, "[FPGA] malloc repack buffer failed\n");
        return -1;
    }

    printf("[FPGA] init OK — ctrl@0x%llx, DDR bufs@0x%llx..0x%llx\n",
           (unsigned long long)CTRL_PHYS,
           (unsigned long long)BUF_A_PHYS,
           (unsigned long long)BUF_C_PHYS + BUF_SIZE - 1);
    return 0;
}

void fpga_cleanup(void) {
    if (g_ctrl   && g_ctrl   != MAP_FAILED) munmap((void*)g_ctrl,   CTRL_SIZE);
    if (g_buf_A  && g_buf_A  != MAP_FAILED) munmap(g_buf_A,  BUF_SIZE);
    if (g_buf_Bd && g_buf_Bd != MAP_FAILED) munmap(g_buf_Bd, BUF_SIZE);
    if (g_buf_Bqs&& g_buf_Bqs!= MAP_FAILED) munmap(g_buf_Bqs,BUF_SIZE);
    if (g_buf_C  && g_buf_C  != MAP_FAILED) munmap(g_buf_C,  BUF_SIZE);
    if (g_mem_fd >= 0) close(g_mem_fd);
    free(g_B_d_buf);
    free(g_B_qs_buf);
    g_ctrl = NULL; g_B_d_buf = NULL; g_B_qs_buf = NULL;
}

// ─────────────────────────────────────────────────────────
int fpga_run_matmul(
    const float* A, const uint16_t* B_d, const int8_t* B_qs,
    float* C, int M, int K, int N, int ith)
{
    // Chỉ thread 0 điều khiển FPGA — các thread khác trả về 1 (đã xử lý)
    if (ith != 0) return 1;

    // Guard: đảm bảo đã init
    if (!g_ctrl || g_ctrl == MAP_FAILED) return 0;

    pthread_mutex_lock(&g_mutex);

    size_t sz_A   = (size_t)M * K * sizeof(float);
    size_t sz_Bd  = (size_t)(K / QK8_0) * N * sizeof(uint16_t);
    size_t sz_Bqs = (size_t)K * N * sizeof(int8_t);
    size_t sz_C   = (size_t)M * N * sizeof(float);

    // Kiểm tra không tràn buffer DDR 64MB
    if (sz_A > BUF_SIZE || sz_Bd > BUF_SIZE || sz_Bqs > BUF_SIZE || sz_C > BUF_SIZE) {
        fprintf(stderr, "[FPGA] Buffer overflow! A=%zuB Bd=%zuB Bqs=%zuB C=%zuB (max=%u)\n",
                sz_A, sz_Bd, sz_Bqs, sz_C, BUF_SIZE);
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }

    // Copy dữ liệu vào DDR physical buffers
    memcpy(g_buf_A,   A,   sz_A);
    memcpy(g_buf_Bd,  B_d, sz_Bd);
    memcpy(g_buf_Bqs, B_qs,sz_Bqs);

    // Ghi địa chỉ + kích thước vào thanh ghi AXI-Lite
    wr32(REG_A_LO,   (uint32_t)(BUF_A_PHYS & 0xFFFFFFFF));
    wr32(REG_A_HI,   (uint32_t)(BUF_A_PHYS >> 32));
    wr32(REG_BD_LO,  (uint32_t)(BUF_BD_PHYS & 0xFFFFFFFF));
    wr32(REG_BD_HI,  (uint32_t)(BUF_BD_PHYS >> 32));
    wr32(REG_BQS_LO, (uint32_t)(BUF_BQS_PHYS & 0xFFFFFFFF));
    wr32(REG_BQS_HI, (uint32_t)(BUF_BQS_PHYS >> 32));
    wr32(REG_C_LO,   (uint32_t)(BUF_C_PHYS & 0xFFFFFFFF));
    wr32(REG_C_HI,   (uint32_t)(BUF_C_PHYS >> 32));
    wr32(REG_M, (uint32_t)M);
    wr32(REG_K, (uint32_t)K);
    wr32(REG_N, (uint32_t)N);

    // AP_START
    wr32(REG_CTRL, 0x1);

    // Chờ AP_DONE (bit1) — timeout 5 giây
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (!(rd32(REG_CTRL) & 0x2)) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long elapsed_ms = (t1.tv_sec  - t0.tv_sec)  * 1000L
                        + (t1.tv_nsec - t0.tv_nsec) / 1000000L;
        if (elapsed_ms > 5000) {
            fprintf(stderr, "[FPGA] TIMEOUT AP_DONE! M=%d K=%d N=%d → fallback CPU\n", M, K, N);
            pthread_mutex_unlock(&g_mutex);
            return 0;
        }
    }

    // Đọc kết quả từ DDR về CPU memory
    memcpy(C, g_buf_C, sz_C);

    pthread_mutex_unlock(&g_mutex);
    return 1;
}

// ─────────────────────────────────────────────────────────
int fpga_try_matmul(
    struct ggml_tensor * src0,   // weight B  (Q8_0)
    struct ggml_tensor * src1,   // activation A (F32)
    struct ggml_tensor * dst,    // output C (F32)
    int ith)
{
    // ── Bước 1: Kiểm tra kiểu dữ liệu ───────────────────
    if (src0->type != GGML_TYPE_Q8_0) return 0;
    if (src1->type != GGML_TYPE_F32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;

    // ── Bước 2: Kích thước ───────────────────────────────
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    // Fallback CPU nếu vượt giới hạn FPGA kernel
    if (K > FPGA_MAX_K || N > FPGA_MAX_N || M > FPGA_MAX_M) {
        // In lần đầu để biết layer nào bị skip
        static int skip_logged = 0;
        if (!skip_logged) {
            fprintf(stderr, "[FPGA] skip (K=%lld N=%lld M=%lld > limit) → CPU\n",
                    (long long)K, (long long)N, (long long)M);
            skip_logged = 1;
        }
        return 0;
    }

    // ── Bước 3: Thread phụ trả về sớm ───────────────────
    // FIX: phải check ith TRƯỚC khi repack để tránh 4 threads
    // ghi đồng thời vào g_B_d_buf / g_B_qs_buf (race condition)
    if (ith != 0) return 1;  // thread 0 sẽ lo phần này

    // ── Bước 4: Repack Q8_0 → B_d + B_qs (chỉ thread 0) ─
    const int num_blocks = (int)((K * N) / QK8_0);
    const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;

    for (int i = 0; i < num_blocks; i++) {
        g_B_d_buf[i] = blocks[i].d;
        memcpy(&g_B_qs_buf[i * QK8_0], blocks[i].qs, QK8_0);
    }

    // ── Bước 5: Gọi FPGA kernel ──────────────────────────
    const float* A = (const float*)src1->data;
    float*       C = (float*)dst->data;

    return fpga_run_matmul(A, g_B_d_buf, g_B_qs_buf, C,
                           (int)M, (int)K, (int)N, ith);
}