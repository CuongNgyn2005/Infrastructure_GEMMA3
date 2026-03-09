#include "fpga_host.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>

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
    while (!(rd32(REG_CTRL) & 0x2));

    // Đọc kết quả về
    memcpy(C, g_buf_C, sz_C);

    pthread_mutex_unlock(&g_mutex);
    return 1;
}