#include "fpga_host.h"
#include "ggml.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>       // OpenMP parallel repack

// ══════════════════════════════════════════════════════════
//  FPGA DEBUG LOG
// ══════════════════════════════════════════════════════════
#define FPGA_LOG_LEVEL_INFO    1
#define FPGA_LOG_LEVEL_MATMUL  1
#define FPGA_LOG_LEVEL_REG     0   // tắt bớt để giảm noise
#define FPGA_LOG_LEVEL_TIMING  1

#define FPGA_LOG_FILE "/tmp/fpga_debug.log"

static FILE* fpga_log_fp(void) {
    static FILE* fp = NULL;
    if (!fp) {
        fp = fopen(FPGA_LOG_FILE, "a");
        if (!fp) fp = stderr;
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        fprintf(fp, "\n═══════════════════════════════════════\n");
        fprintf(fp, "[FPGA] Log started at ts=%ld\n", ts.tv_sec);
        fprintf(fp, "═══════════════════════════════════════\n");
        fflush(fp);
    }
    return fp;
}

static double fpga_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define FPGA_LOG(level_flag, tag, fmt, ...) \
    do { if (level_flag) { \
        FILE* _fp = fpga_log_fp(); \
        fprintf(_fp, "[FPGA][%-6s] " fmt "\n", tag, ##__VA_ARGS__); \
        fflush(_fp); \
    }} while(0)

#define LOGI(fmt, ...) FPGA_LOG(FPGA_LOG_LEVEL_INFO,   "INFO",   fmt, ##__VA_ARGS__)
#define LOGM(fmt, ...) FPGA_LOG(FPGA_LOG_LEVEL_MATMUL, "MATMUL", fmt, ##__VA_ARGS__)
#define LOGR(fmt, ...) FPGA_LOG(FPGA_LOG_LEVEL_REG,    "REG",    fmt, ##__VA_ARGS__)
#define LOGT(fmt, ...) FPGA_LOG(FPGA_LOG_LEVEL_TIMING, "TIMING", fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) FPGA_LOG(1,                      "ERROR",  fmt, ##__VA_ARGS__)

// ══════════════════════════════════════════════════════════
//  Q8_0 block layout
// ══════════════════════════════════════════════════════════
#define QK8_0 32
typedef struct {
    uint16_t d;
    int8_t   qs[QK8_0];
} block_q8_0_t;

// ══════════════════════════════════════════════════════════
//  Địa chỉ phần cứng — KHÔNG ĐỔI so với bitstream cũ
// ══════════════════════════════════════════════════════════
#define CTRL_PHYS    0x400000000ULL
#define CTRL_SIZE    0x10000

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

// ══════════════════════════════════════════════════════════
// Board-Specific Configuration:
// For ZCU104 boards with different memory layouts
// Detects board automatically and selects appropriate addresses
// ══════════════════════════════════════════════════════════

// BOARD ZCU104-01 (2.0 GB RAM, CMA @ 0x6B800000-0x77800000)
// Buffers placed after CMA ends
#define BUF_A_PHYS_B01   0x77C00000ULL  // 64MB
#define BUF_BD_PHYS_B01  0x78C00000ULL  // 64MB
#define BUF_BQS_PHYS_B01 0x79C00000ULL  // 64MB
#define BUF_C_PHYS_B01   0x7AC00000ULL  // 64MB
#define BUF_SIZE_B01     0x4000000       // 64MB

// BOARD ZCU104-02 (1.7 GB RAM, CMA @ 0x60000000-0x70000000)
// Buffers placed BEFORE CMA (smaller 51MB buffers to fit in 1.7GB)
// Addresses: 0x50000000 - 0x5CCC0000 (all 4 buffers before CMA)
#define BUF_A_PHYS_B02   0x50000000ULL  // 51MB
#define BUF_BD_PHYS_B02  0x53300000ULL  // 51MB
#define BUF_BQS_PHYS_B02 0x56600000ULL  // 51MB
#define BUF_C_PHYS_B02   0x59900000ULL  // 51MB
#define BUF_SIZE_B02     0x3300000       // 51MB

// Auto-detect board from DDR size and set addresses
// Check CMA start address from /proc/meminfo or kernel log
static inline void select_board_config(
    uint64_t *buf_a, uint64_t *buf_bd, uint64_t *buf_bqs,
    uint64_t *buf_c, uint64_t *buf_size) {

    // For now: Use compile-time selection or runtime detection
    // Default to B01 (2GB board) - more common
#ifdef USE_ZCU104_01
    // Default: 2GB board with CMA at 0x6B800000
    *buf_a = BUF_A_PHYS_B01;
    *buf_bd = BUF_BD_PHYS_B01;
    *buf_bqs = BUF_BQS_PHYS_B01;
    *buf_c = BUF_C_PHYS_B01;
    *buf_size = BUF_SIZE_B01;
#else
        // 1.7GB board with CMA at 0x60000000
    *buf_a = BUF_A_PHYS_B02;
    *buf_bd = BUF_BD_PHYS_B02;
    *buf_bqs = BUF_BQS_PHYS_B02;
    *buf_c = BUF_C_PHYS_B02;
    *buf_size = BUF_SIZE_B02;
#endif
}

// Default values for backward compatibility
#define BUF_A_PHYS   0x77C00000ULL   // 64MB (Board 01 default)
#define BUF_BD_PHYS  0x78C00000ULL   // 64MB
#define BUF_BQS_PHYS 0x79C00000ULL   // 64MB
#define BUF_C_PHYS   0x7AC00000ULL   // 64MB
#define BUF_SIZE     0x4000000        // 64MB

// Matrix dimension limits based on buffer size
// Board 01 (64MB): Supports larger matrices
#define FPGA_MAX_K_B01   8192
#define FPGA_MAX_N_B01   8192

// Board 02 (51MB): Reduced dimensions to fit smaller buffers
#define FPGA_MAX_K_B02   6144   // Reduced from 8192
#define FPGA_MAX_N_B02   6144   // Reduced from 8192

// Default to Board 01 limits (backward compatible)
#define FPGA_MAX_K   8192
#define FPGA_MAX_N   8192
#define FPGA_MAX_M   512
#define FPGA_MIN_M   8

// ══════════════════════════════════════════════════════════
//  Global state
// ══════════════════════════════════════════════════════════
static int                g_mem_fd   = -1;
static volatile uint32_t* g_ctrl     = NULL;
static void*              g_buf_A    = NULL;
static void*              g_buf_Bd   = NULL;
static void*              g_buf_Bqs  = NULL;
static void*              g_buf_C    = NULL;
static pthread_mutex_t    g_mutex    = PTHREAD_MUTEX_INITIALIZER;
static uint16_t*          g_B_d_buf  = NULL;
static int8_t*            g_B_qs_buf = NULL;
static long long          g_fpga_count = 0;
static long long          g_cpu_count  = 0;

// ── B-cache: tránh copy B_d/B_qs lên DDR khi weight không đổi ──
// Weight của mỗi layer không thay đổi giữa các lần inference
// → chỉ copy lần đầu, các lần sau skip → tiết kiệm ~4ms/call
static const void* g_cached_B_ptr  = NULL;  // src0->data lần trước
static int64_t     g_cached_B_K    = 0;
static int64_t     g_cached_B_N    = 0;

static void wr32(uint32_t off, uint32_t val) { g_ctrl[off/4] = val; }
static uint32_t rd32(uint32_t off)           { return g_ctrl[off/4]; }

// ══════════════════════════════════════════════════════════
//  fpga_init
// ══════════════════════════════════════════════════════════
int fpga_init(void) {
    LOGI("opening /dev/mem...");

    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) {
        LOGE("open /dev/mem failed (cần chạy sudo)");
        perror("[FPGA] open /dev/mem");
        return -1;
    }
    LOGI("/dev/mem opened OK (fd=%d)", g_mem_fd);

    g_ctrl = (volatile uint32_t*)mmap(
        NULL, CTRL_SIZE, PROT_READ|PROT_WRITE,
        MAP_SHARED, g_mem_fd, CTRL_PHYS);
    if (g_ctrl == MAP_FAILED) {
        LOGE("mmap ctrl@0x%08llX failed", (unsigned long long)CTRL_PHYS);
        return -1;
    }
    LOGI("ctrl mmap OK phys=0x%08llX virt=%p",
         (unsigned long long)CTRL_PHYS, (void*)g_ctrl);

    g_buf_A   = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_A_PHYS);
    g_buf_Bd  = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_BD_PHYS);
    g_buf_Bqs = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_BQS_PHYS);
    g_buf_C   = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, BUF_C_PHYS);

    if (g_buf_A   == MAP_FAILED) { LOGE("mmap buf_A failed");   return -1; }
    if (g_buf_Bd  == MAP_FAILED) { LOGE("mmap buf_Bd failed");  return -1; }
    if (g_buf_Bqs == MAP_FAILED) { LOGE("mmap buf_Bqs failed"); return -1; }
    if (g_buf_C   == MAP_FAILED) { LOGE("mmap buf_C failed");   return -1; }

    LOGI("DDR mmap OK: A@0x%08llX Bd@0x%08llX Bqs@0x%08llX C@0x%08llX",
         (unsigned long long)BUF_A_PHYS,  (unsigned long long)BUF_BD_PHYS,
         (unsigned long long)BUF_BQS_PHYS,(unsigned long long)BUF_C_PHYS);

    g_B_d_buf  = (uint16_t*)malloc((FPGA_MAX_K * FPGA_MAX_N / QK8_0) * sizeof(uint16_t));
    g_B_qs_buf = (int8_t*)  malloc( FPGA_MAX_K * FPGA_MAX_N          * sizeof(int8_t));
    if (!g_B_d_buf || !g_B_qs_buf) {
        LOGE("malloc repack buffer failed");
        free(g_B_d_buf);
        return -1;
    }
    LOGI("repack buffers alloc OK");
    LOGI("OpenMP threads available: %d", omp_get_max_threads());

    uint32_t ctrl_val = rd32(REG_CTRL);
    LOGI("REG_CTRL sanity = 0x%08X (mong đợi AP_IDLE=0x4 hoặc 0x6)", ctrl_val);
    if (ctrl_val == 0xFFFFFFFF) {
        LOGE("REG_CTRL=0xFFFFFFFF → AXI không phản hồi! Bitstream chưa nạp?");
    } else if (ctrl_val & 0x4) {
        LOGI("AP_IDLE=1 → kernel sẵn sàng nhận lệnh");
    }

    printf("[FPGA] init OK — ctrl@0x%08llX, DDR bufs@0x%08llX..0x%08llX\n",
           (unsigned long long)CTRL_PHYS,
           (unsigned long long)BUF_A_PHYS,
           (unsigned long long)BUF_C_PHYS + BUF_SIZE - 1);
    LOGI("═══ FPGA INIT COMPLETE ═══");
    return 0;
}

// ══════════════════════════════════════════════════════════
//  fpga_cleanup
// ══════════════════════════════════════════════════════════
void fpga_cleanup(void) {
    LOGI("cleanup — final stats: FPGA=%lld CPU=%lld", g_fpga_count, g_cpu_count);
    if (g_ctrl    && g_ctrl    != MAP_FAILED) munmap((void*)(uintptr_t)g_ctrl, CTRL_SIZE);
    if (g_buf_A   && g_buf_A   != MAP_FAILED) munmap(g_buf_A,   BUF_SIZE);
    if (g_buf_Bd  && g_buf_Bd  != MAP_FAILED) munmap(g_buf_Bd,  BUF_SIZE);
    if (g_buf_Bqs && g_buf_Bqs != MAP_FAILED) munmap(g_buf_Bqs, BUF_SIZE);
    if (g_buf_C   && g_buf_C   != MAP_FAILED) munmap(g_buf_C,   BUF_SIZE);
    if (g_mem_fd  >= 0) close(g_mem_fd);
    free(g_B_d_buf);
    free(g_B_qs_buf);
    g_ctrl = NULL; g_B_d_buf = NULL; g_B_qs_buf = NULL;
    g_cached_B_ptr = NULL;
    LOGI("cleanup done");
}

// ══════════════════════════════════════════════════════════
//  fpga_run_matmul_internal
//
//  M_pad : M gửi FPGA (bội 16)
//  M_real: số hàng thực cần trong C (M_real <= M_pad)
//  b_changed: 1=B đã thay đổi cần copy DDR, 0=skip copy B
// ══════════════════════════════════════════════════════════
static int fpga_run_matmul_internal(
    const float*    A,
    const uint16_t* B_d,
    const int8_t*   B_qs,
    float*          C,
    int M_pad, int M_real,
    int K, int N,
    int b_changed)      // ← tham số mới
{
    pthread_mutex_lock(&g_mutex);

    size_t sz_A      = (size_t)M_pad * K * sizeof(float);
    size_t sz_Bd     = (size_t)(K / QK8_0) * N * sizeof(uint16_t);
    size_t sz_Bqs    = (size_t)K * N * sizeof(int8_t);
    size_t sz_C_pad  = (size_t)M_pad * N * sizeof(float);
    size_t sz_C_real = (size_t)M_real * N * sizeof(float);

    if (sz_A > BUF_SIZE || sz_Bd > BUF_SIZE || sz_Bqs > BUF_SIZE || sz_C_pad > BUF_SIZE) {
        LOGE("buffer overflow! A=%zuKB Bd=%zuKB Bqs=%zuKB C=%zuKB (max=%uKB)",
             sz_A/1024, sz_Bd/1024, sz_Bqs/1024, sz_C_pad/1024, BUF_SIZE/1024);
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }

    // ── Chờ AP_IDLE ──
    double t_idle = fpga_now_ms();
    while (!(rd32(REG_CTRL) & 0x4)) {
        if (fpga_now_ms() - t_idle > 1000.0) {
            LOGE("AP_IDLE timeout (1s) ctrl=0x%08X", rd32(REG_CTRL));
            pthread_mutex_unlock(&g_mutex);
            return 0;
        }
    }

    // ── Copy dữ liệu vào DDR ──
    double t_copy = fpga_now_ms();
    memcpy(g_buf_A, A, sz_A);
    LOGT("memcpy→DDR: A=%zuKB in %.2f ms", sz_A/1024, fpga_now_ms() - t_copy);

    if (b_changed) {
        // B weight thay đổi → copy lên DDR
        double t_b = fpga_now_ms();
        memcpy(g_buf_Bd,  B_d,  sz_Bd);
        memcpy(g_buf_Bqs, B_qs, sz_Bqs);
        LOGT("memcpy→DDR: Bd=%zuKB Bqs=%zuKB (MISS) in %.2f ms",
             sz_Bd/1024, sz_Bqs/1024, fpga_now_ms() - t_b);
    } else {
        LOGT("B-cache HIT — skip copy Bd=%zuKB Bqs=%zuKB",
             sz_Bd/1024, sz_Bqs/1024);
    }

    // ── Ghi register ──
    wr32(REG_A_LO,   (uint32_t)(BUF_A_PHYS   & 0xFFFFFFFF));
    wr32(REG_A_HI,   (uint32_t)(BUF_A_PHYS   >> 32));
    wr32(REG_BD_LO,  (uint32_t)(BUF_BD_PHYS  & 0xFFFFFFFF));
    wr32(REG_BD_HI,  (uint32_t)(BUF_BD_PHYS  >> 32));
    wr32(REG_BQS_LO, (uint32_t)(BUF_BQS_PHYS & 0xFFFFFFFF));
    wr32(REG_BQS_HI, (uint32_t)(BUF_BQS_PHYS >> 32));
    wr32(REG_C_LO,   (uint32_t)(BUF_C_PHYS   & 0xFFFFFFFF));
    wr32(REG_C_HI,   (uint32_t)(BUF_C_PHYS   >> 32));
    wr32(REG_M, (uint32_t)M_pad);
    wr32(REG_K, (uint32_t)K);
    wr32(REG_N, (uint32_t)N);

    // ── AP_START ──
    double t_start = fpga_now_ms();
    wr32(REG_CTRL, 0x1);

    // ── Chờ AP_DONE — timeout 5s ──
    while (!(rd32(REG_CTRL) & 0x2)) {
        if (fpga_now_ms() - t_start > 5000.0) {
            LOGE("AP_DONE TIMEOUT (5s)! M_pad=%d K=%d N=%d ctrl=0x%08X",
                 M_pad, K, N, rd32(REG_CTRL));
            pthread_mutex_unlock(&g_mutex);
            return 0;
        }
    }
    LOGT("kernel exec: %.2f ms (M_pad=%d M_real=%d K=%d N=%d)",
         fpga_now_ms() - t_start, M_pad, M_real, K, N);

    // ── Copy kết quả về — chỉ M_real hàng ──
    double t_read = fpga_now_ms();
    memcpy(C, g_buf_C, sz_C_real);
    LOGT("memcpy←DDR: C_real=%zuKB (M_real=%d) in %.2f ms",
         sz_C_real/1024, M_real, fpga_now_ms() - t_read);
    LOGT("TOTAL fpga_run_matmul: %.2f ms", fpga_now_ms() - t_copy);

    pthread_mutex_unlock(&g_mutex);
    return 1;
}

// ── Wrapper public ──
int fpga_run_matmul(
    const float* A, const uint16_t* B_d, const int8_t* B_qs,
    float* C, int M, int K, int N, int ith)
{
    if (ith != 0) return 1;
    if (!g_ctrl || g_ctrl == MAP_FAILED) return 0;
    return fpga_run_matmul_internal(A, B_d, B_qs, C, M, M, K, N, 1);
}

// ══════════════════════════════════════════════════════════
//  fpga_try_matmul — hook từ ggml
// ══════════════════════════════════════════════════════════
int fpga_try_matmul(
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    int ith)
{
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    if (ith == 0) {
        LOGM("called M=%-4lld K=%-5lld N=%-5lld | src0=%d src1=%d",
             (long long)M, (long long)K, (long long)N,
             src0->type, src1->type);
    }

    // ── Check type ──
    if (src0->type != GGML_TYPE_Q8_0) {
        if (ith == 0) LOGM("SKIP src0 type=%d (need Q8_0=8)", src0->type);
        g_cpu_count++; return 0;
    }
    if (src1->type != GGML_TYPE_F32) {
        if (ith == 0) LOGM("SKIP src1 type=%d (need F32=0)", src1->type);
        g_cpu_count++; return 0;
    }
    if (dst->type != GGML_TYPE_F32) {
        if (ith == 0) LOGM("SKIP dst type=%d (need F32=0)", dst->type);
        g_cpu_count++; return 0;
    }

    // ── Check 2D ──
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1) {
        if (ith == 0) LOGM("SKIP batch ne[2]=%lld/%lld ne[3]=%lld/%lld",
             (long long)src0->ne[2], (long long)src1->ne[2],
             (long long)src0->ne[3], (long long)src1->ne[3]);
        g_cpu_count++; return 0;
    }

    // ── Check K, N align ──
    if (K % 64 != 0 || N % 64 != 0) {
        if (ith == 0) LOGM("SKIP K%%64=%lld N%%64=%lld → CPU",
             (long long)(K%64), (long long)(N%64));
        g_cpu_count++; return 0;
    }

    // ── Check size tuyệt đối ──
    if (K > FPGA_MAX_K || N > FPGA_MAX_N || M > FPGA_MAX_M) {
        if (ith == 0) LOGM("SKIP size M=%lld K=%lld N=%lld > max(%d,%d,%d)",
             (long long)M, (long long)K, (long long)N,
             FPGA_MAX_M, FPGA_MAX_K, FPGA_MAX_N);
        g_cpu_count++; return 0;
    }

    // ── Check M tối thiểu ──
    if (M < FPGA_MIN_M) {
        if (ith == 0) LOGM("SKIP M=%lld < %d (decode) → CPU",
             (long long)M, FPGA_MIN_M);
        g_cpu_count++; return 0;
    }

    // ── Tính M_pad ──
    const int64_t M_pad = ((M + 15) / 16) * 16;
    if (M_pad > FPGA_MAX_M) {
        if (ith == 0) LOGM("SKIP M_pad=%lld > FPGA_MAX_M=%d", (long long)M_pad, FPGA_MAX_M);
        g_cpu_count++; return 0;
    }

    if (ith == 0) {
        if (M_pad != M)
            LOGM("M=%lld → pad→%lld", (long long)M, (long long)M_pad);
    }

    // Thread phụ return sớm
    if (ith != 0) return 1;

    // ══ Thread 0 ══

    LOGM(">>> FPGA #%lld: M=%lld(pad→%lld) K=%lld N=%lld",
         g_fpga_count + 1, (long long)M, (long long)M_pad,
         (long long)K, (long long)N);

    // ── B-cache check ──
    // So sánh con trỏ src0->data, K, N để biết weight có đổi không
    const void* cur_B_ptr = src0->data;
    int b_changed = (cur_B_ptr != g_cached_B_ptr ||
                     K != g_cached_B_K ||
                     N != g_cached_B_N) ? 1 : 0;

    if (b_changed) {
        LOGT("B-cache MISS (ptr=%p K=%lld N=%lld) → repack + copy DDR",
             cur_B_ptr, (long long)K, (long long)N);
    } else {
        LOGT("B-cache HIT (same weight) → skip repack + DDR copy");
    }

    // ── Repack Q8_0 với OpenMP (chỉ khi B thay đổi) ──
    if (b_changed) {
        double t_repack = fpga_now_ms();
        const int num_blocks = (int)((K * N) / QK8_0);
        const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;

        // OpenMP parallel: chia đều num_blocks cho 4 threads
        // Mỗi thread xử lý range riêng → không có race condition
        #pragma omp parallel for schedule(static) num_threads(4)
        for (int i = 0; i < num_blocks; i++) {
            g_B_d_buf[i] = blocks[i].d;
            // Dùng direct assignment thay memcpy cho QK8_0=32 bytes
            // Compiler sẽ vector hóa với ARM NEON
            const int8_t* src_qs = blocks[i].qs;
            int8_t*       dst_qs = &g_B_qs_buf[i * QK8_0];
            for (int j = 0; j < QK8_0; j++)
                dst_qs[j] = src_qs[j];
        }

        LOGT("repack %d blocks (OpenMP 4T) in %.2f ms",
             num_blocks, fpga_now_ms() - t_repack);

        // Cập nhật cache key
        g_cached_B_ptr = cur_B_ptr;
        g_cached_B_K   = K;
        g_cached_B_N   = N;
    }

    // ── Tạo A_pad nếu M không align ──
    const float* A_src    = (const float*)src1->data;
    float*       A_use    = (float*)A_src;
    float*       A_pad_buf = NULL;

    if (M_pad != M) {
        double t_pad = fpga_now_ms();
        A_pad_buf = (float*)calloc(M_pad * K, sizeof(float));
        if (!A_pad_buf) {
            LOGE("calloc A_pad failed → CPU fallback");
            g_cpu_count++; return 0;
        }
        memcpy(A_pad_buf, A_src, (size_t)M * K * sizeof(float));
        A_use = A_pad_buf;
        LOGT("A_pad alloc+memcpy (%zuKB) in %.2f ms",
             (size_t)M_pad * K * sizeof(float) / 1024,
             fpga_now_ms() - t_pad);
    }

    // ── Gọi FPGA ──
    float* C = (float*)dst->data;
    int ret = fpga_run_matmul_internal(
        A_use, g_B_d_buf, g_B_qs_buf, C,
        (int)M_pad, (int)M,
        (int)K, (int)N,
        b_changed);

    free(A_pad_buf);

    if (ret) {
        g_fpga_count++;
        LOGM("<<< FPGA OK — total: FPGA=%lld CPU=%lld", g_fpga_count, g_cpu_count);
    } else {
        g_cpu_count++;
        LOGM("<<< FPGA FAILED → CPU — total: FPGA=%lld CPU=%lld", g_fpga_count, g_cpu_count);
    }
    return ret;
}