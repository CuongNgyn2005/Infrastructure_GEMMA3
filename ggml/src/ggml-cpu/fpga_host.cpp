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
#include <cmath>       // for isnan(), isinf()
// Fallback converter FP16 -> FP32
static inline float safe_fp16_to_fp32(uint16_t h) {
    uint32_t s = (uint32_t)((h >> 15) & 1U);
    uint32_t e = (uint32_t)((h >> 10) & 0x1FU);
    uint32_t m = (uint32_t)(h & 0x3FFU);
    uint32_t b;
    if      (e == 0U)  b = s << 31;
    else if (e == 31U) b = (s << 31) | 0x7F800000U | (m << 13);
    else               b = (s << 31) | ((e + 112U) << 23) | (m << 13);
    union { uint32_t i; float f; } u; u.i = b; return u.f;
}

// Fallback converter FP32 -> FP16
static inline uint16_t safe_fp32_to_fp16(float f) {
    union { float f; uint32_t i; } u; u.f = f;
    uint32_t i = u.i;
    uint32_t s = (i >> 16) & 0x8000;
    int e = ((i >> 23) & 0xff) - 127 + 15;
    uint32_t m = (i >> 13) & 0x3ff;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | m;
}
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

#define QK8_0 32
typedef struct {
    uint16_t d;
    int8_t   qs[QK8_0];
} block_q8_0_t;

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
#define REG_LAYER_ID 0x58   // Layer identifier
#define REG_SEQ_POS  0x60   // Sequence position
#define REG_IS_ATTN  0x68   // Attention flag

// ══════════════════════════════════════════════════════════
// Board-Specific Configuration: ZCU104 (mem=1792M)
// Vùng Nhớ: An toàn 0x70000000 - 0x7FFFFFFF (tránh kernel Linux)
// ══════════════════════════════════════════════════════════
#define BUF_A_PHYS       0x70000000ULL  // Buf A:   70000000 -> 74000000
#define BUF_BD_PHYS      0x74000000ULL  // Buf Bd:  74000000 -> 78000000
#define BUF_BQS_PHYS     0x78000000ULL  // Buf Bqs: 78000000 -> 7C000000
#define BUF_C_PHYS       0x7C000000ULL  // Buf C:   7C000000 -> 80000000
#define BUF_SIZE         0x4000000      // 64MB mỗi mảng

#define FPGA_MAX_K   8192
#define FPGA_MAX_N   8192
#define FPGA_MAX_M   512
#define FPGA_MIN_M   8

static int                g_mem_fd   = -1;
static volatile uint32_t* g_ctrl     = NULL;
static void* g_buf_A    = NULL;
static void* g_buf_Bd   = NULL;
static void* g_buf_Bqs  = NULL;
static void* g_buf_C    = NULL;
static pthread_mutex_t    g_mutex    = PTHREAD_MUTEX_INITIALIZER;
static uint16_t* g_B_d_buf  = NULL;
static int8_t* g_B_qs_buf = NULL;
static long long          g_fpga_count = 0;
static long long          g_cpu_count  = 0;

static uint64_t    g_cached_B_key  = 0;
static int64_t     g_cached_B_K    = 0;
static int64_t     g_cached_B_N    = 0;

static uint64_t g_buf_A_phys   = BUF_A_PHYS;
static uint64_t g_buf_BD_phys  = BUF_BD_PHYS;
static uint64_t g_buf_BQS_phys = BUF_BQS_PHYS;
static uint64_t g_buf_C_phys   = BUF_C_PHYS;
static uint64_t g_buf_size     = BUF_SIZE;
static int64_t  g_fpga_max_k   = FPGA_MAX_K;
static int64_t  g_fpga_max_n   = FPGA_MAX_N;

static void wr32(uint32_t off, uint32_t val) { g_ctrl[off/4] = val; }
static uint32_t rd32(uint32_t off)           { return g_ctrl[off/4]; }

int fpga_init(void) {
    LOGI("opening /dev/mem...");
    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) {
        LOGE("open /dev/mem failed (cần chạy sudo)");
        return -1;
    }
    LOGI("/dev/mem opened OK (fd=%d)", g_mem_fd);

    LOGI("Configured for Board (mem=1792M): Safe hidden RAM [0x70000000 - 0x7FFFFFFF]");

    g_ctrl = (volatile uint32_t*)mmap(NULL, CTRL_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, CTRL_PHYS);
    if (g_ctrl == MAP_FAILED) return -1;

    g_buf_A   = mmap(NULL, g_buf_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, g_buf_A_phys);
    g_buf_Bd  = mmap(NULL, g_buf_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, g_buf_BD_phys);
    g_buf_Bqs = mmap(NULL, g_buf_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, g_buf_BQS_phys);
    g_buf_C   = mmap(NULL, g_buf_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, g_buf_C_phys);

    g_B_d_buf  = (uint16_t*)malloc((FPGA_MAX_K * FPGA_MAX_N / QK8_0) * sizeof(uint16_t));
    g_B_qs_buf = (int8_t*)  malloc( FPGA_MAX_K * FPGA_MAX_N          * sizeof(int8_t));

    uint32_t ctrl_val = rd32(REG_CTRL);
    LOGI("REG_CTRL sanity = 0x%08X", ctrl_val);
    
    printf("[FPGA] init OK — ctrl@0x%08llX, DDR bufs@0x%llX..0x%llX (%luMB)\n",
           (unsigned long long)CTRL_PHYS, (unsigned long long)g_buf_A_phys,
           (unsigned long long)g_buf_C_phys + g_buf_size - 1, g_buf_size / (1024*1024));
    return 0;
}

void fpga_cleanup(void) {
    if (g_ctrl && g_ctrl != MAP_FAILED) munmap((void*)(uintptr_t)g_ctrl, CTRL_SIZE);
    if (g_buf_A && g_buf_A != MAP_FAILED) munmap(g_buf_A, g_buf_size);
    if (g_buf_Bd && g_buf_Bd != MAP_FAILED) munmap(g_buf_Bd, g_buf_size);
    if (g_buf_Bqs && g_buf_Bqs != MAP_FAILED) munmap(g_buf_Bqs, g_buf_size);
    if (g_buf_C && g_buf_C != MAP_FAILED) munmap(g_buf_C, g_buf_size);
    if (g_mem_fd >= 0) close(g_mem_fd);
    free(g_B_d_buf); free(g_B_qs_buf);
}
// Context tracking for current operation
static int g_current_layer_id = 0;
 int g_current_seq_pos = 0;
static int g_is_attention_op = 0;

void fpga_set_context(int layer_id, int seq_pos, int is_attn) {
    g_current_layer_id = layer_id;
    g_current_seq_pos = seq_pos;
    g_is_attention_op = is_attn;
}
static int fpga_run_matmul_internal(
    const float* A,
    const uint16_t* B_d,
    const int8_t* B_qs,
    float* C,
    int M_pad, int M_real,
    int K, int N,
    int b_changed)
{
    //pthread_mutex_lock(&g_mutex);

    size_t sz_A      = (size_t)M_pad * K * sizeof(float);
    size_t sz_Bd     = (size_t)(K / QK8_0) * N * sizeof(uint16_t);
    size_t sz_Bqs    = (size_t)K * N * sizeof(int8_t);
    size_t sz_C_real = (size_t)M_real * N * sizeof(float);

    LOGT("=============================================");
    LOGT("[DEBUG DATA] Soi data HOST gui di (M_pad=%d, K=%d, N=%d)", M_pad, K, N);
    
    const float* dbg_A = (const float*)A;
    int a_has_nan = 0;
    for(size_t i = 0; i < (size_t)M_pad * K; i++) {
        if (std::isnan(dbg_A[i]) || std::isinf(dbg_A[i])) { a_has_nan = 1; break; }
    }
    if (a_has_nan) { LOGE("[FATAL ERROR] Ma tran A da chua san NaN tu CPU truyen xuong!"); } 
    else { LOGT("  [OK] Ma tran A sach se, khong co NaN."); }
    
    LOGT("  A[0..7]: %f, %f, %f, %f, %f, %f, %f, %f",
         dbg_A[0], dbg_A[1], dbg_A[2], dbg_A[3], dbg_A[4], dbg_A[5], dbg_A[6], dbg_A[7]);

    if (b_changed) {
        const int8_t* dbg_Bqs = (const int8_t*)B_qs;
        LOGT("  B_qs[0..7]: %d, %d, %d, %d, %d, %d, %d, %d",
             dbg_Bqs[0], dbg_Bqs[1], dbg_Bqs[2], dbg_Bqs[3], dbg_Bqs[4], dbg_Bqs[5], dbg_Bqs[6], dbg_Bqs[7]);

        const uint16_t* dbg_Bd = (const uint16_t*)B_d;
        LOGT("  B_d[0..3] (Hex fp16): 0x%04X, 0x%04X, 0x%04X, 0x%04X",
             dbg_Bd[0], dbg_Bd[1], dbg_Bd[2], dbg_Bd[3]);
    }
    LOGT("=============================================");

    while (!(rd32(REG_CTRL) & 0x4)) { }

    memcpy(g_buf_A, A, sz_A);

    if (b_changed) {
        memcpy(g_buf_Bd,  B_d,  sz_Bd);
        memcpy(g_buf_Bqs, B_qs, sz_Bqs);
    }

    __sync_synchronize();
    wr32(REG_A_LO,   (uint32_t)(g_buf_A_phys   & 0xFFFFFFFF));
    wr32(REG_A_HI,   (uint32_t)(g_buf_A_phys   >> 32));
    wr32(REG_BD_LO,  (uint32_t)(g_buf_BD_phys  & 0xFFFFFFFF));
    wr32(REG_BD_HI,  (uint32_t)(g_buf_BD_phys  >> 32));
    wr32(REG_BQS_LO, (uint32_t)(g_buf_BQS_phys & 0xFFFFFFFF));
    wr32(REG_BQS_HI, (uint32_t)(g_buf_BQS_phys >> 32));
    wr32(REG_C_LO,   (uint32_t)(g_buf_C_phys   & 0xFFFFFFFF));
    wr32(REG_C_HI,   (uint32_t)(g_buf_C_phys   >> 32));
    wr32(REG_M, (uint32_t)M_pad);
    wr32(REG_K, (uint32_t)K);
    wr32(REG_N, (uint32_t)N);

    wr32(REG_CTRL, 0x1);
    int timeout = 5000000; 
    while (!(rd32(REG_CTRL) & 0x2)) {
        timeout--;
        if (timeout <= 0) {
            fprintf(stderr, "[FPGA ERROR] HARDWARE TIMEOUT! REG_CTRL = 0x%08X\n", rd32(REG_CTRL));
            break; // Ép thoát vòng lặp để không treo máy
        }
    }

    memcpy(C, g_buf_C, sz_C_real);
    
    
    LOGE("--- KET QUA TU FPGA ---");
    int has_nan_inf = 0;
    for (int i = 0; i < 8; i++) {
        float val = C[i];
        if (std::isnan(val) || std::isinf(val)) has_nan_inf = 1;
        LOGE("C[%d] = %f", i, val);
    }
    if (has_nan_inf) LOGE("[DEBUG] Output tra ve co NaN/Inf!");

    //pthread_mutex_unlock(&g_mutex);
    return 1;
}

extern "C" int fpga_try_matmul(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ith)
{
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    if (src0->type != GGML_TYPE_Q8_0) { g_cpu_count++; return 0; }
    if (src1->type != GGML_TYPE_F32)  { g_cpu_count++; return 0; }
    if (dst->type  != GGML_TYPE_F32)  { g_cpu_count++; return 0; }
    if (K % 64 != 0 || N % 64 != 0)   { g_cpu_count++; return 0; }
    if (K > g_fpga_max_k || N > g_fpga_max_n || M > FPGA_MAX_M) { g_cpu_count++; return 0; }
    if (M < FPGA_MIN_M) { g_cpu_count++; return 0; }

    const int64_t M_pad = ((M + 15) / 16) * 16;
    if (M_pad > FPGA_MAX_M) { g_cpu_count++; return 0; }

    if (ith != 0) return 1;

    uint64_t cur_B_key = ((uintptr_t)src0->data) ^ (uint64_t)K ^ (uint64_t)N;
    int b_changed = (cur_B_key != g_cached_B_key || K != g_cached_B_K || N != g_cached_B_N) ? 1 : 0;

    if (b_changed) {
        const int num_blocks = (int)((K * N) / QK8_0);
        const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;

        #pragma omp parallel for schedule(static) num_threads(4)
        for (int i = 0; i < num_blocks; i++) {
            g_B_d_buf[i] = blocks[i].d;
            const int8_t* src_qs = blocks[i].qs;
            int8_t* dst_qs = &g_B_qs_buf[i * QK8_0];
            for (int j = 0; j < QK8_0; j++)
                dst_qs[j] = src_qs[j];
        }
        g_cached_B_key = cur_B_key;
        g_cached_B_K   = K;
        g_cached_B_N   = N;
    }

    const float* A_src = (const float*)src1->data;
    const float* A_use = A_src;
    float* A_pad_buf = NULL;

    if (M_pad != M) {
        A_pad_buf = (float*)calloc(M_pad * K, sizeof(float));
        memcpy(A_pad_buf, A_src, (size_t)M * K * sizeof(float));
        A_use = (const float*)A_pad_buf;
    }

    float* C = (float*)dst->data;
    int ret = fpga_run_matmul_internal(A_use, g_B_d_buf, g_B_qs_buf, C, (int)M_pad, (int)M, (int)K, (int)N, b_changed);

    free(A_pad_buf);
    if (ret) g_fpga_count++; else g_cpu_count++;
    return ret;
}
// ===================================================================
// HÀM CHẠY ATTENTION (IS_ATTN = 1)
// ===================================================================
static int fpga_run_attention(
    const float* Q,
    const uint16_t* K_scales,
    const int8_t* K_V_data,
    float* output,
    int head_dim,  // <--- Đã sửa: Truyền head_dim thay vì M, K, N
    int layer_id,
    int seq_pos,
    int is_attn)
{
    // 1. Ghi tham số ngữ cảnh
    wr32(REG_LAYER_ID, (uint32_t)layer_id);
    wr32(REG_SEQ_POS,  (uint32_t)seq_pos);
    wr32(REG_IS_ATTN,  (uint32_t)is_attn);
    
    // 2. Chuẩn bị dữ liệu Q, K, V (Động theo head_dim)
    memcpy(g_buf_A, Q, head_dim * sizeof(float));
    memcpy(g_buf_Bqs, K_V_data, 2 * head_dim * sizeof(int8_t));
    memcpy(g_buf_Bd, K_scales, 2 * sizeof(uint16_t));
    
    __sync_synchronize();

    // 3. Truyền địa chỉ
    wr32(REG_A_LO,   (uint32_t)(g_buf_A_phys   & 0xFFFFFFFF));
    wr32(REG_A_HI,   (uint32_t)(g_buf_A_phys   >> 32));
    wr32(REG_BD_LO,  (uint32_t)(g_buf_BD_phys  & 0xFFFFFFFF));
    wr32(REG_BD_HI,  (uint32_t)(g_buf_BD_phys  >> 32));
    wr32(REG_BQS_LO, (uint32_t)(g_buf_BQS_phys & 0xFFFFFFFF));
    wr32(REG_BQS_HI, (uint32_t)(g_buf_BQS_phys >> 32));
    wr32(REG_C_LO,   (uint32_t)(g_buf_C_phys   & 0xFFFFFFFF));
    wr32(REG_C_HI,   (uint32_t)(g_buf_C_phys   >> 32));
    
    // MƯỢN THANH GHI K ĐỂ TRUYỀN HEAD_DIM XUỐNG MẠCH CỨNG
    wr32(REG_M, 1);
    wr32(REG_K, (uint32_t)head_dim);
    wr32(REG_N, 1);

    // 4. Kích hoạt FPGA
    wr32(REG_CTRL, 0x1);
    int timeout = 5000000; 
    while (!(rd32(REG_CTRL) & 0x2)) {
        timeout--;
        if (timeout <= 0) {
            fprintf(stderr, "[FPGA ERROR] HARDWARE TIMEOUT! REG_CTRL = 0x%08X\n", rd32(REG_CTRL));
            break;
        }
    }

    // 5. Đọc kết quả
    memcpy(output, g_buf_C, head_dim * sizeof(float));
    
    return 1;
}
// ===================================================================
// HÀM MỞ RỘNG ĐỂ GIAO TIẾP VỚI GGML-CPU.C
// ===================================================================
extern "C" int fpga_try_matmul_extended(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ith,
    int layer_id,
    int seq_pos,
    int is_attention)
{
   if (!src0 || !src1 || !dst) return 0;

    // Tính toán kích thước thật từ tensor
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const int64_t M_pad = ((M + 15) / 16) * 16;

    // =========================================================
    // 1. KIỂM TRA ĐIỀU KIỆN TỪ CHỐI (DÀNH CHO TẤT CẢ CÁC THREAD)
    // Nếu rớt bài kiểm tra này, toàn bộ các Thread phải return 0
    // =========================================================
    if (is_attention == 0) {
        // Chỉ kiểm tra gắt gao với MatMul (Nhánh 1)
        if (src0->type != GGML_TYPE_Q8_0) return 0;
        if (src1->type != GGML_TYPE_F32)  return 0;
        if (dst->type  != GGML_TYPE_F32)  return 0;
        if (K % 64 != 0 || N % 64 != 0)   return 0;
        if (K > g_fpga_max_k || N > g_fpga_max_n || M > FPGA_MAX_M) return 0;
        if (M < FPGA_MIN_M) return 0;
    }
    
    if (M_pad > FPGA_MAX_M) return 0; // Áp dụng cho cả MatMul và Attention

    // =========================================================
    // 2. CHỐT CHẠY FPGA. Cho phép các Thread phụ đi ngủ.
    // =========================================================
    if (ith != 0) return 1; 

    // Chỉ còn lại duy nhất Thread 0 ở đây
    pthread_mutex_lock(&g_mutex);
    fpga_set_context(layer_id, seq_pos, is_attention);
    
    fprintf(stderr, "\n[FPGA HOST] >>> Thread 0 Locked Mutex. is_attn=%d, layer_id=%d, M=%d, K=%d, N=%d\n", 
            is_attention, layer_id, (int)M_pad, (int)K, (int)N);

    if (is_attention == 1) {
        // --- NHÁNH ATTENTION - CÁCH 2: LƯỢNG TỬ HÓA ON-THE-FLY ---
        
        // Trong Llama.cpp, dst chính là Node FLASH_ATTN_EXT
        // Nó chứa: src[0] = Q, src[1] = K, src[2] = V
        const struct ggml_tensor * Q_tensor = dst->src[0];
        const struct ggml_tensor * K_tensor = dst->src[1];
        const struct ggml_tensor * V_tensor = dst->src[2];

        // Lấy con trỏ thô
        const float* Q = (const float*)Q_tensor->data;
        const uint16_t* K_data = (const uint16_t*)K_tensor->data;
        const uint16_t* V_data = (const uint16_t*)V_tensor->data;

        int head_dim = Q_tensor->ne[0]; 
        if (head_dim > 512) return 0;
        // Trích xuất Token mới nhất (Luôn nằm ở cuối mảng K, V)
        int current_idx = K_tensor->ne[1] - 1;
        if (current_idx < 0) current_idx = 0;

        // Nhảy đến đúng địa chỉ của Token hiện tại nhờ Stride (nb[1])
        const uint16_t* k_current = (const uint16_t*)((const char*)K_data + current_idx * K_tensor->nb[1]);
        const uint16_t* v_current = (const uint16_t*)((const char*)V_data + current_idx * V_tensor->nb[1]);

        int8_t kv_quantized[1024]; // Chứa tối đa 512 cho K, 512 cho V
        uint16_t kv_scales[2];    
        
        float k_f32[512]; 
        float v_f32[512]; 
        
        // ---------------------------------------------------
        // Lượng tử hóa K (Symmetric INT8 Quantization)
        // ---------------------------------------------------
        float max_k = 0.0f;
        // (Đã xóa dòng khai báo lặp float k_f32[64] ở đây)
        
        for (int i = 0; i < head_dim; i++) {
            k_f32[i] = safe_fp16_to_fp32(k_current[i]);
            if (fabs(k_f32[i]) > max_k) max_k = fabs(k_f32[i]);
        }
        float scale_k = max_k / 127.0f;
        kv_scales[0] = safe_fp32_to_fp16(scale_k);
        float inv_scale_k = scale_k > 0 ? 1.0f / scale_k : 0.0f;
        
        for (int i = 0; i < head_dim; i++) {
            kv_quantized[i] = (int8_t)roundf(k_f32[i] * inv_scale_k);
        }

        // ---------------------------------------------------
        // Lượng tử hóa V (Symmetric INT8 Quantization)
        // ---------------------------------------------------
        float max_v = 0.0f;
        // (Đã xóa dòng khai báo lặp float v_f32[64] ở đây)
        
        for (int i = 0; i < head_dim; i++) {
            v_f32[i] = safe_fp16_to_fp32(v_current[i]);
            if (fabs(v_f32[i]) > max_v) max_v = fabs(v_f32[i]);
        }
        float scale_v = max_v / 127.0f;
        kv_scales[1] = safe_fp32_to_fp16(scale_v);
        float inv_scale_v = scale_v > 0 ? 1.0f / scale_v : 0.0f;
        
        for (int i = 0; i < head_dim; i++) {
            kv_quantized[head_dim + i] = (int8_t)roundf(v_f32[i] * inv_scale_v);
        }

        // Trỏ mảng đầu ra C
        float* output = (float*)dst->data;

        fprintf(stderr, "[FPGA HOST] Calling fpga_run_attention (Layer %d, Pos %d, HeadDim %d)...\n", 
                layer_id, seq_pos, head_dim);
        
        // Đẩy thẳng xuống FPGA (SỬA: Chỉ truyền head_dim)
        fpga_run_attention(Q, kv_scales, kv_quantized, output, head_dim, layer_id, seq_pos, 1);
        
        fprintf(stderr, "[FPGA HOST] fpga_run_attention DONE!\n");
    }
    else {
        // --- NHÁNH MATMUL (MLP) ---
        // Tương tự logic fpga_try_matmul cũ
        uint64_t cur_B_key = ((uintptr_t)src0->data) ^ (uint64_t)K ^ (uint64_t)N;
        int b_changed = (cur_B_key != g_cached_B_key || K != g_cached_B_K || N != g_cached_B_N) ? 1 : 0;

        if (b_changed) {
            const int num_blocks = (int)((K * N) / QK8_0);
            const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;

            #pragma omp parallel for schedule(static) num_threads(4)
            for (int i = 0; i < num_blocks; i++) {
                g_B_d_buf[i] = blocks[i].d;
                const int8_t* src_qs = blocks[i].qs;
                int8_t* dst_qs = &g_B_qs_buf[i * QK8_0];
                for (int j = 0; j < QK8_0; j++)
                    dst_qs[j] = src_qs[j];
            }
            g_cached_B_key = cur_B_key;
            g_cached_B_K   = K;
            g_cached_B_N   = N;
        }

        const float* A_src = (const float*)src1->data;
        float* A_pad_buf = NULL;
        const float* A_use = A_src;

        if (M_pad != M) {
            A_pad_buf = (float*)calloc(M_pad * K, sizeof(float));
            memcpy(A_pad_buf, A_src, (size_t)M * K * sizeof(float));
            A_use = (const float*)A_pad_buf;
        }

        float* C = (float*)dst->data;
        
        // Ghi metadata trước khi chạy mạch Matmul thường
        wr32(REG_LAYER_ID, (uint32_t)layer_id);
        wr32(REG_SEQ_POS,  (uint32_t)seq_pos);
        wr32(REG_IS_ATTN,  (uint32_t)0);

        fprintf(stderr, "[FPGA HOST] Calling fpga_run_matmul_internal...\n");
        
        fpga_run_matmul_internal(A_use, g_B_d_buf, g_B_qs_buf, C, (int)M_pad, (int)M, (int)K, (int)N, b_changed);
        if (A_pad_buf) free(A_pad_buf);
        
        fprintf(stderr, "[FPGA HOST] fpga_run_matmul_internal DONE!\n");
    }

    g_fpga_count++;
    pthread_mutex_unlock(&g_mutex);
    fprintf(stderr, "[FPGA HOST] <<< Mutex Unlocked. Returning 1.\n\n");
    
    return 1;
}
extern "C" void fpga_reset_kv_cache(void) {
    g_current_seq_pos = 0;
}