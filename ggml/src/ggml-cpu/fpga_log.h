#pragma once
#include <stdio.h>
#include <time.h>

// Bật/tắt từng tầng log — 0=tắt, 1=bật
#define FPGA_LOG_LEVEL_INFO   1   // init, cleanup, lỗi nghiêm trọng
#define FPGA_LOG_LEVEL_MATMUL 1   // mỗi lần matmul: M,K,N, FPGA hay CPU
#define FPGA_LOG_LEVEL_REG    0   // giá trị register AXI (rất verbose)
#define FPGA_LOG_LEVEL_TIMING 1   // đo thời gian từng matmul

// File log — NULL = chỉ in ra stderr
#define FPGA_LOG_FILE "/tmp/fpga_debug.log"

// ---- Internal ----
static inline FILE* fpga_log_get_fp(void) {
    static FILE* fp = NULL;
    if (!fp) {
#ifdef FPGA_LOG_FILE
        fp = fopen(FPGA_LOG_FILE, "a");
        if (!fp) fp = stderr;
#else
        fp = stderr;
#endif
    }
    return fp;
}

static inline double fpga_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define FPGA_LOG(level, fmt, ...) \
    do { if (FPGA_LOG_LEVEL_##level) { \
        FILE* _fp = fpga_log_get_fp(); \
        fprintf(_fp, "[FPGA][%-6s] " fmt "\n", #level, ##__VA_ARGS__); \
        fflush(_fp); \
    }} while(0)

// Shortcuts
#define FPGA_LOGI(fmt, ...)   FPGA_LOG(INFO,   fmt, ##__VA_ARGS__)
#define FPGA_LOGM(fmt, ...)   FPGA_LOG(MATMUL, fmt, ##__VA_ARGS__)
#define FPGA_LOGR(fmt, ...)   FPGA_LOG(REG,    fmt, ##__VA_ARGS__)
#define FPGA_LOGT(fmt, ...)   FPGA_LOG(TIMING, fmt, ##__VA_ARGS__)