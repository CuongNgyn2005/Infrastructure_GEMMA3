#include "fpga_host.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>

#define FPGA_LOG_FILE "/tmp/fpga_debug.log"
#define FPGA_HOST_TRACE_VERSION "zcu104-gemma3-q8-zdma-ddr-to-ip-v1"

#define MY_IP_BASE_ADDRESS 0x00000000A0000000LL
#define REG_BASE_PHYS      0x00000000A0000000LL
#define LMM_BASE_PHYS      0x00000000A0000000LL

#define DMA_BASE_PHYS      0x00000000fd500000LL
#define DMA_MMAP_SIZE      0x0000000000010000LL

#define DDR_BASE_PHYS      0x0000000800000000LL
#define DDR_MMAP_SIZE      0x0000000080000000LL

static int g_log_flush_every = 1;
static int g_log_pending_lines = 0;

static FILE * fpga_log_fp(void) {
    static FILE * fp = nullptr;
    if (!fp) {
        fp = fopen(FPGA_LOG_FILE, "a");
        if (!fp) {
            fp = stderr;
        }

        const time_t now = time(nullptr);
        fprintf(fp, "\n============================================================\n");
        fprintf(fp, "[FPGA] ZDMA DDR-to-IP log started at %ld\n", (long) now);
        fprintf(fp, "============================================================\n");
        fflush(fp);
    }
    return fp;
}

static void fpga_log_finish_line(FILE * fp, bool force_flush) {
    g_log_pending_lines++;
    if (force_flush || g_log_flush_every <= 1 || g_log_pending_lines >= g_log_flush_every) {
        fflush(fp);
        g_log_pending_lines = 0;
    }
}

static void fpga_log_line(bool enabled, const char * tag, bool force_flush, const char * fmt, ...) {
    if (!enabled) {
        return;
    }

    FILE * fp = fpga_log_fp();
    fprintf(fp, "[FPGA][%s] ", tag);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fpga_log_finish_line(fp, force_flush);
}

#define LOGI(fmt, ...)     fpga_log_line(true, "INFO",    false, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...)     fpga_log_line(true, "ERROR",   true,  fmt, ##__VA_ARGS__)
#define LOGDMA(fmt, ...)   fpga_log_line(g_dma_timing_enabled, "DMA", true, fmt, ##__VA_ARGS__)
#define LOGIP(fmt, ...)    fpga_log_line(g_ip_timing_enabled, "IPTIME", true, fmt, ##__VA_ARGS__)
#define LOGSTAGE(fmt, ...) fpga_log_line(true, "STAGE",   true,  fmt, ##__VA_ARGS__)
#define LOGTOKEN(fmt, ...) fpga_log_line(true, "TOKEN",   true,  fmt, ##__VA_ARGS__)
#define LOGDATA(fmt, ...)  fpga_log_line(g_trace_data_enabled, "DATA", true, fmt, ##__VA_ARGS__)

static constexpr uint64_t VPU_BASE_PHYS      = MY_IP_BASE_ADDRESS;
static constexpr size_t   VPU_REG_MMAP_MIN   = 0x00001000;
static constexpr size_t   VPU_DEV_MEM_MMAP   = 0x10000000;
static constexpr size_t   DDR_DEV_MEM_MMAP   = 0x80000000;

static constexpr uint32_t REG_CTRL           = 0x00000000;
static constexpr uint32_t REG_STATUS         = 0x00000010;
static constexpr uint32_t REG_ROWS           = 0x00000020;
static constexpr uint32_t REG_COLS           = 0x00000030;
static constexpr uint32_t REG_COL_BEATS      = 0x00000040;
static constexpr uint32_t REG_SCALE          = 0x00000050;
static constexpr uint32_t REG_MODE           = 0x00000060;
static constexpr uint32_t REG_LIMITS         = 0x00000070;
static constexpr uint32_t REG_PROGRESS       = 0x00000080;
static constexpr uint32_t REG_CAPS           = 0x00000090;

static constexpr uint32_t CTRL_START         = 0x00000001;
static constexpr uint32_t CTRL_CLEAR_DONE    = 0x00000002;
static constexpr uint32_t STATUS_DONE        = 0x00000001;
static constexpr uint32_t STATUS_BUSY        = 0x00000002;
static constexpr uint32_t STATUS_ERROR       = 0x00000004;

static constexpr uint32_t ACT_BASE           = 0x00010000;
static constexpr uint32_t ACT_END            = 0x00020000;
static constexpr uint32_t WEIGHT_BASE        = 0x00100000;
static constexpr uint32_t WEIGHT_END         = 0x00200000;
static constexpr uint32_t RESULT_BASE        = 0x00200000;
static constexpr uint32_t RESULT_END         = 0x00210000;
static constexpr size_t   DDR_REQUIRED_BYTES = RESULT_END;

static constexpr int      VPU_NUM_LANES      = 16;
static constexpr int      VPU_QK8_0          = 32;
static constexpr int      VPU_BLOCK_BEATS    = VPU_QK8_0 / VPU_NUM_LANES;
static constexpr int      VPU_RESULT_PACK_LANES = 4;
static constexpr int      VPU_PACKED_Q8_MAX_BLOCKS = 16;
static constexpr int      VPU_DEFAULT_ROWS   = 256;
static constexpr int      VPU_DEFAULT_BEATS  = 256;
static constexpr int      VPU_DEFAULT_COLS   = VPU_DEFAULT_BEATS * VPU_NUM_LANES;
static constexpr uint32_t VPU_MODE_PACKED_Q8 = 0x00000001;
static constexpr uint32_t VPU_FP16_ONE       = 0x00003C00;

static constexpr long long FPGA_DEFAULT_DMA_TIMEOUT_US = 5000000LL;
static constexpr long long FPGA_DEFAULT_IP_TIMEOUT_US  = 5000000LL;
static constexpr int      FPGA_DEFAULT_STATUS_EVERY    = 128;
static constexpr int      FPGA_DEFAULT_PROFILE_EVERY   = 128;
static constexpr long long FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS = 1000000LL;

typedef uint32_t U32;

struct dma_ctrl {
    U32 ZDMA_ERR_CTRL;
    U32 dmy0[63];
    U32 ZDMA_CH_ISR;
    U32 ZDMA_CH_IMR;
    U32 ZDMA_CH_IEN;
    U32 ZDMA_CH_IDS;
    U32 ZDMA_CH_CTRL0;
    U32 ZDMA_CH_CTRL1;
    U32 ZDMA_CH_FCI;
    U32 ZDMA_CH_STATUS;
    U32 ZDMA_CH_DATA_ATTR;
    U32 ZDMA_CH_DSCR_ATTR;
    U32 ZDMA_CH_SRC_DSCR_WORD0;
    U32 ZDMA_CH_SRC_DSCR_WORD1;
    U32 ZDMA_CH_SRC_DSCR_WORD2;
    U32 ZDMA_CH_SRC_DSCR_WORD3;
    U32 ZDMA_CH_DST_DSCR_WORD0;
    U32 ZDMA_CH_DST_DSCR_WORD1;
    U32 ZDMA_CH_DST_DSCR_WORD2;
    U32 ZDMA_CH_DST_DSCR_WORD3;
    U32 ZDMA_CH_WR_ONLY_WORD0;
    U32 ZDMA_CH_WR_ONLY_WORD1;
    U32 ZDMA_CH_WR_ONLY_WORD2;
    U32 ZDMA_CH_WR_ONLY_WORD3;
    U32 ZDMA_CH_SRC_START_LSB;
    U32 ZDMA_CH_SRC_START_MSB;
    U32 ZDMA_CH_DST_START_LSB;
    U32 ZDMA_CH_DST_START_MSB;
    U32 dmy1[9];
    U32 ZDMA_CH_RATE_CTRL;
    U32 ZDMA_CH_IRQ_SRC_ACCT;
    U32 ZDMA_CH_IRQ_DST_ACCT;
    U32 dmy2[26];
    U32 ZDMA_CH_CTRL2;
};

static_assert(offsetof(dma_ctrl, ZDMA_CH_ISR) == 0x100, "unexpected ZDMA_CH_ISR offset");
static_assert(offsetof(dma_ctrl, ZDMA_CH_CTRL2) == 0x200, "unexpected ZDMA_CH_CTRL2 offset");

static constexpr uint32_t ZDMA_STATUS_STATE_MASK = 0x00000003;
static constexpr uint32_t ZDMA_CTRL2_START       = 0x00000001;
static constexpr uint32_t ZDMA_DATA_ATTR_AXCACHE = 0x04C3D30F;

typedef struct {
    uint16_t d;
    int8_t   qs[VPU_QK8_0];
} block_q8_0_t;

static_assert(sizeof(block_q8_0_t) == sizeof(uint16_t) + VPU_QK8_0, "unexpected q8_0 block layout");

typedef struct {
    long long prep_us;
    long long dma_act_us;
    long long dma_weight_us;
    long long dma_result_us;
    long long ip_compute_us;
    size_t    activation_bytes;
    size_t    weight_bytes;
    size_t    result_bytes;
    long long vpu_runs;
} fpga_stage_totals_t;

typedef struct {
    std::vector<block_q8_0_t> act_blocks_all;
    std::vector<float>        act_scales;
    std::vector<float>        weight_scales;
    std::vector<int32_t>      partial;
    std::vector<float>        accum;
    const struct ggml_tensor * cached_src1;
    const void *               cached_src1_data;
    int64_t                    cached_m;
    int64_t                    cached_k;
    size_t                     cached_nb0;
    size_t                     cached_nb1;
    bool                       activation_cache_valid;
} fpga_scratch_t;

static int                 g_mem_fd        = -1;
static volatile dma_ctrl * g_dma           = nullptr;
static volatile uint8_t *  g_vpu           = nullptr;
static uint8_t *           g_ddr           = nullptr;
static void *              g_dma_map_base  = nullptr;
static void *              g_vpu_map_base  = nullptr;
static void *              g_ddr_map_base  = nullptr;
static size_t              g_dma_map_size  = 0;
static size_t              g_vpu_map_size  = 0;
static size_t              g_ddr_map_size  = 0;
static std::string         g_dma_map_source;
static std::string         g_vpu_map_source;
static std::string         g_ddr_map_source;
static pthread_mutex_t     g_mutex         = PTHREAD_MUTEX_INITIALIZER;
static fpga_scratch_t      g_scratch;

static long long           g_fpga_start_us = 0;
static long long           g_fpga_count    = 0;
static long long           g_fpga_vpu_runs = 0;
static long long           g_reject_count  = 0;
static long long           g_activation_cache_hits = 0;
static long long           g_activation_cache_misses = 0;
static long long           g_last_token_us = 0;
static int                 g_last_token_seq = INT_MIN;
static long long           g_token_matmuls = 0;

static int                 g_vpu_max_rows  = VPU_DEFAULT_ROWS;
static int                 g_vpu_max_beats = VPU_DEFAULT_BEATS;
static int                 g_vpu_max_cols  = VPU_DEFAULT_COLS;
static int                 g_packed_q8_supported = 0;
static int                 g_packed_q8_max_blocks = 1;
static int                 g_packed_q8_result_words = VPU_DEFAULT_ROWS;

static bool                g_dma_timing_enabled = true;
static bool                g_ip_timing_enabled = true;
static bool                g_status_stderr = false;
static bool                g_trace_data_enabled = false;
static bool                g_cleanup_done = false;
static bool                g_atexit_registered = false;
static bool                g_abort_on_cpu_fallback = true;
static int                 g_profile_every = FPGA_DEFAULT_PROFILE_EVERY;
static int                 g_ip_status_every = FPGA_DEFAULT_STATUS_EVERY;
static long long           g_dma_timeout_us = FPGA_DEFAULT_DMA_TIMEOUT_US;
static long long           g_ip_timeout_us = FPGA_DEFAULT_IP_TIMEOUT_US;
static long long           g_large_matrix_min_macs = FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS;
static double              g_fpga_clock_mhz = 0.0;

static int g_current_layer_id = 0;
int        g_current_seq_pos  = 0;
static int g_is_attention_op  = 0;

static long long now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (long long) tv.tv_sec * 1000000LL + (long long) tv.tv_usec;
}

static bool env_flag_enabled(const char * name) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "1") == 0 ||
           strcmp(value, "true") == 0 ||
           strcmp(value, "TRUE") == 0 ||
           strcmp(value, "yes") == 0 ||
           strcmp(value, "YES") == 0 ||
           strcmp(value, "on") == 0 ||
           strcmp(value, "ON") == 0;
}

static bool env_flag_disabled(const char * name) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "0") == 0 ||
           strcmp(value, "false") == 0 ||
           strcmp(value, "FALSE") == 0 ||
           strcmp(value, "no") == 0 ||
           strcmp(value, "NO") == 0 ||
           strcmp(value, "off") == 0 ||
           strcmp(value, "OFF") == 0;
}

static int env_int_value(const char * name, int fallback, int min_value, int max_value) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(value, &end, 0);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return (int) std::max<long>(min_value, std::min<long>(parsed, max_value));
}

static long long env_int64_value(const char * name, long long fallback, long long min_value, long long max_value) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(value, &end, 0);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return std::max(min_value, std::min(parsed, max_value));
}

static double env_double_value(const char * name, double fallback, double min_value, double max_value) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const double parsed = strtod(value, &end);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return std::max(min_value, std::min(parsed, max_value));
}

static void fpga_fatal(const char * fmt, ...) {
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[FPGA][ERROR] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fflush(fp);

    fprintf(stderr, "[FPGA][ERROR] ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    abort();
}

static inline void mmio_fence(void) {
    __sync_synchronize();
}

static inline bool dma_is_mapped(void) {
    return g_dma != nullptr && g_dma_map_base != nullptr && g_dma_map_base != MAP_FAILED;
}

static inline bool vpu_is_mapped(void) {
    return g_vpu != nullptr && g_vpu_map_base != nullptr && g_vpu_map_base != MAP_FAILED;
}

static inline bool ddr_is_mapped(void) {
    return g_ddr != nullptr && g_ddr_map_base != nullptr && g_ddr_map_base != MAP_FAILED;
}

static inline uint32_t vpu_rd32(uint32_t off) {
    return *(volatile uint32_t *) (g_vpu + off);
}

static inline void vpu_wr32(uint32_t off, uint32_t val) {
    *(volatile uint32_t *) (g_vpu + off) = val;
}

static size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool range_fits(uint32_t off, size_t bytes, uint32_t begin, uint32_t end) {
    return bytes > 0 && off >= begin && off < end && bytes <= (size_t) (end - off);
}

static bool ddr_range_fits(uint32_t off, size_t bytes) {
    return ddr_is_mapped() && bytes > 0 && (uint64_t) off + (uint64_t) bytes <= (uint64_t) g_ddr_map_size;
}

static uint8_t * ddr_ptr(uint32_t off, size_t bytes) {
    if (!ddr_range_fits(off, bytes)) {
        fpga_fatal("DDR mapped range overflow off=0x%08x bytes=%zu mapped_size=0x%zx", off, bytes, g_ddr_map_size);
    }
    return g_ddr + off;
}

static const char * tensor_name_or_unknown(const struct ggml_tensor * tensor) {
    return (tensor && tensor->name[0] != '\0') ? tensor->name : "?";
}

static int infer_layer_id_from_name(const char * name, int fallback) {
    if (!name || name[0] == '\0') {
        return fallback;
    }

    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) == 1 ||
        sscanf(name, "layers.%d.", &layer) == 1 ||
        sscanf(name, "model.layers.%d.", &layer) == 1) {
        return layer;
    }
    return fallback;
}

static long long matrix_mac_count(int64_t k, int64_t n, int64_t m) {
    if (k <= 0 || n <= 0 || m <= 0) {
        return 0;
    }
    if (k > LLONG_MAX / n || k * n > LLONG_MAX / m) {
        return LLONG_MAX;
    }
    return (long long) (k * n * m);
}

static const char * decode_or_prefill(int64_t m) {
    return m == 1 ? "decode" : "prefill";
}

static bool read_text_file(const std::string & path, std::string * out) {
    FILE * fp = fopen(path.c_str(), "r");
    if (!fp) {
        return false;
    }
    char buf[256];
    if (!fgets(buf, sizeof(buf), fp)) {
        fclose(fp);
        return false;
    }
    fclose(fp);
    *out = buf;
    while (!out->empty() && (out->back() == '\n' || out->back() == '\r' || out->back() == ' ' || out->back() == '\t')) {
        out->pop_back();
    }
    return true;
}

static bool find_uio_device(const char * wanted_name, std::string * dev_path, size_t * map_size) {
    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        return false;
    }

    bool found = false;
    struct dirent * ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        const std::string uio = ent->d_name;
        std::string name;
        if (!read_text_file("/sys/class/uio/" + uio + "/name", &name)) {
            continue;
        }
        if (name != wanted_name) {
            continue;
        }

        std::string size_text;
        if (!read_text_file("/sys/class/uio/" + uio + "/maps/map0/size", &size_text)) {
            continue;
        }
        char * end = nullptr;
        errno = 0;
        const unsigned long long parsed = strtoull(size_text.c_str(), &end, 0);
        if (errno != 0 || end == size_text.c_str() || parsed == 0ULL) {
            continue;
        }

        *dev_path = "/dev/" + uio;
        *map_size = (size_t) parsed;
        found = true;
        break;
    }

    closedir(dir);
    return found;
}

static bool map_uio_region(
        const char * uio_name,
        size_t required_size,
        const char * tag,
        void ** map_base,
        size_t * map_size,
        std::string * source) {
    std::string dev_path;
    size_t uio_size = 0;
    if (!find_uio_device(uio_name, &dev_path, &uio_size)) {
        return false;
    }
    if (uio_size < required_size) {
        LOGE("UIO %s for %s is too small: size=0x%zx required=0x%zx; trying /dev/mem fallback",
             uio_name, tag, uio_size, required_size);
        return false;
    }

    int fd = open(dev_path.c_str(), O_RDWR | O_SYNC);
    if (fd < 0) {
        LOGE("open %s for %s failed errno=%d (%s); trying /dev/mem fallback",
             dev_path.c_str(), tag, errno, strerror(errno));
        return false;
    }
    void * ptr = mmap(nullptr, uio_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int saved_errno = errno;
    close(fd);
    if (ptr == MAP_FAILED) {
        LOGE("mmap %s for %s size=0x%zx failed errno=%d (%s); trying /dev/mem fallback",
             dev_path.c_str(), tag, uio_size, saved_errno, strerror(saved_errno));
        return false;
    }

    *map_base = ptr;
    *map_size = uio_size;
    *source = dev_path + "(" + uio_name + ",O_SYNC)";
    LOGDMA("mapped %s via UIO name=%s dev=%s virt=%p size=0x%zx",
           tag, uio_name, dev_path.c_str(), ptr, uio_size);
    return true;
}

static bool ensure_mem_fd(void) {
    if (g_mem_fd >= 0) {
        return true;
    }
    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) {
        LOGE("open /dev/mem failed errno=%d (%s). Run with sudo.", errno, strerror(errno));
        return false;
    }
    return true;
}

static bool map_devmem_region(
        uint64_t phys,
        size_t bytes,
        const char * tag,
        void ** map_base,
        size_t * map_size,
        std::string * source) {
    if (!ensure_mem_fd()) {
        return false;
    }
    void * ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g_mem_fd, (off_t) phys);
    if (ptr == MAP_FAILED) {
        LOGE("mmap /dev/mem %s phys=0x%llx size=0x%zx failed errno=%d (%s)",
             tag, (unsigned long long) phys, bytes, errno, strerror(errno));
        return false;
    }
    *map_base = ptr;
    *map_size = bytes;
    *source = "/dev/mem(O_SYNC)";
    LOGDMA("mapped %s via /dev/mem phys=0x%llx virt=%p size=0x%zx",
           tag, (unsigned long long) phys, ptr, bytes);
    return true;
}

static bool map_region_prefer_uio(
        const char * uio_name,
        uint64_t phys,
        size_t devmem_size,
        size_t required_size,
        const char * tag,
        void ** map_base,
        size_t * map_size,
        std::string * source) {
    if (map_uio_region(uio_name, required_size, tag, map_base, map_size, source)) {
        return true;
    }
    return map_devmem_region(phys, devmem_size, tag, map_base, map_size, source);
}

static bool map_registers_dma_ddr(void) {
    if (!map_region_prefer_uio("dma-controller", DMA_BASE_PHYS, DMA_MMAP_SIZE,
                               sizeof(dma_ctrl), "ZDMA", &g_dma_map_base,
                               &g_dma_map_size, &g_dma_map_source)) {
        return false;
    }
    g_dma = (volatile dma_ctrl *) g_dma_map_base;

    if (!map_region_prefer_uio("MY_IP", REG_BASE_PHYS, VPU_DEV_MEM_MMAP,
                               VPU_REG_MMAP_MIN, "MY_IP/VPU", &g_vpu_map_base,
                               &g_vpu_map_size, &g_vpu_map_source)) {
        return false;
    }
    g_vpu = (volatile uint8_t *) g_vpu_map_base;

    if (!map_region_prefer_uio("ddr_high", DDR_BASE_PHYS, DDR_DEV_MEM_MMAP,
                               DDR_REQUIRED_BYTES, "ddr_high", &g_ddr_map_base,
                               &g_ddr_map_size, &g_ddr_map_source)) {
        return false;
    }
    g_ddr = (uint8_t *) g_ddr_map_base;
    return true;
}

static bool msync_ddr_range(uint32_t off, size_t bytes, bool invalidate, const char * tag) {
    if (!ddr_range_fits(off, bytes)) {
        LOGE("DDR msync range overflow tag=%s off=0x%08x bytes=%zu mapped_size=0x%zx",
             tag, off, bytes, g_ddr_map_size);
        return false;
    }

    const long page = sysconf(_SC_PAGESIZE);
    const uintptr_t begin = (uintptr_t) (g_ddr + off);
    const uintptr_t aligned_begin = begin & ~((uintptr_t) page - 1U);
    const uintptr_t end = begin + bytes;
    const size_t len = align_up_size((size_t) (end - aligned_begin), (size_t) page);
    const int flags = MS_SYNC | (invalidate ? MS_INVALIDATE : 0);
    mmio_fence();
    if (msync((void *) aligned_begin, len, flags) != 0) {
        LOGE("msync ddr_high tag=%s off=0x%08x bytes=%zu invalidate=%d errno=%d (%s)",
             tag, off, bytes, invalidate ? 1 : 0, errno, strerror(errno));
        return false;
    }
    mmio_fence();
    return true;
}

static bool phys_to_ddr_offset(uint64_t phys, size_t bytes, uint32_t * off) {
    if (phys < DDR_BASE_PHYS) {
        return false;
    }
    const uint64_t delta = phys - DDR_BASE_PHYS;
    if (delta > UINT32_MAX || delta + bytes > g_ddr_map_size) {
        return false;
    }
    *off = (uint32_t) delta;
    return true;
}

static void zdma_set_addr(volatile U32 * lo, volatile U32 * hi, uint64_t value) {
    *lo = (U32) (value & 0xFFFFFFFFULL);
    *hi = (U32) (value >> 32);
}

static void zdma_dump(const char * tag) {
    LOGE("ZDMA dump tag=%s status=0x%08x isr=0x%08x ctrl0=0x%08x ctrl1=0x%08x ctrl2=0x%08x data_attr=0x%08x",
         tag ? tag : "?",
         g_dma ? g_dma->ZDMA_CH_STATUS : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_ISR : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_CTRL0 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_CTRL1 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_CTRL2 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_DATA_ATTR : 0xFFFFFFFFU);
}

static void zdma_clear_descriptors(void) {
    g_dma->ZDMA_CH_SRC_DSCR_WORD0 = 0;
    g_dma->ZDMA_CH_SRC_DSCR_WORD1 = 0;
    g_dma->ZDMA_CH_SRC_DSCR_WORD2 = 0;
    g_dma->ZDMA_CH_SRC_DSCR_WORD3 = 0;
    g_dma->ZDMA_CH_DST_DSCR_WORD0 = 0;
    g_dma->ZDMA_CH_DST_DSCR_WORD1 = 0;
    g_dma->ZDMA_CH_DST_DSCR_WORD2 = 0;
    g_dma->ZDMA_CH_DST_DSCR_WORD3 = 0;
    g_dma->ZDMA_CH_WR_ONLY_WORD0  = 0;
    g_dma->ZDMA_CH_WR_ONLY_WORD1  = 0;
    g_dma->ZDMA_CH_WR_ONLY_WORD2  = 0;
    g_dma->ZDMA_CH_WR_ONLY_WORD3  = 0;
    g_dma->ZDMA_CH_SRC_START_LSB  = 0;
    g_dma->ZDMA_CH_SRC_START_MSB  = 0;
    g_dma->ZDMA_CH_DST_START_LSB  = 0;
    g_dma->ZDMA_CH_DST_START_MSB  = 0;
}

static bool fpga_dma_init(void) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA register pointer is not mapped");
        return false;
    }

    g_dma->ZDMA_ERR_CTRL          = 0x00000001;
    g_dma->ZDMA_CH_ISR            = 0x00000000;
    g_dma->ZDMA_CH_IMR            = 0x00000FFF;
    g_dma->ZDMA_CH_IEN            = 0x00000000;
    g_dma->ZDMA_CH_IDS            = 0x00000000;
    g_dma->ZDMA_CH_CTRL0          = 0x00000080;
    g_dma->ZDMA_CH_CTRL1          = 0x000003FF;
    g_dma->ZDMA_CH_FCI            = 0x00000000;
    g_dma->ZDMA_CH_STATUS         = 0x00000000;
    g_dma->ZDMA_CH_DATA_ATTR      = ZDMA_DATA_ATTR_AXCACHE;
    g_dma->ZDMA_CH_DSCR_ATTR      = 0x00000000;
    zdma_clear_descriptors();
    g_dma->ZDMA_CH_RATE_CTRL      = 0x00000000;
    g_dma->ZDMA_CH_IRQ_SRC_ACCT   = 0x00000000;
    g_dma->ZDMA_CH_IRQ_DST_ACCT   = 0x00000000;
    g_dma->ZDMA_CH_CTRL2          = 0x00000000;
    mmio_fence();

    LOGDMA("ZDMA init base=0x%llx virt=%p status=0x%08x isr=0x%08x ctrl0=0x%08x ctrl1=0x%08x data_attr=0x%08x",
           (unsigned long long) DMA_BASE_PHYS,
           (void *) g_dma,
           g_dma->ZDMA_CH_STATUS,
           g_dma->ZDMA_CH_ISR,
           g_dma->ZDMA_CH_CTRL0,
           g_dma->ZDMA_CH_CTRL1,
           g_dma->ZDMA_CH_DATA_ATTR);
    return true;
}

static bool fpga_dma_copy(uint64_t src_phys, uint64_t dst_phys, size_t bytes, const char * tag) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA is not mapped for tag=%s", tag ? tag : "?");
        return false;
    }
    if (bytes == 0 || bytes > UINT32_MAX) {
        LOGE("invalid ZDMA byte count tag=%s bytes=%zu", tag ? tag : "?", bytes);
        return false;
    }

    uint32_t src_ddr_off = 0;
    if (phys_to_ddr_offset(src_phys, bytes, &src_ddr_off)) {
        if (!msync_ddr_range(src_ddr_off, bytes, false, tag)) {
            return false;
        }
    }

    zdma_set_addr(&g_dma->ZDMA_CH_SRC_DSCR_WORD0, &g_dma->ZDMA_CH_SRC_DSCR_WORD1, src_phys);
    g_dma->ZDMA_CH_SRC_DSCR_WORD2 = (U32) bytes;
    g_dma->ZDMA_CH_SRC_DSCR_WORD3 = 0;
    zdma_set_addr(&g_dma->ZDMA_CH_DST_DSCR_WORD0, &g_dma->ZDMA_CH_DST_DSCR_WORD1, dst_phys);
    g_dma->ZDMA_CH_DST_DSCR_WORD2 = (U32) bytes;
    g_dma->ZDMA_CH_DST_DSCR_WORD3 = 0;
    mmio_fence();

    const long long t0 = now_us();
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_START;
    mmio_fence();

    uint32_t status = 0;
    uint32_t state = 0;
    long long polls = 0;
    while (true) {
        status = g_dma->ZDMA_CH_STATUS;
        state = status & ZDMA_STATUS_STATE_MASK;
        if ((state == 0U || state == 3U) && polls > 0) {
            break;
        }
        if (now_us() - t0 > g_dma_timeout_us) {
            LOGE("ZDMA timeout tag=%s src=0x%llx dst=0x%llx bytes=%zu status=0x%08x state=%u",
                 tag ? tag : "?",
                 (unsigned long long) src_phys,
                 (unsigned long long) dst_phys,
                 bytes,
                 status,
                 state);
            zdma_dump(tag);
            return false;
        }
        polls++;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }

    const long long t1 = now_us();
    uint32_t dst_ddr_off = 0;
    if (phys_to_ddr_offset(dst_phys, bytes, &dst_ddr_off)) {
        if (!msync_ddr_range(dst_ddr_off, bytes, true, tag)) {
            return false;
        }
    }

    LOGDMA("tag=%s src=0x%llx dst=0x%llx bytes=%zu units=bytes ms=%.3f MiB/s=%.1f status=0x%08x isr=0x%08x polls=%lld",
           tag ? tag : "?",
           (unsigned long long) src_phys,
           (unsigned long long) dst_phys,
           bytes,
           (double) (t1 - t0) / 1000.0,
           (t1 > t0) ? (double) bytes * 1000000.0 / ((double) (t1 - t0) * 1024.0 * 1024.0) : 0.0,
           status,
           g_dma->ZDMA_CH_ISR,
           polls);
    return true;
}

static bool fpga_dma_write_to_ip(uint32_t offset, size_t bytes, const char * tag) {
    return fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) offset,
                         LMM_BASE_PHYS + (uint64_t) offset,
                         bytes,
                         tag);
}

static bool fpga_dma_read_from_ip(uint32_t offset, size_t bytes, const char * tag) {
    return fpga_dma_copy(LMM_BASE_PHYS + (uint64_t) offset,
                         DDR_BASE_PHYS + (uint64_t) offset,
                         bytes,
                         tag);
}

static void configure_vpu(int rows, int col_beats, uint32_t mode) {
    const int cols = col_beats * VPU_NUM_LANES;
    vpu_wr32(REG_ROWS, (uint32_t) rows);
    vpu_wr32(REG_COLS, (uint32_t) cols);
    vpu_wr32(REG_COL_BEATS, (uint32_t) col_beats);
    vpu_wr32(REG_SCALE, VPU_FP16_ONE);
    vpu_wr32(REG_MODE, mode);
    mmio_fence();
}

static bool wait_vpu_done(uint32_t * final_status) {
    const long long t0 = now_us();
    long long polls = 0;
    while (true) {
        const uint32_t status = vpu_rd32(REG_STATUS);
        if (final_status) {
            *final_status = status;
        }
        if ((g_ip_status_every > 0) && ((polls % g_ip_status_every) == 0)) {
            LOGIP("poll=%lld status=0x%08x progress=0x%08x", polls, status, vpu_rd32(REG_PROGRESS));
        }
        if (status & STATUS_ERROR) {
            LOGE("VPU reported error status=0x%08x progress=0x%08x",
                 status, vpu_rd32(REG_PROGRESS));
            return false;
        }
        if (status & STATUS_DONE) {
            return true;
        }
        if (now_us() - t0 > g_ip_timeout_us) {
            LOGE("VPU timeout status=0x%08x progress=0x%08x",
                 status, vpu_rd32(REG_PROGRESS));
            return false;
        }
        polls++;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }
}

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t) ((h >> 15) & 1U);
    const uint32_t e = (uint32_t) ((h >> 10) & 0x1FU);
    const uint32_t m = (uint32_t) (h & 0x03FFU);
    uint32_t b;

    if (e == 0U) {
        if (m == 0U) {
            b = s << 31;
        } else {
            uint32_t mant = m;
            uint32_t exp = 113U;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFU;
            b = (s << 31) | (exp << 23) | (mant << 13);
        }
    } else if (e == 31U) {
        b = (s << 31) | 0x7F800000U | (m << 13);
    } else {
        b = (s << 31) | ((e + 112U) << 23) | (m << 13);
    }

    union {
        uint32_t i;
        float    f;
    } u;
    u.i = b;
    return u.f;
}

static inline uint16_t fp32_to_fp16(float f) {
    union {
        float    f;
        uint32_t i;
    } u;
    u.f = f;

    const uint32_t sign = (u.i >> 16) & 0x8000U;
    int32_t exp = (int32_t) ((u.i >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = u.i & 0x007FFFFFU;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        mant = (mant | 0x00800000U) >> (1 - exp);
        return (uint16_t) (sign | ((mant + 0x00001000U) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00U);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | ((mant + 0x00001000U) >> 13));
}

static void quantize_q8_0_block(const float * x, ptrdiff_t stride_bytes, block_q8_0_t * y) {
    float amax = 0.0f;
    for (int i = 0; i < VPU_QK8_0; ++i) {
        const float v = *(const float *) ((const char *) x + (ptrdiff_t) i * stride_bytes);
        amax = std::max(amax, std::fabs(v));
    }

    const float d_raw = amax / 127.0f;
    const float id = d_raw != 0.0f ? 1.0f / d_raw : 0.0f;
    y->d = fp32_to_fp16(d_raw);

    for (int i = 0; i < VPU_QK8_0; ++i) {
        const float v = *(const float *) ((const char *) x + (ptrdiff_t) i * stride_bytes);
        const int q = (int) std::round(v * id);
        y->qs[i] = (int8_t) std::max(-128, std::min(127, q));
    }
}

static void quantize_activation_vector_to(
        const struct ggml_tensor * src1,
        int64_t m,
        int64_t k,
        block_q8_0_t * out) {
    const int64_t nb = k / VPU_QK8_0;
    const char * base = (const char *) src1->data + m * src1->nb[1];
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * block_base = (const float *) (base + ib * VPU_QK8_0 * src1->nb[0]);
        quantize_q8_0_block(block_base, src1->nb[0], &out[(size_t) ib]);
    }
}

static void ensure_quantized_activation_matrix(
        const struct ggml_tensor * src1,
        int64_t m,
        int64_t k,
        std::vector<block_q8_0_t> & act_blocks_all,
        std::vector<float> & act_scales) {
    const bool cache_hit =
        g_scratch.activation_cache_valid &&
        g_scratch.cached_src1 == src1 &&
        g_scratch.cached_src1_data == src1->data &&
        g_scratch.cached_m == m &&
        g_scratch.cached_k == k &&
        g_scratch.cached_nb0 == src1->nb[0] &&
        g_scratch.cached_nb1 == src1->nb[1];

    if (cache_hit) {
        g_activation_cache_hits++;
        return;
    }

    const int64_t nb = k / VPU_QK8_0;
    act_blocks_all.resize((size_t) (m * nb));
    act_scales.resize((size_t) (m * nb));
    for (int64_t col = 0; col < m; ++col) {
        block_q8_0_t * col_blocks = &act_blocks_all[(size_t) (col * nb)];
        quantize_activation_vector_to(src1, col, k, col_blocks);
        for (int64_t ib = 0; ib < nb; ++ib) {
            act_scales[(size_t) (col * nb + ib)] = fp16_to_fp32(col_blocks[(size_t) ib].d);
        }
    }

    g_scratch.cached_src1 = src1;
    g_scratch.cached_src1_data = src1->data;
    g_scratch.cached_m = m;
    g_scratch.cached_k = k;
    g_scratch.cached_nb0 = src1->nb[0];
    g_scratch.cached_nb1 = src1->nb[1];
    g_scratch.activation_cache_valid = true;
    g_activation_cache_misses++;
}

static const block_q8_0_t * weight_block(
        const struct ggml_tensor * src0,
        int64_t row,
        int64_t block) {
    const char * row_base = (const char *) src0->data + row * src0->nb[1];
    return (const block_q8_0_t *) row_base + block;
}

static void store_dst_value(
        const struct ggml_tensor * dst,
        int64_t row,
        int64_t col,
        float value) {
    char * base = (char *) dst->data;
    *(float *) (base + row * dst->nb[0] + col * dst->nb[1]) = value;
}

static void write_i8x16_to_ddr(uint32_t off, const int8_t * lanes) {
    memcpy(ddr_ptr(off, 16U), lanes, 16U);
}

static void read_result_i32x4_from_ddr(uint32_t result_word, int32_t out[4]) {
    memcpy(out, ddr_ptr(RESULT_BASE + result_word * 16U, 16U), 16U);
}

static bool run_vpu_window_transfer(
        int rows,
        int col_beats,
        uint32_t mode,
        size_t act_bytes,
        size_t weight_bytes,
        size_t result_bytes,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m,
        uint16_t tile_id,
        fpga_stage_totals_t * totals) {
    vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
    configure_vpu(rows, col_beats, mode);

    const long long dma_act0 = now_us();
    if (!fpga_dma_write_to_ip(ACT_BASE, act_bytes, "ACT")) {
        return false;
    }
    const long long dma_act1 = now_us();

    const long long dma_weight0 = now_us();
    if (!fpga_dma_write_to_ip(WEIGHT_BASE, weight_bytes, "WEIGHT")) {
        return false;
    }
    const long long dma_weight1 = now_us();

    mmio_fence();
    const long long ip0 = now_us();
    vpu_wr32(REG_CTRL, CTRL_START);
    mmio_fence();

    uint32_t vpu_status = 0;
    if (!wait_vpu_done(&vpu_status)) {
        LOGE("VPU failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld tile=%u status=0x%08x progress=0x%08x",
             tensor_name ? tensor_name : "?",
             layer_id,
             (long long) k,
             (long long) n,
             (long long) m,
             tile_id,
             vpu_status,
             vpu_rd32(REG_PROGRESS));
        return false;
    }
    const long long ip1 = now_us();

    const long long dma_result0 = now_us();
    if (!fpga_dma_read_from_ip(RESULT_BASE, result_bytes, "RESULT")) {
        return false;
    }
    const long long dma_result1 = now_us();

    if (totals) {
        totals->dma_act_us += dma_act1 - dma_act0;
        totals->dma_weight_us += dma_weight1 - dma_weight0;
        totals->dma_result_us += dma_result1 - dma_result0;
        totals->ip_compute_us += ip1 - ip0;
        totals->activation_bytes += act_bytes;
        totals->weight_bytes += weight_bytes;
        totals->result_bytes += result_bytes;
        totals->vpu_runs++;
    }

    LOGIP("tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u rows=%d col_beats=%d mode=0x%x ip_ms=%.3f status=0x%08x progress=0x%08x",
          tensor_name ? tensor_name : "?",
          layer_id,
          (long long) k,
          (long long) n,
          (long long) m,
          tile_id,
          rows,
          col_beats,
          mode,
          (double) (ip1 - ip0) / 1000.0,
          vpu_status,
          vpu_rd32(REG_PROGRESS));
    return true;
}

static bool fpga_dma_basic_self_test(void) {
    int8_t ones[VPU_QK8_0];
    for (int i = 0; i < VPU_QK8_0; ++i) {
        ones[i] = 1;
    }
    for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) beat * 16U, ones + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + (uint32_t) beat * 16U, ones + beat * VPU_NUM_LANES);
    }

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(1, VPU_BLOCK_BEATS, 0,
                                 VPU_QK8_0,
                                 VPU_QK8_0,
                                 16U,
                                 "selftest.basic", -1, 32, 1, 1, 0, &totals)) {
        return false;
    }

    int32_t lanes[4] = {};
    read_result_i32x4_from_ddr(0, lanes);
    LOGI("basic ZDMA-to-IP self-test result=%d expected=32", lanes[0]);
    return lanes[0] == 32;
}

static bool fpga_dma_packed_self_test(void) {
    int8_t act0[VPU_QK8_0];
    int8_t act1[VPU_QK8_0];
    int8_t w_row0_block0[VPU_QK8_0];
    int8_t w_row0_block1[VPU_QK8_0];
    int8_t w_row1_block0[VPU_QK8_0];
    int8_t w_row1_block1[VPU_QK8_0];
    for (int i = 0; i < VPU_QK8_0; ++i) {
        act0[i] = 1;
        act1[i] = 2;
        w_row0_block0[i] = 1;
        w_row0_block1[i] = 1;
        w_row1_block0[i] = -1;
        w_row1_block1[i] = 3;
    }

    for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) beat * 16U, act0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, act1 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + (uint32_t) beat * 16U, w_row0_block0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, w_row0_block1 + beat * VPU_NUM_LANES);
        const uint32_t row1_base = (uint32_t) (2 * VPU_BLOCK_BEATS * 16);
        write_i8x16_to_ddr(WEIGHT_BASE + row1_base + (uint32_t) beat * 16U, w_row1_block0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + row1_base + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, w_row1_block1 + beat * VPU_NUM_LANES);
    }

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(2, 4, VPU_MODE_PACKED_Q8,
                                 4U * 16U,
                                 2U * 4U * 16U,
                                 16U,
                                 "selftest.packed", -1, 64, 2, 1, 1, &totals)) {
        return false;
    }

    int32_t lanes[4] = {};
    read_result_i32x4_from_ddr(0, lanes);
    LOGI("packed ZDMA-to-IP self-test results=[%d,%d,%d,%d] expected=[32,64,-32,192]",
         lanes[0], lanes[1], lanes[2], lanes[3]);
    return lanes[0] == 32 && lanes[1] == 64 && lanes[2] == -32 && lanes[3] == 192;
}

static bool fpga_validate_tensors(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst,
        const char ** reason) {
    if (!src0 || !src1 || !dst) {
        *reason = "unsupported DMA-to-IP tiling case: null tensor";
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        *reason = "unsupported DMA-to-IP tiling case: requires Q8_0 x F32 -> F32";
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (k <= 0 || n <= 0 || m <= 0) {
        *reason = "unsupported DMA-to-IP tiling case: empty tensor";
        return false;
    }
    if (k != src1->ne[0] || n != dst->ne[0] || m != dst->ne[1]) {
        *reason = "unsupported DMA-to-IP tiling case: shape mismatch";
        return false;
    }
    if (k % VPU_QK8_0 != 0) {
        *reason = "unsupported DMA-to-IP tiling case: K is not divisible by 32";
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst->ne[2]  != 1 || dst->ne[3]  != 1) {
        *reason = "unsupported DMA-to-IP tiling case: batched tensor dimensions";
        return false;
    }
    if (src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        *reason = "unsupported DMA-to-IP tiling case: non-F32 row stride";
        return false;
    }
    if (!g_packed_q8_supported) {
        *reason = "unsupported DMA-to-IP tiling case: packed Q8 capability unavailable";
        return false;
    }
    return true;
}

static int packed_q8_group_blocks_for_rows(int rows, int remaining_blocks) {
    const int beat_limited_blocks = std::max(1, g_vpu_max_beats / VPU_BLOCK_BEATS);
    const int result_limited_blocks =
        std::max(1, (g_packed_q8_result_words * VPU_RESULT_PACK_LANES) / std::max(1, rows));
    int blocks = std::min(g_packed_q8_max_blocks, beat_limited_blocks);
    blocks = std::min(blocks, result_limited_blocks);
    blocks = std::min(blocks, remaining_blocks);
    return std::max(1, blocks);
}

static bool fpga_dma_run_q8_group(
        const struct ggml_tensor * src0,
        const block_q8_0_t * act_group,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        std::vector<int32_t> & partial,
        std::vector<float> & weight_scales,
        fpga_stage_totals_t * totals,
        uint16_t tile_id,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m) {
    if (rows <= 0 || rows > g_vpu_max_rows || group_blocks <= 0) {
        LOGE("unsupported DMA-to-IP tiling case: rows=%d max_rows=%d group_blocks=%d",
             rows, g_vpu_max_rows, group_blocks);
        return false;
    }

    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    if (group_beats > g_vpu_max_beats) {
        LOGE("unsupported DMA-to-IP tiling case: group_beats=%d max_beats=%d",
             group_beats, g_vpu_max_beats);
        return false;
    }

    const uint32_t result_values = (uint32_t) rows * (uint32_t) group_blocks;
    const uint32_t result_words = (result_values + (uint32_t) VPU_RESULT_PACK_LANES - 1U) /
                                  (uint32_t) VPU_RESULT_PACK_LANES;
    if (result_words > (uint32_t) g_packed_q8_result_words) {
        LOGE("unsupported DMA-to-IP tiling case: result_words=%u cap=%d", result_words, g_packed_q8_result_words);
        return false;
    }

    const size_t act_bytes = (size_t) group_beats * 16U;
    const size_t weight_bytes = (size_t) rows * (size_t) group_beats * 16U;
    const size_t result_bytes = (size_t) result_words * 16U;
    if (!range_fits(ACT_BASE, act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(WEIGHT_BASE, weight_bytes, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(RESULT_BASE, result_bytes, RESULT_BASE, RESULT_END) ||
        !ddr_range_fits(ACT_BASE, act_bytes) ||
        !ddr_range_fits(WEIGHT_BASE, weight_bytes) ||
        !ddr_range_fits(RESULT_BASE, result_bytes)) {
        LOGE("unsupported DMA-to-IP tiling case: window overflow act=%zu weight=%zu result=%zu ddr_size=0x%zx",
             act_bytes, weight_bytes, result_bytes, g_ddr_map_size);
        return false;
    }

    const long long prep0 = now_us();
    weight_scales.resize((size_t) rows * (size_t) group_blocks);
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * wb = weight_block(src0, row0 + row, k_block0 + gb);
            weight_scales[(size_t) row * (size_t) group_blocks + (size_t) gb] = fp16_to_fp32(wb->d);
            for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                const uint32_t word_index = (uint32_t) row * (uint32_t) group_beats +
                                            (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS +
                                            (uint32_t) beat;
                write_i8x16_to_ddr(WEIGHT_BASE + word_index * 16U, wb->qs + beat * VPU_NUM_LANES);
            }
        }
    }

    for (int gb = 0; gb < group_blocks; ++gb) {
        const block_q8_0_t & act = act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act.qs + beat * VPU_NUM_LANES);
        }
    }
    if (totals) {
        totals->prep_us += now_us() - prep0;
    }

    if (!run_vpu_window_transfer(rows, group_beats, VPU_MODE_PACKED_Q8,
                                 act_bytes, weight_bytes, result_bytes,
                                 tensor_name, layer_id, k, n, m, tile_id, totals)) {
        return false;
    }

    partial.resize((size_t) result_values);
    int32_t lanes[VPU_RESULT_PACK_LANES] = {};
    for (uint32_t word = 0; word < result_words; ++word) {
        read_result_i32x4_from_ddr(word, lanes);
        for (int lane = 0; lane < VPU_RESULT_PACK_LANES; ++lane) {
            const uint32_t idx = word * (uint32_t) VPU_RESULT_PACK_LANES + (uint32_t) lane;
            if (idx < result_values) {
                partial[(size_t) idx] = lanes[lane];
            }
        }
    }
    return true;
}

static long long estimate_vpu_runs(int64_t k, int64_t n, int64_t m) {
    const int64_t nb = k / VPU_QK8_0;
    long long runs_per_m = 0;
    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, (int) (nb - ib0));
            runs_per_m++;
            ib0 += group_blocks;
        }
    }
    return runs_per_m * m;
}

static bool fpga_hw_q8_0_matmul_dma_to_ip(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst,
        fpga_stage_totals_t * totals,
        const char * tensor_name,
        int layer_id) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int64_t nb = k / VPU_QK8_0;

    std::vector<block_q8_0_t> & act_blocks_all = g_scratch.act_blocks_all;
    std::vector<float> & act_scales = g_scratch.act_scales;
    std::vector<float> & weight_scales = g_scratch.weight_scales;
    std::vector<int32_t> & partial = g_scratch.partial;
    std::vector<float> & accum = g_scratch.accum;

    const long long quant0 = now_us();
    ensure_quantized_activation_matrix(src1, m, k, act_blocks_all, act_scales);
    if (totals) {
        totals->prep_us += now_us() - quant0;
    }

    uint16_t tile_id = 0;
    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        accum.assign((size_t) (m * rows), 0.0f);

        for (int64_t ib0 = 0; ib0 < nb;) {
            const int remaining_blocks = (int) (nb - ib0);
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats = group_blocks * VPU_BLOCK_BEATS;

            LOGSTAGE("tile tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d group_beats=%d tile_id=%u partial_accum=1 transfer=zdma_ddr_to_ip",
                     tensor_name ? tensor_name : "?",
                     layer_id,
                     (long long) row0,
                     rows,
                     (long long) ib0,
                     group_blocks,
                     group_beats,
                     tile_id);

            for (int64_t col = 0; col < m; ++col) {
                const block_q8_0_t * act_group =
                    &act_blocks_all[(size_t) (col * nb + ib0)];
                if (!fpga_dma_run_q8_group(src0, act_group, row0, rows, ib0, group_blocks,
                                           partial, weight_scales, totals, tile_id++,
                                           tensor_name, layer_id, k, n, m)) {
                    return false;
                }

                float * accum_col = &accum[(size_t) (col * rows)];
                for (int row = 0; row < rows; ++row) {
                    for (int gb = 0; gb < group_blocks; ++gb) {
                        const int64_t ib = ib0 + gb;
                        const int32_t raw = partial[(size_t) row * (size_t) group_blocks + (size_t) gb];
                        accum_col[(size_t) row] +=
                            (float) raw *
                            act_scales[(size_t) (col * nb + ib)] *
                            weight_scales[(size_t) row * (size_t) group_blocks + (size_t) gb];
                    }
                }
            }
            ib0 += group_blocks;
        }

        for (int64_t col = 0; col < m; ++col) {
            const float * accum_col = &accum[(size_t) (col * rows)];
            for (int row = 0; row < rows; ++row) {
                store_dst_value(dst, row0 + row, col, accum_col[(size_t) row]);
            }
        }
    }
    return true;
}

int fpga_init(void) {
    if (dma_is_mapped() && vpu_is_mapped() && ddr_is_mapped()) {
        return 0;
    }

    const char * path = getenv("FPGA_PATH");
    if (path && strcmp(path, "dma") != 0 && strcmp(path, "auto") != 0 && strcmp(path, "zdma") != 0) {
        fpga_fatal("FPGA_PATH=%s is not allowed in ZDMA DDR-to-IP build; set FPGA_PATH=dma/zdma or leave it unset", path);
    }
    if (env_flag_enabled("FPGA_DISABLE")) {
        fpga_fatal("FPGA_DISABLE is set, but this build must not silently fall back to CPU");
    }

    g_dma_timing_enabled = !env_flag_disabled("FPGA_DMA_TIMING");
    if (env_flag_enabled("FPGA_DMA_TIMING")) {
        g_dma_timing_enabled = true;
    }
    g_ip_timing_enabled = !env_flag_disabled("FPGA_IP_TIMING");
    if (env_flag_enabled("FPGA_IP_TIMING")) {
        g_ip_timing_enabled = true;
    }
    g_status_stderr = env_flag_enabled("FPGA_STATUS_STDERR");
    g_trace_data_enabled = env_flag_enabled("FPGA_TRACE_DATA");
    g_log_flush_every = env_int_value("FPGA_LOG_FLUSH_EVERY", 1, 1, 1000000);
    g_profile_every = env_int_value("FPGA_PROFILE_EVERY", FPGA_DEFAULT_PROFILE_EVERY, 0, 1000000);
    g_ip_status_every = env_int_value("FPGA_IP_STATUS_EVERY", FPGA_DEFAULT_STATUS_EVERY, 0, 1000000);
    g_dma_timeout_us = env_int64_value("FPGA_DMA_TIMEOUT_US", FPGA_DEFAULT_DMA_TIMEOUT_US, 1000, LLONG_MAX);
    g_ip_timeout_us = env_int64_value("FPGA_IP_TIMEOUT_US", FPGA_DEFAULT_IP_TIMEOUT_US, 1000, LLONG_MAX);
    g_large_matrix_min_macs = env_int64_value(
        "FPGA_LARGE_MATRIX_MIN_MACS", FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS, 0, LLONG_MAX);
    g_fpga_clock_mhz = env_double_value("FPGA_CLOCK_MHZ", 0.0, 0.0, 10000.0);
    g_abort_on_cpu_fallback = !env_flag_disabled("FPGA_ABORT_ON_CPU_FALLBACK");

    if (!map_registers_dma_ddr()) {
        fpga_fatal("ZDMA DDR-to-IP FPGA init failed; refusing CPU fallback");
    }
    if (!fpga_dma_init()) {
        fpga_fatal("ZDMA init failed; refusing CPU fallback");
    }

    g_fpga_start_us = now_us();
    g_cleanup_done = false;
    g_scratch.activation_cache_valid = false;
    if (!g_atexit_registered) {
        atexit(fpga_cleanup);
        g_atexit_registered = true;
    }

    const uint32_t limits = vpu_rd32(REG_LIMITS);
    const uint32_t caps = vpu_rd32(REG_CAPS);
    const int limit_rows  = (int) (limits & 0xFFFFU);
    const int limit_beats = (int) ((limits >> 16) & 0xFFFFU);
    if (limit_rows > 0 && limit_rows <= VPU_DEFAULT_ROWS) {
        g_vpu_max_rows = limit_rows;
    }
    if (limit_beats > 0 && limit_beats <= VPU_DEFAULT_BEATS) {
        g_vpu_max_beats = limit_beats;
        g_vpu_max_cols = g_vpu_max_beats * VPU_NUM_LANES;
    }

    const bool caps_valid = caps != 0U && caps != 0xFFFFFFFFU;
    if (caps_valid && ((caps & 0x1U) != 0U)) {
        const int cap_blocks = (int) ((caps >> 8) & 0xFFU);
        const int cap_result_words = (int) ((caps >> 16) & 0xFFFFU);
        if (cap_blocks > 0 && cap_result_words > 0) {
            g_packed_q8_supported = 1;
            g_packed_q8_max_blocks = std::min(cap_blocks, g_vpu_max_beats / VPU_BLOCK_BEATS);
            g_packed_q8_result_words = cap_result_words;
        }
    }
    if (env_flag_enabled("FPGA_FORCE_PACKED_Q8") && !g_packed_q8_supported) {
        g_packed_q8_supported = 1;
        g_packed_q8_max_blocks = std::min(VPU_PACKED_Q8_MAX_BLOCKS, g_vpu_max_beats / VPU_BLOCK_BEATS);
        g_packed_q8_result_words =
            (g_vpu_max_rows * g_packed_q8_max_blocks + VPU_RESULT_PACK_LANES - 1) /
            VPU_RESULT_PACK_LANES;
    }

    LOGI("host trace version: %s", FPGA_HOST_TRACE_VERSION);
    LOGI("runtime path request: FPGA_PATH=%s selected=zdma_ddr_to_ip", path ? path : "dma(default)");
    LOGDMA("bases my_ip=0x%llx reg=0x%llx lmm=0x%llx dma=0x%llx ddr=0x%llx",
           (unsigned long long) MY_IP_BASE_ADDRESS,
           (unsigned long long) REG_BASE_PHYS,
           (unsigned long long) LMM_BASE_PHYS,
           (unsigned long long) DMA_BASE_PHYS,
           (unsigned long long) DDR_BASE_PHYS);
    LOGDMA("mappings dma=%s virt=%p size=0x%zx vpu=%s virt=%p size=0x%zx ddr=%s virt=%p size=0x%zx",
           g_dma_map_source.c_str(), (void *) g_dma, g_dma_map_size,
           g_vpu_map_source.c_str(), (void *) g_vpu, g_vpu_map_size,
           g_ddr_map_source.c_str(), (void *) g_ddr, g_ddr_map_size);
    LOGI("VPU windows act=0x%08x weight=0x%08x result=0x%08x data_movement=ZDMA_bulk_copy no_axi_stream_main=1",
         ACT_BASE, WEIGHT_BASE, RESULT_BASE);
    LOGI("VPU limits rows=%d col_beats=%d cols=%d raw_limits=0x%08x caps=0x%08x packed_q8=%d max_group_blocks=%d result_words=%d",
         g_vpu_max_rows, g_vpu_max_beats, g_vpu_max_cols, limits, caps,
         g_packed_q8_supported, g_packed_q8_max_blocks, g_packed_q8_result_words);
    if (g_vpu_max_beats == 256) {
        LOGE("MAX_COL_BEATS=256 detected; DMA-to-IP path will run, but this large BRAM setting is still suspicious for timing/resource use");
    }
    LOGI("cache coherency: ddr_high mapped through %s; using msync barriers before DMA reads from DRAM and after DMA writes to DRAM",
         g_ddr_map_source.c_str());
    LOGI("fallback policy: FPGA_ABORT_ON_CPU_FALLBACK=%d default_no_cpu_matmul_fallback=1",
         g_abort_on_cpu_fallback ? 1 : 0);

    if (!g_packed_q8_supported) {
        fpga_fatal("REG_CAPS does not expose packed_q8 capability; refusing CPU fallback");
    }
    if (!fpga_dma_basic_self_test()) {
        fpga_fatal("basic ZDMA-to-IP self-test failed; refusing CPU fallback");
    }
    if (!fpga_dma_packed_self_test()) {
        fpga_fatal("packed Q8 ZDMA-to-IP self-test failed; refusing CPU fallback");
    }

    return 0;
}

void fpga_cleanup(void) {
    pthread_mutex_lock(&g_mutex);
    if (g_cleanup_done) {
        pthread_mutex_unlock(&g_mutex);
        return;
    }
    g_cleanup_done = true;

    if (g_ddr_map_base && g_ddr_map_base != MAP_FAILED) {
        munmap(g_ddr_map_base, g_ddr_map_size);
    }
    g_ddr_map_base = nullptr;
    g_ddr = nullptr;

    if (g_vpu_map_base && g_vpu_map_base != MAP_FAILED) {
        munmap(g_vpu_map_base, g_vpu_map_size);
    }
    g_vpu_map_base = nullptr;
    g_vpu = nullptr;

    if (g_dma_map_base && g_dma_map_base != MAP_FAILED) {
        munmap(g_dma_map_base, g_dma_map_size);
    }
    g_dma_map_base = nullptr;
    g_dma = nullptr;

    if (g_mem_fd >= 0) {
        close(g_mem_fd);
        g_mem_fd = -1;
    }

    const long long elapsed_us = g_fpga_start_us > 0 ? now_us() - g_fpga_start_us : 0;
    LOGI("cleanup complete fpga_calls=%lld vpu_runs=%lld rejects=%lld elapsed_s=%.3f activation_cache_hits=%lld misses=%lld",
         g_fpga_count,
         g_fpga_vpu_runs,
         g_reject_count,
         elapsed_us > 0 ? (double) elapsed_us / 1000000.0 : 0.0,
         g_activation_cache_hits,
         g_activation_cache_misses);
    fflush(fpga_log_fp());
    pthread_mutex_unlock(&g_mutex);
}

extern "C" int fpga_run_matmul(
        const float *    A,
        const uint16_t * B_d,
        const int8_t *   B_qs,
        float *          C,
        int M,
        int K,
        int N,
        int ith) {
    (void) A;
    (void) B_d;
    (void) B_qs;
    (void) C;
    (void) M;
    (void) K;
    (void) N;
    (void) ith;
    LOGE("legacy low-level fpga_run_matmul is disabled; ZDMA-to-IP path requires ggml tensor hook");
    return 0;
}

void fpga_set_context(int layer_id, int seq_pos, int is_attn) {
    g_current_layer_id = layer_id;
    g_current_seq_pos  = seq_pos;
    g_is_attention_op  = is_attn;
}

extern "C" int fpga_try_matmul(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst,
        int ith) {
    return fpga_try_matmul_extended(src0, src1, dst, ith, 0, g_current_seq_pos, 0);
}

static void log_token_boundary_if_needed(int seq_pos) {
    const long long now = now_us();
    if (g_last_token_seq == INT_MIN) {
        g_last_token_seq = seq_pos;
        g_last_token_us = now;
        g_token_matmuls = 0;
        return;
    }
    if (seq_pos != g_last_token_seq) {
        const double token_ms = (double) (now - g_last_token_us) / 1000.0;
        LOGTOKEN("seq=%d prev_seq=%d matmuls=%lld token_ms=%.3f est_tokens_s=%.3f",
                 seq_pos,
                 g_last_token_seq,
                 g_token_matmuls,
                 token_ms,
                 token_ms > 0.0 ? 1000.0 / token_ms : 0.0);
        g_last_token_seq = seq_pos;
        g_last_token_us = now;
        g_token_matmuls = 0;
    }
}

extern "C" int fpga_try_matmul_extended(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst,
        int ith,
        int layer_id,
        int seq_pos,
        int is_attention) {
    const char * tensor_name = tensor_name_or_unknown(src0);
    const int effective_layer_id = infer_layer_id_from_name(tensor_name, layer_id);

    if (is_attention) {
        return 0;
    }

    const char * reason = nullptr;
    if (!fpga_validate_tensors(src0, src1, dst, &reason)) {
        if (ith == 0) {
            const int64_t k = src0 ? src0->ne[0] : 0;
            const int64_t n = src0 ? src0->ne[1] : 0;
            const int64_t m = src1 ? src1->ne[1] : 0;
            pthread_mutex_lock(&g_mutex);
            g_reject_count++;
            LOGE("matmul rejected tensor=%s layer=%d shape=K%lld_N%lld_M%lld reason=%s action=%s",
                 tensor_name,
                 effective_layer_id,
                 (long long) k,
                 (long long) n,
                 (long long) m,
                 reason ? reason : "unknown",
                 g_abort_on_cpu_fallback ? "abort_no_cpu_fallback" : "return_to_cpu");
            pthread_mutex_unlock(&g_mutex);
            if (g_abort_on_cpu_fallback) {
                fpga_fatal("CPU fallback matmul blocked tensor=%s layer=%d reason=%s",
                           tensor_name, effective_layer_id, reason ? reason : "unknown");
            }
        }
        return 0;
    }

    if (!dma_is_mapped() || !vpu_is_mapped() || !ddr_is_mapped()) {
        if (ith == 0) {
            LOGE("FPGA/ZDMA/VPU/DDR is not initialized for tensor=%s", tensor_name);
            if (g_abort_on_cpu_fallback) {
                fpga_fatal("FPGA/ZDMA/VPU/DDR is not initialized; refusing CPU fallback");
            }
        }
        return 0;
    }

    if (ith != 0) {
        return 1;
    }

    pthread_mutex_lock(&g_mutex);
    log_token_boundary_if_needed(seq_pos);

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int64_t q8_blocks = k / VPU_QK8_0;
    const long long macs = matrix_mac_count(k, n, m);
    const long long row_tiles = (n + g_vpu_max_rows - 1) / g_vpu_max_rows;
    const int first_tile_rows = (int) std::min<int64_t>(n, g_vpu_max_rows);
    const int max_group_blocks = packed_q8_group_blocks_for_rows(first_tile_rows, (int) q8_blocks);
    const long long estimated_runs = estimate_vpu_runs(k, n, m);
    fpga_stage_totals_t totals = {};

    const long long t0 = now_us();
    LOGSTAGE("tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld path=zdma_ddr_to_ip row_tiles=%lld group_tiles_per_rowtile~=%lld q8_blocks=%lld max_group_blocks=%d vpu_runs=%lld",
             tensor_name,
             effective_layer_id,
             seq_pos,
             decode_or_prefill(m),
             (long long) k,
             (long long) n,
             (long long) m,
             row_tiles,
             (q8_blocks + max_group_blocks - 1) / max_group_blocks,
             (long long) q8_blocks,
             max_group_blocks,
             estimated_runs);

    const bool hw_ok = fpga_hw_q8_0_matmul_dma_to_ip(src0, src1, dst, &totals, tensor_name, effective_layer_id);
    const long long t1 = now_us();
    if (!hw_ok) {
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal("ZDMA-to-IP/VPU matmul failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld; refusing CPU fallback",
                   tensor_name, effective_layer_id, (long long) k, (long long) n, (long long) m);
    }

    g_fpga_count++;
    g_token_matmuls++;
    g_fpga_vpu_runs += totals.vpu_runs;
    const double total_ms = (double) (t1 - t0) / 1000.0;
    const double prep_ms = (double) totals.prep_us / 1000.0;
    const double dma_act_ms = (double) totals.dma_act_us / 1000.0;
    const double dma_weight_ms = (double) totals.dma_weight_us / 1000.0;
    const double dma_in_ms = dma_act_ms + dma_weight_ms;
    const double ip_ms = (double) totals.ip_compute_us / 1000.0;
    const double dma_out_ms = (double) totals.dma_result_us / 1000.0;

    const char * dominant = "prep";
    double dominant_ms = prep_ms;
    if (dma_in_ms > dominant_ms) {
        dominant = "dma_input";
        dominant_ms = dma_in_ms;
    }
    if (ip_ms > dominant_ms) {
        dominant = "ip_compute";
        dominant_ms = ip_ms;
    }
    if (dma_out_ms > dominant_ms) {
        dominant = "dma_output";
        dominant_ms = dma_out_ms;
    }

    const size_t effective_bytes =
        totals.activation_bytes + totals.weight_bytes + totals.result_bytes;
    const double gmac_s = total_ms > 0.0 ? (double) macs / (total_ms * 1000000.0) : 0.0;
    const double mib_s = total_ms > 0.0 ? (double) effective_bytes * 1000.0 /
        (total_ms * 1024.0 * 1024.0) : 0.0;
    const double cycles_per_run =
        (g_fpga_clock_mhz > 0.0 && totals.vpu_runs > 0) ?
        ((double) totals.ip_compute_us * g_fpga_clock_mhz / (double) totals.vpu_runs) : 0.0;

    LOGSTAGE("tensor=%s layer=%d shape=K%lld_N%lld_M%lld path=zdma_ddr_to_ip prep_ms=%.3f input_dma_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f ip_compute_ms=%.3f output_dma_ms=%.3f total_ms=%.3f dominant=%s effective_GMAC/s=%.3f effective_MiB/s=%.1f act_bytes=%zu weight_bytes=%zu result_bytes=%zu",
             tensor_name,
             effective_layer_id,
             (long long) k,
             (long long) n,
             (long long) m,
             prep_ms,
             dma_in_ms,
             dma_act_ms,
             dma_weight_ms,
             ip_ms,
             dma_out_ms,
             total_ms,
             dominant,
             gmac_s,
             mib_s,
             totals.activation_bytes,
             totals.weight_bytes,
             totals.result_bytes);
    LOGIP("vpu_runs=%lld ip_ms=%.3f cycles_per_run=%.1f clock_mhz=%.3f",
          totals.vpu_runs, ip_ms, cycles_per_run, g_fpga_clock_mhz);

    if (g_status_stderr && (g_fpga_count == 1 || (g_profile_every > 0 && (g_fpga_count % g_profile_every) == 0))) {
        fprintf(stderr,
                "[FPGA][STAGE] tensor=%s layer=%d K=%lld N=%lld M=%lld total_ms=%.3f dma_in_ms=%.3f ip_ms=%.3f dma_out_ms=%.3f\n",
                tensor_name,
                effective_layer_id,
                (long long) k,
                (long long) n,
                (long long) m,
                total_ms,
                dma_in_ms,
                ip_ms,
                dma_out_ms);
        fflush(stderr);
    }

    if (macs >= g_large_matrix_min_macs) {
        LOGSTAGE("large tensor=%s layer=%d macs=%lld row_tiles=%lld q8_blocks=%lld vpu_runs=%lld no_cpu_fallback=1",
                 tensor_name,
                 effective_layer_id,
                 macs,
                 row_tiles,
                 (long long) q8_blocks,
                 totals.vpu_runs);
    }

    (void) dominant_ms;
    (void) g_current_layer_id;
    (void) g_is_attention_op;
    pthread_mutex_unlock(&g_mutex);
    return 1;
}

extern "C" void fpga_reset_kv_cache(void) {
    g_current_seq_pos = 0;
    g_scratch.activation_cache_valid = false;
}
