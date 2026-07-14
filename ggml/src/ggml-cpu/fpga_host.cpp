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
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>

#define FPGA_LOG_FILE "/tmp/fpga_debug.log"
#define FPGA_HOST_TRACE_VERSION "zcu104-gemma3-q8-c0-cache-metrics-v16"

#define MY_IP_BASE_ADDRESS 0x00000000A0000000LL
#define REG_BASE_PHYS      0x00000000A0000000LL
#define LMM_BASE_PHYS      0x00000000A0000000LL

#define DMA_BASE_PHYS      0x00000000fd500000LL
#define DMA_MMAP_SIZE      0x0000000000010000LL

#define DDR_BASE_PHYS      0x0000000800000000LL
#define DDR_MMAP_SIZE      0x0000000080000000LL

static int g_log_flush_every = 256;
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

static unsigned long long fpga_ptr_addr(const volatile void * ptr) {
    return (unsigned long long) reinterpret_cast<uintptr_t>(ptr);
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

#define LOGI(fmt, ...)      fpga_log_line(true, "INFO",    false, fmt, ##__VA_ARGS__)
#define LOGINIT(fmt, ...)   fpga_log_line(g_init_verbose, "INFO", false, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...)      fpga_log_line(true, "ERROR",   true,  fmt, ##__VA_ARGS__)
#define LOGDMA(fmt, ...)    fpga_log_line(g_dma_timing_enabled, "DMA", false, fmt, ##__VA_ARGS__)
#define LOGIP(fmt, ...)     fpga_log_line(g_ip_timing_enabled, "IPTIME", false, fmt, ##__VA_ARGS__)
#define LOGSTAGE(fmt, ...)  fpga_log_line(g_stage_timing_enabled, "STAGE", false, fmt, ##__VA_ARGS__)
#define LOGTOKEN(fmt, ...)  fpga_log_line(g_stage_timing_enabled, "TOKEN", false, fmt, ##__VA_ARGS__)
#define LOGDATA(fmt, ...)   fpga_log_line(g_trace_data_enabled, "DATA", false, fmt, ##__VA_ARGS__)

static constexpr uint64_t VPU_BASE_PHYS      = MY_IP_BASE_ADDRESS;
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
static constexpr uint32_t REG_STREAM_PROTOCOL_VERSION = 0x000000F4;
static constexpr uint32_t REG_BITSTREAM_ID   = 0x000000F8;
static constexpr uint32_t REG_BANK           = 0x00000100;
static constexpr uint32_t REG_JOB_ID         = 0x00000110;
static constexpr uint32_t REG_BANK_STAT      = 0x00000120;
static constexpr uint32_t REG_ACTIVE_JOB     = 0x00000130;
static constexpr uint32_t REG_DONE_JOB       = 0x00000140;
static constexpr uint32_t REG_SLOT_STATE     = 0x00000150;
static constexpr uint32_t REG_TENSOR_ID      = 0x00000160;
static constexpr uint32_t REG_ROW0           = 0x00000170;
static constexpr uint32_t REG_K_BLOCK0       = 0x00000180;
static constexpr uint32_t REG_GROUP_BLOCKS   = 0x00000190;
static constexpr uint32_t REG_TOKEN_ID       = 0x000001A0;
static constexpr uint32_t REG_DESC_FLAGS     = 0x000001B0;
static constexpr uint32_t REG_SPU_CAPS       = 0x000000F0;
static constexpr uint32_t REG_SPU_STREAM_COUNT = 0x000001C0;
static constexpr uint32_t REG_SPU_STREAM_DONE  = 0x000001C4;
static constexpr uint32_t REG_SPU_STREAM_DROP  = 0x000001D0;
static constexpr uint32_t REG_SPU_STREAM_OUT   = 0x000001D4;
static constexpr uint32_t REG_SPU_STREAM_ERROR = 0x000001D8;
static constexpr uint32_t REG_SPU_STREAM_LAST_JOB = 0x000001E8;
static constexpr uint32_t REG_SPU_STREAM_LAST_BANK = 0x000001EC;

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
static constexpr uint32_t SPU_OUT_BASE       = 0x00340000;
static constexpr uint32_t SPU_OUT_END        = 0x00380000;
static constexpr uint32_t SPU_PARAM_BASE     = 0x00380000;
static constexpr uint32_t SPU_PARAM_END      = 0x003C0000;
static constexpr uint32_t SPU_SCRATCH_END    = 0x00400000;
// All host-visible MY_IP registers and local-memory windows used by this
// driver lie below this offset.  The Vivado segment is 256 MiB, but the
// compatibility /dev/mem path must not map the whole segment unnecessarily.
static constexpr size_t   VPU_DEVMEM_COMPAT_MMAP = SPU_SCRATCH_END;
static constexpr size_t   DDR_REQUIRED_BYTES = SPU_SCRATCH_END;
static constexpr uint32_t WEIGHT_CACHE_BASE  = 0x01000000;
static constexpr size_t   WEIGHT_CACHE_ALIGN = 4096;

static constexpr int      VPU_NUM_LANES      = 16;
static constexpr int      VPU_QK8_0          = 32;
static constexpr int      VPU_BLOCK_BEATS    = VPU_QK8_0 / VPU_NUM_LANES;
static constexpr int      VPU_RESULT_PACK_LANES = 4;
static constexpr int      VPU_PACKED_Q8_MAX_BLOCKS = 64;
static constexpr int      VPU_DEFAULT_ROWS   = 256;
static constexpr int      VPU_SAFE_RUNTIME_ROWS = 256;
static constexpr int      VPU_DEFAULT_BEATS  = 128;
static constexpr int      VPU_DEFAULT_COLS   = VPU_DEFAULT_BEATS * VPU_NUM_LANES;
static constexpr int      VPU_LEGACY_PACKED_Q8_MAX_BLOCKS = 16;
static constexpr int      VPU_LEGACY_BEATS   = 32;
static constexpr uint32_t VPU_MODE_PACKED_Q8 = 0x00000001;
static constexpr uint32_t VPU_FP16_ONE       = 0x00003C00;
static constexpr uint32_t VPU_CAP_PACKED_Q8  = 0x00000001;
static constexpr uint32_t VPU_CAP_COMPACT_WEIGHT_LAYOUT = 0x00000002;
static constexpr uint32_t VPU_CAP_PINGPONG_BANKS        = 0x00000008;
static constexpr uint32_t VPU_CAP_JOB_DESCRIPTOR         = 0x00000010;
static constexpr uint32_t VPU_CAP_SPU_RAW_STREAM         = 0x00000020;
static constexpr uint32_t VPU_CAP_SPU_Q8_SCALE_STREAM    = 0x00000040;
static constexpr uint32_t SPU_CAP_VPU_RAW_STREAM         = 0x00000100;
static constexpr uint32_t SPU_CAP_VPU_Q8_SCALE_STREAM    = 0x00000200;
static constexpr uint32_t SPU_CAP_SILU_MUL               = 0x00000004;
static constexpr uint32_t SPU_CAP_RMSNORM                = 0x00000008;
static constexpr uint32_t SPU_CAP_ROPE                   = 0x00000010;
static constexpr uint32_t SPU_CAP_SOFTMAX                = 0x00000020;
static constexpr uint32_t FPGA_REQUIRED_STREAM_PROTOCOL_VERSION = 1;
static constexpr uint32_t FPGA_EXPECTED_BITSTREAM_ID = 0x56505531U; // "VPU1"

static constexpr long long FPGA_DEFAULT_DMA_TIMEOUT_US = 5000000LL;
static constexpr long long FPGA_DEFAULT_IP_TIMEOUT_US  = 5000000LL;
static constexpr int      FPGA_DEFAULT_STATUS_EVERY    = 0;
static constexpr int      FPGA_DEFAULT_PROFILE_EVERY   = 1;
static constexpr int      FPGA_DEFAULT_DETAIL_EVERY    = 0;
static constexpr long long FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS = 1000000LL;
static constexpr long long FPGA_STREAM_POLL_LOG_INTERVAL_US = 50000LL;

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
    long long dma_scale_us;
    long long dma_result_us;
    long long ip_compute_us;
    long long host_result_us;
    long long host_accum_us;
    long long weight_cache_hits;
    long long weight_cache_misses;
    long long weight_cache_lookup_us;
    long long weight_cache_crc_us;
    long long weight_pack_us;
    size_t    activation_bytes;
    size_t    weight_bytes;
    size_t    scale_bytes;
    size_t    result_bytes;
    long long vpu_runs;
} fpga_stage_totals_t;

enum fpga_slot_state_t : uint32_t {
    FPGA_SLOT_FREE           = 0,
    FPGA_SLOT_CPU_PACKING    = 1,
    FPGA_SLOT_DMA_FILLING    = 2,
    FPGA_SLOT_READY          = 3,
    FPGA_SLOT_COMPUTING      = 4,
    FPGA_SLOT_RESULT_READY   = 5,
    FPGA_SLOT_DMA_DRAINING   = 6,
    FPGA_SLOT_HOST_CONSUMING = 7,
};

typedef struct {
    int64_t  row0;
    int      rows;
    int64_t  k_block0;
    int      group_blocks;
    int      group_beats;
    uint32_t ddr_off;
    size_t   bytes;
    size_t   scale_off;
} fpga_weight_tile_cache_t;

typedef struct {
    const struct ggml_tensor * tensor;
    const void *               data;
    int64_t                    k;
    int64_t                    n;
    size_t                     nb1;
    int                        max_rows;
    int                        max_beats;
    int                        max_group_blocks;
    uint32_t                   base_off;
    size_t                     bytes;
    uint32_t                   header_off;
    uint32_t                   payload_crc32;
    bool                       valid;
    bool                       crc_validated;
    std::vector<fpga_weight_tile_cache_t> tiles;
    std::vector<float>                  scales;
} fpga_weight_cache_entry_t;

typedef struct {
    uint32_t magic;
    uint32_t format_version;
    uint32_t tensor_hash;
    uint32_t tile_shape;
    uint32_t ddr_offset;
    uint32_t byte_length;
    uint32_t crc32;
    uint32_t valid;
} fpga_weight_cache_header_t;

static_assert(sizeof(fpga_weight_cache_header_t) == 32,
              "unexpected weight-cache header layout");

static constexpr uint32_t FPGA_WEIGHT_CACHE_MAGIC = 0x46504348U; // "FPCH"
static constexpr uint32_t FPGA_WEIGHT_CACHE_FORMAT_VERSION = 1U;

typedef struct {
    int bank;
    uint32_t job_id;
    uint32_t tile_id;
    uint32_t tensor_id;
    int64_t row0;
    int rows;
    int64_t k_block0;
    int group_blocks;
    int group_beats;
    int64_t col;
    size_t act_bytes;
    size_t weight_bytes;
    size_t scale_bytes;
    size_t spu_result_bytes;
    size_t result_bytes;
    uint32_t result_values;
    uint32_t result_words;
    uint32_t scale_words;
    uint32_t weight_src_off;
    bool weight_cache_hit;
    const block_q8_0_t * act_group;
    const struct ggml_tensor * src0;
    const fpga_weight_cache_entry_t * weight_cache;
    std::vector<int32_t> partial;
    std::vector<float> weight_scales;
    long long result_clear_us;
    long long dma_act_us;
    long long dma_weight_us;
    long long dma_scale_us;
    long long dma_result_us;
    long long ip_start_us;
    long long ip_compute_us;
    long long host_result_us;
    uint32_t vpu_status;
    uint32_t spu_stream_count_before;
    uint32_t spu_stream_done_before;
    uint32_t spu_stream_out_before;
    uint32_t spu_stream_drop_before;
    uint32_t spu_stream_error_before;
} fpga_tile_job_t;

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
static size_t              g_ddr_advertised_size = 0;
static size_t              g_ddr_requested_map_size = DDR_REQUIRED_BYTES;
static std::string         g_dma_map_source;
static std::string         g_vpu_map_source;
static std::string         g_ddr_map_source;
static pthread_mutex_t     g_mutex         = PTHREAD_MUTEX_INITIALIZER;
static fpga_scratch_t      g_scratch;
static std::vector<fpga_weight_cache_entry_t> g_weight_cache;

static long long           g_fpga_start_us = 0;
static long long           g_fpga_count    = 0;
static long long           g_fpga_vpu_runs = 0;
static long long           g_reject_count  = 0;
static long long           g_activation_cache_hits = 0;
static long long           g_activation_cache_misses = 0;
static long long           g_weight_cache_builds = 0;
static long long           g_weight_cache_hits = 0;
static long long           g_weight_cache_misses = 0;
static long long           g_weight_cache_bytes = 0;
static long long           g_weight_cache_lookup_us = 0;
static long long           g_weight_cache_crc_us = 0;
static long long           g_weight_pack_us = 0;
static long long           g_last_token_us = 0;
static int                 g_last_token_seq = INT_MIN;
static long long           g_token_matmuls = 0;
static long long           g_vocab_projection_bypass_count = 0;

static int                 g_vpu_max_rows  = VPU_DEFAULT_ROWS;
static int                 g_vpu_max_beats = VPU_DEFAULT_BEATS;
static int                 g_vpu_max_cols  = VPU_DEFAULT_COLS;
static int                 g_packed_q8_supported = 0;
static int                 g_packed_q8_max_blocks = 1;
static int                 g_packed_q8_result_words = VPU_DEFAULT_ROWS;
static bool                g_vpu_pingpong_supported = false;
static bool                g_vpu_descriptor_supported = false;
static bool                g_spu_q8_scale_stream_supported = false;
static bool                g_spu_silu_supported = false;
static bool                g_spu_rmsnorm_supported = false;
static bool                g_spu_rope_supported = false;
static bool                g_spu_softmax_supported = false;
static bool                g_pingpong_scheduler_enabled = false;
static uint32_t            g_next_job_id = 1;

static bool                g_dma_timing_enabled = false;
static bool                g_ip_timing_enabled = false;
static bool                g_stage_timing_enabled = true;
static bool                g_init_verbose = false;
static bool                g_status_stderr = false;
static bool                g_trace_data_enabled = false;
static bool                g_cleanup_done = false;
static bool                g_atexit_registered = false;
static bool                g_abort_on_cpu_fallback = true;
static bool                g_uio_inventory_logged = false;
static bool                g_allow_devmem_fallback = false;
static bool                g_allow_vpu_devmem_compat = true;
static bool                g_strict_coherency = false;
static bool                g_coherency_platform_whitelisted = false;
static bool                g_run_coherency_stress = false;
static bool                g_ddr_msync_unsupported_logged = false;
static bool                g_weight_cache_enabled = false;
static bool                g_weight_cache_full_logged = false;
static bool                g_weight_cache_crc_verify_each_lookup = false;
static bool                g_activation_cache_enabled = false;
static bool                g_contract_check_abort = false;
static bool                g_clear_result_before_run = false;
static bool                g_contract_raw_repair_enabled = false;
static bool                g_vocab_projection_cpu_bypass = false;
static int                 g_profile_every = FPGA_DEFAULT_PROFILE_EVERY;
static int                 g_ip_status_every = FPGA_DEFAULT_STATUS_EVERY;
static int                 g_detail_every = FPGA_DEFAULT_DETAIL_EVERY;
static int                 g_contract_check_limit = 0;
static int                 g_contract_raw_retry_limit = 1;
static int                 g_runtime_max_rows = VPU_SAFE_RUNTIME_ROWS;
static int64_t             g_vocab_projection_min_n = 65536;
static long long           g_dma_timeout_us = FPGA_DEFAULT_DMA_TIMEOUT_US;
static long long           g_ip_timeout_us = FPGA_DEFAULT_IP_TIMEOUT_US;
static long long           g_large_matrix_min_macs = FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS;
static double              g_fpga_clock_mhz = 0.0;
static double              g_contract_atol = 1.0e-3;
static double              g_contract_rtol = 1.0e-4;
static long long           g_contract_checks_done = 0;
static long long           g_contract_raw_mismatches = 0;
static long long           g_contract_raw_repairs = 0;
static long long           g_contract_value_mismatches = 0;
static uint32_t            g_weight_cache_next_off = WEIGHT_CACHE_BASE;
static uint32_t            g_weight_cache_end_off  = WEIGHT_CACHE_BASE;
static uint32_t            g_stream_protocol_version = 0;
static uint32_t            g_bitstream_id = 0;
static long long           g_weight_cache_budget_mb = 0;

static int g_current_layer_id = 0;
int        g_current_seq_pos  = 0;
static int g_is_attention_op  = 0;
static long long g_attention_bypass_count = 0;

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

static bool should_log_detail_run(long long run_id) {
    return g_detail_every > 0 && ((run_id % g_detail_every) == 0);
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

static uint32_t fpga_next_job_id(void) {
    uint32_t id = g_next_job_id++;
    if (id == 0U) {
        id = g_next_job_id++;
    }
    return id;
}

static uint32_t fpga_tensor_id_from_ptr(const struct ggml_tensor * tensor) {
    uintptr_t value = (uintptr_t) tensor;
    value ^= value >> 32;
    value ^= value >> 16;
    return (uint32_t) value;
}

static uint32_t fpga_slot_state_word(int bank, fpga_slot_state_t input_state, fpga_slot_state_t output_state) {
    const uint32_t b = (uint32_t) (bank & 1);
    return ((uint32_t) input_state  << (4U * b)) |
           ((uint32_t) output_state << (16U + 4U * b));
}

static void vpu_select_banks(int wr_bank, int rd_bank) {
    if (!g_vpu_pingpong_supported) {
        return;
    }
    vpu_wr32(REG_BANK, ((uint32_t) (wr_bank & 1)) | (((uint32_t) (rd_bank & 1)) << 1));
}

static void vpu_write_tile_descriptor(const fpga_tile_job_t & job,
                                      fpga_slot_state_t input_state,
                                      fpga_slot_state_t output_state,
                                      uint32_t flags) {
    if (!g_vpu_descriptor_supported) {
        return;
    }
    vpu_wr32(REG_JOB_ID, job.job_id);
    vpu_wr32(REG_SLOT_STATE, fpga_slot_state_word(job.bank, input_state, output_state));
    vpu_wr32(REG_TENSOR_ID, job.tensor_id);
    vpu_wr32(REG_ROW0, (uint32_t) job.row0);
    vpu_wr32(REG_K_BLOCK0, (uint32_t) job.k_block0);
    vpu_wr32(REG_GROUP_BLOCKS, (uint32_t) job.group_blocks);
    vpu_wr32(REG_TOKEN_ID, (uint32_t) g_current_seq_pos);
    vpu_wr32(REG_DESC_FLAGS, flags);
}

static bool vpu_verify_done_job(const fpga_tile_job_t & job, uint32_t status) {
    if (!g_vpu_pingpong_supported) {
        return true;
    }

    const uint32_t bank_stat = vpu_rd32(REG_BANK_STAT);
    const uint32_t done_job = g_vpu_descriptor_supported ? vpu_rd32(REG_DONE_JOB) : job.job_id;
    const int done_bank = (int) ((bank_stat >> 9) & 1U);
    if (done_bank != (job.bank & 1)) {
        LOGE("VPU done bank mismatch tensor=%s tile=%u job=%u expected_bank=%d done_bank=%d status=0x%08x bank_stat=0x%08x",
             job.src0 ? job.src0->name : "?",
             job.tile_id,
             job.job_id,
             job.bank & 1,
             done_bank,
             status,
             bank_stat);
        return false;
    }
    if (g_vpu_descriptor_supported && done_job != job.job_id) {
        LOGE("VPU done job mismatch tensor=%s tile=%u expected_job=%u done_job=%u bank=%d status=0x%08x bank_stat=0x%08x active_job=%u",
             job.src0 ? job.src0->name : "?",
             job.tile_id,
             job.job_id,
             done_job,
             job.bank & 1,
             status,
             bank_stat,
             vpu_rd32(REG_ACTIVE_JOB));
        return false;
    }
    return true;
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

static void ddr_zero_range32(uint32_t off, size_t bytes) {
    if ((off & 0x3U) != 0 || (bytes & 0x3U) != 0) {
        fpga_fatal("DDR zero requires 32-bit alignment off=0x%08x bytes=%zu", off, bytes);
    }
    volatile uint32_t * dst = (volatile uint32_t *) ddr_ptr(off, bytes);
    const size_t words = bytes / sizeof(uint32_t);
    for (size_t i = 0; i < words; ++i) {
        dst[i] = 0U;
    }
    mmio_fence();
}

static void ddr_write_i8x16(uint32_t off, const int8_t * lanes) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR i8x16 write requires 32-bit alignment off=0x%08x", off);
    }
    volatile uint32_t * dst = (volatile uint32_t *) ddr_ptr(off, 16U);
    for (int w = 0; w < 4; ++w) {
        const uint8_t b0 = (uint8_t) lanes[4 * w + 0];
        const uint8_t b1 = (uint8_t) lanes[4 * w + 1];
        const uint8_t b2 = (uint8_t) lanes[4 * w + 2];
        const uint8_t b3 = (uint8_t) lanes[4 * w + 3];
        dst[w] = ((uint32_t) b0) |
                 ((uint32_t) b1 << 8) |
                 ((uint32_t) b2 << 16) |
                 ((uint32_t) b3 << 24);
    }
}

static void ddr_write_u32(uint32_t off, uint32_t value) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR u32 write requires 32-bit alignment off=0x%08x", off);
    }
    volatile uint32_t * dst = (volatile uint32_t *) ddr_ptr(off, sizeof(uint32_t));
    *dst = value;
}

static void ddr_read_i32x4(uint32_t off, int32_t out[4]) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR i32x4 read requires 32-bit alignment off=0x%08x", off);
    }
    volatile const uint32_t * src = (volatile const uint32_t *) ddr_ptr(off, 16U);
    for (int w = 0; w < 4; ++w) {
        out[w] = (int32_t) src[w];
    }
}

static uint32_t fpga_crc32_ddr(uint32_t off, size_t bytes) {
    volatile const uint8_t * data = (volatile const uint8_t *) ddr_ptr(off, bytes);
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= (uint32_t) data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
        }
    }
    return ~crc;
}

static void ddr_write_weight_cache_header(
        uint32_t off,
        const fpga_weight_cache_header_t & header) {
    if ((off & 0x3U) != 0U) {
        fpga_fatal("weight-cache header requires 32-bit alignment off=0x%08x", off);
    }
    volatile uint32_t * dst = (volatile uint32_t *) ddr_ptr(off, sizeof(header));
    const uint32_t * src = (const uint32_t *) &header;
    for (size_t i = 0; i < sizeof(header) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    mmio_fence();
}

static fpga_weight_cache_header_t ddr_read_weight_cache_header(uint32_t off) {
    fpga_weight_cache_header_t header = {};
    if ((off & 0x3U) != 0U) {
        return header;
    }
    volatile const uint32_t * src =
        (volatile const uint32_t *) ddr_ptr(off, sizeof(header));
    uint32_t * dst = (uint32_t *) &header;
    for (size_t i = 0; i < sizeof(header) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    return header;
}

static int64_t ddr_read_spu_q16_row(uint32_t off, uint16_t * row_id) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR SPU row read requires 32-bit alignment off=0x%08x", off);
    }
    volatile const uint32_t * src = (volatile const uint32_t *) ddr_ptr(off, 16U);
    if (row_id) {
        *row_id = (uint16_t) (src[0] & 0xffffU);
    }
    uint64_t accum_u =
        ((uint64_t) (src[0] >> 16)) |
        ((uint64_t) src[1] << 16) |
        ((uint64_t) (src[2] & 0xffffU) << 48);
    return (int64_t) accum_u;
}

static size_t weight_window_bytes_for_rows(int rows, int active_beats) {
    if (rows <= 0 || active_beats <= 0) {
        return 0;
    }
    return (size_t) rows * (size_t) active_beats * 16U;
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

static bool parse_uio_map_size(const std::string & uio, size_t * map_size) {
    std::string size_text;
    if (!read_text_file("/sys/class/uio/" + uio + "/maps/map0/size", &size_text)) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(size_text.c_str(), &end, 0);
    if (errno != 0 || end == size_text.c_str() || parsed == 0ULL) {
        return false;
    }

    *map_size = (size_t) parsed;
    return true;
}

static bool parse_uio_map_addr(const std::string & uio, uint64_t * map_addr) {
    std::string addr_text;
    if (!read_text_file("/sys/class/uio/" + uio + "/maps/map0/addr", &addr_text)) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(addr_text.c_str(), &end, 0);
    if (errno != 0 || end == addr_text.c_str()) {
        return false;
    }

    *map_addr = (uint64_t) parsed;
    return true;
}

static bool parse_uio_map_offset(const std::string & uio, uint64_t * map_offset) {
    std::string offset_text;
    if (!read_text_file("/sys/class/uio/" + uio + "/maps/map0/offset", &offset_text)) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(offset_text.c_str(), &end, 0);
    if (errno != 0 || end == offset_text.c_str()) {
        return false;
    }

    *map_offset = (uint64_t) parsed;
    return true;
}

static bool uio_name_from_dev_path(const std::string & dev_path, std::string * uio_name) {
    const size_t slash = dev_path.find_last_of('/');
    const std::string base = slash == std::string::npos ? dev_path : dev_path.substr(slash + 1);
    if (base.compare(0, 3, "uio") != 0) {
        return false;
    }
    *uio_name = base;
    return true;
}

static void log_uio_inventory_once(void) {
    if (g_uio_inventory_logged) {
        return;
    }
    g_uio_inventory_logged = true;

    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        LOGINIT("UIO inventory unavailable: /sys/class/uio cannot be opened errno=%d (%s)",
                errno, strerror(errno));
        return;
    }

    bool any = false;
    struct dirent * ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        const std::string uio = ent->d_name;
        std::string name;
        std::string addr;
        std::string size;
        read_text_file("/sys/class/uio/" + uio + "/name", &name);
        read_text_file("/sys/class/uio/" + uio + "/maps/map0/addr", &addr);
        read_text_file("/sys/class/uio/" + uio + "/maps/map0/size", &size);
        LOGINIT("UIO inventory dev=/dev/%s name=%s addr=%s size=%s",
                uio.c_str(),
                name.empty() ? "?" : name.c_str(),
                addr.empty() ? "?" : addr.c_str(),
                size.empty() ? "?" : size.c_str());
        any = true;
    }
    closedir(dir);

    if (!any) {
        LOGINIT("UIO inventory: /sys/class/uio exists but contains no uio devices");
    }
}

static bool find_uio_device(const char * wanted_name, std::string * dev_path, size_t * map_size) {
    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        LOGINIT("UIO lookup for name=%s failed: /sys/class/uio cannot be opened errno=%d (%s)",
                wanted_name, errno, strerror(errno));
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

        size_t parsed = 0;
        if (!parse_uio_map_size(uio, &parsed)) {
            continue;
        }

        *dev_path = "/dev/" + uio;
        *map_size = parsed;
        found = true;
        break;
    }

    closedir(dir);
    if (!found) {
        LOGINIT("UIO name=%s not found; trying physical-address match",
                wanted_name);
        log_uio_inventory_once();
    }
    return found;
}

static bool find_uio_device_by_addr(uint64_t wanted_addr, std::string * dev_path, size_t * map_size, std::string * resolved_name) {
    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        LOGINIT("UIO address lookup for addr=0x%llx failed: /sys/class/uio cannot be opened errno=%d (%s)",
                (unsigned long long) wanted_addr, errno, strerror(errno));
        return false;
    }

    bool found = false;
    struct dirent * ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        const std::string uio = ent->d_name;
        uint64_t addr = 0;
        size_t size = 0;
        if (!parse_uio_map_addr(uio, &addr) || !parse_uio_map_size(uio, &size)) {
            continue;
        }
        if (addr != wanted_addr) {
            continue;
        }

        *dev_path = "/dev/" + uio;
        *map_size = size;
        if (!read_text_file("/sys/class/uio/" + uio + "/name", resolved_name)) {
            *resolved_name = "?";
        }
        found = true;
        break;
    }

    closedir(dir);
    if (!found) {
        LOGINIT("UIO addr=0x%llx not found",
                (unsigned long long) wanted_addr);
    }
    return found;
}

static bool find_uio_device_from_ref(const char * ref, std::string * dev_path, size_t * map_size, std::string * resolved_name) {
    if (!ref || ref[0] == '\0') {
        return false;
    }

    std::string text(ref);
    std::string uio;
    if (text.compare(0, 8, "/dev/uio") == 0) {
        const size_t slash = text.find_last_of('/');
        uio = slash == std::string::npos ? text : text.substr(slash + 1);
        *dev_path = text;
    } else if (text.compare(0, 3, "uio") == 0) {
        uio = text;
        *dev_path = "/dev/" + text;
    } else {
        if (!find_uio_device(ref, dev_path, map_size)) {
            return false;
        }
        *resolved_name = text;
        return true;
    }

    if (!parse_uio_map_size(uio, map_size)) {
        LOGE("UIO override %s has no readable map0 size under /sys/class/uio/%s", ref, uio.c_str());
        return false;
    }
    if (!read_text_file("/sys/class/uio/" + uio + "/name", resolved_name)) {
        *resolved_name = "?";
    }
    return true;
}

static bool map_uio_region(
        const char * uio_name,
        const char * env_name,
        uint64_t phys,
        size_t required_size,
        const char * tag,
        size_t requested_map_size,
        void ** map_base,
        size_t * map_size,
        size_t * advertised_size,
        std::string * source) {
    std::string dev_path;
    std::string resolved_name;
    size_t uio_size = 0;
    const char * override_ref = env_name ? getenv(env_name) : nullptr;
    if (override_ref && override_ref[0] != '\0') {
        if (!find_uio_device_from_ref(override_ref, &dev_path, &uio_size, &resolved_name)) {
            LOGE("%s=%s could not be resolved for %s",
                 env_name, override_ref, tag);
            return false;
        }
        LOGINIT("using %s=%s for %s resolved_dev=%s resolved_name=%s",
                env_name, override_ref, tag, dev_path.c_str(), resolved_name.c_str());
    } else {
        if (!find_uio_device(uio_name, &dev_path, &uio_size)) {
            if (!find_uio_device_by_addr(phys, &dev_path, &uio_size, &resolved_name)) {
                return false;
            }
            LOGINIT("using UIO addr match for %s expected_name=%s phys=0x%llx resolved_dev=%s resolved_name=%s",
                    tag, uio_name, (unsigned long long) phys, dev_path.c_str(), resolved_name.c_str());
        } else {
            resolved_name = uio_name;
        }
    }
    if (uio_size < required_size) {
        LOGE("UIO %s for %s is too small: size=0x%zx required=0x%zx",
             dev_path.c_str(), tag, uio_size, required_size);
        return false;
    }

    std::string uio;
    uint64_t uio_addr = 0;
    uint64_t uio_offset = 0;
    if (!uio_name_from_dev_path(dev_path, &uio) ||
        !parse_uio_map_addr(uio, &uio_addr) ||
        !parse_uio_map_offset(uio, &uio_offset)) {
        LOGE("UIO %s for %s has unreadable map0 addr/offset metadata", dev_path.c_str(), tag);
        return false;
    }
    if (uio_addr != phys || uio_offset != 0U) {
        LOGE("UIO %s for %s does not match required resource: expected_phys=0x%llx got_phys=0x%llx map_offset=0x%llx",
             dev_path.c_str(), tag, (unsigned long long) phys,
             (unsigned long long) uio_addr, (unsigned long long) uio_offset);
        return false;
    }

    const size_t actual_map_size = requested_map_size == 0 ? uio_size : requested_map_size;
    if (actual_map_size < required_size || actual_map_size > uio_size) {
        LOGE("UIO %s for %s cannot satisfy requested mapping: requested=0x%zx required=0x%zx advertised=0x%zx",
             dev_path.c_str(), tag, actual_map_size, required_size, uio_size);
        return false;
    }

    int fd = open(dev_path.c_str(), O_RDWR | O_SYNC);
    if (fd < 0) {
        LOGE("open %s for %s failed errno=%d (%s)",
             dev_path.c_str(), tag, errno, strerror(errno));
        return false;
    }
    void * ptr = mmap(nullptr, actual_map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int saved_errno = errno;
    close(fd);
    if (ptr == MAP_FAILED) {
        LOGE("mmap %s for %s size=0x%zx failed errno=%d (%s)",
             dev_path.c_str(), tag, actual_map_size, saved_errno, strerror(saved_errno));
        return false;
    }

    *map_base = ptr;
    *map_size = actual_map_size;
    if (advertised_size) {
        *advertised_size = uio_size;
    }
    *source = dev_path + "(" + resolved_name + ",O_SYNC)";
    LOGINIT("mapped %s via UIO expected_name=%s resolved_name=%s dev=%s phys=0x%llx virt=%p mapped_size=0x%zx advertised_size=0x%zx",
            tag, uio_name, resolved_name.c_str(), dev_path.c_str(),
            (unsigned long long) uio_addr, ptr, actual_map_size, uio_size);
    return true;
}

static bool ensure_mem_fd(void) {
    if (g_mem_fd >= 0) {
        return true;
    }
    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) {
        const int saved_errno = errno;
        struct stat st;
        if (stat("/dev/mem", &st) == 0) {
            LOGE("/dev/mem mode=0%o uid=%u gid=%u process_uid=%u process_euid=%u",
                 (unsigned int) (st.st_mode & 07777),
                 (unsigned int) st.st_uid,
                 (unsigned int) st.st_gid,
                 (unsigned int) getuid(),
                 (unsigned int) geteuid());
        }
        LOGE("open /dev/mem failed errno=%d (%s). If process_euid is 0, this is kernel/device policy, not a sudo problem.",
             saved_errno, strerror(saved_errno));
        LOGE("Use UIO mappings instead: set FPGA_DMA_UIO, FPGA_VPU_UIO, FPGA_DDR_UIO to /dev/uioX or to the UIO names shown above");
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
    LOGINIT("mapped %s via /dev/mem phys=0x%llx virt=%p size=0x%zx",
            tag, (unsigned long long) phys, ptr, bytes);
    return true;
}

static bool map_region_prefer_uio(
        const char * uio_name,
        const char * env_name,
        uint64_t phys,
        size_t devmem_size,
        size_t required_size,
        const char * tag,
        size_t requested_map_size,
        bool allow_devmem_fallback,
        const char * fallback_policy,
        void ** map_base,
        size_t * map_size,
        size_t * advertised_size,
        std::string * source) {
    if (map_uio_region(uio_name, env_name, phys, required_size, tag, requested_map_size,
                       map_base, map_size, advertised_size, source)) {
        return true;
    }
    if (!allow_devmem_fallback) {
        LOGE("mapping denied for %s expected_phys=0x%llx required=0x%zx policy=%s; install/fix the UIO device tree node or explicitly enable the documented compatibility policy",
             tag, (unsigned long long) phys, required_size,
             fallback_policy ? fallback_policy : "uio_required");
        return false;
    }
    if (fallback_policy && strcmp(fallback_policy, "vpu_devmem_compat") == 0) {
        LOGI("MY_IP UIO is unavailable; using bounded /dev/mem compatibility mapping for %s phys=0x%llx size=0x%zx. Set FPGA_VPU_UIO_REQUIRED=1 or FPGA_VPU_DEVMEM_COMPAT=0 to require UIO.",
             tag, (unsigned long long) phys, devmem_size);
    } else {
        LOGE("DIAGNOSTIC ONLY: FPGA_ALLOW_DEVMEM=1 permits /dev/mem mapping for %s; this is not a production-safe mapping policy",
             tag);
    }
    const size_t actual_map_size = requested_map_size == 0 ? devmem_size : requested_map_size;
    const bool ok = map_devmem_region(phys, actual_map_size, tag, map_base, map_size, source);
    if (ok && advertised_size) {
        *advertised_size = actual_map_size;
    }
    return ok;
}

static bool map_registers_dma_ddr(void) {
    if (!map_region_prefer_uio("dma-controller", "FPGA_DMA_UIO", DMA_BASE_PHYS, DMA_MMAP_SIZE,
                               sizeof(dma_ctrl), "ZDMA", 0,
                               g_allow_devmem_fallback, "uio_required", &g_dma_map_base,
                               &g_dma_map_size, nullptr, &g_dma_map_source)) {
        return false;
    }
    g_dma = (volatile dma_ctrl *) g_dma_map_base;

    if (!map_region_prefer_uio("MY_IP", "FPGA_VPU_UIO", REG_BASE_PHYS, VPU_DEVMEM_COMPAT_MMAP,
                               VPU_DEVMEM_COMPAT_MMAP, "MY_IP/VPU", 0,
                               g_allow_devmem_fallback || g_allow_vpu_devmem_compat,
                               g_allow_devmem_fallback ? "diagnostic_all_resources" : "vpu_devmem_compat",
                               &g_vpu_map_base,
                               &g_vpu_map_size, nullptr, &g_vpu_map_source)) {
        return false;
    }
    g_vpu = (volatile uint8_t *) g_vpu_map_base;

    if (!map_region_prefer_uio("ddr_high", "FPGA_DDR_UIO", DDR_BASE_PHYS, DDR_DEV_MEM_MMAP,
                               DDR_REQUIRED_BYTES, "ddr_high", g_ddr_requested_map_size,
                               g_allow_devmem_fallback, "uio_required",
                               &g_ddr_map_base, &g_ddr_map_size, &g_ddr_advertised_size,
                               &g_ddr_map_source)) {
        return false;
    }
    g_ddr = (uint8_t *) g_ddr_map_base;
    return true;
}

static bool configure_ddr_mapping_policy(void) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        LOGE("cannot determine page size for ddr_high mapping");
        return false;
    }

    g_weight_cache_budget_mb = 0;
    g_ddr_requested_map_size = align_up_size(DDR_REQUIRED_BYTES, (size_t) page_size);
    if (!g_weight_cache_enabled) {
        LOGINIT("DDR map policy: cache=disabled requested_size=0x%zx", g_ddr_requested_map_size);
        return true;
    }

    const char * cache_mb_text = getenv("FPGA_WEIGHT_CACHE_MB");
    const long long cache_mb = env_int64_value("FPGA_WEIGHT_CACHE_MB", 0, 0, 16384);
    if (!cache_mb_text || cache_mb_text[0] == '\0' || cache_mb <= 0) {
        LOGE("FPGA_WEIGHT_CACHE=1 requires a positive FPGA_WEIGHT_CACHE_MB; refusing an unbounded DDR cache");
        return false;
    }

    const uint64_t payload_bytes = (uint64_t) cache_mb * 1024ULL * 1024ULL;
    const uint64_t required_bytes = (uint64_t) WEIGHT_CACHE_BASE + payload_bytes;
    if (required_bytes > DDR_DEV_MEM_MMAP) {
        LOGE("requested FPGA_WEIGHT_CACHE_MB=%lld requires 0x%llx bytes, above DDR high window 0x%zx",
             cache_mb, (unsigned long long) required_bytes, DDR_DEV_MEM_MMAP);
        return false;
    }

    g_weight_cache_budget_mb = cache_mb;
    g_ddr_requested_map_size =
        align_up_size((size_t) required_bytes, (size_t) page_size);
    LOGINIT("DDR map policy: cache=enabled budget_mb=%lld requested_size=0x%zx",
            g_weight_cache_budget_mb, g_ddr_requested_map_size);
    return true;
}

static void configure_weight_cache(void) {
    g_weight_cache.clear();
    g_weight_cache_next_off = WEIGHT_CACHE_BASE;
    g_weight_cache_end_off = WEIGHT_CACHE_BASE;
    g_weight_cache_full_logged = false;

    if (!g_weight_cache_enabled) {
        LOGINIT("weight tile cache disabled; only ACT/WEIGHT/RESULT scratch DDR windows will be touched");
        return;
    }
    if (!ddr_is_mapped() || g_ddr_map_size <= WEIGHT_CACHE_BASE) {
        g_weight_cache_enabled = false;
        LOGE("weight tile cache disabled: ddr_high size=0x%zx is too small for base=0x%08x",
             g_ddr_map_size, WEIGHT_CACHE_BASE);
        return;
    }

    if (g_weight_cache_budget_mb <= 0) {
        g_weight_cache_enabled = false;
        LOGE("weight tile cache disabled: no validated FPGA_WEIGHT_CACHE_MB budget is available");
        return;
    }

    size_t available = g_ddr_map_size - (size_t) WEIGHT_CACHE_BASE;
    const size_t budget_bytes = (size_t) g_weight_cache_budget_mb * 1024U * 1024U;
    available = std::min(available, budget_bytes);
    available = (available / WEIGHT_CACHE_ALIGN) * WEIGHT_CACHE_ALIGN;
    if (available < WEIGHT_CACHE_ALIGN) {
        g_weight_cache_enabled = false;
        LOGE("weight tile cache disabled: available bytes after limit is only %zu", available);
        return;
    }

    const uint64_t end = (uint64_t) WEIGHT_CACHE_BASE + (uint64_t) available;
    g_weight_cache_end_off = end > UINT32_MAX ? UINT32_MAX : (uint32_t) end;
    LOGI("weight tile cache enabled base=0x%08x end=0x%08x bytes=%zu budget_mb=%lld mapped_ddr=0x%zx advertised_ddr=0x%zx",
         WEIGHT_CACHE_BASE, g_weight_cache_end_off,
         (size_t) (g_weight_cache_end_off - WEIGHT_CACHE_BASE),
         g_weight_cache_budget_mb,
         g_ddr_map_size,
         g_ddr_advertised_size);
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
        const int saved_errno = errno;
        const bool likely_uncached_mapping =
            g_ddr_map_source.find("O_SYNC") != std::string::npos ||
            g_ddr_map_source.find("/dev/uio") != std::string::npos;
        if (!g_strict_coherency && !env_flag_enabled("FPGA_STRICT_MSYNC") &&
            likely_uncached_mapping &&
            (saved_errno == EINVAL || saved_errno == ENODEV)) {
            if (!g_ddr_msync_unsupported_logged) {
                LOGI("msync unsupported for ddr_high source=%s errno=%d (%s); continuing with CPU barriers/O_SYNC mapping assumption",
                     g_ddr_map_source.c_str(), saved_errno, strerror(saved_errno));
                g_ddr_msync_unsupported_logged = true;
            }
            mmio_fence();
            return true;
        }
        if ((saved_errno == EINVAL || saved_errno == ENODEV) &&
            g_strict_coherency && g_coherency_platform_whitelisted) {
            LOGI("strict coherency whitelist accepts unsupported msync for ddr_high source=%s; CPU barriers remain enabled",
                 g_ddr_map_source.c_str());
            mmio_fence();
            return true;
        }
        LOGE("msync ddr_high tag=%s off=0x%08x bytes=%zu invalidate=%d errno=%d (%s) source=%s aligned=0x%llx len=0x%zx flags=0x%x",
             tag, off, bytes, invalidate ? 1 : 0, saved_errno, strerror(saved_errno),
             g_ddr_map_source.c_str(), (unsigned long long) aligned_begin, len, flags);
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

    LOGINIT("ZDMA init base=0x%llx virt=0x%llx status=0x%08x isr=0x%08x ctrl0=0x%08x ctrl1=0x%08x data_attr=0x%08x",
            (unsigned long long) DMA_BASE_PHYS,
            fpga_ptr_addr(g_dma),
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

static bool fpga_ddr_coherency_stress_test(void) {
    const int iterations = env_int_value("FPGA_COHERENCY_STRESS_ITERS", 2048, 1, 100000);
    static constexpr size_t kBytes = 64U;
    static_assert((kBytes % sizeof(uint32_t)) == 0U, "coherency pattern alignment");
    if (!ddr_range_fits(ACT_BASE, kBytes)) {
        LOGE("coherency stress cannot access ACT staging off=0x%08x bytes=%zu", ACT_BASE, kBytes);
        return false;
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        volatile uint32_t * pattern = (volatile uint32_t *) ddr_ptr(ACT_BASE, kBytes);
        uint32_t expected[kBytes / sizeof(uint32_t)] = {};
        for (size_t word = 0; word < kBytes / sizeof(uint32_t); ++word) {
            expected[word] = 0xA5C30000U ^ ((uint32_t) iteration * 0x9E3779B9U) ^ (uint32_t) word;
            pattern[word] = expected[word];
        }
        if (!msync_ddr_range(ACT_BASE, kBytes, false, "coherency_write")) {
            return false;
        }
        if (!fpga_dma_write_to_ip(ACT_BASE, kBytes, "coherency_ddr_to_ip")) {
            return false;
        }

        for (size_t word = 0; word < kBytes / sizeof(uint32_t); ++word) {
            pattern[word] = 0U;
        }
        if (!msync_ddr_range(ACT_BASE, kBytes, false, "coherency_clear")) {
            return false;
        }
        if (!fpga_dma_read_from_ip(ACT_BASE, kBytes, "coherency_ip_to_ddr")) {
            return false;
        }

        for (size_t word = 0; word < kBytes / sizeof(uint32_t); ++word) {
            const uint32_t actual = pattern[word];
            if (actual != expected[word]) {
                LOGE("coherency stress mismatch iteration=%d word=%zu expected=0x%08x actual=0x%08x",
                     iteration, word, expected[word], actual);
                return false;
            }
        }
    }

    LOGI("coherency stress passed iterations=%d bytes_per_iteration=%zu source=%s strict=%d whitelist=%d",
         iterations, kBytes, g_ddr_map_source.c_str(), g_strict_coherency ? 1 : 0,
         g_coherency_platform_whitelisted ? 1 : 0);
    return true;
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
        if (g_ip_timing_enabled && (g_ip_status_every > 0) && ((polls % g_ip_status_every) == 0)) {
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

static bool wait_spu_stream_outputs(const fpga_tile_job_t & job) {
    const uint32_t target_out = job.spu_stream_out_before + (uint32_t) job.rows;
    const uint32_t expected_raw = job.spu_stream_count_before +
        (uint32_t) job.rows * (uint32_t) job.group_blocks;
    const uint32_t expected_done = job.spu_stream_done_before + 1U;
    const long long t0 = now_us();
    long long last_log_us = t0;
    long long polls = 0;
    uint32_t last_count = job.spu_stream_count_before;
    uint32_t last_done = job.spu_stream_done_before;
    uint32_t last_out = job.spu_stream_out_before;
    uint32_t last_drop = job.spu_stream_drop_before;
    uint32_t last_error = job.spu_stream_error_before;
    while (true) {
        const long long now = now_us();
        const uint32_t count = vpu_rd32(REG_SPU_STREAM_COUNT);
        const uint32_t done_count = vpu_rd32(REG_SPU_STREAM_DONE);
        const uint32_t out_count = vpu_rd32(REG_SPU_STREAM_OUT);
        const uint32_t drop_count = vpu_rd32(REG_SPU_STREAM_DROP);
        const uint32_t error_count = vpu_rd32(REG_SPU_STREAM_ERROR);
        if (drop_count != job.spu_stream_drop_before ||
            error_count != job.spu_stream_error_before) {
            LOGE("SPU stream failed job=%u bank=%d expected_raw=%u expected_out=%u count=%u done=%u out=%u drop_before=%u drop_now=%u error_before=%u error_now=%u active_job=%u done_job=%u bank_stat=0x%08x progress=0x%08x last_job=%u last_bank=%u",
                 job.job_id, job.bank, expected_raw, target_out, count, done_count, out_count,
                 job.spu_stream_drop_before, drop_count, job.spu_stream_error_before, error_count,
                 vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB), vpu_rd32(REG_BANK_STAT),
                 vpu_rd32(REG_PROGRESS), vpu_rd32(REG_SPU_STREAM_LAST_JOB),
                 vpu_rd32(REG_SPU_STREAM_LAST_BANK));
            return false;
        }
        if (done_count >= expected_done && count != expected_raw) {
            LOGE("SPU stream protocol_mismatch job=%u bank=%d expected_raw=%u count=%u expected_done=%u done=%u out=%u expected_out=%u active_job=%u done_job=%u bank_stat=0x%08x",
                 job.job_id, job.bank, expected_raw, count, expected_done, done_count,
                 out_count, target_out, vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB),
                 vpu_rd32(REG_BANK_STAT));
            return false;
        }
        if (out_count >= target_out) {
            if (count != expected_raw || done_count != expected_done) {
                LOGE("SPU stream completion_mismatch job=%u bank=%d expected_raw=%u count=%u expected_done=%u done=%u expected_out=%u out=%u",
                     job.job_id, job.bank, expected_raw, count, expected_done, done_count,
                     target_out, out_count);
                return false;
            }
            return true;
        }
        const bool counters_changed =
            count != last_count || done_count != last_done || out_count != last_out ||
            drop_count != last_drop || error_count != last_error;
        if (counters_changed || now - last_log_us >= FPGA_STREAM_POLL_LOG_INTERVAL_US) {
            LOGIP("SPU poll job=%u bank=%d expected_raw=%u expected_out=%u count=%u done=%u out=%u drop=%u error=%u active_job=%u done_job=%u bank_stat=0x%08x status=0x%08x progress=0x%08x last_job=%u last_bank=%u",
                  job.job_id, job.bank, expected_raw, target_out, count, done_count, out_count,
                  drop_count, error_count, vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB),
                  vpu_rd32(REG_BANK_STAT), vpu_rd32(REG_STATUS), vpu_rd32(REG_PROGRESS),
                  vpu_rd32(REG_SPU_STREAM_LAST_JOB), vpu_rd32(REG_SPU_STREAM_LAST_BANK));
            last_count = count;
            last_done = done_count;
            last_out = out_count;
            last_drop = drop_count;
            last_error = error_count;
            last_log_us = now;
        }
        if (now - t0 > g_ip_timeout_us) {
            LOGE("SPU stream timeout job=%u bank=%d expected_raw=%u expected_out=%u count=%u done=%u out=%u drop=%u error=%u active_job=%u done_job=%u bank_stat=0x%08x progress=0x%08x",
                 job.job_id, job.bank, expected_raw, target_out, count, done_count, out_count,
                 drop_count, error_count, vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB),
                 vpu_rd32(REG_BANK_STAT), vpu_rd32(REG_PROGRESS));
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
        std::vector<float> & act_scales,
        bool store_act_scales) {
    const int64_t nb = k / VPU_QK8_0;
    const bool cache_hit =
        g_activation_cache_enabled &&
        g_scratch.activation_cache_valid &&
        g_scratch.cached_src1 == src1 &&
        g_scratch.cached_src1_data == src1->data &&
        g_scratch.cached_m == m &&
        g_scratch.cached_k == k &&
        g_scratch.cached_nb0 == src1->nb[0] &&
        g_scratch.cached_nb1 == src1->nb[1] &&
        (!store_act_scales || act_scales.size() == (size_t) (m * nb));

    if (cache_hit) {
        g_activation_cache_hits++;
        return;
    }

    act_blocks_all.resize((size_t) (m * nb));
    if (store_act_scales) {
        act_scales.resize((size_t) (m * nb));
    } else {
        act_scales.clear();
    }
    for (int64_t col = 0; col < m; ++col) {
        block_q8_0_t * col_blocks = &act_blocks_all[(size_t) (col * nb)];
        quantize_activation_vector_to(src1, col, k, col_blocks);
        if (store_act_scales) {
            for (int64_t ib = 0; ib < nb; ++ib) {
                act_scales[(size_t) (col * nb + ib)] = fp16_to_fp32(col_blocks[(size_t) ib].d);
            }
        }
    }

    if (g_activation_cache_enabled) {
        g_scratch.cached_src1 = src1;
        g_scratch.cached_src1_data = src1->data;
        g_scratch.cached_m = m;
        g_scratch.cached_k = k;
        g_scratch.cached_nb0 = src1->nb[0];
        g_scratch.cached_nb1 = src1->nb[1];
        g_scratch.activation_cache_valid = true;
    } else {
        g_scratch.activation_cache_valid = false;
    }
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
    ddr_write_i8x16(off, lanes);
}

static void read_result_i32x4_from_ddr(uint32_t result_word, int32_t out[4]) {
    ddr_read_i32x4(RESULT_BASE + result_word * 16U, out);
}

static float load_dst_value(
        const struct ggml_tensor * dst,
        int64_t row,
        int64_t col) {
    const char * base = (const char *) dst->data;
    return *(const float *) (base + row * dst->nb[0] + col * dst->nb[1]);
}

static bool fpga_nonfinite_equivalent(double got, double expected) {
    if (std::isnan(got) && std::isnan(expected)) {
        return true;
    }
    if (std::isinf(got) && std::isinf(expected)) {
        return std::signbit(got) == std::signbit(expected);
    }
    return false;
}

static int32_t q8_0_raw_dot(const int8_t * a, const int8_t * w) {
    int32_t acc = 0;
    for (int i = 0; i < VPU_QK8_0; ++i) {
        acc += (int32_t) a[i] * (int32_t) w[i];
    }
    return acc;
}

static long long fpga_contract_count_raw_mismatches(
        const struct ggml_tensor * src0,
        const block_q8_0_t * act_group,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        std::vector<int32_t> & partial,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id,
        int attempt,
        bool log_mismatches,
        bool repair_mismatches) {
    long long mismatches = 0;
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * wb = weight_block(src0, row0 + row, k_block0 + gb);
            const int32_t expected = q8_0_raw_dot(act_group[gb].qs, wb->qs);
            const size_t partial_idx = (size_t) row * (size_t) group_blocks + (size_t) gb;
            const int32_t got = partial[partial_idx];
            if (got != expected) {
                if (log_mismatches && mismatches < 4) {
                    LOGE("CONTRACT_RAW_MISMATCH tensor=%s layer=%d tile=%u attempt=%d row=%lld block=%lld got=%d expected=%d act_d=%.9g weight_d=%.9g",
                         tensor_name ? tensor_name : "?",
                         layer_id,
                         tile_id,
                         attempt,
                         (long long) (row0 + row),
                         (long long) (k_block0 + gb),
                         got,
                         expected,
                         fp16_to_fp32(act_group[gb].d),
                         fp16_to_fp32(wb->d));
                }
                if (repair_mismatches) {
                    partial[partial_idx] = expected;
                }
                mismatches++;
            }
        }
    }
    return mismatches;
}

static bool fpga_contract_check_dst_values(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst,
        const std::vector<block_q8_0_t> & act_blocks_all,
        const std::vector<float> & act_scales,
        const char * tensor_name,
        int layer_id) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = dst->ne[1];
    const int64_t nb = k / VPU_QK8_0;
    long long bad = 0;
    long long nonfinite_equal = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;

    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < n; ++row) {
            float ref = 0.0f;
            for (int64_t ib = 0; ib < nb; ++ib) {
                const block_q8_0_t & act = act_blocks_all[(size_t) (col * nb + ib)];
                const block_q8_0_t * wb = weight_block(src0, row, ib);
                const int32_t raw = q8_0_raw_dot(act.qs, wb->qs);
                ref +=
                    (float) raw *
                    act_scales[(size_t) (col * nb + ib)] *
                    fp16_to_fp32(wb->d);
            }

            const double got = (double) load_dst_value(dst, row, col);
            const double expected = (double) ref;

            if (!std::isfinite(got) || !std::isfinite(expected)) {
                if (fpga_nonfinite_equivalent(got, expected)) {
                    nonfinite_equal++;
                    continue;
                }
                if (bad < 4) {
                    const double err_nan = std::numeric_limits<double>::quiet_NaN();
                    LOGE("CONTRACT_VALUE_MISMATCH tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g abs=%.9g rel=%.9g",
                         tensor_name ? tensor_name : "?",
                         layer_id,
                         (long long) row,
                         (long long) col,
                         got,
                         expected,
                         err_nan,
                         err_nan);
                }
                bad++;
                continue;
            }

            const double abs_err = std::fabs(got - expected);
            const double rel_err = abs_err / (std::fabs(expected) + 1.0e-12);
            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
            if (abs_err > g_contract_atol && rel_err > g_contract_rtol) {
                if (bad < 4) {
                    LOGE("CONTRACT_VALUE_MISMATCH tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g abs=%.9g rel=%.9g",
                         tensor_name ? tensor_name : "?",
                         layer_id,
                         (long long) row,
                         (long long) col,
                         got,
                         expected,
                         abs_err,
                         rel_err);
                }
                bad++;
            }
        }
    }

    if (bad > 0) {
        g_contract_value_mismatches += bad;
        LOGE("CONTRACT_VALUE_SUMMARY tensor=%s layer=%d checked=%lld bad=%lld nonfinite_equal=%lld max_abs=%.9g max_rel=%.9g atol=%.9g rtol=%.9g action=%s",
             tensor_name ? tensor_name : "?",
             layer_id,
             (long long) (n * m),
             bad,
             nonfinite_equal,
             max_abs,
             max_rel,
             g_contract_atol,
             g_contract_rtol,
             g_contract_check_abort ? "abort" : "log_only");
        return !g_contract_check_abort;
    }

    LOGI("CONTRACT_VALUE_PASS tensor=%s layer=%d checked=%lld nonfinite_equal=%lld max_abs=%.9g max_rel=%.9g",
         tensor_name ? tensor_name : "?",
         layer_id,
         (long long) (n * m),
         nonfinite_equal,
         max_abs,
         max_rel);
    return true;
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
        uint32_t tile_id,
        fpga_stage_totals_t * totals) {
    vpu_select_banks(0, 0);
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
    vpu_select_banks(0, 0);
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

    if (g_ip_timing_enabled && should_log_detail_run(tile_id)) {
        LOGIP("run tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u rows=%d col_beats=%d mode=0x%x act_dma_ms=%.3f weight_dma_ms=%.3f ip_ms=%.3f result_dma_ms=%.3f status=0x%08x progress=0x%08x",
              tensor_name ? tensor_name : "?",
              layer_id,
              (long long) k,
              (long long) n,
              (long long) m,
              tile_id,
              rows,
              col_beats,
              mode,
              (double) (dma_act1 - dma_act0) / 1000.0,
              (double) (dma_weight1 - dma_weight0) / 1000.0,
              (double) (ip1 - ip0) / 1000.0,
              (double) (dma_result1 - dma_result0) / 1000.0,
              vpu_status,
              vpu_rd32(REG_PROGRESS));
    }
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

    const int packed_rows = 2;
    const int packed_group_beats = 4;
    const size_t packed_weight_bytes = weight_window_bytes_for_rows(packed_rows, packed_group_beats);
    ddr_zero_range32(WEIGHT_BASE, packed_weight_bytes);

    for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) beat * 16U, act0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, act1 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + (uint32_t) beat * 16U, w_row0_block0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, w_row0_block1 + beat * VPU_NUM_LANES);
        const uint32_t row1_base = (uint32_t) packed_group_beats * 16U;
        write_i8x16_to_ddr(WEIGHT_BASE + row1_base + (uint32_t) beat * 16U, w_row1_block0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(WEIGHT_BASE + row1_base + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, w_row1_block1 + beat * VPU_NUM_LANES);
    }

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(packed_rows, packed_group_beats, VPU_MODE_PACKED_Q8,
                                 4U * 16U,
                                 packed_weight_bytes,
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

static int32_t read_result_i32_flat(uint32_t index) {
    int32_t lanes[VPU_RESULT_PACK_LANES] = {};
    read_result_i32x4_from_ddr(index / (uint32_t) VPU_RESULT_PACK_LANES, lanes);
    return lanes[index % (uint32_t) VPU_RESULT_PACK_LANES];
}

static bool fpga_dma_row_limit_self_test(void) {
    const int rows = g_vpu_max_rows;
    const int group_blocks = std::min(2, g_packed_q8_max_blocks);
    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    const uint32_t result_values = (uint32_t) rows * (uint32_t) group_blocks;
    const uint32_t result_words =
        (result_values + (uint32_t) VPU_RESULT_PACK_LANES - 1U) / (uint32_t) VPU_RESULT_PACK_LANES;

    if (rows <= 2 || group_blocks <= 0 || result_words > (uint32_t) g_packed_q8_result_words) {
        LOGI("row-limit self-test skipped rows=%d group_blocks=%d result_words=%u cap=%d",
             rows, group_blocks, result_words, g_packed_q8_result_words);
        return true;
    }

    const size_t act_bytes = (size_t) group_beats * 16U;
    const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
    const size_t result_bytes = (size_t) result_words * 16U;
    if (!range_fits(ACT_BASE, act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(WEIGHT_BASE, weight_bytes, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(RESULT_BASE, result_bytes, RESULT_BASE, RESULT_END)) {
        LOGE("row-limit self-test window overflow rows=%d group_beats=%d act=%zu weight=%zu result=%zu",
             rows, group_beats, act_bytes, weight_bytes, result_bytes);
        return false;
    }

    int8_t act[VPU_PACKED_Q8_MAX_BLOCKS][VPU_QK8_0];
    for (int gb = 0; gb < group_blocks; ++gb) {
        const int8_t act_value = (int8_t) (gb + 1);
        for (int i = 0; i < VPU_QK8_0; ++i) {
            act[gb][i] = act_value;
        }
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act[gb] + beat * VPU_NUM_LANES);
        }
    }

    ddr_zero_range32(WEIGHT_BASE, weight_bytes);
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const int8_t weight_value = (int8_t) (((row + gb) % 5) - 2);
            int8_t weight[VPU_QK8_0];
            for (int i = 0; i < VPU_QK8_0; ++i) {
                weight[i] = weight_value;
            }
            for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                const uint32_t word_index = (uint32_t) row * (uint32_t) group_beats +
                                            (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS +
                                            (uint32_t) beat;
                write_i8x16_to_ddr(WEIGHT_BASE + word_index * 16U, weight + beat * VPU_NUM_LANES);
            }
        }
    }
    mmio_fence();

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(rows, group_beats, VPU_MODE_PACKED_Q8,
                                 act_bytes, weight_bytes, result_bytes,
                                 "selftest.row_limit", -1, VPU_QK8_0 * group_blocks, rows, 1, 2, &totals)) {
        return false;
    }

    const int probe_rows[3] = {0, rows / 2, rows - 1};
    for (int probe = 0; probe < 3; ++probe) {
        const int row = probe_rows[probe];
        for (int gb = 0; gb < group_blocks; ++gb) {
            const int32_t got = read_result_i32_flat((uint32_t) row * (uint32_t) group_blocks + (uint32_t) gb);
            const int32_t expected = (int32_t) VPU_QK8_0 * (int32_t) (gb + 1) *
                (int32_t) (((row + gb) % 5) - 2);
            if (got != expected) {
                LOGE("row-limit self-test mismatch rows=%d row=%d block=%d got=%d expected=%d",
                     rows, row, gb, got, expected);
                return false;
            }
        }
    }

    LOGI("row-limit self-test passed rows=%d group_blocks=%d result_words=%u",
         rows, group_blocks, result_words);
    return true;
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

static bool weight_cache_entry_matches(
        const fpga_weight_cache_entry_t & entry,
        const struct ggml_tensor * src0) {
    return entry.valid &&
           entry.tensor == src0 &&
           entry.data == src0->data &&
           entry.k == src0->ne[0] &&
           entry.n == src0->ne[1] &&
           entry.nb1 == (size_t) src0->nb[1] &&
           entry.max_rows == g_vpu_max_rows &&
           entry.max_beats == g_vpu_max_beats &&
           entry.max_group_blocks == g_packed_q8_max_blocks;
}

static bool validate_weight_cache_entry(
        fpga_weight_cache_entry_t * entry,
        fpga_stage_totals_t * totals) {
    if (!entry || !entry->valid || !ddr_range_fits(entry->header_off, sizeof(fpga_weight_cache_header_t)) ||
        !ddr_range_fits(entry->base_off, entry->bytes)) {
        return false;
    }

    const fpga_weight_cache_header_t header = ddr_read_weight_cache_header(entry->header_off);
    const uint32_t expected_tensor_hash = fpga_tensor_id_from_ptr(entry->tensor);
    const uint32_t expected_tile_shape =
        ((uint32_t) entry->max_rows & 0xFFFFU) |
        (((uint32_t) entry->max_group_blocks & 0xFFFFU) << 16);
    const bool header_ok =
        header.magic == FPGA_WEIGHT_CACHE_MAGIC &&
        header.format_version == FPGA_WEIGHT_CACHE_FORMAT_VERSION &&
        header.tensor_hash == expected_tensor_hash &&
        header.tile_shape == expected_tile_shape &&
        header.ddr_offset == entry->base_off &&
        header.byte_length == entry->bytes &&
        header.crc32 == entry->payload_crc32 &&
        header.valid == 1U;
    if (!header_ok) {
        LOGE("weight tile cache header/CRC mismatch tensor=%s header_off=0x%08x payload_off=0x%08x bytes=%zu header_ok=%d expected_crc=0x%08x actual_crc=0x%08x; entry invalidated",
             tensor_name_or_unknown(entry->tensor), entry->header_off, entry->base_off, entry->bytes,
             0, entry->payload_crc32, 0U);
        entry->valid = false;
        return false;
    }

    // A full CRC reads the entire tensor through an uncached UIO mapping.  On
    // this board that would cost hundreds of milliseconds per decode token.
    // The header is checked on every lookup; the payload CRC is checked once
    // after construction, or on every lookup only when explicitly requested
    // for a corruption diagnostic.
    if (!entry->crc_validated || g_weight_cache_crc_verify_each_lookup) {
        const long long crc0 = now_us();
        const uint32_t computed_crc = fpga_crc32_ddr(entry->base_off, entry->bytes);
        const long long crc_us = now_us() - crc0;
        if (totals) {
            totals->weight_cache_crc_us += crc_us;
        }
        g_weight_cache_crc_us += crc_us;
        if (computed_crc != entry->payload_crc32) {
            LOGE("weight tile cache CRC mismatch tensor=%s header_off=0x%08x payload_off=0x%08x bytes=%zu expected_crc=0x%08x actual_crc=0x%08x; entry invalidated",
                 tensor_name_or_unknown(entry->tensor), entry->header_off, entry->base_off,
                 entry->bytes, entry->payload_crc32, computed_crc);
            entry->valid = false;
            return false;
        }
        entry->crc_validated = true;
    }
    return true;
}

static fpga_weight_cache_entry_t * find_weight_cache_entry(
        const struct ggml_tensor * src0,
        fpga_stage_totals_t * totals) {
    for (fpga_weight_cache_entry_t & entry : g_weight_cache) {
        if (weight_cache_entry_matches(entry, src0)) {
            if (validate_weight_cache_entry(&entry, totals)) {
                return &entry;
            }
        }
    }
    return nullptr;
}

static fpga_weight_cache_entry_t * build_weight_cache_entry(
        const struct ggml_tensor * src0,
        fpga_stage_totals_t * totals) {
    if (!g_weight_cache_enabled || !ddr_is_mapped()) {
        return nullptr;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t nb = k / VPU_QK8_0;
    std::vector<fpga_weight_tile_cache_t> tiles;
    size_t total_bytes = 0;
    size_t total_scales = 0;

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int remaining_blocks = (int) (nb - ib0);
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats = group_blocks * VPU_BLOCK_BEATS;
            const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
            fpga_weight_tile_cache_t tile = {};
            tile.row0 = row0;
            tile.rows = rows;
            tile.k_block0 = ib0;
            tile.group_blocks = group_blocks;
            tile.group_beats = group_beats;
            tile.bytes = weight_bytes;
            tile.scale_off = total_scales;
            tiles.push_back(tile);
            total_bytes += align_up_size(weight_bytes, WEIGHT_CACHE_ALIGN);
            total_scales += (size_t) rows * (size_t) group_blocks;
            ib0 += group_blocks;
        }
    }

    const uint64_t header_off = align_up_size((size_t) g_weight_cache_next_off, WEIGHT_CACHE_ALIGN);
    const uint64_t payload_off = align_up_size(
        (size_t) header_off + sizeof(fpga_weight_cache_header_t), WEIGHT_CACHE_ALIGN);
    const uint64_t cache_end = payload_off + (uint64_t) total_bytes;
    if (cache_end > (uint64_t) g_weight_cache_end_off || cache_end > UINT32_MAX) {
        if (!g_weight_cache_full_logged) {
            LOGE("weight tile cache full; tensor=%s needs=%zu available=%zu. Continuing with per-run weight packing for uncached tensors.",
                 tensor_name_or_unknown(src0),
                 total_bytes,
                 g_weight_cache_end_off > g_weight_cache_next_off ?
                    (size_t) (g_weight_cache_end_off - g_weight_cache_next_off) : 0U);
            g_weight_cache_full_logged = true;
        }
        return nullptr;
    }

    fpga_weight_cache_entry_t entry = {};
    entry.tensor = src0;
    entry.data = src0->data;
    entry.k = k;
    entry.n = n;
    entry.nb1 = (size_t) src0->nb[1];
    entry.max_rows = g_vpu_max_rows;
    entry.max_beats = g_vpu_max_beats;
    entry.max_group_blocks = g_packed_q8_max_blocks;
    entry.header_off = (uint32_t) header_off;
    entry.base_off = (uint32_t) payload_off;
    entry.bytes = total_bytes;
    entry.valid = false;
    entry.crc_validated = false;
    entry.tiles = std::move(tiles);
    entry.scales.resize(total_scales);

    const fpga_weight_cache_header_t pending_header = {
        FPGA_WEIGHT_CACHE_MAGIC,
        FPGA_WEIGHT_CACHE_FORMAT_VERSION,
        fpga_tensor_id_from_ptr(src0),
        ((uint32_t) entry.max_rows & 0xFFFFU) |
            (((uint32_t) entry.max_group_blocks & 0xFFFFU) << 16),
        entry.base_off,
        (uint32_t) entry.bytes,
        0U,
        0U,
    };
    const long long pack0 = now_us();
    ddr_write_weight_cache_header(entry.header_off, pending_header);
    ddr_zero_range32(entry.base_off, entry.bytes);
    if (!msync_ddr_range(entry.header_off,
                         (size_t) (entry.base_off - entry.header_off) + entry.bytes,
                         false, "weight_cache_prepare")) {
        return nullptr;
    }

    uint32_t next_off = entry.base_off;
    for (fpga_weight_tile_cache_t & tile : entry.tiles) {
        tile.ddr_off = next_off;
        if (!ddr_range_fits(tile.ddr_off, tile.bytes)) {
            LOGE("weight tile cache range overflow tensor=%s off=0x%08x bytes=%zu",
                 tensor_name_or_unknown(src0), tile.ddr_off, tile.bytes);
            return nullptr;
        }

        for (int row = 0; row < tile.rows; ++row) {
            for (int gb = 0; gb < tile.group_blocks; ++gb) {
                const block_q8_0_t * wb = weight_block(src0, tile.row0 + row, tile.k_block0 + gb);
                entry.scales[tile.scale_off + (size_t) row * (size_t) tile.group_blocks + (size_t) gb] =
                    fp16_to_fp32(wb->d);
                for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                    const uint32_t word_index =
                        (uint32_t) row * (uint32_t) tile.group_beats +
                        (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS +
                        (uint32_t) beat;
                    ddr_write_i8x16(tile.ddr_off + word_index * 16U,
                                    wb->qs + beat * VPU_NUM_LANES);
                }
            }
        }

        next_off = (uint32_t) align_up_size((size_t) tile.ddr_off + tile.bytes, WEIGHT_CACHE_ALIGN);
    }

    mmio_fence();
    if (!msync_ddr_range(entry.base_off, entry.bytes, false, "weight_cache_payload")) {
        return nullptr;
    }
    const long long crc0 = now_us();
    entry.payload_crc32 = fpga_crc32_ddr(entry.base_off, entry.bytes);
    const long long crc_us = now_us() - crc0;
    if (totals) {
        totals->weight_cache_crc_us += crc_us;
    }
    g_weight_cache_crc_us += crc_us;
    fpga_weight_cache_header_t committed_header = pending_header;
    committed_header.crc32 = entry.payload_crc32;
    committed_header.valid = 1U;
    ddr_write_weight_cache_header(entry.header_off, committed_header);
    if (!msync_ddr_range(entry.header_off, sizeof(committed_header), false, "weight_cache_commit")) {
        return nullptr;
    }
    entry.valid = true;
    entry.crc_validated = true;
    const long long pack_us = now_us() - pack0;
    if (totals) {
        totals->weight_pack_us += pack_us;
    }
    g_weight_pack_us += pack_us;
    g_weight_cache_next_off = next_off;
    g_weight_cache_bytes += (long long) entry.bytes;
    g_weight_cache_builds++;
    LOGI("weight tile cache build tensor=%s tiles=%zu bytes=%zu scales=%zu header=0x%08x base=0x%08x crc32=0x%08x pack_ms=%.3f crc_ms=%.3f next=0x%08x",
         tensor_name_or_unknown(src0),
         entry.tiles.size(),
         entry.bytes,
         entry.scales.size(),
         entry.header_off,
         entry.base_off,
         entry.payload_crc32,
         (double) pack_us / 1000.0,
         (double) crc_us / 1000.0,
         g_weight_cache_next_off);

    g_weight_cache.push_back(std::move(entry));
    return &g_weight_cache.back();
}

static fpga_weight_cache_entry_t * get_weight_cache_entry(
        const struct ggml_tensor * src0,
        fpga_stage_totals_t * totals) {
    const long long lookup0 = now_us();
    fpga_weight_cache_entry_t * entry = find_weight_cache_entry(src0, totals);
    const long long lookup_us = now_us() - lookup0;
    if (totals) {
        totals->weight_cache_lookup_us += lookup_us;
    }
    g_weight_cache_lookup_us += lookup_us;
    if (entry) {
        return entry;
    }
    return build_weight_cache_entry(src0, totals);
}

static bool fpga_prepare_q8_tile_job(
        fpga_tile_job_t & job,
        const struct ggml_tensor * src0,
        const block_q8_0_t * act_group,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        int64_t col,
        uint32_t weight_tile_index,
        const fpga_weight_cache_entry_t * weight_cache,
        uint32_t tile_id,
        int bank,
        fpga_stage_totals_t * totals) {
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
    const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
    const size_t result_bytes = (size_t) result_words * 16U;
    const size_t scale_bytes = (size_t) result_words * 16U;
    const size_t spu_result_bytes = (size_t) rows * 16U;
    if (!range_fits(ACT_BASE, act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(WEIGHT_BASE, weight_bytes, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(RESULT_BASE, result_bytes, RESULT_BASE, RESULT_END) ||
        !range_fits(SPU_PARAM_BASE, scale_bytes, SPU_PARAM_BASE, SPU_PARAM_END) ||
        !range_fits(SPU_OUT_BASE, spu_result_bytes, SPU_OUT_BASE, SPU_OUT_END) ||
        !ddr_range_fits(ACT_BASE, act_bytes) ||
        !ddr_range_fits(WEIGHT_BASE, weight_bytes) ||
        !ddr_range_fits(RESULT_BASE, result_bytes) ||
        !ddr_range_fits(SPU_PARAM_BASE, scale_bytes) ||
        !ddr_range_fits(SPU_OUT_BASE, spu_result_bytes)) {
        LOGE("unsupported DMA-to-IP tiling case: window overflow act=%zu weight=%zu result=%zu scale=%zu spu_out=%zu ddr_size=0x%zx",
             act_bytes, weight_bytes, result_bytes, scale_bytes, spu_result_bytes, g_ddr_map_size);
        return false;
    }

    std::vector<int32_t> partial_storage;
    std::vector<float> weight_scale_storage;
    partial_storage.swap(job.partial);
    weight_scale_storage.swap(job.weight_scales);
    job = {};
    partial_storage.clear();
    weight_scale_storage.clear();
    job.partial.swap(partial_storage);
    job.weight_scales.swap(weight_scale_storage);

    job.bank = bank & 1;
    job.job_id = fpga_next_job_id();
    job.tile_id = tile_id;
    job.tensor_id = fpga_tensor_id_from_ptr(src0);
    job.row0 = row0;
    job.rows = rows;
    job.k_block0 = k_block0;
    job.group_blocks = group_blocks;
    job.group_beats = group_beats;
    job.col = col;
    job.act_bytes = act_bytes;
    job.weight_bytes = weight_bytes;
    job.scale_bytes = scale_bytes;
    job.spu_result_bytes = spu_result_bytes;
    job.result_bytes = result_bytes;
    job.result_values = result_values;
    job.result_words = result_words;
    job.scale_words = result_words;
    job.weight_src_off = WEIGHT_BASE;
    job.act_group = act_group;
    job.src0 = src0;
    job.weight_cache = weight_cache;

    const long long prep0 = now_us();
    vpu_write_tile_descriptor(job, FPGA_SLOT_CPU_PACKING, FPGA_SLOT_FREE, 0x00000001U);

    if (weight_cache && weight_tile_index < weight_cache->tiles.size()) {
        const fpga_weight_tile_cache_t & tile = weight_cache->tiles[weight_tile_index];
        if (tile.row0 == row0 &&
            tile.rows == rows &&
            tile.k_block0 == k_block0 &&
            tile.group_blocks == group_blocks &&
            tile.group_beats == group_beats &&
            tile.bytes == weight_bytes) {
            job.weight_src_off = tile.ddr_off;
            job.weight_cache_hit = true;
        }
    }

    if (!job.weight_cache_hit) {
        for (int row = 0; row < rows; ++row) {
            for (int gb = 0; gb < group_blocks; ++gb) {
                const block_q8_0_t * wb = weight_block(src0, row0 + row, k_block0 + gb);
                for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                    const uint32_t word_index = (uint32_t) row * (uint32_t) group_beats +
                                                (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS +
                                                (uint32_t) beat;
                    write_i8x16_to_ddr(WEIGHT_BASE + word_index * 16U, wb->qs + beat * VPU_NUM_LANES);
                }
            }
        }
    }

    ddr_zero_range32(SPU_PARAM_BASE, job.scale_bytes);
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const uint32_t linear = (uint32_t) row * (uint32_t) group_blocks + (uint32_t) gb;
            const uint32_t word = linear / (uint32_t) VPU_RESULT_PACK_LANES;
            const uint32_t lane = linear % (uint32_t) VPU_RESULT_PACK_LANES;
            const block_q8_0_t * wb = weight_block(src0, row0 + row, k_block0 + gb);
            const uint32_t packed_scale =
                (uint32_t) act_group[gb].d |
                ((uint32_t) wb->d << 16);
            ddr_write_u32(SPU_PARAM_BASE + word * 16U + lane * 4U, packed_scale);
        }
    }

    for (int gb = 0; gb < group_blocks; ++gb) {
        const block_q8_0_t & act = act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act.qs + beat * VPU_NUM_LANES);
        }
    }
    mmio_fence();
    vpu_write_tile_descriptor(job, FPGA_SLOT_READY, FPGA_SLOT_FREE, 0x00000001U);

    if (totals) {
        totals->prep_us += now_us() - prep0;
        if (job.weight_cache_hit) {
            totals->weight_cache_hits++;
        } else {
            totals->weight_cache_misses++;
        }
    }
    return true;
}

static bool fpga_submit_q8_tile_job(
        fpga_tile_job_t & job,
        fpga_stage_totals_t * totals,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m,
        int attempt) {
    vpu_select_banks(job.bank, job.bank);
    vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
    configure_vpu(job.rows, job.group_beats, VPU_MODE_PACKED_Q8);
    vpu_write_tile_descriptor(job, FPGA_SLOT_DMA_FILLING, FPGA_SLOT_FREE, 0x00000101U);

    long long result_clear0 = 0;
    long long result_clear1 = 0;
    if (g_clear_result_before_run && !g_spu_q8_scale_stream_supported) {
        result_clear0 = now_us();
        ddr_zero_range32(RESULT_BASE, job.result_bytes);
        vpu_select_banks(job.bank, job.bank);
        if (!fpga_dma_write_to_ip(RESULT_BASE, job.result_bytes, "RESULT_CLEAR")) {
            return false;
        }
        result_clear1 = now_us();
        job.result_clear_us = result_clear1 - result_clear0;
    }

    const long long dma_act0 = now_us();
    vpu_select_banks(job.bank, job.bank);
    if (!fpga_dma_write_to_ip(ACT_BASE, job.act_bytes, "ACT")) {
        return false;
    }
    const long long dma_act1 = now_us();

    const long long dma_weight0 = now_us();
    vpu_select_banks(job.bank, job.bank);
    if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) job.weight_src_off,
                       LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                       job.weight_bytes,
                       "WEIGHT")) {
        return false;
    }
    const long long dma_weight1 = now_us();

    const long long dma_scale0 = now_us();
    if (!fpga_dma_write_to_ip(SPU_PARAM_BASE, job.scale_bytes, "SPU_SCALE")) {
        return false;
    }
    const long long dma_scale1 = now_us();

    job.dma_act_us = dma_act1 - dma_act0;
    job.dma_weight_us = dma_weight1 - dma_weight0;
    job.dma_scale_us = dma_scale1 - dma_scale0;
    job.spu_stream_count_before = vpu_rd32(REG_SPU_STREAM_COUNT);
    job.spu_stream_done_before = vpu_rd32(REG_SPU_STREAM_DONE);
    job.spu_stream_out_before = vpu_rd32(REG_SPU_STREAM_OUT);
    job.spu_stream_drop_before = vpu_rd32(REG_SPU_STREAM_DROP);
    job.spu_stream_error_before = vpu_rd32(REG_SPU_STREAM_ERROR);
    if (totals) {
        totals->dma_act_us += job.dma_act_us;
        totals->dma_weight_us += job.dma_weight_us;
        totals->dma_scale_us += job.dma_scale_us;
        totals->dma_result_us += g_clear_result_before_run ? job.result_clear_us : 0;
        totals->activation_bytes += job.act_bytes;
        totals->weight_bytes += job.weight_bytes;
        totals->scale_bytes += job.scale_bytes;
        totals->result_bytes += job.spu_result_bytes;
        totals->vpu_runs++;
    }

    vpu_write_tile_descriptor(job, FPGA_SLOT_COMPUTING, FPGA_SLOT_FREE, 0x00000101U);
    mmio_fence();
    job.ip_start_us = now_us();
    vpu_wr32(REG_CTRL, CTRL_START);
    mmio_fence();

    if (g_ip_timing_enabled && should_log_detail_run(job.tile_id)) {
        LOGIP("submit tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u job=%u bank=%d attempt=%d rows=%d col_beats=%d mode=0x%x result_clear_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f weight_cache=%d",
              tensor_name ? tensor_name : "?",
              layer_id,
              (long long) k,
              (long long) n,
              (long long) m,
              job.tile_id,
              job.job_id,
              job.bank,
              attempt,
              job.rows,
              job.group_beats,
              VPU_MODE_PACKED_Q8,
              g_clear_result_before_run ? (double) job.result_clear_us / 1000.0 : 0.0,
              (double) job.dma_act_us / 1000.0,
              (double) job.dma_weight_us / 1000.0,
              (double) job.dma_scale_us / 1000.0,
              job.weight_cache_hit ? 1 : 0);
    }
    return true;
}

static bool fpga_wait_and_drain_q8_tile_job(
        fpga_tile_job_t & job,
        fpga_stage_totals_t * totals,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m,
        int attempt) {
    uint32_t vpu_status = 0;
    if (!wait_vpu_done(&vpu_status)) {
        LOGE("VPU failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld tile=%u job=%u bank=%d attempt=%d status=0x%08x progress=0x%08x",
             tensor_name ? tensor_name : "?",
             layer_id,
             (long long) k,
             (long long) n,
             (long long) m,
             job.tile_id,
             job.job_id,
             job.bank,
             attempt,
             vpu_status,
             vpu_rd32(REG_PROGRESS));
        return false;
    }
    const long long ip1 = now_us();
    job.vpu_status = vpu_status;
    job.ip_compute_us = ip1 - job.ip_start_us;
    if (!vpu_verify_done_job(job, vpu_status)) {
        return false;
    }
    if (!wait_spu_stream_outputs(job)) {
        return false;
    }

    vpu_write_tile_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_DMA_DRAINING, 0x00000101U);
    vpu_select_banks(job.bank, job.bank);
    const long long dma_result0 = now_us();
    if (!fpga_dma_read_from_ip(SPU_OUT_BASE, job.spu_result_bytes, "SPU_OUT")) {
        return false;
    }
    const long long dma_result1 = now_us();
    job.dma_result_us = dma_result1 - dma_result0;
    vpu_write_tile_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_HOST_CONSUMING, 0x00000101U);

    if (totals) {
        totals->dma_result_us += job.dma_result_us;
        totals->ip_compute_us += job.ip_compute_us;
    }

    if (g_ip_timing_enabled && should_log_detail_run(job.tile_id)) {
        LOGIP("complete tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u job=%u bank=%d attempt=%d rows=%d col_beats=%d ip_ms=%.3f spu_out_dma_ms=%.3f status=0x%08x progress=0x%08x spu_out=%u",
              tensor_name ? tensor_name : "?",
              layer_id,
              (long long) k,
              (long long) n,
              (long long) m,
              job.tile_id,
              job.job_id,
              job.bank,
              attempt,
              job.rows,
              job.group_beats,
              (double) job.ip_compute_us / 1000.0,
              (double) job.dma_result_us / 1000.0,
              vpu_status,
              vpu_rd32(REG_PROGRESS),
              vpu_rd32(REG_SPU_STREAM_OUT));
    }
    return true;
}

static void fpga_unpack_q8_tile_job(fpga_tile_job_t & job, fpga_stage_totals_t * totals) {
    const long long result_unpack0 = now_us();
    job.partial.resize((size_t) job.result_values);
    int32_t lanes[VPU_RESULT_PACK_LANES] = {};
    for (uint32_t word = 0; word < job.result_words; ++word) {
        read_result_i32x4_from_ddr(word, lanes);
        for (int lane = 0; lane < VPU_RESULT_PACK_LANES; ++lane) {
            const uint32_t idx = word * (uint32_t) VPU_RESULT_PACK_LANES + (uint32_t) lane;
            if (idx < job.result_values) {
                job.partial[(size_t) idx] = lanes[lane];
            }
        }
    }
    job.host_result_us = now_us() - result_unpack0;
    if (totals) {
        totals->host_result_us += job.host_result_us;
    }
    vpu_write_tile_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_FREE, 0x00000000U);
}

static void fpga_accumulate_q8_tile_job(
        const fpga_tile_job_t & job,
        const std::vector<float> & act_scales,
        int64_t nb,
        std::vector<float> & accum,
        fpga_stage_totals_t * totals) {
    const long long accum0 = now_us();
    float * accum_col = &accum[(size_t) (job.col * job.rows)];
    for (int row = 0; row < job.rows; ++row) {
        for (int gb = 0; gb < job.group_blocks; ++gb) {
            const int64_t ib = job.k_block0 + gb;
            const int32_t raw = job.partial[(size_t) row * (size_t) job.group_blocks + (size_t) gb];
            accum_col[(size_t) row] +=
                (float) raw *
                act_scales[(size_t) (job.col * nb + ib)] *
                job.weight_scales[(size_t) row * (size_t) job.group_blocks + (size_t) gb];
        }
    }
    if (totals) {
        totals->host_accum_us += now_us() - accum0;
    }
}

static bool fpga_accumulate_pl_scaled_q8_tile_job(
        const fpga_tile_job_t & job,
        std::vector<float> & accum,
        fpga_stage_totals_t * totals) {
    const long long result0 = now_us();
    float * accum_col = &accum[(size_t) (job.col * job.rows)];
    for (int row = 0; row < job.rows; ++row) {
        uint16_t row_id = 0xffffU;
        const int64_t q16 = ddr_read_spu_q16_row(SPU_OUT_BASE + (uint32_t) row * 16U, &row_id);
        if (row_id != (uint16_t) row) {
            LOGE("SPU_OUT row mismatch job=%u tile=%u bank=%d row=%d got_row=%u q16=%lld",
                 job.job_id,
                 job.tile_id,
                 job.bank,
                 row,
                 (unsigned) row_id,
                 (long long) q16);
            return false;
        }
        accum_col[(size_t) row] += (float) q16 * (1.0f / 65536.0f);
    }
    const long long result1 = now_us();
    if (totals) {
        totals->host_result_us += result1 - result0;
    }
    vpu_write_tile_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_FREE, 0x00000000U);
    return true;
}

static bool fpga_dma_run_q8_group(
        const struct ggml_tensor * src0,
        const block_q8_0_t * act_group,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        uint32_t weight_tile_index,
        const fpga_weight_cache_entry_t * weight_cache,
        std::vector<int32_t> & partial,
        std::vector<float> & weight_scales,
        fpga_stage_totals_t * totals,
        uint32_t tile_id,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m,
        bool contract_check_active) {
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
    const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
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
    uint32_t weight_src_off = WEIGHT_BASE;
    bool weight_cache_hit = false;
    weight_scales.resize((size_t) rows * (size_t) group_blocks);

    if (weight_cache && weight_tile_index < weight_cache->tiles.size()) {
        const fpga_weight_tile_cache_t & tile = weight_cache->tiles[weight_tile_index];
        if (tile.row0 == row0 &&
            tile.rows == rows &&
            tile.k_block0 == k_block0 &&
            tile.group_blocks == group_blocks &&
            tile.group_beats == group_beats &&
            tile.bytes == weight_bytes) {
            weight_src_off = tile.ddr_off;
            weight_cache_hit = true;
            std::copy(weight_cache->scales.begin() + (ptrdiff_t) tile.scale_off,
                      weight_cache->scales.begin() + (ptrdiff_t) tile.scale_off +
                        (ptrdiff_t) ((size_t) rows * (size_t) group_blocks),
                      weight_scales.begin());
        }
    }

    if (!weight_cache_hit) {
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
    }

    for (int gb = 0; gb < group_blocks; ++gb) {
        const block_q8_0_t & act = act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act.qs + beat * VPU_NUM_LANES);
        }
    }
    mmio_fence();
    if (totals) {
        totals->prep_us += now_us() - prep0;
        if (weight_cache_hit) {
            totals->weight_cache_hits++;
        } else {
            totals->weight_cache_misses++;
        }
    }

    const int attempt_count = contract_check_active ? (1 + g_contract_raw_retry_limit) : 1;
    for (int attempt = 0; attempt < attempt_count; ++attempt) {
        vpu_select_banks(0, 0);
        vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
        configure_vpu(rows, group_beats, VPU_MODE_PACKED_Q8);

        long long result_clear0 = 0;
        long long result_clear1 = 0;
        if (g_clear_result_before_run) {
            result_clear0 = now_us();
            ddr_zero_range32(RESULT_BASE, result_bytes);
            if (!fpga_dma_write_to_ip(RESULT_BASE, result_bytes, "RESULT_CLEAR")) {
                return false;
            }
            result_clear1 = now_us();
        }

        const long long dma_act0 = now_us();
        if (!fpga_dma_write_to_ip(ACT_BASE, act_bytes, "ACT")) {
            return false;
        }
        const long long dma_act1 = now_us();

        const long long dma_weight0 = now_us();
        if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) weight_src_off,
                           LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                           weight_bytes,
                           "WEIGHT")) {
            return false;
        }
        const long long dma_weight1 = now_us();

        mmio_fence();
        const long long ip0 = now_us();
        vpu_wr32(REG_CTRL, CTRL_START);
        mmio_fence();

        uint32_t vpu_status = 0;
        if (!wait_vpu_done(&vpu_status)) {
            LOGE("VPU failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld tile=%u attempt=%d status=0x%08x progress=0x%08x",
                 tensor_name ? tensor_name : "?",
                 layer_id,
                 (long long) k,
                 (long long) n,
                 (long long) m,
                 tile_id,
                 attempt,
                 vpu_status,
                 vpu_rd32(REG_PROGRESS));
            return false;
        }
        const long long ip1 = now_us();

        const long long dma_result0 = now_us();
        vpu_select_banks(0, 0);
        if (!fpga_dma_read_from_ip(RESULT_BASE, result_bytes, "RESULT")) {
            return false;
        }
        const long long dma_result1 = now_us();

        if (totals) {
            totals->dma_act_us += dma_act1 - dma_act0;
            totals->dma_weight_us += dma_weight1 - dma_weight0;
            totals->dma_result_us += (dma_result1 - dma_result0) +
                                     (g_clear_result_before_run ? (result_clear1 - result_clear0) : 0);
            totals->ip_compute_us += ip1 - ip0;
            totals->activation_bytes += act_bytes;
            totals->weight_bytes += weight_bytes;
            totals->result_bytes += result_bytes;
            totals->vpu_runs++;
        }

        if (g_ip_timing_enabled && should_log_detail_run(tile_id)) {
            LOGIP("run tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u attempt=%d rows=%d col_beats=%d mode=0x%x result_clear_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f ip_ms=%.3f result_dma_ms=%.3f status=0x%08x progress=0x%08x weight_cache=%d",
                  tensor_name ? tensor_name : "?",
                  layer_id,
                  (long long) k,
                  (long long) n,
                  (long long) m,
                  tile_id,
                  attempt,
                  rows,
                  group_beats,
                  VPU_MODE_PACKED_Q8,
                  g_clear_result_before_run ? (double) (result_clear1 - result_clear0) / 1000.0 : 0.0,
                  (double) (dma_act1 - dma_act0) / 1000.0,
                  (double) (dma_weight1 - dma_weight0) / 1000.0,
                  (double) (ip1 - ip0) / 1000.0,
                  (double) (dma_result1 - dma_result0) / 1000.0,
                  vpu_status,
                  vpu_rd32(REG_PROGRESS),
                  weight_cache_hit ? 1 : 0);
        }

        const long long result_unpack0 = now_us();
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
        if (totals) {
            totals->host_result_us += now_us() - result_unpack0;
        }

        if (!contract_check_active) {
            return true;
        }

        const bool final_attempt = (attempt + 1) >= attempt_count;
        const bool repair_this_attempt = final_attempt && g_contract_raw_repair_enabled;
        const long long raw_mismatches = fpga_contract_count_raw_mismatches(
            src0, act_group, row0, rows, k_block0, group_blocks,
            partial, tensor_name, layer_id, tile_id, attempt, true, repair_this_attempt);
        if (raw_mismatches == 0) {
            if (attempt > 0) {
                LOGI("CONTRACT_RAW_RETRY_PASS tensor=%s layer=%d tile=%u attempts=%d",
                     tensor_name ? tensor_name : "?",
                     layer_id,
                     tile_id,
                     attempt + 1);
            }
            return true;
        }

        if (!final_attempt) {
            LOGE("CONTRACT_RAW_RETRY tensor=%s layer=%d tile=%u attempt=%d mismatches=%lld next_attempt=%d",
                 tensor_name ? tensor_name : "?",
                 layer_id,
                 tile_id,
                 attempt,
                 raw_mismatches,
                 attempt + 1);
            continue;
        }

        g_contract_raw_mismatches += raw_mismatches;
        if (repair_this_attempt) {
            g_contract_raw_repairs += raw_mismatches;
        }
        LOGE("CONTRACT_RAW_SUMMARY tensor=%s layer=%d tile=%u attempt=%d mismatches=%lld action=%s",
             tensor_name ? tensor_name : "?",
             layer_id,
             tile_id,
             attempt,
             raw_mismatches,
             repair_this_attempt ? "repair_continue" :
             (g_contract_check_abort ? "abort" : "log_only"));
        if (repair_this_attempt) {
            return true;
        }
        return !g_contract_check_abort;
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

static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst,
        const std::vector<block_q8_0_t> & act_blocks_all,
        const std::vector<float> & act_scales,
        const fpga_weight_cache_entry_t * weight_cache,
        fpga_stage_totals_t * totals,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m,
        int64_t nb,
        std::vector<float> & accum) {
    (void) act_scales;
    (void) nb;
    uint32_t tile_id = 0;
    uint32_t weight_tile_index = 0;
    fpga_tile_job_t slots[2] = {};

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        accum.assign((size_t) (m * rows), 0.0f);

        fpga_tile_job_t * running = nullptr;
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int remaining_blocks = (int) (nb - ib0);
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats = group_blocks * VPU_BLOCK_BEATS;

            for (int64_t col = 0; col < m; ++col) {
                const int bank = (int) (tile_id & 1U);
                fpga_tile_job_t & prepared = slots[bank];
                const block_q8_0_t * act_group =
                    &act_blocks_all[(size_t) (col * nb + ib0)];

                if (should_log_detail_run(tile_id)) {
                    LOGSTAGE("tile tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d group_beats=%d tile_id=%u bank=%d pipeline=pingpong partial_accum=1 transfer=zdma_ddr_to_ip",
                             tensor_name ? tensor_name : "?",
                             layer_id,
                             (long long) row0,
                             rows,
                             (long long) ib0,
                             group_blocks,
                             group_beats,
                             tile_id,
                             bank);
                }

                if (!fpga_prepare_q8_tile_job(prepared, src0, act_group, row0, rows, ib0, group_blocks,
                                             col, weight_tile_index, weight_cache, tile_id, bank, totals)) {
                    return false;
                }

                if (running) {
                    if (!fpga_wait_and_drain_q8_tile_job(*running, totals, tensor_name, layer_id,
                                                         k, n, m, 0)) {
                        return false;
                    }
                    if (!fpga_submit_q8_tile_job(prepared, totals, tensor_name, layer_id,
                                                 k, n, m, 0)) {
                        return false;
                    }
                    if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals)) {
                        return false;
                    }
                    running = &prepared;
                } else {
                    if (!fpga_submit_q8_tile_job(prepared, totals, tensor_name, layer_id,
                                                 k, n, m, 0)) {
                        return false;
                    }
                    running = &prepared;
                }

                tile_id++;
            }

            ib0 += group_blocks;
            weight_tile_index++;
        }

        if (running) {
            if (!fpga_wait_and_drain_q8_tile_job(*running, totals, tensor_name, layer_id,
                                                 k, n, m, 0)) {
                return false;
            }
            if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals)) {
                return false;
            }
        }

        const long long store0 = now_us();
        for (int64_t col = 0; col < m; ++col) {
            const float * accum_col = &accum[(size_t) (col * rows)];
            for (int row = 0; row < rows; ++row) {
                store_dst_value(dst, row0 + row, col, accum_col[(size_t) row]);
            }
        }
        if (totals) {
            totals->host_accum_us += now_us() - store0;
        }
    }
    return true;
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

    const bool contract_check_active =
        (g_contract_check_limit > 0) &&
        (g_contract_checks_done < (long long) g_contract_check_limit);
    const bool use_pl_scale_path =
        g_pingpong_scheduler_enabled &&
        g_spu_q8_scale_stream_supported &&
        !contract_check_active;

    const long long quant0 = now_us();
    ensure_quantized_activation_matrix(src1, m, k, act_blocks_all, act_scales,
                                       !use_pl_scale_path);
    if (totals) {
        totals->prep_us += now_us() - quant0;
    }

    const long long weight_cache0 = now_us();
    fpga_weight_cache_entry_t * weight_cache = get_weight_cache_entry(src0, totals);
    if (totals) {
        totals->prep_us += now_us() - weight_cache0;
    }

    if (use_pl_scale_path) {
        return fpga_hw_q8_0_matmul_dma_to_ip_pipelined(
            src0, dst, act_blocks_all, act_scales, weight_cache, totals,
            tensor_name, layer_id, k, n, m, nb, accum);
    }

    uint32_t tile_id = 0;
    uint32_t weight_tile_index = 0;
    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        accum.assign((size_t) (m * rows), 0.0f);

        for (int64_t ib0 = 0; ib0 < nb;) {
            const int remaining_blocks = (int) (nb - ib0);
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats = group_blocks * VPU_BLOCK_BEATS;

            if (should_log_detail_run(tile_id)) {
                LOGSTAGE("tile tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d group_beats=%d tile_id=%u partial_accum=1 transfer=zdma_ddr_to_ip",
                         tensor_name ? tensor_name : "?",
                         layer_id,
                         (long long) row0,
                         rows,
                         (long long) ib0,
                         group_blocks,
                         group_beats,
                         tile_id);
            }

            for (int64_t col = 0; col < m; ++col) {
                const block_q8_0_t * act_group =
                    &act_blocks_all[(size_t) (col * nb + ib0)];
                if (!fpga_dma_run_q8_group(src0, act_group, row0, rows, ib0, group_blocks,
                                           weight_tile_index, weight_cache,
                                           partial, weight_scales, totals, tile_id++,
                                           tensor_name, layer_id, k, n, m,
                                           contract_check_active)) {
                    return false;
                }

                const long long accum0 = now_us();
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
                if (totals) {
                    totals->host_accum_us += now_us() - accum0;
                }
            }
            ib0 += group_blocks;
            weight_tile_index++;
        }

        const long long store0 = now_us();
        for (int64_t col = 0; col < m; ++col) {
            const float * accum_col = &accum[(size_t) (col * rows)];
            for (int row = 0; row < rows; ++row) {
                store_dst_value(dst, row0 + row, col, accum_col[(size_t) row]);
            }
        }
        if (totals) {
            totals->host_accum_us += now_us() - store0;
        }
    }
    if (contract_check_active) {
        g_contract_checks_done++;
        if (!fpga_contract_check_dst_values(src0, dst, act_blocks_all, act_scales,
                                            tensor_name, layer_id)) {
            return false;
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

    g_stage_timing_enabled = !env_flag_disabled("FPGA_STAGE_TIMING");
    g_dma_timing_enabled = env_flag_enabled("FPGA_DMA_TIMING");
    g_ip_timing_enabled = env_flag_enabled("FPGA_IP_TIMING");
    g_init_verbose = env_flag_enabled("FPGA_INIT_VERBOSE");
    g_status_stderr = env_flag_enabled("FPGA_STATUS_STDERR");
    g_trace_data_enabled = env_flag_enabled("FPGA_TRACE_DATA");
    g_weight_cache_enabled = env_flag_enabled("FPGA_WEIGHT_CACHE");
    g_activation_cache_enabled = env_flag_enabled("FPGA_ACTIVATION_CACHE");
    g_allow_devmem_fallback = env_flag_enabled("FPGA_ALLOW_DEVMEM");
    g_allow_vpu_devmem_compat =
        !env_flag_enabled("FPGA_VPU_UIO_REQUIRED") &&
        !env_flag_disabled("FPGA_VPU_DEVMEM_COMPAT");
    g_strict_coherency = env_flag_enabled("FPGA_STRICT_COHERENCY");
    g_coherency_platform_whitelisted = env_flag_enabled("FPGA_COHERENCY_PLATFORM_VERIFIED");
    g_run_coherency_stress = env_flag_enabled("FPGA_COHERENCY_STRESS") || g_strict_coherency;
    g_contract_check_abort = env_flag_enabled("FPGA_CONTRACT_ABORT");
    g_clear_result_before_run = env_flag_enabled("FPGA_CLEAR_RESULT");
    // A repair changes the value consumed by the model, so it is a diagnostic
    // opt-in only.  Correctness runs must report mismatches, not hide them.
    g_contract_raw_repair_enabled = env_flag_enabled("FPGA_CONTRACT_RAW_REPAIR");
    g_weight_cache_crc_verify_each_lookup = env_flag_enabled("FPGA_WEIGHT_CACHE_CRC_EACH_LOOKUP");
    g_vocab_projection_cpu_bypass =
        env_flag_enabled("FPGA_VOCAB_PROJECTION_CPU") &&
        !env_flag_enabled("FPGA_ACCELERATE_VOCAB");
    g_log_flush_every = env_int_value("FPGA_LOG_FLUSH_EVERY", 256, 1, 1000000);
    g_profile_every = env_int_value("FPGA_PROFILE_EVERY", FPGA_DEFAULT_PROFILE_EVERY, 0, 1000000);
    g_ip_status_every = env_int_value("FPGA_IP_STATUS_EVERY", FPGA_DEFAULT_STATUS_EVERY, 0, 1000000);
    g_detail_every = env_int_value("FPGA_DETAIL_EVERY", FPGA_DEFAULT_DETAIL_EVERY, 0, 1000000);
    g_contract_check_limit = env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000);
    g_contract_raw_retry_limit = env_int_value("FPGA_CONTRACT_RAW_RETRY", 1, 0, 8);
    g_runtime_max_rows = env_int_value("FPGA_RUNTIME_MAX_ROWS", VPU_SAFE_RUNTIME_ROWS, 1, VPU_DEFAULT_ROWS);
    g_vocab_projection_min_n = env_int64_value("FPGA_VOCAB_PROJECTION_MIN_N", 65536, 1024, LLONG_MAX);
    if (env_flag_enabled("FPGA_TILE_TIMING") && g_detail_every == 0) {
        g_detail_every = 1;
    }
    g_dma_timeout_us = env_int64_value("FPGA_DMA_TIMEOUT_US", FPGA_DEFAULT_DMA_TIMEOUT_US, 1000, LLONG_MAX);
    g_ip_timeout_us = env_int64_value("FPGA_IP_TIMEOUT_US", FPGA_DEFAULT_IP_TIMEOUT_US, 1000, LLONG_MAX);
    g_large_matrix_min_macs = env_int64_value(
        "FPGA_LARGE_MATRIX_MIN_MACS", FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS, 0, LLONG_MAX);
    const double fpga_clock_hz = env_double_value("FPGA_CLOCK_HZ", 0.0, 0.0, 1.0e12);
    if (fpga_clock_hz > 0.0) {
        g_fpga_clock_mhz = fpga_clock_hz / 1.0e6;
    } else {
        g_fpga_clock_mhz = env_double_value("FPGA_CLOCK_MHZ", 0.0, 0.0, 1.0e12);
        if (g_fpga_clock_mhz > 100000.0) {
            g_fpga_clock_mhz /= 1.0e6;
        }
    }
    g_contract_atol = env_double_value("FPGA_CONTRACT_ATOL", 1.0e-3, 0.0, 1.0e9);
    g_contract_rtol = env_double_value("FPGA_CONTRACT_RTOL", 1.0e-4, 0.0, 1.0e9);
    g_abort_on_cpu_fallback = !env_flag_disabled("FPGA_ABORT_ON_CPU_FALLBACK");

    if (!configure_ddr_mapping_policy()) {
        fpga_fatal("DDR cache/mapping policy is invalid; refusing FPGA initialization");
    }
    if (!map_registers_dma_ddr()) {
        fpga_fatal("ZDMA DDR-to-IP FPGA init failed; refusing CPU fallback");
    }
    configure_weight_cache();
    if (!fpga_dma_init()) {
        fpga_fatal("ZDMA init failed; refusing CPU fallback");
    }
    if (g_run_coherency_stress && !fpga_ddr_coherency_stress_test()) {
        fpga_fatal("DDR coherency stress test failed; refusing FPGA execution");
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
    const uint32_t spu_caps = vpu_rd32(REG_SPU_CAPS);
    g_stream_protocol_version = vpu_rd32(REG_STREAM_PROTOCOL_VERSION);
    g_bitstream_id = vpu_rd32(REG_BITSTREAM_ID);
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
    if (caps_valid && ((caps & VPU_CAP_PACKED_Q8) != 0U)) {
        const bool compact_weight_layout = (caps & VPU_CAP_COMPACT_WEIGHT_LAYOUT) != 0U;
        const int cap_blocks = (int) ((caps >> 8) & 0xFFU);
        const int cap_result_words = (int) ((caps >> 16) & 0xFFFFU);
        if (compact_weight_layout && cap_blocks > 0 && cap_result_words > 0) {
            g_packed_q8_supported = 1;
            g_packed_q8_max_blocks = std::min(cap_blocks, g_vpu_max_beats / VPU_BLOCK_BEATS);
            g_packed_q8_result_words = cap_result_words;
        } else if (!compact_weight_layout) {
            LOGE("REG_CAPS=0x%08x exposes packed_q8 without compact weight layout; host requires compact active-stride layout", caps);
        }
    }
    if (!caps_valid) {
        fpga_fatal("REG_CAPS=0x%08x is invalid; refusing legacy capability assumptions", caps);
    }
    if (env_flag_enabled("FPGA_FORCE_PACKED_Q8")) {
        fpga_fatal("FPGA_FORCE_PACKED_Q8 is not permitted in the production host; the bitstream must advertise packed-Q8 capability");
    }
    const bool bitstream_id_compatible = g_bitstream_id == FPGA_EXPECTED_BITSTREAM_ID;
    if (!bitstream_id_compatible) {
        LOGI("bitstream identity is legacy/unknown got=0x%08x expected=0x%08x; raw CPU-scale compatibility path remains enabled, but PL-scale and pipeline opt-ins are prohibited",
             g_bitstream_id, FPGA_EXPECTED_BITSTREAM_ID);
    }
    if (g_runtime_max_rows > 0 && g_runtime_max_rows < g_vpu_max_rows) {
        g_vpu_max_rows = g_runtime_max_rows;
    }
    g_packed_q8_result_words =
        (g_vpu_max_rows * g_packed_q8_max_blocks + VPU_RESULT_PACK_LANES - 1) /
        VPU_RESULT_PACK_LANES;
    g_vpu_pingpong_supported = caps_valid && ((caps & VPU_CAP_PINGPONG_BANKS) != 0U);
    g_vpu_descriptor_supported = caps_valid && ((caps & VPU_CAP_JOB_DESCRIPTOR) != 0U);
    const bool spu_caps_valid = spu_caps != 0U && spu_caps != 0xFFFFFFFFU;
    g_spu_silu_supported = spu_caps_valid && ((spu_caps & SPU_CAP_SILU_MUL) != 0U);
    g_spu_rmsnorm_supported = spu_caps_valid && ((spu_caps & SPU_CAP_RMSNORM) != 0U);
    g_spu_rope_supported = spu_caps_valid && ((spu_caps & SPU_CAP_ROPE) != 0U);
    g_spu_softmax_supported = spu_caps_valid && ((spu_caps & SPU_CAP_SOFTMAX) != 0U);
    g_spu_q8_scale_stream_supported =
        caps_valid &&
        spu_caps_valid &&
        bitstream_id_compatible &&
        (g_stream_protocol_version == FPGA_REQUIRED_STREAM_PROTOCOL_VERSION) &&
        ((caps & VPU_CAP_SPU_RAW_STREAM) != 0U) &&
        ((caps & VPU_CAP_SPU_Q8_SCALE_STREAM) != 0U) &&
        ((spu_caps & SPU_CAP_VPU_RAW_STREAM) != 0U) &&
        ((spu_caps & SPU_CAP_VPU_Q8_SCALE_STREAM) != 0U) &&
        env_flag_enabled("FPGA_PL_SCALE_ENABLE") &&
        !env_flag_enabled("FPGA_PL_SCALE_DISABLE");
    g_pingpong_scheduler_enabled =
        g_vpu_pingpong_supported &&
        g_vpu_descriptor_supported &&
        bitstream_id_compatible &&
        env_flag_enabled("FPGA_PIPELINE_ENABLE") &&
        !env_flag_enabled("FPGA_PIPELINE_DISABLE");
    if (env_flag_enabled("FPGA_PL_SCALE_ENABLE") && !g_spu_q8_scale_stream_supported) {
        fpga_fatal("FPGA_PL_SCALE_ENABLE requested but stream capability/protocol is not compatible: caps=0x%08x spu_caps=0x%08x protocol=0x%08x required=%u",
                   caps, spu_caps, g_stream_protocol_version, FPGA_REQUIRED_STREAM_PROTOCOL_VERSION);
    }
    if (env_flag_enabled("FPGA_PIPELINE_ENABLE") && !g_pingpong_scheduler_enabled) {
        fpga_fatal("FPGA_PIPELINE_ENABLE requested but ping-pong descriptor capability is unavailable caps=0x%08x", caps);
    }
    vpu_select_banks(0, 0);

    const char * mapping_policy =
        g_allow_devmem_fallback ? "diagnostic_all_resources" :
        (g_allow_vpu_devmem_compat ? "uio_dma_ddr+vpu_devmem_compat" : "uio_required");

    LOGI("ready version=%s path=%s rows=%d host_row_limit=%d col_beats=%d cols=%d packed_q8=%d max_group_blocks=%d result_words=%d pingpong_cap=%d descriptor_cap=%d scheduler=%d scheduler_policy=opt_in pl_scale=%d pl_scale_policy=opt_in stream_protocol=0x%08x bitstream_id=0x%08x spu_silu=%d spu_rms=%d spu_rope=%d spu_softmax=%d weight_cache=%d activation_cache=%d vocab_cpu_bypass=%d vocab_min_n=%lld contract_check=%d contract_abort=%d raw_retry=%d raw_repair=%d result_clear=%d strict_coherency=%d clock_mhz=%.3f profile_log=1 dma_detail=%d ip_detail=%d detail_every=%d flush_every=%d",
         FPGA_HOST_TRACE_VERSION,
         path ? path : "dma(default)",
         g_vpu_max_rows,
         g_runtime_max_rows,
         g_vpu_max_beats, g_vpu_max_cols,
         g_packed_q8_supported, g_packed_q8_max_blocks, g_packed_q8_result_words,
         g_vpu_pingpong_supported ? 1 : 0,
         g_vpu_descriptor_supported ? 1 : 0,
         g_pingpong_scheduler_enabled ? 1 : 0,
         g_spu_q8_scale_stream_supported ? 1 : 0,
         g_stream_protocol_version,
         g_bitstream_id,
         g_spu_silu_supported ? 1 : 0,
         g_spu_rmsnorm_supported ? 1 : 0,
         g_spu_rope_supported ? 1 : 0,
         g_spu_softmax_supported ? 1 : 0,
         g_weight_cache_enabled ? 1 : 0,
         g_activation_cache_enabled ? 1 : 0,
         g_vocab_projection_cpu_bypass ? 1 : 0,
         (long long) g_vocab_projection_min_n,
         g_contract_check_limit,
         g_contract_check_abort ? 1 : 0,
         g_contract_raw_retry_limit,
         g_contract_raw_repair_enabled ? 1 : 0,
         g_clear_result_before_run ? 1 : 0,
         g_strict_coherency ? 1 : 0,
         g_fpga_clock_mhz,
         g_dma_timing_enabled ? 1 : 0,
         g_ip_timing_enabled ? 1 : 0,
         g_detail_every,
         g_log_flush_every);
    LOGI("manifest host_version=%s host_build=\"%s %s\" limits=0x%08x caps=0x%08x spu_caps=0x%08x stream_protocol=0x%08x required_protocol=%u bitstream_id=0x%08x expected_bitstream_id=0x%08x bitstream_id_compatible=%d pingpong_cap=%d descriptor_cap=%d scheduler=%d scheduler_policy=opt_in pl_scale=%d pl_scale_policy=opt_in devmem_all_resources=%d vpu_devmem_compat=%d mapping_policy=%s spu_silu=%d spu_rms=%d spu_rope=%d spu_softmax=%d bases my_ip=0x%llx dma=0x%llx ddr=0x%llx windows act=0x%08x weight=0x%08x result=0x%08x spu_out=0x%08x spu_param=0x%08x block_q8_0_size=%zu compact_weight_layout_required=1",
         FPGA_HOST_TRACE_VERSION,
         __DATE__,
         __TIME__,
         limits,
         caps,
         spu_caps,
         g_stream_protocol_version,
         FPGA_REQUIRED_STREAM_PROTOCOL_VERSION,
         g_bitstream_id,
         FPGA_EXPECTED_BITSTREAM_ID,
         bitstream_id_compatible ? 1 : 0,
         g_vpu_pingpong_supported ? 1 : 0,
         g_vpu_descriptor_supported ? 1 : 0,
         g_pingpong_scheduler_enabled ? 1 : 0,
         g_spu_q8_scale_stream_supported ? 1 : 0,
         g_allow_devmem_fallback ? 1 : 0,
         g_allow_vpu_devmem_compat ? 1 : 0,
         mapping_policy,
         g_spu_silu_supported ? 1 : 0,
         g_spu_rmsnorm_supported ? 1 : 0,
         g_spu_rope_supported ? 1 : 0,
         g_spu_softmax_supported ? 1 : 0,
         (unsigned long long) MY_IP_BASE_ADDRESS,
         (unsigned long long) DMA_BASE_PHYS,
         (unsigned long long) DDR_BASE_PHYS,
         ACT_BASE,
         WEIGHT_BASE,
         RESULT_BASE,
         SPU_OUT_BASE,
         SPU_PARAM_BASE,
         sizeof(block_q8_0_t));
    LOGINIT("bases my_ip=0x%llx reg=0x%llx lmm=0x%llx dma=0x%llx ddr=0x%llx",
            (unsigned long long) MY_IP_BASE_ADDRESS,
            (unsigned long long) REG_BASE_PHYS,
            (unsigned long long) LMM_BASE_PHYS,
            (unsigned long long) DMA_BASE_PHYS,
            (unsigned long long) DDR_BASE_PHYS);
    LOGI("mappings policy=%s dma=%s virt=0x%llx size=0x%zx vpu=%s virt=0x%llx size=0x%zx ddr=%s virt=0x%llx mapped_size=0x%zx advertised_size=0x%zx",
            mapping_policy,
            g_dma_map_source.c_str(), fpga_ptr_addr(g_dma), g_dma_map_size,
            g_vpu_map_source.c_str(), fpga_ptr_addr(g_vpu), g_vpu_map_size,
            g_ddr_map_source.c_str(), fpga_ptr_addr(g_ddr), g_ddr_map_size,
            g_ddr_advertised_size);
    LOGINIT("VPU windows act=0x%08x weight=0x%08x result=0x%08x spu_out=0x%08x spu_param=0x%08x data_movement=ZDMA_bulk_copy no_axi_stream_main=1",
            ACT_BASE, WEIGHT_BASE, RESULT_BASE, SPU_OUT_BASE, SPU_PARAM_BASE);
    LOGINIT("VPU raw_limits=0x%08x caps=0x%08x spu_caps=0x%08x", limits, caps, spu_caps);
    if (g_vpu_max_beats == 256) {
        LOGE("MAX_COL_BEATS=256 detected; DMA-to-IP path will run, but this large BRAM setting is still suspicious for timing/resource use");
    }
    LOGI("cache coherency source=%s strict=%d whitelist=%d stress=%d; msync barriers are issued before DDR-to-device and after device-to-DDR transfers",
         g_ddr_map_source.c_str(), g_strict_coherency ? 1 : 0,
         g_coherency_platform_whitelisted ? 1 : 0, g_run_coherency_stress ? 1 : 0);
    LOGINIT("fallback policy: FPGA_ABORT_ON_CPU_FALLBACK=%d default_no_cpu_matmul_fallback=1",
            g_abort_on_cpu_fallback ? 1 : 0);

    if (!g_packed_q8_supported) {
        fpga_fatal("REG_CAPS=0x%08x does not expose packed_q8 capability; refusing CPU fallback", caps);
    }
    if (!fpga_dma_basic_self_test()) {
        fpga_fatal("basic ZDMA-to-IP self-test failed; refusing CPU fallback");
    }
    if (!fpga_dma_packed_self_test()) {
        fpga_fatal("packed Q8 ZDMA-to-IP self-test failed; refusing CPU fallback");
    }
    if (!fpga_dma_row_limit_self_test()) {
        fpga_fatal("row-limit packed Q8 self-test failed; refusing CPU fallback");
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

    if (ddr_is_mapped()) {
        for (fpga_weight_cache_entry_t & entry : g_weight_cache) {
            if (entry.valid && ddr_range_fits(entry.header_off, sizeof(fpga_weight_cache_header_t))) {
                fpga_weight_cache_header_t header = ddr_read_weight_cache_header(entry.header_off);
                header.valid = 0U;
                ddr_write_weight_cache_header(entry.header_off, header);
                entry.valid = false;
            }
        }
        if (!g_weight_cache.empty()) {
            LOGI("weight tile cache metadata invalidated entries=%zu; payload was left intact", g_weight_cache.size());
        }
        g_weight_cache.clear();
    }

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
    LOGI("cleanup complete fpga_calls=%lld vpu_runs=%lld rejects=%lld attention_cpu_bypass=%lld vocab_projection_cpu_bypass=%lld elapsed_s=%.3f pingpong_cap=%d descriptor_cap=%d scheduler=%d activation_cache_enabled=%d activation_cache_hits=%lld misses=%lld weight_cache_builds=%lld hits=%lld misses=%lld bytes=%lld cache_lookup_ms=%.3f cache_crc_ms=%.3f weight_pack_ms=%.3f contract_checks=%lld raw_mismatches=%lld raw_repairs=%lld value_mismatches=%lld",
         g_fpga_count,
         g_fpga_vpu_runs,
         g_reject_count,
         g_attention_bypass_count,
         g_vocab_projection_bypass_count,
         elapsed_us > 0 ? (double) elapsed_us / 1000000.0 : 0.0,
         g_vpu_pingpong_supported ? 1 : 0,
         g_vpu_descriptor_supported ? 1 : 0,
         g_pingpong_scheduler_enabled ? 1 : 0,
         g_activation_cache_enabled ? 1 : 0,
         g_activation_cache_hits,
         g_activation_cache_misses,
         g_weight_cache_builds,
         g_weight_cache_hits,
         g_weight_cache_misses,
         g_weight_cache_bytes,
         (double) g_weight_cache_lookup_us / 1000.0,
         (double) g_weight_cache_crc_us / 1000.0,
         (double) g_weight_pack_us / 1000.0,
         g_contract_checks_done,
         g_contract_raw_mismatches,
         g_contract_raw_repairs,
         g_contract_value_mismatches);
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

static bool should_bypass_vocab_projection_to_cpu(
        const char * tensor_name,
        int64_t k,
        int64_t n,
        int64_t m) {
    (void) tensor_name;
    return g_vocab_projection_cpu_bypass &&
           m == 1 &&
           k > 0 &&
           n >= g_vocab_projection_min_n;
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
    const int64_t probe_k = src0 ? src0->ne[0] : 0;
    const int64_t probe_n = src0 ? src0->ne[1] : 0;
    const int64_t probe_m = src1 ? src1->ne[1] : 0;

    if (is_attention) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_attention_bypass_count++;
            if (g_attention_bypass_count == 1) {
                LOGI("attention path is currently bypassed to CPU; FPGA timing log below only covers Q8_0 matmul/GEMV hooks");
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return 0;
    }

    if (should_bypass_vocab_projection_to_cpu(tensor_name, probe_k, probe_n, probe_m)) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_vocab_projection_bypass_count++;
            if (g_vocab_projection_bypass_count == 1) {
                LOGI("vocab projection bypassed to CPU tensor=%s shape=K%lld_N%lld_M%lld threshold_n=%lld; unset FPGA_VOCAB_PROJECTION_CPU or set FPGA_ACCELERATE_VOCAB=1 to force FPGA",
                     tensor_name,
                     (long long) probe_k,
                     (long long) probe_n,
                     (long long) probe_m,
                     (long long) g_vocab_projection_min_n);
            }
            pthread_mutex_unlock(&g_mutex);
        }
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
    if (should_log_detail_run(g_fpga_count)) {
        LOGSTAGE("start tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld path=zdma_ddr_to_ip row_tiles=%lld group_tiles_per_rowtile~=%lld q8_blocks=%lld max_group_blocks=%d vpu_runs=%lld",
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
    }

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
    g_weight_cache_hits += totals.weight_cache_hits;
    g_weight_cache_misses += totals.weight_cache_misses;
    const double total_ms = (double) (t1 - t0) / 1000.0;
    const double prep_ms = (double) totals.prep_us / 1000.0;
    const double dma_act_ms = (double) totals.dma_act_us / 1000.0;
    const double dma_weight_ms = (double) totals.dma_weight_us / 1000.0;
    const double dma_scale_ms = (double) totals.dma_scale_us / 1000.0;
    const double dma_in_ms = dma_act_ms + dma_weight_ms + dma_scale_ms;
    const double ip_ms = (double) totals.ip_compute_us / 1000.0;
    const double dma_out_ms = (double) totals.dma_result_us / 1000.0;
    const double host_result_ms = (double) totals.host_result_us / 1000.0;
    const double host_accum_ms = (double) totals.host_accum_us / 1000.0;
    const double cache_lookup_ms = (double) totals.weight_cache_lookup_us / 1000.0;
    const double cache_crc_ms = (double) totals.weight_cache_crc_us / 1000.0;
    const double weight_pack_ms = (double) totals.weight_pack_us / 1000.0;

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
    if (host_result_ms > dominant_ms) {
        dominant = g_spu_q8_scale_stream_supported ? "host_scaled_row_read" : "host_result_unpack";
        dominant_ms = host_result_ms;
    }
    if (host_accum_ms > dominant_ms) {
        dominant = "host_accum";
        dominant_ms = host_accum_ms;
    }

    const size_t effective_bytes =
        totals.activation_bytes + totals.weight_bytes + totals.scale_bytes + totals.result_bytes;
    const double gmac_s = total_ms > 0.0 ? (double) macs / (total_ms * 1000000.0) : 0.0;
    const double mib_s = total_ms > 0.0 ? (double) effective_bytes * 1000.0 /
        (total_ms * 1024.0 * 1024.0) : 0.0;
    const double cycles_per_run =
        (g_fpga_clock_mhz > 0.0 && totals.vpu_runs > 0) ?
        ((double) totals.ip_compute_us * g_fpga_clock_mhz / (double) totals.vpu_runs) : 0.0;

    if (g_profile_every > 0 && (g_fpga_count == 1 || (g_fpga_count % g_profile_every) == 0)) {
        LOGSTAGE("tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld row_tiles=%lld group_tiles=%lld q8_blocks=%lld vpu_runs=%lld prep_ms=%.3f cache_lookup_ms=%.3f cache_crc_ms=%.3f weight_pack_ms=%.3f dma_input_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f ip_compute_ms=%.3f dma_output_ms=%.3f host_result_ms=%.3f host_accum_ms=%.3f total_ms=%.3f dominant=%s pl_scale=%d effective_GMAC/s=%.3f effective_MiB/s=%.1f act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu weight_cache_hits=%lld weight_cache_misses=%lld cycles_per_run=%.1f",
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
                 totals.vpu_runs,
                 prep_ms,
                 cache_lookup_ms,
                 cache_crc_ms,
                 weight_pack_ms,
                 dma_in_ms,
                 dma_act_ms,
                 dma_weight_ms,
                 dma_scale_ms,
                 ip_ms,
                 dma_out_ms,
                 host_result_ms,
                 host_accum_ms,
                 total_ms,
                 dominant,
                 g_spu_q8_scale_stream_supported ? 1 : 0,
                 gmac_s,
                 mib_s,
                 totals.activation_bytes,
                 totals.weight_bytes,
                 totals.scale_bytes,
                 totals.result_bytes,
                 totals.weight_cache_hits,
                 totals.weight_cache_misses,
                 cycles_per_run);
    }
    LOGIP("summary vpu_runs=%lld ip_ms=%.3f cycles_per_run=%.1f clock_mhz=%.3f",
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

    if (macs >= g_large_matrix_min_macs && should_log_detail_run(g_fpga_count)) {
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
