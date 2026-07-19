#include "fpga_host.h"
#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <atomic>
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
#define FPGA_HOST_TRACE_VERSION "zcu104-gemma3-q8-v55-spu-scale-sequential-contract"

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
#define LOGPROOF(fmt, ...)  fpga_log_line(true, "INFO",    true,  fmt, ##__VA_ARGS__)
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
// A UIO resource size only proves that Linux will mmap the address range; it
// does not prove that a large sequential write is safe for the deployed DDR
// interconnect/bitstream.  Keep the first cache experiment bounded until the
// board-specific address preflight and graduated cache tests are recorded.
// Larger requests require an explicit operator acknowledgement and are never
// enabled by the primary command accidentally.
static constexpr long long WEIGHT_CACHE_DEFAULT_MAX_MB = 16;
// Avoid a repeated, unsupported msync only for cache payloads large enough to
// make the call itself disruptive.  Per-tile scratch transfers retain v16's
// conservative msync attempt even after a UIO driver reports EINVAL.
static constexpr size_t   WEIGHT_CACHE_LARGE_MSYNC_BYTES = 16U * 1024U * 1024U;

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
static constexpr float    VPU_FP16_MAX_FINITE = 65504.0f;
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
// The ZDMA hardware accepts much larger descriptors, but the legacy board
// path has shown a normal-run-only staging corruption on 512 KiB WEIGHT
// copies.  Keep each submitted descriptor bounded to 64 KiB.  Consecutive
// chunks retain the same contiguous destination window and do not alter the
// VPU's tile, Q8 layout, or arithmetic contract.
static constexpr size_t    FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES = 64U * 1024U;
static constexpr int      FPGA_DEFAULT_STATUS_EVERY    = 0;
static constexpr int      FPGA_DEFAULT_PROFILE_EVERY   = 1;
static constexpr int      FPGA_DEFAULT_DETAIL_EVERY    = 0;
static constexpr long long FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS = 1000000LL;
static constexpr long long FPGA_STREAM_POLL_LOG_INTERVAL_US = 50000LL;
static constexpr size_t    FPGA_DMA_TRACE_DEPTH = 24U;

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
    U32 ZDMA_CH_SRC_CUR_PYLD_LSB;
    U32 ZDMA_CH_SRC_CUR_PYLD_MSB;
    U32 ZDMA_CH_DST_CUR_PYLD_LSB;
    U32 ZDMA_CH_DST_CUR_PYLD_MSB;
    U32 ZDMA_CH_SRC_CUR_DSCR_LSB;
    U32 ZDMA_CH_SRC_CUR_DSCR_MSB;
    U32 ZDMA_CH_DST_CUR_DSCR_LSB;
    U32 ZDMA_CH_DST_CUR_DSCR_MSB;
    U32 ZDMA_CH_TOTAL_BYTE;
    U32 ZDMA_CH_RATE_CTRL;
    U32 ZDMA_CH_IRQ_SRC_ACCT;
    U32 ZDMA_CH_IRQ_DST_ACCT;
    U32 dmy2[26];
    U32 ZDMA_CH_CTRL2;
};

static_assert(offsetof(dma_ctrl, ZDMA_CH_ISR) == 0x100, "unexpected ZDMA_CH_ISR offset");
static_assert(offsetof(dma_ctrl, ZDMA_CH_TOTAL_BYTE) == 0x188, "unexpected ZDMA_CH_TOTAL_BYTE offset");
static_assert(offsetof(dma_ctrl, ZDMA_CH_CTRL2) == 0x200, "unexpected ZDMA_CH_CTRL2 offset");

static constexpr uint32_t ZDMA_STATUS_STATE_MASK = 0x00000003;
static constexpr uint32_t ZDMA_CTRL2_START       = 0x00000001;
static constexpr uint32_t ZDMA_CTRL2_EN          = 0x00000001;
static constexpr uint32_t ZDMA_ISR_CLEAR_ALL     = 0x00000FFF;
static constexpr uint32_t ZDMA_ISR_DMA_DONE      = 0x00000400;
static constexpr uint32_t ZDMA_ISR_ERROR_MASK    = 0x00000BF9;
static constexpr uint32_t ZDMA_DATA_ATTR_AXCACHE = 0x04C3D30F;

// Use GGML's canonical block type rather than maintaining a private layout.
// This makes an upstream Q8_0 ABI change a compile-time failure instead of a
// silent scale/quant offset mismatch in the FPGA staging path.
using block_q8_0_t = block_q8_0;
static_assert(QK8_0 == VPU_QK8_0, "FPGA/GGML Q8_0 block length mismatch");
static_assert(sizeof(block_q8_0_t) == sizeof(ggml_half) + VPU_QK8_0,
              "FPGA/GGML Q8_0 block size mismatch");
static_assert(offsetof(block_q8_0_t, d) == 0U, "unexpected Q8_0 scale offset");
static_assert(offsetof(block_q8_0_t, qs) == sizeof(ggml_half),
              "unexpected Q8_0 quant offset");

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
    long long activation_scale_fp16_overflows;
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
    // Used only by FPGA_INPUT_INTEGRITY_CHECK.  It records the logical F32
    // activation columns before a raw-FPGA matmul, independent of padding in
    // src1->nb[1], so a host output store cannot silently damage the next
    // consumer in a multi-column graph.
    std::vector<uint8_t>      activation_input_snapshot;
    std::vector<block_q8_0_t> weight_tile_snapshot;
    std::vector<uint8_t>      weight_tensor_snapshot;
    std::vector<float>        contract_actual;
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

typedef struct {
    long long fp16_scale_overflows;
    float     max_abs;
    float     max_scale;
    int64_t   first_overflow_col;
    int64_t   first_overflow_block;
    float     first_overflow_abs;
    float     first_overflow_scale;
} fpga_activation_quant_stats_t;

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
// Count host-hook decisions independently from VPU runs.  A successful text
// response alone does not prove that every eligible Q8 GEMV used the FPGA:
// GGML may legally retain attention and the vocabulary projection on CPU.
// These atomics give an end-of-process, per-run coverage proof without
// changing routing or numerical results.
static std::atomic<long long> g_matmul_hook_calls{0};
static std::atomic<long long> g_q8_candidate_calls{0};
static std::atomic<long long> g_q8_intentional_cpu_bypass_calls{0};
static std::atomic<long long> g_q8_unavailable_cpu_fallback_calls{0};
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
static bool                g_dma_trace_enabled = false;
static bool                g_cleanup_done = false;
static bool                g_abort_on_cpu_fallback = true;
static bool                g_uio_inventory_logged = false;
static bool                g_allow_devmem_fallback = false;
static bool                g_allow_vpu_devmem_compat = true;
static bool                g_strict_coherency = false;
static bool                g_coherency_platform_whitelisted = false;
static bool                g_run_coherency_stress = false;
static bool                g_ddr_msync_unsupported_logged = false;
static bool                g_ddr_msync_unavailable = false;
static bool                g_weight_cache_enabled = false;
static bool                g_weight_cache_full_logged = false;
static bool                g_weight_cache_crc_verify_each_lookup = false;
static bool                g_activation_cache_enabled = false;
// Qualification-only input ownership proof.  This is intentionally opt-in:
// it copies each logical F32 activation matrix before/after a raw FPGA
// matmul, which is appropriate for validating M>1 graph layouts but not for
// production throughput measurement.
static bool                g_activation_input_integrity_check = false;
static bool                g_contract_check_abort = false;
static bool                g_contract_forensic_replay = true;
// Full byte-for-byte staging scans are valuable for a single forensic replay,
// but reading every ACT/WEIGHT byte twice per tile through the UIO mapping is
// not part of the numerical contract and dominates C0 timing.  Keep the
// bounded ordering fence on every launch; make the exhaustive scan explicit.
static bool                g_contract_deep_staging = false;
// Contract mode must prove the raw FPGA path without taking ownership of the
// GGML destination tensor.  In this explicit shadow mode, the host retains
// its hardware result in contract_actual for checking while the upstream
// threaded kernel writes dst.  It is not a hardware-unavailable CPU fallback
// and is never the production-acceleration route.
static bool                g_contract_cpu_shadow_dst = false;
// Raw-F32 propagation is useful only when deliberately investigating a
// downstream attention/KV divergence.  It is not a C0/C1 success criterion:
// a raw/value contract can pass while a later attention score diverges.  Keep
// it separate from the ordinary C0/C1 shadow route so a stale legacy setting
// cannot turn a validation command into that forensic experiment accidentally.
static bool                g_contract_raw_propagation_diagnostic = false;
static bool                g_contract_legacy_canonical_override_ignored = false;
// Diagnostic-only: validate the Q8_0 source tensors in the normal GGML
// execution order, but return control to CPU before any model GEMV is sent to
// ZDMA/VPU.  This distinguishes a bad/mutable GGUF source from a fault that
// only appears after raw FPGA launches.  The ggml caller skips FPGA init in
// this mode, so source audit never maps or self-tests board hardware.
static bool                g_q8_source_audit_only = false;
static bool                g_q8_source_audit_mode_logged = false;
static bool                g_clear_result_before_run = false;
static bool                g_contract_raw_repair_enabled = false;
// The fused path has not yet passed an end-to-end language/logit A/B test on
// the legacy board bitstream.  Keep the explicit partial -> scale ->
// accumulate order as the production default.  Fusion remains an opt-in
// performance experiment after that test passes.
static bool                g_fuse_raw_result_accum = false;
// The tied embedding / vocabulary projection has 262144 rows for Gemma3-1B.
// It is the final, top-token-sensitive operation and the current legacy
// bitstream advertises neither the required stream protocol nor bitstream ID.
// Keep it on the established CPU GGML path by default until FPGA logits pass
// an end-to-end A/B comparison.  All block GEMVs remain eligible for FPGA.
static bool                g_vocab_projection_cpu_bypass = true;
// A bitstream without the required identity/protocol has already produced
// raw-contract failures on the board.  Primary inference must prefer a
// readable answer over silently consuming corrupt FPGA outputs.  Contract
// runs remain allowed so the fault can be isolated without changing this
// safety policy.
static bool                g_legacy_raw_cpu_bypass = false;
static long long           g_legacy_raw_cpu_bypass_count = 0;
static int                 g_profile_every = FPGA_DEFAULT_PROFILE_EVERY;
static int                 g_ip_status_every = FPGA_DEFAULT_STATUS_EVERY;
static int                 g_detail_every = FPGA_DEFAULT_DETAIL_EVERY;
static int                 g_contract_check_limit = 0;
// P2 has its own bounded contract.  It proves the VPU->SPU path before a
// scaled result is permitted to own a live GGML destination.  This must not
// be combined with C0: C0 proves raw VPU arithmetic while this contract proves
// fixed-point scale/accumulate and SPU output ownership.
static int                 g_spu_scale_contract_check_limit = 0;
static double              g_spu_scale_contract_atol = 2.0e-3;
static int                 g_contract_raw_retry_limit = 1;
static int                 g_runtime_max_rows = VPU_SAFE_RUNTIME_ROWS;
static int64_t             g_vocab_projection_min_n = 65536;
static long long           g_dma_timeout_us = FPGA_DEFAULT_DMA_TIMEOUT_US;
static long long           g_ip_timeout_us = FPGA_DEFAULT_IP_TIMEOUT_US;
static size_t              g_zdma_max_transfer_bytes = FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES;
static long long           g_large_matrix_min_macs = FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS;
static double              g_fpga_clock_mhz = 0.0;
static double              g_contract_atol = 1.0e-3;
static double              g_contract_rtol = 1.0e-4;
static long long           g_contract_checks_done = 0;
static long long           g_contract_raw_mismatches = 0;
static long long           g_contract_raw_repairs = 0;
static long long           g_contract_value_mismatches = 0;
static long long           g_contract_cpu_shadow_dst_values = 0;
static long long           g_contract_staging_restage_count = 0;
// C0 is a bounded hardware qualification. Once its requested number of
// checked GEMVs has completed, later graph GEMVs must execute only in the
// native CPU backend. Launching unverified VPU work after that boundary can
// turn a later CPU-side failure into misleading C0 evidence.
static long long           g_contract_limit_cpu_bypass_count = 0;
static bool                g_contract_limit_cpu_bypass_logged = false;
static long long           g_spu_scale_contract_checks_done = 0;
static long long           g_spu_scale_contract_limit_cpu_bypass_count = 0;
static bool                g_spu_scale_contract_limit_cpu_bypass_logged = false;
static long long           g_spu_scale_contract_tile_checks = 0;
static long long           g_spu_scale_contract_raw_mismatches = 0;
static long long           g_spu_scale_contract_value_mismatches = 0;
static long long           g_spu_scale_contract_cpu_shadow_dst_values = 0;
static bool                g_spu_scale_contract_cpu_shadow_dst = false;
static long long           g_q8_source_audit_checks = 0;
static long long           g_q8_source_audit_failures = 0;
static long long           g_activation_input_integrity_checks = 0;
static long long           g_activation_input_integrity_failures = 0;
static long long           g_activation_scale_fp16_overflows = 0;
// Set only while a C0 source preflight rejects an immutable Q8 tensor.  This
// lets the outer hook report the real cause instead of mislabelling it as a
// ZDMA/VPU transfer failure.
static bool                g_contract_source_validation_failed = false;
// Set by llama-model-loader only after its complete upstream tensor validation
// succeeds.  C0 starts before normal FPGA initialization, so this handshake
// prevents a partially copied frontend build from bypassing the loader gate
// and reaching MY_IP/ZDMA with an invalid model source.
static std::atomic<bool>   g_contract_loader_validation_passed{false};

typedef struct {
    uint32_t status;
    uint32_t isr;
    uint32_t ctrl2;
    long long polls;
    bool saw_enabled;
} zdma_completion_info_t;

typedef struct {
    bool valid;
    unsigned long long sequence;
    char tag[48];
    uint64_t src_phys;
    uint64_t dst_phys;
    size_t bytes;
    uint32_t pre_status;
    uint32_t pre_isr;
    uint32_t pre_ctrl2;
    uint32_t total_bytes_before_clear;
    uint32_t pre_vpu_status;
    uint32_t pre_vpu_progress;
    uint32_t post_status;
    uint32_t post_isr;
    uint32_t post_ctrl2;
    uint32_t total_bytes_after_transfer;
    uint32_t post_vpu_status;
    uint32_t post_vpu_progress;
    long long elapsed_us;
    long long polls;
    bool saw_enabled;
} fpga_dma_trace_record_t;

static fpga_dma_trace_record_t g_dma_trace[FPGA_DMA_TRACE_DEPTH] = {};
static unsigned long long g_dma_trace_sequence = 0;

typedef struct {
    bool    valid;
    int     local_row;
    int     group_block;
    int64_t global_row;
    int64_t k_block;
} fpga_raw_mismatch_location_t;
static uint32_t            g_weight_cache_next_off = WEIGHT_CACHE_BASE;
static uint32_t            g_weight_cache_end_off  = WEIGHT_CACHE_BASE;
static uint32_t            g_stream_protocol_version = 0;
static uint32_t            g_bitstream_id = 0;
static long long           g_weight_cache_budget_mb = 0;

static int g_current_layer_id = 0;
static int g_current_seq_pos  = 0;
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
#if defined(__aarch64__) || defined(__arm__)
    // The ZCU104 host is Arm.  A compiler/CPU fence alone does not guarantee
    // that a posted device write reached ZDMA before the next MMIO poll.
    // DSB SY orders completion of the descriptor and CTRL2 writes with the
    // subsequent read of the same peripheral.
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

extern "C" int fpga_source_audit_only_requested(void) {
    // ggml-cpu calls this before pthread_once(fpga_init), so it must not map
    // MY_IP, ZDMA, or DDR high.
    return env_flag_enabled("FPGA_SOURCE_AUDIT_ONLY") ? 1 : 0;
}

extern "C" int fpga_contract_check_requested(void) {
    // This query is called before fpga_init() by llama-cli.  It must not
    // access MMIO, map memory, or change any persistent host state.
    return env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000) > 0 ? 1 : 0;
}

extern "C" void fpga_mark_model_tensor_validation_passed(void) {
    // This function is intentionally board-free: it can run while main has
    // deferred fpga_init() for C0.  Do not treat a normal model load as C0
    // evidence; the loader calls it only for C0/source-audit validation.
    if (fpga_contract_check_requested() || fpga_source_audit_only_requested()) {
        g_contract_loader_validation_passed.store(true, std::memory_order_release);
    }
}

int fpga_model_tensor_validation_passed(void) {
    return g_contract_loader_validation_passed.load(std::memory_order_acquire) ? 1 : 0;
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
    if (!ddr_is_mapped() || bytes == 0U) {
        return false;
    }
    const size_t offset = (size_t) off;
    return offset <= g_ddr_map_size && bytes <= g_ddr_map_size - offset;
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

static const uint32_t * fpga_crc32_table() {
    // This host serializes matmul/cache work with g_mutex, therefore lazy
    // initialization is safe and avoids a large static initializer.
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t byte = 0; byte < 256U; ++byte) {
            uint32_t value = byte;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value >> 1) ^ ((value & 1U) ? 0xEDB88320U : 0U);
            }
            table[byte] = value;
        }
        initialized = true;
    }
    return table;
}

static uint32_t fpga_crc32_update(uint32_t crc, const uint8_t * data, size_t bytes) {
    const uint32_t * table = fpga_crc32_table();
    for (size_t i = 0; i < bytes; ++i) {
        crc = table[(crc ^ (uint32_t) data[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc;
}

static uint32_t fpga_crc32_update_zeros(uint32_t crc, size_t bytes) {
    static const uint8_t zeros[256] = {};
    while (bytes != 0U) {
        const size_t chunk = std::min(bytes, sizeof(zeros));
        crc = fpga_crc32_update(crc, zeros, chunk);
        bytes -= chunk;
    }
    return crc;
}

static uint32_t fpga_crc32_ddr(uint32_t off, size_t bytes) {
    // This diagnostic path deliberately reads through the uncached DDR UIO
    // mapping.  It must never be used while constructing the cache: a full
    // cache can be about 1 GiB and the old bit-at-a-time implementation made
    // the first prefill appear to hang the ZCU104.
    volatile const uint8_t * data = (volatile const uint8_t *) ddr_ptr(off, bytes);
    const uint32_t * table = fpga_crc32_table();
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < bytes; ++i) {
        crc = table[(crc ^ (uint32_t) data[i]) & 0xFFU] ^ (crc >> 8);
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

    // A MY_IP UIO resource can advertise the entire Vivado segment.  This
    // driver uses only the established 4 MiB register/local-memory ABI, so
    // never turn an advertised UIO size into a larger process mapping.
    if (!map_region_prefer_uio("MY_IP", "FPGA_VPU_UIO", REG_BASE_PHYS, VPU_DEVMEM_COMPAT_MMAP,
                               VPU_DEVMEM_COMPAT_MMAP, "MY_IP/VPU", VPU_DEVMEM_COMPAT_MMAP,
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

    const bool large_cache_confirmed = env_flag_enabled("FPGA_WEIGHT_CACHE_LARGE_CONFIRMED");
    if (cache_mb > WEIGHT_CACHE_DEFAULT_MAX_MB && !large_cache_confirmed) {
        const uint64_t cache_start = (uint64_t) DDR_BASE_PHYS + (uint64_t) WEIGHT_CACHE_BASE;
        const uint64_t cache_end = cache_start + (uint64_t) cache_mb * 1024ULL * 1024ULL;
        LOGE("refusing unconfirmed large weight cache: FPGA_WEIGHT_CACHE_MB=%lld exceeds default_max_mb=%lld; requested_phys=[0x%llx,0x%llx). No DDR mapping or cache write was performed. Run the read-only ddr_high/reserved-memory preflight and graduated cache tests first; FPGA_WEIGHT_CACHE_LARGE_CONFIRMED is an operator acknowledgement, not a safety check",
             cache_mb,
             WEIGHT_CACHE_DEFAULT_MAX_MB,
             (unsigned long long) cache_start,
             (unsigned long long) cache_end);
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
    LOGINIT("DDR map policy: cache=enabled budget_mb=%lld requested_size=0x%zx large_confirmed=%d",
            g_weight_cache_budget_mb, g_ddr_requested_map_size,
            large_cache_confirmed ? 1 : 0);
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
    // A UIO ddr_high mapping on this board reports EINVAL for msync.  Keep
    // v16's conservative per-tile attempt for ordinary ACT/WEIGHT/RESULT DMA
    // ranges, but do not repeat that unsupported call for one large cache
    // payload (potentially 1.1 GiB) during warm-up.
    if (g_ddr_msync_unavailable && bytes >= WEIGHT_CACHE_LARGE_MSYNC_BYTES) {
        mmio_fence();
        return true;
    }
    if (msync((void *) aligned_begin, len, flags) != 0) {
        const int saved_errno = errno;
        const bool likely_uncached_mapping =
            g_ddr_map_source.find("O_SYNC") != std::string::npos ||
            g_ddr_map_source.find("/dev/uio") != std::string::npos;
        if (!g_strict_coherency && !env_flag_enabled("FPGA_STRICT_MSYNC") &&
            likely_uncached_mapping &&
            (saved_errno == EINVAL || saved_errno == ENODEV)) {
            if (!g_ddr_msync_unsupported_logged) {
                LOGI("msync unsupported for ddr_high source=%s errno=%d (%s); generic-uio physical maps are normally non-cached, but cacheability is kernel-owned. Continuing with ordered volatile DDR accesses; this message alone does not prove a cache fault.",
                     g_ddr_map_source.c_str(), saved_errno, strerror(saved_errno));
                g_ddr_msync_unsupported_logged = true;
            }
            g_ddr_msync_unavailable = true;
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
    if (!off || bytes == 0U || phys < DDR_BASE_PHYS) {
        return false;
    }
    const uint64_t delta = phys - DDR_BASE_PHYS;
    if (delta > UINT32_MAX || delta > (uint64_t) g_ddr_map_size ||
        (uint64_t) bytes > (uint64_t) g_ddr_map_size - delta) {
        return false;
    }
    *off = (uint32_t) delta;
    return true;
}

static bool phys_range_fits(
        uint64_t phys,
        size_t bytes,
        uint64_t base,
        size_t mapped_size) {
    if (bytes == 0U || phys < base) {
        return false;
    }
    const uint64_t delta = phys - base;
    return delta <= (uint64_t) mapped_size &&
           (uint64_t) bytes <= (uint64_t) mapped_size - delta;
}

static void zdma_set_addr(volatile U32 * lo, volatile U32 * hi, uint64_t value) {
    *lo = (U32) (value & 0xFFFFFFFFULL);
    *hi = (U32) (value >> 32);
}

static uint64_t zdma_read_addr(volatile U32 * lo, volatile U32 * hi) {
    return (uint64_t) *lo | ((uint64_t) *hi << 32);
}

// ZDMA_CH_TOTAL_BYTE is write-one-to-clear.  This is the same read-then-write
// sequence used by XZDma_TotalByteClear() in AMD's standalone driver.  The
// counter belongs to the channel, not to a single descriptor, so leaving it
// uncleared makes a long inference eventually assert BYTE_CNT_OVRFL.
static uint32_t zdma_total_byte_clear(void) {
    if (!dma_is_mapped()) {
        return 0;
    }
    const uint32_t total = g_dma->ZDMA_CH_TOTAL_BYTE;
    if (total != 0U) {
        g_dma->ZDMA_CH_TOTAL_BYTE = total;
        mmio_fence();
    }
    return total;
}

static void zdma_format_error_mask(uint32_t isr, char * out, size_t out_size) {
    struct zdma_error_name_t {
        uint32_t mask;
        const char * name;
    };
    static constexpr zdma_error_name_t kErrors[] = {
        { 0x00000001U, "INV_APB" },
        { 0x00000008U, "BYTE_CNT_OVRFL" },
        { 0x00000010U, "SRC_IRQ_ACCT_OVRFL" },
        { 0x00000020U, "DST_IRQ_ACCT_OVRFL" },
        { 0x00000040U, "AXI_RD_SRC_DSCR" },
        { 0x00000080U, "AXI_RD_DST_DSCR" },
        { 0x00000100U, "AXI_RD_DATA" },
        { 0x00000200U, "AXI_WR_DATA" },
        { 0x00000800U, "DMA_PAUSE" },
    };

    if (out_size == 0U) {
        return;
    }
    out[0] = '\0';
    size_t used = 0;
    const uint32_t errors = isr & ZDMA_ISR_ERROR_MASK;
    if (errors == 0U) {
        snprintf(out, out_size, "none");
        return;
    }
    for (const zdma_error_name_t & entry : kErrors) {
        if ((errors & entry.mask) == 0U || used >= out_size) {
            continue;
        }
        const int written = snprintf(out + used, out_size - used, "%s%s",
                                     used == 0U ? "" : "|", entry.name);
        if (written <= 0) {
            break;
        }
        const size_t advanced = (size_t) written;
        if (advanced >= out_size - used) {
            used = out_size - 1U;
            break;
        }
        used += advanced;
    }
}

static void zdma_dump(const char * tag) {
    const uint32_t isr = g_dma ? g_dma->ZDMA_CH_ISR : 0xFFFFFFFFU;
    char errors[160];
    zdma_format_error_mask(isr, errors, sizeof(errors));
    LOGE("ZDMA dump tag=%s status=0x%08x isr=0x%08x errors=%s ctrl0=0x%08x ctrl1=0x%08x ctrl2=0x%08x total_bytes=0x%08x data_attr=0x%08x cur_src=0x%llx cur_dst=0x%llx src_desc=[0x%08x,0x%08x,0x%08x,0x%08x] dst_desc=[0x%08x,0x%08x,0x%08x,0x%08x]",
         tag ? tag : "?",
         g_dma ? g_dma->ZDMA_CH_STATUS : 0xFFFFFFFFU,
         isr,
         errors,
         g_dma ? g_dma->ZDMA_CH_CTRL0 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_CTRL1 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_CTRL2 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_TOTAL_BYTE : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_DATA_ATTR : 0xFFFFFFFFU,
         g_dma ? (unsigned long long) zdma_read_addr(&g_dma->ZDMA_CH_SRC_CUR_PYLD_LSB,
                                                      &g_dma->ZDMA_CH_SRC_CUR_PYLD_MSB) : 0ULL,
         g_dma ? (unsigned long long) zdma_read_addr(&g_dma->ZDMA_CH_DST_CUR_PYLD_LSB,
                                                      &g_dma->ZDMA_CH_DST_CUR_PYLD_MSB) : 0ULL,
         g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD0 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD1 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD2 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD3 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD0 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD1 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD2 : 0xFFFFFFFFU,
         g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD3 : 0xFFFFFFFFU);
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
    // ZDMA_CH_ISR is write-one-to-clear.  Clear any sticky result left by a
    // prior process before the first descriptor is programmed.
    g_dma->ZDMA_CH_ISR            = ZDMA_ISR_CLEAR_ALL;
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
    const uint32_t stale_total_bytes = zdma_total_byte_clear();

    LOGINIT("ZDMA init base=0x%llx virt=0x%llx status=0x%08x isr=0x%08x ctrl0=0x%08x ctrl1=0x%08x data_attr=0x%08x stale_total_bytes=0x%08x total_bytes_after_clear=0x%08x completion_gate=isr_ack_then_dma_done_and_ctrl2_en_clear",
            (unsigned long long) DMA_BASE_PHYS,
            fpga_ptr_addr(g_dma),
            g_dma->ZDMA_CH_STATUS,
            g_dma->ZDMA_CH_ISR,
            g_dma->ZDMA_CH_CTRL0,
            g_dma->ZDMA_CH_CTRL1,
            g_dma->ZDMA_CH_DATA_ATTR,
            stale_total_bytes,
            g_dma->ZDMA_CH_TOTAL_BYTE);
    return true;
}

// AMD defines CTRL2.EN as hardware-cleared after a DMA operation finishes.
// Descriptor registers must remain stable while EN is set.  The legacy host
// accepted STATUS.state==3 as completion and could therefore rewrite a new
// descriptor while the channel was still enabled.  That race matches the
// observed pattern: a long normal sequence corrupts a staged weight tile,
// while a slower forensic replay of the exact tile is correct.
static bool zdma_wait_channel_disabled(const char * tag, const char * phase) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA completion gate has no mapped channel tag=%s phase=%s",
             tag ? tag : "?", phase ? phase : "?");
        return false;
    }

    const long long t0 = now_us();
    long long polls = 0;
    while ((g_dma->ZDMA_CH_CTRL2 & ZDMA_CTRL2_EN) != 0U) {
        if (now_us() - t0 > g_dma_timeout_us) {
            LOGE("ZDMA EN timeout tag=%s phase=%s status=0x%08x state=%u isr=0x%08x ctrl2=0x%08x polls=%lld",
                 tag ? tag : "?",
                 phase ? phase : "?",
                 g_dma->ZDMA_CH_STATUS,
                 g_dma->ZDMA_CH_STATUS & ZDMA_STATUS_STATE_MASK,
                 g_dma->ZDMA_CH_ISR,
                 g_dma->ZDMA_CH_CTRL2,
                 polls);
            zdma_dump(tag);
            return false;
        }
        ++polls;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }
    mmio_fence();
    return true;
}

// DMA_DONE is sticky and W1C.  A DSB orders the clear write but does not by
// itself prove that a subsequent CPU load sees the cleared event.  Without
// this acknowledgement, the post-START completion loop can accept a previous
// transfer's DONE bit while CTRL2.EN is still sampled as idle, then let the
// host reuse ACT/WEIGHT staging before the new transfer has started.
static bool zdma_clear_isr_for_new_transfer(const char * tag) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA ISR-clear gate has no mapped channel tag=%s", tag ? tag : "?");
        return false;
    }

    g_dma->ZDMA_CH_ISR = ZDMA_ISR_CLEAR_ALL;
    mmio_fence();

    const long long t0 = now_us();
    long long polls = 0;
    const uint32_t completion_or_error = ZDMA_ISR_DMA_DONE | ZDMA_ISR_ERROR_MASK;
    while (true) {
        const uint32_t isr = g_dma->ZDMA_CH_ISR;
        if ((isr & completion_or_error) == 0U) {
            mmio_fence();
            return true;
        }
        if (now_us() - t0 > g_dma_timeout_us) {
            char errors[160];
            zdma_format_error_mask(isr, errors, sizeof(errors));
            LOGE("ZDMA ISR clear timeout tag=%s isr=0x%08x errors=%s ctrl2=0x%08x polls=%lld; refusing to start with a stale completion event",
                 tag ? tag : "?", isr, errors, g_dma->ZDMA_CH_CTRL2, polls);
            zdma_dump(tag);
            return false;
        }
        ++polls;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }
}

// CTRL2.EN is a channel-state bit, not an event for the descriptor we just
// programmed.  Polling only for EN==0 immediately after writing START can
// observe the old disabled state before the posted start write is accepted.
// The caller has positively cleared the old W1C event, so DMA_DONE here is a
// completion generated by the descriptor just launched.
static bool zdma_wait_transfer_complete(const char * tag, zdma_completion_info_t * info) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA completion gate has no mapped channel tag=%s",
             tag ? tag : "?");
        return false;
    }

    const long long t0 = now_us();
    long long polls = 0;
    bool saw_enabled = false;
    while (true) {
        const uint32_t ctrl2 = g_dma->ZDMA_CH_CTRL2;
        const uint32_t isr = g_dma->ZDMA_CH_ISR;
        const uint32_t status = g_dma->ZDMA_CH_STATUS;
        const bool enabled = (ctrl2 & ZDMA_CTRL2_EN) != 0U;
        const bool dma_done = (isr & ZDMA_ISR_DMA_DONE) != 0U;

        saw_enabled = saw_enabled || enabled;
        if (info) {
            info->status = status;
            info->isr = isr;
            info->ctrl2 = ctrl2;
            info->polls = polls;
            info->saw_enabled = saw_enabled;
        }
        if ((isr & ZDMA_ISR_ERROR_MASK) != 0U) {
            char errors[160];
            zdma_format_error_mask(isr, errors, sizeof(errors));
            LOGE("ZDMA error tag=%s status=0x%08x state=%u isr=0x%08x errors=%s ctrl2=0x%08x total_bytes=0x%08x saw_enabled=%d polls=%lld",
                 tag ? tag : "?",
                 status,
                 status & ZDMA_STATUS_STATE_MASK,
                 isr,
                 errors,
                 ctrl2,
                 g_dma->ZDMA_CH_TOTAL_BYTE,
                 saw_enabled ? 1 : 0,
                 polls);
            zdma_dump(tag);
            return false;
        }
        if (dma_done && !enabled) {
            mmio_fence();
            return true;
        }
        if (now_us() - t0 > g_dma_timeout_us) {
            LOGE("ZDMA completion timeout tag=%s status=0x%08x state=%u isr=0x%08x ctrl2=0x%08x total_bytes=0x%08x dma_done=%d saw_enabled=%d polls=%lld",
                 tag ? tag : "?",
                 status,
                 status & ZDMA_STATUS_STATE_MASK,
                 isr,
                 ctrl2,
                 g_dma->ZDMA_CH_TOTAL_BYTE,
                 dma_done ? 1 : 0,
                 saw_enabled ? 1 : 0,
                 polls);
            zdma_dump(tag);
            return false;
        }
        ++polls;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }
}

// Store a bounded history in RAM. It is enabled automatically by a contract
// run and emitted only on a raw mismatch or a DMA-completion failure, so it
// does not create a giant log or perturb normal primary-command timing.
static void fpga_dma_trace_record(
        const char * tag,
        uint64_t src_phys,
        uint64_t dst_phys,
        size_t bytes,
        uint32_t pre_status,
        uint32_t pre_isr,
        uint32_t pre_ctrl2,
        uint32_t total_bytes_before_clear,
        uint32_t pre_vpu_status,
        uint32_t pre_vpu_progress,
        uint32_t total_bytes_after_transfer,
        uint32_t post_vpu_status,
        uint32_t post_vpu_progress,
        long long elapsed_us,
        const zdma_completion_info_t & completion) {
    if (!g_dma_trace_enabled) {
        return;
    }
    const unsigned long long sequence = ++g_dma_trace_sequence;
    fpga_dma_trace_record_t & record =
        g_dma_trace[(sequence - 1U) % FPGA_DMA_TRACE_DEPTH];
    record = {};
    record.valid = true;
    record.sequence = sequence;
    snprintf(record.tag, sizeof(record.tag), "%s", tag ? tag : "?");
    record.src_phys = src_phys;
    record.dst_phys = dst_phys;
    record.bytes = bytes;
    record.pre_status = pre_status;
    record.pre_isr = pre_isr;
    record.pre_ctrl2 = pre_ctrl2;
    record.total_bytes_before_clear = total_bytes_before_clear;
    record.pre_vpu_status = pre_vpu_status;
    record.pre_vpu_progress = pre_vpu_progress;
    record.post_status = completion.status;
    record.post_isr = completion.isr;
    record.post_ctrl2 = completion.ctrl2;
    record.total_bytes_after_transfer = total_bytes_after_transfer;
    record.post_vpu_status = post_vpu_status;
    record.post_vpu_progress = post_vpu_progress;
    record.elapsed_us = elapsed_us;
    record.polls = completion.polls;
    record.saw_enabled = completion.saw_enabled;
}

static void fpga_dma_trace_dump(
        const char * reason,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id,
        const char * failed_transfer_tag) {
    if (!g_dma_trace_enabled || g_dma_trace_sequence == 0U) {
        return;
    }
    const unsigned long long first = g_dma_trace_sequence > FPGA_DMA_TRACE_DEPTH ?
        g_dma_trace_sequence - FPGA_DMA_TRACE_DEPTH + 1U : 1U;
    LOGE("DMA_TRACE_BEGIN reason=%s tensor=%s layer=%d tile=%u failed_transfer=%s first_seq=%llu last_seq=%llu depth=%zu",
         reason ? reason : "?",
         tensor_name ? tensor_name : "?",
         layer_id,
         tile_id,
         failed_transfer_tag ? failed_transfer_tag : "none",
         first,
         g_dma_trace_sequence,
         FPGA_DMA_TRACE_DEPTH);
    for (unsigned long long sequence = first; sequence <= g_dma_trace_sequence; ++sequence) {
        const fpga_dma_trace_record_t & record =
            g_dma_trace[(sequence - 1U) % FPGA_DMA_TRACE_DEPTH];
        if (!record.valid || record.sequence != sequence) {
            continue;
        }
        LOGE("DMA_TRACE seq=%llu tag=%s src=0x%llx dst=0x%llx bytes=%zu elapsed_us=%lld polls=%lld saw_enabled=%d total_before_clear=0x%08x pre_status=0x%08x pre_isr=0x%08x pre_ctrl2=0x%08x pre_vpu_status=0x%08x pre_vpu_progress=0x%08x post_status=0x%08x post_isr=0x%08x post_ctrl2=0x%08x total_after=0x%08x post_vpu_status=0x%08x post_vpu_progress=0x%08x dma_done=%d",
             record.sequence,
             record.tag,
             (unsigned long long) record.src_phys,
             (unsigned long long) record.dst_phys,
             record.bytes,
             record.elapsed_us,
             record.polls,
             record.saw_enabled ? 1 : 0,
             record.total_bytes_before_clear,
             record.pre_status,
             record.pre_isr,
             record.pre_ctrl2,
             record.pre_vpu_status,
             record.pre_vpu_progress,
             record.post_status,
             record.post_isr,
             record.post_ctrl2,
             record.total_bytes_after_transfer,
             record.post_vpu_status,
             record.post_vpu_progress,
             (record.post_isr & ZDMA_ISR_DMA_DONE) != 0U ? 1 : 0);
    }
    LOGE("DMA_TRACE_END reason=%s tensor=%s layer=%d tile=%u failed_transfer=%s",
         reason ? reason : "?",
         tensor_name ? tensor_name : "?",
         layer_id,
         tile_id,
         failed_transfer_tag ? failed_transfer_tag : "none");
}

static bool fpga_dma_copy_one(uint64_t src_phys, uint64_t dst_phys, size_t bytes, const char * tag) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA is not mapped for tag=%s", tag ? tag : "?");
        return false;
    }
    if (bytes == 0 || bytes > UINT32_MAX) {
        LOGE("invalid ZDMA byte count tag=%s bytes=%zu", tag ? tag : "?", bytes);
        return false;
    }

    // Never rewrite descriptors until the preceding transfer's hardware EN
    // bit is clear.  STATUS is retained for diagnostics only; it is not a
    // sufficient ownership/completion gate.
    if (!zdma_wait_channel_disabled(tag, "before_descriptor")) {
        return false;
    }
    const uint32_t total_bytes_before_clear = zdma_total_byte_clear();

    uint32_t src_ddr_off = 0;
    if (phys_to_ddr_offset(src_phys, bytes, &src_ddr_off)) {
        if (!msync_ddr_range(src_ddr_off, bytes, false, tag)) {
            return false;
        }
    }

    if (!zdma_clear_isr_for_new_transfer(tag)) {
        return false;
    }
    zdma_set_addr(&g_dma->ZDMA_CH_SRC_DSCR_WORD0, &g_dma->ZDMA_CH_SRC_DSCR_WORD1, src_phys);
    g_dma->ZDMA_CH_SRC_DSCR_WORD2 = (U32) bytes;
    // DMA_DONE is a channel-completion event.  Do not request per-descriptor
    // interrupts: this user-space driver polls the sticky ISR and must not
    // introduce a new UIO interrupt-delivery dependency.
    g_dma->ZDMA_CH_SRC_DSCR_WORD3 = 0U;
    zdma_set_addr(&g_dma->ZDMA_CH_DST_DSCR_WORD0, &g_dma->ZDMA_CH_DST_DSCR_WORD1, dst_phys);
    g_dma->ZDMA_CH_DST_DSCR_WORD2 = (U32) bytes;
    g_dma->ZDMA_CH_DST_DSCR_WORD3 = 0U;
    mmio_fence();

    uint32_t pre_status = 0;
    uint32_t pre_isr = 0;
    uint32_t pre_ctrl2 = 0;
    uint32_t pre_vpu_status = 0;
    uint32_t pre_vpu_progress = 0;
    if (g_dma_trace_enabled) {
        pre_status = g_dma->ZDMA_CH_STATUS;
        pre_isr = g_dma->ZDMA_CH_ISR;
        pre_ctrl2 = g_dma->ZDMA_CH_CTRL2;
        if (vpu_is_mapped()) {
            pre_vpu_status = vpu_rd32(REG_STATUS);
            pre_vpu_progress = vpu_rd32(REG_PROGRESS);
        }
    }

    const long long t0 = now_us();
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_START;
    mmio_fence();

    zdma_completion_info_t completion = {};
    if (!zdma_wait_transfer_complete(tag, &completion)) {
        const uint32_t total_bytes_after_transfer = g_dma->ZDMA_CH_TOTAL_BYTE;
        const uint32_t post_vpu_status = vpu_is_mapped() ? vpu_rd32(REG_STATUS) : 0U;
        const uint32_t post_vpu_progress = vpu_is_mapped() ? vpu_rd32(REG_PROGRESS) : 0U;
        fpga_dma_trace_record(tag, src_phys, dst_phys, bytes, pre_status, pre_isr,
                              pre_ctrl2, total_bytes_before_clear,
                              pre_vpu_status, pre_vpu_progress, total_bytes_after_transfer,
                              post_vpu_status, post_vpu_progress,
                              now_us() - t0, completion);
        fpga_dma_trace_dump("transfer_completion_failure", nullptr, -1, 0U, tag);
        LOGE("ZDMA transfer did not complete tag=%s src=0x%llx dst=0x%llx bytes=%zu",
             tag ? tag : "?",
             (unsigned long long) src_phys,
             (unsigned long long) dst_phys,
             bytes);
        return false;
    }

    const long long t1 = now_us();
    const uint32_t total_bytes_after_transfer = g_dma->ZDMA_CH_TOTAL_BYTE;
    const uint32_t post_vpu_status = vpu_is_mapped() ? vpu_rd32(REG_STATUS) : 0U;
    const uint32_t post_vpu_progress = vpu_is_mapped() ? vpu_rd32(REG_PROGRESS) : 0U;
    fpga_dma_trace_record(tag, src_phys, dst_phys, bytes, pre_status, pre_isr,
                          pre_ctrl2, total_bytes_before_clear,
                          pre_vpu_status, pre_vpu_progress, total_bytes_after_transfer,
                          post_vpu_status, post_vpu_progress,
                          t1 - t0, completion);
    const uint32_t status = completion.status;
    const uint32_t state = status & ZDMA_STATUS_STATE_MASK;
    const uint32_t isr = completion.isr;
    uint32_t dst_ddr_off = 0;
    if (phys_to_ddr_offset(dst_phys, bytes, &dst_ddr_off)) {
        if (!msync_ddr_range(dst_ddr_off, bytes, true, tag)) {
            return false;
        }
    }

    LOGDMA("tag=%s src=0x%llx dst=0x%llx bytes=%zu units=bytes ms=%.3f MiB/s=%.1f completion=isr_ack_then_dma_done_and_ctrl2_en_clear status=0x%08x state=%u isr=0x%08x",
           tag ? tag : "?",
           (unsigned long long) src_phys,
           (unsigned long long) dst_phys,
           bytes,
           (double) (t1 - t0) / 1000.0,
           (t1 > t0) ? (double) bytes * 1000000.0 / ((double) (t1 - t0) * 1024.0 * 1024.0) : 0.0,
           status,
           state,
           isr);
    return true;
}

static bool fpga_dma_copy(uint64_t src_phys, uint64_t dst_phys, size_t bytes, const char * tag);

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

// A CPU barrier orders the PS stores, but does not create a read transaction
// through the VPU AXI slave.  The C0 forensic replay, which performs register
// readbacks between ACT/WEIGHT DMA and CTRL_START, is correct for the exact
// tile that fails in the fast normal path.  Use the same non-destructive
// readback fence in production sequencing; it is two register reads, not a
// timer delay and not a DDR cache operation.
static void fpga_ip_dma_readback_fence(void) {
    mmio_fence();
    if (vpu_is_mapped()) {
        const uint32_t status = vpu_rd32(REG_STATUS);
        const uint32_t progress = vpu_rd32(REG_PROGRESS);
        (void) status;
        (void) progress;
    }
    mmio_fence();
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

// Submit a long linear copy as ordered, non-overlapping descriptors.  The
// legacy bitstream cannot observe a VPU start until the caller has returned,
// so splitting a copy here does not expose a partially loaded ACT/WEIGHT
// window to the VPU.  It does, however, keep the ZDMA/IP interconnect away
// from the 512 KiB descriptor pattern implicated by the contract log.
static bool fpga_dma_copy(uint64_t src_phys, uint64_t dst_phys, size_t bytes, const char * tag) {
    if (bytes == 0U || bytes > UINT32_MAX) {
        LOGE("invalid ZDMA byte count tag=%s bytes=%zu", tag ? tag : "?", bytes);
        return false;
    }
    const bool ddr_to_ip =
        phys_range_fits(src_phys, bytes, DDR_BASE_PHYS, g_ddr_map_size) &&
        phys_range_fits(dst_phys, bytes, LMM_BASE_PHYS, g_vpu_map_size);
    const bool ip_to_ddr =
        phys_range_fits(src_phys, bytes, LMM_BASE_PHYS, g_vpu_map_size) &&
        phys_range_fits(dst_phys, bytes, DDR_BASE_PHYS, g_ddr_map_size);
    if (!ddr_to_ip && !ip_to_ddr) {
        LOGE("ZDMA physical range rejected tag=%s src=0x%llx dst=0x%llx bytes=%zu ddr=[0x%llx,+0x%zx) my_ip=[0x%llx,+0x%zx); require one bounded DDR<->MY_IP transfer",
             tag ? tag : "?",
             (unsigned long long) src_phys,
             (unsigned long long) dst_phys,
             bytes,
             (unsigned long long) DDR_BASE_PHYS,
             g_ddr_map_size,
             (unsigned long long) LMM_BASE_PHYS,
             g_vpu_map_size);
        return false;
    }
    if (g_zdma_max_transfer_bytes < 16U ||
        (g_zdma_max_transfer_bytes & 0xFU) != 0U) {
        LOGE("invalid ZDMA descriptor policy max_bytes=%zu; require a positive 16-byte multiple",
             g_zdma_max_transfer_bytes);
        return false;
    }

    const size_t chunk_bytes = std::min(bytes, g_zdma_max_transfer_bytes);
    const size_t chunk_count = bytes / chunk_bytes + (bytes % chunk_bytes != 0U ? 1U : 0U);
    for (size_t chunk = 0U; chunk < chunk_count; ++chunk) {
        const size_t offset = chunk * chunk_bytes;
        const size_t this_bytes = std::min(chunk_bytes, bytes - offset);
        char chunk_tag[48];
        if (chunk_count == 1U) {
            snprintf(chunk_tag, sizeof(chunk_tag), "%s", tag ? tag : "?");
        } else {
            snprintf(chunk_tag, sizeof(chunk_tag), "%s[%zu/%zu]",
                     tag ? tag : "?", chunk + 1U, chunk_count);
        }
        if (!fpga_dma_copy_one(src_phys + (uint64_t) offset,
                               dst_phys + (uint64_t) offset,
                               this_bytes,
                               chunk_tag)) {
            return false;
        }
    }
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

static bool quantize_activation_vector_to(
        const struct ggml_tensor * src1,
        int64_t m,
        int64_t k,
        block_q8_0_t * out,
        float * stored_scales,
        fpga_activation_quant_stats_t * stats,
        const char * consumer_tensor_name,
        int consumer_layer_id,
        int64_t * bad_block,
        int * bad_lane,
        float * bad_value) {
    const int64_t nb = k / VPU_QK8_0;
    const char * base = (const char *) src1->data + m * src1->nb[1];

    // The FPGA hook is entered before ggml-cpu converts src1 into its vec-dot
    // type.  Use the exact same architecture-selected Q8_0 converter here so
    // rounding and, critically, the FP16-stored block scale match the CPU
    // kernel.  A private quantizer or an FP32 "exact" scale creates a different
    // numerical backend and can accumulate hidden-state drift across layers.
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * block_base = (const float *) (base + ib * VPU_QK8_0 * (int64_t) sizeof(float));
        float block_max_abs = 0.0f;
        for (int lane = 0; lane < VPU_QK8_0; ++lane) {
            const float value = block_base[lane];
            if (!std::isfinite(value)) {
                long long nan_count = 0;
                long long inf_count = 0;
                long long finite_count = 0;
                int64_t first_nonfinite = -1;
                float finite_min = INFINITY;
                float finite_max = -INFINITY;
                for (int64_t i = 0; i < k; ++i) {
                    const float probe = *(const float *) (base + i * (int64_t) sizeof(float));
                    if (std::isnan(probe)) {
                        if (first_nonfinite < 0) {
                            first_nonfinite = i;
                        }
                        nan_count++;
                    } else if (std::isinf(probe)) {
                        if (first_nonfinite < 0) {
                            first_nonfinite = i;
                        }
                        inf_count++;
                    } else {
                        finite_count++;
                        finite_min = std::min(finite_min, probe);
                        finite_max = std::max(finite_max, probe);
                    }
                }
                if (finite_count == 0) {
                    finite_min = NAN;
                    finite_max = NAN;
                }
                LOGE("ACTIVATION_NONFINITE_DETAIL consumer=%s consumer_layer=%d source=%s col=%lld first_index=%lld first_block=%lld first_lane=%d value=%.9g nan_count=%lld inf_count=%lld finite_count=%lld finite_min=%.9g finite_max=%.9g src1_type=%d src1_ne=[%lld,%lld,%lld,%lld] src1_nb=[%lld,%lld,%lld,%lld]",
                     consumer_tensor_name ? consumer_tensor_name : "?",
                     consumer_layer_id,
                     tensor_name_or_unknown(src1),
                     (long long) m,
                     (long long) first_nonfinite,
                     (long long) ib,
                     lane,
                     value,
                     nan_count,
                     inf_count,
                     finite_count,
                     finite_min,
                     finite_max,
                     (int) src1->type,
                     (long long) src1->ne[0], (long long) src1->ne[1],
                     (long long) src1->ne[2], (long long) src1->ne[3],
                     (long long) src1->nb[0], (long long) src1->nb[1],
                     (long long) src1->nb[2], (long long) src1->nb[3]);
                if (bad_block) {
                    *bad_block = ib;
                }
                if (bad_lane) {
                    *bad_lane = lane;
                }
                if (bad_value) {
                    *bad_value = value;
                }
                return false;
            }
            block_max_abs = std::max(block_max_abs, std::fabs(value));
        }

        if (stats) {
            stats->max_abs = std::max(stats->max_abs, block_max_abs);
            const float raw_scale = block_max_abs / 127.0f;
            stats->max_scale = std::max(stats->max_scale, raw_scale);
            if (raw_scale > VPU_FP16_MAX_FINITE) {
                if (stats->fp16_scale_overflows == 0) {
                    stats->first_overflow_col = m;
                    stats->first_overflow_block = ib;
                    stats->first_overflow_abs = block_max_abs;
                    stats->first_overflow_scale = raw_scale;
                }
                stats->fp16_scale_overflows++;
            }
        }
    }

    static_assert(sizeof(block_q8_0_t) == sizeof(block_q8_0), "FPGA/GGML Q8_0 layout mismatch");
    quantize_row_q8_0((const float *) base, out, k);
    if (stored_scales) {
        for (int64_t ib = 0; ib < nb; ++ib) {
            stored_scales[(size_t) ib] = fp16_to_fp32(out[(size_t) ib].d);
        }
    }
    return true;
}

static bool ensure_quantized_activation_matrix(
        const struct ggml_tensor * src1,
        int64_t m,
        int64_t k,
        std::vector<block_q8_0_t> & act_blocks_all,
        std::vector<float> & act_scales,
        bool store_act_scales,
        fpga_stage_totals_t * totals,
        const char * tensor_name,
        int layer_id) {
    const int64_t nb = k / VPU_QK8_0;
    if (nb <= 0 || (uint64_t) m > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) nb) {
        LOGE("activation quantization allocation overflow tensor=%s layer=%d K=%lld M=%lld blocks_per_col=%lld",
             tensor_name ? tensor_name : "?", layer_id,
             (long long) k, (long long) m, (long long) nb);
        return false;
    }
    const size_t total_blocks = (size_t) m * (size_t) nb;
    const bool cache_hit =
        g_activation_cache_enabled &&
        g_scratch.activation_cache_valid &&
        g_scratch.cached_src1 == src1 &&
        g_scratch.cached_src1_data == src1->data &&
        g_scratch.cached_m == m &&
        g_scratch.cached_k == k &&
        g_scratch.cached_nb0 == src1->nb[0] &&
        g_scratch.cached_nb1 == src1->nb[1] &&
        (!store_act_scales || act_scales.size() == total_blocks);

    if (cache_hit) {
        g_activation_cache_hits++;
        return true;
    }

    act_blocks_all.resize(total_blocks);
    if (store_act_scales) {
        act_scales.resize(total_blocks);
    } else {
        act_scales.clear();
    }
    fpga_activation_quant_stats_t stats = {};
    stats.first_overflow_col = -1;
    stats.first_overflow_block = -1;
    for (int64_t col = 0; col < m; ++col) {
        block_q8_0_t * col_blocks = &act_blocks_all[(size_t) col * (size_t) nb];
        int64_t bad_block = -1;
        int bad_lane = -1;
        float bad_value = 0.0f;
        float * stored_scales = store_act_scales ?
            &act_scales[(size_t) col * (size_t) nb] : nullptr;
        if (!quantize_activation_vector_to(src1, col, k, col_blocks, stored_scales,
                                           &stats, tensor_name, layer_id,
                                           &bad_block, &bad_lane, &bad_value)) {
            LOGE("ACTIVATION_NONFINITE tensor=%s layer=%d col=%lld block=%lld lane=%d value=%.9g; refusing to quantize invalid F32 activation",
                 tensor_name ? tensor_name : "?",
                 layer_id,
                 (long long) col,
                 (long long) bad_block,
                 bad_lane,
                 bad_value);
            return false;
        }
    }

    if (stats.fp16_scale_overflows > 0) {
        g_activation_scale_fp16_overflows += stats.fp16_scale_overflows;
        if (totals) {
            totals->activation_scale_fp16_overflows += stats.fp16_scale_overflows;
        }
        LOGE("ACTIVATION_SCALE_FP16_OVERFLOW tensor=%s layer=%d count=%lld first_col=%lld first_block=%lld first_amax=%.9g first_scale=%.9g max_amax=%.9g max_scale=%.9g; GGML Q8_0 stores d as FP16, so FPGA execution must stop instead of substituting an FP32-only scale",
             tensor_name ? tensor_name : "?",
             layer_id,
             stats.fp16_scale_overflows,
             (long long) stats.first_overflow_col,
             (long long) stats.first_overflow_block,
             stats.first_overflow_abs,
             stats.first_overflow_scale,
             stats.max_abs,
             stats.max_scale);
        return false;
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
    return true;
}

static const block_q8_0_t * weight_block_from_base(
        const struct ggml_tensor * src0,
        const void * data_base,
        int64_t row,
        int64_t block) {
    const char * row_base = (const char *) data_base + row * src0->nb[1];
    return (const block_q8_0_t *) row_base + block;
}

static const block_q8_0_t * weight_block(
        const struct ggml_tensor * src0,
        int64_t row,
        int64_t block) {
    return weight_block_from_base(src0, src0->data, row, block);
}

static bool checked_size_add(size_t lhs, size_t rhs, size_t * out) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

static bool checked_size_mul(size_t lhs, size_t rhs, size_t * out) {
    if (lhs != 0U && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

typedef struct {
    uintptr_t begin;
    uintptr_t end;
    size_t    bytes;
} fpga_tensor_access_range_t;

// The FPGA hook owns the complete MUL_MAT result, so a wrong tensor stride
// would otherwise corrupt the CPU allocator and only surface later as a
// misleading "double free or corruption" during shutdown.  Calculate the
// exact byte interval that the host will read/write before the first VPU
// launch; this also lets the host reject an output that aliases a live F32
// activation tensor.
static bool fpga_tensor_access_range(
        const struct ggml_tensor * tensor,
        int64_t dim0,
        int64_t dim1,
        size_t element_bytes,
        fpga_tensor_access_range_t * range) {
    if (!tensor || !tensor->data || dim0 <= 0 || dim1 <= 0) {
        return false;
    }
    if ((uint64_t) dim0 > (uint64_t) std::numeric_limits<size_t>::max() ||
        (uint64_t) dim1 > (uint64_t) std::numeric_limits<size_t>::max()) {
        return false;
    }

    size_t dim0_offset = 0U;
    size_t dim1_offset = 0U;
    size_t last_offset = 0U;
    size_t required_bytes = 0U;
    if (!checked_size_mul((size_t) (dim0 - 1), tensor->nb[0], &dim0_offset) ||
        !checked_size_mul((size_t) (dim1 - 1), tensor->nb[1], &dim1_offset) ||
        !checked_size_add(dim0_offset, dim1_offset, &last_offset) ||
        !checked_size_add(last_offset, element_bytes, &required_bytes)) {
        return false;
    }

    const uintptr_t begin = (uintptr_t) tensor->data;
    if (begin > std::numeric_limits<uintptr_t>::max() - required_bytes) {
        return false;
    }
    if (range) {
        range->begin = begin;
        range->end = begin + required_bytes;
        range->bytes = required_bytes;
    }
    return true;
}

static bool fpga_tensor_ranges_overlap(
        const fpga_tensor_access_range_t & lhs,
        const fpga_tensor_access_range_t & rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

static bool fpga_capture_activation_input_snapshot(
        const struct ggml_tensor * src1,
        int64_t k,
        int64_t m,
        std::vector<uint8_t> & snapshot) {
    size_t column_bytes = 0U;
    size_t total_bytes = 0U;
    if (!src1 || !src1->data ||
        !checked_size_mul((size_t) k, sizeof(float), &column_bytes) ||
        !checked_size_mul((size_t) m, column_bytes, &total_bytes)) {
        return false;
    }

    snapshot.resize(total_bytes);
    const char * const base = (const char *) src1->data;
    for (int64_t col = 0; col < m; ++col) {
        memcpy(snapshot.data() + (size_t) col * column_bytes,
               base + (size_t) col * src1->nb[1], column_bytes);
    }
    return true;
}

static bool fpga_verify_activation_input_snapshot(
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst,
        int64_t k,
        int64_t m,
        const std::vector<uint8_t> & snapshot,
        const char * consumer_tensor_name,
        int consumer_layer_id) {
    size_t column_bytes = 0U;
    size_t expected_bytes = 0U;
    if (!src1 || !dst || !src1->data ||
        !checked_size_mul((size_t) k, sizeof(float), &column_bytes) ||
        !checked_size_mul((size_t) m, column_bytes, &expected_bytes) ||
        snapshot.size() != expected_bytes) {
        LOGE("FPGA_INPUT_INTEGRITY_INTERNAL_ERROR tensor=%s layer=%d reason=invalid_snapshot_shape K=%lld M=%lld snapshot_bytes=%zu",
             consumer_tensor_name ? consumer_tensor_name : "?", consumer_layer_id,
             (long long) k, (long long) m, snapshot.size());
        return false;
    }

    const char * const base = (const char *) src1->data;
    for (int64_t col = 0; col < m; ++col) {
        const uint8_t * const expected = snapshot.data() + (size_t) col * column_bytes;
        const uint8_t * const actual = (const uint8_t *) base + (size_t) col * src1->nb[1];
        if (memcmp(expected, actual, column_bytes) == 0) {
            continue;
        }

        size_t first_byte = 0U;
        while (first_byte < column_bytes && expected[first_byte] == actual[first_byte]) {
            ++first_byte;
        }
        const size_t first_index = first_byte / sizeof(float);
        uint32_t expected_bits = 0U;
        uint32_t actual_bits = 0U;
        if (first_index < (size_t) k) {
            memcpy(&expected_bits, expected + first_index * sizeof(float), sizeof(expected_bits));
            memcpy(&actual_bits, actual + first_index * sizeof(float), sizeof(actual_bits));
        }
        fpga_tensor_access_range_t src_range = {};
        fpga_tensor_access_range_t dst_range = {};
        const bool src_range_ok = fpga_tensor_access_range(src1, k, m, sizeof(float), &src_range);
        const bool dst_range_ok = fpga_tensor_access_range(dst, dst->ne[0], dst->ne[1], sizeof(float), &dst_range);
        LOGE("FPGA_INPUT_MUTATION tensor=%s layer=%d source=%s col=%lld index=%zu byte=%zu expected_bits=0x%08x actual_bits=0x%08x src=[0x%llx,0x%llx) dst=[0x%llx,0x%llx) ranges_overlap=%d; raw FPGA matmul modified its F32 input",
             consumer_tensor_name ? consumer_tensor_name : "?", consumer_layer_id,
             tensor_name_or_unknown(src1), (long long) col, first_index, first_byte,
             expected_bits, actual_bits,
             (unsigned long long) (src_range_ok ? src_range.begin : 0U),
             (unsigned long long) (src_range_ok ? src_range.end : 0U),
             (unsigned long long) (dst_range_ok ? dst_range.begin : 0U),
             (unsigned long long) (dst_range_ok ? dst_range.end : 0U),
             src_range_ok && dst_range_ok && fpga_tensor_ranges_overlap(src_range, dst_range) ? 1 : 0);
        return false;
    }
    return true;
}

static void store_dst_value(
        const struct ggml_tensor * dst,
        int64_t row,
        int64_t col,
        float value) {
    char * base = (char *) dst->data;
    const size_t offset = (size_t) row * dst->nb[0] + (size_t) col * dst->nb[1];
    memcpy(base + offset, &value, sizeof(value));
}

static void write_i8x16_to_ddr(uint32_t off, const int8_t * lanes) {
    ddr_write_i8x16(off, lanes);
}

static void read_result_i32x4_from_ddr(uint32_t result_word, int32_t out[4]) {
    ddr_read_i32x4(RESULT_BASE + result_word * 16U, out);
}

// When a supposedly immutable GGUF Q8 block is invalid, distinguish an
// invalid file from a process-memory mutation.  A C0 failure at this point is
// before VPU launch, so this evidence must come from the host address space,
// not from ZDMA or PMAU.  This helper runs only on an error path.
typedef struct {
    bool      found;
    uintptr_t start;
    uintptr_t end;
    uint64_t  file_offset;
    char      perms[5];
    char      path[768];
} fpga_proc_map_info_t;

static void fpga_trim_leading_space(char * value) {
    if (!value) {
        return;
    }
    char * first = value;
    while (*first == ' ' || *first == '\t') {
        ++first;
    }
    if (first != value) {
        memmove(value, first, strlen(first) + 1U);
    }
}

static bool fpga_find_process_mapping(uintptr_t address, fpga_proc_map_info_t * info) {
    if (!info) {
        return false;
    }
    *info = {};

    FILE * const maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        return false;
    }

    char line[1024] = {};
    while (fgets(line, sizeof(line), maps)) {
        unsigned long long start = 0U;
        unsigned long long end = 0U;
        unsigned long long file_offset = 0U;
        unsigned long long inode = 0U;
        char perms[5] = {};
        char dev[32] = {};
        char path[768] = {};
        const int fields = sscanf(line, "%llx-%llx %4s %llx %31s %llu %767[^\n]",
                                  &start, &end, perms, &file_offset, dev, &inode, path);
        if (fields < 6 || address < (uintptr_t) start || address >= (uintptr_t) end) {
            continue;
        }

        info->found = true;
        info->start = (uintptr_t) start;
        info->end = (uintptr_t) end;
        info->file_offset = (uint64_t) file_offset;
        snprintf(info->perms, sizeof(info->perms), "%s", perms);
        if (fields >= 7) {
            fpga_trim_leading_space(path);
            snprintf(info->path, sizeof(info->path), "%s", path);
        }
        fclose(maps);
        return true;
    }

    fclose(maps);
    return false;
}

static void fpga_log_source_file_provenance(const void * source, size_t bytes) {
    constexpr size_t MAX_PROBE_BYTES = sizeof(block_q8_0_t);
    if (!source || bytes == 0U) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=%p bytes=%zu result=invalid_request",
             source, bytes);
        return;
    }

    fpga_proc_map_info_t map = {};
    const uintptr_t address = (uintptr_t) source;
    if (!fpga_find_process_mapping(address, &map)) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu result=map_not_found errno=%d (%s)",
             (unsigned long long) address, bytes, errno, strerror(errno));
        return;
    }

    const size_t bytes_in_mapping = (size_t) (map.end - address);
    const size_t probe_bytes = std::min(std::min(bytes, MAX_PROBE_BYTES), bytes_in_mapping);
    if (probe_bytes == 0U) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu map=[0x%llx,0x%llx) result=empty_map_probe",
             (unsigned long long) address, bytes,
             (unsigned long long) map.start, (unsigned long long) map.end);
        return;
    }
    const bool file_backed = map.path[0] == '/';
    if (!file_backed) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu map=[0x%llx,0x%llx) perms=%s file_offset=0x%llx path=%s result=not_file_backed",
             (unsigned long long) address, bytes,
             (unsigned long long) map.start, (unsigned long long) map.end,
             map.perms, (unsigned long long) map.file_offset,
             map.path[0] ? map.path : "[anonymous]");
        return;
    }

    const int fd = open(map.path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu map=[0x%llx,0x%llx) perms=%s file_offset=0x%llx path=%s result=file_open_failed errno=%d (%s)",
             (unsigned long long) address, bytes,
             (unsigned long long) map.start, (unsigned long long) map.end,
             map.perms, (unsigned long long) map.file_offset, map.path,
             errno, strerror(errno));
        return;
    }

    uint8_t file_bytes[MAX_PROBE_BYTES] = {};
    const uint64_t mapped_file_offset = map.file_offset + (uint64_t) (address - map.start);
    const ssize_t read_bytes = pread(fd, file_bytes, probe_bytes, (off_t) mapped_file_offset);
    const int saved_errno = errno;
    close(fd);

    const bool complete_read = read_bytes == (ssize_t) probe_bytes;
    const bool source_matches_file = complete_read &&
        memcmp(file_bytes, source, probe_bytes) == 0;
    LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu probe_bytes=%zu map=[0x%llx,0x%llx) perms=%s map_file_offset=0x%llx source_file_offset=0x%llx path=%s pread=%zd source_matches_file=%d file_bytes=[%02x,%02x,%02x,%02x] source_bytes=[%02x,%02x,%02x,%02x]%s",
         (unsigned long long) address, bytes, probe_bytes,
         (unsigned long long) map.start, (unsigned long long) map.end,
         map.perms, (unsigned long long) map.file_offset,
         (unsigned long long) mapped_file_offset, map.path, read_bytes,
         source_matches_file ? 1 : 0,
         file_bytes[0], file_bytes[1], file_bytes[2], file_bytes[3],
         ((const uint8_t *) source)[0], ((const uint8_t *) source)[1],
         ((const uint8_t *) source)[2], ((const uint8_t *) source)[3],
         complete_read ? "" : strerror(saved_errno));
}

static float load_dst_value(
        const struct ggml_tensor * dst,
        int64_t row,
        int64_t col) {
    const char * base = (const char *) dst->data;
    const size_t offset = (size_t) row * dst->nb[0] + (size_t) col * dst->nb[1];
    float value = 0.0f;
    memcpy(&value, base + offset, sizeof(value));
    return value;
}

static int32_t q8_0_raw_dot(const int8_t * a, const int8_t * w) {
    int32_t acc = 0;
    for (int i = 0; i < VPU_QK8_0; ++i) {
        acc += (int32_t) a[i] * (int32_t) w[i];
    }
    return acc;
}

// A Q8_0 x Q8_0 dot can become NaN only when at least one FP16 block scale is
// non-finite (the signed int8 raw dot and its finite FP32 product cannot
// overflow at this geometry).  Validate the immutable snapshot before any
// ZDMA/VPU work so a bad scale is reported at its source instead of later as a
// vague destination-value failure.
static bool fpga_contract_validate_weight_scales(
        const struct ggml_tensor * src0,
        const void * weight_data_base,
        const char * tensor_name,
        int layer_id) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t nb = k / VPU_QK8_0;
    long long zero_scales = 0;
    float min_abs_scale = INFINITY;
    float max_abs_scale = 0.0f;

    for (int64_t row = 0; row < n; ++row) {
        for (int64_t block = 0; block < nb; ++block) {
            const block_q8_0_t * const snapshot =
                weight_block_from_base(src0, weight_data_base, row, block);
            const float scale = fp16_to_fp32(snapshot->d);
            if (!std::isfinite(scale)) {
                const block_q8_0_t * const live = weight_block(src0, row, block);
                const bool live_matches_snapshot =
                    memcmp(live, snapshot, sizeof(*snapshot)) == 0;
                block_q8_0_t read_a = {};
                block_q8_0_t read_b = {};
                memcpy(&read_a, live, sizeof(read_a));
                std::atomic_thread_fence(std::memory_order_seq_cst);
                memcpy(&read_b, live, sizeof(read_b));
                const bool source_reads_stable =
                    memcmp(&read_a, &read_b, sizeof(read_a)) == 0;
                const bool upstream_row_valid =
                    ggml_validate_row_data(src0->type, src0->data, ggml_nbytes(src0));
                const size_t byte_offset =
                    (size_t) row * (size_t) src0->nb[1] +
                    (size_t) block * (size_t) src0->nb[0];
                LOGE("CONTRACT_WEIGHT_SCALE_NONFINITE tensor=%s layer=%d row=%lld block=%lld byte_offset=%zu d_bits=0x%04x scale=%.9g live_d_bits=0x%04x live_scale=%.9g live_matches_snapshot=%d source_reads_stable=%d upstream_q8_validate=%s src0_type=%d src0_nb=[%lld,%lld,%lld,%lld] snapshot_bytes=%zu raw_bytes=[%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x] qs_first8=[%d,%d,%d,%d,%d,%d,%d,%d]; refusing VPU launch because finite raw dots cannot yield a finite F32 result",
                     tensor_name ? tensor_name : "?",
                     layer_id,
                     (long long) row,
                     (long long) block,
                     byte_offset,
                     (unsigned) snapshot->d,
                     scale,
                     (unsigned) live->d,
                     fp16_to_fp32(live->d),
                     live_matches_snapshot ? 1 : 0,
                     source_reads_stable ? 1 : 0,
                     upstream_row_valid ? "pass" : "fail",
                     (int) src0->type,
                     (long long) src0->nb[0], (long long) src0->nb[1],
                     (long long) src0->nb[2], (long long) src0->nb[3],
                     ggml_nbytes(src0),
                     ((const uint8_t *) snapshot)[0], ((const uint8_t *) snapshot)[1],
                     ((const uint8_t *) snapshot)[2], ((const uint8_t *) snapshot)[3],
                     ((const uint8_t *) snapshot)[4], ((const uint8_t *) snapshot)[5],
                     ((const uint8_t *) snapshot)[6], ((const uint8_t *) snapshot)[7],
                     ((const uint8_t *) snapshot)[8], ((const uint8_t *) snapshot)[9],
                     (int) snapshot->qs[0], (int) snapshot->qs[1],
                     (int) snapshot->qs[2], (int) snapshot->qs[3],
                     (int) snapshot->qs[4], (int) snapshot->qs[5],
                     (int) snapshot->qs[6], (int) snapshot->qs[7]);
                fpga_log_source_file_provenance(live, sizeof(*live));
                return false;
            }
            const float abs_scale = std::fabs(scale);
            min_abs_scale = std::min(min_abs_scale, abs_scale);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
            if (scale == 0.0f) {
                zero_scales++;
            }
        }
    }

    if (!std::isfinite(min_abs_scale)) {
        min_abs_scale = 0.0f;
    }
    LOGI("CONTRACT_WEIGHT_SCALE_AUDIT tensor=%s layer=%d blocks=%lld zero_scales=%lld min_abs=%.9g max_abs=%.9g snapshot_bytes=%zu result=pass",
         tensor_name ? tensor_name : "?",
         layer_id,
         (long long) (n * nb),
         zero_scales,
         min_abs_scale,
         max_abs_scale,
         ggml_nbytes(src0));
    return true;
}

static bool fpga_audit_q8_source_only(
        const struct ggml_tensor * src0,
        const char * tensor_name,
        int layer_id) {
    if (!src0 || src0->type != GGML_TYPE_Q8_0 || !src0->data) {
        return true;
    }

    g_q8_source_audit_checks++;
    const bool valid = fpga_contract_validate_weight_scales(
        src0, src0->data, tensor_name, layer_id);
    if (!valid) {
        g_q8_source_audit_failures++;
        LOGE("Q8_SOURCE_AUDIT_FAIL tensor=%s layer=%d check=%lld action=stop_before_model_zdma_vpu_gemv; this run did not submit this tensor to ZDMA/VPU",
             tensor_name ? tensor_name : "?", layer_id,
             g_q8_source_audit_checks);
        return false;
    }

    LOGI("Q8_SOURCE_AUDIT_PASS tensor=%s layer=%d check=%lld action=cpu_matmul_only",
         tensor_name ? tensor_name : "?", layer_id,
         g_q8_source_audit_checks);
    return true;
}

static void fpga_contract_log_q8_nonfinite_provenance(
        const struct ggml_tensor * src0,
        const block_q8_0_t * weight,
        const block_q8_0_t * act,
        int64_t row,
        int64_t col,
        const char * tensor_name,
        int layer_id,
        float kernel_reference) {
    const int64_t nb = src0->ne[0] / VPU_QK8_0;
    float scalar_reference = 0.0f;
    int64_t first_bad_block = -1;
    int32_t first_bad_raw = 0;
    float first_bad_act_scale = 0.0f;
    float first_bad_weight_scale = 0.0f;
    float first_bad_term = 0.0f;
    const char * first_bad_kind = "scalar_accumulator";

    for (int64_t block = 0; block < nb; ++block) {
        const float act_scale = fp16_to_fp32(act[block].d);
        const float weight_scale = fp16_to_fp32(weight[block].d);
        const int32_t raw = q8_0_raw_dot(act[block].qs, weight[block].qs);
        const float term = (float) raw * act_scale * weight_scale;
        if (first_bad_block < 0 &&
            (!std::isfinite(act_scale) || !std::isfinite(weight_scale) ||
             !std::isfinite(term) || !std::isfinite(scalar_reference + term))) {
            first_bad_block = block;
            first_bad_raw = raw;
            first_bad_act_scale = act_scale;
            first_bad_weight_scale = weight_scale;
            first_bad_term = term;
            if (!std::isfinite(act_scale)) {
                first_bad_kind = "activation_scale";
            } else if (!std::isfinite(weight_scale)) {
                first_bad_kind = "weight_scale";
            } else if (!std::isfinite(term)) {
                first_bad_kind = "scaled_term";
            }
        }
        scalar_reference += term;
    }

    LOGE("CONTRACT_Q8_NONFINITE_PROVENANCE tensor=%s layer=%d row=%lld col=%lld kernel_reference=%.9g scalar_reference=%.9g first_bad_kind=%s first_bad_block=%lld raw=%d act_d_bits=0x%04x act_scale=%.9g weight_d_bits=0x%04x weight_scale=%.9g term=%.9g",
         tensor_name ? tensor_name : "?",
         layer_id,
         (long long) row,
         (long long) col,
         kernel_reference,
         scalar_reference,
         first_bad_kind,
         (long long) first_bad_block,
         first_bad_raw,
         first_bad_block >= 0 ? (unsigned) act[first_bad_block].d : 0U,
         first_bad_act_scale,
         first_bad_block >= 0 ? (unsigned) weight[first_bad_block].d : 0U,
         first_bad_weight_scale,
         first_bad_term);
}

// Stage exactly the packed Q8 payload consumed by one legacy VPU launch.
// Keeping this in one helper is intentional: normal launch and forensic
// replay must write byte-for-byte identical ACT/WEIGHT layouts.
static void fpga_stage_q8_group_payload(
        const block_q8_0_t * weight_snapshot,
        const block_q8_0_t * act_group,
        int rows,
        int group_blocks,
        bool write_weight_payload,
        uint32_t weight_dst_off) {
    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    if (write_weight_payload) {
        for (int row = 0; row < rows; ++row) {
            for (int gb = 0; gb < group_blocks; ++gb) {
                const block_q8_0_t * wb =
                    &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
                for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                    const uint32_t word_index = (uint32_t) row * (uint32_t) group_beats +
                                                (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS +
                                                (uint32_t) beat;
                    write_i8x16_to_ddr(weight_dst_off + word_index * 16U,
                                       wb->qs + beat * VPU_NUM_LANES);
                }
            }
        }
    }

    for (int gb = 0; gb < group_blocks; ++gb) {
        const block_q8_0_t & act = act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS +
                                        (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U,
                               act.qs + beat * VPU_NUM_LANES);
        }
    }
    mmio_fence();
}

// ddr_high is exposed through UIO/O_SYNC, but this board reports EINVAL for
// msync().  A DSB followed by reads from both ends of the written range drains
// posted PS stores before ZDMA becomes the reader.  The full C0 audit below is
// stronger; this bounded fence remains enabled in normal execution.
static void fpga_ddr_staging_readback_commit(uint32_t off, size_t bytes) {
    if (bytes == 0U || (off & 0x3U) != 0U || (bytes & 0x3U) != 0U) {
        fpga_fatal("DDR staging commit requires a non-empty 32-bit range off=0x%08x bytes=%zu",
                   off, bytes);
    }
    mmio_fence();
    volatile const uint32_t * const first =
        (volatile const uint32_t *) ddr_ptr(off, sizeof(uint32_t));
    volatile const uint32_t * const last =
        (volatile const uint32_t *) ddr_ptr(off + (uint32_t) bytes - sizeof(uint32_t),
                                            sizeof(uint32_t));
    const uint32_t first_value = *first;
    const uint32_t last_value = *last;
    (void) first_value;
    (void) last_value;
    mmio_fence();
}

static bool fpga_contract_verify_staged_q8_group(
        const block_q8_0_t * weight_snapshot,
        const block_q8_0_t * act_group,
        int rows,
        int group_blocks,
        uint32_t weight_src_off,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id,
        const char * phase) {
    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    for (int gb = 0; gb < group_blocks; ++gb) {
        const volatile int8_t * const staged =
            (volatile const int8_t *) ddr_ptr(
                ACT_BASE + (uint32_t) gb * VPU_QK8_0, VPU_QK8_0);
        for (int lane = 0; lane < VPU_QK8_0; ++lane) {
            if (staged[lane] != act_group[gb].qs[lane]) {
                LOGE("CONTRACT_STAGING_BOUNDARY_FAIL phase=%s kind=ACT tensor=%s layer=%d tile=%u block=%d lane=%d expected=%d actual=%d",
                     phase ? phase : "?", tensor_name ? tensor_name : "?", layer_id, tile_id, gb, lane,
                     (int) act_group[gb].qs[lane], (int) staged[lane]);
                return false;
            }
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t & expected =
                weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
            const uint32_t off = weight_src_off +
                ((uint32_t) row * (uint32_t) group_beats +
                 (uint32_t) gb * VPU_BLOCK_BEATS) * 16U;
            const volatile int8_t * const staged =
                (volatile const int8_t *) ddr_ptr(off, VPU_QK8_0);
            for (int lane = 0; lane < VPU_QK8_0; ++lane) {
                if (staged[lane] != expected.qs[lane]) {
                    LOGE("CONTRACT_STAGING_BOUNDARY_FAIL phase=%s kind=WEIGHT tensor=%s layer=%d tile=%u row=%d block=%d lane=%d off=0x%08x expected=%d actual=%d",
                         phase ? phase : "?", tensor_name ? tensor_name : "?", layer_id, tile_id, row, gb,
                         lane, off + (uint32_t) lane, (int) expected.qs[lane],
                         (int) staged[lane]);
                    return false;
                }
            }
        }
    }
    mmio_fence();
    return true;
}

// The current FPD ZDMA path has no hardware coherency guarantee for its data
// transactions.  The v34 failure was two adjacent 32-byte Q8 blocks in one
// stale DDR cache line; the exact tile passed only after the forensic replay
// performed a full staged-byte readback.  In C0/C1, make that proof part of
// the normal launch sequence and allow one pre-VPU re-stage.  This never
// repairs FPGA results or changes a production tensor: it only prevents a
// known-invalid DDR source from reaching the VPU during a contract run.
static bool fpga_stage_q8_group_with_contract_guard(
        const block_q8_0_t * weight_snapshot,
        const block_q8_0_t * act_group,
        int rows,
        int group_blocks,
        bool write_weight_payload,
        uint32_t weight_src_off,
        size_t act_bytes,
        size_t weight_bytes,
        bool guard_enabled,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id) {
    const int attempts = guard_enabled ? 2 : 1;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        fpga_stage_q8_group_payload(weight_snapshot, act_group, rows,
                                    group_blocks, write_weight_payload, weight_src_off);
        fpga_ddr_staging_readback_commit(ACT_BASE, act_bytes);
        fpga_ddr_staging_readback_commit(weight_src_off, weight_bytes);

        if (!guard_enabled || fpga_contract_verify_staged_q8_group(
                weight_snapshot, act_group, rows, group_blocks,
                weight_src_off, tensor_name, layer_id, tile_id,
                attempt == 0 ? "after_stage" : "after_restage")) {
            if (attempt > 0) {
                g_contract_staging_restage_count++;
                LOGI("CONTRACT_STAGING_RESTAGE_RECOVERED tensor=%s layer=%d tile=%u attempts=%d; the corrected source was verified before VPU start",
                     tensor_name ? tensor_name : "?", layer_id, tile_id, attempt + 1);
            }
            return true;
        }

        LOGE("CONTRACT_STAGING_RESTAGE tensor=%s layer=%d tile=%u attempt=%d reason=pre_vpu_ddr_source_mismatch",
             tensor_name ? tensor_name : "?", layer_id, tile_id, attempt + 1);
    }

    LOGE("CONTRACT_STAGING_RESTAGE_FAILED tensor=%s layer=%d tile=%u attempts=%d; refusing to launch VPU with an unverified ACT/WEIGHT DDR source",
         tensor_name ? tensor_name : "?", layer_id, tile_id, attempts);
    return false;
}

// A completed ACT DMA is a read from DDR_HIGH and must not modify the
// separately staged WEIGHT source.  v35 found a byte change in WEIGHT after
// ACT completed, before the WEIGHT DMA or VPU start.  Preserve that evidence,
// then re-stage once so a C0 run can continue to collect raw-contract data.
// A recovered staging fault is still a C0 failure for primary-FPGA admission:
// the cleanup summary records staging_restages and must remain zero.
static bool fpga_contract_restage_after_act_dma(
        const block_q8_0_t * weight_snapshot,
        const block_q8_0_t * act_group,
        int rows,
        int group_blocks,
        bool write_weight_payload,
        uint32_t weight_src_off,
        size_t act_bytes,
        size_t weight_bytes,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id) {
    const uint64_t act_src_begin = DDR_BASE_PHYS + (uint64_t) ACT_BASE;
    const uint64_t act_src_end = act_src_begin + (uint64_t) act_bytes;
    const uint64_t act_dst_begin = LMM_BASE_PHYS + (uint64_t) ACT_BASE;
    const uint64_t act_dst_end = act_dst_begin + (uint64_t) act_bytes;
    const uint64_t weight_src_begin = DDR_BASE_PHYS + (uint64_t) weight_src_off;
    const uint64_t weight_src_end = weight_src_begin + (uint64_t) weight_bytes;
    const bool source_ranges_disjoint =
        act_src_end <= weight_src_begin || weight_src_end <= act_src_begin;

    LOGE("CONTRACT_STAGING_ACT_DMA_CONTEXT tensor=%s layer=%d tile=%u act_src=[0x%llx,0x%llx) act_dst=[0x%llx,0x%llx) weight_src=[0x%llx,0x%llx) source_ranges_disjoint=%d write_weight_payload=%d",
         tensor_name ? tensor_name : "?", layer_id, tile_id,
         (unsigned long long) act_src_begin,
         (unsigned long long) act_src_end,
         (unsigned long long) act_dst_begin,
         (unsigned long long) act_dst_end,
         (unsigned long long) weight_src_begin,
         (unsigned long long) weight_src_end,
         source_ranges_disjoint ? 1 : 0,
         write_weight_payload ? 1 : 0);
    zdma_dump("contract_staging_changed_after_act_dma");
    fpga_dma_trace_dump("staging_changed_after_act_dma", tensor_name, layer_id,
                        tile_id, "ACT");

    if (!write_weight_payload) {
        LOGE("CONTRACT_STAGING_RESTAGE_FAILED tensor=%s layer=%d tile=%u reason=weight_cache_source_changed_after_act_dma",
             tensor_name ? tensor_name : "?", layer_id, tile_id);
        return false;
    }

    fpga_stage_q8_group_payload(weight_snapshot, act_group, rows, group_blocks,
                                true, weight_src_off);
    fpga_ddr_staging_readback_commit(ACT_BASE, act_bytes);
    fpga_ddr_staging_readback_commit(weight_src_off, weight_bytes);
    if (!fpga_contract_verify_staged_q8_group(
            weight_snapshot, act_group, rows, group_blocks, weight_src_off,
            tensor_name, layer_id, tile_id, "after_act_dma_restage")) {
        LOGE("CONTRACT_STAGING_RESTAGE_FAILED tensor=%s layer=%d tile=%u reason=post_restage_source_mismatch",
             tensor_name ? tensor_name : "?", layer_id, tile_id);
        return false;
    }

    g_contract_staging_restage_count++;
    LOGE("CONTRACT_STAGING_RESTAGE_RECOVERED tensor=%s layer=%d tile=%u phase=after_act_dma; raw contract will continue, but this run is ineligible for primary-FPGA admission",
         tensor_name ? tensor_name : "?", layer_id, tile_id);
    return true;
}

static bool fpga_contract_verify_weight_source_snapshot(
        const struct ggml_tensor * src0,
        const block_q8_0_t * weight_snapshot,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id) {
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * const live =
                weight_block(src0, row0 + row, k_block0 + gb);
            const block_q8_0_t * const snapshot =
                &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
            if (memcmp(live, snapshot, sizeof(*snapshot)) != 0) {
                int first_bad = 0;
                const uint8_t * const live_bytes = (const uint8_t *) live;
                const uint8_t * const snapshot_bytes = (const uint8_t *) snapshot;
                while (first_bad < (int) sizeof(*snapshot) &&
                       live_bytes[first_bad] == snapshot_bytes[first_bad]) {
                    first_bad++;
                }
                LOGE("CONTRACT_WEIGHT_SOURCE_MUTATION tensor=%s layer=%d tile=%u row=%lld block=%lld byte=%d snapshot=%u live=%u snapshot_d=0x%04x live_d=0x%04x; immutable GGUF weight changed during one VPU launch",
                     tensor_name ? tensor_name : "?", layer_id, tile_id,
                     (long long) (row0 + row), (long long) (k_block0 + gb),
                     first_bad,
                     first_bad < (int) sizeof(*snapshot) ? snapshot_bytes[first_bad] : 0U,
                     first_bad < (int) sizeof(*snapshot) ? live_bytes[first_bad] : 0U,
                     (unsigned) snapshot->d, (unsigned) live->d);
                return false;
            }
        }
    }
    return true;
}

static bool fpga_contract_log_staging_audit(
        const block_q8_0_t * act,
        const block_q8_0_t * weight,
        int local_row,
        int group_block,
        int group_beats,
        uint32_t weight_src_off,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id,
        const char * phase) {
    const uint32_t act_off = ACT_BASE + (uint32_t) group_block * VPU_BLOCK_BEATS * 16U;
    const uint32_t weight_off = weight_src_off +
        ((uint32_t) local_row * (uint32_t) group_beats +
         (uint32_t) group_block * VPU_BLOCK_BEATS) * 16U;
    const volatile int8_t * const staged_act =
        (volatile const int8_t *) ddr_ptr(act_off, VPU_QK8_0);
    const volatile int8_t * const staged_weight =
        (volatile const int8_t *) ddr_ptr(weight_off, VPU_QK8_0);
    int act_first_bad = -1;
    int weight_first_bad = -1;
    int act_expected = 0;
    int act_actual = 0;
    int weight_expected = 0;
    int weight_actual = 0;
    for (int i = 0; i < VPU_QK8_0; ++i) {
        const int got_act = (int) staged_act[i];
        const int got_weight = (int) staged_weight[i];
        if (act_first_bad < 0 && got_act != (int) act->qs[i]) {
            act_first_bad = i;
            act_expected = (int) act->qs[i];
            act_actual = got_act;
        }
        if (weight_first_bad < 0 && got_weight != (int) weight->qs[i]) {
            weight_first_bad = i;
            weight_expected = (int) weight->qs[i];
            weight_actual = got_weight;
        }
    }
    const bool intact = act_first_bad < 0 && weight_first_bad < 0;
    fpga_log_line(true, intact ? "INFO" : "ERROR", !intact,
         "CONTRACT_STAGING_AUDIT phase=%s integrity=%s tensor=%s layer=%d tile=%u local_row=%d group_block=%d group_beats=%d act_off=0x%08x weight_off=0x%08x weight_src_off=0x%08x act_first_bad=%d act_expected=%d act_actual=%d weight_first_bad=%d weight_expected=%d weight_actual=%d act_d_bits=0x%04x weight_d_bits=0x%04x status=0x%08x progress=0x%08x",
         phase ? phase : "post_result",
         intact ? "pass" : "fail",
         tensor_name ? tensor_name : "?",
         layer_id,
         tile_id,
         local_row,
         group_block,
         group_beats,
         act_off,
         weight_off,
         weight_src_off,
         act_first_bad,
         act_expected,
         act_actual,
         weight_first_bad,
         weight_expected,
         weight_actual,
         (unsigned) act->d,
         (unsigned) weight->d,
         vpu_rd32(REG_STATUS),
         vpu_rd32(REG_PROGRESS));
    return intact;
}

static long long fpga_contract_count_raw_mismatches(
        const block_q8_0_t * weight_snapshot,
        const block_q8_0_t * act_group,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        uint32_t weight_src_off,
        std::vector<int32_t> & partial,
        const char * tensor_name,
        int layer_id,
        uint32_t tile_id,
        int attempt,
        bool log_mismatches,
        bool repair_mismatches,
        fpga_raw_mismatch_location_t * first_mismatch) {
    long long mismatches = 0;
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * wb =
                &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
            const int32_t expected = q8_0_raw_dot(act_group[gb].qs, wb->qs);
            const size_t partial_idx = (size_t) row * (size_t) group_blocks + (size_t) gb;
            const int32_t got = partial[partial_idx];
            if (got != expected) {
                if (first_mismatch && !first_mismatch->valid) {
                    first_mismatch->valid = true;
                    first_mismatch->local_row = row;
                    first_mismatch->group_block = gb;
                    first_mismatch->global_row = row0 + row;
                    first_mismatch->k_block = k_block0 + gb;
                }
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
                    fpga_contract_log_staging_audit(
                        &act_group[gb], wb, row, gb, group_blocks * VPU_BLOCK_BEATS,
                        weight_src_off, tensor_name, layer_id, tile_id, "post_result");
                    if (mismatches == 0) {
                        fpga_dma_trace_dump("raw_mismatch", tensor_name, layer_id, tile_id, nullptr);
                    }
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

// A raw mismatch tells us that the model output is not trustworthy.  On an
// aborting contract run, replay only that one failing VPU job and inspect the
// same 32-byte activation/weight block after each boundary.  This is bounded
// to one tile (at most 294,912 bytes of re-staging on the current geometry),
// never builds a weight cache, and does not alter normal inference timing.
static void fpga_contract_forensic_replay(
        const block_q8_0_t * weight_snapshot,
        const block_q8_0_t * act_group,
        int rows,
        int group_blocks,
        uint32_t weight_src_off,
        bool weight_cache_hit,
        uint32_t tile_id,
        const char * tensor_name,
        int layer_id,
        const fpga_raw_mismatch_location_t & mismatch) {
    if (!mismatch.valid) {
        return;
    }

    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    const size_t act_bytes = (size_t) group_beats * 16U;
    const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
    const uint32_t result_values = (uint32_t) rows * (uint32_t) group_blocks;
    const uint32_t result_words =
        (result_values + (uint32_t) VPU_RESULT_PACK_LANES - 1U) /
        (uint32_t) VPU_RESULT_PACK_LANES;
    const size_t result_bytes = (size_t) result_words * 16U;
    const block_q8_0_t * const weight = &weight_snapshot[
        (size_t) mismatch.local_row * (size_t) group_blocks +
        (size_t) mismatch.group_block];
    const block_q8_0_t * const act = &act_group[mismatch.group_block];

    LOGE("CONTRACT_FORENSIC_BEGIN tensor=%s layer=%d tile=%u row=%lld local_row=%d block=%lld group_block=%d cache_hit=%d",
         tensor_name ? tensor_name : "?",
         layer_id,
         tile_id,
         (long long) mismatch.global_row,
         mismatch.local_row,
         (long long) mismatch.k_block,
         mismatch.group_block,
         weight_cache_hit ? 1 : 0);

    // Scratch weights are overwritten from the immutable GGUF source.  A
    // cache hit remains read-only by design; probing it still identifies a
    // corrupt cache payload without touching the large cache range.
    fpga_stage_q8_group_payload(weight_snapshot, act_group, rows,
                                group_blocks, !weight_cache_hit, weight_src_off);
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row,
                                    mismatch.group_block, group_beats,
                                    weight_src_off, tensor_name, layer_id,
                                    tile_id, "forensic_after_restage");

    vpu_select_banks(0, 0);
    vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
    configure_vpu(rows, group_beats, VPU_MODE_PACKED_Q8);

    if (!fpga_dma_write_to_ip(ACT_BASE, act_bytes, "FORENSIC_ACT")) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=act_dma tensor=%s layer=%d tile=%u",
             tensor_name ? tensor_name : "?", layer_id, tile_id);
        return;
    }
    fpga_ip_dma_readback_fence();
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row,
                                    mismatch.group_block, group_beats,
                                    weight_src_off, tensor_name, layer_id,
                                    tile_id, "forensic_after_act_dma");

    if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) weight_src_off,
                       LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                       weight_bytes, "FORENSIC_WEIGHT")) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=weight_dma tensor=%s layer=%d tile=%u",
             tensor_name ? tensor_name : "?", layer_id, tile_id);
        return;
    }
    fpga_ip_dma_readback_fence();
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row,
                                    mismatch.group_block, group_beats,
                                    weight_src_off, tensor_name, layer_id,
                                    tile_id, "forensic_after_weight_dma");

    vpu_wr32(REG_CTRL, CTRL_START);
    mmio_fence();
    uint32_t vpu_status = 0;
    if (!wait_vpu_done(&vpu_status)) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=vpu_wait tensor=%s layer=%d tile=%u status=0x%08x progress=0x%08x",
             tensor_name ? tensor_name : "?", layer_id, tile_id,
             vpu_status, vpu_rd32(REG_PROGRESS));
        return;
    }
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row,
                                    mismatch.group_block, group_beats,
                                    weight_src_off, tensor_name, layer_id,
                                    tile_id, "forensic_after_vpu");

    vpu_select_banks(0, 0);
    if (!fpga_dma_read_from_ip(RESULT_BASE, result_bytes, "FORENSIC_RESULT")) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=result_dma tensor=%s layer=%d tile=%u",
             tensor_name ? tensor_name : "?", layer_id, tile_id);
        return;
    }
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row,
                                    mismatch.group_block, group_beats,
                                    weight_src_off, tensor_name, layer_id,
                                    tile_id, "forensic_after_result_dma");

    const uint32_t raw_index =
        (uint32_t) mismatch.local_row * (uint32_t) group_blocks +
        (uint32_t) mismatch.group_block;
    int32_t lanes[VPU_RESULT_PACK_LANES] = {};
    read_result_i32x4_from_ddr(raw_index / (uint32_t) VPU_RESULT_PACK_LANES, lanes);
    const int32_t got = lanes[raw_index % (uint32_t) VPU_RESULT_PACK_LANES];
    const int32_t expected = q8_0_raw_dot(act->qs, weight->qs);
    LOGE("CONTRACT_FORENSIC_RAW tensor=%s layer=%d tile=%u row=%lld block=%lld got=%d expected=%d status=0x%08x progress=0x%08x",
         tensor_name ? tensor_name : "?", layer_id, tile_id,
         (long long) mismatch.global_row, (long long) mismatch.k_block,
         got, expected, vpu_status, vpu_rd32(REG_PROGRESS));
}

static bool fpga_contract_check_output_values(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst,
        const std::vector<block_q8_0_t> & act_blocks_all,
        const void * weight_data_base,
        const char * tensor_name,
        int layer_id,
        bool cpu_shadow_dst,
        double atol,
        double rtol,
        bool abort_on_mismatch,
        const char * contract_tag,
        long long * mismatch_total,
        long long * shadow_value_total) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = dst->ne[1];
    const int64_t nb = k / VPU_QK8_0;
    long long bad = 0;
    long long nonfinite = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    const size_t value_count = (size_t) n * (size_t) m;
    if (cpu_shadow_dst && g_scratch.contract_actual.size() != value_count) {
        LOGE("%s_CPU_SHADOW_LAYOUT tensor=%s layer=%d actual_values=%zu expected_values=%zu",
             contract_tag,
             tensor_name ? tensor_name : "?",
             layer_id,
             g_scratch.contract_actual.size(),
             value_count);
        return false;
    }

    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < n; ++row) {
            float ref = 0.0f;
            const block_q8_0_t * act = &act_blocks_all[(size_t) (col * nb)];
            const block_q8_0_t * weight =
                weight_block_from_base(src0, weight_data_base, row, 0);
            ggml_vec_dot_q8_0_q8_0((int) k, &ref, 0, weight, 0, act, 0, 1);
            const size_t value_index = (size_t) col * (size_t) n + (size_t) row;
            const double got = cpu_shadow_dst ?
                (double) g_scratch.contract_actual[value_index] :
                (double) load_dst_value(dst, row, col);
            const double expected = (double) ref;

            if (!std::isfinite(got) || !std::isfinite(expected)) {
                if (bad < 4) {
                    LOGE("%s_NONFINITE tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g; matching NaN/Inf is a correctness failure",
                         contract_tag,
                         tensor_name ? tensor_name : "?",
                         layer_id,
                         (long long) row,
                         (long long) col,
                         got,
                         expected);
                    fpga_contract_log_q8_nonfinite_provenance(
                        src0, weight, act, row, col, tensor_name, layer_id, ref);
                }
                nonfinite++;
                bad++;
                continue;
            }

            const double abs_err = std::fabs(got - expected);
            const double rel_err = abs_err / (std::fabs(expected) + 1.0e-12);
            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
            if (abs_err > atol && rel_err > rtol) {
                if (bad < 4) {
                    LOGE("%s_MISMATCH tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g abs=%.9g rel=%.9g",
                         contract_tag,
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
        if (mismatch_total) {
            *mismatch_total += bad;
        }
        LOGE("%s_SUMMARY tensor=%s layer=%d checked=%lld bad=%lld nonfinite=%lld max_abs=%.9g max_rel=%.9g atol=%.9g rtol=%.9g action=%s",
             contract_tag,
             tensor_name ? tensor_name : "?",
             layer_id,
             (long long) (n * m),
             bad,
             nonfinite,
             max_abs,
             max_rel,
             atol,
             rtol,
             abort_on_mismatch ? "abort" : "log_only");
        return !abort_on_mismatch;
    }

    LOGI("%s_PASS tensor=%s layer=%d checked=%lld nonfinite=0 max_abs=%.9g max_rel=%.9g reference=ggml_vec_dot_q8_0_q8_0",
         contract_tag,
         tensor_name ? tensor_name : "?",
         layer_id,
         (long long) (n * m),
         max_abs,
         max_rel);
    if (cpu_shadow_dst) {
        // ggml-cpu.c receives FPGA_MATMUL_CONTRACT_CPU_SHADOW and continues
        // into its upstream threaded kernel.  Do not write dst here: doing so
        // would race that kernel and would replace its output with a second
        // implementation during a contract run.
        if (shadow_value_total) {
            *shadow_value_total += (long long) value_count;
        }
        LOGI("%s_CPU_SHADOW_DST tensor=%s layer=%d values=%zu hardware_result=validated native_ggml_dst=deferred purpose=contract_isolation_not_cpu_fallback",
             contract_tag,
             tensor_name ? tensor_name : "?",
             layer_id,
             value_count);
    }
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
        bool require_packed_q8_capability,
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
    if ((uint64_t) k > (uint64_t) std::numeric_limits<size_t>::max() ||
        (uint64_t) n > (uint64_t) std::numeric_limits<size_t>::max() ||
        (uint64_t) m > (uint64_t) std::numeric_limits<size_t>::max()) {
        *reason = "unsupported DMA-to-IP tiling case: tensor dimensions exceed host size arithmetic";
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
    size_t src0_row_bytes = 0U;
    size_t src1_row_bytes = 0U;
    size_t dst_row_bytes = 0U;
    size_t activation_block_count = 0U;
    if (!checked_size_mul((size_t) (k / VPU_QK8_0), sizeof(block_q8_0_t), &src0_row_bytes) ||
        !checked_size_mul((size_t) k, sizeof(float), &src1_row_bytes) ||
        !checked_size_mul((size_t) n, sizeof(float), &dst_row_bytes) ||
        !checked_size_mul((size_t) m, (size_t) (k / VPU_QK8_0), &activation_block_count)) {
        *reason = "unsupported DMA-to-IP tiling case: tensor layout size overflows host allocation arithmetic";
        return false;
    }
    (void) activation_block_count;
    if (src0->nb[0] != sizeof(block_q8_0_t) ||
        src0->nb[1] < src0_row_bytes ||
        src1->nb[1] < src1_row_bytes ||
        dst->nb[1] < dst_row_bytes) {
        *reason = "unsupported DMA-to-IP tiling case: tensor stride is smaller than its logical row";
        return false;
    }
    if ((src0->nb[1] % alignof(block_q8_0_t)) != 0U ||
        (src1->nb[1] % alignof(float)) != 0U ||
        (dst->nb[1] % alignof(float)) != 0U ||
        ((uintptr_t) src0->data % alignof(block_q8_0_t)) != 0U ||
        ((uintptr_t) src1->data % alignof(float)) != 0U ||
        ((uintptr_t) dst->data % alignof(float)) != 0U) {
        *reason = "unsupported DMA-to-IP tiling case: typed Q8/F32 data or padded row stride is misaligned";
        return false;
    }
    fpga_tensor_access_range_t src0_range = {};
    fpga_tensor_access_range_t src1_range = {};
    fpga_tensor_access_range_t dst_range = {};
    if (!fpga_tensor_access_range(src0, k / VPU_QK8_0, n, sizeof(block_q8_0_t), &src0_range) ||
        !fpga_tensor_access_range(src1, k, m, sizeof(float), &src1_range) ||
        !fpga_tensor_access_range(dst, n, m, sizeof(float), &dst_range) ||
        src0_range.bytes > ggml_nbytes(src0) ||
        src1_range.bytes > ggml_nbytes(src1) ||
        dst_range.bytes > ggml_nbytes(dst)) {
        *reason = "unsupported DMA-to-IP tiling case: tensor byte span does not cover host read/write layout";
        return false;
    }

    if (fpga_tensor_ranges_overlap(src0_range, dst_range)) {
        *reason = "unsupported DMA-to-IP tiling case: destination aliases immutable weight storage";
        return false;
    }
    if (fpga_tensor_ranges_overlap(src1_range, dst_range)) {
        // Q, K, and V consume the same normalized activation tensor in the
        // Gemma graph.  An accelerator path cannot own a dst interval that
        // overlaps that live input; it would turn a valid later K/V input into
        // unrelated result bits.  The native CPU route has no such ownership
        // transfer, so reject this layout rather than silently corrupt it.
        *reason = "unsupported DMA-to-IP tiling case: destination aliases live F32 activation storage";
        return false;
    }
    if (require_packed_q8_capability && !g_packed_q8_supported) {
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
    // Do not pre-zero entry.bytes here.  With a 1100 MiB cache that was a
    // complete uncached DDR write before the useful data was even packed, and
    // was the first half of the apparent ZCU104 hang.  Each tile below writes
    // its complete payload; only its small alignment padding needs clearing so
    // that the payload CRC remains deterministic.
    if (!msync_ddr_range(entry.header_off, sizeof(pending_header), false,
                         "weight_cache_prepare_header")) {
        return nullptr;
    }

    uint32_t next_off = entry.base_off;
    uint32_t payload_crc_state = 0xFFFFFFFFU;
    size_t payload_progress = 0;
    size_t next_progress_log = 32U * 1024U * 1024U;
    std::vector<uint8_t> crc_tile_bytes;
    LOGI("weight tile cache build start tensor=%s tiles=%zu payload_mib=%.3f header=0x%08x base=0x%08x; CRC is streamed from cacheable host tile buffers (no full uncached DDR sweep)",
         tensor_name_or_unknown(src0),
         entry.tiles.size(),
         (double) entry.bytes / (1024.0 * 1024.0),
         entry.header_off,
         entry.base_off);
    for (fpga_weight_tile_cache_t & tile : entry.tiles) {
        tile.ddr_off = next_off;
        if (!ddr_range_fits(tile.ddr_off, tile.bytes)) {
            LOGE("weight tile cache range overflow tensor=%s off=0x%08x bytes=%zu",
                 tensor_name_or_unknown(src0), tile.ddr_off, tile.bytes);
            return nullptr;
        }

        crc_tile_bytes.resize(tile.bytes);
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
                    const size_t byte_offset = (size_t) word_index * 16U;
                    const int8_t * const source = wb->qs + beat * VPU_NUM_LANES;
                    ddr_write_i8x16(tile.ddr_off + word_index * 16U, source);
                    memcpy(crc_tile_bytes.data() + byte_offset, source, 16U);
                }
            }
        }

        payload_crc_state = fpga_crc32_update(
            payload_crc_state, crc_tile_bytes.data(), crc_tile_bytes.size());
        const size_t padded_tile_bytes = align_up_size(tile.bytes, WEIGHT_CACHE_ALIGN);
        const size_t padding_bytes = padded_tile_bytes - tile.bytes;
        if (padding_bytes != 0U) {
            // tile.bytes is 16-byte aligned and WEIGHT_CACHE_ALIGN is a
            // multiple of four, hence this preserves ddr_zero_range32's
            // alignment contract without touching the rest of the cache.
            ddr_zero_range32(tile.ddr_off + (uint32_t) tile.bytes, padding_bytes);
            payload_crc_state = fpga_crc32_update_zeros(payload_crc_state, padding_bytes);
        }
        payload_progress += padded_tile_bytes;
        if (payload_progress >= next_progress_log || payload_progress == entry.bytes) {
            LOGI("weight tile cache build progress tensor=%s packed_mib=%.3f/%.3f",
                 tensor_name_or_unknown(src0),
                 (double) payload_progress / (1024.0 * 1024.0),
                 (double) entry.bytes / (1024.0 * 1024.0));
            while (next_progress_log <= payload_progress) {
                next_progress_log += 32U * 1024U * 1024U;
            }
        }
        next_off = (uint32_t) align_up_size((size_t) tile.ddr_off + tile.bytes, WEIGHT_CACHE_ALIGN);
    }

    mmio_fence();
    if (!msync_ddr_range(entry.base_off, entry.bytes, false, "weight_cache_payload")) {
        return nullptr;
    }
    // The CRC was accumulated while the cacheable source tile was packed.
    // Re-reading the whole payload through UIO here was a second near-1 GiB
    // uncached access and must be reserved for the explicit diagnostic flag
    // FPGA_WEIGHT_CACHE_CRC_EACH_LOOKUP=1 only.
    entry.payload_crc32 = ~payload_crc_state;
    const long long crc_us = 0;
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
    LOGI("weight tile cache build complete tensor=%s tiles=%zu bytes=%zu scales=%zu header=0x%08x base=0x%08x crc32=0x%08x crc_mode=streamed_host pack_ms=%.3f crc_ms=%.3f next=0x%08x",
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
        const void * weight_data_base,
        const block_q8_0_t * act_group,
        int64_t row0,
        int rows,
        int64_t k_block0,
        int group_blocks,
        uint32_t weight_tile_index,
        const fpga_weight_cache_entry_t * weight_cache,
        std::vector<int32_t> & partial,
        std::vector<float> & weight_scales,
        float * accum_col,
        const float * act_scales_group,
        bool * accumulated_on_unpack,
        fpga_stage_totals_t * totals,
        uint32_t tile_id,
        const char * tensor_name,
        int layer_id,
        int64_t k,
        int64_t n,
        int64_t m,
        bool contract_check_active) {
    if (accumulated_on_unpack) {
        *accumulated_on_unpack = false;
    }
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
    const float * weight_scale_values = nullptr;
    std::vector<block_q8_0_t> & weight_snapshot = g_scratch.weight_tile_snapshot;
    weight_snapshot.resize((size_t) rows * (size_t) group_blocks);
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb] =
                *weight_block_from_base(src0, weight_data_base,
                                        row0 + row, k_block0 + gb);
        }
    }

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
            // The cache owns immutable scale metadata in the same row-major
            // layout used by accumulation.  Avoid copying it into a temporary
            // vector for every decode tile.
            weight_scale_values = weight_cache->scales.data() + tile.scale_off;
            if (contract_check_active || !g_fuse_raw_result_accum) {
                // Contract mode deliberately preserves the pre-fusion path so
                // it can validate raw values before the caller accumulates.
                weight_scales.assign(weight_scale_values,
                                     weight_scale_values + (size_t) rows * (size_t) group_blocks);
            }
        }
    }

    if (!weight_cache_hit) {
        weight_scales.resize((size_t) rows * (size_t) group_blocks);
        for (int row = 0; row < rows; ++row) {
            for (int gb = 0; gb < group_blocks; ++gb) {
                const block_q8_0_t * wb =
                    &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
                weight_scales[(size_t) row * (size_t) group_blocks + (size_t) gb] = fp16_to_fp32(wb->d);
            }
        }
        weight_scale_values = weight_scales.data();
    }
    if (!weight_scale_values) {
        LOGE("weight scale metadata is unavailable tensor=%s tile=%u", tensor_name ? tensor_name : "?", tile_id);
        return false;
    }

    const bool staging_guard_active = contract_check_active && g_contract_deep_staging;
    if (!fpga_stage_q8_group_with_contract_guard(
            weight_snapshot.data(), act_group, rows, group_blocks,
            !weight_cache_hit, weight_src_off, act_bytes, weight_bytes,
            staging_guard_active, tensor_name, layer_id, tile_id)) {
        return false;
    }
    if (totals) {
        totals->prep_us += now_us() - prep0;
        if (weight_cache_hit) {
            totals->weight_cache_hits++;
        } else {
            totals->weight_cache_misses++;
        }
    }

    const bool raw_contract_active =
        contract_check_active || g_contract_raw_repair_enabled;
    const int attempt_count = raw_contract_active ?
        (1 + g_contract_raw_retry_limit) : 1;
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
        // Verify at the exact ACT-DMA boundary.  This deliberately precedes
        // the optional VPU-register readback fence so a failure can be
        // attributed to the ZDMA read itself rather than to a later CPU/MMIO
        // observation step.
        if (staging_guard_active && !fpga_contract_verify_staged_q8_group(
                weight_snapshot.data(), act_group, rows, group_blocks,
                weight_src_off, tensor_name, layer_id, tile_id,
                "after_act_dma")) {
            if (!fpga_contract_restage_after_act_dma(
                    weight_snapshot.data(), act_group, rows, group_blocks,
                    !weight_cache_hit, weight_src_off, act_bytes, weight_bytes,
                    tensor_name, layer_id, tile_id)) {
                LOGE("contract staging changed across ACT DMA tensor=%s layer=%d tile=%u",
                     tensor_name ? tensor_name : "?", layer_id, tile_id);
                return false;
            }
        }
        fpga_ip_dma_readback_fence();
        fpga_ddr_staging_readback_commit(weight_src_off, weight_bytes);
        const long long dma_act1 = now_us();

        const long long dma_weight0 = now_us();
        if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) weight_src_off,
                           LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                           weight_bytes,
                           "WEIGHT")) {
            return false;
        }
        fpga_ip_dma_readback_fence();
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

        if (!contract_check_active && g_fuse_raw_result_accum) {
            // The legacy raw-result bitstream needs the PS to apply Q8 scales,
            // but it does not need an intermediate partial[] vector.  Consume
            // each 128-bit result word once and accumulate values in the same
            // row-major/group-major order as the non-fused implementation.
            //
            // Do not derive row/group with division and modulo for every raw
            // result.  On the PS that arithmetic was slower than retaining
            // partial[], so v45 advances an explicit row/group cursor and
            // writes the accumulator once per completed row.  It neither
            // changes the VPU result nor reorders floating-point additions.
            const long long result_unpack0 = now_us();
            uint32_t row = 0;
            uint32_t gb = 0;
            float row_accum = rows > 0 ? accum_col[0] : 0.0f;
            int32_t lanes[VPU_RESULT_PACK_LANES] = {};
            for (uint32_t word = 0; word < result_words; ++word) {
                read_result_i32x4_from_ddr(word, lanes);
                const uint32_t word_base = word * (uint32_t) VPU_RESULT_PACK_LANES;
                const uint32_t lane_count =
                    std::min((uint32_t) VPU_RESULT_PACK_LANES, result_values - word_base);
                for (uint32_t lane = 0; lane < lane_count; ++lane) {
                    if (row >= (uint32_t) rows || gb >= (uint32_t) group_blocks) {
                        LOGE("fused raw-result cursor overflow tensor=%s tile=%u word=%u row=%u group=%u rows=%d group_blocks=%d result_values=%u",
                             tensor_name ? tensor_name : "?", tile_id, word,
                             row, gb, rows, group_blocks, result_values);
                        return false;
                    }
                    const uint32_t scale_index = row * (uint32_t) group_blocks + gb;
                    row_accum +=
                        (float) lanes[lane] * act_scales_group[gb] *
                        weight_scale_values[scale_index];
                    ++gb;
                    if (gb == (uint32_t) group_blocks) {
                        accum_col[row] = row_accum;
                        ++row;
                        gb = 0;
                        if (row < (uint32_t) rows) {
                            row_accum = accum_col[row];
                        }
                    }
                }
            }
            if (row != (uint32_t) rows || gb != 0U) {
                LOGE("fused raw-result cursor incomplete tensor=%s tile=%u rows_done=%u expected_rows=%d remaining_group=%u result_values=%u",
                     tensor_name ? tensor_name : "?", tile_id, row, rows,
                     gb, result_values);
                return false;
            }
            if (totals) {
                totals->host_result_us += now_us() - result_unpack0;
            }
            if (accumulated_on_unpack) {
                *accumulated_on_unpack = true;
            }
            return true;
        }

        // Contract mode retains the raw vector so every value can be compared
        // to the CPU Q8_0 golden reference before any accumulation happens.
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

        // The raw reference is intentionally diagnostic-only.  v21 called
        // this O(rows * blocks * 32) CPU check for every production tile even
        // when FPGA_CONTRACT_CHECK=0; it produced log-only mismatch storms and
        // hid the fact that no contract was actually enabled.  Keep the
        // normal partial -> scale -> accumulate computation, but validate raw
        // values only for an explicit contract/repair run.
        if (!raw_contract_active) {
            return true;
        }

        if (!fpga_contract_verify_weight_source_snapshot(
                src0, weight_snapshot.data(), row0, rows, k_block0,
                group_blocks, tensor_name, layer_id, tile_id)) {
            return false;
        }

        const bool final_attempt = (attempt + 1) >= attempt_count;
        const bool repair_this_attempt = final_attempt && g_contract_raw_repair_enabled;
        fpga_raw_mismatch_location_t first_mismatch = {};
        const long long raw_mismatches = fpga_contract_count_raw_mismatches(
            weight_snapshot.data(), act_group, row0, rows, k_block0, group_blocks,
            weight_src_off,
            partial, tensor_name, layer_id, tile_id, attempt, true, repair_this_attempt,
            &first_mismatch);
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
        if (g_contract_forensic_replay && g_contract_check_abort &&
            !repair_this_attempt && first_mismatch.valid) {
            fpga_contract_forensic_replay(weight_snapshot.data(), act_group,
                                          rows, group_blocks, weight_src_off, weight_cache_hit,
                                          tile_id, tensor_name, layer_id, first_mismatch);
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
    const bool cpu_shadow_dst = g_contract_cpu_shadow_dst;
    g_contract_source_validation_failed = false;
    const size_t weight_tensor_bytes = ggml_nbytes(src0);
    const void * weight_data_base = src0->data;
    if (contract_check_active) {
        // Capture the complete immutable tensor before the first tile.  A
        // per-tile snapshot alone can miss corruption caused by an earlier
        // launch when the damaged row is not consumed until a later tile.
        g_scratch.weight_tensor_snapshot.resize(weight_tensor_bytes);
        memcpy(g_scratch.weight_tensor_snapshot.data(), src0->data,
               weight_tensor_bytes);
        weight_data_base = g_scratch.weight_tensor_snapshot.data();
        if (!fpga_contract_validate_weight_scales(
                src0, weight_data_base, tensor_name, layer_id)) {
            g_contract_source_validation_failed = true;
            return false;
        }
    } else {
        g_scratch.weight_tensor_snapshot.clear();
    }
    const bool use_pl_scale_path =
        g_pingpong_scheduler_enabled &&
        g_spu_q8_scale_stream_supported &&
        !contract_check_active &&
        !cpu_shadow_dst;

    const long long quant0 = now_us();
    if (!ensure_quantized_activation_matrix(src1, m, k, act_blocks_all, act_scales,
                                            !use_pl_scale_path, totals,
                                            tensor_name, layer_id)) {
        return false;
    }
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
    if (contract_check_active && cpu_shadow_dst) {
        g_scratch.contract_actual.assign((size_t) n * (size_t) m, 0.0f);
    } else {
        g_scratch.contract_actual.clear();
    }
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
                float * const accum_col = &accum[(size_t) (col * rows)];
                const float * const act_scales_group =
                    &act_scales[(size_t) (col * nb + ib0)];
                bool accumulated_on_unpack = false;
                if (!fpga_dma_run_q8_group(src0, weight_data_base, act_group,
                                           row0, rows, ib0, group_blocks,
                                           weight_tile_index, weight_cache,
                                           partial, weight_scales, accum_col, act_scales_group,
                                           &accumulated_on_unpack, totals, tile_id++,
                                           tensor_name, layer_id, k, n, m,
                                           contract_check_active)) {
                    return false;
                }

                if (!accumulated_on_unpack) {
                    const long long accum0 = now_us();
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
            }
            ib0 += group_blocks;
            weight_tile_index++;
        }

        const long long store0 = now_us();
        for (int64_t col = 0; col < m; ++col) {
            const float * accum_col = &accum[(size_t) (col * rows)];
            for (int row = 0; row < rows; ++row) {
                if (contract_check_active && cpu_shadow_dst) {
                    g_scratch.contract_actual[(size_t) col * (size_t) n +
                                              (size_t) (row0 + row)] = accum_col[(size_t) row];
                } else if (!cpu_shadow_dst) {
                    store_dst_value(dst, row0 + row, col, accum_col[(size_t) row]);
                }
            }
        }
        if (totals) {
            totals->host_accum_us += now_us() - store0;
        }
    }
    if (contract_check_active) {
        if (memcmp(src0->data, weight_data_base, weight_tensor_bytes) != 0) {
            const uint8_t * const live = (const uint8_t *) src0->data;
            const uint8_t * const baseline = (const uint8_t *) weight_data_base;
            size_t first_bad = 0U;
            while (first_bad < weight_tensor_bytes && live[first_bad] == baseline[first_bad]) {
                first_bad++;
            }
            LOGE("CONTRACT_WEIGHT_TENSOR_MUTATION tensor=%s layer=%d byte=%zu baseline=%u live=%u bytes=%zu; immutable GGUF tensor changed during FPGA matmul",
                 tensor_name ? tensor_name : "?", layer_id, first_bad,
                 first_bad < weight_tensor_bytes ? baseline[first_bad] : 0U,
                 first_bad < weight_tensor_bytes ? live[first_bad] : 0U,
                 weight_tensor_bytes);
            return false;
        }
        g_contract_checks_done++;
        if (!fpga_contract_check_output_values(src0, dst, act_blocks_all,
                                               weight_data_base, tensor_name, layer_id)) {
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
    if (env_flag_enabled("FPGA_ACTIVATION_CACHE")) {
        fpga_fatal("FPGA_ACTIVATION_CACHE is unsupported: the existing key is pointer/shape based and cannot prove immutable activation contents; refusing a stale-activation cache");
    }
    g_activation_cache_enabled = false;
    g_allow_devmem_fallback = env_flag_enabled("FPGA_ALLOW_DEVMEM");
    g_allow_vpu_devmem_compat =
        !env_flag_enabled("FPGA_VPU_UIO_REQUIRED") &&
        !env_flag_disabled("FPGA_VPU_DEVMEM_COMPAT");
    g_strict_coherency = env_flag_enabled("FPGA_STRICT_COHERENCY");
    g_coherency_platform_whitelisted = env_flag_enabled("FPGA_COHERENCY_PLATFORM_VERIFIED");
    g_run_coherency_stress = env_flag_enabled("FPGA_COHERENCY_STRESS") || g_strict_coherency;
    g_contract_check_abort = env_flag_enabled("FPGA_CONTRACT_ABORT");
    g_contract_forensic_replay = !env_flag_disabled("FPGA_CONTRACT_FORENSIC_REPLAY");
    g_clear_result_before_run = env_flag_enabled("FPGA_CLEAR_RESULT");
    // A repair changes the value consumed by the model, so it is a diagnostic
    // opt-in only.  Correctness runs must report mismatches, not hide them.
    g_contract_raw_repair_enabled = env_flag_enabled("FPGA_CONTRACT_RAW_REPAIR");
    g_fuse_raw_result_accum = env_flag_enabled("FPGA_FUSE_RAW_RESULT_ACCUM");
    g_weight_cache_crc_verify_each_lookup = env_flag_enabled("FPGA_WEIGHT_CACHE_CRC_EACH_LOOKUP");
    g_vocab_projection_cpu_bypass =
        !env_flag_disabled("FPGA_VOCAB_PROJECTION_CPU") &&
        !env_flag_enabled("FPGA_ACCELERATE_VOCAB");
    if (!g_fuse_raw_result_accum) {
        LOGPROOF("v23 correctness policy: raw-result fusion is disabled by default; set FPGA_FUSE_RAW_RESULT_ACCUM=1 only after end-to-end CPU/FPGA logits A/B passes");
    }
    if (g_vocab_projection_cpu_bypass) {
        LOGPROOF("v23 correctness policy: vocabulary projection remains on CPU; set FPGA_ACCELERATE_VOCAB=1 only after the deployed bitstream reports protocol=1/id=0x56505531 and logits A/B passes");
    }
    LOGPROOF("v50 numerical policy: C0 validates every immutable Q8_0 weight scale before the first VPU launch, uses ggml quantize_row_q8_0 plus stored FP16 scales, compares the hardware result against ggml_vec_dot_q8_0_q8_0, and rejects every NaN/Inf");
    LOGPROOF("v38 ZDMA policy: every descriptor is preceded by a W1C ISR clear that is read back until DMA_DONE and all error bits are zero. The completion loop therefore accepts only a newly generated DONE event. C0 keeps the same ACT/WEIGHT staging sequence as primary raw GEMV; staging_restages must remain zero before primary FPGA use.");
    LOGPROOF("v44 C0 loader gate: FPGA_CONTRACT_CHECK requires a completed upstream GGUF tensor-validation handshake before this host maps MY_IP/ZDMA/DDRHIGH or launches a model VPU transfer.");
    g_log_flush_every = env_int_value("FPGA_LOG_FLUSH_EVERY", 256, 1, 1000000);
    g_profile_every = env_int_value("FPGA_PROFILE_EVERY", FPGA_DEFAULT_PROFILE_EVERY, 0, 1000000);
    g_ip_status_every = env_int_value("FPGA_IP_STATUS_EVERY", FPGA_DEFAULT_STATUS_EVERY, 0, 1000000);
    g_detail_every = env_int_value("FPGA_DETAIL_EVERY", FPGA_DEFAULT_DETAIL_EVERY, 0, 1000000);
    g_contract_check_limit = env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000);
    g_q8_source_audit_only = env_flag_enabled("FPGA_SOURCE_AUDIT_ONLY");
    if (g_q8_source_audit_only && g_contract_check_limit > 0) {
        fpga_fatal("FPGA_SOURCE_AUDIT_ONLY and FPGA_CONTRACT_CHECK cannot be enabled together; source audit must not launch model GEMVs through ZDMA/VPU");
    }
    if (g_contract_check_limit > 0 && !fpga_model_tensor_validation_passed()) {
        fpga_fatal("C0 loader-validation handshake is missing; no board MMIO was mapped. Rebuild and deploy the coupled frontend files ggml/src/ggml-cpu/fpga_host.{cpp,h}, src/llama-model-loader.cpp, and tools/main/main.cpp from the same source tree, then rerun C0. Do not copy only fpga_host.cpp");
    }
    if (g_contract_check_limit > 0) {
        LOGPROOF("C0 loader-validation handshake=pass; full GGUF tensor validation completed before FPGA initialization");
    }
    g_contract_deep_staging =
        g_contract_check_limit > 0 && !env_flag_disabled("FPGA_CONTRACT_DEEP_STAGING");
    // C0/C1 validates hardware first, but must not overwrite dst while the
    // GGML worker pool owns it.  The ordinary contract route therefore stores
    // the verified hardware result privately and lets the native kernel write
    // dst.  Raw propagation remains an explicit forensic opt-in.
    g_contract_raw_propagation_diagnostic =
        g_contract_check_limit > 0 &&
        env_flag_enabled("FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC");
    const char * const legacy_canonical_override = getenv("FPGA_CONTRACT_CANONICAL_DST");
    g_contract_legacy_canonical_override_ignored =
        g_contract_check_limit > 0 &&
        legacy_canonical_override != nullptr &&
        env_flag_disabled("FPGA_CONTRACT_CANONICAL_DST");
    g_contract_cpu_shadow_dst =
        g_contract_check_limit > 0 && !g_contract_raw_propagation_diagnostic;
    if (g_contract_cpu_shadow_dst) {
        LOGPROOF("v50 C0/C1 shadow policy: every eligible GEMV still runs on ZDMA/VPU and is raw/value checked, but its result is retained privately while upstream GGML writes dst. This avoids a contract-mode dst race and is not a CPU fallback. Use FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC=1 only for raw-F32 propagation.");
        if (g_contract_legacy_canonical_override_ignored) {
            LOGPROOF("v50 compatibility: FPGA_CONTRACT_CANONICAL_DST=0 is ignored for C0/C1. Use FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC=1 only for a forensic raw-propagation run.");
        }
    } else if (g_contract_check_limit > 0) {
        LOGPROOF("v49 forensic policy: FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC=1; accepted raw-F32 FPGA results will propagate into CPU attention/KV. This mode diagnoses end-to-end sensitivity and is not a C0/C1 pass.");
    }
    g_dma_trace_enabled = env_flag_enabled("FPGA_DMA_AUDIT") || g_contract_check_limit > 0;
    memset(g_dma_trace, 0, sizeof(g_dma_trace));
    g_dma_trace_sequence = 0;
    g_contract_raw_retry_limit = env_int_value("FPGA_CONTRACT_RAW_RETRY", 1, 0, 8);
    g_runtime_max_rows = env_int_value("FPGA_RUNTIME_MAX_ROWS", VPU_SAFE_RUNTIME_ROWS, 1, VPU_DEFAULT_ROWS);
    g_vocab_projection_min_n = env_int64_value("FPGA_VOCAB_PROJECTION_MIN_N", 65536, 1024, LLONG_MAX);
    if (env_flag_enabled("FPGA_TILE_TIMING") && g_detail_every == 0) {
        g_detail_every = 1;
    }
    g_dma_timeout_us = env_int64_value("FPGA_DMA_TIMEOUT_US", FPGA_DEFAULT_DMA_TIMEOUT_US, 1000, LLONG_MAX);
    g_ip_timeout_us = env_int64_value("FPGA_IP_TIMEOUT_US", FPGA_DEFAULT_IP_TIMEOUT_US, 1000, LLONG_MAX);
    g_zdma_max_transfer_bytes = (size_t) env_int64_value(
        "FPGA_ZDMA_MAX_TRANSFER_BYTES",
        (long long) FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES,
        16,
        (long long) UINT32_MAX);
    g_zdma_max_transfer_bytes &= ~(size_t) 0xFU;
    if (g_zdma_max_transfer_bytes == 0U) {
        fpga_fatal("FPGA_ZDMA_MAX_TRANSFER_BYTES must be at least 16 and 16-byte aligned");
    }
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
    g_activation_input_integrity_check =
        env_flag_enabled("FPGA_INPUT_INTEGRITY_CHECK");
    if (g_contract_check_limit > 0 && !g_activation_input_integrity_check) {
        // C0 deliberately exercises M>1 layouts. It must always prove that
        // the host/VPU path leaves each live F32 source activation intact.
        g_activation_input_integrity_check = true;
        LOGPROOF("C0 input-integrity policy: FPGA_CONTRACT_CHECK automatically enables the F32 src1 snapshot/verify guard; FPGA_INPUT_INTEGRITY_CHECK=1 is implied for this qualification run");
    }

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
    const bool stream_protocol_compatible =
        g_stream_protocol_version == FPGA_REQUIRED_STREAM_PROTOCOL_VERSION;
    const bool raw_fpga_compatible = bitstream_id_compatible && stream_protocol_compatible;
    const bool legacy_raw_diagnostic_override =
        env_flag_enabled("FPGA_ALLOW_UNVERIFIED_LEGACY_RAW");
    if (!raw_fpga_compatible) {
        LOGPROOF("raw FPGA ABI is incompatible bitstream_id=0x%08x expected_id=0x%08x id_ok=%d stream_protocol=0x%08x required_protocol=%u protocol_ok=%d; primary raw GEMV is CPU-quarantined and PL-scale/pipeline opt-ins are prohibited",
                 g_bitstream_id, FPGA_EXPECTED_BITSTREAM_ID, bitstream_id_compatible ? 1 : 0,
                 g_stream_protocol_version, FPGA_REQUIRED_STREAM_PROTOCOL_VERSION,
                 stream_protocol_compatible ? 1 : 0);
    }
    g_legacy_raw_cpu_bypass =
        !raw_fpga_compatible &&
        g_contract_check_limit == 0 &&
        !legacy_raw_diagnostic_override;
    if (g_legacy_raw_cpu_bypass) {
        LOGPROOF("raw FPGA quarantine: block GEMVs will use CPU because the ID/protocol ABI gate failed; set FPGA_ALLOW_UNVERIFIED_LEGACY_RAW=1 only for a controlled raw-contract diagnostic run");
    } else if (!raw_fpga_compatible && legacy_raw_diagnostic_override) {
        LOGPROOF("DIAGNOSTIC OVERRIDE FPGA_ALLOW_UNVERIFIED_LEGACY_RAW=1 bypasses the raw ID/protocol quarantine; this is not production compatibility or board-success evidence");
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
        raw_fpga_compatible &&
        env_flag_enabled("FPGA_PIPELINE_ENABLE") &&
        !env_flag_enabled("FPGA_PIPELINE_DISABLE");
    if (env_flag_enabled("FPGA_PL_SCALE_ENABLE") && !g_spu_q8_scale_stream_supported) {
        fpga_fatal("FPGA_PL_SCALE_ENABLE requested but stream capability/protocol is not compatible: caps=0x%08x spu_caps=0x%08x protocol=0x%08x required=%u",
                   caps, spu_caps, g_stream_protocol_version, FPGA_REQUIRED_STREAM_PROTOCOL_VERSION);
    }
    if (env_flag_enabled("FPGA_PIPELINE_ENABLE") && !g_pingpong_scheduler_enabled) {
        fpga_fatal("FPGA_PIPELINE_ENABLE requested but ping-pong descriptor capability is unavailable caps=0x%08x", caps);
    }
    LOGPROOF("raw FPGA compatibility gate id_ok=%d protocol_ok=%d raw_compatible=%d legacy_diagnostic_override=%d route=%s",
             bitstream_id_compatible ? 1 : 0,
             stream_protocol_compatible ? 1 : 0,
             raw_fpga_compatible ? 1 : 0,
             legacy_raw_diagnostic_override ? 1 : 0,
             g_legacy_raw_cpu_bypass ? "cpu_quarantine" :
             (g_contract_check_limit > 0 ? "contract_diagnostic" : "raw_fpga"));
    vpu_select_banks(0, 0);

    const char * mapping_policy =
        g_allow_devmem_fallback ? "diagnostic_all_resources" :
        (g_allow_vpu_devmem_compat ? "uio_dma_ddr+vpu_devmem_compat" : "uio_required");

    LOGPROOF("ready version=%s path=%s rows=%d host_row_limit=%d col_beats=%d cols=%d packed_q8=%d max_group_blocks=%d result_words=%d zdma_max_transfer_bytes=%zu pingpong_cap=%d descriptor_cap=%d scheduler=%d scheduler_policy=opt_in pl_scale=%d pl_scale_policy=opt_in stream_protocol=0x%08x bitstream_id=0x%08x spu_silu=%d spu_rms=%d spu_rope=%d spu_softmax=%d weight_cache=%d activation_cache=%d input_integrity_check=%d vocab_cpu_bypass=%d legacy_raw_cpu_bypass=%d vocab_min_n=%lld contract_check=%d source_audit_only=%d contract_abort=%d contract_forensics=%d contract_deep_staging=%d contract_cpu_shadow_dst=%d contract_raw_propagation_diagnostic=%d raw_retry=%d raw_repair=%d raw_accum_fused=%d result_clear=%d strict_coherency=%d clock_mhz=%.3f profile_log=1 dma_detail=%d ip_detail=%d detail_every=%d flush_every=%d",
         FPGA_HOST_TRACE_VERSION,
         path ? path : "dma(default)",
         g_vpu_max_rows,
         g_runtime_max_rows,
         g_vpu_max_beats, g_vpu_max_cols,
         g_packed_q8_supported, g_packed_q8_max_blocks, g_packed_q8_result_words,
         g_zdma_max_transfer_bytes,
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
         g_activation_input_integrity_check ? 1 : 0,
         g_vocab_projection_cpu_bypass ? 1 : 0,
         g_legacy_raw_cpu_bypass ? 1 : 0,
         (long long) g_vocab_projection_min_n,
         g_contract_check_limit,
         g_q8_source_audit_only ? 1 : 0,
         g_contract_check_abort ? 1 : 0,
         g_contract_forensic_replay ? 1 : 0,
         g_contract_deep_staging ? 1 : 0,
         g_contract_cpu_shadow_dst ? 1 : 0,
         g_contract_raw_propagation_diagnostic ? 1 : 0,
         g_contract_raw_retry_limit,
         g_contract_raw_repair_enabled ? 1 : 0,
         g_fuse_raw_result_accum ? 1 : 0,
         g_clear_result_before_run ? 1 : 0,
         g_strict_coherency ? 1 : 0,
         g_fpga_clock_mhz,
         g_dma_timing_enabled ? 1 : 0,
         g_ip_timing_enabled ? 1 : 0,
         g_detail_every,
         g_log_flush_every);
    LOGPROOF("ZDMA trace policy enabled=%d depth=%zu trigger=raw_mismatch_or_transfer_failure contract_check=%d explicit_env=FPGA_DMA_AUDIT",
             g_dma_trace_enabled ? 1 : 0,
             FPGA_DMA_TRACE_DEPTH,
             g_contract_check_limit);
    if (g_activation_input_integrity_check) {
        LOGPROOF("FPGA_INPUT_INTEGRITY_CHECK=1: each raw FPGA matmul snapshots its logical F32 src1 before launch and verifies it after dst ownership returns; this is a qualification guard for M>1 graph layouts, not a CPU fallback or numerical replacement");
    }
    LOGPROOF("ZDMA byte-counter policy clear=before_every_transfer current=0x%08x error_mask=0x%08x; BYTE_CNT_OVRFL is fatal and produces a bounded DMA trace",
             g_dma ? g_dma->ZDMA_CH_TOTAL_BYTE : 0U,
             ZDMA_ISR_ERROR_MASK);
    LOGPROOF("ZDMA descriptor policy max_transfer_bytes=%zu; larger ACT/WEIGHT/RESULT copies are submitted as ordered contiguous chunks and preserve the VPU data layout",
             g_zdma_max_transfer_bytes);
    LOGPROOF("manifest host_version=%s host_build=\"%s %s\" limits=0x%08x caps=0x%08x spu_caps=0x%08x stream_protocol=0x%08x required_protocol=%u bitstream_id=0x%08x expected_bitstream_id=0x%08x bitstream_id_compatible=%d pingpong_cap=%d descriptor_cap=%d scheduler=%d scheduler_policy=opt_in pl_scale=%d pl_scale_policy=opt_in devmem_all_resources=%d vpu_devmem_compat=%d mapping_policy=%s spu_silu=%d spu_rms=%d spu_rope=%d spu_softmax=%d bases my_ip=0x%llx dma=0x%llx ddr=0x%llx windows act=0x%08x weight=0x%08x result=0x%08x spu_out=0x%08x spu_param=0x%08x block_q8_0_size=%zu compact_weight_layout_required=1",
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
    LOGPROOF("mappings policy=%s dma=%s virt=0x%llx size=0x%zx vpu=%s virt=0x%llx size=0x%zx ddr=%s virt=0x%llx mapped_size=0x%zx advertised_size=0x%zx",
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
    LOGPROOF("cache coherency source=%s strict=%d whitelist=%d stress=%d; msync barriers are issued before DDR-to-device and after device-to-DDR transfers",
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
    LOGPROOF("cleanup begin lifecycle=explicit-before-backend-free ddr_mapped=%d vpu_mapped=%d dma_mapped=%d weight_cache_entries=%zu",
             ddr_is_mapped() ? 1 : 0,
             vpu_is_mapped() ? 1 : 0,
             dma_is_mapped() ? 1 : 0,
             g_weight_cache.size());

    // No local source establishes a safe software abort/reset sequence for an
    // in-flight ZDMA descriptor.  Therefore cleanup must observe the
    // documented hardware-cleared CTRL2.EN state before invalidating DDR
    // metadata or unmapping any participant in the transfer.
    if (dma_is_mapped() && !zdma_wait_channel_disabled("cleanup", "before_unmap")) {
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal("ZDMA EN remained set during cleanup; descriptors, DDR/VPU/DMA mappings, and /dev/mem fd were intentionally left owned by the process. No undocumented reset/abort write was issued");
    }

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
    const long long hook_calls = g_matmul_hook_calls.load(std::memory_order_relaxed);
    const long long q8_candidates = g_q8_candidate_calls.load(std::memory_order_relaxed);
    const long long q8_intentional_cpu =
        g_q8_intentional_cpu_bypass_calls.load(std::memory_order_relaxed);
    const long long q8_unavailable_cpu =
        g_q8_unavailable_cpu_fallback_calls.load(std::memory_order_relaxed);
    const long long q8_expected_fpga =
        q8_candidates >= q8_intentional_cpu + q8_unavailable_cpu ?
        q8_candidates - q8_intentional_cpu - q8_unavailable_cpu : -1;
    const bool q8_coverage_complete =
        q8_expected_fpga >= 0 &&
        q8_unavailable_cpu == 0 &&
        g_fpga_count == q8_expected_fpga;
    const char * fpga_block_gemv_mode =
        q8_candidates == 0 ? "not_applicable" :
        g_legacy_raw_cpu_bypass_count > 0 ? "cpu_quarantine" :
        q8_coverage_complete ? "active" : "incomplete";
    LOGPROOF("FPGA_GEMV_COVERAGE hook_calls=%lld q8_candidates=%lld q8_expected_fpga=%lld q8_hw_completed=%lld q8_intentional_cpu_bypass=%lld q8_unavailable_cpu_fallback=%lld routing_verdict=%s fpga_block_gemv=%s",
             hook_calls,
             q8_candidates,
             q8_expected_fpga,
             g_fpga_count,
             q8_intentional_cpu,
             q8_unavailable_cpu,
             q8_coverage_complete ? "complete" : "incomplete",
             fpga_block_gemv_mode);
    LOGPROOF("cleanup complete fpga_calls=%lld vpu_runs=%lld rejects=%lld attention_cpu_bypass=%lld vocab_projection_cpu_bypass=%lld legacy_raw_cpu_bypass=%lld elapsed_s=%.3f pingpong_cap=%d descriptor_cap=%d scheduler=%d activation_cache_enabled=%d activation_cache_hits=%lld misses=%lld weight_cache_builds=%lld hits=%lld misses=%lld bytes=%lld cache_lookup_ms=%.3f cache_crc_ms=%.3f weight_pack_ms=%.3f activation_scale_fp16_overflows=%lld input_integrity_checks=%lld input_integrity_failures=%lld contract_checks=%lld contract_limit_cpu_bypass=%lld raw_mismatches=%lld raw_repairs=%lld value_mismatches=%lld contract_cpu_shadow_dst_values=%lld staging_restages=%lld q8_source_audit_checks=%lld q8_source_audit_failures=%lld",
         g_fpga_count,
         g_fpga_vpu_runs,
         g_reject_count,
         g_attention_bypass_count,
         g_vocab_projection_bypass_count,
         g_legacy_raw_cpu_bypass_count,
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
         g_activation_scale_fp16_overflows,
         g_activation_input_integrity_checks,
         g_activation_input_integrity_failures,
         g_contract_checks_done,
         g_contract_limit_cpu_bypass_count,
         g_contract_raw_mismatches,
         g_contract_raw_repairs,
         g_contract_value_mismatches,
         g_contract_cpu_shadow_dst_values,
         g_contract_staging_restage_count,
         g_q8_source_audit_checks,
         g_q8_source_audit_failures);
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

extern "C" int fpga_get_sequence_position(void) {
    return g_current_seq_pos;
}

extern "C" void fpga_advance_sequence_position(int n_tokens) {
    if (n_tokens < 0 || g_current_seq_pos > INT_MAX - n_tokens) {
        fpga_fatal("invalid FPGA sequence advance current=%d delta=%d",
                   g_current_seq_pos, n_tokens);
    }
    g_current_seq_pos += n_tokens;
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

    // This path deliberately runs before every board-facing policy branch.
    // It must validate the same Q8 tensors the normal graph would consume,
    // including the vocabulary projection, while leaving all work to CPU.
    const bool source_audit_only = fpga_source_audit_only_requested() != 0;
    if (source_audit_only) {
        if (env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000) > 0) {
            fpga_fatal("FPGA_SOURCE_AUDIT_ONLY and FPGA_CONTRACT_CHECK cannot be enabled together; source audit must not launch model GEMVs through ZDMA/VPU");
        }
        if (ith != 0) {
            return 0;
        }
        const char * reason = nullptr;
        if (!fpga_validate_tensors(src0, src1, dst, false, &reason)) {
            LOGI("Q8_SOURCE_AUDIT_SKIP tensor=%s layer=%d reason=%s",
                 tensor_name, effective_layer_id, reason ? reason : "unsupported tensor");
            return 0;
        }
        pthread_mutex_lock(&g_mutex);
        if (!g_q8_source_audit_mode_logged) {
            g_q8_source_audit_only = true;
            g_log_flush_every = env_int_value("FPGA_LOG_FLUSH_EVERY", 256, 1, 1000000);
            LOGPROOF("Q8_SOURCE_AUDIT_MODE version=%s host_hardware_init=skipped board_mmio=not_mapped zdma_selftests=not_run action=validate_q8_source_then_cpu_matmul",
                     FPGA_HOST_TRACE_VERSION);
            g_q8_source_audit_mode_logged = true;
        }
        const bool source_ok = fpga_audit_q8_source_only(
            src0, tensor_name, effective_layer_id);
        pthread_mutex_unlock(&g_mutex);
        if (!source_ok) {
            fpga_fatal("Q8 source audit failed tensor=%s layer=%d; refusing to continue with a numerically invalid source tensor",
                       tensor_name, effective_layer_id);
        }
        return 0;
    }

    if (is_attention) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_attention_bypass_count++;
            if (g_attention_bypass_count == 1) {
                LOGPROOF("attention path is currently bypassed to CPU; FPGA timing log below only covers Q8_0 matmul/GEMV hooks");
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return 0;
    }

    // Only count the single thread that owns this matmul result.  The GGML
    // CPU scheduler may call the hook from multiple workers, but non-zero
    // workers return success below without launching their own VPU job.
    const bool q8_f32_f32_candidate =
        src0 && src1 && dst &&
        src0->type == GGML_TYPE_Q8_0 &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32;
    if (ith == 0) {
        g_matmul_hook_calls.fetch_add(1, std::memory_order_relaxed);
        if (q8_f32_f32_candidate) {
            g_q8_candidate_calls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // `FPGA_CONTRACT_CHECK=N` used to stop only the comparison loop. The host
    // still launched later raw VPU jobs, but returned CPU shadow for their
    // destinations. That is neither a complete hardware qualification nor a
    // CPU baseline, and it obscures the first source of a later non-finite
    // activation. Make N a strict qualification boundary: the first N eligible
    // GEMVs execute, stage, and validate on FPGA; all later eligible GEMVs use
    // the upstream CPU kernel without being counted as FPGA unavailability or
    // a production fallback.
    if (q8_f32_f32_candidate && g_contract_cpu_shadow_dst &&
        g_contract_check_limit > 0 &&
        g_contract_checks_done >= (long long) g_contract_check_limit) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_contract_limit_cpu_bypass_count++;
            if (!g_contract_limit_cpu_bypass_logged) {
                LOGPROOF("C0 qualification boundary reached checked=%lld limit=%d; subsequent eligible Q8 GEMVs use the native CPU kernel without ZDMA/VPU launch. This is an explicit bounded-C0 policy, not FPGA unavailability or a production fallback.",
                         g_contract_checks_done, g_contract_check_limit);
                g_contract_limit_cpu_bypass_logged = true;
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return FPGA_MATMUL_NOT_HANDLED;
    }

    if (should_bypass_vocab_projection_to_cpu(tensor_name, probe_k, probe_n, probe_m)) {
        if (ith == 0) {
            if (q8_f32_f32_candidate) {
                g_q8_intentional_cpu_bypass_calls.fetch_add(1, std::memory_order_relaxed);
            }
            pthread_mutex_lock(&g_mutex);
            g_vocab_projection_bypass_count++;
            if (g_vocab_projection_bypass_count == 1) {
                LOGPROOF("vocab projection bypassed to CPU tensor=%s shape=K%lld_N%lld_M%lld threshold_n=%lld; unset FPGA_VOCAB_PROJECTION_CPU or set FPGA_ACCELERATE_VOCAB=1 to force FPGA",
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

    if (g_legacy_raw_cpu_bypass) {
        if (ith == 0) {
            if (q8_f32_f32_candidate) {
                g_q8_intentional_cpu_bypass_calls.fetch_add(1, std::memory_order_relaxed);
            }
            pthread_mutex_lock(&g_mutex);
            g_legacy_raw_cpu_bypass_count++;
            if (g_legacy_raw_cpu_bypass_count == 1) {
                LOGPROOF("legacy raw FPGA path bypassed to CPU tensor=%s shape=K%lld_N%lld_M%lld; this preserves language correctness while contract diagnostics repair the unverified bitstream/transfer path",
                     tensor_name,
                     (long long) probe_k,
                     (long long) probe_n,
                     (long long) probe_m);
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return 0;
    }

    const char * reason = nullptr;
    if (!fpga_validate_tensors(src0, src1, dst, true, &reason)) {
        if (ith == 0) {
            if (q8_f32_f32_candidate) {
                g_q8_unavailable_cpu_fallback_calls.fetch_add(1, std::memory_order_relaxed);
            }
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
            if (q8_f32_f32_candidate) {
                g_q8_unavailable_cpu_fallback_calls.fetch_add(1, std::memory_order_relaxed);
            }
            LOGE("FPGA/ZDMA/VPU/DDR is not initialized for tensor=%s", tensor_name);
            if (g_abort_on_cpu_fallback) {
                fpga_fatal("FPGA/ZDMA/VPU/DDR is not initialized; refusing CPU fallback");
            }
        }
        return 0;
    }

    if (ith != 0) {
        return g_contract_cpu_shadow_dst ?
            FPGA_MATMUL_CONTRACT_CPU_SHADOW : FPGA_MATMUL_FPGA_DST;
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
    const bool verify_activation_input = g_activation_input_integrity_check;
    if (verify_activation_input) {
        if (!fpga_capture_activation_input_snapshot(
                src1, k, m, g_scratch.activation_input_snapshot)) {
            LOGE("FPGA_INPUT_INTEGRITY_INTERNAL_ERROR tensor=%s layer=%d reason=snapshot_capture_failed K=%lld M=%lld",
                 tensor_name, effective_layer_id, (long long) k, (long long) m);
            pthread_mutex_unlock(&g_mutex);
            fpga_fatal("FPGA activation-input integrity snapshot failed tensor=%s layer=%d; refusing CPU fallback",
                       tensor_name, effective_layer_id);
        }
        g_activation_input_integrity_checks++;
    }

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

    bool hw_ok = fpga_hw_q8_0_matmul_dma_to_ip(src0, src1, dst, &totals, tensor_name, effective_layer_id);
    if (hw_ok && verify_activation_input &&
        !fpga_verify_activation_input_snapshot(
            src1, dst, k, m, g_scratch.activation_input_snapshot,
            tensor_name, effective_layer_id)) {
        g_activation_input_integrity_failures++;
        hw_ok = false;
    }
    const long long t1 = now_us();
    if (!hw_ok) {
        const bool source_validation_failed = g_contract_source_validation_failed;
        pthread_mutex_unlock(&g_mutex);
        if (source_validation_failed) {
            fpga_fatal("C0 source validation failed before any model ZDMA/VPU launch for tensor=%s layer=%d; the active GGUF contains a non-finite Q8_0 scale. Replace or re-copy the model, then rerun C0. No CPU fallback or scale repair was applied",
                       tensor_name, effective_layer_id);
        }
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
        LOGSTAGE("tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld row_tiles=%lld group_tiles=%lld q8_blocks=%lld vpu_runs=%lld prep_ms=%.3f cache_lookup_ms=%.3f cache_crc_ms=%.3f weight_pack_ms=%.3f activation_scale_fp16_overflows=%lld dma_input_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f ip_compute_ms=%.3f dma_output_ms=%.3f host_result_ms=%.3f host_accum_ms=%.3f total_ms=%.3f dominant=%s pl_scale=%d raw_accum_fused=%d effective_GMAC/s=%.3f effective_MiB/s=%.1f act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu weight_cache_hits=%lld weight_cache_misses=%lld cycles_per_run=%.1f",
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
                 totals.activation_scale_fp16_overflows,
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
                 g_fuse_raw_result_accum ? 1 : 0,
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
    return g_contract_cpu_shadow_dst ?
        FPGA_MATMUL_CONTRACT_CPU_SHADOW : FPGA_MATMUL_FPGA_DST;
}

extern "C" void fpga_reset_kv_cache(void) {
    g_current_seq_pos = 0;
    g_scratch.activation_cache_valid = false;
}
