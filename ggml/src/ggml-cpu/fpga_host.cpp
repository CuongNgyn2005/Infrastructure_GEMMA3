#include "fpga_host.h"
#include "fpga_log.h"
#include "fpga_q8_layout.h"
#include "fpga_q8_pack.h"

#include "ggml.h"
#include "quants.h"

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
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
#include <limits>
#include <string>
#include <vector>

#define FPGA_HOST_TRACE_VERSION "kv260-gemma3-q8-v1-uio-ddr-high"

// Phase 6 is an intentionally small, line-oriented diagnostic capture.  This
// finite bound limits both retained accounting and owner log volume; it is not
// a hardware capability or a production tiling limit.
static constexpr size_t FPGA_P6_RECORD_LIMIT_HARD_MAX = 4096U;

#define MY_IP_BASE_ADDRESS 0x00000000A0000000LL
#define REG_BASE_PHYS      0x00000000A0000000LL
#define LMM_BASE_PHYS      0x00000000A0000000LL

#define DMA_BASE_PHYS 0x00000000fd500000LL
// The verified ZDMA UIO resource is exactly one page:
// [0xFD500000, 0xFD501000).  Mapping a speculative 64 KiB aperture can cross
// into an unrelated peripheral even when current register offsets are small.
#define DMA_MMAP_SIZE 0x0000000000001000LL

// The KV260 device tree exposes the upper 2 GiB bank as the physical UIO
// resource "ddr_high" at [0x800000000, 0x880000000).  Keep the host staging
// aperture at 256 MiB so existing offsets and bounds remain unchanged while
// avoiding Linux-owned low DDR.
static constexpr uint64_t DDR_BASE_PHYS     = 0x0000000800000000ULL;
static constexpr size_t   DDR_REGION_SIZE   = 0x0000000010000000ULL;  // 256 MiB
static constexpr uint64_t DDR_END_EXCLUSIVE = DDR_BASE_PHYS + (uint64_t) DDR_REGION_SIZE;

static int                g_log_flush_every                   = 256;
// P2 has a separate MY_IP/SPU ABI and its initialization must be observable
// on the terminal even when the file log is unavailable or buffered.
static bool               g_p2_init_requested                 = false;
static bool               g_init_verbose                      = false;
static bool               g_summary_detail_after_error       = false;
// P2/P3 contract qualification is deliberately tile-bounded across eligible
// matrices.  The global Q16-verified count is the authority: an intermediate
// matrix remains CPU-shadowed and qualification continues with the next
// eligible matrix until the exact requested tile retires.  No later tile may
// be admitted after that cumulative boundary.
static int                g_p2_tile_limit                     = 0;
static bool               g_p2_allow_multitile                = false;
static bool               g_p2_tile_contract_boundary_reached = false;
static long long          g_p2_tile_q16_checks                = 0;
static long long          g_p2_matrix_contract_checks         = 0;
static bool               g_p2_tile_trace_enabled             = false;
// Phase 6 is a deliberately narrow ARM-host golden capture.  It observes the
// deployed host's existing FP32 accumulator operation after the P2 Q16 tile
// check; it does not compare with, or prescribe, a future PL FP32 result.
typedef struct {
    bool        enabled;
    std::string tensor;
    int         layer;
    int64_t     tile_row0;
    int64_t     row_first;
    int         row_count;
    int         expected_k_chunks;
    size_t      record_limit;
    size_t      records_emitted;
    int         chunks_seen;
    int64_t     next_k_block0;
    // The selected tensor may be evaluated during a multi-token warm-up
    // before its single-token decode GEMV.  Record the M=1 sighting
    // separately so cleanup can report a missing decode capture precisely.
    bool        selected_m1_matmul_seen;
    bool        selected_non_m1_defer_logged;
    bool        capture_complete;
} fpga_p6_accum_trace_t;

static fpga_p6_accum_trace_t g_p6_accum_trace = {};
// Keep routine P2 breadcrumbs in /tmp/fpga_debug.log.  Their stderr mirrors
// are an explicit diagnostic opt-in so normal inference remains readable.
static bool               g_p2_terminal_trace_enabled         = false;
// v60 keeps the expensive post-SPU_OUT breadcrumbs opt-in.  Qualification
// already has its own tile trace; this switch isolates the suspected terminal
// boundary without changing the normal P2 descriptor sequence.
static bool               g_p2_boundary_diagnostics_enabled   = false;
// P2 event telemetry is intentionally buffered and file-only.  It observes
// completed host operations but must never become a timing fence, a terminal
// breadcrumb, or a prerequisite for scheduling.
static bool               g_p2_event_trace_enabled             = false;
static uint32_t           g_p1_preload_breadcrumbs             = 0;
static constexpr uint32_t FPGA_P1_PRELOAD_BREADCRUMB_LIMIT     = 8U;
static bool               g_p1_preload_trace_enabled           = false;
static uint32_t           g_p2_trace_job_id                   = 0;
static uint32_t           g_p2_trace_tile_id                  = 0;
static int                g_p2_trace_bank                     = -1;
static unsigned long long g_p2_dma_transfer_sequence          = 0;
static std::string        g_p2_trace_dma_tag;
// The first model ACT transfer emits the exhaustive terminal trace only when
// FPGA_P2_FIRST_ACT_TRACE=1. Qualification has its separate tile-trace path.
static bool               g_p2_first_act_dma_trace_enabled = false;
static bool               g_p2_first_act_dma_trace_active = false;
static bool               g_p2_first_act_dma_trace_done   = false;

static void fpga_p2_init_breadcrumb(const char * fmt, ...);

// A board stop can lose buffered /tmp/fpga_debug.log output.  P2 boundary
// diagnostics therefore write the same marker to the debug log and terminal,
// flushing both sinks before any following MMIO or DDR operation.
static void fpga_p2_boundary_marker(const char * fmt, ...);

static void fpga_p2_dma_breadcrumb(const char * fmt, ...);

static unsigned long long fpga_ptr_addr(const volatile void * ptr) {
    return (unsigned long long) reinterpret_cast<uintptr_t>(ptr);
}

static void fpga_log_line(bool enabled, const char * tag, bool force_flush, const char * fmt, ...);

#define LOGI(fmt, ...)     fpga_log_line(true, "INFO", false, fmt, ##__VA_ARGS__)
#define LOGPROOF(fmt, ...) fpga_log_line(true, "INFO", true, fmt, ##__VA_ARGS__)
#define LOGINIT(fmt, ...)  fpga_log_line(g_init_verbose, "INFO", false, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...)     fpga_log_line(true, "ERROR", true, fmt, ##__VA_ARGS__)
#define LOGDMA(fmt, ...)   fpga_log_line(g_dma_timing_enabled, "DMA", false, fmt, ##__VA_ARGS__)
#define LOGIP(fmt, ...)    fpga_log_line(g_ip_timing_enabled, "IPTIME", false, fmt, ##__VA_ARGS__)
#define LOGSTAGE(fmt, ...) fpga_log_line(g_stage_timing_enabled, "STAGE", false, fmt, ##__VA_ARGS__)
#define LOGDATA(fmt, ...)  fpga_log_line(g_trace_data_enabled, "DATA", false, fmt, ##__VA_ARGS__)

static constexpr uint64_t VPU_BASE_PHYS          = MY_IP_BASE_ADDRESS;
static constexpr double   P2_PL_SCALE_VALUE_ATOL = 0.002;

static constexpr uint32_t REG_CTRL                    = 0x00000000;
static constexpr uint32_t REG_STATUS                  = 0x00000010;
static constexpr uint32_t REG_ROWS                    = 0x00000020;
static constexpr uint32_t REG_COLS                    = 0x00000030;
static constexpr uint32_t REG_COL_BEATS               = 0x00000040;
static constexpr uint32_t REG_SCALE                   = 0x00000050;
static constexpr uint32_t REG_MODE                    = 0x00000060;
static constexpr uint32_t REG_LIMITS                  = 0x00000070;
static constexpr uint32_t REG_PROGRESS                = 0x00000080;
static constexpr uint32_t REG_CAPS                    = 0x00000090;
static constexpr uint32_t REG_STREAM_PROTOCOL_VERSION = 0x000000F4;
static constexpr uint32_t REG_BITSTREAM_ID            = 0x000000F8;
// FPGA_PL_SCALE_ENABLE is a distinct P2 ABI.  Do not infer it from the raw
// VPU1/protocol1 identity: previous raw-compatible images did not guarantee
// the SPU_PARAM/SPU_OUT packing or stream-finality contract.
static constexpr uint32_t REG_P2_STREAM_ABI           = 0x000000FC;
static constexpr uint32_t REG_BANK                    = 0x00000100;
static constexpr uint32_t REG_JOB_ID                  = 0x00000110;
static constexpr uint32_t REG_BANK_STAT               = 0x00000120;
static constexpr uint32_t REG_ACTIVE_JOB              = 0x00000130;
static constexpr uint32_t REG_DONE_JOB                = 0x00000140;
static constexpr uint32_t REG_SLOT_STATE              = 0x00000150;
static constexpr uint32_t REG_TENSOR_ID               = 0x00000160;
static constexpr uint32_t REG_ROW0                    = 0x00000170;
static constexpr uint32_t REG_K_BLOCK0                = 0x00000180;
static constexpr uint32_t REG_GROUP_BLOCKS            = 0x00000190;
static constexpr uint32_t REG_TOKEN_ID                = 0x000001A0;
static constexpr uint32_t REG_DESC_FLAGS              = 0x000001B0;
static constexpr uint32_t REG_SPU_CAPS                = 0x000000F0;
static constexpr uint32_t REG_SPU_STREAM_COUNT        = 0x000001C0;
static constexpr uint32_t REG_SPU_STREAM_DONE         = 0x000001C4;
static constexpr uint32_t REG_SPU_STREAM_DROP         = 0x000001D0;
static constexpr uint32_t REG_SPU_STREAM_OUT          = 0x000001D4;
static constexpr uint32_t REG_SPU_STREAM_ERROR        = 0x000001D8;
static constexpr uint32_t REG_SPU_STREAM_LAST_JOB     = 0x000001E8;
static constexpr uint32_t REG_SPU_STREAM_LAST_BANK    = 0x000001EC;
static constexpr uint32_t REG_SPU_STREAM_STATUS       = 0x000001F8;
static constexpr uint32_t REG_STREAM_MODE             = 0x000001FC;
static constexpr uint32_t REG_P3_SPLIT_SCALE_ABI       = 0x00000200;
static constexpr uint32_t REG_SPU_STREAM_ENTRY_DONE    = 0x00000210;
static constexpr uint32_t REG_SPU_STREAM_FINAL_WRITE   = 0x00000214;
static constexpr uint32_t REG_SPU_STREAM_P3_REJECT     = 0x00000218;
static constexpr uint32_t REG_SPU_STREAM_P3_STATUS     = 0x0000021C;
static constexpr uint32_t REG_SPU_CTRL                = 0x000000A0;

static constexpr uint32_t CTRL_START      = 0x00000001;
static constexpr uint32_t CTRL_CLEAR_DONE = 0x00000002;
static constexpr uint32_t STATUS_DONE     = 0x00000001;
static constexpr uint32_t STATUS_BUSY     = 0x00000002;
static constexpr uint32_t STATUS_ERROR    = 0x00000004;
static constexpr uint32_t BANK_STAT_ACTIVE_BANK = 0x00000100;
static constexpr uint32_t BANK_STAT_DONE_BANK   = 0x00000200;
static constexpr uint32_t BANK_STAT_BUSY        = 0x00010000;
static constexpr uint32_t BANK_STAT_DONE        = 0x00020000;
static constexpr uint32_t BANK_STAT_ERROR       = 0x00040000;

static constexpr uint32_t ACT_BASE               = 0x00010000;
static constexpr uint32_t ACT_END                = 0x00020000;
static constexpr uint32_t WEIGHT_BASE            = 0x00100000;
static constexpr uint32_t WEIGHT_END             = 0x00200000;
static constexpr uint32_t RESULT_BASE            = 0x00200000;
static constexpr uint32_t RESULT_END             = 0x00210000;
static constexpr uint32_t SPU_OUT_BASE           = 0x00340000;
static constexpr uint32_t SPU_OUT_END            = 0x00380000;
static constexpr uint32_t SPU_PARAM_BASE         = 0x00380000;
static constexpr uint32_t SPU_PARAM_END          = 0x003C0000;
static constexpr uint32_t SPU_SCRATCH_BASE       = 0x003C0000;
static constexpr uint32_t SPU_SCRATCH_END        = 0x00400000;
static_assert((WEIGHT_BASE % alignof(uint32_t)) == 0U, "WEIGHT_BASE must be 32-bit aligned");
// All host-visible MY_IP registers and local-memory windows used by this
// driver lie below this offset.  The Vivado segment is 256 MiB, but the
// compatibility /dev/mem path must not map the whole segment unnecessarily.
static constexpr size_t   VPU_DEVMEM_COMPAT_MMAP = SPU_SCRATCH_END;
static constexpr size_t   DDR_REQUIRED_BYTES     = SPU_SCRATCH_END;
static constexpr uint32_t WEIGHT_CACHE_BASE      = 0x01000000;
static constexpr uint32_t P2_WEIGHT_RESIDENCY_END = 0x03000000;
static constexpr size_t   WEIGHT_CACHE_ALIGN     = 4096;
static_assert((DDR_BASE_PHYS & 0xFFFULL) == 0ULL, "DDR_BASE_PHYS must be page aligned");
static_assert((DDR_REGION_SIZE & 0xFFFULL) == 0ULL, "DDR_REGION_SIZE must be page aligned");
static_assert(DDR_BASE_PHYS >= 0x0000000800000000ULL && DDR_END_EXCLUSIVE <= 0x0000000880000000ULL,
              "KV260 FPGA DDR staging aperture must stay inside reserved ddr_high");
static_assert(DDR_REQUIRED_BYTES <= DDR_REGION_SIZE, "FPGA scratch windows exceed the reserved DDR carveout");
static_assert((uint64_t) WEIGHT_CACHE_BASE < (uint64_t) DDR_REGION_SIZE,
              "weight-cache base lies outside the reserved DDR carveout");
// A UIO resource size only proves that Linux will mmap the address range; it
// does not prove that a large sequential write is safe for the deployed DDR
// interconnect/bitstream.  Keep the first cache experiment bounded until the
// board-specific address preflight and graduated cache tests are recorded.
// Larger requests require an explicit operator acknowledgement and are never
// enabled by the primary command accidentally.
static constexpr long long WEIGHT_CACHE_DEFAULT_MAX_MB    = 16;
// P2 residency is deliberately a different experiment from the legacy
// weight cache. It owns a fixed, non-evicting directory of sealed tiles and
// never calls msync().
static constexpr long long P2_WEIGHT_RESIDENCY_MAX_MB     = 32;
// Protocol-2/VPU2 stores each beat as an adjacent even/odd row pair.  This
// version is deliberately part of the sealed-residency identity: a v1 tile
// is byte-valid but semantically row-major, and must never be reused by VPU2.
static constexpr uint32_t  P2_WEIGHT_RESIDENCY_LAYOUT_V2  = 2U;
static constexpr size_t    P2_WEIGHT_RESIDENCY_SLOT_CAPACITY = 128U;
// Residency lookups are bounded without walking the slot directory.  Linear
// probing is deterministic, capped, and never overwrites or evicts a seal.
static constexpr size_t    P2_WEIGHT_RESIDENCY_INDEX_BUCKETS = 1024U;
static constexpr size_t    P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES = 8U;
static constexpr uint32_t  P2_WEIGHT_RESIDENCY_NO_SLOT       = UINT32_MAX;
static constexpr uint32_t  P2_WEIGHT_RESIDENCY_INVALID_SLOT  = UINT32_MAX - 1U;
static_assert(P2_WEIGHT_RESIDENCY_INDEX_BUCKETS >= P2_WEIGHT_RESIDENCY_SLOT_CAPACITY,
              "P2 residency index must not be smaller than its sealed-slot directory");
static_assert((uint64_t) P2_WEIGHT_RESIDENCY_END - (uint64_t) WEIGHT_CACHE_BASE ==
                  (uint64_t) P2_WEIGHT_RESIDENCY_MAX_MB * 1024ULL * 1024ULL,
              "P2 residency budget must remain inside [0x01000000,0x03000000)");
// Avoid a repeated, unsupported msync only for cache payloads large enough to
// make the call itself disruptive.  Per-tile scratch transfers retain v16's
// conservative msync attempt even after a UIO driver reports EINVAL.
static constexpr size_t    WEIGHT_CACHE_LARGE_MSYNC_BYTES = 16U * 1024U * 1024U;

static constexpr int      VPU_NUM_LANES                         = 16;
static constexpr int      VPU_QK8_0                             = 32;
static constexpr int      VPU_BLOCK_BEATS                       = VPU_QK8_0 / VPU_NUM_LANES;
static constexpr int      VPU_RESULT_PACK_LANES                 = 4;
static constexpr int      VPU_PACKED_Q8_MAX_BLOCKS              = 64;
static constexpr int      VPU_DEFAULT_ROWS                      = 256;
static constexpr int      VPU_SAFE_RUNTIME_ROWS                 = 256;
static constexpr int      VPU_DEFAULT_BEATS                     = 128;
static constexpr int      VPU_DEFAULT_COLS                      = VPU_DEFAULT_BEATS * VPU_NUM_LANES;
static constexpr int      VPU_LEGACY_PACKED_Q8_MAX_BLOCKS       = 16;
static constexpr int      VPU_LEGACY_BEATS                      = 32;
static constexpr uint32_t VPU_MODE_PACKED_Q8                    = 0x00000001;
static constexpr uint32_t VPU_MODE_P2_TWO_ROW                   = 0x00000010;
static constexpr uint32_t VPU_FP16_ONE                          = 0x00003C00;
static constexpr float    VPU_FP16_MAX_FINITE                   = 65504.0f;
static constexpr uint32_t VPU_CAP_PACKED_Q8                     = 0x00000001;
static constexpr uint32_t VPU_CAP_COMPACT_WEIGHT_LAYOUT         = 0x00000002;
static constexpr uint32_t VPU_CAP_PINGPONG_BANKS                = 0x00000008;
static constexpr uint32_t VPU_CAP_JOB_DESCRIPTOR                = 0x00000010;
static constexpr uint32_t VPU_CAP_SPU_RAW_STREAM                = 0x00000020;
static constexpr uint32_t VPU_CAP_SPU_Q8_SCALE_STREAM           = 0x00000040;
static constexpr uint32_t VPU_CAP_P2_TWO_ROW_TRANSPORT          = 0x00000080;
static constexpr uint32_t SPU_CAP_VPU_RAW_STREAM                = 0x00000100;
static constexpr uint32_t SPU_CAP_VPU_Q8_SCALE_STREAM           = 0x00000200;
static constexpr uint32_t SPU_CAP_VPU_Q8_SCALE_PAIR_STREAM      = 0x00000400;
static constexpr uint32_t SPU_CAP_P3_SPLIT_SCALE                = 0x00000800;
static constexpr uint32_t SPU_CAP_SILU_MUL                      = 0x00000004;
static constexpr uint32_t SPU_CAP_RMSNORM                       = 0x00000008;
static constexpr uint32_t SPU_CAP_ROPE                          = 0x00000010;
static constexpr uint32_t SPU_CAP_SOFTMAX                       = 0x00000020;
static constexpr uint32_t FPGA_REQUIRED_STREAM_PROTOCOL_VERSION = 2;
static constexpr uint32_t FPGA_EXPECTED_BITSTREAM_ID            = 0x56505532U;  // "VPU2"
// "P2", ABI v3.  In addition to SPU_PARAM/SPU_OUT, it binds pair-interleaved
// padded WEIGHT beats: index=((row>>1)*group_beats+beat)*2+(row&1).
// It binds the following layout: SPU_PARAM=0x00380000
// contains four {weight_scale_fp16,act_scale_fp16} entries per 128-bit word;
// SPU_OUT=0x00340000 stores {q16.16_accum,row_id} per 128-bit row; and VPU
// two-row transport requires both VPU and SPU capability advertisements.
static constexpr uint32_t FPGA_REQUIRED_P2_STREAM_ABI           = 0x50320003U;
static constexpr uint32_t FPGA_REQUIRED_P3_SPLIT_SCALE_ABI      = 0x50330001U;
static constexpr uint32_t SPU_STREAM_STATUS_QUIESCENT           = 0x00000010U;
static constexpr uint32_t SPU_P3_STATUS_LOCK_VALID              = 0x00000010U;
static constexpr uint32_t SPU_P3_STATUS_MODE_RETAINED           = 0x00000080U;
static constexpr int      P3_MAX_ROWS                           = 256;
static constexpr int      P3_MAX_GROUP_BLOCKS                   = 64;
static constexpr uint32_t SPU_CTRL_SOFT_RESET                   = 0x00000004U;
static constexpr uint32_t P2_REQUIRED_SPU_WORD_CAPACITY = (VPU_DEFAULT_ROWS * VPU_PACKED_Q8_MAX_BLOCKS + 3U) / 4U;

static constexpr long long FPGA_DEFAULT_DMA_TIMEOUT_US          = 5000000LL;
static constexpr long long FPGA_DEFAULT_IP_TIMEOUT_US           = 5000000LL;
// The ZDMA hardware accepts much larger descriptors, but the legacy board
// path has shown a normal-run-only staging corruption on 512 KiB WEIGHT
// copies.  Keep each submitted descriptor bounded to 64 KiB.  Consecutive
// chunks retain the same contiguous destination window and do not alter the
// VPU's tile, Q8 layout, or arithmetic contract.
static constexpr size_t    FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES = 64U * 1024U;
static constexpr int       FPGA_DEFAULT_STATUS_EVERY            = 0;
static constexpr int       FPGA_DEFAULT_PROFILE_EVERY           = 0;
static constexpr int       FPGA_DEFAULT_DETAIL_EVERY            = 0;
static constexpr int       FPGA_DEFAULT_SUMMARY_DETAIL_EVERY    = 16;
static constexpr long long FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS   = 1000000LL;
static constexpr long long FPGA_STREAM_POLL_LOG_INTERVAL_US     = 50000LL;
static constexpr size_t    FPGA_DMA_TRACE_DEPTH                 = 24U;

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
static_assert(sizeof(dma_ctrl) <= DMA_MMAP_SIZE, "ZDMA register struct exceeds verified UIO page aperture");

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
static_assert(sizeof(block_q8_0_t) == sizeof(ggml_half) + VPU_QK8_0, "FPGA/GGML Q8_0 block size mismatch");
static_assert(offsetof(block_q8_0_t, d) == 0U, "unexpected Q8_0 scale offset");
static_assert(offsetof(block_q8_0_t, qs) == sizeof(ggml_half), "unexpected Q8_0 quant offset");

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
    // Low-overhead preparation decomposition. These counters distinguish
    // source lookup/residency, direct WEIGHT packing, scale-table packing,
    // and ACT packing without emitting one log line per tile.
    long long prep_weight_select_us;
    long long prep_direct_weight_pack_us;
    long long prep_scale_pack_us;
    long long prep_act_pack_us;
    long long activation_scale_fp16_overflows;
    // P1 is explicitly opt-in. Keep its timing separate from ordinary input
    // DMA so an owner can compare preload duration and post-retirement launch
    // bubble without changing default stage-telemetry semantics.
    long long input_preload_us;
    long long preload_launch_bubble_us;
    long long input_preload_jobs;
    // Scheduler handoff telemetry. All timestamps use CLOCK_MONOTONIC.
    // "late" means preparation completed after the running job output was
    // already ready; "headroom" means it completed before that boundary.
    long long scheduler_prepare_overlap_us;
    long long scheduler_prepare_late_us;
    long long scheduler_prepare_headroom_us;
    long long scheduler_preload_overlap_us;
    long long scheduler_output_to_launch_us;
    long long scheduler_retire_to_launch_us;
    long long scheduler_handoffs;
    long long scheduler_prepare_late_jobs;
    long long scheduler_preload_overlap_jobs;
    // Bank-aware telemetry.  H2IP includes ACT + WEIGHT + SCALE and, when
    // P1 preload is active, the already-completed ACT/WEIGHT preload time.
    long long bank_h2ip_us[2];
    long long bank_compute_us[2];
    long long bank_ip2host_us[2];
    long long bank_host_read_us[2];
    long long bank_jobs[2];
    long long first_ip_launch_mono_us;
    long long last_ip_output_ready_mono_us;
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
    const struct ggml_tensor *            tensor;
    const void *                          data;
    int64_t                               k;
    int64_t                               n;
    size_t                                nb1;
    int                                   max_rows;
    int                                   max_beats;
    int                                   max_group_blocks;
    uint32_t                              base_off;
    size_t                                bytes;
    uint32_t                              header_off;
    uint32_t                              payload_crc32;
    bool                                  valid;
    bool                                  crc_validated;
    std::vector<fpga_weight_tile_cache_t> tiles;
    std::vector<float>                    scales;
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

static_assert(sizeof(fpga_weight_cache_header_t) == 32, "unexpected weight-cache header layout");

static constexpr uint32_t FPGA_WEIGHT_CACHE_MAGIC          = 0x46504348U;  // "FPCH"
static constexpr uint32_t FPGA_WEIGHT_CACHE_FORMAT_VERSION = 2U;

// This is intentionally a fixed, non-evicting directory. Every slot carries
// the complete immutable tensor/tile identity and deployed ABI identity needed
// to prove that a later DMA source is the exact payload that was sealed.
typedef struct {
    bool                      enabled;
    bool                      building;
    bool                      sealed;
    bool                      poisoned;
    const struct ggml_tensor * tensor;
    const void *              data;
    enum ggml_type            type;
    int64_t                   ne[GGML_MAX_DIMS];
    size_t                    nb[GGML_MAX_DIMS];
    uint32_t                  layout_version;
    int64_t                   row0;
    int                       rows;
    int64_t                   k_block0;
    int                       group_blocks;
    int                       group_beats;
    size_t                    qs_bytes;
    size_t                    scale_bytes;
    size_t                    allocation_bytes;
    uint32_t                  qs_off;
    uint32_t                  scale_off;
    uint32_t                  crc32;
    uint32_t                  seal;
    uint64_t                  epoch;
    uint32_t                  stream_protocol;
    uint32_t                  bitstream_id;
    uint32_t                  p2_abi;
    uint64_t                  key_hash;
    std::vector<uint16_t>     scale_bits;
    size_t                    scale_count;
    uint32_t                  scale_crc32;
    uint32_t                  metadata_seal;
    // This is published only after the build's DDR readback, CRC, and seal
    // have all completed.  It makes ordinary reuse validation O(1) without
    // pretending that an unsealed host vector is immutable.
    uint64_t                  metadata_validated_epoch;
} fpga_p2_resident_tile_t;

typedef struct {
    int                               bank;
    uint32_t                          job_id;
    unsigned long long                matmul_call_id;
    int                               graph_seq;
    int                               layer_id;
    int64_t                           shape_k;
    int64_t                           shape_n;
    int64_t                           shape_m;
    bool                              cpu_shadow_dst;
    bool                              pingpong_scheduler;
    const char *                      tensor_name;
    uint32_t                          tile_id;
    uint32_t                          tensor_id;
    int64_t                           row0;
    int                               rows;
    int64_t                           k_block0;
    int                               group_blocks;
    int                               group_beats;
    int64_t                           col;
    size_t                            act_bytes;
    size_t                            weight_bytes;
    size_t                            scale_bytes;
    // P3 keeps the P2 result ABI but replaces packed {weight,activation}
    // entries with dense FP16 tables in the selected PARAM/SCRATCH halves.
    bool                              p3_split_scale;
    uint32_t                          p3_param_off;
    uint32_t                          p3_scratch_off;
    size_t                            p3_weight_scale_bytes;
    size_t                            p3_activation_scale_bytes;
    size_t                            spu_result_bytes;
    size_t                            result_bytes;
    uint32_t                          result_values;
    uint32_t                          result_words;
    uint32_t                          scale_words;
    uint32_t                          weight_src_off;
    bool                              weight_cache_hit;
    bool                              p2_residency_hit;
    uint32_t                          p2_residency_slot;
    uint64_t                          p2_residency_epoch;
    uint32_t                          p2_residency_seal;
    const block_q8_0_t *              act_group;
    const struct ggml_tensor *        src0;
    const fpga_weight_cache_entry_t * weight_cache;
    std::vector<int32_t>              partial;
    std::vector<float>                weight_scales;
    long long                         result_clear_us;
    long long                         dma_act_us;
    long long                         dma_weight_us;
    long long                         dma_scale_us;
    long long                         dma_result_us;
    long long                         ip_start_us;
    long long                         ip_compute_us;
    long long                         host_result_us;
    long long                         event_prep_begin_us;
    long long                         event_prep_done_us;
    long long                         event_preload_begin_us;
    long long                         event_preload_done_us;
    long long                         event_submit_begin_us;
    long long                         event_input_transfer_begin_us;
    long long                         event_launch_us;
    long long                         event_vpu_done_us;
    long long                         event_spu_finality_us;
    long long                         event_retire_us;
    long long                         handoff_prev_output_ready_us;
    long long                         handoff_prev_retire_us;
    uint32_t                          vpu_status;
    uint32_t                          spu_stream_count_before;
    uint32_t                          spu_stream_done_before;
    uint32_t                          spu_stream_out_before;
    uint32_t                          spu_stream_drop_before;
    uint32_t                          spu_stream_error_before;
    uint32_t                          spu_stream_entry_done_before;
    uint32_t                          spu_stream_final_write_before;
    uint32_t                          spu_stream_p3_reject_before;
    // An input preload transfers only ACT and WEIGHT into an inactive bank.
    // This snapshot is redundant by design: deferred launch proves it owns
    // the exact staged tile, rather than a same-shaped slot reuse.
    bool                              input_preloaded;
    bool                              input_preload_poisoned;
    uint32_t                          preload_key_job_id;
    const struct ggml_tensor *        preload_key_tensor;
    uint32_t                          preload_key_tensor_id;
    int64_t                           preload_key_row0;
    int                               preload_key_rows;
    int64_t                           preload_key_col;
    int64_t                           preload_key_k_block0;
    int                               preload_key_group_blocks;
    size_t                            preload_key_act_bytes;
    size_t                            preload_key_weight_bytes;
    size_t                            preload_key_scale_bytes;
    uint32_t                          preload_key_weight_src_off;
    uint32_t                          preload_key_weight_layout_version;
    uint32_t                          preload_key_p2_residency_slot;
    uint64_t                          preload_key_p2_residency_epoch;
    uint32_t                          preload_key_p2_residency_seal;
    int                               preload_key_bank;
    long long                         input_preload_us;
    long long                         input_preload_done_us;
    long long                         preload_ready_to_launch_us;
} fpga_tile_job_t;

typedef struct {
    std::vector<block_q8_0_t>  act_blocks_all;
    // Used only by FPGA_INPUT_INTEGRITY_CHECK.  It records the logical F32
    // activation columns before a raw-FPGA matmul, independent of padding in
    // src1->nb[1], so a host output store cannot silently damage the next
    // consumer in a multi-column graph.
    std::vector<uint8_t>       activation_input_snapshot;
    std::vector<block_q8_0_t>  weight_tile_snapshot;
    std::vector<uint8_t>       weight_tensor_snapshot;
    std::vector<float>         contract_actual;
    std::vector<float>         act_scales;
    std::vector<float>         weight_scales;
    std::vector<int32_t>       partial;
    std::vector<float>         accum;
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

static int                 g_mem_fd       = -1;
static volatile dma_ctrl * g_dma          = nullptr;
static volatile uint8_t *  g_vpu          = nullptr;
static uint8_t *           g_ddr          = nullptr;
static void *              g_dma_map_base = nullptr;
static void *              g_vpu_map_base = nullptr;
enum class fpga_mapping_kind : uint8_t {
    UNKNOWN = 0,
    UIO_PHYSICAL,
    DEVMEM_PHYSICAL,
};

static const char * fpga_mapping_kind_name(fpga_mapping_kind kind) {
    switch (kind) {
        case fpga_mapping_kind::UIO_PHYSICAL:
            return "uio_physical";
        case fpga_mapping_kind::DEVMEM_PHYSICAL:
            return "devmem_physical";
        default:
            return "unknown";
    }
}

static void *                                 g_ddr_map_base           = nullptr;
static size_t                                 g_dma_map_size           = 0;
static size_t                                 g_vpu_map_size           = 0;
static size_t                                 g_ddr_map_size           = 0;
static size_t                                 g_ddr_advertised_size    = 0;
static size_t                                 g_ddr_requested_map_size = DDR_REQUIRED_BYTES;
static std::string                            g_dma_map_source;
static std::string                            g_vpu_map_source;
static std::string                            g_ddr_map_source;
// FPGA_WEIGHT_PATH_BENCH is an in-process, passive diagnostic.  It records
// only the live tiler geometry and keeps tensor references private to this
// process; it never serializes or logs host pointers.
typedef struct {
    const struct ggml_tensor * src0;
    const void *               weight_data_base;
    int64_t                    source_k;
    int64_t                    source_n;
    size_t                     source_nb0;
    size_t                     source_nb1;
    size_t                     source_span_bytes;
    int64_t                    row0;
    int                        rows;
    int64_t                    k_block0;
    int                        group_blocks;
    int                        group_beats;
    size_t                     payload_bytes;
} fpga_weight_path_bench_job_t;

typedef struct {
    bool                                    enabled;
    bool                                    passive_mapping;
    bool                                    replayed;
    int                                     graph_seq;
    uint64_t                                payload_bytes;
    std::vector<fpga_weight_path_bench_job_t> jobs;
    std::vector<uint32_t>                   cached_payload;
} fpga_weight_path_bench_trace_t;

static fpga_weight_path_bench_trace_t g_weight_path_bench = {};
// This is set by the mapper that created g_ddr, not inferred later from a
// display string or the O_SYNC open flag.  P2's no-msync policy is legal for an admitted bounded physical
// mapping: exact UIO is preferred, with /dev/mem compatibility using O_SYNC.
static fpga_mapping_kind                      g_ddr_mapping_kind = fpga_mapping_kind::UNKNOWN;
static pthread_mutex_t                        g_mutex            = PTHREAD_MUTEX_INITIALIZER;
static fpga_scratch_t                         g_scratch;
static std::vector<fpga_weight_cache_entry_t> g_weight_cache;

static long long              g_fpga_start_us = 0;
static long long              g_fpga_count    = 0;
static long long              g_fpga_vpu_runs = 0;
// Count host-hook decisions independently from VPU runs.  A successful text
// response alone does not prove that every eligible Q8 GEMV used the FPGA:
// GGML may legally retain attention and the vocabulary projection on CPU.
// These atomics give an end-of-process, per-run coverage proof without
// changing routing or numerical results.
static std::atomic<long long> g_matmul_hook_calls{ 0 };
static std::atomic<long long> g_q8_candidate_calls{ 0 };
static std::atomic<long long> g_q8_intentional_cpu_bypass_calls{ 0 };
static std::atomic<long long> g_q8_unavailable_cpu_fallback_calls{ 0 };
static long long              g_reject_count                  = 0;
static long long              g_activation_cache_hits         = 0;
static long long              g_activation_cache_misses       = 0;
static long long              g_weight_cache_builds           = 0;
static long long              g_weight_cache_hits             = 0;
static long long              g_weight_cache_misses           = 0;
static long long              g_weight_cache_bytes            = 0;
static long long              g_weight_cache_lookup_us        = 0;
static long long              g_weight_cache_crc_us           = 0;
static long long              g_weight_pack_us                = 0;
static long long              g_last_token_us                 = 0;
static int                    g_last_token_seq                = INT_MIN;
static long long              g_token_matmuls                 = 0;
static int64_t                g_last_epoch_first_hook_m       = 0;
static long long              g_vocab_projection_bypass_count = 0;

static int      g_vpu_max_rows                  = VPU_DEFAULT_ROWS;
static int      g_vpu_max_beats                 = VPU_DEFAULT_BEATS;
static int      g_vpu_max_cols                  = VPU_DEFAULT_COLS;
static int      g_packed_q8_supported           = 0;
static int      g_packed_q8_max_blocks          = 1;
static int      g_packed_q8_result_words        = VPU_DEFAULT_ROWS;
static bool     g_vpu_pingpong_supported        = false;
static bool     g_vpu_descriptor_supported      = false;
static bool     g_spu_q8_scale_stream_supported = false;
static uint32_t g_p2_stream_abi_signature       = 0;
static uint32_t g_p3_split_scale_abi_signature  = 0;
static uint32_t g_spu_stream_status             = 0;
static uint32_t g_spu_word_capacity             = 0;
static bool     g_p3_split_scale_requested      = false;
static bool     g_p3_split_scale_active         = false;
// Set only after mappings, identity/capability admission, and host self-tests
// have all completed.  A second fpga_init() must observe this before changing
// any route, P3 state, counters, mapping, MMIO, or DMA state.
static bool     g_fpga_init_complete             = false;
static bool     g_p3_split_scale_admitted        = false;
static bool     g_p3_mode_committed             = false;
static int      g_committed_stream_mode        = -1;
static long long g_p3_jobs                      = 0;
static long long g_p3_param_dma_bytes           = 0;
static long long g_p3_scratch_dma_bytes         = 0;
static long long g_p3_retire_pass_logs          = 0;
static long long g_p3_retire_pass_suppressed    = 0;
// This diagnostic is intentionally opt-in: the production verifier performs
// no clock calls unless FPGA_P3_RETIRE_TIMING=1 is supplied at fresh init.
// Its scope is host monotonic elapsed time from immediately before the existing
// nine MMIO reads through the existing exact-retirement predicate. It does not
// measure device-internal latency.
static bool     g_p3_retire_timing_enabled      = false;
static uint64_t g_p3_retire_timing_calls        = 0;
static uint64_t g_p3_retire_timing_passes       = 0;
static uint64_t g_p3_retire_timing_failures     = 0;
static uint64_t g_p3_retire_timing_valid_samples = 0;
static uint64_t g_p3_retire_timing_clock_errors = 0;
static uint64_t g_p3_retire_timing_mmio_reads   = 0;
static uint64_t g_p3_retire_timing_core_total_ns = 0;
static uint64_t g_p3_retire_timing_core_min_ns  = 0;
static uint64_t g_p3_retire_timing_core_max_ns  = 0;
static constexpr long long P3_PRODUCTION_RETIRE_LOG_INTERVAL = 256;
static bool     g_spu_silu_supported            = false;
static bool     g_spu_rmsnorm_supported         = false;
static bool     g_spu_rope_supported            = false;
static bool     g_spu_softmax_supported         = false;
static bool     g_pingpong_scheduler_enabled    = false;
// Existing capability bits do not prove that an inactive-bank write select is
// harmless while VPU is active. P1 is admitted by default only on the
// validated production ping-pong route; FPGA_P2_INPUT_PRELOAD=0 opts out.
static bool     g_p2_input_preload_enabled      = false;
// File-only aggregate telemetry for a completed graph sequence.  This is
// opt-in because even a host timestamp belongs outside the production fast
// path unless the owner is actively measuring scheduler behavior.
static bool     g_p1_sched_summary_enabled      = false;
// FPGA_TOKEN_TIMING controls TOKEN_TIMING records. Aggregate collection stays
// active when BOTTLENECK_SUMMARY or PINGPONG_TIMING needs the same counters.
static bool     g_token_timing_enabled           = false;
static bool     g_token_timing_collection_enabled = false;
static bool     g_pingpong_timing_enabled        = false;
// Emits only aggregate token/category/scheduler/ZDMA summaries. It is meant
// for long throughput runs where per-tile PINGPONG/P2_EVT logging would
// materially perturb host scheduling.
static bool     g_bottleneck_summary_enabled     = false;
static int      g_summary_detail_every           = FPGA_DEFAULT_SUMMARY_DETAIL_EVERY;
static long long g_summary_detail_decode_tokens  = 0;
static uint32_t g_next_job_id                    = 1;

static bool              g_dma_timing_enabled                         = false;
static bool              g_ip_timing_enabled                          = false;
static bool              g_stage_timing_enabled                       = false;
static bool              g_status_stderr                              = false;
static bool              g_trace_data_enabled                         = false;
static bool              g_dma_trace_enabled                          = false;
static bool              g_cleanup_done                               = false;
static bool              g_abort_on_cpu_fallback                      = true;
static bool              g_uio_inventory_logged                       = false;
static bool              g_allow_devmem_fallback                      = false;
static bool              g_allow_vpu_devmem_compat                    = true;
static bool              g_strict_coherency                           = false;
static bool              g_coherency_platform_whitelisted             = false;
static bool              g_run_coherency_stress                       = false;
static bool              g_ddr_msync_unsupported_logged               = false;
static bool              g_ddr_msync_unavailable                      = false;
static bool              g_weight_cache_enabled                       = false;
static bool              g_p2_weight_residency_env_requested           = false;
static bool              g_p2_weight_residency_requested              = false;
static bool              g_p2_weight_residency_enabled                = false;
static bool              g_p2_weight_residency_diagnostic             = false;
static bool              g_p2_residency_trace_enabled                  = false;
static bool              g_p2_residency_verify_metadata               = false;
static long long         g_p2_weight_residency_budget_mb              = 0;
static uint64_t          g_p2_weight_residency_epoch                  = 0;
static std::array<fpga_p2_resident_tile_t, P2_WEIGHT_RESIDENCY_SLOT_CAPACITY> g_p2_resident_tiles = {};
static std::array<uint32_t, P2_WEIGHT_RESIDENCY_INDEX_BUCKETS> g_p2_residency_index = {};
static size_t            g_p2_resident_tile_count                     = 0;
static size_t            g_p2_residency_next_slot                     = 0;
static uint32_t          g_p2_residency_next_off                      = WEIGHT_CACHE_BASE;
static long long         g_p2_residency_builds                        = 0;
static long long         g_p2_residency_hits                          = 0;
static long long         g_p2_residency_misses                        = 0;
static long long         g_p2_residency_build_failures                = 0;
static long long         g_p2_residency_logical_bytes                 = 0;
static long long         g_p2_residency_miss_alignment                = 0;
static long long         g_p2_residency_miss_shape                    = 0;
static long long         g_p2_residency_miss_collision                = 0;
static long long         g_p2_residency_miss_poison                   = 0;
static long long         g_p2_residency_miss_stale                    = 0;
static long long         g_p2_residency_miss_mismatch                 = 0;
static long long         g_p2_residency_miss_capacity                 = 0;
static long long         g_p2_residency_miss_quiescence               = 0;
static long long         g_p2_residency_miss_range                    = 0;
static long long         g_p2_residency_miss_verify                   = 0;
static long long         g_p2_residency_probe_count                   = 0;
static long long         g_p2_residency_probe_exhausted               = 0;
static long long         g_p2_residency_host_metadata_hits            = 0;
static long long         g_p2_residency_host_metadata_invalidations   = 0;
static long long         g_p2_residency_volatile_ddr_reads            = 0;
static long long         g_p2_residency_build_us                      = 0;
static long long         g_p2_residency_select_us                     = 0;
static long long         g_p2_residency_metadata_validate_us          = 0;
static long long         g_p2_residency_resident_param_us             = 0;
static long long         g_p2_residency_direct_weight_pack_us         = 0;
static uint64_t          g_p2_residency_direct_weight_pack_bytes      = 0;
// V78 deliberately accelerates only the CPU stores that prepare the existing
// P2 WEIGHT window.  The worker never owns a DMA, descriptor, IP, or SPU
// action; g_mutex still serializes every hardware job around this local work.
static int               g_p2_pack_workers_requested                  = 2;
static int               g_p2_pack_workers_active                     = 2;
static long long         g_p2_pack_parallel_jobs                       = 0;
static uint64_t          g_p2_pack_parallel_bytes                      = 0;
static long long         g_p2_pack_serial_threshold_skips              = 0;
static long long         g_p2_pack_main_us                             = 0;
static long long         g_p2_pack_helper_service_us                   = 0;
static long long         g_p2_pack_caller_wait_us                      = 0;
static constexpr size_t  FPGA_P2_PACK_PARALLEL_MIN_BYTES               = 256U * 1024U;

// A task has no device ownership: it is a prevalidated, disjoint range of
// volatile WEIGHT words in the already-admitted DDR mapping.  The caller owns
// all range checks and does not start DMA until it observes completion for the
// matching generation.
typedef struct {
    const struct ggml_tensor * src0;
    const void *               weight_data_base;
    int64_t                    row0;
    int64_t                    k_block0;
    int                        rows;
    int                        group_blocks;
    int                        group_beats;
    size_t                     pair_begin;
    size_t                     pair_end;
    volatile uint32_t *        dst_words;
    size_t                     expected_words;
    uint64_t                   generation;
} fpga_p2_pack_worker_task_t;

static pthread_mutex_t        g_p2_pack_worker_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t         g_p2_pack_worker_task_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t         g_p2_pack_worker_done_cv = PTHREAD_COND_INITIALIZER;
static pthread_t              g_p2_pack_worker_thread = {};
static fpga_p2_pack_worker_task_t g_p2_pack_worker_task = {};
static bool                   g_p2_pack_worker_created = false;
static bool                   g_p2_pack_worker_stop_requested = false;
static bool                   g_p2_pack_worker_task_pending = false;
static bool                   g_p2_pack_worker_busy = false;
static uint64_t               g_p2_pack_worker_next_generation = 0;
static uint64_t               g_p2_pack_worker_completed_generation = 0;
static bool                   g_p2_pack_worker_completed_success = false;
static size_t                 g_p2_pack_worker_completed_words = 0;
static long long              g_p2_pack_worker_completed_service_us = 0;
static long long         g_p2_residency_avoided_cpu_pack_bytes        = 0;
// Residency still transfers each selected tile from DDR to the IP.  Keep this
// explicit so a diagnostic log never claims a traffic reduction it cannot
// prove from the current RTL/descriptor contract.
static long long         g_p2_residency_avoided_ddr_to_ip_bytes       = 0;
static bool              g_weight_cache_full_logged                   = false;
static bool              g_weight_cache_crc_verify_each_lookup        = false;
static bool              g_activation_cache_enabled                   = false;
// Qualification-only input ownership proof.  This is intentionally opt-in:
// it copies each logical F32 activation matrix before/after a raw FPGA
// matmul, which is appropriate for validating M>1 graph layouts but not for
// production throughput measurement.
static bool              g_activation_input_integrity_check           = false;
static bool              g_contract_check_abort                       = false;
static bool              g_contract_forensic_replay                   = true;
// Full byte-for-byte staging scans are valuable for a single forensic replay,
// but reading every ACT/WEIGHT byte twice per tile through the UIO mapping is
// not part of the numerical contract and dominates C0 timing.  Keep the
// bounded ordering fence on every launch; make the exhaustive scan explicit.
static bool              g_contract_deep_staging                      = false;
// Contract mode must prove the raw FPGA path without taking ownership of the
// GGML destination tensor.  In this explicit shadow mode, the host retains
// its hardware result in contract_actual for checking while the upstream
// threaded kernel writes dst.  It is not a hardware-unavailable CPU fallback
// and is never the production-acceleration route.
static bool              g_contract_cpu_shadow_dst                    = false;
// Raw-F32 propagation is useful only when deliberately investigating a
// downstream attention/KV divergence.  It is not a C0/C1 success criterion:
// a raw/value contract can pass while a later attention score diverges.  Keep
// it separate from the ordinary C0/C1 shadow route so a stale legacy setting
// cannot turn a validation command into that forensic experiment accidentally.
static bool              g_contract_raw_propagation_diagnostic        = false;
static bool              g_contract_legacy_canonical_override_ignored = false;
// Diagnostic-only: validate the Q8_0 source tensors in the normal GGML
// execution order, but return control to CPU before any model GEMV is sent to
// ZDMA/VPU.  This distinguishes a bad/mutable GGUF source from a fault that
// only appears after raw FPGA launches.  The ggml caller skips FPGA init in
// this mode, so source audit never maps or self-tests board hardware.
static bool              g_q8_source_audit_only                       = false;
static bool              g_q8_source_audit_mode_logged                = false;
static bool              g_clear_result_before_run                    = false;
static bool              g_contract_raw_repair_enabled                = false;
// The fused path has not yet passed an end-to-end language/logit A/B test on
// the legacy board bitstream.  Keep the explicit partial -> scale ->
// accumulate order as the production default.  Fusion remains an opt-in
// performance experiment after that test passes.
static bool              g_fuse_raw_result_accum                      = false;
// The tied embedding / vocabulary projection has 262144 rows for Gemma3-1B.
// It is the final, top-token-sensitive operation and the current legacy
// bitstream advertises neither the required stream protocol nor bitstream ID.
// Keep it on the established CPU GGML path by default until FPGA logits pass
// an end-to-end A/B comparison.  All block GEMVs remain eligible for FPGA.
static bool              g_vocab_projection_cpu_bypass                = true;
// A bitstream without the required identity/protocol has already produced
// raw-contract failures on the board.  Primary inference must prefer a
// readable answer over silently consuming corrupt FPGA outputs.  Contract
// runs remain allowed so the fault can be isolated without changing this
// safety policy.
static bool              g_legacy_raw_cpu_bypass                      = false;
static long long         g_legacy_raw_cpu_bypass_count                = 0;
static int               g_profile_every                              = FPGA_DEFAULT_PROFILE_EVERY;
static int               g_ip_status_every                            = FPGA_DEFAULT_STATUS_EVERY;
static int               g_detail_every                               = FPGA_DEFAULT_DETAIL_EVERY;
static int               g_contract_check_limit                       = 0;
// P2 is deliberately independent from raw C0: it validates the canonical
// VPU->SPU scale/accumulate ABI and never consumes the raw RESULT window.
static int               g_pl_scale_contract_check_limit              = 0;
static int               g_contract_raw_retry_limit                   = 1;
static int               g_runtime_max_rows                           = VPU_SAFE_RUNTIME_ROWS;
static int64_t           g_vocab_projection_min_n                     = 65536;
static long long         g_dma_timeout_us                             = FPGA_DEFAULT_DMA_TIMEOUT_US;
static long long         g_ip_timeout_us                              = FPGA_DEFAULT_IP_TIMEOUT_US;
static size_t            g_zdma_max_transfer_bytes                    = FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES;
static long long         g_large_matrix_min_macs                      = FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS;
static double            g_fpga_clock_mhz                             = 0.0;
static double            g_contract_atol                              = 1.0e-3;
static double            g_contract_rtol                              = 1.0e-4;
static long long         g_contract_checks_done                       = 0;
static long long         g_pl_scale_jobs                              = 0;
static long long         g_pl_scale_banks                             = 0;
static long long         g_pl_scale_stream_drops                      = 0;
static long long         g_pl_scale_stream_errors                     = 0;
static long long         g_contract_raw_mismatches                    = 0;
static long long         g_contract_raw_repairs                       = 0;
static long long         g_contract_value_mismatches                  = 0;
static long long         g_contract_cpu_shadow_dst_values             = 0;
static long long         g_contract_staging_restage_count             = 0;
// C0 is a bounded hardware qualification. Once its requested number of
// checked GEMVs has completed, later graph GEMVs must execute only in the
// native CPU backend. Launching unverified VPU work after that boundary can
// turn a later CPU-side failure into misleading C0 evidence.
static long long         g_contract_limit_cpu_bypass_count            = 0;
static bool              g_contract_limit_cpu_bypass_logged           = false;
static long long         g_q8_source_audit_checks                     = 0;
static long long         g_q8_source_audit_failures                   = 0;
static long long         g_activation_input_integrity_checks          = 0;
static long long         g_activation_input_integrity_failures        = 0;
static long long         g_activation_scale_fp16_overflows            = 0;
// Set only while a C0 source preflight rejects an immutable Q8 tensor.  This
// lets the outer hook report the real cause instead of mislabelling it as a
// ZDMA/VPU transfer failure.
static bool              g_contract_source_validation_failed          = false;
// Set by llama-model-loader only after its complete upstream tensor validation
// succeeds.  C0 starts before normal FPGA initialization, so this handshake
// prevents a partially copied frontend build from bypassing the loader gate
// and reaching MY_IP/ZDMA with an invalid model source.
static std::atomic<bool> g_contract_loader_validation_passed{ false };

typedef struct {
    uint32_t  status;
    uint32_t  isr;
    uint32_t  ctrl2;
    long long polls;
    bool      saw_enabled;
} zdma_completion_info_t;

typedef struct {
    bool               valid;
    unsigned long long sequence;
    char               tag[48];
    uint64_t           src_phys;
    uint64_t           dst_phys;
    size_t             bytes;
    uint32_t           pre_status;
    uint32_t           pre_isr;
    uint32_t           pre_ctrl2;
    uint32_t           total_bytes_before_clear;
    uint32_t           pre_vpu_status;
    uint32_t           pre_vpu_progress;
    uint32_t           post_status;
    uint32_t           post_isr;
    uint32_t           post_ctrl2;
    uint32_t           total_bytes_after_transfer;
    uint32_t           post_vpu_status;
    uint32_t           post_vpu_progress;
    long long          elapsed_us;
    long long          polls;
    bool               saw_enabled;
} fpga_dma_trace_record_t;

static fpga_dma_trace_record_t g_dma_trace[FPGA_DMA_TRACE_DEPTH] = {};
static unsigned long long      g_dma_trace_sequence              = 0;

typedef struct {
    bool    valid;
    int     local_row;
    int     group_block;
    int64_t global_row;
    int64_t k_block;
} fpga_raw_mismatch_location_t;

static uint32_t  g_weight_cache_next_off   = WEIGHT_CACHE_BASE;
static uint32_t  g_weight_cache_end_off    = WEIGHT_CACHE_BASE;
static uint32_t  g_stream_protocol_version = 0;
static uint32_t  g_bitstream_id            = 0;
static long long g_weight_cache_budget_mb  = 0;

static int       g_current_layer_id       = 0;
static int       g_current_seq_pos        = 0;
static int       g_is_attention_op        = 0;
static long long g_attention_bypass_count = 0;
// These values are set while g_mutex serializes the owner matmul.  They are
// copied into each P2 tile job during preparation, so buffered telemetry
// remains attributable even when a later job has been prepared.
static unsigned long long g_next_matmul_call_id      = 1U;
static unsigned long long g_active_matmul_call_id    = 0U;
static int                g_active_matmul_graph_seq  = 0;
static int                g_active_matmul_layer_id   = 0;
static int64_t            g_active_matmul_shape_k    = 0;
static int64_t            g_active_matmul_shape_n    = 0;
static int64_t            g_active_matmul_shape_m    = 0;
static bool               g_active_matmul_cpu_shadow = false;
static bool               g_active_matmul_pingpong   = false;
static const char *       g_active_matmul_tensor_name = nullptr;

typedef struct {
    bool      active;
    int       graph_seq;
    long long matmuls;
    long long vpu_runs;
    long long pingpong_pairs;
    long long preload_attempts;
    long long preload_admitted_while_active;
    long long preload_terminal_skip;
    long long serial_submit_after_no_preload;
    long long input_preload_us;
    long long preload_launch_bubble_us;
    long long ip_compute_us;
    long long dma_act_us;
    long long dma_weight_us;
    long long matrix_wall_us;
} fpga_p1_sched_summary_t;

static fpga_p1_sched_summary_t g_p1_sched_summary = {};

enum fpga_bottleneck_category_t : int {
    FPGA_BOTTLENECK_ATTN = 0,
    FPGA_BOTTLENECK_FFN_GATE,
    FPGA_BOTTLENECK_FFN_UP,
    FPGA_BOTTLENECK_FFN_DOWN,
    FPGA_BOTTLENECK_OTHER,
    FPGA_BOTTLENECK_CATEGORY_COUNT,
};

typedef struct {
    bool      active;
    int       graph_seq;
    int64_t   first_m;
    long long start_mono_us;
    long long last_update_mono_us;
    long long first_ip_launch_mono_us;
    long long last_ip_output_ready_mono_us;
    long long matmuls;
    long long vpu_runs;
    long long prep_us;
    long long prep_weight_select_us;
    long long prep_direct_weight_pack_us;
    long long prep_scale_pack_us;
    long long prep_act_pack_us;
    long long matmul_wall_us;
    long long act_dma_us;
    long long weight_dma_us;
    long long scale_dma_us;
    long long preload_us;
    long long ip_compute_us;
    long long ip2host_dma_us;
    long long host_read_us;
    long long host_accum_us;
    long long scheduler_prepare_overlap_us;
    long long scheduler_prepare_late_us;
    long long scheduler_prepare_headroom_us;
    long long scheduler_preload_overlap_us;
    long long scheduler_output_to_launch_us;
    long long scheduler_retire_to_launch_us;
    long long scheduler_handoffs;
    long long scheduler_prepare_late_jobs;
    long long scheduler_preload_overlap_jobs;
    long long zdma_descriptors;
    long long zdma_polls;
    long long zdma_zero_poll_descriptors;
    long long zdma_saw_enabled_descriptors;
    long long zdma_elapsed_us;
    size_t    zdma_bytes;
    long long zdma_act_descriptors;
    long long zdma_weight_descriptors;
    long long zdma_scale_descriptors;
    long long zdma_result_descriptors;
    long long zdma_other_descriptors;
    long long category_matmuls[FPGA_BOTTLENECK_CATEGORY_COUNT];
    long long category_runs[FPGA_BOTTLENECK_CATEGORY_COUNT];
    long long category_wall_us[FPGA_BOTTLENECK_CATEGORY_COUNT];
    long long category_prep_us[FPGA_BOTTLENECK_CATEGORY_COUNT];
    long long category_compute_us[FPGA_BOTTLENECK_CATEGORY_COUNT];
    long long category_dma_us[FPGA_BOTTLENECK_CATEGORY_COUNT];
    long long bank_h2ip_us[2];
    long long bank_compute_us[2];
    long long bank_ip2host_us[2];
    long long bank_host_read_us[2];
    long long bank_jobs[2];
    size_t    activation_bytes;
    size_t    weight_bytes;
    size_t    scale_bytes;
    size_t    result_bytes;
} fpga_token_timing_t;

static fpga_token_timing_t g_token_timing = {};
// These counters are exported to llama_perf_sampler_print(). They represent
// only completed single-token decode evaluations, never prefill batches.
static fpga_perf_decode_data g_fpga_perf_decode = {};

static void fpga_p1_sched_summary_emit(const char * reason);

static void fpga_p1_sched_summary_begin_graph(int graph_seq);

static long long now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (long long) tv.tv_sec * 1000000LL + (long long) tv.tv_usec;
}

// P1 breadcrumbs are diagnostic-only. Ordinary records require
// FPGA_P1_PRELOAD_TRACE=1; force is reserved for failures and flushes them.
static void fpga_p1_preload_breadcrumb(bool force, const char * fmt, ...);

// Wall clock is useful for long-running summaries, but it can step while NTP
// disciplines the board.  P2 event durations must instead use a monotonic
// clock so adjacent event deltas remain meaningful.
static long long monotonic_now_us(void) {
    struct timespec ts = {};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (long long) ts.tv_sec * 1000000LL + (long long) ts.tv_nsec / 1000LL;
}

// Keep the default path free of telemetry clock reads.  Call sites may always
// capture through this helper; it returns a neutral value while P2_EVT is off.
static long long p2_event_now_us(void);

static uint64_t fpga_saturating_add_u64(uint64_t lhs, uint64_t rhs) {
    return lhs > UINT64_MAX - rhs ? UINT64_MAX : lhs + rhs;
}

// This helper is called only from the opt-in P3 retirement timing path.  A
// failed or unusable clock sample is telemetry-only: it cannot alter the
// verifier's nine reads, predicate, ordering, or failure result.
static bool fpga_p3_retire_timing_now_ns(uint64_t * out_ns);

static const char * p2_bank_label(int bank) {
    return (bank & 1) == 0 ? "PING" : "PONG";
}

static fpga_bottleneck_category_t fpga_bottleneck_category(const char * tensor_name);

static void fpga_token_timing_reset(void);

static bool fpga_token_timing_emit(int next_graph_seq,
                                   int64_t     ubatch_tokens,
                                   const char * reason,
                                   long long    end_mono_us,
                                   bool         force_flush = false);

static bool fpga_token_timing_emit_final(long long end_mono_us);

static void fpga_token_timing_begin(int graph_seq, int64_t m);

static void fpga_token_timing_accumulate(const fpga_stage_totals_t & totals, long long matmul_wall_us,
                                                 const char * tensor_name);

static void p2_event_trace(const fpga_tile_job_t & job,
                           const char *            event,
                           long long               event_mono_us,
                           const char *            duration_name,
                           long long               duration_us);

static void p2_matmul_finish_trace(long long event_mono_us, long long matmul_start_us, bool contract_tile_only);

static bool env_flag_enabled(const char * name) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0 ||
           strcmp(value, "yes") == 0 || strcmp(value, "YES") == 0 || strcmp(value, "on") == 0 ||
           strcmp(value, "ON") == 0;
}

static bool env_flag_disabled(const char * name) {
    const char * value = getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0 ||
           strcmp(value, "no") == 0 || strcmp(value, "NO") == 0 || strcmp(value, "off") == 0 ||
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
    char * end        = nullptr;
    errno             = 0;
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
    char * end             = nullptr;
    errno                  = 0;
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
    char * end          = nullptr;
    errno               = 0;
    const double parsed = strtod(value, &end);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return std::max(min_value, std::min(parsed, max_value));
}

static void fpga_fatal(const char * fmt, ...) {
    g_committed_stream_mode = -1;
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
    g_summary_detail_after_error = true;
    if (g_token_timing_collection_enabled && g_token_timing.active) {
        fpga_token_timing_emit(g_token_timing.graph_seq, 0, "error", monotonic_now_us());
    }
    fpga_p2_init_breadcrumb("phase=failure reason=fpga_fatal");
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
    // MY_IP, ZDMA, or DDR high. A requested qualification takes precedence:
    // the existing contract gate must defer initialization until validation,
    // where fpga_init() rejects the mutually exclusive environment settings.
    const bool qualification_requested = env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000) > 0 ||
                                         env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000) > 0;
    return env_flag_enabled("FPGA_SOURCE_AUDIT_ONLY") && !qualification_requested ? 1 : 0;
}

extern "C" int fpga_contract_check_requested(void) {
    // This query is called before fpga_init() by llama-cli. It deliberately
    // covers both raw C0 and P2 because the existing frontend uses this
    // board-free gate to defer initialization until loader validation passes.
    return (env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000) > 0 ||
            env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000) > 0) ?
               1 :
               0;
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
    return ((uint32_t) input_state << (4U * b)) | ((uint32_t) output_state << (16U + 4U * b));
}

static void vpu_select_banks(int wr_bank, int rd_bank) {
    if (!g_vpu_pingpong_supported) {
        return;
    }
    vpu_wr32(REG_BANK, ((uint32_t) (wr_bank & 1)) | (((uint32_t) (rd_bank & 1)) << 1));
}

static void vpu_write_tile_descriptor(const fpga_tile_job_t & job,
                                      fpga_slot_state_t       input_state,
                                      fpga_slot_state_t       output_state,
                                      uint32_t                flags) {
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

struct fpga_p2_descriptor_words_t {
    uint32_t bank;
    uint32_t bank_stat;
    uint32_t job_id;
    uint32_t slot_state;
    uint32_t tensor_id;
    uint32_t row0;
    uint32_t k_block0;
    uint32_t group_blocks;
    uint32_t token_id;
    uint32_t desc_flags;
};

static void fpga_p2_post_spu_descriptor_write_word(const fpga_tile_job_t & job,
                                                   const char *            phase,
                                                   uint32_t                off,
                                                   uint32_t                value) {
    fpga_p2_boundary_marker(
        "P2_POST_SPU_DESCRIPTOR edge=before phase=%s job=%u bank=%d tile=%u off=0x%08x value=0x%08x", phase, job.job_id,
        job.bank, job.tile_id, off, value);
    vpu_wr32(off, value);
    mmio_fence();
    fpga_p2_boundary_marker("P2_POST_SPU_DESCRIPTOR edge=after phase=%s job=%u bank=%d tile=%u off=0x%08x value=0x%08x",
                            phase, job.job_id, job.bank, job.tile_id, off, value);
}

// Keep the descriptor register order unchanged.  v60 only replaces the final
// post-SPU_OUT writes with individually bracketed writes when the explicit
// boundary diagnostic is enabled.
static bool fpga_write_post_spu_descriptor(const fpga_tile_job_t & job,
                                           fpga_slot_state_t       input_state,
                                           fpga_slot_state_t       output_state,
                                           uint32_t                flags,
                                           const char *            phase,
                                           bool                    final_free_free) {
    // Preserve vpu_write_tile_descriptor()'s capability gate exactly: an
    // unsupported descriptor block must not receive diagnostic MMIO writes.
    if (!g_vpu_descriptor_supported) {
        return true;
    }
    // P1 cannot infer retirement from a write alone: its deferred launch
    // needs an exact FREE/FREE readback even when verbose boundary tracing is
    // off.  Other descriptor phases keep the established quiet fast path.
    if (!g_p2_boundary_diagnostics_enabled && !(final_free_free && g_p2_input_preload_enabled)) {
        vpu_write_tile_descriptor(job, input_state, output_state, flags);
        return true;
    }

    const uint32_t slot_state = fpga_slot_state_word(job.bank, input_state, output_state);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_JOB_ID, job.job_id);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_SLOT_STATE, slot_state);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_TENSOR_ID, job.tensor_id);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_ROW0, (uint32_t) job.row0);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_K_BLOCK0, (uint32_t) job.k_block0);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_GROUP_BLOCKS, (uint32_t) job.group_blocks);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_TOKEN_ID, (uint32_t) g_current_seq_pos);
    fpga_p2_post_spu_descriptor_write_word(job, phase, REG_DESC_FLAGS, flags);

    if (!final_free_free) {
        return true;
    }

    // The final FREE/FREE readback is a diagnostic fail-closed gate.  A stale
    // descriptor or bank must not be reported as a completed P2 tile.
    fpga_p2_boundary_marker(
        "P2_POST_SPU_FREE_READBACK edge=before phase=%s job=%u bank=%d tile=%u expected_slot_state=0x%08x "
        "expected_desc_flags=0x%08x",
        phase, job.job_id, job.bank, job.tile_id, slot_state, flags);
    mmio_fence();
    const fpga_p2_descriptor_words_t actual = {
        vpu_rd32(REG_BANK),      vpu_rd32(REG_BANK_STAT),  vpu_rd32(REG_JOB_ID),   vpu_rd32(REG_SLOT_STATE),
        vpu_rd32(REG_TENSOR_ID), vpu_rd32(REG_ROW0),       vpu_rd32(REG_K_BLOCK0), vpu_rd32(REG_GROUP_BLOCKS),
        vpu_rd32(REG_TOKEN_ID),  vpu_rd32(REG_DESC_FLAGS),
    };
    const uint32_t bank          = (uint32_t) (job.bank & 1);
    const uint32_t expected_bank = bank | (bank << 1);
    const bool     exact_match = (actual.bank & 0x3U) == expected_bank && (actual.bank_stat & 0x3U) == expected_bank &&
                                 actual.job_id == job.job_id && actual.slot_state == slot_state &&
                                 actual.tensor_id == job.tensor_id && actual.row0 == (uint32_t) job.row0 &&
                                 actual.k_block0 == (uint32_t) job.k_block0 &&
                                 actual.group_blocks == (uint32_t) job.group_blocks &&
                                 actual.token_id == (uint32_t) g_current_seq_pos && actual.desc_flags == flags;
    fpga_p2_boundary_marker(
        "P2_POST_SPU_FREE_READBACK edge=after phase=%s job=%u bank=%d tile=%u expected_bank=0x%08x "
        "expected_slot_state=0x%08x expected_desc_flags=0x%08x actual_bank=0x%08x actual_bank_stat=0x%08x "
        "actual_job_id=0x%08x actual_slot_state=0x%08x actual_tensor_id=0x%08x actual_row0=0x%08x "
        "actual_k_block0=0x%08x actual_group_blocks=0x%08x actual_token_id=0x%08x actual_desc_flags=0x%08x "
        "exact_match=%d",
        phase, job.job_id, job.bank, job.tile_id, expected_bank, slot_state, flags, actual.bank, actual.bank_stat,
        actual.job_id, actual.slot_state, actual.tensor_id, actual.row0, actual.k_block0, actual.group_blocks,
        actual.token_id, actual.desc_flags, exact_match ? 1 : 0);
    if (!exact_match) {
        LOGE(
            "P2_POST_SPU_FREE_READBACK_FAIL phase=%s job=%u bank=%d tile=%u expected_bank=0x%08x "
            "expected_slot_state=0x%08x expected_desc_flags=0x%08x actual_bank=0x%08x actual_bank_stat=0x%08x "
            "actual_job_id=0x%08x actual_slot_state=0x%08x actual_tensor_id=0x%08x actual_row0=0x%08x "
            "actual_k_block0=0x%08x actual_group_blocks=0x%08x actual_token_id=0x%08x actual_desc_flags=0x%08x "
            "action=abort_before_tile_boundary_no_cpu_native",
            phase, job.job_id, job.bank, job.tile_id, expected_bank, slot_state, flags, actual.bank, actual.bank_stat,
            actual.job_id, actual.slot_state, actual.tensor_id, actual.row0, actual.k_block0, actual.group_blocks,
            actual.token_id, actual.desc_flags);
        fpga_p2_boundary_marker(
            "P2_POST_SPU_FREE_READBACK edge=complete status=fail reason=exact_descriptor_or_bank_mismatch phase=%s "
            "job=%u bank=%d tile=%u action=abort_before_tile_boundary_no_cpu_native",
            phase, job.job_id, job.bank, job.tile_id);
        return false;
    }
    return true;
}

static void fpga_p2_descriptor_commit_breadcrumb_before(const fpga_tile_job_t &            job,
                                                        const fpga_p2_descriptor_words_t & expected);

static void fpga_p2_descriptor_commit_breadcrumb_after(const fpga_tile_job_t &            job,
                                                       const fpga_p2_descriptor_words_t & expected,
                                                       const fpga_p2_descriptor_words_t & actual,
                                                       bool                               match);

static bool fpga_commit_p2_dma_filling_descriptor(const fpga_tile_job_t & job) {
    if (!g_p2_init_requested) {
        return true;
    }

    const uint32_t                   bank          = (uint32_t) (job.bank & 1);
    const uint32_t                   expected_bank = bank | (bank << 1);
    const fpga_p2_descriptor_words_t expected      = {
        expected_bank,
        expected_bank,
        job.job_id,
        fpga_slot_state_word(job.bank, FPGA_SLOT_DMA_FILLING, FPGA_SLOT_FREE),
        job.tensor_id,
        (uint32_t) job.row0,
        (uint32_t) job.k_block0,
        (uint32_t) job.group_blocks,
        (uint32_t) g_current_seq_pos,
        0x00000101U,
    };
    fpga_p2_descriptor_commit_breadcrumb_before(job, expected);

    // Commit all posted MY_IP descriptor/bank writes before reading back the
    // exact final DMA_FILLING state that owns the upcoming ACT transfer.
    mmio_fence();
    const fpga_p2_descriptor_words_t actual = {
        vpu_rd32(REG_BANK),      vpu_rd32(REG_BANK_STAT),  vpu_rd32(REG_JOB_ID),   vpu_rd32(REG_SLOT_STATE),
        vpu_rd32(REG_TENSOR_ID), vpu_rd32(REG_ROW0),       vpu_rd32(REG_K_BLOCK0), vpu_rd32(REG_GROUP_BLOCKS),
        vpu_rd32(REG_TOKEN_ID),  vpu_rd32(REG_DESC_FLAGS),
    };
    const bool match = (actual.bank & 0x3U) == expected.bank && (actual.bank_stat & 0x3U) == expected.bank_stat &&
                       actual.job_id == expected.job_id && actual.slot_state == expected.slot_state &&
                       actual.tensor_id == expected.tensor_id && actual.row0 == expected.row0 &&
                       actual.k_block0 == expected.k_block0 && actual.group_blocks == expected.group_blocks &&
                       actual.token_id == expected.token_id && actual.desc_flags == expected.desc_flags;
    fpga_p2_descriptor_commit_breadcrumb_after(job, expected, actual, match);
    if (!match) {
        LOGE(
            "P2 descriptor commit mismatch job=%u bank=%d tile=%u expected bank_bits=0x%08x bank_stat_bits=0x%08x "
            "job_id=0x%08x slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x group_blocks=0x%08x "
            "token_id=0x%08x desc_flags=0x%08x actual bank=0x%08x bank_stat=0x%08x job_id=0x%08x slot_state=0x%08x "
            "tensor_id=0x%08x row0=0x%08x k_block0=0x%08x group_blocks=0x%08x token_id=0x%08x desc_flags=0x%08x "
            "action=abort_before_act_dma no_retry=1 no_reset=1 no_fallback=1",
            job.job_id, job.bank, job.tile_id, expected.bank, expected.bank_stat, expected.job_id, expected.slot_state,
            expected.tensor_id, expected.row0, expected.k_block0, expected.group_blocks, expected.token_id,
            expected.desc_flags, actual.bank, actual.bank_stat, actual.job_id, actual.slot_state, actual.tensor_id,
            actual.row0, actual.k_block0, actual.group_blocks, actual.token_id, actual.desc_flags);
        return false;
    }
    return true;
}

static bool vpu_verify_done_job(const fpga_tile_job_t & job, uint32_t status) {
    if (!g_vpu_pingpong_supported) {
        return true;
    }

    const uint32_t bank_stat = vpu_rd32(REG_BANK_STAT);
    const uint32_t done_job  = g_vpu_descriptor_supported ? vpu_rd32(REG_DONE_JOB) : job.job_id;
    const int      done_bank = (int) ((bank_stat >> 9) & 1U);
    if (done_bank != (job.bank & 1)) {
        LOGE(
            "VPU done bank mismatch tensor=%s tile=%u job=%u expected_bank=%d done_bank=%d status=0x%08x "
            "bank_stat=0x%08x",
            job.src0 ? job.src0->name : "?", job.tile_id, job.job_id, job.bank & 1, done_bank, status, bank_stat);
        return false;
    }
    if (g_vpu_descriptor_supported && done_job != job.job_id) {
        LOGE(
            "VPU done job mismatch tensor=%s tile=%u expected_job=%u done_job=%u bank=%d status=0x%08x "
            "bank_stat=0x%08x active_job=%u",
            job.src0 ? job.src0->name : "?", job.tile_id, job.job_id, done_job, job.bank & 1, status, bank_stat,
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

// P2 packs four {weight_scale_fp16[31:16], activation_scale_fp16[15:0]}
// entries in each 128-bit SPU_PARAM word.  Keep this arithmetic independent
// of mappings and job state so both the producer and the pre-mapping self-test
// share exactly the same checked contract.
typedef struct {
    size_t entries;
    size_t words;
    size_t bytes;
    size_t tail_entries;
} fpga_p2_scale_shape_t;

static inline uint32_t fpga_p2_pack_scale_entry(uint16_t activation_scale, uint16_t weight_scale) {
    return (uint32_t) activation_scale | ((uint32_t) weight_scale << 16);
}

static bool fpga_p2_checked_scale_shape(int                   rows,
                                        int                   group_blocks,
                                        uint32_t              job_result_values,
                                        uint32_t              job_result_words,
                                        size_t                job_scale_bytes,
                                        uint32_t              scale_off,
                                        fpga_p2_scale_shape_t * shape) {
    if (!shape || rows <= 0 || group_blocks <= 0 || (scale_off & 0xFU) != 0U) {
        return false;
    }

    const size_t rows_size         = (size_t) rows;
    const size_t group_blocks_size = (size_t) group_blocks;
    if (rows_size > std::numeric_limits<size_t>::max() / group_blocks_size) {
        return false;
    }
    const size_t entries = rows_size * group_blocks_size;
    if (entries > std::numeric_limits<uint32_t>::max() ||
        entries > std::numeric_limits<size_t>::max() - ((size_t) VPU_RESULT_PACK_LANES - 1U)) {
        return false;
    }
    const size_t words =
        (entries + (size_t) VPU_RESULT_PACK_LANES - 1U) / (size_t) VPU_RESULT_PACK_LANES;
    if (words == 0U || words > std::numeric_limits<uint32_t>::max() ||
        words > std::numeric_limits<size_t>::max() / 16U) {
        return false;
    }
    const size_t bytes = words * 16U;
    if ((bytes & 0xFU) != 0U || job_result_values != (uint32_t) entries ||
        job_result_words != (uint32_t) words || job_scale_bytes != bytes) {
        return false;
    }

    if (words > std::numeric_limits<size_t>::max() / (size_t) VPU_RESULT_PACK_LANES) {
        return false;
    }
    const size_t covered_entries = words * (size_t) VPU_RESULT_PACK_LANES;
    const size_t tail_entries    = covered_entries - entries;
    if (tail_entries > (size_t) VPU_RESULT_PACK_LANES - 1U) {
        return false;
    }

    *shape = {entries, words, bytes, tail_entries};
    return true;
}

static bool fpga_p2_scale_layout_case(int rows, int group_blocks);

static bool fpga_p2_scale_layout_self_test(void);

static uint8_t * ddr_ptr(uint32_t off, size_t bytes) {
    if (!ddr_range_fits(off, bytes)) {
        fpga_fatal("DDR mapped range overflow off=0x%08x bytes=%zu mapped_size=0x%zx", off, bytes, g_ddr_map_size);
    }
    return g_ddr + off;
}

// The direct P2 WEIGHT producer validates its complete range once, then emits
// only naturally aligned volatile 32-bit stores through this pointer.
static volatile uint32_t * ddr_checked_u32_ptr(uint32_t off, size_t bytes) {
    if (off == 0U || bytes == 0U || (off & 0x3U) != 0U || (bytes & 0x3U) != 0U || !ddr_range_fits(off, bytes)) {
        fpga_fatal("DDR u32 pointer precondition failed off=0x%08x bytes=%zu mapped_size=0x%zx", off, bytes,
                   g_ddr_map_size);
    }
    uint8_t * const raw = ddr_ptr(off, bytes);
    if ((reinterpret_cast<uintptr_t>(raw) % alignof(uint32_t)) != 0U) {
        fpga_fatal("DDR u32 pointer alignment failed off=0x%08x bytes=%zu address=%p alignment=%zu", off, bytes,
                   (void *) raw, alignof(uint32_t));
    }
    return reinterpret_cast<volatile uint32_t *>(raw);
}

static inline uint32_t ddr_pack_i8x4_le(const int8_t * lanes) {
    const uint8_t b0 = (uint8_t) lanes[0];
    const uint8_t b1 = (uint8_t) lanes[1];
    const uint8_t b2 = (uint8_t) lanes[2];
    const uint8_t b3 = (uint8_t) lanes[3];
    return ((uint32_t) b0) | ((uint32_t) b1 << 8) | ((uint32_t) b2 << 16) | ((uint32_t) b3 << 24);
}

static inline void ddr_store_i8x16_words(volatile uint32_t * dst, const int8_t * lanes) {
    for (int w = 0; w < 4; ++w) {
        dst[w] = ddr_pack_i8x4_le(lanes + 4 * w);
    }
}

static void ddr_zero_range32(uint32_t off, size_t bytes) {
    if ((off & 0x3U) != 0 || (bytes & 0x3U) != 0) {
        fpga_fatal("DDR zero requires 32-bit alignment off=0x%08x bytes=%zu", off, bytes);
    }
    volatile uint32_t * dst   = (volatile uint32_t *) ddr_ptr(off, bytes);
    const size_t        words = bytes / sizeof(uint32_t);
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
    ddr_store_i8x16_words(dst, lanes);
}

static void ddr_write_u32(uint32_t off, uint32_t value) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR u32 write requires 32-bit alignment off=0x%08x", off);
    }
    volatile uint32_t * dst = (volatile uint32_t *) ddr_ptr(off, sizeof(uint32_t));
    *dst                    = value;
}

static void ddr_read_i32x4(uint32_t off, int32_t out[4]) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR i32x4 read requires 32-bit alignment off=0x%08x", off);
    }
    const volatile uint32_t * src = (volatile const uint32_t *) ddr_ptr(off, 16U);
    for (int w = 0; w < 4; ++w) {
        out[w] = (int32_t) src[w];
    }
}

static const uint32_t * fpga_crc32_table() {
    // This host serializes matmul/cache work with g_mutex, therefore lazy
    // initialization is safe and avoids a large static initializer.
    static uint32_t table[256];
    static bool     initialized = false;
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
        crc                = fpga_crc32_update(crc, zeros, chunk);
        bytes -= chunk;
    }
    return crc;
}

static uint32_t fpga_crc32_ddr(uint32_t off, size_t bytes) {
    // This diagnostic path deliberately reads through the uncached DDR UIO
    // mapping.  It must never be used while constructing the cache: a full
    // cache can be about 1 GiB and the old bit-at-a-time implementation made
    // the first prefill appear to hang the ZCU104.
    const volatile uint8_t * data  = (volatile const uint8_t *) ddr_ptr(off, bytes);
    const uint32_t *         table = fpga_crc32_table();
    uint32_t                 crc   = 0xFFFFFFFFU;
    for (size_t i = 0; i < bytes; ++i) {
        crc = table[(crc ^ (uint32_t) data[i]) & 0xFFU] ^ (crc >> 8);
    }
    return ~crc;
}

static void ddr_write_weight_cache_header(uint32_t off, const fpga_weight_cache_header_t & header) {
    if ((off & 0x3U) != 0U) {
        fpga_fatal("weight-cache header requires 32-bit alignment off=0x%08x", off);
    }
    volatile uint32_t * dst = (volatile uint32_t *) ddr_ptr(off, sizeof(header));
    const uint32_t *    src = (const uint32_t *) &header;
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
    const volatile uint32_t * src = (volatile const uint32_t *) ddr_ptr(off, sizeof(header));
    uint32_t *                dst = (uint32_t *) &header;
    for (size_t i = 0; i < sizeof(header) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    return header;
}

static int64_t ddr_read_spu_q16_row(uint32_t off, uint16_t * row_id, bool trace_q16_verifier_row0 = false) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR SPU row read requires 32-bit alignment off=0x%08x", off);
    }
    const volatile uint32_t * src = (volatile const uint32_t *) ddr_ptr(off, 16U);
    if (trace_q16_verifier_row0) {
        fpga_p2_boundary_marker("P2_Q16_DDR_READ edge=before row=0 word=0 off=0x%08x", off);
    }
    const uint32_t word0 = src[0];
    if (trace_q16_verifier_row0) {
        fpga_p2_boundary_marker("P2_Q16_DDR_READ edge=after row=0 word=0 off=0x%08x value=0x%08x", off, word0);
        fpga_p2_boundary_marker("P2_Q16_DDR_READ edge=before row=0 word=1 off=0x%08x", off + 4U);
    }
    const uint32_t word1 = src[1];
    if (trace_q16_verifier_row0) {
        fpga_p2_boundary_marker("P2_Q16_DDR_READ edge=after row=0 word=1 off=0x%08x value=0x%08x", off + 4U, word1);
        fpga_p2_boundary_marker("P2_Q16_DDR_READ edge=before row=0 word=2 off=0x%08x", off + 8U);
    }
    const uint32_t word2 = src[2];
    if (trace_q16_verifier_row0) {
        fpga_p2_boundary_marker("P2_Q16_DDR_READ edge=after row=0 word=2 off=0x%08x value=0x%08x", off + 8U, word2);
    }
    if (row_id) {
        *row_id = (uint16_t) (word0 & 0xffffU);
    }
    uint64_t accum_u = ((uint64_t) (word0 >> 16)) | ((uint64_t) word1 << 16) | ((uint64_t) (word2 & 0xffffU) << 48);
    return (int64_t) accum_u;
}

// Canonical Protocol-2/VPU2 WEIGHT ABI.  The RTL consumes adjacent words for
// an even/odd row pair at each beat, rather than a complete row followed by
// the next row.  The final odd row always has an allocated, zero companion.


// physical_row permits the sole padded companion row (physical_row == rows)
// for odd logical row counts.  Logical data producers must pass row < rows.



static void fpga_weight_layout_zero_padded_companion(uint32_t base, int rows, int group_beats) {
    if ((rows & 1) == 0) {
        return;
    }
    for (int beat = 0; beat < group_beats; ++beat) {
        uint32_t off = 0;
        if (!fpga_weight_layout_word_offset(base, rows, group_beats, rows, beat, &off)) {
            fpga_fatal("P2 padded WEIGHT companion offset overflow base=0x%08x rows=%d group_beats=%d beat=%d",
                       base, rows, group_beats, beat);
        }
        ddr_zero_range32(off, 16U);
    }
}

static bool fpga_i8x16_le_pack_self_test(void);

static bool fpga_weight_layout_host_self_test(void);

static const char * tensor_name_or_unknown(const struct ggml_tensor * tensor) {
    return (tensor && tensor->name[0] != '\0') ? tensor->name : "?";
}

static int infer_layer_id_from_name(const char * name, int fallback) {
    if (!name || name[0] == '\0') {
        return fallback;
    }

    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) == 1 || sscanf(name, "layers.%d.", &layer) == 1 ||
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

static bool fpga_ddr_iomem_preflight(void) {
    FILE * fp = fopen("/proc/iomem", "r");
    if (!fp) {
        LOGE("cannot open /proc/iomem for DDR ownership preflight errno=%d (%s)", errno, strerror(errno));
        return false;
    }

    const uint64_t requested_begin = DDR_BASE_PHYS;
    const uint64_t requested_end   = DDR_END_EXCLUSIVE;
    char           line[512]       = {};
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long begin         = 0ULL;
        unsigned long long end_inclusive = 0ULL;
        char               owner[256]    = {};
        if (sscanf(line, " %llx-%llx : %255[^\n]", &begin, &end_inclusive, owner) != 3) {
            continue;
        }

        if (strstr(owner, "System RAM") == nullptr) {
            continue;
        }

        const uint64_t ram_begin = (uint64_t) begin;
        const uint64_t ram_end   = end_inclusive == ULLONG_MAX ? UINT64_MAX : (uint64_t) end_inclusive + 1ULL;
        const bool     overlaps  = requested_begin < ram_end && ram_begin < requested_end;
        if (overlaps) {
            fclose(fp);
            LOGE(
                "unsafe FPGA DDR carveout overlaps Linux System RAM: "
                "fpga=[0x%llx,0x%llx) system_ram=[0x%llx,0x%llx); "
                "fix the boot-time device tree or mem= limit before running",
                (unsigned long long) requested_begin, (unsigned long long) requested_end,
                (unsigned long long) ram_begin, (unsigned long long) ram_end);
            return false;
        }
    }

    fclose(fp);
    LOGINIT(
        "DDR ownership preflight passed phys=[0x%llx,0x%llx) "
        "region_size=0x%zx; no overlap with /proc/iomem System RAM",
        (unsigned long long) requested_begin, (unsigned long long) requested_end, DDR_REGION_SIZE);
    return true;
}

static bool parse_uio_map_size(const std::string & uio, size_t * map_size) {
    std::string size_text;
    if (!read_text_file("/sys/class/uio/" + uio + "/maps/map0/size", &size_text)) {
        return false;
    }

    char * end                      = nullptr;
    errno                           = 0;
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

    char * end                      = nullptr;
    errno                           = 0;
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

    char * end                      = nullptr;
    errno                           = 0;
    const unsigned long long parsed = strtoull(offset_text.c_str(), &end, 0);
    if (errno != 0 || end == offset_text.c_str()) {
        return false;
    }

    *map_offset = (uint64_t) parsed;
    return true;
}

static bool uio_name_from_dev_path(const std::string & dev_path, std::string * uio_name) {
    const size_t      slash = dev_path.find_last_of('/');
    const std::string base  = slash == std::string::npos ? dev_path : dev_path.substr(slash + 1);
    if (base.compare(0, 3, "uio") != 0) {
        return false;
    }
    *uio_name = base;
    return true;
}

static void log_uio_inventory_once(void);

static bool find_uio_device(const char * wanted_name, std::string * dev_path, size_t * map_size) {
    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        LOGINIT("UIO lookup for name=%s failed: /sys/class/uio cannot be opened errno=%d (%s)", wanted_name, errno,
                strerror(errno));
        return false;
    }

    bool            found = false;
    struct dirent * ent   = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        const std::string uio = ent->d_name;
        std::string       name;
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
        found     = true;
        break;
    }

    closedir(dir);
    if (!found) {
        LOGINIT("UIO name=%s not found; trying physical-address match", wanted_name);
        log_uio_inventory_once();
    }
    return found;
}

static bool find_uio_device_by_addr(uint64_t      wanted_addr,
                                    std::string * dev_path,
                                    size_t *      map_size,
                                    std::string * resolved_name) {
    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        LOGINIT("UIO address lookup for addr=0x%llx failed: /sys/class/uio cannot be opened errno=%d (%s)",
                (unsigned long long) wanted_addr, errno, strerror(errno));
        return false;
    }

    bool            found = false;
    struct dirent * ent   = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        const std::string uio  = ent->d_name;
        uint64_t          addr = 0;
        size_t            size = 0;
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
        LOGINIT("UIO addr=0x%llx not found", (unsigned long long) wanted_addr);
    }
    return found;
}

static bool find_uio_device_from_ref(const char *  ref,
                                     std::string * dev_path,
                                     size_t *      map_size,
                                     std::string * resolved_name) {
    if (!ref || ref[0] == '\0') {
        return false;
    }

    std::string text(ref);
    std::string uio;
    if (text.compare(0, 8, "/dev/uio") == 0) {
        const size_t slash = text.find_last_of('/');
        uio                = slash == std::string::npos ? text : text.substr(slash + 1);
        *dev_path          = text;
    } else if (text.compare(0, 3, "uio") == 0) {
        uio       = text;
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

static bool map_uio_region(const char *        uio_name,
                           const char *        env_name,
                           uint64_t            phys,
                           size_t              advertised_min_size,
                           size_t              required_size,
                           const char *        tag,
                           size_t              requested_map_size,
                           void **             map_base,
                           size_t *            map_size,
                           size_t *            advertised_size,
                           std::string *       source,
                           fpga_mapping_kind * mapping_kind) {
    std::string  dev_path;
    std::string  resolved_name;
    size_t       uio_size     = 0;
    const char * override_ref = env_name ? getenv(env_name) : nullptr;
    if (override_ref && override_ref[0] != '\0') {
        if (!find_uio_device_from_ref(override_ref, &dev_path, &uio_size, &resolved_name)) {
            LOGE("%s=%s could not be resolved for %s", env_name, override_ref, tag);
            return false;
        }
        LOGINIT("using %s=%s for %s resolved_dev=%s resolved_name=%s", env_name, override_ref, tag, dev_path.c_str(),
                resolved_name.c_str());
    } else {
        // UIO names are not necessarily unique.  KV260 exposes all eight ZDMA
        // channels as "dma-controller", so a name-first lookup can bind an
        // arbitrary channel.  The physical address is the resource identity.
        if (!find_uio_device_by_addr(phys, &dev_path, &uio_size, &resolved_name)) {
            if (!find_uio_device(uio_name, &dev_path, &uio_size)) {
                return false;
            }
            resolved_name = uio_name;
        } else {
            LOGINIT("using UIO addr match for %s expected_name=%s phys=0x%llx resolved_dev=%s resolved_name=%s", tag,
                    uio_name, (unsigned long long) phys, dev_path.c_str(), resolved_name.c_str());
        }
    }
    // DDR is a production data-plane resource, not an address-only alias.
    if (strcmp(uio_name, "ddr_high") == 0 && resolved_name != uio_name) {
        LOGE("UIO %s for %s resolved_name=%s; require exact resource name=%s", dev_path.c_str(), tag,
             resolved_name.c_str(), uio_name);
        return false;
    }
    if (uio_size < advertised_min_size) {
        LOGE("UIO %s for %s is too small: advertised=0x%zx required_advertised=0x%zx", dev_path.c_str(), tag,
             uio_size, advertised_min_size);
        return false;
    }

    std::string uio;
    uint64_t    uio_addr   = 0;
    uint64_t    uio_offset = 0;
    if (!uio_name_from_dev_path(dev_path, &uio) || !parse_uio_map_addr(uio, &uio_addr) ||
        !parse_uio_map_offset(uio, &uio_offset)) {
        LOGE("UIO %s for %s has unreadable map0 addr/offset metadata", dev_path.c_str(), tag);
        return false;
    }
    if (uio_addr != phys || uio_offset != 0U) {
        LOGE("UIO %s for %s does not match required resource: expected_phys=0x%llx got_phys=0x%llx map_offset=0x%llx",
             dev_path.c_str(), tag, (unsigned long long) phys, (unsigned long long) uio_addr,
             (unsigned long long) uio_offset);
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
        LOGE("open %s for %s failed errno=%d (%s)", dev_path.c_str(), tag, errno, strerror(errno));
        return false;
    }
    void *    ptr         = mmap(nullptr, actual_map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int saved_errno = errno;
    close(fd);
    if (ptr == MAP_FAILED) {
        LOGE("mmap %s for %s size=0x%zx failed errno=%d (%s)", dev_path.c_str(), tag, actual_map_size, saved_errno,
             strerror(saved_errno));
        return false;
    }

    *map_base = ptr;
    *map_size = actual_map_size;
    if (advertised_size) {
        *advertised_size = uio_size;
    }
    if (mapping_kind) {
        *mapping_kind = fpga_mapping_kind::UIO_PHYSICAL;
    }
    *source = dev_path + "(" + resolved_name + ",O_SYNC)";
    LOGINIT(
        "mapped %s via UIO expected_name=%s resolved_name=%s dev=%s phys=0x%llx virt=%p mapped_size=0x%zx "
        "advertised_size=0x%zx",
        tag, uio_name, resolved_name.c_str(), dev_path.c_str(), (unsigned long long) uio_addr, ptr, actual_map_size,
        uio_size);
    return true;
}

static bool ensure_mem_fd(void) {
    if (g_mem_fd >= 0) {
        return true;
    }
    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) {
        const int   saved_errno = errno;
        struct stat st;
        if (stat("/dev/mem", &st) == 0) {
            LOGE("/dev/mem mode=0%o uid=%u gid=%u process_uid=%u process_euid=%u", (unsigned int) (st.st_mode & 07777),
                 (unsigned int) st.st_uid, (unsigned int) st.st_gid, (unsigned int) getuid(), (unsigned int) geteuid());
        }
        LOGE(
            "open /dev/mem failed errno=%d (%s). If process_euid is 0, this is kernel/device policy, not a sudo "
            "problem.",
            saved_errno, strerror(saved_errno));
        LOGE(
            "Use UIO mappings instead: set FPGA_DMA_UIO, FPGA_VPU_UIO, FPGA_DDR_UIO to /dev/uioX or to the UIO names "
            "shown above");
        return false;
    }
    return true;
}

static bool map_devmem_region(uint64_t            phys,
                              size_t              bytes,
                              const char *        tag,
                              void **             map_base,
                              size_t *            map_size,
                              std::string *       source,
                              fpga_mapping_kind * mapping_kind) {
    if (!ensure_mem_fd()) {
        return false;
    }
    void * ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g_mem_fd, (off_t) phys);
    if (ptr == MAP_FAILED) {
        LOGE("mmap /dev/mem %s phys=0x%llx size=0x%zx failed errno=%d (%s)", tag, (unsigned long long) phys, bytes,
             errno, strerror(errno));
        return false;
    }
    *map_base = ptr;
    *map_size = bytes;
    *source   = "/dev/mem(O_SYNC)";
    if (mapping_kind) {
        *mapping_kind = fpga_mapping_kind::DEVMEM_PHYSICAL;
    }
    LOGINIT("mapped %s via /dev/mem phys=0x%llx virt=%p size=0x%zx", tag, (unsigned long long) phys, ptr, bytes);
    return true;
}

static bool map_region_prefer_uio(const char *        uio_name,
                                  const char *        env_name,
                                  uint64_t            phys,
                                  size_t              devmem_size,
                                  size_t              advertised_min_size,
                                  size_t              required_size,
                                  const char *        tag,
                                  size_t              requested_map_size,
                                  bool                allow_devmem_fallback,
                                  const char *        fallback_policy,
                                  void **             map_base,
                                  size_t *            map_size,
                                  size_t *            advertised_size,
                                  std::string *       source,
                                  fpga_mapping_kind * mapping_kind) {
    if (map_uio_region(uio_name, env_name, phys, advertised_min_size, required_size, tag, requested_map_size, map_base, map_size,
                       advertised_size, source, mapping_kind)) {
        return true;
    }
    if (!allow_devmem_fallback) {
        LOGE(
            "mapping denied for %s expected_phys=0x%llx required=0x%zx policy=%s; install/fix the UIO device tree node "
            "or explicitly enable the documented compatibility policy",
            tag, (unsigned long long) phys, required_size, fallback_policy ? fallback_policy : "uio_required");
        return false;
    }
    if (fallback_policy && strcmp(fallback_policy, "vpu_devmem_compat") == 0) {
        LOGI(
            "MY_IP UIO is unavailable; using bounded /dev/mem compatibility mapping for %s phys=0x%llx size=0x%zx. Set "
            "FPGA_VPU_UIO_REQUIRED=1 or FPGA_VPU_DEVMEM_COMPAT=0 to require UIO.",
            tag, (unsigned long long) phys, devmem_size);
    } else {
        LOGE(
            "DIAGNOSTIC ONLY: FPGA_ALLOW_DEVMEM=1 permits /dev/mem mapping for %s; this is not a production-safe "
            "mapping policy",
            tag);
    }
    const size_t actual_map_size = requested_map_size == 0 ? devmem_size : requested_map_size;
    const bool   ok = map_devmem_region(phys, actual_map_size, tag, map_base, map_size, source, mapping_kind);
    if (ok && advertised_size) {
        *advertised_size = actual_map_size;
    }
    return ok;
}

static bool map_registers_dma_ddr(void) {
    fpga_p2_init_breadcrumb("phase=mapping_zdma begin policy=uio_preferred");
    if (!map_region_prefer_uio("dma-controller", "FPGA_DMA_UIO", DMA_BASE_PHYS, DMA_MMAP_SIZE, sizeof(dma_ctrl), sizeof(dma_ctrl), "ZDMA",
                               0, g_allow_devmem_fallback, "uio_required", &g_dma_map_base, &g_dma_map_size, nullptr,
                               &g_dma_map_source, nullptr)) {
        return false;
    }
    g_dma = (volatile dma_ctrl *) g_dma_map_base;
    fpga_p2_init_breadcrumb("phase=mapping_zdma pass source=%s size=0x%zx", g_dma_map_source.c_str(), g_dma_map_size);

    // A MY_IP UIO resource can advertise the entire Vivado segment.  This
    // driver uses only the established 4 MiB register/local-memory ABI, so
    // never turn an advertised UIO size into a larger process mapping.
    fpga_p2_init_breadcrumb(
        "phase=mapping_my_ip begin required_phys=0x%llx required_offset=0 required_size=0x%zx policy=uio_preferred",
        (unsigned long long) REG_BASE_PHYS, VPU_DEVMEM_COMPAT_MMAP);
    if (!map_region_prefer_uio("MY_IP", "FPGA_VPU_UIO", REG_BASE_PHYS, VPU_DEVMEM_COMPAT_MMAP, VPU_DEVMEM_COMPAT_MMAP, VPU_DEVMEM_COMPAT_MMAP,
                               "MY_IP/VPU", VPU_DEVMEM_COMPAT_MMAP,
                               g_allow_devmem_fallback || g_allow_vpu_devmem_compat,
                               g_allow_devmem_fallback ? "diagnostic_all_resources" : "vpu_devmem_compat",
                               &g_vpu_map_base, &g_vpu_map_size, nullptr, &g_vpu_map_source, nullptr)) {
        return false;
    }
    g_vpu = (volatile uint8_t *) g_vpu_map_base;
    fpga_p2_init_breadcrumb("phase=mapping_my_ip pass source=%s phys=0x%llx offset=0 size=0x%zx",
                            g_vpu_map_source.c_str(), (unsigned long long) REG_BASE_PHYS, g_vpu_map_size);

    fpga_p2_init_breadcrumb(
        "phase=mapping_ddr begin policy=uio_preferred "
        "phys=[0x%llx,0x%llx) required_size=0x%zx",
        (unsigned long long) DDR_BASE_PHYS, (unsigned long long) DDR_END_EXCLUSIVE, DDR_REQUIRED_BYTES);
    if (!fpga_ddr_iomem_preflight()) {
        return false;
    }
    g_ddr_mapping_kind = fpga_mapping_kind::UNKNOWN;
    if (!map_region_prefer_uio("ddr_high", "FPGA_DDR_UIO", DDR_BASE_PHYS, DDR_REGION_SIZE, DDR_REGION_SIZE, DDR_REQUIRED_BYTES,
                               "ddr_high", g_ddr_requested_map_size, g_allow_devmem_fallback, "uio_required",
                               &g_ddr_map_base, &g_ddr_map_size, &g_ddr_advertised_size, &g_ddr_map_source,
                               &g_ddr_mapping_kind)) {
        return false;
    }
    g_ddr = (uint8_t *) g_ddr_map_base;
    fpga_p2_init_breadcrumb("phase=mapping_ddr pass source=%s map_kind=%s size=0x%zx", g_ddr_map_source.c_str(),
                            fpga_mapping_kind_name(g_ddr_mapping_kind), g_ddr_map_size);
    return true;
}

static bool fpga_weight_path_bench_host_self_test(void);

// The weight-path benchmark intentionally has no ZDMA ownership.  It maps
// only the verified UIO resources needed to read identity/capability words
// and to make bounded CPU stores into the existing WEIGHT staging window.
static bool map_weight_path_bench_vpu_ddr(void);

static bool fpga_weight_path_bench_read_capabilities(void) {
    if (!vpu_is_mapped()) {
        return false;
    }
    const uint32_t limits = vpu_rd32(REG_LIMITS);
    const uint32_t caps   = vpu_rd32(REG_CAPS);
    g_stream_protocol_version = vpu_rd32(REG_STREAM_PROTOCOL_VERSION);
    g_bitstream_id            = vpu_rd32(REG_BITSTREAM_ID);
    g_p2_stream_abi_signature = vpu_rd32(REG_P2_STREAM_ABI);

    const int rows  = (int) (limits & 0xFFFFU);
    const int beats = (int) ((limits >> 16) & 0xFFFFU);
    const int blocks = (int) ((caps >> 8) & 0xFFU);
    const int result_words = (int) ((caps >> 16) & 0xFFFFU);
    const bool caps_valid = caps != 0U && caps != 0xFFFFFFFFU;
    const bool identity_ok = g_bitstream_id == FPGA_EXPECTED_BITSTREAM_ID &&
                             g_stream_protocol_version == FPGA_REQUIRED_STREAM_PROTOCOL_VERSION &&
                             g_p2_stream_abi_signature == FPGA_REQUIRED_P2_STREAM_ABI;
    const bool layout_ok = caps_valid && (caps & VPU_CAP_PACKED_Q8) != 0U &&
                           (caps & VPU_CAP_COMPACT_WEIGHT_LAYOUT) != 0U &&
                           (caps & VPU_CAP_P2_TWO_ROW_TRANSPORT) != 0U && rows > 0 && rows <= VPU_DEFAULT_ROWS &&
                           beats > 0 && beats <= VPU_DEFAULT_BEATS && blocks > 0 && result_words > 0;
    if (!identity_ok || !layout_ok) {
        LOGE("WEIGHT_PATH_BENCH_CAPS_REJECT protocol=0x%08x bitstream_id=0x%08x p2_abi=0x%08x limits=0x%08x "
             "caps=0x%08x identity_ok=%d layout_ok=%d action=no_ddr_write_no_zdma_no_vpu_start",
             g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature, limits, caps,
             identity_ok ? 1 : 0, layout_ok ? 1 : 0);
        return false;
    }
    g_vpu_max_rows           = rows;
    g_vpu_max_beats          = beats;
    g_vpu_max_cols           = beats * VPU_NUM_LANES;
    g_runtime_max_rows = env_int_value("FPGA_RUNTIME_MAX_ROWS", VPU_SAFE_RUNTIME_ROWS, 1, VPU_DEFAULT_ROWS);
    if (g_runtime_max_rows < g_vpu_max_rows) {
        g_vpu_max_rows = g_runtime_max_rows;
    }
    g_packed_q8_supported    = 1;
    g_packed_q8_max_blocks   = std::min(blocks, beats / VPU_BLOCK_BEATS);
    if (g_packed_q8_max_blocks <= 0) {
        return false;
    }
    // Match the admitted normal host tiler: capability reporting is checked
    // above, then its safe runtime tile geometry becomes the grouping limit.
    g_packed_q8_result_words =
        (g_vpu_max_rows * g_packed_q8_max_blocks + VPU_RESULT_PACK_LANES - 1) / VPU_RESULT_PACK_LANES;
    LOGINIT("WEIGHT_PATH_BENCH_PASSIVE_READY version=%s protocol=0x%08x bitstream_id=0x%08x p2_abi=0x%08x "
            "rows=%d host_row_limit=%d beats=%d max_group_blocks=%d result_words=%d vpu_map=0x%zx ddr_map=0x%zx "
            "zdma=not_mapped dma_init=not_run vpu_start=not_run",
            FPGA_HOST_TRACE_VERSION, g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature,
            g_vpu_max_rows, g_runtime_max_rows, g_vpu_max_beats, g_packed_q8_max_blocks,
            g_packed_q8_result_words, g_vpu_map_size, g_ddr_map_size);
    return true;
}

static bool fpga_weight_path_bench_environment_allowed(void) {
    if (env_flag_enabled("FPGA_DISABLE")) {
        fpga_fatal("FPGA_WEIGHT_PATH_BENCH=1 conflicts with FPGA_DISABLE=1; explicit disable wins before any UIO mapping");
    }
    const bool incompatible = env_flag_enabled("FPGA_WEIGHT_CACHE") ||
                              env_flag_enabled("FPGA_P2_WEIGHT_RESIDENCY") ||
                              env_flag_enabled("FPGA_P2_WEIGHT_RESIDENCY_DIAGNOSTIC") ||
                              env_flag_enabled("FPGA_P2_RESIDENCY_TRACE") ||
                              env_flag_enabled("FPGA_P2_RESIDENCY_VERIFY_METADATA") ||
                              env_flag_enabled("FPGA_P3_SPLIT_SCALE") ||
                              env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000) != 0 ||
                              env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000) != 0 ||
                              env_flag_enabled("FPGA_SOURCE_AUDIT_ONLY") ||
                              env_flag_enabled("FPGA_PIPELINE_ENABLE") ||
                              env_flag_enabled("FPGA_P2_INPUT_PRELOAD") ||
                              env_flag_enabled("FPGA_ABORT_ON_CPU_FALLBACK");
    if (incompatible) {
        fpga_fatal("FPGA_WEIGHT_PATH_BENCH=1 rejects cache/residency/P3/contract/source-audit/preload/pipeline/explicit-"
                   "abort modes; benchmark routes the captured m==1 sequence to native CPU and never launches ZDMA/VPU");
    }
    return true;
}

static bool configure_ddr_mapping_policy(void) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        LOGE("cannot determine page size for ddr_high mapping");
        return false;
    }

    g_weight_cache_budget_mb = 0;
    g_p2_weight_residency_budget_mb = 0;
    g_ddr_requested_map_size = align_up_size(DDR_REQUIRED_BYTES, (size_t) page_size);
    if (g_p2_weight_residency_requested) {
        const long long residency_mb = P2_WEIGHT_RESIDENCY_MAX_MB;
        const uint64_t required_bytes = (uint64_t) WEIGHT_CACHE_BASE + (uint64_t) residency_mb * 1024ULL * 1024ULL;
        if (required_bytes > DDR_REGION_SIZE) {
            LOGE("P2 residency range exceeds reserved DDR: budget_mb=%lld required=0x%llx region=0x%zx", residency_mb,
                 (unsigned long long) required_bytes, DDR_REGION_SIZE);
            return false;
        }
        g_p2_weight_residency_budget_mb = residency_mb;
        g_ddr_requested_map_size = align_up_size((size_t) required_bytes, (size_t) page_size);
        LOGINIT(
            "P2_RESIDENCY_CONFIG requested=1 diagnostic=1 trace=%d verify_metadata=%d base_off=0x%08x phys=[0x%llx,0x%llx) budget_mb=%lld requested_map=0x%zx "
            "policy=fixed_non_evicting_directory_index_no_msync slots=%zu index_buckets=%zu",
            g_p2_residency_trace_enabled ? 1 : 0, g_p2_residency_verify_metadata ? 1 : 0, WEIGHT_CACHE_BASE,
            (unsigned long long) (DDR_BASE_PHYS + WEIGHT_CACHE_BASE),
            (unsigned long long) (DDR_BASE_PHYS + required_bytes), residency_mb, g_ddr_requested_map_size,
            P2_WEIGHT_RESIDENCY_SLOT_CAPACITY, P2_WEIGHT_RESIDENCY_INDEX_BUCKETS);
        return true;
    }
    if (!g_weight_cache_enabled) {
        LOGINIT("DDR map policy: cache=disabled requested_size=0x%zx", g_ddr_requested_map_size);
        return true;
    }

    const char *    cache_mb_text = getenv("FPGA_WEIGHT_CACHE_MB");
    const long long cache_mb      = env_int64_value("FPGA_WEIGHT_CACHE_MB", 0, 0, 16384);
    if (!cache_mb_text || cache_mb_text[0] == '\0' || cache_mb <= 0) {
        LOGE("FPGA_WEIGHT_CACHE=1 requires a positive FPGA_WEIGHT_CACHE_MB; refusing an unbounded DDR cache");
        return false;
    }

    const bool large_cache_confirmed = env_flag_enabled("FPGA_WEIGHT_CACHE_LARGE_CONFIRMED");
    if (cache_mb > WEIGHT_CACHE_DEFAULT_MAX_MB && !large_cache_confirmed) {
        const uint64_t cache_start = (uint64_t) DDR_BASE_PHYS + (uint64_t) WEIGHT_CACHE_BASE;
        const uint64_t cache_end   = cache_start + (uint64_t) cache_mb * 1024ULL * 1024ULL;
        LOGE(
            "refusing unconfirmed large weight cache: FPGA_WEIGHT_CACHE_MB=%lld exceeds default_max_mb=%lld; "
            "requested_phys=[0x%llx,0x%llx). No DDR mapping or cache write was performed. Run the read-only "
            "ddr_high/reserved-memory preflight and graduated cache tests first; FPGA_WEIGHT_CACHE_LARGE_CONFIRMED is "
            "an operator acknowledgement, not a safety check",
            cache_mb, WEIGHT_CACHE_DEFAULT_MAX_MB, (unsigned long long) cache_start, (unsigned long long) cache_end);
        return false;
    }

    const uint64_t payload_bytes  = (uint64_t) cache_mb * 1024ULL * 1024ULL;
    const uint64_t required_bytes = (uint64_t) WEIGHT_CACHE_BASE + payload_bytes;
    if (required_bytes > DDR_REGION_SIZE) {
        LOGE("requested FPGA_WEIGHT_CACHE_MB=%lld requires 0x%llx bytes, above reserved FPGA DDR region 0x%zx",
             cache_mb, (unsigned long long) required_bytes, DDR_REGION_SIZE);
        return false;
    }

    g_weight_cache_budget_mb = cache_mb;
    g_ddr_requested_map_size = align_up_size((size_t) required_bytes, (size_t) page_size);
    LOGINIT("DDR map policy: cache=enabled budget_mb=%lld requested_size=0x%zx large_confirmed=%d",
            g_weight_cache_budget_mb, g_ddr_requested_map_size, large_cache_confirmed ? 1 : 0);
    return true;
}

static void configure_weight_cache(void) {
    g_weight_cache.clear();
    g_weight_cache_next_off    = WEIGHT_CACHE_BASE;
    g_weight_cache_end_off     = WEIGHT_CACHE_BASE;
    g_weight_cache_full_logged = false;

    if (!g_weight_cache_enabled) {
        LOGINIT("weight tile cache disabled; only ACT/WEIGHT/RESULT scratch DDR windows will be touched");
        return;
    }
    if (!ddr_is_mapped() || g_ddr_map_size <= WEIGHT_CACHE_BASE) {
        g_weight_cache_enabled = false;
        LOGE("weight tile cache disabled: ddr_high size=0x%zx is too small for base=0x%08x", g_ddr_map_size,
             WEIGHT_CACHE_BASE);
        return;
    }

    if (g_weight_cache_budget_mb <= 0) {
        g_weight_cache_enabled = false;
        LOGE("weight tile cache disabled: no validated FPGA_WEIGHT_CACHE_MB budget is available");
        return;
    }

    size_t       available    = g_ddr_map_size - (size_t) WEIGHT_CACHE_BASE;
    const size_t budget_bytes = (size_t) g_weight_cache_budget_mb * 1024U * 1024U;
    available                 = std::min(available, budget_bytes);
    available                 = (available / WEIGHT_CACHE_ALIGN) * WEIGHT_CACHE_ALIGN;
    if (available < WEIGHT_CACHE_ALIGN) {
        g_weight_cache_enabled = false;
        LOGE("weight tile cache disabled: available bytes after limit is only %zu", available);
        return;
    }

    const uint64_t end     = (uint64_t) WEIGHT_CACHE_BASE + (uint64_t) available;
    g_weight_cache_end_off = end > UINT32_MAX ? UINT32_MAX : (uint32_t) end;
    LOGI(
        "weight tile cache enabled base=0x%08x end=0x%08x bytes=%zu budget_mb=%lld mapped_ddr=0x%zx "
        "advertised_ddr=0x%zx",
        WEIGHT_CACHE_BASE, g_weight_cache_end_off, (size_t) (g_weight_cache_end_off - WEIGHT_CACHE_BASE),
        g_weight_cache_budget_mb, g_ddr_map_size, g_ddr_advertised_size);
}

static bool msync_ddr_range(uint32_t off, size_t bytes, bool invalidate, const char * tag) {
    if (!ddr_range_fits(off, bytes)) {
        LOGE("DDR msync range overflow tag=%s off=0x%08x bytes=%zu mapped_size=0x%zx", tag, off, bytes, g_ddr_map_size);
        return false;
    }

    const long      page          = sysconf(_SC_PAGESIZE);
    const uintptr_t begin         = (uintptr_t) (g_ddr + off);
    const uintptr_t aligned_begin = begin & ~((uintptr_t) page - 1U);
    const uintptr_t end           = begin + bytes;
    const size_t    len           = align_up_size((size_t) (end - aligned_begin), (size_t) page);
    const int       flags         = MS_SYNC | (invalidate ? MS_INVALIDATE : 0);
    mmio_fence();
    // A UIO ddr_high mapping on this board may report EINVAL for msync.  Keep
    // v16's conservative per-tile attempt for ordinary ACT/WEIGHT/RESULT DMA
    // ranges, but do not repeat that unsupported call for one large cache
    // payload (potentially 1.1 GiB) during warm-up.
    if (g_ddr_msync_unavailable && bytes >= WEIGHT_CACHE_LARGE_MSYNC_BYTES) {
        mmio_fence();
        return true;
    }
    if (msync((void *) aligned_begin, len, flags) != 0) {
        const int  saved_errno             = errno;
        const bool likely_uncached_mapping = g_ddr_map_source.find("O_SYNC") != std::string::npos ||
                                             g_ddr_map_source.find("/dev/uio") != std::string::npos;
        if (!g_strict_coherency && !env_flag_enabled("FPGA_STRICT_MSYNC") && likely_uncached_mapping &&
            (saved_errno == EINVAL || saved_errno == ENODEV)) {
            if (!g_ddr_msync_unsupported_logged) {
                LOGI(
                    "msync unsupported for ddr_high source=%s errno=%d (%s); generic-uio physical maps are normally "
                    "non-cached, but cacheability is kernel-owned. Continuing with ordered volatile DDR accesses; this "
                    "message alone does not prove a cache fault.",
                    g_ddr_map_source.c_str(), saved_errno, strerror(saved_errno));
                g_ddr_msync_unsupported_logged = true;
            }
            g_ddr_msync_unavailable = true;
            mmio_fence();
            return true;
        }
        if ((saved_errno == EINVAL || saved_errno == ENODEV) && g_strict_coherency &&
            g_coherency_platform_whitelisted) {
            LOGI(
                "strict coherency whitelist accepts unsupported msync for ddr_high source=%s; CPU barriers remain "
                "enabled",
                g_ddr_map_source.c_str());
            mmio_fence();
            return true;
        }
        LOGE(
            "msync ddr_high tag=%s off=0x%08x bytes=%zu invalidate=%d errno=%d (%s) source=%s aligned=0x%llx len=0x%zx "
            "flags=0x%x",
            tag, off, bytes, invalidate ? 1 : 0, saved_errno, strerror(saved_errno), g_ddr_map_source.c_str(),
            (unsigned long long) aligned_begin, len, flags);
        return false;
    }
    mmio_fence();
    return true;
}

// Defined with the staging helpers below.  P2 uses this exact bounded
// first/last-word readback after CPU writes to an admitted physical DDR aperture.
static void fpga_ddr_staging_readback_commit(uint32_t off, size_t bytes);

static bool p2_physical_ddr_mapping_active(void) {
    return g_p2_init_requested &&
           (g_ddr_mapping_kind == fpga_mapping_kind::UIO_PHYSICAL ||
            (g_allow_devmem_fallback && g_ddr_mapping_kind == fpga_mapping_kind::DEVMEM_PHYSICAL));
}

static bool p2_msync_or_stress_requested(void) {
    return g_strict_coherency || g_run_coherency_stress || env_flag_enabled("FPGA_STRICT_MSYNC") ||
           env_flag_enabled("FPGA_COHERENCY_STRESS");
}

static void fpga_p2_ddr_sync_breadcrumb(const char * tag,
                                        const char * direction,
                                        uint32_t     off,
                                        size_t       bytes,
                                        const char * ordering);

// `msync()` is a file-backed writeback primitive, not a DMA-coherency
// primitive for the physical pages exposed by generic-uio.  The owner trace
// proves this syscall can hang before the first ZDMA descriptor is touched.
// P2 prefers the explicit UIO mapping class. For the fixed board contract,
// bounded /dev/mem(O_SYNC) compatibility uses the same volatile readback + DSB
// ordering and still avoids msync. Raw v52/C0 keeps the legacy msync path.
static bool fpga_p2_ddr_sync(uint32_t off, size_t bytes, bool device_to_cpu, const char * tag) {
    if (!g_p2_init_requested) {
        return msync_ddr_range(off, bytes, device_to_cpu, tag);
    }
    if (!ddr_range_fits(off, bytes)) {
        LOGE("P2 DDR sync range overflow tag=%s off=0x%08x bytes=%zu mapped_size=0x%zx", tag ? tag : "?", off, bytes,
             g_ddr_map_size);
        return false;
    }
    if (!p2_physical_ddr_mapping_active()) {
        LOGE(
            "P2 DDR sync requires an admitted physical ddr_high mapping tag=%s map_kind=%s; refusing before "
            "data-plane transfer",
            tag ? tag : "?", fpga_mapping_kind_name(g_ddr_mapping_kind));
        return false;
    }
    if (p2_msync_or_stress_requested()) {
        LOGE(
            "P2 rejects FPGA_STRICT_COHERENCY/FPGA_STRICT_MSYNC/FPGA_COHERENCY_STRESS for admitted physical ddr_high "
            "tag=%s; no msync syscall or data-plane transfer was issued",
            tag ? tag : "?");
        return false;
    }

    if (device_to_cpu) {
        mmio_fence();
        fpga_p2_ddr_sync_breadcrumb(tag, "device_to_cpu", off, bytes, "dsb_before_cpu_read");
    } else {
        fpga_ddr_staging_readback_commit(off, bytes);
        fpga_p2_ddr_sync_breadcrumb(tag, "cpu_to_device", off, bytes, "dsb_readback");
    }
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

static bool phys_range_fits(uint64_t phys, size_t bytes, uint64_t base, size_t mapped_size) {
    if (bytes == 0U || phys < base) {
        return false;
    }
    const uint64_t delta = phys - base;
    return delta <= (uint64_t) mapped_size && (uint64_t) bytes <= (uint64_t) mapped_size - delta;
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

static void zdma_format_error_mask(uint32_t isr, char * out, size_t out_size);

static void zdma_dump(const char * tag);

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

// P2 must not take ownership of a channel left live by another owner/process.
// This is intentionally read-only and must run before fpga_dma_init() writes
// controller state, descriptors, W1C registers, or CTRL2.
static bool p2_zdma_preinit_passive_gate(void) {
    if (!dma_is_mapped()) {
        LOGE("P2 pre-init ZDMA gate has no mapped channel");
        return false;
    }

    const uint32_t ctrl2  = g_dma->ZDMA_CH_CTRL2;
    const uint32_t status = g_dma->ZDMA_CH_STATUS;
    const uint32_t isr    = g_dma->ZDMA_CH_ISR;
    const uint32_t total  = g_dma->ZDMA_CH_TOTAL_BYTE;
    LOGINIT("P2_ZDMA_PREINIT passive=1 ctrl2=0x%08x status=0x%08x isr=0x%08x total_bytes=0x%08x en=%d action=%s",
             ctrl2, status, isr, total, (ctrl2 & ZDMA_CTRL2_EN) != 0U ? 1 : 0,
             (ctrl2 & ZDMA_CTRL2_EN) != 0U ? "fail_closed_no_write" : "init_permitted");
    fpga_p2_init_breadcrumb(
        "phase=zdma_preinit ctrl2=0x%08x status=0x%08x isr=0x%08x total_bytes=0x%08x en=%d action=%s", ctrl2, status,
        isr, total, (ctrl2 & ZDMA_CTRL2_EN) != 0U ? 1 : 0,
        (ctrl2 & ZDMA_CTRL2_EN) != 0U ? "fail_closed_no_write" : "init_permitted");
    return (ctrl2 & ZDMA_CTRL2_EN) == 0U;
}

static bool fpga_dma_init(void) {
    if (!dma_is_mapped()) {
        LOGE("ZDMA register pointer is not mapped");
        return false;
    }

    g_dma->ZDMA_ERR_CTRL     = 0x00000001;
    // ZDMA_CH_ISR is write-one-to-clear.  Clear any sticky result left by a
    // prior process before the first descriptor is programmed.
    g_dma->ZDMA_CH_ISR       = ZDMA_ISR_CLEAR_ALL;
    g_dma->ZDMA_CH_IMR       = 0x00000FFF;
    g_dma->ZDMA_CH_IEN       = 0x00000000;
    g_dma->ZDMA_CH_IDS       = 0x00000000;
    g_dma->ZDMA_CH_CTRL0     = 0x00000080;
    g_dma->ZDMA_CH_CTRL1     = 0x000003FF;
    g_dma->ZDMA_CH_FCI       = 0x00000000;
    g_dma->ZDMA_CH_STATUS    = 0x00000000;
    g_dma->ZDMA_CH_DATA_ATTR = ZDMA_DATA_ATTR_AXCACHE;
    g_dma->ZDMA_CH_DSCR_ATTR = 0x00000000;
    zdma_clear_descriptors();
    g_dma->ZDMA_CH_RATE_CTRL    = 0x00000000;
    g_dma->ZDMA_CH_IRQ_SRC_ACCT = 0x00000000;
    g_dma->ZDMA_CH_IRQ_DST_ACCT = 0x00000000;
    g_dma->ZDMA_CH_CTRL2        = 0x00000000;
    mmio_fence();
    const uint32_t stale_total_bytes = zdma_total_byte_clear();

    LOGINIT(
        "ZDMA init base=0x%llx virt=0x%llx status=0x%08x isr=0x%08x ctrl0=0x%08x ctrl1=0x%08x data_attr=0x%08x "
        "stale_total_bytes=0x%08x total_bytes_after_clear=0x%08x "
        "completion_gate=isr_ack_then_dma_done_and_ctrl2_en_clear",
        (unsigned long long) DMA_BASE_PHYS, fpga_ptr_addr(g_dma), g_dma->ZDMA_CH_STATUS, g_dma->ZDMA_CH_ISR,
        g_dma->ZDMA_CH_CTRL0, g_dma->ZDMA_CH_CTRL1, g_dma->ZDMA_CH_DATA_ATTR, stale_total_bytes,
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
        LOGE("ZDMA completion gate has no mapped channel tag=%s phase=%s", tag ? tag : "?", phase ? phase : "?");
        return false;
    }

    const long long t0    = now_us();
    long long       polls = 0;
    while ((g_dma->ZDMA_CH_CTRL2 & ZDMA_CTRL2_EN) != 0U) {
        if (now_us() - t0 > g_dma_timeout_us) {
            LOGE("ZDMA EN timeout tag=%s phase=%s status=0x%08x state=%u isr=0x%08x ctrl2=0x%08x polls=%lld",
                 tag ? tag : "?", phase ? phase : "?", g_dma->ZDMA_CH_STATUS,
                 g_dma->ZDMA_CH_STATUS & ZDMA_STATUS_STATE_MASK, g_dma->ZDMA_CH_ISR, g_dma->ZDMA_CH_CTRL2, polls);
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

    const long long t0                  = now_us();
    long long       polls               = 0;
    const uint32_t  completion_or_error = ZDMA_ISR_DMA_DONE | ZDMA_ISR_ERROR_MASK;
    while (true) {
        const uint32_t isr = g_dma->ZDMA_CH_ISR;
        if ((isr & completion_or_error) == 0U) {
            mmio_fence();
            return true;
        }
        if (now_us() - t0 > g_dma_timeout_us) {
            char errors[160];
            zdma_format_error_mask(isr, errors, sizeof(errors));
            LOGE(
                "ZDMA ISR clear timeout tag=%s isr=0x%08x errors=%s ctrl2=0x%08x polls=%lld; refusing to start with a "
                "stale completion event",
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
        LOGE("ZDMA completion gate has no mapped channel tag=%s", tag ? tag : "?");
        return false;
    }

    const long long t0          = now_us();
    long long       polls       = 0;
    bool            saw_enabled = false;
    while (true) {
        const uint32_t ctrl2    = g_dma->ZDMA_CH_CTRL2;
        const uint32_t isr      = g_dma->ZDMA_CH_ISR;
        const uint32_t status   = g_dma->ZDMA_CH_STATUS;
        const bool     enabled  = (ctrl2 & ZDMA_CTRL2_EN) != 0U;
        const bool     dma_done = (isr & ZDMA_ISR_DMA_DONE) != 0U;

        saw_enabled = saw_enabled || enabled;
        if (info) {
            info->status      = status;
            info->isr         = isr;
            info->ctrl2       = ctrl2;
            info->polls       = polls;
            info->saw_enabled = saw_enabled;
        }
        if ((isr & ZDMA_ISR_ERROR_MASK) != 0U) {
            char errors[160];
            zdma_format_error_mask(isr, errors, sizeof(errors));
            LOGE(
                "ZDMA error tag=%s status=0x%08x state=%u isr=0x%08x errors=%s ctrl2=0x%08x total_bytes=0x%08x "
                "saw_enabled=%d polls=%lld",
                tag ? tag : "?", status, status & ZDMA_STATUS_STATE_MASK, isr, errors, ctrl2, g_dma->ZDMA_CH_TOTAL_BYTE,
                saw_enabled ? 1 : 0, polls);
            zdma_dump(tag);
            return false;
        }
        if (dma_done && !enabled) {
            mmio_fence();
            return true;
        }
        if (now_us() - t0 > g_dma_timeout_us) {
            LOGE(
                "ZDMA completion timeout tag=%s status=0x%08x state=%u isr=0x%08x ctrl2=0x%08x total_bytes=0x%08x "
                "dma_done=%d saw_enabled=%d polls=%lld",
                tag ? tag : "?", status, status & ZDMA_STATUS_STATE_MASK, isr, ctrl2, g_dma->ZDMA_CH_TOTAL_BYTE,
                dma_done ? 1 : 0, saw_enabled ? 1 : 0, polls);
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
static void fpga_dma_trace_record(const char *                   tag,
                                  uint64_t                       src_phys,
                                  uint64_t                       dst_phys,
                                  size_t                         bytes,
                                  uint32_t                       pre_status,
                                  uint32_t                       pre_isr,
                                  uint32_t                       pre_ctrl2,
                                  uint32_t                       total_bytes_before_clear,
                                  uint32_t                       pre_vpu_status,
                                  uint32_t                       pre_vpu_progress,
                                  uint32_t                       total_bytes_after_transfer,
                                  uint32_t                       post_vpu_status,
                                  uint32_t                       post_vpu_progress,
                                  long long                      elapsed_us,
                                  const zdma_completion_info_t & completion);

static void fpga_dma_trace_dump(const char * reason,
                                const char * tensor_name,
                                int          layer_id,
                                uint32_t     tile_id,
                                const char * failed_transfer_tag);

// P2 uses descriptor/bank ownership as part of its ABI.  Read back exactly
// the descriptor words just committed; a mismatch is a controller/mapping
// contract failure, not something a CPU route may conceal.
static bool p2_zdma_verify_descriptor_commit(uint64_t src_phys,
                                             uint64_t dst_phys,
                                             size_t   bytes,
                                             uint32_t pre_status,
                                             uint32_t pre_isr,
                                             uint32_t pre_ctrl2,
                                             uint32_t pre_total,
                                             bool     first_act_detail) {
    mmio_fence();
    const uint32_t src0      = g_dma->ZDMA_CH_SRC_DSCR_WORD0;
    const uint32_t src1      = g_dma->ZDMA_CH_SRC_DSCR_WORD1;
    const uint32_t src2      = g_dma->ZDMA_CH_SRC_DSCR_WORD2;
    const uint32_t src3      = g_dma->ZDMA_CH_SRC_DSCR_WORD3;
    const uint32_t dst0      = g_dma->ZDMA_CH_DST_DSCR_WORD0;
    const uint32_t dst1      = g_dma->ZDMA_CH_DST_DSCR_WORD1;
    const uint32_t dst2      = g_dma->ZDMA_CH_DST_DSCR_WORD2;
    const uint32_t dst3      = g_dma->ZDMA_CH_DST_DSCR_WORD3;
    const uint32_t bank_stat = vpu_is_mapped() ? vpu_rd32(REG_BANK_STAT) : 0xFFFFFFFFU;
    const bool     matches   = src0 == (uint32_t) src_phys && src1 == (uint32_t) (src_phys >> 32) &&
                               src2 == (uint32_t) bytes && src3 == 0U && dst0 == (uint32_t) dst_phys &&
                               dst1 == (uint32_t) (dst_phys >> 32) && dst2 == (uint32_t) bytes && dst3 == 0U;
    if (first_act_detail) {
        fpga_p2_dma_breadcrumb(
            "step=descriptor_commit src=0x%llx dst=0x%llx bytes=%zu pre_status=0x%08x pre_isr=0x%08x pre_ctrl2=0x%08x "
            "pre_total=0x%08x src_desc=[0x%08x,0x%08x,0x%08x,0x%08x] dst_desc=[0x%08x,0x%08x,0x%08x,0x%08x] "
            "bank_stat=0x%08x match=%d",
            (unsigned long long) src_phys, (unsigned long long) dst_phys, bytes, pre_status, pre_isr, pre_ctrl2,
            pre_total, src0, src1, src2, src3, dst0, dst1, dst2, dst3, bank_stat, matches ? 1 : 0);
    }
    if (!matches) {
        LOGE(
            "P2 ZDMA descriptor readback mismatch src=0x%llx dst=0x%llx bytes=%zu "
            "src_desc=[0x%08x,0x%08x,0x%08x,0x%08x] dst_desc=[0x%08x,0x%08x,0x%08x,0x%08x] bank_stat=0x%08x",
            (unsigned long long) src_phys, (unsigned long long) dst_phys, bytes, src0, src1, src2, src3, dst0, dst1,
            dst2, dst3, bank_stat);
        return false;
    }
    return true;
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

    const bool p2_descriptor_commit = g_p2_init_requested;
    // A qualification must show every transfer edge, including each split
    // WEIGHT descriptor.  Outside qualification the existing first-ACT trace
    // remains the only detailed P2 path.
    const bool p2_transfer_detail =
        g_p2_init_requested &&
        (g_p2_tile_trace_enabled || (g_p2_first_act_dma_trace_active && tag != nullptr && strcmp(tag, "ACT") == 0));
    if (p2_transfer_detail) {
        ++g_p2_dma_transfer_sequence;
        g_p2_trace_dma_tag = tag ? tag : "?";
        fpga_p2_dma_breadcrumb("step=entry src=0x%llx dst=0x%llx bytes=%zu", (unsigned long long) src_phys,
                               (unsigned long long) dst_phys, bytes);
        fpga_p2_dma_breadcrumb(
            "step=channel_disabled_gate edge=before ctrl2=0x%08x status=0x%08x isr=0x%08x total_bytes=0x%08x",
            g_dma->ZDMA_CH_CTRL2, g_dma->ZDMA_CH_STATUS, g_dma->ZDMA_CH_ISR, g_dma->ZDMA_CH_TOTAL_BYTE);
    }

    // Never rewrite descriptors until the preceding transfer's hardware EN
    // bit is clear.  STATUS is retained for diagnostics only; it is not a
    // sufficient ownership/completion gate.
    if (!zdma_wait_channel_disabled(tag, "before_descriptor")) {
        if (p2_transfer_detail) {
            fpga_p2_dma_breadcrumb(
                "step=channel_disabled_gate edge=fail ctrl2=0x%08x status=0x%08x isr=0x%08x total_bytes=0x%08x",
                g_dma->ZDMA_CH_CTRL2, g_dma->ZDMA_CH_STATUS, g_dma->ZDMA_CH_ISR, g_dma->ZDMA_CH_TOTAL_BYTE);
        }
        return false;
    }
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb(
            "step=channel_disabled_gate edge=after ctrl2=0x%08x status=0x%08x isr=0x%08x total_bytes=0x%08x",
            g_dma->ZDMA_CH_CTRL2, g_dma->ZDMA_CH_STATUS, g_dma->ZDMA_CH_ISR, g_dma->ZDMA_CH_TOTAL_BYTE);
    }
    const uint32_t total_bytes_before_clear = zdma_total_byte_clear();
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb("step=total_clear w1c_value=0x%08x total_bytes_after=0x%08x", total_bytes_before_clear,
                               g_dma->ZDMA_CH_TOTAL_BYTE);
    }

    uint32_t src_ddr_off = 0;
    if (phys_to_ddr_offset(src_phys, bytes, &src_ddr_off)) {
        if (p2_transfer_detail) {
            fpga_p2_dma_breadcrumb(
                "step=source_sync edge=before tag=%s direction=cpu_to_device offset=0x%08x bytes=%zu map_kind=%s "
                "action=no_msync ordering=dsb_readback",
                tag ? tag : "?", src_ddr_off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind));
        }
        if (!fpga_p2_ddr_sync(src_ddr_off, bytes, false, tag)) {
            if (p2_transfer_detail) {
                fpga_p2_dma_breadcrumb(
                    "step=source_sync edge=fail tag=%s direction=cpu_to_device offset=0x%08x bytes=%zu map_kind=%s "
                    "action=no_msync ordering=dsb_readback",
                    tag ? tag : "?", src_ddr_off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind));
            }
            return false;
        }
        if (p2_transfer_detail) {
            fpga_p2_dma_breadcrumb(
                "step=source_sync edge=after tag=%s direction=cpu_to_device offset=0x%08x bytes=%zu map_kind=%s "
                "action=no_msync ordering=dsb_readback",
                tag ? tag : "?", src_ddr_off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind));
        }
    }

    const uint32_t isr_before_clear = p2_transfer_detail ? g_dma->ZDMA_CH_ISR : 0U;
    if (!zdma_clear_isr_for_new_transfer(tag)) {
        if (p2_transfer_detail) {
            fpga_p2_dma_breadcrumb("step=isr_w1c_ack edge=fail isr_before=0x%08x write_mask=0x%08x isr_after=0x%08x",
                                   isr_before_clear, ZDMA_ISR_CLEAR_ALL, g_dma->ZDMA_CH_ISR);
        }
        return false;
    }
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb("step=isr_w1c_ack isr_before=0x%08x write_mask=0x%08x isr_after=0x%08x",
                               isr_before_clear, ZDMA_ISR_CLEAR_ALL, g_dma->ZDMA_CH_ISR);
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

    uint32_t pre_status       = g_dma_trace_enabled || p2_descriptor_commit ? g_dma->ZDMA_CH_STATUS : 0U;
    uint32_t pre_isr          = g_dma_trace_enabled || p2_descriptor_commit ? g_dma->ZDMA_CH_ISR : 0U;
    uint32_t pre_ctrl2        = g_dma_trace_enabled || p2_descriptor_commit ? g_dma->ZDMA_CH_CTRL2 : 0U;
    uint32_t pre_vpu_status   = 0;
    uint32_t pre_vpu_progress = 0;
    if (p2_descriptor_commit &&
        !p2_zdma_verify_descriptor_commit(src_phys, dst_phys, bytes, pre_status, pre_isr, pre_ctrl2,
                                          total_bytes_before_clear, p2_transfer_detail)) {
        return false;
    }
    if (g_dma_trace_enabled) {
        if (vpu_is_mapped()) {
            pre_vpu_status   = vpu_rd32(REG_STATUS);
            pre_vpu_progress = vpu_rd32(REG_PROGRESS);
        }
    }

    const long long t0 = now_us();
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb("step=ctrl2_start edge=before ctrl2=0x%08x", g_dma->ZDMA_CH_CTRL2);
    }
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_START;
    mmio_fence();
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb("step=ctrl2_start edge=after_write_fence ctrl2=0x%08x", g_dma->ZDMA_CH_CTRL2);
    }

    zdma_completion_info_t completion = {};
    if (!zdma_wait_transfer_complete(tag, &completion)) {
        const uint32_t total_bytes_after_transfer = g_dma->ZDMA_CH_TOTAL_BYTE;
        const uint32_t post_vpu_status            = vpu_is_mapped() ? vpu_rd32(REG_STATUS) : 0U;
        const uint32_t post_vpu_progress          = vpu_is_mapped() ? vpu_rd32(REG_PROGRESS) : 0U;
        if (p2_transfer_detail) {
            fpga_p2_dma_breadcrumb(
                "step=completion edge=fail status=0x%08x isr=0x%08x ctrl2=0x%08x total_bytes=0x%08x bank_stat=0x%08x "
                "polls=%lld saw_enabled=%d",
                completion.status, completion.isr, completion.ctrl2, total_bytes_after_transfer,
                vpu_is_mapped() ? vpu_rd32(REG_BANK_STAT) : 0xFFFFFFFFU, completion.polls,
                completion.saw_enabled ? 1 : 0);
        }
        fpga_dma_trace_record(tag, src_phys, dst_phys, bytes, pre_status, pre_isr, pre_ctrl2, total_bytes_before_clear,
                              pre_vpu_status, pre_vpu_progress, total_bytes_after_transfer, post_vpu_status,
                              post_vpu_progress, now_us() - t0, completion);
        fpga_dma_trace_dump("transfer_completion_failure", nullptr, -1, 0U, tag);
        LOGE("ZDMA transfer did not complete tag=%s src=0x%llx dst=0x%llx bytes=%zu", tag ? tag : "?",
             (unsigned long long) src_phys, (unsigned long long) dst_phys, bytes);
        return false;
    }

    const long long t1                         = now_us();
    const uint32_t  total_bytes_after_transfer = g_dma->ZDMA_CH_TOTAL_BYTE;
    const uint32_t  post_vpu_status            = vpu_is_mapped() ? vpu_rd32(REG_STATUS) : 0U;
    const uint32_t  post_vpu_progress          = vpu_is_mapped() ? vpu_rd32(REG_PROGRESS) : 0U;
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb(
            "step=completion edge=pass status=0x%08x isr=0x%08x ctrl2=0x%08x total_bytes=0x%08x bank_stat=0x%08x "
            "polls=%lld saw_enabled=%d",
            completion.status, completion.isr, completion.ctrl2, total_bytes_after_transfer,
            vpu_is_mapped() ? vpu_rd32(REG_BANK_STAT) : 0xFFFFFFFFU, completion.polls, completion.saw_enabled ? 1 : 0);
    }
    fpga_dma_trace_record(tag, src_phys, dst_phys, bytes, pre_status, pre_isr, pre_ctrl2, total_bytes_before_clear,
                          pre_vpu_status, pre_vpu_progress, total_bytes_after_transfer, post_vpu_status,
                          post_vpu_progress, t1 - t0, completion);
    const uint32_t status      = completion.status;
    const uint32_t state       = status & ZDMA_STATUS_STATE_MASK;
    const uint32_t isr         = completion.isr;
    uint32_t       dst_ddr_off = 0;
    if (phys_to_ddr_offset(dst_phys, bytes, &dst_ddr_off)) {
        if (!fpga_p2_ddr_sync(dst_ddr_off, bytes, true, tag)) {
            return false;
        }
    }

    if (g_bottleneck_summary_enabled && g_token_timing.active) {
        g_token_timing.zdma_descriptors++;
        g_token_timing.zdma_bytes += bytes;
        g_token_timing.zdma_elapsed_us += t1 - t0;
        g_token_timing.zdma_polls += completion.polls;
        if (completion.polls == 0) {
            g_token_timing.zdma_zero_poll_descriptors++;
        }
        if (completion.saw_enabled) {
            g_token_timing.zdma_saw_enabled_descriptors++;
        }
        const char * const effective_tag = tag ? tag : "";
        if (strstr(effective_tag, "WEIGHT") != nullptr) {
            g_token_timing.zdma_weight_descriptors++;
        } else if (strstr(effective_tag, "ACT") != nullptr) {
            g_token_timing.zdma_act_descriptors++;
        } else if (strstr(effective_tag, "SCALE") != nullptr || strstr(effective_tag, "PARAM") != nullptr) {
            g_token_timing.zdma_scale_descriptors++;
        } else if (strstr(effective_tag, "SPU_OUT") != nullptr || strstr(effective_tag, "RESULT") != nullptr) {
            g_token_timing.zdma_result_descriptors++;
        } else {
            g_token_timing.zdma_other_descriptors++;
        }
    }

    LOGDMA(
        "tag=%s src=0x%llx dst=0x%llx bytes=%zu units=bytes ms=%.3f MiB/s=%.1f "
        "completion=isr_ack_then_dma_done_and_ctrl2_en_clear status=0x%08x state=%u isr=0x%08x",
        tag ? tag : "?", (unsigned long long) src_phys, (unsigned long long) dst_phys, bytes,
        (double) (t1 - t0) / 1000.0,
        (t1 > t0) ? (double) bytes * 1000000.0 / ((double) (t1 - t0) * 1024.0 * 1024.0) : 0.0, status, state, isr);
    return true;
}

static bool fpga_dma_copy(uint64_t src_phys, uint64_t dst_phys, size_t bytes, const char * tag);

static bool fpga_dma_write_to_ip(uint32_t offset, size_t bytes, const char * tag) {
    return fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) offset, LMM_BASE_PHYS + (uint64_t) offset, bytes, tag);
}

static bool fpga_dma_read_from_ip(uint32_t offset, size_t bytes, const char * tag) {
    return fpga_dma_copy(LMM_BASE_PHYS + (uint64_t) offset, DDR_BASE_PHYS + (uint64_t) offset, bytes, tag);
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
        const uint32_t status   = vpu_rd32(REG_STATUS);
        const uint32_t progress = vpu_rd32(REG_PROGRESS);
        (void) status;
        (void) progress;
    }
    mmio_fence();
}

static bool fpga_ddr_coherency_stress_test(void) {
    const int               iterations = env_int_value("FPGA_COHERENCY_STRESS_ITERS", 2048, 1, 100000);
    static constexpr size_t kBytes     = 64U;
    static_assert((kBytes % sizeof(uint32_t)) == 0U, "coherency pattern alignment");
    if (!ddr_range_fits(ACT_BASE, kBytes)) {
        LOGE("coherency stress cannot access ACT staging off=0x%08x bytes=%zu", ACT_BASE, kBytes);
        return false;
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        volatile uint32_t * pattern                             = (volatile uint32_t *) ddr_ptr(ACT_BASE, kBytes);
        uint32_t            expected[kBytes / sizeof(uint32_t)] = {};
        for (size_t word = 0; word < kBytes / sizeof(uint32_t); ++word) {
            expected[word] = 0xA5C30000U ^ ((uint32_t) iteration * 0x9E3779B9U) ^ (uint32_t) word;
            pattern[word]  = expected[word];
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
                LOGE("coherency stress mismatch iteration=%d word=%zu expected=0x%08x actual=0x%08x", iteration, word,
                     expected[word], actual);
                return false;
            }
        }
    }

    LOGI("coherency stress passed iterations=%d bytes_per_iteration=%zu source=%s strict=%d whitelist=%d", iterations,
         kBytes, g_ddr_map_source.c_str(), g_strict_coherency ? 1 : 0, g_coherency_platform_whitelisted ? 1 : 0);
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
    const bool ddr_to_ip = phys_range_fits(src_phys, bytes, DDR_BASE_PHYS, g_ddr_map_size) &&
                           phys_range_fits(dst_phys, bytes, LMM_BASE_PHYS, g_vpu_map_size);
    const bool ip_to_ddr = phys_range_fits(src_phys, bytes, LMM_BASE_PHYS, g_vpu_map_size) &&
                           phys_range_fits(dst_phys, bytes, DDR_BASE_PHYS, g_ddr_map_size);
    if (!ddr_to_ip && !ip_to_ddr) {
        LOGE(
            "ZDMA physical range rejected tag=%s src=0x%llx dst=0x%llx bytes=%zu ddr=[0x%llx,+0x%zx) "
            "my_ip=[0x%llx,+0x%zx); require one bounded DDR<->MY_IP transfer",
            tag ? tag : "?", (unsigned long long) src_phys, (unsigned long long) dst_phys, bytes,
            (unsigned long long) DDR_BASE_PHYS, g_ddr_map_size, (unsigned long long) LMM_BASE_PHYS, g_vpu_map_size);
        return false;
    }
    if (g_zdma_max_transfer_bytes < 16U || (g_zdma_max_transfer_bytes & 0xFU) != 0U) {
        LOGE("invalid ZDMA descriptor policy max_bytes=%zu; require a positive 16-byte multiple",
             g_zdma_max_transfer_bytes);
        return false;
    }

    const size_t chunk_bytes = std::min(bytes, g_zdma_max_transfer_bytes);
    const size_t chunk_count = bytes / chunk_bytes + (bytes % chunk_bytes != 0U ? 1U : 0U);
    for (size_t chunk = 0U; chunk < chunk_count; ++chunk) {
        const size_t offset     = chunk * chunk_bytes;
        const size_t this_bytes = std::min(chunk_bytes, bytes - offset);
        char         chunk_tag[48];
        if (chunk_count == 1U) {
            snprintf(chunk_tag, sizeof(chunk_tag), "%s", tag ? tag : "?");
        } else {
            snprintf(chunk_tag, sizeof(chunk_tag), "%s[%zu/%zu]", tag ? tag : "?", chunk + 1U, chunk_count);
        }
        if (!fpga_dma_copy_one(src_phys + (uint64_t) offset, dst_phys + (uint64_t) offset, this_bytes, chunk_tag)) {
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
    const long long t0    = now_us();
    long long       polls = 0;
    while (true) {
        const uint32_t status = vpu_rd32(REG_STATUS);
        if (final_status) {
            *final_status = status;
        }
        if (g_ip_timing_enabled && (g_ip_status_every > 0) && ((polls % g_ip_status_every) == 0)) {
            LOGIP("poll=%lld status=0x%08x progress=0x%08x", polls, status, vpu_rd32(REG_PROGRESS));
        }
        if (status & STATUS_ERROR) {
            LOGE("VPU reported error status=0x%08x progress=0x%08x", status, vpu_rd32(REG_PROGRESS));
            return false;
        }
        if (status & STATUS_DONE) {
            return true;
        }
        if (now_us() - t0 > g_ip_timeout_us) {
            LOGE("VPU timeout status=0x%08x progress=0x%08x", status, vpu_rd32(REG_PROGRESS));
            return false;
        }
        polls++;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }
}

// The VPU DONE bit means its final raw token was accepted by the SPU FIFO; it
// does not mean that the SPU has completed its scale lookup, Q16 accumulation,
// or SPU_OUT write.  P2 must prove this ownership boundary before reusing the
// shared SPU_PARAM/SPU_OUT windows.
static bool wait_spu_stream_quiescent(const char * context, bool require_zero_counters) {
    const long long t0    = now_us();
    long long       polls = 0;
    while (true) {
        const uint32_t status        = vpu_rd32(REG_SPU_STREAM_STATUS);
        const uint32_t count         = vpu_rd32(REG_SPU_STREAM_COUNT);
        const uint32_t done          = vpu_rd32(REG_SPU_STREAM_DONE);
        const uint32_t out           = vpu_rd32(REG_SPU_STREAM_OUT);
        const uint32_t drop          = vpu_rd32(REG_SPU_STREAM_DROP);
        const uint32_t error         = vpu_rd32(REG_SPU_STREAM_ERROR);
        const bool     counters_zero = count == 0U && done == 0U && out == 0U && drop == 0U && error == 0U;
        if ((status & SPU_STREAM_STATUS_QUIESCENT) != 0U && (!require_zero_counters || counters_zero)) {
            g_spu_stream_status = status;
            return true;
        }
        if (now_us() - t0 > g_ip_timeout_us) {
            LOGE(
                "SPU stream quiescence timeout context=%s status=0x%08x count=%u done=%u out=%u drop=%u error=%u "
                "require_zero=%d",
                context ? context : "?", status, count, done, out, drop, error, require_zero_counters ? 1 : 0);
            return false;
        }
        polls++;
        if ((polls & 0x3FF) == 0) {
            sched_yield();
        }
    }
}

static bool wait_spu_stream_outputs(const fpga_tile_job_t & job) {
    const uint32_t  target_out    = job.spu_stream_out_before + (uint32_t) job.rows;
    const uint32_t  expected_raw  = job.spu_stream_count_before + (uint32_t) job.rows * (uint32_t) job.group_blocks;
    const uint32_t  expected_done = job.spu_stream_done_before + 1U;
    const long long t0            = now_us();
    long long       last_log_us   = t0;
    long long       polls         = 0;
    uint32_t        last_count    = job.spu_stream_count_before;
    uint32_t        last_done     = job.spu_stream_done_before;
    uint32_t        last_out      = job.spu_stream_out_before;
    uint32_t        last_drop     = job.spu_stream_drop_before;
    uint32_t        last_error    = job.spu_stream_error_before;
    while (true) {
        const long long now         = now_us();
        const uint32_t  count       = vpu_rd32(REG_SPU_STREAM_COUNT);
        const uint32_t  done_count  = vpu_rd32(REG_SPU_STREAM_DONE);
        const uint32_t  out_count   = vpu_rd32(REG_SPU_STREAM_OUT);
        const uint32_t  drop_count  = vpu_rd32(REG_SPU_STREAM_DROP);
        const uint32_t  error_count = vpu_rd32(REG_SPU_STREAM_ERROR);
        if (drop_count != job.spu_stream_drop_before || error_count != job.spu_stream_error_before) {
            LOGE(
                "SPU stream failed job=%u bank=%d expected_raw=%u expected_out=%u count=%u done=%u out=%u "
                "drop_before=%u drop_now=%u error_before=%u error_now=%u active_job=%u done_job=%u bank_stat=0x%08x "
                "progress=0x%08x last_job=%u last_bank=%u",
                job.job_id, job.bank, expected_raw, target_out, count, done_count, out_count,
                job.spu_stream_drop_before, drop_count, job.spu_stream_error_before, error_count,
                vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB), vpu_rd32(REG_BANK_STAT), vpu_rd32(REG_PROGRESS),
                vpu_rd32(REG_SPU_STREAM_LAST_JOB), vpu_rd32(REG_SPU_STREAM_LAST_BANK));
            return false;
        }
        if (done_count >= expected_done && count != expected_raw) {
            LOGE(
                "SPU stream protocol_mismatch job=%u bank=%d expected_raw=%u count=%u expected_done=%u done=%u out=%u "
                "expected_out=%u active_job=%u done_job=%u bank_stat=0x%08x",
                job.job_id, job.bank, expected_raw, count, expected_done, done_count, out_count, target_out,
                vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB), vpu_rd32(REG_BANK_STAT));
            return false;
        }
        if (out_count >= target_out) {
            const uint32_t last_job  = vpu_rd32(REG_SPU_STREAM_LAST_JOB);
            const uint32_t last_bank = vpu_rd32(REG_SPU_STREAM_LAST_BANK);
            if (count != expected_raw || done_count != expected_done || last_job != job.job_id ||
                (last_bank & 1U) != (uint32_t) (job.bank & 1)) {
                LOGE(
                    "SPU stream completion_mismatch job=%u bank=%d expected_raw=%u count=%u expected_done=%u done=%u "
                    "expected_out=%u out=%u last_job=%u last_bank=%u",
                    job.job_id, job.bank, expected_raw, count, expected_done, done_count, target_out, out_count,
                    last_job, last_bank);
                return false;
            }
            if (!wait_spu_stream_quiescent("stream completion", false)) {
                return false;
            }
            return true;
        }
        const bool counters_changed = count != last_count || done_count != last_done || out_count != last_out ||
                                      drop_count != last_drop || error_count != last_error;
        if (counters_changed || now - last_log_us >= FPGA_STREAM_POLL_LOG_INTERVAL_US) {
            LOGIP(
                "SPU poll job=%u bank=%d expected_raw=%u expected_out=%u count=%u done=%u out=%u drop=%u error=%u "
                "active_job=%u done_job=%u bank_stat=0x%08x status=0x%08x progress=0x%08x last_job=%u last_bank=%u",
                job.job_id, job.bank, expected_raw, target_out, count, done_count, out_count, drop_count, error_count,
                vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB), vpu_rd32(REG_BANK_STAT), vpu_rd32(REG_STATUS),
                vpu_rd32(REG_PROGRESS), vpu_rd32(REG_SPU_STREAM_LAST_JOB), vpu_rd32(REG_SPU_STREAM_LAST_BANK));
            last_count  = count;
            last_done   = done_count;
            last_out    = out_count;
            last_drop   = drop_count;
            last_error  = error_count;
            last_log_us = now;
        }
        if (now - t0 > g_ip_timeout_us) {
            LOGE(
                "SPU stream timeout job=%u bank=%d expected_raw=%u expected_out=%u count=%u done=%u out=%u drop=%u "
                "error=%u active_job=%u done_job=%u bank_stat=0x%08x progress=0x%08x",
                job.job_id, job.bank, expected_raw, target_out, count, done_count, out_count, drop_count, error_count,
                vpu_rd32(REG_ACTIVE_JOB), vpu_rd32(REG_DONE_JOB), vpu_rd32(REG_BANK_STAT), vpu_rd32(REG_PROGRESS));
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
    uint32_t       b;

    if (e == 0U) {
        if (m == 0U) {
            b = s << 31;
        } else {
            uint32_t mant = m;
            uint32_t exp  = 113U;
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

static bool quantize_activation_vector_to(const struct ggml_tensor *      src1,
                                          int64_t                         m,
                                          int64_t                         k,
                                          block_q8_0_t *                  out,
                                          float *                         stored_scales,
                                          fpga_activation_quant_stats_t * stats,
                                          const char *                    consumer_tensor_name,
                                          int                             consumer_layer_id,
                                          int64_t *                       bad_block,
                                          int *                           bad_lane,
                                          float *                         bad_value) {
    const int64_t nb   = k / VPU_QK8_0;
    const char *  base = (const char *) src1->data + m * src1->nb[1];

    // The FPGA hook is entered before ggml-cpu converts src1 into its vec-dot
    // type.  Use the exact same architecture-selected Q8_0 converter here so
    // rounding and, critically, the FP16-stored block scale match the CPU
    // kernel.  A private quantizer or an FP32 "exact" scale creates a different
    // numerical backend and can accumulate hidden-state drift across layers.
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * block_base    = (const float *) (base + ib * VPU_QK8_0 * (int64_t) sizeof(float));
        float         block_max_abs = 0.0f;
        for (int lane = 0; lane < VPU_QK8_0; ++lane) {
            const float value = block_base[lane];
            if (!std::isfinite(value)) {
                long long nan_count       = 0;
                long long inf_count       = 0;
                long long finite_count    = 0;
                int64_t   first_nonfinite = -1;
                float     finite_min      = INFINITY;
                float     finite_max      = -INFINITY;
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
                LOGE(
                    "ACTIVATION_NONFINITE_DETAIL consumer=%s consumer_layer=%d source=%s col=%lld first_index=%lld "
                    "first_block=%lld first_lane=%d value=%.9g nan_count=%lld inf_count=%lld finite_count=%lld "
                    "finite_min=%.9g finite_max=%.9g src1_type=%d src1_ne=[%lld,%lld,%lld,%lld] "
                    "src1_nb=[%lld,%lld,%lld,%lld]",
                    consumer_tensor_name ? consumer_tensor_name : "?", consumer_layer_id, tensor_name_or_unknown(src1),
                    (long long) m, (long long) first_nonfinite, (long long) ib, lane, value, nan_count, inf_count,
                    finite_count, finite_min, finite_max, (int) src1->type, (long long) src1->ne[0],
                    (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3], (long long) src1->nb[0],
                    (long long) src1->nb[1], (long long) src1->nb[2], (long long) src1->nb[3]);
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
            stats->max_abs        = std::max(stats->max_abs, block_max_abs);
            const float raw_scale = block_max_abs / 127.0f;
            stats->max_scale      = std::max(stats->max_scale, raw_scale);
            if (raw_scale > VPU_FP16_MAX_FINITE) {
                if (stats->fp16_scale_overflows == 0) {
                    stats->first_overflow_col   = m;
                    stats->first_overflow_block = ib;
                    stats->first_overflow_abs   = block_max_abs;
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

static bool ensure_quantized_activation_matrix(const struct ggml_tensor *  src1,
                                               int64_t                     m,
                                               int64_t                     k,
                                               std::vector<block_q8_0_t> & act_blocks_all,
                                               std::vector<float> &        act_scales,
                                               bool                        store_act_scales,
                                               fpga_stage_totals_t *       totals,
                                               const char *                tensor_name,
                                               int                         layer_id) {
    const int64_t nb = k / VPU_QK8_0;
    if (nb <= 0 || (uint64_t) m > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) nb) {
        LOGE("activation quantization allocation overflow tensor=%s layer=%d K=%lld M=%lld blocks_per_col=%lld",
             tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) m, (long long) nb);
        return false;
    }
    const size_t total_blocks = (size_t) m * (size_t) nb;
    const bool cache_hit = g_activation_cache_enabled && g_scratch.activation_cache_valid &&
                           g_scratch.cached_src1 == src1 && g_scratch.cached_src1_data == src1->data &&
                           g_scratch.cached_m == m && g_scratch.cached_k == k && g_scratch.cached_nb0 == src1->nb[0] &&
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
    stats.first_overflow_col            = -1;
    stats.first_overflow_block          = -1;
    for (int64_t col = 0; col < m; ++col) {
        block_q8_0_t * col_blocks    = &act_blocks_all[(size_t) col * (size_t) nb];
        int64_t        bad_block     = -1;
        int            bad_lane      = -1;
        float          bad_value     = 0.0f;
        float *        stored_scales = store_act_scales ? &act_scales[(size_t) col * (size_t) nb] : nullptr;
        if (!quantize_activation_vector_to(src1, col, k, col_blocks, stored_scales, &stats, tensor_name, layer_id,
                                           &bad_block, &bad_lane, &bad_value)) {
            LOGE(
                "ACTIVATION_NONFINITE tensor=%s layer=%d col=%lld block=%lld lane=%d value=%.9g; refusing to quantize "
                "invalid F32 activation",
                tensor_name ? tensor_name : "?", layer_id, (long long) col, (long long) bad_block, bad_lane, bad_value);
            return false;
        }
    }

    if (stats.fp16_scale_overflows > 0) {
        g_activation_scale_fp16_overflows += stats.fp16_scale_overflows;
        if (totals) {
            totals->activation_scale_fp16_overflows += stats.fp16_scale_overflows;
        }
        LOGE(
            "ACTIVATION_SCALE_FP16_OVERFLOW tensor=%s layer=%d count=%lld first_col=%lld first_block=%lld "
            "first_amax=%.9g first_scale=%.9g max_amax=%.9g max_scale=%.9g; GGML Q8_0 stores d as FP16, so FPGA "
            "execution must stop instead of substituting an FP32-only scale",
            tensor_name ? tensor_name : "?", layer_id, stats.fp16_scale_overflows, (long long) stats.first_overflow_col,
            (long long) stats.first_overflow_block, stats.first_overflow_abs, stats.first_overflow_scale, stats.max_abs,
            stats.max_scale);
        return false;
    }

    if (g_activation_cache_enabled) {
        g_scratch.cached_src1            = src1;
        g_scratch.cached_src1_data       = src1->data;
        g_scratch.cached_m               = m;
        g_scratch.cached_k               = k;
        g_scratch.cached_nb0             = src1->nb[0];
        g_scratch.cached_nb1             = src1->nb[1];
        g_scratch.activation_cache_valid = true;
    } else {
        g_scratch.activation_cache_valid = false;
    }
    g_activation_cache_misses++;
    return true;
}

static const block_q8_0_t * weight_block_from_base(const struct ggml_tensor * src0,
                                                   const void *               data_base,
                                                   int64_t                    row,
                                                   int64_t                    block) {
    const char * row_base = (const char *) data_base + row * src0->nb[1];
    return (const block_q8_0_t *) row_base + block;
}

static const block_q8_0_t * weight_block(const struct ggml_tensor * src0, int64_t row, int64_t block) {
    return weight_block_from_base(src0, src0->data, row, block);
}

// The stream-mode register is not a best-effort hint: RTL accepts a write
// only while its FIFO is quiescent, no P3 bank is locked, and command-mode SPU
// work is idle.  Readback is mandatory because an ignored write would mix P2
// packed entries with P3 dense tables.  This function is called before any P3
// staging/DMA, so an admission failure has not altered the P3 data plane.
static bool fpga_set_split_scale_mode(uint32_t requested_mode, const char * context) {
    const uint32_t requested = requested_mode & 1U;
    if (g_committed_stream_mode == (int) requested) {
        return true;
    }

    // The cached value is invalid while a real transition is being checked.
    // Any failed MMIO readback therefore leaves the next caller forced through
    // the full quiescence, write, and verification sequence.
    g_committed_stream_mode = -1;
    const uint32_t stream_status = vpu_rd32(REG_SPU_STREAM_STATUS);
    const uint32_t p3_status = vpu_rd32(REG_SPU_STREAM_P3_STATUS);
    const uint32_t spu_status = vpu_rd32(REG_SPU_CTRL);
    const bool quiescent = (stream_status & SPU_STREAM_STATUS_QUIESCENT) != 0U;
    const bool p3_locked = (p3_status & SPU_P3_STATUS_LOCK_VALID) != 0U;
    const bool spu_busy = (spu_status & 1U) != 0U;
    if (!quiescent || p3_locked || spu_busy) {
        LOGE("P3_MODE_TRANSITION_REJECT context=%s requested=%u stream_status=0x%08x p3_status=0x%08x spu_status=0x%08x "
             "reason=not_quiescent_or_locked action=no_p3_data_plane_write",
             context ? context : "?", requested, stream_status, p3_status, spu_status);
        return false;
    }
    mmio_fence();
    vpu_wr32(REG_STREAM_MODE, requested);
    mmio_fence();
    const uint32_t mode_readback = vpu_rd32(REG_STREAM_MODE);
    const uint32_t stream_after = vpu_rd32(REG_SPU_STREAM_STATUS);
    const uint32_t p3_after = vpu_rd32(REG_SPU_STREAM_P3_STATUS);
    const bool mode_matches = (mode_readback & 1U) == requested;
    const bool retained_matches = requested == 0U || (p3_after & SPU_P3_STATUS_MODE_RETAINED) != 0U;
    const bool still_quiescent = (stream_after & SPU_STREAM_STATUS_QUIESCENT) != 0U;
    const bool still_unlocked = (p3_after & SPU_P3_STATUS_LOCK_VALID) == 0U;
    if (!mode_matches || !retained_matches || !still_quiescent || !still_unlocked) {
        LOGE("P3_MODE_TRANSITION_FAIL context=%s requested=%u readback=0x%08x stream_status=0x%08x p3_status=0x%08x "
             "mode_matches=%d retained=%d quiescent=%d unlocked=%d action=no_p3_data_plane_write",
             context ? context : "?", requested, mode_readback, stream_after, p3_after, mode_matches ? 1 : 0,
             retained_matches ? 1 : 0, still_quiescent ? 1 : 0, still_unlocked ? 1 : 0);
        return false;
    }
    g_committed_stream_mode = (int) requested;
    LOGI("P3_MODE_TRANSITION_PASS context=%s mode=%u stream_status=0x%08x p3_status=0x%08x",
         context ? context : "?", requested, stream_after, p3_after);
    return true;
}

static bool fpga_p3_verify_retirement(const fpga_tile_job_t & job) {
    if (!job.p3_split_scale) {
        return true;
    }
    uint64_t retire_timing_start_ns = 0;
    bool     retire_timing_start_valid = false;
    if (g_p3_retire_timing_enabled) {
        g_p3_retire_timing_calls = fpga_saturating_add_u64(g_p3_retire_timing_calls, 1U);
        g_p3_retire_timing_mmio_reads = fpga_saturating_add_u64(g_p3_retire_timing_mmio_reads, 9U);
        retire_timing_start_valid = fpga_p3_retire_timing_now_ns(&retire_timing_start_ns);
        if (!retire_timing_start_valid) {
            g_p3_retire_timing_clock_errors = fpga_saturating_add_u64(g_p3_retire_timing_clock_errors, 1U);
        }
    }
    const uint32_t accepted = vpu_rd32(REG_SPU_STREAM_COUNT) - job.spu_stream_count_before;
    const uint32_t done = vpu_rd32(REG_SPU_STREAM_DONE) - job.spu_stream_done_before;
    const uint32_t entries = vpu_rd32(REG_SPU_STREAM_ENTRY_DONE) - job.spu_stream_entry_done_before;
    const uint32_t finals = vpu_rd32(REG_SPU_STREAM_FINAL_WRITE) - job.spu_stream_final_write_before;
    const uint32_t reject = vpu_rd32(REG_SPU_STREAM_P3_REJECT) - job.spu_stream_p3_reject_before;
    const uint32_t drops = vpu_rd32(REG_SPU_STREAM_DROP) - job.spu_stream_drop_before;
    const uint32_t errors = vpu_rd32(REG_SPU_STREAM_ERROR) - job.spu_stream_error_before;
    const uint32_t expected_entries = (uint32_t) job.rows * (uint32_t) job.group_blocks;
    const uint32_t stream_status = vpu_rd32(REG_SPU_STREAM_STATUS);
    const uint32_t p3_status = vpu_rd32(REG_SPU_STREAM_P3_STATUS);
    const bool exact = accepted == expected_entries && entries == expected_entries && done == 1U &&
                       finals == (uint32_t) job.rows && reject == 0U && drops == 0U && errors == 0U &&
                       (stream_status & SPU_STREAM_STATUS_QUIESCENT) != 0U &&
                       (p3_status & SPU_P3_STATUS_LOCK_VALID) == 0U &&
                       (p3_status & SPU_P3_STATUS_MODE_RETAINED) != 0U;
    if (g_p3_retire_timing_enabled) {
        if (exact) {
            g_p3_retire_timing_passes = fpga_saturating_add_u64(g_p3_retire_timing_passes, 1U);
        } else {
            g_p3_retire_timing_failures = fpga_saturating_add_u64(g_p3_retire_timing_failures, 1U);
        }
        if (retire_timing_start_valid) {
            uint64_t retire_timing_end_ns = 0;
            if (!fpga_p3_retire_timing_now_ns(&retire_timing_end_ns) || retire_timing_end_ns < retire_timing_start_ns) {
                g_p3_retire_timing_clock_errors = fpga_saturating_add_u64(g_p3_retire_timing_clock_errors, 1U);
            } else {
                const uint64_t retire_timing_core_ns = retire_timing_end_ns - retire_timing_start_ns;
                g_p3_retire_timing_valid_samples = fpga_saturating_add_u64(g_p3_retire_timing_valid_samples, 1U);
                g_p3_retire_timing_core_total_ns =
                    fpga_saturating_add_u64(g_p3_retire_timing_core_total_ns, retire_timing_core_ns);
                if (g_p3_retire_timing_valid_samples == 1U || retire_timing_core_ns < g_p3_retire_timing_core_min_ns) {
                    g_p3_retire_timing_core_min_ns = retire_timing_core_ns;
                }
                if (retire_timing_core_ns > g_p3_retire_timing_core_max_ns) {
                    g_p3_retire_timing_core_max_ns = retire_timing_core_ns;
                }
            }
        }
    }
    if (!exact) {
        LOGE("P3_RETIRE_FAIL job=%u bank=%d accepted=%u expected=%u entries=%u done=%u finals=%u expected_finals=%d "
             "reject=%u drops=%u errors=%u stream_status=0x%08x p3_status=0x%08x action=abort_after_p3_commit",
             job.job_id, job.bank, accepted, expected_entries, entries, done, finals, job.rows, reject, drops, errors,
             stream_status, p3_status);
        return false;
    }
    g_p3_jobs++;
    // Qualification keeps an immediately durable record for every checked
    // tile. Production already fails closed on every retirement mismatch, so
    // force-flushing all tens of thousands of successful tiles only measures
    // filesystem latency. Keep the first four and periodic checkpoints; the
    // cleanup summary remains the exact authority for the total job count.
    const bool qualification_retire_proof = g_pl_scale_contract_check_limit > 0;
    const bool production_retire_checkpoint =
        g_p3_jobs <= 4 || (g_p3_jobs % P3_PRODUCTION_RETIRE_LOG_INTERVAL) == 0;
    if (qualification_retire_proof || production_retire_checkpoint) {
        g_p3_retire_pass_logs++;
        LOGI(
            "P3_RETIRE_PASS job=%u bank=%d accepted=%u entries=%u done=1 finals=%u mode_retained=1 "
            "lock_released=1 telemetry=%s",
            job.job_id, job.bank, accepted, entries, finals,
            qualification_retire_proof ? "qualification_every_tile" : "production_checkpoint");
    } else {
        g_p3_retire_pass_suppressed++;
    }
    return true;
}

static inline uint32_t fpga_p3_pack_fp16_pair(uint16_t lane0, uint16_t lane1) {
    return (uint32_t) lane0 | ((uint32_t) lane1 << 16);
}

// This is a host-only arithmetic/layout check.  It deliberately never maps
// hardware and proves that dense eight-lane words retain the source half bits
// verbatim, including a NaN payload that must never be repaired by staging.
static bool fpga_p3_split_scale_host_self_test(void);

static bool fpga_p2_cumulative_tile_limit_reached(long long q16_checks, int tile_limit) {
    return tile_limit > 0 && q16_checks >= (long long) tile_limit;
}

// The terminal boundary is global to the qualification run, not local to the
// current matrix.  A matrix may therefore complete cleanly before the exact
// cumulative tile limit, but a reached limit must always have completed the
// existing DMA/SPU/descriptor boundary sequence.
static bool fpga_p2_cumulative_tile_state_consistent(long long q16_checks, int tile_limit, bool boundary_reached) {
    return fpga_p2_cumulative_tile_limit_reached(q16_checks, tile_limit) == boundary_reached;
}

static bool fpga_p6_parse_required_i64(const char * name, int64_t min_value, int64_t * value) {
    const char * const text = getenv(name);
    if (!text || text[0] == '\0' || !value) {
        return false;
    }
    errno       = 0;
    char * end  = nullptr;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value) {
        return false;
    }
    *value = (int64_t) parsed;
    return true;
}

// P6 captures the deployed ARM accumulation only for the selected decode
// GEMV.  The same tensor can legitimately occur with M>1 during warm-up;
// that work must stay native CPU and must not enter any FPGA accounting.
static bool fpga_p6_selected_m1_eligible(bool selected_name_layer, int64_t shape_m) {
    return selected_name_layer && shape_m == 1;
}

// Invoke this only after the ordered cleanup sequence has released all safe
// host/PL resources.  A configured P6 run without a complete golden capture
// is incomplete evidence even when the selected M=1 GEMV was never reached.
static bool fpga_p6_cleanup_requires_failure(bool fpga_initialized, const fpga_p6_accum_trace_t & trace) {
    return fpga_initialized && trace.enabled && !trace.capture_complete;
}

static bool fpga_p6_job_selected(const fpga_tile_job_t & job) {
    const fpga_p6_accum_trace_t & trace = g_p6_accum_trace;
    if (!trace.enabled || !job.tensor_name || trace.tensor != job.tensor_name || trace.layer != job.layer_id ||
        trace.tile_row0 != job.row0 || job.col != 0 || job.shape_m != 1) {
        return false;
    }
    const int64_t row_end = trace.row_first + (int64_t) trace.row_count;
    const int64_t tile_end = job.row0 + (int64_t) job.rows;
    return trace.row_first >= job.row0 && row_end >= trace.row_first && row_end <= tile_end;
}

// Admit one whole selected K chunk before writing any trace record.  This is
// intentionally host-only state: an error leaves the native CPU destination
// ownership intact and stops the qualification rather than concealing it.
static bool fpga_p6_accept_chunk(const fpga_tile_job_t & job, int * ordinal, bool * final_chunk) {
    fpga_p6_accum_trace_t & trace = g_p6_accum_trace;
    if (!trace.enabled || !ordinal || !final_chunk || !fpga_p6_job_selected(job) || trace.capture_complete) {
        return false;
    }
    const int64_t nb = job.shape_k / VPU_QK8_0;
    if (job.shape_k <= 0 || job.shape_k % VPU_QK8_0 != 0 || job.group_blocks <= 0 || job.k_block0 < 0 ||
        job.k_block0 > nb || (int64_t) job.group_blocks > nb - job.k_block0 ||
        trace.records_emitted > trace.record_limit ||
        (size_t) trace.row_count > trace.record_limit - trace.records_emitted) {
        return false;
    }
    if ((trace.chunks_seen == 0 && job.k_block0 != 0) ||
        (trace.chunks_seen > 0 && job.k_block0 != trace.next_k_block0) ||
        trace.chunks_seen >= trace.expected_k_chunks) {
        return false;
    }
    const bool actual_final = job.k_block0 + (int64_t) job.group_blocks == nb;
    const bool expected_final = trace.chunks_seen + 1 == trace.expected_k_chunks;
    if (actual_final != expected_final) {
        return false;
    }
    *ordinal             = trace.chunks_seen;
    *final_chunk         = actual_final;
    trace.next_k_block0  = job.k_block0 + (int64_t) job.group_blocks;
    trace.chunks_seen++;
    return true;
}

static bool fpga_p6_accum_trace_host_self_test(void);

// Host-only policy check for the cumulative qualification boundary.  The
// sequence models several eligible matrices and proves that a limit larger
// than the first matrix neither aborts at that matrix boundary nor admits a
// tile after the global limit.  It never maps or touches FPGA resources.
static bool fpga_p2_cumulative_tile_limit_host_self_test(void);

// Emits a complete range of pair-major VPU2 WEIGHT words.  This helper is
// deliberately side-effect limited: it neither validates mapping addresses
// nor touches DMA/IP state, logs, locks g_mutex, or aborts.  The caller has
// already admitted the full WEIGHT payload and gives each producer a disjoint
// contiguous pair range.

static void * fpga_p2_pack_worker_main(void *) {
    for (;;) {
        pthread_mutex_lock(&g_p2_pack_worker_mutex);
        while (!g_p2_pack_worker_stop_requested && !g_p2_pack_worker_task_pending) {
            (void) pthread_cond_wait(&g_p2_pack_worker_task_cv, &g_p2_pack_worker_mutex);
        }
        if (g_p2_pack_worker_stop_requested) {
            pthread_mutex_unlock(&g_p2_pack_worker_mutex);
            return nullptr;
        }
        const fpga_p2_pack_worker_task_t task = g_p2_pack_worker_task;
        g_p2_pack_worker_task_pending = false;
        g_p2_pack_worker_busy = true;
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);

        const long long service0 = now_us();
        size_t written_words = 0U;
        const bool range_success = fpga_pack_direct_weight_pair_range(
            task.dst_words, task.src0, task.weight_data_base, task.row0, task.k_block0, task.rows,
            task.group_blocks, task.group_beats, task.pair_begin, task.pair_end, &written_words);
        const bool success = range_success && written_words == task.expected_words;
        // Completion is never published before all volatile WEIGHT stores are
        // globally observed.  The caller performs its own DSB before DMA.
        mmio_fence();

        pthread_mutex_lock(&g_p2_pack_worker_mutex);
        g_p2_pack_worker_busy = false;
        g_p2_pack_worker_completed_generation = task.generation;
        g_p2_pack_worker_completed_success = success;
        g_p2_pack_worker_completed_words = written_words;
        g_p2_pack_worker_completed_service_us = now_us() - service0;
        pthread_cond_broadcast(&g_p2_pack_worker_done_cv);
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    }
}

static bool fpga_p2_pack_worker_start(void) {
    if (g_p2_pack_workers_requested != 2) {
        return true;
    }
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    if (g_p2_pack_worker_created || g_p2_pack_worker_task_pending || g_p2_pack_worker_busy) {
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        return false;
    }
    g_p2_pack_worker_stop_requested = false;
    g_p2_pack_worker_completed_generation = 0U;
    g_p2_pack_worker_completed_success = false;
    g_p2_pack_worker_completed_words = 0U;
    g_p2_pack_worker_completed_service_us = 0;
    const int create_rc = pthread_create(&g_p2_pack_worker_thread, nullptr, fpga_p2_pack_worker_main, nullptr);
    if (create_rc == 0) {
        g_p2_pack_worker_created = true;
    }
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    return create_rc == 0;
}

static bool fpga_p2_pack_worker_next_generation(uint64_t * generation) {
    if (!generation) {
        return false;
    }
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    const bool ready = g_p2_pack_worker_created && !g_p2_pack_worker_stop_requested &&
                       !g_p2_pack_worker_task_pending && !g_p2_pack_worker_busy &&
                       g_p2_pack_worker_next_generation != UINT64_MAX;
    if (ready) {
        *generation = ++g_p2_pack_worker_next_generation;
    }
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    return ready;
}

static bool fpga_p2_pack_worker_submit(const fpga_p2_pack_worker_task_t & task) {
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    if (!g_p2_pack_worker_created || g_p2_pack_worker_stop_requested || g_p2_pack_worker_task_pending ||
        g_p2_pack_worker_busy || task.generation == 0U) {
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        return false;
    }
    g_p2_pack_worker_task = task;
    g_p2_pack_worker_task_pending = true;
    pthread_cond_signal(&g_p2_pack_worker_task_cv);
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    return true;
}

static bool fpga_p2_pack_worker_wait(uint64_t generation, size_t * words, long long * service_us, long long * wait_us) {
    if (!words || !service_us || !wait_us || generation == 0U) {
        return false;
    }
    const long long wait0 = now_us();
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    while (g_p2_pack_worker_created && g_p2_pack_worker_completed_generation < generation) {
        (void) pthread_cond_wait(&g_p2_pack_worker_done_cv, &g_p2_pack_worker_mutex);
    }
    const bool success = g_p2_pack_worker_created && g_p2_pack_worker_completed_generation == generation &&
                         g_p2_pack_worker_completed_success;
    *words = g_p2_pack_worker_completed_words;
    *service_us = g_p2_pack_worker_completed_service_us;
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    *wait_us = now_us() - wait0;
    return success;
}

// Called under g_mutex during explicit cleanup.  A join error is surfaced to
// the caller before any UIO DDR mapping can be invalidated.
static bool fpga_p2_pack_worker_stop(void) {
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    if (!g_p2_pack_worker_created) {
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        return true;
    }
    if (g_p2_pack_worker_task_pending || g_p2_pack_worker_busy) {
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        return false;
    }
    g_p2_pack_worker_stop_requested = true;
    pthread_cond_broadcast(&g_p2_pack_worker_task_cv);
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);

    if (pthread_join(g_p2_pack_worker_thread, nullptr) != 0) {
        return false;
    }

    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    g_p2_pack_worker_created = false;
    g_p2_pack_worker_stop_requested = false;
    g_p2_pack_worker_task_pending = false;
    g_p2_pack_worker_busy = false;
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    return true;
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
static bool fpga_tensor_access_range(const struct ggml_tensor *   tensor,
                                     int64_t                      dim0,
                                     int64_t                      dim1,
                                     size_t                       element_bytes,
                                     fpga_tensor_access_range_t * range) {
    if (!tensor || !tensor->data || dim0 <= 0 || dim1 <= 0) {
        return false;
    }
    if ((uint64_t) dim0 > (uint64_t) std::numeric_limits<size_t>::max() ||
        (uint64_t) dim1 > (uint64_t) std::numeric_limits<size_t>::max()) {
        return false;
    }

    size_t dim0_offset    = 0U;
    size_t dim1_offset    = 0U;
    size_t last_offset    = 0U;
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
        range->end   = begin + required_bytes;
        range->bytes = required_bytes;
    }
    return true;
}

static bool fpga_tensor_ranges_overlap(const fpga_tensor_access_range_t & lhs, const fpga_tensor_access_range_t & rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

static bool fpga_capture_activation_input_snapshot(const struct ggml_tensor * src1,
                                                   int64_t                    k,
                                                   int64_t                    m,
                                                   std::vector<uint8_t> &     snapshot) {
    size_t column_bytes = 0U;
    size_t total_bytes  = 0U;
    if (!src1 || !src1->data || !checked_size_mul((size_t) k, sizeof(float), &column_bytes) ||
        !checked_size_mul((size_t) m, column_bytes, &total_bytes)) {
        return false;
    }

    snapshot.resize(total_bytes);
    const char * const base = (const char *) src1->data;
    for (int64_t col = 0; col < m; ++col) {
        memcpy(snapshot.data() + (size_t) col * column_bytes, base + (size_t) col * src1->nb[1], column_bytes);
    }
    return true;
}

static bool fpga_verify_activation_input_snapshot(const struct ggml_tensor *   src1,
                                                  const struct ggml_tensor *   dst,
                                                  int64_t                      k,
                                                  int64_t                      m,
                                                  const std::vector<uint8_t> & snapshot,
                                                  const char *                 consumer_tensor_name,
                                                  int                          consumer_layer_id) {
    size_t column_bytes   = 0U;
    size_t expected_bytes = 0U;
    if (!src1 || !dst || !src1->data || !checked_size_mul((size_t) k, sizeof(float), &column_bytes) ||
        !checked_size_mul((size_t) m, column_bytes, &expected_bytes) || snapshot.size() != expected_bytes) {
        LOGE(
            "FPGA_INPUT_INTEGRITY_INTERNAL_ERROR tensor=%s layer=%d reason=invalid_snapshot_shape K=%lld M=%lld "
            "snapshot_bytes=%zu",
            consumer_tensor_name ? consumer_tensor_name : "?", consumer_layer_id, (long long) k, (long long) m,
            snapshot.size());
        return false;
    }

    const char * const base = (const char *) src1->data;
    for (int64_t col = 0; col < m; ++col) {
        const uint8_t * const expected = snapshot.data() + (size_t) col * column_bytes;
        const uint8_t * const actual   = (const uint8_t *) base + (size_t) col * src1->nb[1];
        if (memcmp(expected, actual, column_bytes) == 0) {
            continue;
        }

        size_t first_byte = 0U;
        while (first_byte < column_bytes && expected[first_byte] == actual[first_byte]) {
            ++first_byte;
        }
        const size_t first_index   = first_byte / sizeof(float);
        uint32_t     expected_bits = 0U;
        uint32_t     actual_bits   = 0U;
        if (first_index < (size_t) k) {
            memcpy(&expected_bits, expected + first_index * sizeof(float), sizeof(expected_bits));
            memcpy(&actual_bits, actual + first_index * sizeof(float), sizeof(actual_bits));
        }
        fpga_tensor_access_range_t src_range    = {};
        fpga_tensor_access_range_t dst_range    = {};
        const bool                 src_range_ok = fpga_tensor_access_range(src1, k, m, sizeof(float), &src_range);
        const bool dst_range_ok = fpga_tensor_access_range(dst, dst->ne[0], dst->ne[1], sizeof(float), &dst_range);
        LOGE(
            "FPGA_INPUT_MUTATION tensor=%s layer=%d source=%s col=%lld index=%zu byte=%zu expected_bits=0x%08x "
            "actual_bits=0x%08x src=[0x%llx,0x%llx) dst=[0x%llx,0x%llx) ranges_overlap=%d; raw FPGA matmul modified "
            "its F32 input",
            consumer_tensor_name ? consumer_tensor_name : "?", consumer_layer_id, tensor_name_or_unknown(src1),
            (long long) col, first_index, first_byte, expected_bits, actual_bits,
            (unsigned long long) (src_range_ok ? src_range.begin : 0U),
            (unsigned long long) (src_range_ok ? src_range.end : 0U),
            (unsigned long long) (dst_range_ok ? dst_range.begin : 0U),
            (unsigned long long) (dst_range_ok ? dst_range.end : 0U),
            src_range_ok && dst_range_ok && fpga_tensor_ranges_overlap(src_range, dst_range) ? 1 : 0);
        return false;
    }
    return true;
}

static void store_dst_value(const struct ggml_tensor * dst, int64_t row, int64_t col, float value) {
    char *       base   = (char *) dst->data;
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
        unsigned long long start       = 0U;
        unsigned long long end         = 0U;
        unsigned long long file_offset = 0U;
        unsigned long long inode       = 0U;
        char               perms[5]    = {};
        char               dev[32]     = {};
        char               path[768]   = {};
        const int fields = sscanf(line, "%llx-%llx %4s %llx %31s %llu %767[^\n]", &start, &end, perms, &file_offset,
                                  dev, &inode, path);
        if (fields < 6 || address < (uintptr_t) start || address >= (uintptr_t) end) {
            continue;
        }

        info->found       = true;
        info->start       = (uintptr_t) start;
        info->end         = (uintptr_t) end;
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

static void fpga_log_source_file_provenance(const void * source, size_t bytes);

static float load_dst_value(const struct ggml_tensor * dst, int64_t row, int64_t col) {
    const char * base   = (const char *) dst->data;
    const size_t offset = (size_t) row * dst->nb[0] + (size_t) col * dst->nb[1];
    float        value  = 0.0f;
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
static bool fpga_contract_validate_weight_scales(const struct ggml_tensor * src0,
                                                 const void *               weight_data_base,
                                                 const char *               tensor_name,
                                                 int                        layer_id);

static bool fpga_audit_q8_source_only(const struct ggml_tensor * src0, const char * tensor_name, int layer_id);

static void fpga_contract_log_q8_nonfinite_provenance(const struct ggml_tensor * src0,
                                                      const block_q8_0_t *       weight,
                                                      const block_q8_0_t *       act,
                                                      int64_t                    row,
                                                      int64_t                    col,
                                                      const char *               tensor_name,
                                                      int                        layer_id,
                                                      float                      kernel_reference);

// Stage exactly the packed Q8 payload consumed by one legacy VPU launch.
// Keeping this in one helper is intentional: normal launch and forensic
// replay must write byte-for-byte identical ACT/WEIGHT layouts.
static void fpga_stage_q8_group_payload(const block_q8_0_t * weight_snapshot,
                                        const block_q8_0_t * act_group,
                                        int                  rows,
                                        int                  group_blocks,
                                        bool                 write_weight_payload,
                                        uint32_t             weight_dst_off) {
    if (group_blocks > INT_MAX / VPU_BLOCK_BEATS) {
        fpga_fatal("unsupported DMA-to-IP tiling case: group_blocks=%d VPU_BLOCK_BEATS=%d arithmetic=overflow",
                   group_blocks, VPU_BLOCK_BEATS);
    }
    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    if (write_weight_payload) {
        for (int row = 0; row < rows; ++row) {
            for (int gb = 0; gb < group_blocks; ++gb) {
                const block_q8_0_t * wb = &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
                for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                    uint32_t word_off = 0;
                    if (!fpga_weight_layout_word_offset(weight_dst_off, rows, group_beats, row,
                                                        gb * VPU_BLOCK_BEATS + beat, &word_off)) {
                        fpga_fatal("P2 WEIGHT staging offset overflow base=0x%08x rows=%d group_beats=%d row=%d gb=%d beat=%d",
                                   weight_dst_off, rows, group_beats, row, gb, beat);
                    }
                    write_i8x16_to_ddr(word_off, wb->qs + beat * VPU_NUM_LANES);
                }
            }
        }
        fpga_weight_layout_zero_padded_companion(weight_dst_off, rows, group_beats);
    }

    for (int gb = 0; gb < group_blocks; ++gb) {
        const block_q8_0_t & act = act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act.qs + beat * VPU_NUM_LANES);
        }
    }
    mmio_fence();
}

// ddr_high is exposed through UIO/O_SYNC, but this board may report EINVAL for
// msync().  A DSB followed by reads from both ends of the written range drains
// posted PS stores before ZDMA becomes the reader.  The full C0 audit below is
// stronger; this bounded fence remains enabled in normal execution.
static void fpga_ddr_staging_readback_commit(uint32_t off, size_t bytes) {
    if (bytes == 0U || (off & 0x3U) != 0U || (bytes & 0x3U) != 0U) {
        fpga_fatal("DDR staging commit requires a non-empty 32-bit range off=0x%08x bytes=%zu", off, bytes);
    }
    mmio_fence();
    const volatile uint32_t * const first = (volatile const uint32_t *) ddr_ptr(off, sizeof(uint32_t));
    const volatile uint32_t * const last =
        (volatile const uint32_t *) ddr_ptr(off + (uint32_t) bytes - sizeof(uint32_t), sizeof(uint32_t));
    const uint32_t first_value = *first;
    const uint32_t last_value  = *last;
    (void) first_value;
    (void) last_value;
    mmio_fence();
}

static bool fpga_contract_verify_staged_q8_group(const block_q8_0_t * weight_snapshot,
                                                 const block_q8_0_t * act_group,
                                                 int                  rows,
                                                 int                  group_blocks,
                                                 uint32_t             weight_src_off,
                                                 const char *         tensor_name,
                                                 int                  layer_id,
                                                 uint32_t             tile_id,
                                                 const char *         phase);

// The current FPD ZDMA path has no hardware coherency guarantee for its data
// transactions.  The v34 failure was two adjacent 32-byte Q8 blocks in one
// stale DDR cache line; the exact tile passed only after the forensic replay
// performed a full staged-byte readback.  In C0/C1, make that proof part of
// the normal launch sequence and allow one pre-VPU re-stage.  This never
// repairs FPGA results or changes a production tensor: it only prevents a
// known-invalid DDR source from reaching the VPU during a contract run.
static bool fpga_stage_q8_group_with_contract_guard(const block_q8_0_t * weight_snapshot,
                                                    const block_q8_0_t * act_group,
                                                    int                  rows,
                                                    int                  group_blocks,
                                                    bool                 write_weight_payload,
                                                    uint32_t             weight_src_off,
                                                    size_t               act_bytes,
                                                    size_t               weight_bytes,
                                                    bool                 guard_enabled,
                                                    const char *         tensor_name,
                                                    int                  layer_id,
                                                    uint32_t             tile_id);

// A completed ACT DMA is a read from DDR_HIGH and must not modify the
// separately staged WEIGHT source.  v35 found a byte change in WEIGHT after
// ACT completed, before the WEIGHT DMA or VPU start.  Preserve that evidence,
// then re-stage once so a C0 run can continue to collect raw-contract data.
// A recovered staging fault is still a C0 failure for primary-FPGA admission:
// the cleanup summary records staging_restages and must remain zero.
static bool fpga_contract_restage_after_act_dma(const block_q8_0_t * weight_snapshot,
                                                const block_q8_0_t * act_group,
                                                int                  rows,
                                                int                  group_blocks,
                                                bool                 write_weight_payload,
                                                uint32_t             weight_src_off,
                                                size_t               act_bytes,
                                                size_t               weight_bytes,
                                                const char *         tensor_name,
                                                int                  layer_id,
                                                uint32_t             tile_id);

static bool fpga_contract_verify_weight_source_snapshot(const struct ggml_tensor * src0,
                                                        const block_q8_0_t *       weight_snapshot,
                                                        int64_t                    row0,
                                                        int                        rows,
                                                        int64_t                    k_block0,
                                                        int                        group_blocks,
                                                        const char *               tensor_name,
                                                        int                        layer_id,
                                                        uint32_t                   tile_id);

static bool fpga_contract_log_staging_audit(const block_q8_0_t * act,
                                            const block_q8_0_t * weight,
                                            int                  local_row,
                                            int                  group_block,
                                            int                  group_beats,
                                            uint32_t             weight_src_off,
                                            const char *         tensor_name,
                                            int                  layer_id,
                                            uint32_t             tile_id,
                                            const char *         phase);

static long long fpga_contract_count_raw_mismatches(const block_q8_0_t *           weight_snapshot,
                                                    const block_q8_0_t *           act_group,
                                                    int64_t                        row0,
                                                    int                            rows,
                                                    int64_t                        k_block0,
                                                    int                            group_blocks,
                                                    uint32_t                       weight_src_off,
                                                    std::vector<int32_t> &         partial,
                                                    const char *                   tensor_name,
                                                    int                            layer_id,
                                                    uint32_t                       tile_id,
                                                    int                            attempt,
                                                    bool                           log_mismatches,
                                                    bool                           repair_mismatches,
                                                    fpga_raw_mismatch_location_t * first_mismatch);

// A raw mismatch tells us that the model output is not trustworthy.  On an
// aborting contract run, replay only that one failing VPU job and inspect the
// same 32-byte activation/weight block after each boundary.  This is bounded
// to one tile (at most 294,912 bytes of re-staging on the current geometry),
// never builds a weight cache, and does not alter normal inference timing.
static void fpga_contract_forensic_replay(const block_q8_0_t *                 weight_snapshot,
                                          const block_q8_0_t *                 act_group,
                                          int                                  rows,
                                          int                                  group_blocks,
                                          uint32_t                             weight_src_off,
                                          bool                                 weight_cache_hit,
                                          uint32_t                             tile_id,
                                          const char *                         tensor_name,
                                          int                                  layer_id,
                                          const fpga_raw_mismatch_location_t & mismatch);

static bool fpga_contract_check_output_values(const struct ggml_tensor *        src0,
                                              const struct ggml_tensor *        dst,
                                              const std::vector<block_q8_0_t> & act_blocks_all,
                                              const void *                      weight_data_base,
                                              const char *                      tensor_name,
                                              int                               layer_id);

static bool fpga_wait_selftest_spu_stream(int          rows,
                                          int          col_beats,
                                          uint32_t     job_id,
                                          uint32_t     count_before,
                                          uint32_t     done_before,
                                          uint32_t     out_before,
                                          uint32_t     drop_before,
                                          uint32_t     error_before,
                                          const char * tensor_name) {
    if ((col_beats % VPU_BLOCK_BEATS) != 0) {
        LOGE("P2 self-test stream cannot derive Q8 block count tensor=%s col_beats=%d block_beats=%d",
             tensor_name ? tensor_name : "?", col_beats, VPU_BLOCK_BEATS);
        return false;
    }

    fpga_tile_job_t stream_job         = {};
    stream_job.bank                    = 0;
    stream_job.job_id                  = job_id;
    stream_job.rows                    = rows;
    stream_job.group_blocks            = col_beats / VPU_BLOCK_BEATS;
    stream_job.spu_stream_count_before = count_before;
    stream_job.spu_stream_done_before  = done_before;
    stream_job.spu_stream_out_before   = out_before;
    stream_job.spu_stream_drop_before  = drop_before;
    stream_job.spu_stream_error_before = error_before;
    if (!wait_spu_stream_outputs(stream_job)) {
        LOGE("P2 self-test stream finality failed tensor=%s job=%u rows=%d blocks=%d", tensor_name ? tensor_name : "?",
             job_id, rows, stream_job.group_blocks);
        return false;
    }
    return wait_spu_stream_quiescent("self-test stream finality", false);
}

static bool fpga_prime_p2_stream_selftest_scales(void) {
    const size_t bytes = (size_t) P2_REQUIRED_SPU_WORD_CAPACITY * 16U;
    if (!range_fits(SPU_PARAM_BASE, bytes, SPU_PARAM_BASE, SPU_PARAM_END) || !ddr_range_fits(SPU_PARAM_BASE, bytes)) {
        LOGE("P2 self-test scale-table range is invalid bytes=%zu capacity_words=%u", bytes,
             P2_REQUIRED_SPU_WORD_CAPACITY);
        return false;
    }
    // Raw compatibility self-tests have no GGML Q8 scale table.  P2 must not
    // let them consume uninitialized SPU_PARAM contents while it drains their
    // mandatory stream tail, so give every implemented entry an explicit
    // FP16 zero/zero scale pair before the tests begin.
    ddr_zero_range32(SPU_PARAM_BASE, bytes);
    return fpga_dma_write_to_ip(SPU_PARAM_BASE, bytes, "P2_SELFTEST_PARAM_ZERO");
}

static bool fpga_reset_p2_stream_after_selftests(void) {
    // Do not reset before all raw self-test stream entries are drained: the
    // reset intentionally clears ownership/counters but leaves BRAM payload
    // intact.  P2 will subsequently overwrite only the rows/scale words it
    // owns for its first tile.
    g_committed_stream_mode = -1;
    vpu_wr32(REG_SPU_CTRL, SPU_CTRL_SOFT_RESET);
    mmio_fence();
    return wait_spu_stream_quiescent("post-selftest soft reset", true);
}

static bool run_vpu_window_transfer(int                   rows,
                                    int                   col_beats,
                                    uint32_t              mode,
                                    size_t                act_bytes,
                                    size_t                weight_bytes,
                                    size_t                result_bytes,
                                    const char *          tensor_name,
                                    int                   layer_id,
                                    int64_t               k,
                                    int64_t               n,
                                    int64_t               m,
                                    uint32_t              tile_id,
                                    fpga_stage_totals_t * totals) {
    const bool p2_stream_fence_required = g_spu_q8_scale_stream_supported;
    uint32_t   selftest_job_id          = 0U;
    uint32_t   stream_count_before      = 0U;
    uint32_t   stream_done_before       = 0U;
    uint32_t   stream_out_before        = 0U;
    uint32_t   stream_drop_before       = 0U;
    uint32_t   stream_error_before      = 0U;
    if (p2_stream_fence_required) {
        // The raw VPU self-tests also emit VPU->SPU entries.  Snapshot before
        // start so no residual FIFO/accumulator work can cross into P2.
        selftest_job_id     = vpu_rd32(REG_JOB_ID);
        stream_count_before = vpu_rd32(REG_SPU_STREAM_COUNT);
        stream_done_before  = vpu_rd32(REG_SPU_STREAM_DONE);
        stream_out_before   = vpu_rd32(REG_SPU_STREAM_OUT);
        stream_drop_before  = vpu_rd32(REG_SPU_STREAM_DROP);
        stream_error_before = vpu_rd32(REG_SPU_STREAM_ERROR);
    }
    vpu_select_banks(0, 0);
    vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
    // Protocol-2/VPU2 pair transport is not optional for raw diagnostics or
    // self-tests: every launch consumes the same padded WEIGHT ABI as P2.
    configure_vpu(rows, col_beats, mode | VPU_MODE_P2_TWO_ROW);

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
             tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, tile_id,
             vpu_status, vpu_rd32(REG_PROGRESS));
        return false;
    }
    const long long ip1 = now_us();

    if (p2_stream_fence_required &&
        !fpga_wait_selftest_spu_stream(rows, col_beats, selftest_job_id, stream_count_before, stream_done_before,
                                       stream_out_before, stream_drop_before, stream_error_before, tensor_name)) {
        return false;
    }

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
        LOGIP(
            "run tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u rows=%d col_beats=%d mode=0x%x act_dma_ms=%.3f "
            "weight_dma_ms=%.3f ip_ms=%.3f result_dma_ms=%.3f status=0x%08x progress=0x%08x",
            tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, tile_id, rows,
            col_beats, mode, (double) (dma_act1 - dma_act0) / 1000.0, (double) (dma_weight1 - dma_weight0) / 1000.0,
            (double) (ip1 - ip0) / 1000.0, (double) (dma_result1 - dma_result0) / 1000.0, vpu_status,
            vpu_rd32(REG_PROGRESS));
    }
    return true;
}

static bool fpga_dma_basic_self_test(void);

static bool fpga_dma_packed_self_test(void);

static int32_t read_result_i32_flat(uint32_t index) {
    int32_t lanes[VPU_RESULT_PACK_LANES] = {};
    read_result_i32x4_from_ddr(index / (uint32_t) VPU_RESULT_PACK_LANES, lanes);
    return lanes[index % (uint32_t) VPU_RESULT_PACK_LANES];
}

static bool fpga_dma_row_limit_self_test(void);

static bool fpga_validate_tensors(const struct ggml_tensor * src0,
                                  const struct ggml_tensor * src1,
                                  const struct ggml_tensor * dst,
                                  bool                       require_packed_q8_capability,
                                  const char **              reason) {
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
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 ||
        dst->ne[3] != 1) {
        *reason = "unsupported DMA-to-IP tiling case: batched tensor dimensions";
        return false;
    }
    if (src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        *reason = "unsupported DMA-to-IP tiling case: non-F32 row stride";
        return false;
    }
    size_t src0_row_bytes         = 0U;
    size_t src1_row_bytes         = 0U;
    size_t dst_row_bytes          = 0U;
    size_t activation_block_count = 0U;
    if (!checked_size_mul((size_t) (k / VPU_QK8_0), sizeof(block_q8_0_t), &src0_row_bytes) ||
        !checked_size_mul((size_t) k, sizeof(float), &src1_row_bytes) ||
        !checked_size_mul((size_t) n, sizeof(float), &dst_row_bytes) ||
        !checked_size_mul((size_t) m, (size_t) (k / VPU_QK8_0), &activation_block_count)) {
        *reason = "unsupported DMA-to-IP tiling case: tensor layout size overflows host allocation arithmetic";
        return false;
    }
    (void) activation_block_count;
    if (src0->nb[0] != sizeof(block_q8_0_t) || src0->nb[1] < src0_row_bytes || src1->nb[1] < src1_row_bytes ||
        dst->nb[1] < dst_row_bytes) {
        *reason = "unsupported DMA-to-IP tiling case: tensor stride is smaller than its logical row";
        return false;
    }
    if ((src0->nb[1] % alignof(block_q8_0_t)) != 0U || (src1->nb[1] % alignof(float)) != 0U ||
        (dst->nb[1] % alignof(float)) != 0U || ((uintptr_t) src0->data % alignof(block_q8_0_t)) != 0U ||
        ((uintptr_t) src1->data % alignof(float)) != 0U || ((uintptr_t) dst->data % alignof(float)) != 0U) {
        *reason = "unsupported DMA-to-IP tiling case: typed Q8/F32 data or padded row stride is misaligned";
        return false;
    }
    fpga_tensor_access_range_t src0_range = {};
    fpga_tensor_access_range_t src1_range = {};
    fpga_tensor_access_range_t dst_range  = {};
    if (!fpga_tensor_access_range(src0, k / VPU_QK8_0, n, sizeof(block_q8_0_t), &src0_range) ||
        !fpga_tensor_access_range(src1, k, m, sizeof(float), &src1_range) ||
        !fpga_tensor_access_range(dst, n, m, sizeof(float), &dst_range) || src0_range.bytes > ggml_nbytes(src0) ||
        src1_range.bytes > ggml_nbytes(src1) || dst_range.bytes > ggml_nbytes(dst)) {
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
    blocks     = std::min(blocks, result_limited_blocks);
    blocks     = std::min(blocks, remaining_blocks);
    return std::max(1, blocks);
}

static bool fpga_weight_path_bench_pack_cached(const fpga_weight_path_bench_job_t & job,
                                                std::vector<uint32_t> &            cached_words);

static bool fpga_weight_path_bench_revalidate_job(const fpga_weight_path_bench_job_t & job);

static bool fpga_weight_path_bench_store_uio(const std::vector<uint32_t> & cached_words, size_t bytes);

static bool fpga_weight_path_bench_pack_uio(const fpga_weight_path_bench_job_t & job);

static bool fpga_weight_path_bench_reset_required(const fpga_weight_path_bench_trace_t & trace,
                                                  int                                    previous_graph_seq,
                                                  int                                    n_tokens);

// This host-only diagnostic has not written DDR until replay begins.  It is
// therefore safe to discard a trace which crossed a graph boundary before its
// one-token replay point.  Callers hold g_mutex.
static void fpga_weight_path_bench_reset_unreplayed_trace(fpga_weight_path_bench_trace_t * trace,
                                                          int                              new_graph_seq,
                                                          const char *                     reason,
                                                          bool                             emit_log);

static bool fpga_weight_path_bench_prepare_capture(fpga_weight_path_bench_trace_t * trace,
                                                   int                              graph_seq,
                                                   const char **                    reason,
                                                   bool                             emit_log);

static bool fpga_weight_path_bench_capture(const struct ggml_tensor * src0, int graph_seq, const char ** reason);

static bool fpga_weight_path_bench_stats(const std::array<long long, 5> & samples,
                                         long long *                       min_us,
                                         long long *                       median_us,
                                         long long *                       max_us);

enum class fpga_weight_path_bench_phase : uint8_t { PACK_CACHED, STORE_UIO, PACK_UIO };

static bool fpga_weight_path_bench_replay_phase(fpga_weight_path_bench_phase phase,
                                                 bool                         store_timing_only,
                                                 long long *                  store_elapsed_us);

static bool fpga_weight_path_bench_emit_phase(const char * name, fpga_weight_path_bench_phase phase);

static bool fpga_weight_path_bench_host_self_test(void) {
    std::array<long long, 5> samples = {{9, 1, 5, 3, 7}};
    long long min_us = 0;
    long long median_us = 0;
    long long max_us = 0;
    size_t overflow_bytes = 0U;
    if (!fpga_weight_path_bench_stats(samples, &min_us, &median_us, &max_us) || min_us != 1 || median_us != 5 ||
        max_us != 9 || fpga_weight_layout_payload_bytes(3, VPU_BLOCK_BEATS, nullptr) ||
        fpga_weight_layout_payload_bytes(INT_MAX, INT_MAX, &overflow_bytes)) {
        return false;
    }
    std::array<block_q8_0_t, 6> blocks = {};
    for (size_t block = 0; block < blocks.size(); ++block) {
        for (int lane = 0; lane < VPU_QK8_0; ++lane) blocks[block].qs[lane] = (int8_t) (block * 17U + (size_t) lane);
    }
    struct ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_Q8_0;
    tensor.ne[0] = 64;
    tensor.ne[1] = 3;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(block_q8_0_t);
    tensor.nb[1] = 2 * sizeof(block_q8_0_t);
    tensor.nb[2] = 3 * tensor.nb[1];
    tensor.nb[3] = tensor.nb[2];
    tensor.data = blocks.data();
    size_t payload_bytes = 0U;
    if (!fpga_weight_layout_payload_bytes(3, 2 * VPU_BLOCK_BEATS, &payload_bytes)) return false;
    fpga_weight_path_bench_job_t job = {&tensor, blocks.data(), 64, 3, tensor.nb[0], tensor.nb[1],
                                        3 * 2 * sizeof(block_q8_0_t), 0, 3, 0, 2, 2 * VPU_BLOCK_BEATS, payload_bytes};
    std::vector<uint32_t> cached(payload_bytes / sizeof(uint32_t));
    std::vector<uint32_t> direct(payload_bytes / sizeof(uint32_t));
    size_t direct_words = 0U;
    const bool direct_ok = fpga_pack_direct_weight_pair_range((volatile uint32_t *) direct.data(), &tensor, blocks.data(),
                                                               0, 0, 3, 2, 2 * VPU_BLOCK_BEATS, 0, 2, &direct_words);
    fpga_weight_path_bench_trace_t trace = {};
    trace.enabled                         = true;
    trace.graph_seq                       = 7;
    trace.payload_bytes                   = 64U;
    trace.jobs.push_back(job);
    trace.cached_payload.assign(4U, 0U);
    // An exact one-token boundary retains the trace so captures from the same
    // graph continue to aggregate.  A later graph must discard only host
    // trace state before it binds the new sequence.
    const char * lifecycle_reason = nullptr;
    const bool same_sequence_aggregates = fpga_weight_path_bench_prepare_capture(&trace, 7, &lifecycle_reason, false) &&
                                          trace.jobs.size() == 1U && trace.payload_bytes == 64U;
    const bool stale_sequence_resets = fpga_weight_path_bench_reset_required(trace, 8, 1) &&
                                       fpga_weight_path_bench_prepare_capture(&trace, 8, &lifecycle_reason, false);
    return fpga_weight_path_bench_pack_cached(job, cached) && direct_ok && direct_words == direct.size() &&
           memcmp(cached.data(), direct.data(), payload_bytes) == 0 && same_sequence_aggregates &&
           stale_sequence_resets && trace.graph_seq == 8 && trace.jobs.empty() && trace.payload_bytes == 0U &&
           trace.cached_payload.empty();
}

static bool fpga_weight_path_bench_replay_at_boundary(int previous_graph_seq, int n_tokens);

static bool weight_cache_entry_matches(const fpga_weight_cache_entry_t & entry, const struct ggml_tensor * src0) {
    return entry.valid && entry.tensor == src0 && entry.data == src0->data && entry.k == src0->ne[0] &&
           entry.n == src0->ne[1] && entry.nb1 == (size_t) src0->nb[1] && entry.max_rows == g_vpu_max_rows &&
           entry.max_beats == g_vpu_max_beats && entry.max_group_blocks == g_packed_q8_max_blocks;
}

static bool validate_weight_cache_entry(fpga_weight_cache_entry_t * entry, fpga_stage_totals_t * totals) {
    if (!entry || !entry->valid || !ddr_range_fits(entry->header_off, sizeof(fpga_weight_cache_header_t)) ||
        !ddr_range_fits(entry->base_off, entry->bytes)) {
        return false;
    }

    const fpga_weight_cache_header_t header               = ddr_read_weight_cache_header(entry->header_off);
    const uint32_t                   expected_tensor_hash = fpga_tensor_id_from_ptr(entry->tensor);
    const uint32_t                   expected_tile_shape =
        ((uint32_t) entry->max_rows & 0xFFFFU) | (((uint32_t) entry->max_group_blocks & 0xFFFFU) << 16);
    const bool header_ok = header.magic == FPGA_WEIGHT_CACHE_MAGIC &&
                           header.format_version == FPGA_WEIGHT_CACHE_FORMAT_VERSION &&
                           header.tensor_hash == expected_tensor_hash && header.tile_shape == expected_tile_shape &&
                           header.ddr_offset == entry->base_off && header.byte_length == entry->bytes &&
                           header.crc32 == entry->payload_crc32 && header.valid == 1U;
    if (!header_ok) {
        LOGE(
            "weight tile cache header/CRC mismatch tensor=%s header_off=0x%08x payload_off=0x%08x bytes=%zu "
            "header_ok=%d expected_crc=0x%08x actual_crc=0x%08x; entry invalidated",
            tensor_name_or_unknown(entry->tensor), entry->header_off, entry->base_off, entry->bytes, 0,
            entry->payload_crc32, 0U);
        entry->valid = false;
        return false;
    }

    // A full CRC reads the entire tensor through an uncached UIO mapping.  On
    // this board that would cost hundreds of milliseconds per decode token.
    // The header is checked on every lookup; the payload CRC is checked once
    // after construction, or on every lookup only when explicitly requested
    // for a corruption diagnostic.
    if (!entry->crc_validated || g_weight_cache_crc_verify_each_lookup) {
        const long long crc0         = now_us();
        const uint32_t  computed_crc = fpga_crc32_ddr(entry->base_off, entry->bytes);
        const long long crc_us       = now_us() - crc0;
        if (totals) {
            totals->weight_cache_crc_us += crc_us;
        }
        g_weight_cache_crc_us += crc_us;
        if (computed_crc != entry->payload_crc32) {
            LOGE(
                "weight tile cache CRC mismatch tensor=%s header_off=0x%08x payload_off=0x%08x bytes=%zu "
                "expected_crc=0x%08x actual_crc=0x%08x; entry invalidated",
                tensor_name_or_unknown(entry->tensor), entry->header_off, entry->base_off, entry->bytes,
                entry->payload_crc32, computed_crc);
            entry->valid = false;
            return false;
        }
        entry->crc_validated = true;
    }
    return true;
}

static fpga_weight_cache_entry_t * find_weight_cache_entry(const struct ggml_tensor * src0,
                                                           fpga_stage_totals_t *      totals) {
    for (fpga_weight_cache_entry_t & entry : g_weight_cache) {
        if (weight_cache_entry_matches(entry, src0)) {
            if (validate_weight_cache_entry(&entry, totals)) {
                return &entry;
            }
        }
    }
    return nullptr;
}

static fpga_weight_cache_entry_t * build_weight_cache_entry(const struct ggml_tensor * src0,
                                                            fpga_stage_totals_t *      totals) {
    if (!g_weight_cache_enabled || !ddr_is_mapped()) {
        return nullptr;
    }

    const int64_t                         k  = src0->ne[0];
    const int64_t                         n  = src0->ne[1];
    const int64_t                         nb = k / VPU_QK8_0;
    std::vector<fpga_weight_tile_cache_t> tiles;
    size_t                                total_bytes  = 0;
    size_t                                total_scales = 0;

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int                remaining_blocks = (int) (nb - ib0);
            const int                group_blocks     = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int                group_beats      = group_blocks * VPU_BLOCK_BEATS;
            const size_t             weight_bytes     = weight_window_bytes_for_rows(rows, group_beats);
            fpga_weight_tile_cache_t tile             = {};
            tile.row0                                 = row0;
            tile.rows                                 = rows;
            tile.k_block0                             = ib0;
            tile.group_blocks                         = group_blocks;
            tile.group_beats                          = group_beats;
            tile.bytes                                = weight_bytes;
            tile.scale_off                            = total_scales;
            tiles.push_back(tile);
            total_bytes += align_up_size(weight_bytes, WEIGHT_CACHE_ALIGN);
            total_scales += (size_t) rows * (size_t) group_blocks;
            ib0 += group_blocks;
        }
    }

    const uint64_t header_off = align_up_size((size_t) g_weight_cache_next_off, WEIGHT_CACHE_ALIGN);
    const uint64_t payload_off =
        align_up_size((size_t) header_off + sizeof(fpga_weight_cache_header_t), WEIGHT_CACHE_ALIGN);
    const uint64_t cache_end = payload_off + (uint64_t) total_bytes;
    if (cache_end > (uint64_t) g_weight_cache_end_off || cache_end > UINT32_MAX) {
        if (!g_weight_cache_full_logged) {
            LOGE(
                "weight tile cache full; tensor=%s needs=%zu available=%zu. Continuing with per-run weight packing for "
                "uncached tensors.",
                tensor_name_or_unknown(src0), total_bytes,
                g_weight_cache_end_off > g_weight_cache_next_off ?
                    (size_t) (g_weight_cache_end_off - g_weight_cache_next_off) :
                    0U);
            g_weight_cache_full_logged = true;
        }
        return nullptr;
    }

    fpga_weight_cache_entry_t entry = {};
    entry.tensor                    = src0;
    entry.data                      = src0->data;
    entry.k                         = k;
    entry.n                         = n;
    entry.nb1                       = (size_t) src0->nb[1];
    entry.max_rows                  = g_vpu_max_rows;
    entry.max_beats                 = g_vpu_max_beats;
    entry.max_group_blocks          = g_packed_q8_max_blocks;
    entry.header_off                = (uint32_t) header_off;
    entry.base_off                  = (uint32_t) payload_off;
    entry.bytes                     = total_bytes;
    entry.valid                     = false;
    entry.crc_validated             = false;
    entry.tiles                     = std::move(tiles);
    entry.scales.resize(total_scales);

    const fpga_weight_cache_header_t pending_header = {
        FPGA_WEIGHT_CACHE_MAGIC,
        FPGA_WEIGHT_CACHE_FORMAT_VERSION,
        fpga_tensor_id_from_ptr(src0),
        ((uint32_t) entry.max_rows & 0xFFFFU) | (((uint32_t) entry.max_group_blocks & 0xFFFFU) << 16),
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
    if (!msync_ddr_range(entry.header_off, sizeof(pending_header), false, "weight_cache_prepare_header")) {
        return nullptr;
    }

    uint32_t             next_off          = entry.base_off;
    uint32_t             payload_crc_state = 0xFFFFFFFFU;
    size_t               payload_progress  = 0;
    size_t               next_progress_log = 32U * 1024U * 1024U;
    std::vector<uint8_t> crc_tile_bytes;
    LOGI(
        "weight tile cache build start tensor=%s tiles=%zu payload_mib=%.3f header=0x%08x base=0x%08x; CRC is streamed "
        "from cacheable host tile buffers (no full uncached DDR sweep)",
        tensor_name_or_unknown(src0), entry.tiles.size(), (double) entry.bytes / (1024.0 * 1024.0), entry.header_off,
        entry.base_off);
    for (fpga_weight_tile_cache_t & tile : entry.tiles) {
        tile.ddr_off = next_off;
        if (!ddr_range_fits(tile.ddr_off, tile.bytes)) {
            LOGE("weight tile cache range overflow tensor=%s off=0x%08x bytes=%zu", tensor_name_or_unknown(src0),
                 tile.ddr_off, tile.bytes);
            return nullptr;
        }

        crc_tile_bytes.resize(tile.bytes);
        for (int row = 0; row < tile.rows; ++row) {
            for (int gb = 0; gb < tile.group_blocks; ++gb) {
                const block_q8_0_t * wb = weight_block(src0, tile.row0 + row, tile.k_block0 + gb);
                entry.scales[tile.scale_off + (size_t) row * (size_t) tile.group_blocks + (size_t) gb] =
                    fp16_to_fp32(wb->d);
                for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                    uint32_t word_off = 0;
                    if (!fpga_weight_layout_word_offset(tile.ddr_off, tile.rows, tile.group_beats, row,
                                                        gb * VPU_BLOCK_BEATS + beat, &word_off)) {
                        LOGE("weight tile cache P2 layout offset overflow tensor=%s tile_row0=%lld rows=%d group_beats=%d row=%d gb=%d beat=%d",
                             tensor_name_or_unknown(src0), (long long) tile.row0, tile.rows, tile.group_beats, row, gb,
                             beat);
                        return nullptr;
                    }
                    const size_t         byte_offset = (size_t) (word_off - tile.ddr_off);
                    const int8_t * const source      = wb->qs + beat * VPU_NUM_LANES;
                    ddr_write_i8x16(word_off, source);
                    memcpy(crc_tile_bytes.data() + byte_offset, source, 16U);
                }
            }
        }
        fpga_weight_layout_zero_padded_companion(tile.ddr_off, tile.rows, tile.group_beats);

        payload_crc_state = fpga_crc32_update(payload_crc_state, crc_tile_bytes.data(), crc_tile_bytes.size());
        const size_t padded_tile_bytes = align_up_size(tile.bytes, WEIGHT_CACHE_ALIGN);
        const size_t padding_bytes     = padded_tile_bytes - tile.bytes;
        if (padding_bytes != 0U) {
            // tile.bytes is 16-byte aligned and WEIGHT_CACHE_ALIGN is a
            // multiple of four, hence this preserves ddr_zero_range32's
            // alignment contract without touching the rest of the cache.
            ddr_zero_range32(tile.ddr_off + (uint32_t) tile.bytes, padding_bytes);
            payload_crc_state = fpga_crc32_update_zeros(payload_crc_state, padding_bytes);
        }
        payload_progress += padded_tile_bytes;
        if (payload_progress >= next_progress_log || payload_progress == entry.bytes) {
            LOGI("weight tile cache build progress tensor=%s packed_mib=%.3f/%.3f", tensor_name_or_unknown(src0),
                 (double) payload_progress / (1024.0 * 1024.0), (double) entry.bytes / (1024.0 * 1024.0));
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
    entry.payload_crc32                         = ~payload_crc_state;
    const long long            crc_us           = 0;
    fpga_weight_cache_header_t committed_header = pending_header;
    committed_header.crc32                      = entry.payload_crc32;
    committed_header.valid                      = 1U;
    ddr_write_weight_cache_header(entry.header_off, committed_header);
    if (!msync_ddr_range(entry.header_off, sizeof(committed_header), false, "weight_cache_commit")) {
        return nullptr;
    }
    entry.valid             = true;
    entry.crc_validated     = true;
    const long long pack_us = now_us() - pack0;
    if (totals) {
        totals->weight_pack_us += pack_us;
    }
    g_weight_pack_us += pack_us;
    g_weight_cache_next_off = next_off;
    g_weight_cache_bytes += (long long) entry.bytes;
    g_weight_cache_builds++;
    LOGI(
        "weight tile cache build complete tensor=%s tiles=%zu bytes=%zu scales=%zu header=0x%08x base=0x%08x "
        "crc32=0x%08x crc_mode=streamed_host pack_ms=%.3f crc_ms=%.3f next=0x%08x",
        tensor_name_or_unknown(src0), entry.tiles.size(), entry.bytes, entry.scales.size(), entry.header_off,
        entry.base_off, entry.payload_crc32, (double) pack_us / 1000.0, (double) crc_us / 1000.0,
        g_weight_cache_next_off);

    g_weight_cache.push_back(std::move(entry));
    return &g_weight_cache.back();
}

static fpga_weight_cache_entry_t * get_weight_cache_entry(const struct ggml_tensor * src0,
                                                          fpga_stage_totals_t *      totals) {
    const long long             lookup0   = now_us();
    fpga_weight_cache_entry_t * entry     = find_weight_cache_entry(src0, totals);
    const long long             lookup_us = now_us() - lookup0;
    if (totals) {
        totals->weight_cache_lookup_us += lookup_us;
    }
    g_weight_cache_lookup_us += lookup_us;
    if (entry) {
        return entry;
    }
    return build_weight_cache_entry(src0, totals);
}

// Routine residency telemetry is opt-in and shares the normal buffered file
// logger.  Forced records remain reserved for validation/invalidation events
// whose evidence must survive a later failure or abort.
static void fpga_p2_residency_log(bool force_flush, const char * event, const char * fmt, ...);

static bool fpga_p2_residency_has_reportable_activity() {
    const bool counter_activity =
        g_p2_resident_tile_count != 0U || g_p2_residency_next_slot != 0U ||
        g_p2_residency_next_off != WEIGHT_CACHE_BASE || g_p2_residency_builds != 0 ||
        g_p2_residency_hits != 0 || g_p2_residency_misses != 0 || g_p2_residency_build_failures != 0 ||
        g_p2_residency_logical_bytes != 0 || g_p2_residency_miss_alignment != 0 ||
        g_p2_residency_miss_shape != 0 || g_p2_residency_miss_collision != 0 ||
        g_p2_residency_miss_poison != 0 || g_p2_residency_miss_stale != 0 ||
        g_p2_residency_miss_mismatch != 0 || g_p2_residency_miss_capacity != 0 ||
        g_p2_residency_miss_quiescence != 0 || g_p2_residency_miss_range != 0 ||
        g_p2_residency_miss_verify != 0 || g_p2_residency_probe_count != 0 ||
        g_p2_residency_probe_exhausted != 0 || g_p2_residency_host_metadata_hits != 0 ||
        g_p2_residency_host_metadata_invalidations != 0 || g_p2_residency_volatile_ddr_reads != 0 ||
        g_p2_residency_build_us != 0 || g_p2_residency_select_us != 0 ||
        g_p2_residency_metadata_validate_us != 0 || g_p2_residency_resident_param_us != 0 ||
        g_p2_residency_direct_weight_pack_us != 0 || g_p2_residency_direct_weight_pack_bytes != 0U ||
        g_p2_residency_avoided_cpu_pack_bytes != 0 || g_p2_residency_avoided_ddr_to_ip_bytes != 0;

    return g_p2_weight_residency_env_requested || g_p2_weight_residency_enabled ||
           g_p2_weight_residency_diagnostic || g_p2_residency_trace_enabled || g_p2_residency_verify_metadata ||
           g_p2_resident_tile_count != 0U || counter_activity;
}

static uint64_t fpga_p2_residency_hash_mix(uint64_t hash, uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    return hash;
}

static uint64_t fpga_p2_residency_key_hash(const struct ggml_tensor * src0,
                                           int64_t                    row0,
                                           int                        rows,
                                           int64_t                    k_block0,
                                           int                        group_blocks,
                                           int                        group_beats,
                                           size_t                     qs_bytes,
                                           size_t                     scale_bytes) {
    uint64_t hash = 0x84222325cbf29ce4ULL;
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) (uintptr_t) src0);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) (uintptr_t) src0->data);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) src0->type);
    hash = fpga_p2_residency_hash_mix(hash, P2_WEIGHT_RESIDENCY_LAYOUT_V2);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) row0);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) rows);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) k_block0);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) group_blocks);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) group_beats);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) qs_bytes);
    hash = fpga_p2_residency_hash_mix(hash, (uint64_t) scale_bytes);
    hash = fpga_p2_residency_hash_mix(hash, g_p2_weight_residency_epoch);
    hash = fpga_p2_residency_hash_mix(hash, g_stream_protocol_version);
    hash = fpga_p2_residency_hash_mix(hash, g_bitstream_id);
    hash = fpga_p2_residency_hash_mix(hash, g_p2_stream_abi_signature);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        hash = fpga_p2_residency_hash_mix(hash, (uint64_t) src0->ne[d]);
        hash = fpga_p2_residency_hash_mix(hash, (uint64_t) src0->nb[d]);
    }
    return hash;
}

static size_t fpga_p2_residency_bucket(uint64_t key_hash) {
    static_assert((P2_WEIGHT_RESIDENCY_INDEX_BUCKETS & (P2_WEIGHT_RESIDENCY_INDEX_BUCKETS - 1U)) == 0U,
                  "P2 residency index bucket count must be a power of two");
    return (size_t) key_hash & (P2_WEIGHT_RESIDENCY_INDEX_BUCKETS - 1U);
}

static uint32_t fpga_p2_residency_metadata_seal(uint32_t tile_seal,
                                                uint32_t scale_crc32,
                                                size_t   scale_count,
                                                uint64_t validated_epoch) {
    uint64_t mix = ((uint64_t) tile_seal << 32U) | scale_crc32;
    mix = fpga_p2_residency_hash_mix(mix, (uint64_t) scale_count);
    mix = fpga_p2_residency_hash_mix(mix, validated_epoch);
    mix ^= 0x4d4554415343414cULL; // "METASCAL"
    uint32_t seal = (uint32_t) mix ^ (uint32_t) (mix >> 32U);
    return seal == 0U ? 1U : seal;
}

static bool fpga_p2_residency_align_up(size_t value, size_t alignment, size_t * out) {
    if (!out || alignment == 0U || (alignment & (alignment - 1U)) != 0U ||
        value > std::numeric_limits<size_t>::max() - (alignment - 1U)) {
        return false;
    }
    *out = (value + alignment - 1U) & ~(alignment - 1U);
    return true;
}

static bool fpga_p2_residency_range_end(uint32_t begin, size_t bytes, uint32_t limit, uint32_t * end) {
    if (!end || begin < WEIGHT_CACHE_BASE || begin > limit || bytes == 0U || bytes > UINT32_MAX) {
        return false;
    }
    const uint64_t end64 = (uint64_t) begin + (uint64_t) bytes;
    if (end64 > limit || end64 > P2_WEIGHT_RESIDENCY_END || end64 > DDR_REGION_SIZE) {
        return false;
    }
    *end = (uint32_t) end64;
    return true;
}

static bool fpga_p2_residency_host_metadata_shape_valid(const fpga_p2_resident_tile_t & tile) {
    return tile.sealed && !tile.poisoned && tile.scale_count != 0U &&
           tile.scale_count == tile.scale_bits.size() &&
           tile.scale_count == tile.scale_bytes / sizeof(uint16_t);
}

static uint32_t fpga_p2_residency_scale_bits_crc32(const std::vector<uint16_t> & scale_bits) {
    uint32_t crc = 0xFFFFFFFFU;
    for (const uint16_t bits : scale_bits) {
        const uint8_t canonical_le[2] = {(uint8_t) (bits & 0xffU), (uint8_t) ((bits >> 8) & 0xffU)};
        crc = fpga_crc32_update(crc, canonical_le, sizeof(canonical_le));
    }
    return ~crc;
}

static bool fpga_p2_residency_host_metadata_valid(const fpga_p2_resident_tile_t & tile) {
    // Normal reuse is deliberately O(1): the sealed vector was read back and
    // CRC-validated during build, and no mutable handle is exposed outside
    // this mutex-protected directory.  A full vector CRC is a diagnostic-only
    // integrity experiment because doing it on every hit defeats residency.
    if (!fpga_p2_residency_host_metadata_shape_valid(tile) ||
        tile.metadata_validated_epoch == 0U || tile.metadata_validated_epoch != tile.epoch ||
        tile.metadata_validated_epoch != g_p2_weight_residency_epoch) {
        return false;
    }
    if (tile.metadata_seal == 0U ||
        tile.metadata_seal != fpga_p2_residency_metadata_seal(tile.seal, tile.scale_crc32, tile.scale_count,
                                                              tile.metadata_validated_epoch)) {
        return false;
    }
    return !g_p2_residency_verify_metadata || fpga_p2_residency_scale_bits_crc32(tile.scale_bits) == tile.scale_crc32;
}

static bool fpga_p2_residency_host_metadata_valid_timed(const fpga_p2_resident_tile_t & tile) {
    const long long validation0 = now_us();
    const bool valid = fpga_p2_residency_host_metadata_valid(tile);
    g_p2_residency_metadata_validate_us += now_us() - validation0;
    return valid;
}

static bool fpga_p2_residency_identity_matches(const fpga_p2_resident_tile_t & tile,
                                               const struct ggml_tensor *      src0,
                                               int64_t                         row0,
                                               int                             rows,
                                               int64_t                         k_block0,
                                               int                             group_blocks,
                                               int                             group_beats,
                                               size_t                          qs_bytes,
                                               size_t                          scale_bytes) {
    size_t qs_padded = 0;
    size_t scale_padded = 0;
    if (!fpga_p2_residency_align_up(qs_bytes, WEIGHT_CACHE_ALIGN, &qs_padded) ||
        !fpga_p2_residency_align_up(scale_bytes, WEIGHT_CACHE_ALIGN, &scale_padded) ||
        qs_padded > std::numeric_limits<size_t>::max() - scale_padded) {
        return false;
    }
    if (!tile.sealed || tile.poisoned || tile.tensor != src0 || tile.data != src0->data || tile.type != src0->type ||
        tile.layout_version != P2_WEIGHT_RESIDENCY_LAYOUT_V2 || tile.row0 != row0 || tile.rows != rows ||
        tile.k_block0 != k_block0 || tile.group_blocks != group_blocks || tile.group_beats != group_beats ||
        tile.qs_bytes != qs_bytes || tile.scale_bytes != scale_bytes || tile.epoch != g_p2_weight_residency_epoch ||
        tile.stream_protocol != g_stream_protocol_version || tile.bitstream_id != g_bitstream_id ||
        tile.p2_abi != g_p2_stream_abi_signature || tile.seal == 0U ||
        (tile.qs_off & (WEIGHT_CACHE_ALIGN - 1U)) != 0U || (tile.scale_off & (WEIGHT_CACHE_ALIGN - 1U)) != 0U ||
        (uint64_t) tile.scale_off != (uint64_t) tile.qs_off + qs_padded ||
        tile.allocation_bytes != qs_padded + scale_padded) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tile.ne[d] != src0->ne[d] || tile.nb[d] != src0->nb[d]) {
            return false;
        }
    }
    const uint64_t phys_begin = DDR_BASE_PHYS + (uint64_t) tile.qs_off;
    const uint64_t phys_end = phys_begin + (uint64_t) tile.allocation_bytes;
    return tile.qs_off >= WEIGHT_CACHE_BASE && (uint64_t) tile.qs_off + tile.allocation_bytes <= P2_WEIGHT_RESIDENCY_END &&
           phys_begin >= DDR_BASE_PHYS && phys_end >= phys_begin && phys_end <= DDR_END_EXCLUSIVE &&
           tile.allocation_bytes != 0U && ddr_range_fits(tile.qs_off, tile.qs_bytes) &&
           ddr_range_fits(tile.scale_off, tile.scale_bytes);
}

static void fpga_p2_residency_poison_slot(uint32_t slot, const char * reason) {
    if (slot >= g_p2_resident_tiles.size()) {
        return;
    }
    fpga_p2_resident_tile_t & tile = g_p2_resident_tiles[slot];
    if (tile.sealed || tile.building || tile.enabled) {
        fpga_p2_residency_log(true, "INVALIDATE", "slot=%u reason=%s epoch=%llu seal=0x%08x action=direct_stage_no_reuse",
                              slot, reason ? reason : "?", (unsigned long long) tile.epoch, tile.seal);
    }
    tile.sealed   = false;
    tile.building = false;
    tile.poisoned = true;
    g_p2_residency_host_metadata_invalidations++;
}

static bool fpga_p2_resident_scale_bits(uint32_t slot, size_t index, uint16_t * bits) {
    if (slot >= g_p2_resident_tiles.size()) {
        return false;
    }
    const fpga_p2_resident_tile_t & tile = g_p2_resident_tiles[slot];
    if (!bits || !fpga_p2_residency_host_metadata_shape_valid(tile) || index >= tile.scale_count) {
        return false;
    }
    *bits = tile.scale_bits[index];
    return true;
}

// Select an exact sealed slot or construct one in an unused range. No
// descriptor, bank selector, START, or DMA programming occurs here.
// A residency validation failure returns INVALID_SLOT so the caller fails
// closed before direct-stage, DMA, or START. Ordinary capacity/collision
// misses return NO_SLOT and directly stage the immutable source.
static uint32_t fpga_p2_residency_select_or_build_impl(const struct ggml_tensor * src0,
                                                        const void *               weight_data_base,
                                                        int64_t                    row0,
                                                        int                        rows,
                                                        int64_t                    k_block0,
                                                        int                        group_blocks,
                                                        int                        group_beats) {
    if (!g_p2_weight_residency_enabled) {
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    const size_t qs_bytes    = weight_window_bytes_for_rows(rows, group_beats);
    const size_t scale_bytes = (size_t) rows * (size_t) group_blocks * sizeof(uint16_t);
    size_t qs_padded = 0;
    size_t scale_padded = 0;
    if (!fpga_p2_residency_align_up(qs_bytes, WEIGHT_CACHE_ALIGN, &qs_padded) ||
        !fpga_p2_residency_align_up(scale_bytes, WEIGHT_CACHE_ALIGN, &scale_padded) ||
        qs_padded > std::numeric_limits<size_t>::max() - scale_padded) {
        g_p2_residency_misses++;
        g_p2_residency_miss_alignment++;
        fpga_p2_residency_log(false, "MISS", "reason=payload_alignment_overflow action=direct_stage");
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    const size_t payload_bytes = qs_padded + scale_padded;
    if (qs_bytes == 0U || scale_bytes == 0U) {
        g_p2_residency_misses++;
        g_p2_residency_miss_shape++;
        fpga_p2_residency_log(false, "MISS", "reason=invalid_payload_shape action=direct_stage qs_bytes=%zu scale_bytes=%zu",
                              qs_bytes, scale_bytes);
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    const uint64_t key_hash =
        fpga_p2_residency_key_hash(src0, row0, rows, k_block0, group_blocks, group_beats, qs_bytes, scale_bytes);
    const size_t bucket = fpga_p2_residency_bucket(key_hash);
    size_t insertion_bucket = P2_WEIGHT_RESIDENCY_INDEX_BUCKETS;
    for (size_t probe = 0; probe < P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES; ++probe) {
        const size_t index_bucket = (bucket + probe) & (P2_WEIGHT_RESIDENCY_INDEX_BUCKETS - 1U);
        g_p2_residency_probe_count++;
        const uint32_t indexed_slot = g_p2_residency_index[index_bucket];
        if (indexed_slot == P2_WEIGHT_RESIDENCY_NO_SLOT) {
            insertion_bucket = index_bucket;
            break;
        }
        if (indexed_slot >= g_p2_resident_tiles.size()) {
            g_p2_residency_misses++;
            g_p2_residency_miss_poison++;
            fpga_p2_residency_log(true, "MISS", "reason=index_poison action=direct_stage bucket=%zu probe=%zu slot=%u",
                                  index_bucket, probe, indexed_slot);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
        const fpga_p2_resident_tile_t & tile = g_p2_resident_tiles[indexed_slot];
        if (tile.poisoned || tile.building || !tile.sealed || !tile.enabled) {
            g_p2_residency_misses++;
            g_p2_residency_miss_poison++;
            g_p2_residency_host_metadata_invalidations++;
            fpga_p2_residency_log(true, "MISS", "reason=poison_or_unsealed action=direct_stage bucket=%zu probe=%zu slot=%u",
                                  index_bucket, probe, indexed_slot);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
        if (tile.key_hash != key_hash) {
            g_p2_residency_miss_collision++;
            continue;
        }
        if (tile.epoch != g_p2_weight_residency_epoch || tile.stream_protocol != g_stream_protocol_version ||
            tile.bitstream_id != g_bitstream_id || tile.p2_abi != g_p2_stream_abi_signature) {
            g_p2_residency_misses++;
            g_p2_residency_miss_stale++;
            fpga_p2_residency_log(true, "MISS", "reason=index_stale_identity action=direct_stage bucket=%zu probe=%zu slot=%u",
                                  index_bucket, probe, indexed_slot);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
        if (!fpga_p2_residency_host_metadata_valid_timed(tile)) {
            g_p2_residency_misses++;
            g_p2_residency_miss_mismatch++;
            fpga_p2_residency_poison_slot(indexed_slot, "host_scale_crc_or_metadata_seal_mismatch");
            fpga_p2_residency_log(true, "MISS", "reason=host_scale_crc_or_metadata_seal_mismatch "
                                  "action=fail_closed_no_direct_stage bucket=%zu probe=%zu slot=%u",
                                  index_bucket, probe, indexed_slot);
            return P2_WEIGHT_RESIDENCY_INVALID_SLOT;
        }
        if (!fpga_p2_residency_identity_matches(tile, src0, row0, rows, k_block0, group_blocks, group_beats,
                                                qs_bytes, scale_bytes)) {
            g_p2_residency_misses++;
            g_p2_residency_miss_mismatch++;
            fpga_p2_residency_poison_slot(indexed_slot, "host_metadata_or_identity_mismatch");
            fpga_p2_residency_log(true, "MISS", "reason=index_identity_mismatch action=fail_closed_no_direct_stage bucket=%zu probe=%zu slot=%u",
                                  index_bucket, probe, indexed_slot);
            return P2_WEIGHT_RESIDENCY_INVALID_SLOT;
        }
        g_p2_residency_hits++;
        g_p2_residency_host_metadata_hits++;
        fpga_p2_residency_log(false, "HIT", "slot=%u bucket=%zu probe=%zu epoch=%llu seal=0x%08x qs_off=0x%08x scale_off=0x%08x "
                                         "bytes=%zu",
                              indexed_slot, index_bucket, probe, (unsigned long long) tile.epoch, tile.seal, tile.qs_off,
                              tile.scale_off, tile.qs_bytes + tile.scale_bytes);
        return indexed_slot;
    }
    if (insertion_bucket == P2_WEIGHT_RESIDENCY_INDEX_BUCKETS) {
        g_p2_residency_misses++;
        g_p2_residency_probe_exhausted++;
        fpga_p2_residency_log(false, "MISS", "reason=index_probe_exhausted action=direct_stage bucket=%zu max_probes=%zu",
                              bucket, P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES);
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    const size_t budget_bytes = (size_t) g_p2_weight_residency_budget_mb * 1024U * 1024U;
    const uint64_t budget_end64 = (uint64_t) WEIGHT_CACHE_BASE + (uint64_t) budget_bytes;
    uint32_t allocation_end = 0;
    if (budget_end64 > P2_WEIGHT_RESIDENCY_END ||
        !fpga_p2_residency_range_end(g_p2_residency_next_off, payload_bytes, (uint32_t) budget_end64, &allocation_end)) {
        g_p2_residency_misses++;
        g_p2_residency_miss_capacity++;
        fpga_p2_residency_log(false, "MISS", "reason=budget_or_physical_range action=direct_stage qs_bytes=%zu scale_bytes=%zu budget=%zu",
                              qs_bytes, scale_bytes, budget_bytes);
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    // A build is optional. While P1 owns either bank, decline immediately so
    // residency never blocks waiting for a running descriptor to retire.
    if (g_pingpong_scheduler_enabled) {
        mmio_fence();
        const uint32_t status_before = vpu_rd32(REG_STATUS);
        const uint32_t bank_stat     = vpu_rd32(REG_BANK_STAT);
        const uint32_t slot_state    = vpu_rd32(REG_SLOT_STATE);
        const uint32_t desc_flags    = vpu_rd32(REG_DESC_FLAGS);
        const uint32_t status_after  = vpu_rd32(REG_STATUS);
        mmio_fence();
        const bool p1_registers_idle = status_before == status_after && (status_before & STATUS_BUSY) == 0U &&
                                       (bank_stat & BANK_STAT_BUSY) == 0U &&
                                       slot_state == fpga_slot_state_word(0, FPGA_SLOT_FREE, FPGA_SLOT_FREE) &&
                                       desc_flags == 0U;
        if (!p1_registers_idle) {
            g_p2_residency_misses++;
            g_p2_residency_miss_quiescence++;
            fpga_p2_residency_log(false, "MISS", "reason=active_p1_registers_not_idle action=defer_direct_stage "
                                          "status_before=0x%08x status_after=0x%08x bank_stat=0x%08x "
                                          "slot_state=0x%08x flags=0x%08x",
                                  status_before, status_after, bank_stat, slot_state, desc_flags);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
    }
    if (!zdma_wait_channel_disabled("P2_RESIDENCY", "before_build") ||
        !wait_spu_stream_quiescent("P2 residency build", false)) {
        g_p2_residency_misses++;
        g_p2_residency_miss_quiescence++;
        fpga_p2_residency_log(false, "MISS", "reason=quiescence_gate action=direct_stage");
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    mmio_fence();
    if (vpu_rd32(REG_SLOT_STATE) != fpga_slot_state_word(0, FPGA_SLOT_FREE, FPGA_SLOT_FREE) ||
        vpu_rd32(REG_DESC_FLAGS) != 0U) {
        g_p2_residency_misses++;
        g_p2_residency_miss_quiescence++;
        fpga_p2_residency_log(false, "MISS", "reason=descriptor_not_free action=direct_stage");
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }

    if (g_p2_residency_next_slot >= g_p2_resident_tiles.size()) {
        g_p2_residency_misses++;
        g_p2_residency_miss_capacity++;
        fpga_p2_residency_log(false, "MISS", "reason=directory_full_non_evicting action=direct_stage slots=%zu",
                              g_p2_resident_tiles.size());
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    const uint32_t slot = (uint32_t) g_p2_residency_next_slot++;
    const long long build0 = now_us();

    fpga_p2_resident_tile_t pending = {};
    pending.enabled        = true;
    pending.building       = true;
    pending.tensor         = src0;
    pending.data           = src0->data;
    pending.type           = src0->type;
    pending.layout_version = P2_WEIGHT_RESIDENCY_LAYOUT_V2;
    pending.row0           = row0;
    pending.rows           = rows;
    pending.k_block0       = k_block0;
    pending.group_blocks   = group_blocks;
    pending.group_beats    = group_beats;
    pending.qs_bytes       = qs_bytes;
    pending.scale_bytes    = scale_bytes;
    pending.allocation_bytes = payload_bytes;
    pending.qs_off         = g_p2_residency_next_off;
    const uint64_t scale_off64 = (uint64_t) pending.qs_off + (uint64_t) qs_padded;
    if (scale_off64 > UINT32_MAX) {
        g_p2_residency_build_us += now_us() - build0;
        g_p2_residency_misses++;
        g_p2_residency_miss_range++;
        fpga_p2_residency_log(true, "MISS", "reason=scale_offset_overflow action=direct_stage slot=%u", slot);
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    pending.scale_off      = (uint32_t) scale_off64;
    pending.epoch          = g_p2_weight_residency_epoch;
    pending.stream_protocol = g_stream_protocol_version;
    pending.bitstream_id    = g_bitstream_id;
    pending.p2_abi          = g_p2_stream_abi_signature;
    pending.key_hash        = key_hash;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        pending.ne[d] = src0->ne[d];
        pending.nb[d] = src0->nb[d];
    }
    if (!ddr_range_fits(pending.qs_off, pending.qs_bytes) || !ddr_range_fits(pending.scale_off, pending.scale_bytes) ||
        allocation_end > P2_WEIGHT_RESIDENCY_END || allocation_end > (uint32_t) DDR_REGION_SIZE ||
        DDR_BASE_PHYS > UINT64_MAX - (uint64_t) allocation_end ||
        DDR_BASE_PHYS + (uint64_t) allocation_end > DDR_END_EXCLUSIVE) {
        g_p2_residency_build_us += now_us() - build0;
        g_p2_residency_misses++;
        g_p2_residency_miss_range++;
        fpga_p2_residency_log(false, "MISS", "reason=map_or_physical_range action=direct_stage slot=%u", slot);
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    // Reserve before touching DDR. A failed slot remains poisoned and its
    // physical range is never recycled during this process epoch.
    g_p2_residency_next_off = allocation_end;
    fpga_p2_residency_log(false, "BUILD_BEGIN", "slot=%u epoch=%llu base=0x%08x qs_bytes=%zu scale_bytes=%zu row0=%lld rows=%d "
                                               "k_block0=%lld group_blocks=%d protocol=0x%08x bitstream=0x%08x p2_abi=0x%08x",
                          slot, (unsigned long long) pending.epoch, pending.qs_off, qs_bytes, scale_bytes,
                          (long long) row0, rows, (long long) k_block0, group_blocks, pending.stream_protocol,
                          pending.bitstream_id, pending.p2_abi);

    std::vector<uint8_t> expected_qs(qs_bytes);
    std::vector<uint8_t> expected_scales(scale_bytes);
    pending.scale_bits.resize(scale_bytes / sizeof(uint16_t));
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * wb = weight_block_from_base(src0, weight_data_base, row0 + row, k_block0 + gb);
            const size_t scale_index = (size_t) row * (size_t) group_blocks + (size_t) gb;
            const uint16_t weight_scale_bits = (uint16_t) wb->d;
            pending.scale_bits[scale_index]        = weight_scale_bits;
            expected_scales[scale_index * 2U]     = (uint8_t) (weight_scale_bits & 0xffU);
            expected_scales[scale_index * 2U + 1U] = (uint8_t) ((weight_scale_bits >> 8) & 0xffU);
            for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                size_t word = 0;
                if (!fpga_weight_layout_word_index(rows, group_beats, row, gb * VPU_BLOCK_BEATS + beat, &word)) {
                    g_p2_residency_build_us += now_us() - build0;
                    g_p2_residency_misses++;
                    g_p2_residency_miss_range++;
                    fpga_p2_residency_log(true, "VERIFY_FAIL", "reason=layout_offset_overflow row=%d gb=%d beat=%d",
                                          row, gb, beat);
                    return P2_WEIGHT_RESIDENCY_NO_SLOT;
                }
                memcpy(expected_qs.data() + word * 16U, wb->qs + beat * VPU_NUM_LANES, 16U);
            }
        }
    }
    volatile uint8_t * dst_qs = (volatile uint8_t *) ddr_ptr(pending.qs_off, qs_bytes);
    volatile uint8_t * dst_scales = (volatile uint8_t *) ddr_ptr(pending.scale_off, scale_bytes);
    for (size_t i = 0; i < qs_bytes; ++i) dst_qs[i] = expected_qs[i];
    for (size_t i = 0; i < scale_bytes; ++i) dst_scales[i] = expected_scales[i];
    // Keep the odd-row companion explicit even though expected_qs was
    // zero-initialized: no prior resident payload may leak into VPU2's pair.
    fpga_weight_layout_zero_padded_companion(pending.qs_off, rows, group_beats);
    if (!fpga_p2_ddr_sync(pending.qs_off, qs_bytes, false, "p2_residency_qs") ||
        !fpga_p2_ddr_sync(pending.scale_off, scale_bytes, false, "p2_residency_scales")) {
        g_p2_residency_build_us += now_us() - build0;
        fpga_p2_residency_log(true, "VERIFY_FAIL", "reason=ddr_write_sync");
        fpga_p2_residency_poison_slot(slot, "ddr_write_sync");
        g_p2_residency_build_failures++;
        g_p2_residency_misses++;
        g_p2_residency_miss_verify++;
        fpga_p2_residency_log(false, "MISS", "reason=ddr_write_sync action=direct_stage slot=%u", slot);
        return P2_WEIGHT_RESIDENCY_NO_SLOT;
    }
    fpga_p2_residency_log(false, "DDR_WRITE_SYNC", "qs_off=0x%08x qs_bytes=%zu scale_off=0x%08x scale_bytes=%zu",
                          pending.qs_off, qs_bytes, pending.scale_off, scale_bytes);
    const volatile uint8_t * verify_qs = (volatile const uint8_t *) ddr_ptr(pending.qs_off, qs_bytes);
    const volatile uint8_t * verify_scales = (volatile const uint8_t *) ddr_ptr(pending.scale_off, scale_bytes);
    for (size_t i = 0; i < qs_bytes; ++i) {
        if (verify_qs[i] != expected_qs[i]) {
            g_p2_residency_build_us += now_us() - build0;
            fpga_p2_residency_log(true, "VERIFY_FAIL", "kind=qs offset=%zu expected=0x%02x actual=0x%02x", i,
                                  expected_qs[i], verify_qs[i]);
            fpga_p2_residency_poison_slot(slot, "qs_readback_mismatch");
            g_p2_residency_build_failures++;
            g_p2_residency_misses++;
            g_p2_residency_miss_verify++;
            fpga_p2_residency_log(false, "MISS", "reason=qs_readback_mismatch action=direct_stage slot=%u", slot);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
    }
    for (size_t i = 0; i < scale_bytes; ++i) {
        if (verify_scales[i] != expected_scales[i]) {
            g_p2_residency_build_us += now_us() - build0;
            fpga_p2_residency_log(true, "VERIFY_FAIL", "kind=scale offset=%zu expected=0x%02x actual=0x%02x", i,
                                  expected_scales[i], verify_scales[i]);
            fpga_p2_residency_poison_slot(slot, "scale_readback_mismatch");
            g_p2_residency_build_failures++;
            g_p2_residency_misses++;
            g_p2_residency_miss_verify++;
            fpga_p2_residency_log(false, "MISS", "reason=scale_readback_mismatch action=direct_stage slot=%u", slot);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
    }
    uint32_t crc = fpga_crc32_update(0xFFFFFFFFU, expected_qs.data(), expected_qs.size());
    pending.crc32 = ~fpga_crc32_update(crc, expected_scales.data(), expected_scales.size());
    pending.scale_count = pending.scale_bits.size();
    pending.scale_crc32 = ~fpga_crc32_update(0xFFFFFFFFU, expected_scales.data(), expected_scales.size());
    fpga_p2_residency_log(false, "VERIFY_PASS", "qs_bytes=%zu scale_bytes=%zu crc32=0x%08x", qs_bytes, scale_bytes,
                          pending.crc32);
    // Two-phase seal: pending metadata is published only after exact readback,
    // then the nonzero seal makes it selectable by a later job.
    fpga_p2_residency_log(false, "SEAL", "phase=prepare epoch=%llu", (unsigned long long) pending.epoch);
    pending.seal = pending.crc32 ^ 0xA5C35A3CU;
    if (pending.seal == 0U) pending.seal = 1U;
    pending.building = false;
    pending.sealed   = true;
    // Publish this only after exact DDR readback, both CRCs, and the tile
    // seal have completed.  Normal reuse may therefore trust the immutable
    // vector count/seal without re-hashing the entire scale vector.
    pending.metadata_validated_epoch = pending.epoch;
    pending.metadata_seal = fpga_p2_residency_metadata_seal(
        pending.seal, pending.scale_crc32, pending.scale_count, pending.metadata_validated_epoch);
    std::atomic_thread_fence(std::memory_order_release);
    g_p2_resident_tiles[slot] = pending;
    std::atomic_thread_fence(std::memory_order_release);
    g_p2_residency_index[insertion_bucket] = slot;
    g_p2_resident_tile_count++;
    g_p2_residency_builds++;
    g_p2_residency_build_us += now_us() - build0;
    g_p2_residency_logical_bytes += (long long) (pending.qs_bytes + pending.scale_bytes);
    fpga_p2_residency_log(false, "SEAL", "phase=commit slot=%u bucket=%zu epoch=%llu metadata_validated_epoch=%llu seal=0x%08x metadata_seal=0x%08x "
                                  "scale_count=%zu scale_crc32=0x%08x qs_off=0x%08x scale_off=0x%08x",
                          slot, insertion_bucket, (unsigned long long) pending.epoch,
                          (unsigned long long) pending.metadata_validated_epoch, pending.seal, pending.metadata_seal,
                          pending.scale_count, pending.scale_crc32, pending.qs_off,
                          pending.scale_off);
    return slot;
}

static uint32_t fpga_p2_residency_select_or_build(const struct ggml_tensor * src0,
                                                   const void *               weight_data_base,
                                                   int64_t                    row0,
                                                   int                        rows,
                                                   int64_t                    k_block0,
                                                   int                        group_blocks,
                                                   int                        group_beats) {
    const long long select0 = now_us();
    const uint32_t slot = fpga_p2_residency_select_or_build_impl(src0, weight_data_base, row0, rows, k_block0,
                                                                   group_blocks, group_beats);
    g_p2_residency_select_us += now_us() - select0;
    return slot;
}

static void p2_trace_first_tile(const fpga_tile_job_t & job, const char * stage, const char * edge);

static void p2_trace_set_job_context(const fpga_tile_job_t & job);

static bool p2_trace_this_tile();

static bool fpga_prepare_q8_tile_job(fpga_tile_job_t &                 job,
                                     const struct ggml_tensor *        src0,
                                     const void *                      weight_data_base,
                                     const block_q8_0_t *              act_group,
                                     int64_t                           row0,
                                     int                               rows,
                                     int64_t                           k_block0,
                                     int                               group_blocks,
                                     int64_t                           col,
                                     uint32_t                          weight_tile_index,
                                     const fpga_weight_cache_entry_t * weight_cache,
                                     uint32_t                          tile_id,
                                     int                               bank,
                                     fpga_stage_totals_t *             totals);

static void p2_trace_first_tile(const fpga_tile_job_t & job, const char * stage, const char * edge) {
    if (!p2_trace_this_tile()) {
        return;
    }
    LOGI("P2_TILE_TRACE stage=%s edge=%s job=%u bank=%d tile=%u", stage, edge, job.job_id, job.bank, job.tile_id);
    if (g_p2_terminal_trace_enabled) {
        fprintf(stderr, "[FPGA][P2] stage=%s edge=%s job=%u bank=%d tile=%u\n", stage, edge, job.job_id, job.bank,
                job.tile_id);
        fflush(stderr);
    }
}

static bool fpga_q8_input_preload_key_matches(const fpga_tile_job_t & job) {
    return job.input_preloaded && !job.input_preload_poisoned && job.preload_key_job_id == job.job_id &&
           job.preload_key_tensor == job.src0 && job.preload_key_tensor_id == job.tensor_id &&
           job.preload_key_row0 == job.row0 && job.preload_key_rows == job.rows && job.preload_key_col == job.col &&
           job.preload_key_k_block0 == job.k_block0 &&
           job.preload_key_group_blocks == job.group_blocks && job.preload_key_act_bytes == job.act_bytes &&
           job.preload_key_weight_bytes == job.weight_bytes && job.preload_key_scale_bytes == job.scale_bytes &&
           job.preload_key_weight_src_off == job.weight_src_off &&
           job.preload_key_weight_layout_version == P2_WEIGHT_RESIDENCY_LAYOUT_V2 &&
           job.preload_key_p2_residency_slot == job.p2_residency_slot &&
           job.preload_key_p2_residency_epoch == job.p2_residency_epoch &&
           job.preload_key_p2_residency_seal == job.p2_residency_seal && job.preload_key_bank == (job.bank & 1);
}

static bool fpga_p2_residency_job_still_valid(const fpga_tile_job_t & job) {
    if (!job.p2_residency_hit) {
        return true;
    }
    const bool valid = g_p2_weight_residency_enabled && job.p2_residency_slot < g_p2_resident_tiles.size() &&
                       job.weight_src_off == g_p2_resident_tiles[job.p2_residency_slot].qs_off &&
                       job.weight_bytes == g_p2_resident_tiles[job.p2_residency_slot].qs_bytes &&
                       job.p2_residency_epoch == g_p2_resident_tiles[job.p2_residency_slot].epoch &&
                       job.p2_residency_seal == g_p2_resident_tiles[job.p2_residency_slot].seal &&
                       fpga_p2_residency_host_metadata_valid_timed(g_p2_resident_tiles[job.p2_residency_slot]) &&
                       fpga_p2_residency_identity_matches(g_p2_resident_tiles[job.p2_residency_slot], job.src0, job.row0,
                                                          job.rows, job.k_block0, job.group_blocks, job.group_beats,
                                                          job.weight_bytes,
                                                          (size_t) job.rows * (size_t) job.group_blocks * sizeof(uint16_t));
    if (!valid) {
        fpga_p2_residency_log(
            true, "INVALIDATE", "reason=post_selection_binding_mismatch job=%u tile=%u slot=%u weight_src_off=0x%08x "
                          "weight_bytes=%zu action=no_dma_no_start",
            job.job_id, job.tile_id, job.p2_residency_slot, job.weight_src_off, job.weight_bytes);
        fpga_p2_residency_poison_slot(job.p2_residency_slot, "post_selection_binding_mismatch");
    }
    return valid;
}

// P1 samples live ownership immediately before changing the inactive write
// bank.  The bracketing status reads make a transition during the MMIO read
// sequence observable, so it cannot be mistaken for either admitted state.
typedef struct {
    uint32_t status_before;
    uint32_t bank_stat;
    uint32_t active_job;
    uint32_t done_job;
    uint32_t descriptor_job;
    uint32_t slot_state;
    uint32_t status_after;
} fpga_p1_preload_register_snapshot_t;

enum fpga_p1_preload_state_t : uint32_t {
    FPGA_P1_PRELOAD_STATE_ACTIVE,
    FPGA_P1_PRELOAD_STATE_TERMINAL,
    FPGA_P1_PRELOAD_STATE_MISMATCH,
};

static constexpr uint32_t FPGA_P1_PRELOAD_RESNAPSHOT_LIMIT = 4U;

static fpga_p1_preload_register_snapshot_t fpga_p1_preload_register_snapshot(void) {
    // A DSB/CPU fence before and after the read sequence prevents host-side
    // reordering around this ownership decision.  This is observational only:
    // it performs no W1C, descriptor, bank, or control write.
    mmio_fence();
    const fpga_p1_preload_register_snapshot_t snapshot = {
        vpu_rd32(REG_STATUS),    vpu_rd32(REG_BANK_STAT), vpu_rd32(REG_ACTIVE_JOB),
        vpu_rd32(REG_DONE_JOB),  vpu_rd32(REG_JOB_ID),     vpu_rd32(REG_SLOT_STATE),
        vpu_rd32(REG_STATUS),
    };
    mmio_fence();
    return snapshot;
}

static fpga_p1_preload_state_t fpga_classify_p1_preload_state(
    const fpga_p1_preload_register_snapshot_t & snapshot, const fpga_tile_job_t & running) {
    const uint32_t status_mask = STATUS_DONE | STATUS_BUSY | STATUS_ERROR;
    const uint32_t bank_mask   = BANK_STAT_BUSY | BANK_STAT_DONE | BANK_STAT_ERROR;
    const uint32_t expected_slot = fpga_slot_state_word(running.bank, FPGA_SLOT_COMPUTING, FPGA_SLOT_FREE);
    const uint32_t status_before = snapshot.status_before & status_mask;
    const uint32_t status_after  = snapshot.status_after & status_mask;
    const bool status_stable = status_before == status_after;
    const bool descriptor_running = snapshot.descriptor_job == running.job_id && snapshot.slot_state == expected_slot;
    const bool active_bank_running =
        (((snapshot.bank_stat & BANK_STAT_ACTIVE_BANK) != 0U) ? 1 : 0) == (running.bank & 1);

    if (status_stable && status_before == STATUS_BUSY && (snapshot.bank_stat & bank_mask) == BANK_STAT_BUSY &&
        active_bank_running && snapshot.active_job == running.job_id && descriptor_running) {
        return FPGA_P1_PRELOAD_STATE_ACTIVE;
    }

    const bool done_bank_running =
        (((snapshot.bank_stat & BANK_STAT_DONE_BANK) != 0U) ? 1 : 0) == (running.bank & 1);
    if (status_stable && status_before == STATUS_DONE && (snapshot.bank_stat & bank_mask) == BANK_STAT_DONE &&
        active_bank_running && done_bank_running && snapshot.active_job == running.job_id &&
        snapshot.done_job == running.job_id && descriptor_running) {
        return FPGA_P1_PRELOAD_STATE_TERMINAL;
    }

    return FPGA_P1_PRELOAD_STATE_MISMATCH;
}

// A BUSY -> DONE pair with the running descriptor still owned by the running
// job is the only mismatch eligible for re-sampling.  The active-bank,
// descriptor, and slot checks remain mandatory; bank status may contain the
// old or new BUSY/DONE value while the VPU completes.  Error bits and every
// other ownership mismatch fail closed without another read sequence.
static bool fpga_p1_preload_snapshot_may_be_transitioning(
    const fpga_p1_preload_register_snapshot_t & snapshot, const fpga_tile_job_t & running) {
    const uint32_t bank_state = snapshot.bank_stat & (BANK_STAT_BUSY | BANK_STAT_DONE | BANK_STAT_ERROR);
    const bool active_bank_running =
        (((snapshot.bank_stat & BANK_STAT_ACTIVE_BANK) != 0U) ? 1 : 0) == (running.bank & 1);
    const uint32_t expected_slot = fpga_slot_state_word(running.bank, FPGA_SLOT_COMPUTING, FPGA_SLOT_FREE);
    const bool bank_status_transition = bank_state == BANK_STAT_BUSY || bank_state == BANK_STAT_DONE ||
                                         bank_state == (BANK_STAT_BUSY | BANK_STAT_DONE);

    return (snapshot.status_before & (STATUS_DONE | STATUS_BUSY | STATUS_ERROR)) == STATUS_BUSY &&
           (snapshot.status_after & (STATUS_DONE | STATUS_BUSY | STATUS_ERROR)) == STATUS_DONE &&
           bank_status_transition && active_bank_running && snapshot.active_job == running.job_id &&
           snapshot.descriptor_job == running.job_id && snapshot.slot_state == expected_slot;
}

// P1 overlap is deliberately narrow.  The running job retains its descriptor,
// VPU configuration, read-bank selection, SPU_PARAM/SPU_OUT, and CTRL state.
// Only ZDMA ACT/WEIGHT writes to the other bank are admitted here.
static bool fpga_preload_q8_tile_inputs(fpga_tile_job_t &       job,
                                        const fpga_tile_job_t & running,
                                        fpga_stage_totals_t *   totals) {
    if (g_p1_sched_summary_enabled) {
        g_p1_sched_summary.preload_attempts++;
    }
    const auto poison = [&job, &running](const char * reason) {
        job.input_preload_poisoned = true;
        LOGE("P1_INPUT_PRELOAD_FAIL job=%u running_job=%u bank=%d running_bank=%d reason=%s action=abort_no_deferred_launch",
             job.job_id, running.job_id, job.bank & 1, running.bank & 1, reason ? reason : "?");
        fpga_p1_preload_breadcrumb(true,
                                   "event=fail job=%u running_job=%u bank=%d running_bank=%d reason=%s action=abort",
                                   job.job_id, running.job_id, job.bank & 1, running.bank & 1, reason ? reason : "?");
        return false;
    };

    if (!g_p2_input_preload_enabled || !g_pingpong_scheduler_enabled || !g_spu_q8_scale_stream_supported) {
        return poison("preload_not_admitted");
    }
    if (job.input_preloaded || job.input_preload_poisoned || job.job_id == 0U || running.job_id == 0U ||
        job.job_id == running.job_id || (job.bank & 1) == (running.bank & 1) || running.ip_start_us <= 0) {
        return poison("invalid_running_or_target_ownership");
    }
    // The scheduler's software pointer is not enough proof of live ownership.
    // A fully terminal N can legitimately be observed here after prepare(N+1)
    // but before this P1 gate.  It is too late to overlap safely, yet N still
    // owns the exact descriptor: skip only that terminal signature and let the
    // existing drain/retire path submit N+1 serially.  No other mixed, stale,
    // or error state is safe to reinterpret as a successful preload.
    fpga_p1_preload_register_snapshot_t snapshot = fpga_p1_preload_register_snapshot();
    fpga_p1_preload_state_t preload_state = fpga_classify_p1_preload_state(snapshot, running);
    if (preload_state == FPGA_P1_PRELOAD_STATE_MISMATCH &&
        fpga_p1_preload_snapshot_may_be_transitioning(snapshot, running)) {
        for (uint32_t retry = 0; retry < FPGA_P1_PRELOAD_RESNAPSHOT_LIMIT; ++retry) {
            snapshot = fpga_p1_preload_register_snapshot();
            preload_state = fpga_classify_p1_preload_state(snapshot, running);
            if (preload_state != FPGA_P1_PRELOAD_STATE_MISMATCH ||
                !fpga_p1_preload_snapshot_may_be_transitioning(snapshot, running)) {
                break;
            }
        }
    }
    if (preload_state == FPGA_P1_PRELOAD_STATE_TERMINAL) {
        if (g_p1_sched_summary_enabled) {
            // This is the one non-fatal no-overlap outcome: N has already
            // reached the exact terminal signature, so N+1 must submit
            // serially after the established drain/retire boundary.
            g_p1_sched_summary.preload_terminal_skip++;
        }
        fpga_p1_preload_breadcrumb(
            false,
            "event=P1_INPUT_PRELOAD_SKIP reason=running_job_terminal_before_preload job=%u running_job=%u "
            "bank=%d running_bank=%d status_before=0x%08x status_after=0x%08x bank_stat=0x%08x active_bank=%d "
            "done_bank=%d active_job=0x%08x done_job=0x%08x descriptor_job=0x%08x slot_state=0x%08x "
            "action=drain_then_serial_submit",
            job.job_id, running.job_id, job.bank & 1, running.bank & 1, snapshot.status_before, snapshot.status_after,
            snapshot.bank_stat, (snapshot.bank_stat & BANK_STAT_ACTIVE_BANK) != 0U ? 1 : 0,
            (snapshot.bank_stat & BANK_STAT_DONE_BANK) != 0U ? 1 : 0, snapshot.active_job, snapshot.done_job,
            snapshot.descriptor_job, snapshot.slot_state);
        return true;
    }
    if (preload_state != FPGA_P1_PRELOAD_STATE_ACTIVE) {
        job.input_preload_poisoned = true;
        LOGE(
            "P1_INPUT_PRELOAD_FAIL job=%u running_job=%u bank=%d running_bank=%d reason=register_snapshot_mismatch "
            "expected_slot_state=0x%08x status_before=0x%08x status_after=0x%08x bank_stat=0x%08x active_job=0x%08x "
            "done_job=0x%08x descriptor_job=0x%08x slot_state=0x%08x action=abort_no_deferred_launch",
            job.job_id, running.job_id, job.bank & 1, running.bank & 1,
            fpga_slot_state_word(running.bank, FPGA_SLOT_COMPUTING, FPGA_SLOT_FREE), snapshot.status_before,
            snapshot.status_after, snapshot.bank_stat, snapshot.active_job, snapshot.done_job, snapshot.descriptor_job,
            snapshot.slot_state);
        fpga_p1_preload_breadcrumb(
            true,
            "event=fail job=%u running_job=%u bank=%d running_bank=%d reason=register_snapshot_mismatch "
            "status_before=0x%08x status_after=0x%08x bank_stat=0x%08x active_job=0x%08x done_job=0x%08x "
            "descriptor_job=0x%08x slot_state=0x%08x action=abort",
            job.job_id, running.job_id, job.bank & 1, running.bank & 1, snapshot.status_before, snapshot.status_after,
            snapshot.bank_stat, snapshot.active_job, snapshot.done_job, snapshot.descriptor_job, snapshot.slot_state);
        return false;
    }
    // fpga_dma_copy() repeats this gate per descriptor, but establish it at
    // the P1 boundary as well: no preload may begin while another ZDMA
    // descriptor owns the channel.
    if (!zdma_wait_channel_disabled("P1_INPUT_PRELOAD", "before_act")) {
        return poison("zdma_not_idle");
    }
    if (!fpga_p2_residency_job_still_valid(job)) {
        return poison("p2_residency_post_selection_mismatch");
    }

    const long long preload0 = now_us();
    const long long event0   = p2_event_now_us();
    job.event_preload_begin_us = event0;
    fpga_p1_preload_breadcrumb(false,
                               "event=admit job=%u running_job=%u target_bank=%d active_bank=%d bank_stat=0x%08x "
                               "active_job=%u",
                               job.job_id, running.job_id, job.bank & 1, running.bank & 1, snapshot.bank_stat,
                               snapshot.active_job);
    // Preserve the running bank as read-bank.  Changing it while N executes
    // could redirect its live operand path; only the inactive write-bank bit
    // is changed for N+1's ACT/WEIGHT ZDMA transfers.
    vpu_select_banks(job.bank, running.bank);
    mmio_fence();
    const uint32_t expected_bank = (uint32_t) (job.bank & 1) | ((uint32_t) (running.bank & 1) << 1);
    const uint32_t actual_bank   = vpu_rd32(REG_BANK);
    if ((actual_bank & 0x3U) != expected_bank) {
        return poison("inactive_bank_select_readback_mismatch");
    }

    if (!fpga_dma_write_to_ip(ACT_BASE, job.act_bytes, "P1_ACT")) {
        return poison("act_dma_failed");
    }
    // Keep the running bank selected for reads during the second input DMA.
    vpu_select_banks(job.bank, running.bank);
    mmio_fence();
    if ((vpu_rd32(REG_BANK) & 0x3U) != expected_bank) {
        return poison("inactive_bank_reselect_readback_mismatch");
    }
    if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) job.weight_src_off, LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                       job.weight_bytes, "P1_WEIGHT")) {
        return poison("weight_dma_failed");
    }
    if (g_p1_sched_summary_enabled) {
        // ACT and WEIGHT have both completed into the inactive bank while
        // the admitted running job was still observed active.
        g_p1_sched_summary.preload_admitted_while_active++;
    }
    if (job.p2_residency_hit) {
        fpga_p2_residency_log(false, "P1_PRELOAD_CACHE_SOURCE",
                              "job=%u bank=%d epoch=%llu seal=0x%08x weight_src_off=0x%08x bytes=%zu",
                              job.job_id, job.bank & 1, (unsigned long long) job.p2_residency_epoch,
                              job.p2_residency_seal, job.weight_src_off, job.weight_bytes);
    }

    job.input_preloaded         = true;
    job.preload_key_job_id      = job.job_id;
    job.preload_key_tensor      = job.src0;
    job.preload_key_tensor_id   = job.tensor_id;
    job.preload_key_row0        = job.row0;
    job.preload_key_rows        = job.rows;
    job.preload_key_col         = job.col;
    job.preload_key_k_block0    = job.k_block0;
    job.preload_key_group_blocks = job.group_blocks;
    job.preload_key_act_bytes   = job.act_bytes;
    job.preload_key_weight_bytes = job.weight_bytes;
    job.preload_key_scale_bytes = job.scale_bytes;
    job.preload_key_weight_src_off = job.weight_src_off;
    job.preload_key_weight_layout_version = P2_WEIGHT_RESIDENCY_LAYOUT_V2;
    job.preload_key_p2_residency_slot = job.p2_residency_slot;
    job.preload_key_p2_residency_epoch = job.p2_residency_epoch;
    job.preload_key_p2_residency_seal = job.p2_residency_seal;
    job.preload_key_bank        = job.bank & 1;
    job.input_preload_us        = now_us() - preload0;
    job.input_preload_done_us   = now_us();
    const long long event_done  = p2_event_now_us();
    job.event_preload_done_us = event_done;
    if (totals) {
        totals->input_preload_us += job.input_preload_us;
        totals->input_preload_jobs++;
    }
    p2_event_trace(job, "P1_INPUT_PRELOAD_DONE", event_done, "preload_us", event_done - event0);
    fpga_p1_preload_breadcrumb(false,
                               "event=success job=%u running_job=%u bank=%d act_bytes=%zu weight_bytes=%zu preload_us=%lld",
                               job.job_id, running.job_id, job.bank & 1, job.act_bytes, job.weight_bytes,
                               job.input_preload_us);
    return true;
}

static bool fpga_submit_q8_tile_job(fpga_tile_job_t &     job,
                                    fpga_stage_totals_t * totals,
                                    const char *          tensor_name,
                                    int                   layer_id,
                                    int64_t               k,
                                    int64_t               n,
                                    int64_t               m,
                                    int                   attempt) {
    if (job.p3_split_scale) {
        if (!g_p3_split_scale_active || !g_p3_mode_committed ||
            (vpu_rd32(REG_STREAM_MODE) & 1U) == 0U ||
            (vpu_rd32(REG_SPU_STREAM_P3_STATUS) & SPU_P3_STATUS_MODE_RETAINED) == 0U) {
            LOGE("P3_SUBMIT_REJECT job=%u active=%d committed=%d mode=0x%08x p3_status=0x%08x "
                 "action=abort_before_p3_dma",
                 job.job_id, g_p3_split_scale_active ? 1 : 0, g_p3_mode_committed ? 1 : 0,
                 vpu_rd32(REG_STREAM_MODE), vpu_rd32(REG_SPU_STREAM_P3_STATUS));
            return false;
        }
    }
    job.event_submit_begin_us = p2_event_now_us();
    if (job.input_preload_poisoned || (job.input_preloaded && !fpga_q8_input_preload_key_matches(job))) {
        LOGE("P1_INPUT_PRELOAD_KEY_FAIL job=%u bank=%d preloaded=%d poisoned=%d action=abort_no_deferred_launch",
             job.job_id, job.bank & 1, job.input_preloaded ? 1 : 0, job.input_preload_poisoned ? 1 : 0);
        return false;
    }
    if (!fpga_p2_residency_job_still_valid(job)) {
        LOGE("P2_RESIDENCY_POST_SELECTION_FAIL job=%u tile=%u action=no_start_no_cpu_fallback", job.job_id,
             job.tile_id);
        return false;
    }
    if (job.input_preloaded) {
        const long long reuse_event = p2_event_now_us();
        const long long preload_hold_us =
            job.input_preload_done_us > 0 ? std::max(0LL, now_us() - job.input_preload_done_us) : 0LL;
        // Do not emit ACT_DMA_DONE/WEIGHT_DMA_DONE for a deferred launch: no
        // input DMA occurs here.  This explicit event preserves event-log
        // semantics for owner-side preload-versus-serial comparisons.
        p2_event_trace(job, "P1_INPUT_REUSE", reuse_event, "preload_hold_us", preload_hold_us);
    }
    if (g_p2_init_requested) {
        p2_trace_set_job_context(job);
    }
    vpu_select_banks(job.bank, job.bank);
    vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
    const uint32_t launch_mode =
        VPU_MODE_PACKED_Q8 | (g_spu_q8_scale_stream_supported ? VPU_MODE_P2_TWO_ROW : 0U);
    configure_vpu(job.rows, job.group_beats, launch_mode);
    vpu_write_tile_descriptor(job, FPGA_SLOT_DMA_FILLING, FPGA_SLOT_FREE, 0x00000101U);
    if (g_p2_init_requested && !fpga_commit_p2_dma_filling_descriptor(job)) {
        return false;
    }
    p2_trace_first_tile(job, "CONTROL_DESCRIPTOR", "after");

    long long result_clear0 = 0;
    long long result_clear1 = 0;
    if (g_clear_result_before_run && !g_spu_q8_scale_stream_supported) {
        result_clear0 = now_us();
        ddr_zero_range32(RESULT_BASE, job.result_bytes);
        vpu_select_banks(job.bank, job.bank);
        if (!fpga_dma_write_to_ip(RESULT_BASE, job.result_bytes, "RESULT_CLEAR")) {
            return false;
        }
        result_clear1       = now_us();
        job.result_clear_us = result_clear1 - result_clear0;
    }

    const long long dma_act0       = now_us();
    const long long event_dma_act0 = p2_event_now_us();
    job.event_input_transfer_begin_us = event_dma_act0;
    if (!job.input_preloaded) {
        p2_trace_first_tile(job, "ACT_DMA", "before");
    }
    const bool first_p2_act_detail =
        g_p2_first_act_dma_trace_enabled && !job.input_preloaded && g_p2_init_requested && job.tile_id == 0U &&
        !g_p2_first_act_dma_trace_done;
    if (first_p2_act_detail) {
        fpga_p2_dma_breadcrumb("step=bank_select edge=before expected_bank=%d reg_bank=0x%08x bank_stat=0x%08x",
                               job.bank & 1, vpu_rd32(REG_BANK), vpu_rd32(REG_BANK_STAT));
    }
    vpu_select_banks(job.bank, job.bank);
    if (first_p2_act_detail) {
        // The P2 gate requires ping-pong/descriptor capability, so this
        // readback proves the bank-select write reached the mapped MY_IP
        // aperture before the first ZDMA descriptor is touched. BANK_STAT is
        // diagnostic only because its write/read-bank bit contract is not
        // part of the host ABI.
        const uint32_t expected_bank = (uint32_t) (job.bank & 1) | ((uint32_t) (job.bank & 1) << 1);
        const uint32_t bank_readback = vpu_rd32(REG_BANK);
        const uint32_t bank_stat     = vpu_rd32(REG_BANK_STAT);
        const bool     bank_matches  = (bank_readback & 0x3U) == expected_bank;
        fpga_p2_dma_breadcrumb(
            "step=bank_select edge=after expected_bank=0x%08x reg_bank=0x%08x bank_stat=0x%08x match=%d", expected_bank,
            bank_readback, bank_stat, bank_matches ? 1 : 0);
        if (!bank_matches) {
            g_p2_first_act_dma_trace_done = true;
            LOGE(
                "P2 bank-select readback mismatch job=%u bank=%d expected_reg_bank=0x%08x actual_reg_bank=0x%08x "
                "bank_stat=0x%08x; refusing ACT DMA",
                job.job_id, job.bank & 1, expected_bank, bank_readback, bank_stat);
            return false;
        }
        g_p2_first_act_dma_trace_active = true;
    }
    const bool act_dma_ok = job.input_preloaded || fpga_dma_write_to_ip(ACT_BASE, job.act_bytes, "ACT");
    if (first_p2_act_detail) {
        g_p2_first_act_dma_trace_active = false;
        g_p2_first_act_dma_trace_done   = true;
    }
    if (!act_dma_ok) {
        return false;
    }
    const long long event_dma_act_done = p2_event_now_us();
    if (!job.input_preloaded) {
        p2_trace_first_tile(job, "ACT_DMA", "after");
        p2_event_trace(job, "ACT_DMA_DONE", event_dma_act_done, "act_dma_us", event_dma_act_done - event_dma_act0);
    }
    const long long dma_act1 = now_us();

    const long long dma_weight0       = now_us();
    const long long event_dma_weight0 = p2_event_now_us();
    if (!job.input_preloaded) {
        p2_trace_first_tile(job, "WEIGHT_DMA", "before");
    }
    vpu_select_banks(job.bank, job.bank);
    if (!job.input_preloaded &&
        !fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) job.weight_src_off, LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                       job.weight_bytes, "WEIGHT")) {
        return false;
    }
    const long long event_dma_weight_done = p2_event_now_us();
    if (!job.input_preloaded) {
        p2_trace_first_tile(job, "WEIGHT_DMA", "after");
        p2_event_trace(job, "WEIGHT_DMA_DONE", event_dma_weight_done, "weight_dma_us",
                       event_dma_weight_done - event_dma_weight0);
    }
    const long long dma_weight1 = now_us();

    const long long dma_scale0 = now_us();
    // A completed prior tile must release SPU ownership before this shared
    // parameter window can be overwritten.
    if (g_spu_q8_scale_stream_supported && !wait_spu_stream_quiescent("before SPU_PARAM DMA", false)) {
        return false;
    }
    const long long event_dma_scale0 = p2_event_now_us();
    p2_trace_first_tile(job, job.p3_split_scale ? "P3_PARAM_DMA" : "SPU_PARAM_DMA", "before");
    if (job.p3_split_scale) {
        // P3 deliberately has no PARAM/SCRATCH preload overlap: both exact
        // dense tables are completed, fenced by ZDMA completion/readback, and
        // only then may CTRL_START publish raw VPU traffic to the SPU.
        if (!fpga_dma_write_to_ip(job.p3_param_off, job.p3_weight_scale_bytes, "P3_PARAM_SCALE") ||
            !fpga_dma_write_to_ip(job.p3_scratch_off, job.p3_activation_scale_bytes, "P3_SCRATCH_SCALE")) {
            return false;
        }
        g_p3_param_dma_bytes += (long long) job.p3_weight_scale_bytes;
        g_p3_scratch_dma_bytes += (long long) job.p3_activation_scale_bytes;
    } else if (!fpga_dma_write_to_ip(SPU_PARAM_BASE, job.scale_bytes, "SPU_SCALE")) {
        return false;
    }
    const long long event_dma_scale_done = p2_event_now_us();
    p2_trace_first_tile(job, job.p3_split_scale ? "P3_SCRATCH_DMA" : "SPU_PARAM_DMA", "after");
    p2_event_trace(job, "SPU_PARAM_DMA_DONE", event_dma_scale_done, "spu_param_dma_us",
                   event_dma_scale_done - event_dma_scale0);
    p2_event_trace(job, "WRITE_DONE", event_dma_scale_done, "write_phase_elapsed_us",
                   event_dma_scale_done - job.event_input_transfer_begin_us);
    const long long dma_scale1 = now_us();

    job.dma_act_us              = job.input_preloaded ? 0 : dma_act1 - dma_act0;
    job.dma_weight_us           = job.input_preloaded ? 0 : dma_weight1 - dma_weight0;
    job.dma_scale_us            = dma_scale1 - dma_scale0;
    job.spu_stream_count_before = vpu_rd32(REG_SPU_STREAM_COUNT);
    job.spu_stream_done_before  = vpu_rd32(REG_SPU_STREAM_DONE);
    job.spu_stream_out_before   = vpu_rd32(REG_SPU_STREAM_OUT);
    job.spu_stream_drop_before  = vpu_rd32(REG_SPU_STREAM_DROP);
    job.spu_stream_error_before = vpu_rd32(REG_SPU_STREAM_ERROR);
    job.spu_stream_entry_done_before = vpu_rd32(REG_SPU_STREAM_ENTRY_DONE);
    job.spu_stream_final_write_before = vpu_rd32(REG_SPU_STREAM_FINAL_WRITE);
    job.spu_stream_p3_reject_before = vpu_rd32(REG_SPU_STREAM_P3_REJECT);
    const int bank_index = job.bank & 1;
    const long long h2ip_us = job.input_preload_us + job.dma_act_us + job.dma_weight_us + job.dma_scale_us;
    if (totals) {
        totals->dma_act_us += job.dma_act_us;
        totals->dma_weight_us += job.dma_weight_us;
        totals->dma_scale_us += job.dma_scale_us;
        totals->dma_result_us += g_clear_result_before_run ? job.result_clear_us : 0;
        totals->bank_h2ip_us[bank_index] += h2ip_us;
        totals->bank_jobs[bank_index]++;
        totals->activation_bytes += job.act_bytes;
        totals->weight_bytes += job.weight_bytes;
        totals->scale_bytes += job.scale_bytes;
        totals->result_bytes += job.spu_result_bytes;
        totals->vpu_runs++;
    }
    if (g_pingpong_timing_enabled) {
        fpga_log_line(
            true, "PINGPONG_TIMING", false,
            "phase=host_to_ip graph_seq=%d layer=%d tensor=%s tile=%u job=%u bank=%d bank_role=%s "
            "preloaded=%d h2ip_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f preload_ms=%.3f "
            "act_bytes=%zu weight_bytes=%zu scale_bytes=%zu",
            job.graph_seq, layer_id, tensor_name ? tensor_name : "?", job.tile_id, job.job_id, bank_index,
            p2_bank_label(bank_index), job.input_preloaded ? 1 : 0, (double) h2ip_us / 1000.0,
            (double) job.dma_act_us / 1000.0, (double) job.dma_weight_us / 1000.0,
            (double) job.dma_scale_us / 1000.0, (double) job.input_preload_us / 1000.0, job.act_bytes,
            job.weight_bytes, job.scale_bytes);
    }

    vpu_write_tile_descriptor(job, FPGA_SLOT_COMPUTING, FPGA_SLOT_FREE, 0x00000101U);
    mmio_fence();
    job.ip_start_us = now_us();
    p2_trace_first_tile(job, "VPU_LAUNCH", "before");
    vpu_wr32(REG_CTRL, CTRL_START);
    mmio_fence();
    job.event_launch_us = p2_event_now_us();
    if (totals && job.handoff_prev_output_ready_us > 0 && job.event_launch_us >= job.handoff_prev_output_ready_us) {
        totals->scheduler_output_to_launch_us += job.event_launch_us - job.handoff_prev_output_ready_us;
        totals->scheduler_handoffs++;
    }
    if (totals && job.handoff_prev_retire_us > 0 && job.event_launch_us >= job.handoff_prev_retire_us) {
        totals->scheduler_retire_to_launch_us += job.event_launch_us - job.handoff_prev_retire_us;
    }
    if (totals && job.event_launch_us > 0 &&
        (totals->first_ip_launch_mono_us == 0 || job.event_launch_us < totals->first_ip_launch_mono_us)) {
        totals->first_ip_launch_mono_us = job.event_launch_us;
    }
    if (job.input_preloaded && job.preload_ready_to_launch_us > 0) {
        const long long launch_bubble_us = now_us() - job.preload_ready_to_launch_us;
        if (totals) {
            totals->preload_launch_bubble_us += launch_bubble_us;
        }
        p2_event_trace(job, "P1_DEFERRED_START", job.event_launch_us, "free_to_start_us", launch_bubble_us);
    }
    p2_trace_first_tile(job, "VPU_LAUNCH", "after");
    p2_event_trace(job, "START", job.event_launch_us, "none_us", 0);

    if (g_ip_timing_enabled && should_log_detail_run(job.tile_id)) {
        LOGIP(
            "submit tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u job=%u bank=%d attempt=%d rows=%d col_beats=%d "
            "mode=0x%x result_clear_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f weight_cache=%d",
            tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, job.tile_id,
            job.job_id, job.bank, attempt, job.rows, job.group_beats, launch_mode,
            g_clear_result_before_run ? (double) job.result_clear_us / 1000.0 : 0.0, (double) job.dma_act_us / 1000.0,
            (double) job.dma_weight_us / 1000.0, (double) job.dma_scale_us / 1000.0, job.weight_cache_hit ? 1 : 0);
    }
    return true;
}

static bool fpga_wait_and_drain_q8_tile_job(fpga_tile_job_t &     job,
                                            fpga_stage_totals_t * totals,
                                            const char *          tensor_name,
                                            int                   layer_id,
                                            int64_t               k,
                                            int64_t               n,
                                            int64_t               m,
                                            int                   attempt) {
    uint32_t vpu_status = 0;
    p2_trace_first_tile(job, "VPU_DONE_WAIT", "before");
    if (!wait_vpu_done(&vpu_status)) {
        LOGE(
            "VPU failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld tile=%u job=%u bank=%d attempt=%d status=0x%08x "
            "progress=0x%08x",
            tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, job.tile_id,
            job.job_id, job.bank, attempt, vpu_status, vpu_rd32(REG_PROGRESS));
        return false;
    }
    const long long ip1 = now_us();
    job.event_vpu_done_us = p2_event_now_us();
    p2_trace_first_tile(job, "VPU_DONE_WAIT", "after");
    p2_event_trace(job, "VPU_DONE", job.event_vpu_done_us, "launch_to_vpu_done_observed_us",
                   job.event_vpu_done_us - job.event_launch_us);
    job.vpu_status    = vpu_status;
    job.ip_compute_us = ip1 - job.ip_start_us;
    if (!vpu_verify_done_job(job, vpu_status)) {
        return false;
    }
    p2_trace_first_tile(job, "SPU_FINALITY_WAIT", "before");
    if (!wait_spu_stream_outputs(job)) {
        g_pl_scale_stream_drops += (long long) (vpu_rd32(REG_SPU_STREAM_DROP) - job.spu_stream_drop_before);
        g_pl_scale_stream_errors += (long long) (vpu_rd32(REG_SPU_STREAM_ERROR) - job.spu_stream_error_before);
        return false;
    }
    if (!fpga_p3_verify_retirement(job)) {
        return false;
    }
    job.event_spu_finality_us = p2_event_now_us();
    p2_trace_first_tile(job, "SPU_FINALITY_WAIT", "after");
    p2_event_trace(job, "SPU_FINALITY", job.event_spu_finality_us, "vpu_done_to_spu_finality_us",
                   job.event_spu_finality_us - job.event_vpu_done_us);
    p2_event_trace(job, "OUTPUT_READY", job.event_spu_finality_us, "launch_to_output_ready_observed_us",
                   job.event_spu_finality_us - job.event_launch_us);
    if (totals && job.event_spu_finality_us > totals->last_ip_output_ready_mono_us) {
        totals->last_ip_output_ready_mono_us = job.event_spu_finality_us;
    }

    if (!fpga_write_post_spu_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_DMA_DRAINING, 0x00000101U,
                                        "spu_out_dma_draining", false)) {
        return false;
    }
    vpu_select_banks(job.bank, job.bank);
    const long long dma_result0       = now_us();
    const long long event_dma_result0 = p2_event_now_us();
    p2_trace_first_tile(job, "SPU_OUT_DMA", "before");
    if (!fpga_dma_read_from_ip(SPU_OUT_BASE, job.spu_result_bytes, "SPU_OUT")) {
        return false;
    }
    const long long event_dma_result_done = p2_event_now_us();
    p2_trace_first_tile(job, "SPU_OUT_DMA", "after");
    p2_event_trace(job, "READ_DONE", event_dma_result_done, "spu_out_dma_us",
                   event_dma_result_done - event_dma_result0);
    const long long dma_result1 = now_us();
    job.dma_result_us           = dma_result1 - dma_result0;
    if (!fpga_write_post_spu_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_HOST_CONSUMING, 0x00000101U,
                                        "spu_out_host_consuming", false)) {
        return false;
    }

    const int bank_index = job.bank & 1;
    if (totals) {
        totals->dma_result_us += job.dma_result_us;
        totals->ip_compute_us += job.ip_compute_us;
        totals->bank_compute_us[bank_index] += job.ip_compute_us;
        totals->bank_ip2host_us[bank_index] += job.dma_result_us;
    }
    if (g_pingpong_timing_enabled) {
        fpga_log_line(
            true, "PINGPONG_TIMING", false,
            "phase=ip_compute_and_ip_to_host graph_seq=%d layer=%d tensor=%s tile=%u job=%u bank=%d bank_role=%s "
            "launch_to_vpu_done_ms=%.3f launch_to_output_ready_ms=%.3f ip_compute_ms=%.3f ip_to_host_dma_ms=%.3f "
            "result_bytes=%zu",
            job.graph_seq, layer_id, tensor_name ? tensor_name : "?", job.tile_id, job.job_id, bank_index,
            p2_bank_label(bank_index),
            job.event_vpu_done_us >= job.event_launch_us ?
                (double) (job.event_vpu_done_us - job.event_launch_us) / 1000.0 :
                0.0,
            job.event_spu_finality_us >= job.event_launch_us ?
                (double) (job.event_spu_finality_us - job.event_launch_us) / 1000.0 :
                0.0,
            (double) job.ip_compute_us / 1000.0, (double) job.dma_result_us / 1000.0,
            job.spu_result_bytes);
    }

    if (g_ip_timing_enabled && should_log_detail_run(job.tile_id)) {
        LOGIP(
            "complete tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u job=%u bank=%d attempt=%d rows=%d "
            "col_beats=%d ip_ms=%.3f spu_out_dma_ms=%.3f status=0x%08x progress=0x%08x spu_out=%u",
            tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, job.tile_id,
            job.job_id, job.bank, attempt, job.rows, job.group_beats, (double) job.ip_compute_us / 1000.0,
            (double) job.dma_result_us / 1000.0, vpu_status, vpu_rd32(REG_PROGRESS), vpu_rd32(REG_SPU_STREAM_OUT));
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
    const int bank_index = job.bank & 1;
    if (totals) {
        totals->host_result_us += job.host_result_us;
        totals->bank_host_read_us[bank_index] += job.host_result_us;
    }
    if (g_pingpong_timing_enabled) {
        fpga_log_line(true, "PINGPONG_TIMING", false,
                      "phase=host_result_read graph_seq=%d tensor=%s tile=%u job=%u bank=%d bank_role=%s "
                      "host_read_ms=%.3f values=%u",
                      job.graph_seq, job.tensor_name ? job.tensor_name : "?", job.tile_id, job.job_id, bank_index,
                      p2_bank_label(bank_index), (double) job.host_result_us / 1000.0, job.result_values);
    }
    vpu_write_tile_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_FREE, 0x00000000U);
}

static void fpga_accumulate_q8_tile_job(const fpga_tile_job_t &    job,
                                        const std::vector<float> & act_scales,
                                        int64_t                    nb,
                                        std::vector<float> &       accum,
                                        fpga_stage_totals_t *      totals) {
    const long long accum0    = now_us();
    float *         accum_col = &accum[(size_t) (job.col * job.rows)];
    for (int row = 0; row < job.rows; ++row) {
        for (int gb = 0; gb < job.group_blocks; ++gb) {
            const int64_t ib  = job.k_block0 + gb;
            const int32_t raw = job.partial[(size_t) row * (size_t) job.group_blocks + (size_t) gb];
            accum_col[(size_t) row] += (float) raw * act_scales[(size_t) (job.col * nb + ib)] *
                                       job.weight_scales[(size_t) row * (size_t) job.group_blocks + (size_t) gb];
        }
    }
    if (totals) {
        totals->host_accum_us += now_us() - accum0;
    }
}

static bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t & job,
                                                  std::vector<float> &    accum,
                                                  fpga_stage_totals_t *   totals,
                                                  bool                     p6_q16_verified) {
    const long long result0   = now_us();
    const long long event_host_read_accum0 = p2_event_now_us();
    float *         accum_col = &accum[(size_t) (job.col * job.rows)];
    const bool      p6_trace_job = g_p6_accum_trace.enabled;
    int             p6_chunk_ordinal = -1;
    bool            p6_final_chunk = false;
    if (p6_trace_job) {
        if (!p6_q16_verified || !fpga_p6_accept_chunk(job, &p6_chunk_ordinal, &p6_final_chunk)) {
            LOGE(
                "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d graph=%d job=%u tile=%u bank=%d col=%lld "
                "q16_verified=%d chunks_seen=%d expected_k_chunks=%d records=%zu record_limit=%zu "
                "reason=selector_or_contiguous_k_contract",
                job.tensor_name ? job.tensor_name : "?", job.layer_id, job.graph_seq, job.job_id, job.tile_id, job.bank,
                (long long) job.col, p6_q16_verified ? 1 : 0, g_p6_accum_trace.chunks_seen,
                g_p6_accum_trace.expected_k_chunks, g_p6_accum_trace.records_emitted,
                g_p6_accum_trace.record_limit);
            return false;
        }
    }
    for (int row = 0; row < job.rows; ++row) {
        const int64_t global_row = job.row0 + (int64_t) row;
        const bool p6_emit_row = p6_trace_job && global_row >= g_p6_accum_trace.row_first &&
                                 global_row < g_p6_accum_trace.row_first + (int64_t) g_p6_accum_trace.row_count;
        uint16_t      row_id = 0xffffU;
        const int64_t q16    = ddr_read_spu_q16_row(SPU_OUT_BASE + (uint32_t) row * 16U, &row_id);
        if (row_id != (uint16_t) row) {
            LOGE("SPU_OUT row mismatch job=%u tile=%u bank=%d row=%d got_row=%u q16=%lld", job.job_id, job.tile_id,
                 job.bank, row, (unsigned) row_id, (long long) q16);
            return false;
        }
        uint32_t before_bits = 0U;
        if (p6_emit_row) {
            memcpy(&before_bits, &accum_col[(size_t) row], sizeof(before_bits));
        }
        accum_col[(size_t) row] += (float) q16 * (1.0f / 65536.0f);
        uint32_t after_bits = 0U;
        if (p6_emit_row) {
            memcpy(&after_bits, &accum_col[(size_t) row], sizeof(after_bits));
            LOGI(
                "P6_ARM_GOLDEN_CAPTURE tensor=%s layer=%d graph=%d job=%u tile=%u bank=%d col=%lld tile_row0=%lld "
                "tile_row=%d global_row=%lld q16=%lld k_chunk_ordinal=%d k_chunk_order=%d k_block0=%lld "
                "group_blocks=%d before_bits=0x%08x after_bits=0x%08x final_chunk=%d",
                job.tensor_name ? job.tensor_name : "?", job.layer_id, job.graph_seq, job.job_id, job.tile_id, job.bank,
                (long long) job.col, (long long) job.row0, row, (long long) global_row, (long long) q16,
                p6_chunk_ordinal, p6_chunk_ordinal + 1, (long long) job.k_block0, job.group_blocks, before_bits,
                after_bits, p6_final_chunk ? 1 : 0);
            g_p6_accum_trace.records_emitted++;
        }
    }
    if (p6_trace_job && p6_final_chunk) {
        const size_t expected_records =
            (size_t) g_p6_accum_trace.row_count * (size_t) g_p6_accum_trace.expected_k_chunks;
        if (g_p6_accum_trace.chunks_seen != g_p6_accum_trace.expected_k_chunks ||
            g_p6_accum_trace.records_emitted != expected_records) {
            LOGE(
                "P6_ARM_GOLDEN_CAPTURE_REJECT reason=missing_final_or_record_count chunks_seen=%d expected_k_chunks=%d "
                "records=%zu expected_records=%zu",
                g_p6_accum_trace.chunks_seen, g_p6_accum_trace.expected_k_chunks,
                g_p6_accum_trace.records_emitted, expected_records);
            return false;
        }
        g_p6_accum_trace.capture_complete = true;
        LOGI(
            "P6_ARM_GOLDEN_CAPTURE_COMPLETE tensor=%s layer=%d graph=%d chunks=%d records=%zu route=cpu_shadow "
            "pl_accum_bits=unavailable pl_bit_match=unavailable status=ARM_GOLDEN_ONLY",
            job.tensor_name ? job.tensor_name : "?", job.layer_id, job.graph_seq, g_p6_accum_trace.chunks_seen,
            g_p6_accum_trace.records_emitted);
    }
    const long long result1 = now_us();
    const long long event_host_read_accum_done = p2_event_now_us();
    const long long host_read_us = result1 - result0;
    const int bank_index = job.bank & 1;
    if (totals) {
        totals->host_result_us += host_read_us;
        totals->bank_host_read_us[bank_index] += host_read_us;
    }
    if (g_pingpong_timing_enabled) {
        fpga_log_line(true, "PINGPONG_TIMING", false,
                      "phase=host_result_read graph_seq=%d tensor=%s tile=%u job=%u bank=%d bank_role=%s "
                      "host_read_ms=%.3f rows=%d format=q16_16_scaled",
                      job.graph_seq, job.tensor_name ? job.tensor_name : "?", job.tile_id, job.job_id, bank_index,
                      p2_bank_label(bank_index), (double) host_read_us / 1000.0, job.rows);
    }
    p2_event_trace(job, "HOST_READ_ACCUM_DONE", event_host_read_accum_done, "host_read_accum_us",
                   event_host_read_accum_done - event_host_read_accum0);
    p2_trace_first_tile(job, "DESCRIPTOR_FREE", "before");
    if (!fpga_write_post_spu_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_FREE, 0x00000000U, "final_free_free", true)) {
        return false;
    }
    const long long retire_us = p2_event_now_us();
    job.event_retire_us = retire_us;
    p2_trace_first_tile(job, "DESCRIPTOR_FREE", "after");
    p2_event_trace(job, "DESCRIPTOR_RETIRED", retire_us, "submit_to_retire_us",
                   retire_us - job.event_submit_begin_us);
    p2_event_trace(job, "TILE_FINISH", retire_us, "submit_to_retire_us", retire_us - job.event_submit_begin_us);
    return true;
}

// The P2/P3 contract is a tile-level transport/numerical qualification, not a
// complete matrix result.  Once the cumulative limit is reached, retire the
// descriptor with a DSB and exact FREE/FREE readback before returning
// CPU-shadow control to GGML.  The global boundary then prevents admission of
// every later hardware tile.
static bool fpga_p2_complete_tile_contract_boundary(const fpga_tile_job_t & job,
                                                    const char *            tensor_name,
                                                    int                     layer_id);

#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
#endif
static bool fpga_spu_q16_contribution(int32_t   raw,
                                      uint16_t  act_scale,
                                      uint16_t  weight_scale,
                                      int64_t * contribution_q16);
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

static bool fpga_pl_scale_contract_verify_q16_tile(const fpga_tile_job_t & job,
                                                   const void *            weight_data_base,
                                                   const char *            tensor_name,
                                                   int                     layer_id);

static bool fpga_pl_scale_contract_check_output_values(const struct ggml_tensor *        src0,
                                                       const struct ggml_tensor *        dst,
                                                       const std::vector<block_q8_0_t> & act_blocks_all,
                                                       const void *                      weight_data_base,
                                                       const char *                      tensor_name,
                                                       int                               layer_id);

static bool fpga_dma_run_q8_group(const struct ggml_tensor *        src0,
                                  const void *                      weight_data_base,
                                  const block_q8_0_t *              act_group,
                                  int64_t                           row0,
                                  int                               rows,
                                  int64_t                           k_block0,
                                  int                               group_blocks,
                                  uint32_t                          weight_tile_index,
                                  const fpga_weight_cache_entry_t * weight_cache,
                                  std::vector<int32_t> &            partial,
                                  std::vector<float> &              weight_scales,
                                  float *                           accum_col,
                                  const float *                     act_scales_group,
                                  bool *                            accumulated_on_unpack,
                                  fpga_stage_totals_t *             totals,
                                  uint32_t                          tile_id,
                                  const char *                      tensor_name,
                                  int                               layer_id,
                                  int64_t                           k,
                                  int64_t                           n,
                                  int64_t                           m,
                                  bool                              contract_check_active) {
    if (accumulated_on_unpack) {
        *accumulated_on_unpack = false;
    }
    if (rows <= 0 || rows > g_vpu_max_rows || group_blocks <= 0) {
        LOGE("unsupported DMA-to-IP tiling case: rows=%d max_rows=%d group_blocks=%d", rows, g_vpu_max_rows,
             group_blocks);
        return false;
    }

    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    if (group_beats > g_vpu_max_beats) {
        LOGE("unsupported DMA-to-IP tiling case: group_beats=%d max_beats=%d", group_beats, g_vpu_max_beats);
        return false;
    }

    const uint32_t result_values = (uint32_t) rows * (uint32_t) group_blocks;
    const uint32_t result_words =
        (result_values + (uint32_t) VPU_RESULT_PACK_LANES - 1U) / (uint32_t) VPU_RESULT_PACK_LANES;
    if (result_words > (uint32_t) g_packed_q8_result_words) {
        LOGE("unsupported DMA-to-IP tiling case: result_words=%u cap=%d", result_words, g_packed_q8_result_words);
        return false;
    }

    const size_t act_bytes    = (size_t) group_beats * 16U;
    const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
    const size_t result_bytes = (size_t) result_words * 16U;
    if (!range_fits(ACT_BASE, act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(WEIGHT_BASE, weight_bytes, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(RESULT_BASE, result_bytes, RESULT_BASE, RESULT_END) || !ddr_range_fits(ACT_BASE, act_bytes) ||
        !ddr_range_fits(WEIGHT_BASE, weight_bytes) || !ddr_range_fits(RESULT_BASE, result_bytes)) {
        LOGE("unsupported DMA-to-IP tiling case: window overflow act=%zu weight=%zu result=%zu ddr_size=0x%zx",
             act_bytes, weight_bytes, result_bytes, g_ddr_map_size);
        return false;
    }

    const long long             prep0               = now_us();
    uint32_t                    weight_src_off      = WEIGHT_BASE;
    bool                        weight_cache_hit    = false;
    const float *               weight_scale_values = nullptr;
    std::vector<block_q8_0_t> & weight_snapshot     = g_scratch.weight_tile_snapshot;
    weight_snapshot.resize((size_t) rows * (size_t) group_blocks);
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb] =
                *weight_block_from_base(src0, weight_data_base, row0 + row, k_block0 + gb);
        }
    }

    if (weight_cache && weight_tile_index < weight_cache->tiles.size()) {
        const fpga_weight_tile_cache_t & tile = weight_cache->tiles[weight_tile_index];
        if (tile.row0 == row0 && tile.rows == rows && tile.k_block0 == k_block0 && tile.group_blocks == group_blocks &&
            tile.group_beats == group_beats && tile.bytes == weight_bytes) {
            weight_src_off      = tile.ddr_off;
            weight_cache_hit    = true;
            // The cache owns immutable scale metadata in the same row-major
            // layout used by accumulation.  Avoid copying it into a temporary
            // vector for every decode tile.
            weight_scale_values = weight_cache->scales.data() + tile.scale_off;
            if (contract_check_active || !g_fuse_raw_result_accum) {
                // Contract mode deliberately preserves the pre-fusion path so
                // it can validate raw values before the caller accumulates.
                weight_scales.assign(weight_scale_values, weight_scale_values + (size_t) rows * (size_t) group_blocks);
            }
        }
    }

    if (!weight_cache_hit) {
        weight_scales.resize((size_t) rows * (size_t) group_blocks);
        for (int row = 0; row < rows; ++row) {
            for (int gb = 0; gb < group_blocks; ++gb) {
                const block_q8_0_t * wb = &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
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
    if (!fpga_stage_q8_group_with_contract_guard(weight_snapshot.data(), act_group, rows, group_blocks,
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

    const bool raw_contract_active = contract_check_active || g_contract_raw_repair_enabled;
    const int  attempt_count       = raw_contract_active ? (1 + g_contract_raw_retry_limit) : 1;
    for (int attempt = 0; attempt < attempt_count; ++attempt) {
        vpu_select_banks(0, 0);
        vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
        configure_vpu(rows, group_beats, VPU_MODE_PACKED_Q8 | VPU_MODE_P2_TWO_ROW);

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
        if (staging_guard_active &&
            !fpga_contract_verify_staged_q8_group(weight_snapshot.data(), act_group, rows, group_blocks, weight_src_off,
                                                  tensor_name, layer_id, tile_id, "after_act_dma")) {
            if (!fpga_contract_restage_after_act_dma(weight_snapshot.data(), act_group, rows, group_blocks,
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
        if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) weight_src_off, LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                           weight_bytes, "WEIGHT")) {
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
            LOGE(
                "VPU failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld tile=%u attempt=%d status=0x%08x "
                "progress=0x%08x",
                tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, tile_id,
                attempt, vpu_status, vpu_rd32(REG_PROGRESS));
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
            totals->dma_result_us +=
                (dma_result1 - dma_result0) + (g_clear_result_before_run ? (result_clear1 - result_clear0) : 0);
            totals->ip_compute_us += ip1 - ip0;
            totals->activation_bytes += act_bytes;
            totals->weight_bytes += weight_bytes;
            totals->result_bytes += result_bytes;
            totals->vpu_runs++;
        }

        if (g_ip_timing_enabled && should_log_detail_run(tile_id)) {
            LOGIP(
                "run tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u attempt=%d rows=%d col_beats=%d mode=0x%x "
                "result_clear_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f ip_ms=%.3f result_dma_ms=%.3f status=0x%08x "
                "progress=0x%08x weight_cache=%d",
                tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, tile_id,
                attempt, rows, group_beats, VPU_MODE_PACKED_Q8,
                g_clear_result_before_run ? (double) (result_clear1 - result_clear0) / 1000.0 : 0.0,
                (double) (dma_act1 - dma_act0) / 1000.0, (double) (dma_weight1 - dma_weight0) / 1000.0,
                (double) (ip1 - ip0) / 1000.0, (double) (dma_result1 - dma_result0) / 1000.0, vpu_status,
                vpu_rd32(REG_PROGRESS), weight_cache_hit ? 1 : 0);
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
            const long long result_unpack0               = now_us();
            uint32_t        row                          = 0;
            uint32_t        gb                           = 0;
            float           row_accum                    = rows > 0 ? accum_col[0] : 0.0f;
            int32_t         lanes[VPU_RESULT_PACK_LANES] = {};
            for (uint32_t word = 0; word < result_words; ++word) {
                read_result_i32x4_from_ddr(word, lanes);
                const uint32_t word_base  = word * (uint32_t) VPU_RESULT_PACK_LANES;
                const uint32_t lane_count = std::min((uint32_t) VPU_RESULT_PACK_LANES, result_values - word_base);
                for (uint32_t lane = 0; lane < lane_count; ++lane) {
                    if (row >= (uint32_t) rows || gb >= (uint32_t) group_blocks) {
                        LOGE(
                            "fused raw-result cursor overflow tensor=%s tile=%u word=%u row=%u group=%u rows=%d "
                            "group_blocks=%d result_values=%u",
                            tensor_name ? tensor_name : "?", tile_id, word, row, gb, rows, group_blocks, result_values);
                        return false;
                    }
                    const uint32_t scale_index = row * (uint32_t) group_blocks + gb;
                    row_accum += (float) lanes[lane] * act_scales_group[gb] * weight_scale_values[scale_index];
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
                LOGE(
                    "fused raw-result cursor incomplete tensor=%s tile=%u rows_done=%u expected_rows=%d "
                    "remaining_group=%u result_values=%u",
                    tensor_name ? tensor_name : "?", tile_id, row, rows, gb, result_values);
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

        if (!fpga_contract_verify_weight_source_snapshot(src0, weight_snapshot.data(), row0, rows, k_block0,
                                                         group_blocks, tensor_name, layer_id, tile_id)) {
            return false;
        }

        const bool                   final_attempt       = (attempt + 1) >= attempt_count;
        const bool                   repair_this_attempt = final_attempt && g_contract_raw_repair_enabled;
        fpga_raw_mismatch_location_t first_mismatch      = {};
        const long long              raw_mismatches      = fpga_contract_count_raw_mismatches(
            weight_snapshot.data(), act_group, row0, rows, k_block0, group_blocks, weight_src_off, partial, tensor_name,
            layer_id, tile_id, attempt, true, repair_this_attempt, &first_mismatch);
        if (raw_mismatches == 0) {
            if (attempt > 0) {
                LOGI("CONTRACT_RAW_RETRY_PASS tensor=%s layer=%d tile=%u attempts=%d", tensor_name ? tensor_name : "?",
                     layer_id, tile_id, attempt + 1);
            }
            return true;
        }

        if (!final_attempt) {
            LOGE("CONTRACT_RAW_RETRY tensor=%s layer=%d tile=%u attempt=%d mismatches=%lld next_attempt=%d",
                 tensor_name ? tensor_name : "?", layer_id, tile_id, attempt, raw_mismatches, attempt + 1);
            continue;
        }

        g_contract_raw_mismatches += raw_mismatches;
        if (repair_this_attempt) {
            g_contract_raw_repairs += raw_mismatches;
        }
        if (g_contract_forensic_replay && g_contract_check_abort && !repair_this_attempt && first_mismatch.valid) {
            fpga_contract_forensic_replay(weight_snapshot.data(), act_group, rows, group_blocks, weight_src_off,
                                          weight_cache_hit, tile_id, tensor_name, layer_id, first_mismatch);
        }
        LOGE("CONTRACT_RAW_SUMMARY tensor=%s layer=%d tile=%u attempt=%d mismatches=%lld action=%s",
             tensor_name ? tensor_name : "?", layer_id, tile_id, attempt, raw_mismatches,
             repair_this_attempt ? "repair_continue" : (g_contract_check_abort ? "abort" : "log_only"));
        if (repair_this_attempt) {
            return true;
        }
        return !g_contract_check_abort;
    }
    return true;
}

static long long estimate_vpu_runs(int64_t k, int64_t n, int64_t m) {
    const int64_t nb         = k / VPU_QK8_0;
    long long     runs_per_m = 0;
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

static bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(const struct ggml_tensor *        src0,
                                                    const struct ggml_tensor *        dst,
                                                    const std::vector<block_q8_0_t> & act_blocks_all,
                                                    const std::vector<float> &        act_scales,
                                                    const fpga_weight_cache_entry_t * weight_cache,
                                                    fpga_stage_totals_t *             totals,
                                                    const char *                      tensor_name,
                                                    int                               layer_id,
                                                    int64_t                           k,
                                                    int64_t                           n,
                                                    int64_t                           m,
                                                    int64_t                           nb,
                                                    std::vector<float> &              accum) {
    (void) act_scales;
    (void) nb;
    uint32_t        tile_id           = 0;
    uint32_t        weight_tile_index = 0;
    fpga_tile_job_t slots[2]          = {};

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        accum.assign((size_t) (m * rows), 0.0f);

        fpga_tile_job_t * running = nullptr;
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int remaining_blocks = (int) (nb - ib0);
            const int group_blocks     = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats      = group_blocks * VPU_BLOCK_BEATS;

            for (int64_t col = 0; col < m; ++col) {
                const int            bank      = (int) (tile_id & 1U);
                fpga_tile_job_t &    prepared  = slots[bank];
                const block_q8_0_t * act_group = &act_blocks_all[(size_t) (col * nb + ib0)];

                if (should_log_detail_run(tile_id)) {
                    LOGSTAGE(
                        "tile tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d group_beats=%d "
                        "tile_id=%u bank=%d pipeline=pingpong partial_accum=1 transfer=zdma_ddr_to_ip",
                        tensor_name ? tensor_name : "?", layer_id, (long long) row0, rows, (long long) ib0,
                        group_blocks, group_beats, tile_id, bank);
                }

                if (!fpga_prepare_q8_tile_job(prepared, src0, src0->data, act_group, row0, rows, ib0, group_blocks, col,
                                              weight_tile_index, weight_cache, tile_id, bank, totals)) {
                    return false;
                }

                if (running) {
                    if (g_p1_sched_summary_enabled) {
                        // A pair is a scheduler handoff opportunity, whether
                        // it overlaps or must submit serially after drain.
                        g_p1_sched_summary.pingpong_pairs++;
                    }
                    // P1 may overlap only the inactive-bank ACT/WEIGHT DMA.
                    // It must not touch descriptor/config/CTRL/SPU windows or
                    // the running job's read bank; all of those remain
                    // deferred until N is fully retired below.
                    if (g_p2_input_preload_enabled && !fpga_preload_q8_tile_inputs(prepared, *running, totals)) {
                        return false;
                    }
                    if (!fpga_wait_and_drain_q8_tile_job(*running, totals, tensor_name, layer_id, k, n, m, 0)) {
                        return false;
                    }
                    if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals, false)) {
                        return false;
                    }
                    if (totals && running->event_launch_us > 0 && running->event_spu_finality_us > 0 &&
                        prepared.event_prep_begin_us > 0 && prepared.event_prep_done_us > 0) {
                        const long long overlap_begin = std::max(prepared.event_prep_begin_us, running->event_launch_us);
                        const long long overlap_end = std::min(prepared.event_prep_done_us, running->event_spu_finality_us);
                        if (overlap_end > overlap_begin) {
                            totals->scheduler_prepare_overlap_us += overlap_end - overlap_begin;
                        }
                        if (prepared.event_prep_done_us > running->event_spu_finality_us) {
                            totals->scheduler_prepare_late_us += prepared.event_prep_done_us - running->event_spu_finality_us;
                            totals->scheduler_prepare_late_jobs++;
                        } else {
                            totals->scheduler_prepare_headroom_us += running->event_spu_finality_us - prepared.event_prep_done_us;
                        }
                    }
                    if (totals && prepared.event_preload_begin_us > 0 && prepared.event_preload_done_us > 0 &&
                        running->event_launch_us > 0 && running->event_spu_finality_us > 0) {
                        const long long overlap_begin = std::max(prepared.event_preload_begin_us, running->event_launch_us);
                        const long long overlap_end = std::min(prepared.event_preload_done_us, running->event_spu_finality_us);
                        if (overlap_end > overlap_begin) {
                            totals->scheduler_preload_overlap_us += overlap_end - overlap_begin;
                            totals->scheduler_preload_overlap_jobs++;
                        }
                    }
                    prepared.handoff_prev_output_ready_us = running->event_spu_finality_us;
                    prepared.handoff_prev_retire_us = running->event_retire_us;
                    // fpga_accumulate_pl_scaled_q8_tile_job performs the
                    // final FREE/FREE descriptor readback.  Only after that
                    // boundary may this deferred descriptor/config/SPU_PARAM
                    // launch run; preloaded ACT/WEIGHT are reused verbatim.
                    if (prepared.input_preloaded) {
                        prepared.preload_ready_to_launch_us = now_us();
                    } else if (g_p1_sched_summary_enabled) {
                        g_p1_sched_summary.serial_submit_after_no_preload++;
                    }
                    if (!fpga_submit_q8_tile_job(prepared, totals, tensor_name, layer_id, k, n, m, 0)) {
                        return false;
                    }
                    running = &prepared;
                } else {
                    if (!fpga_submit_q8_tile_job(prepared, totals, tensor_name, layer_id, k, n, m, 0)) {
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
            if (!fpga_wait_and_drain_q8_tile_job(*running, totals, tensor_name, layer_id, k, n, m, 0)) {
                return false;
            }
            if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals, false)) {
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

static bool fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_single_bank(const struct ggml_tensor *        src0,
                                                               const struct ggml_tensor *        dst,
                                                               const std::vector<block_q8_0_t> & act_blocks_all,
                                                               const fpga_weight_cache_entry_t * weight_cache,
                                                               const void *                      weight_data_base,
                                                               fpga_stage_totals_t *             totals,
                                                               const char *                      tensor_name,
                                                               int                               layer_id,
                                                               int64_t                           k,
                                                               int64_t                           n,
                                                               int64_t                           m,
                                                               int64_t                           nb,
                                                               bool                 pl_scale_contract_active,
                                                               std::vector<float> & accum) {
    if (g_p3_split_scale_active) {
        // This is the irrevocable P3 boundary.  It precedes both dense-table
        // DDR writes and ZDMA descriptors, so an incompatible mode can still
        // stop cleanly without a speculative P3 payload transfer.
        if (!fpga_set_split_scale_mode(1U, "P3 first tile before staging")) {
            return false;
        }
        g_p3_mode_committed = true;
    } else if (!fpga_set_split_scale_mode(0U, "P2 single-bank route")) {
        return false;
    }
    uint32_t tile_id           = 0;
    uint32_t weight_tile_index = 0;
    if (pl_scale_contract_active) {
        // v58 validates whole P2 tiles only.  A partial tile sequence is not
        // a matrix value contract, so never allocate or later compare a
        // fabricated partially-filled matrix result.
        g_scratch.contract_actual.clear();
    } else {
        g_scratch.contract_actual.clear();
    }

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        // The P6 matrix remains CPU-shadowed.  It therefore may omit
        // nonselected row tiles entirely instead of staging them just to
        // reach the requested tile selector.
        if (g_p6_accum_trace.enabled && row0 != g_p6_accum_trace.tile_row0) {
            continue;
        }
        accum.assign((size_t) m * (size_t) rows, 0.0f);
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, (int) (nb - ib0));
            for (int64_t col = 0; col < m; ++col) {
                fpga_tile_job_t            job       = {};
                const block_q8_0_t * const act_group = &act_blocks_all[(size_t) (col * nb + ib0)];
                if (should_log_detail_run(tile_id)) {
                    LOGSTAGE(
                        "P2_SINGLE_BANK_TILE tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d "
                        "tile=%u job_policy=one_outstanding bank=0 route=vpu_to_spu",
                        tensor_name ? tensor_name : "?", layer_id, (long long) row0, rows, (long long) ib0,
                        group_blocks, tile_id);
                }
                if (!fpga_prepare_q8_tile_job(job, src0, weight_data_base, act_group, row0, rows, ib0, group_blocks,
                                              col, weight_tile_index, weight_cache, tile_id, 0, totals) ||
                    !fpga_submit_q8_tile_job(job, totals, tensor_name, layer_id, k, n, m, 0) ||
                    !fpga_wait_and_drain_q8_tile_job(job, totals, tensor_name, layer_id, k, n, m, 0)) {
                    return false;
                }
                g_pl_scale_jobs++;
                g_pl_scale_banks++;
                if (pl_scale_contract_active &&
                    !fpga_pl_scale_contract_verify_q16_tile(job, weight_data_base, tensor_name, layer_id)) {
                    return false;
                }
                if (pl_scale_contract_active) {
                    ++g_p2_tile_q16_checks;
                }
                if (!fpga_accumulate_pl_scaled_q8_tile_job(job, accum, totals, pl_scale_contract_active)) {
                    return false;
                }
                if (pl_scale_contract_active &&
                    fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit)) {
                    if (g_p6_accum_trace.enabled && !g_p6_accum_trace.capture_complete) {
                        LOGE(
                            "P6_ARM_GOLDEN_CAPTURE_REJECT reason=terminal_p2_boundary_without_complete_capture "
                            "chunks_seen=%d expected_k_chunks=%d records=%zu",
                            g_p6_accum_trace.chunks_seen, g_p6_accum_trace.expected_k_chunks,
                            g_p6_accum_trace.records_emitted);
                        return false;
                    }
                    if (!fpga_p2_complete_tile_contract_boundary(job, tensor_name, layer_id)) {
                        return false;
                    }
                    // This is a successful *tile* contract.  The current
                    // MUL_MAT remains CPU-shadowed and all later eligible
                    // GEMVs are routed natively by fpga_try_matmul_extended.
                    return true;
                }
                tile_id++;
            }
            ib0 += group_blocks;
            weight_tile_index++;
        }
        const long long store0 = now_us();
        for (int64_t col = 0; col < m; ++col) {
            const float * const accum_col = &accum[(size_t) col * (size_t) rows];
            for (int row = 0; row < rows; ++row) {
                if (!pl_scale_contract_active) {
                    store_dst_value(dst, row0 + row, col, accum_col[(size_t) row]);
                }
            }
        }
        if (totals) {
            totals->host_accum_us += now_us() - store0;
        }
    }
    if (g_p6_accum_trace.enabled && !g_p6_accum_trace.capture_complete) {
        LOGE(
            "P6_ARM_GOLDEN_CAPTURE_REJECT reason=missing_selected_tile_or_final_chunk tensor=%s layer=%d "
            "tile_row0=%lld chunks_seen=%d expected_k_chunks=%d records=%zu",
            tensor_name ? tensor_name : "?", layer_id, (long long) g_p6_accum_trace.tile_row0,
            g_p6_accum_trace.chunks_seen, g_p6_accum_trace.expected_k_chunks, g_p6_accum_trace.records_emitted);
        return false;
    }
    if (pl_scale_contract_active) {
        if (g_p2_tile_contract_boundary_reached ||
            fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit)) {
            LOGE("P3_TILE_LIMIT_INTERNAL_FAIL tensor=%s layer=%d matrix_tiles=%u cumulative_q16=%lld tile_limit=%d "
                 "boundary=%d reason=matrix_returned_without_exact_boundary",
                 tensor_name ? tensor_name : "?", layer_id, tile_id, g_p2_tile_q16_checks, g_p2_tile_limit,
                 g_p2_tile_contract_boundary_reached ? 1 : 0);
            return false;
        }
        LOGINIT(
            "P3_TILE_MATRIX_CONTINUE tensor=%s layer=%d matrix_tiles=%u cumulative_q16=%lld tile_limit=%d "
            "remaining=%lld matrix_value_contract=not_attempted "
            "action=cpu_shadow_current_matmul_then_continue_qualification",
            tensor_name ? tensor_name : "?", layer_id, tile_id, g_p2_tile_q16_checks, g_p2_tile_limit,
            (long long) g_p2_tile_limit - g_p2_tile_q16_checks);
        return true;
    }
    return true;
}

static bool fpga_hw_q8_0_matmul_dma_to_ip(const struct ggml_tensor * src0,
                                          const struct ggml_tensor * src1,
                                          const struct ggml_tensor * dst,
                                          fpga_stage_totals_t *      totals,
                                          const char *               tensor_name,
                                          int                        layer_id) {
    // This guard is deliberately before activation/weight staging and before
    // any mode-0 write.  A requested P3 route is never allowed to silently
    // degrade into P2 after a lifecycle/re-entry fault.
    if (g_p3_split_scale_requested) {
        LOGINIT(
            "P3_DISPATCH_STATE requested=1 active=%d admitted=%d init_complete=%d p2_requested=%d mode_committed=%d "
            "action=validate_before_p2_p3_staging_or_mode_write",
            g_p3_split_scale_active ? 1 : 0, g_p3_split_scale_admitted ? 1 : 0, g_fpga_init_complete ? 1 : 0,
            g_p2_init_requested ? 1 : 0, g_p3_mode_committed ? 1 : 0);
        if (!g_fpga_init_complete || !g_p2_init_requested || !g_p3_split_scale_active ||
            !g_p3_split_scale_admitted) {
            LOGE(
                "P3_DISPATCH_REJECT requested=1 active=%d admitted=%d init_complete=%d p2_requested=%d "
                "mode_committed=%d action=abort_before_p2_p3_staging_or_mode_write_no_fallback",
                g_p3_split_scale_active ? 1 : 0, g_p3_split_scale_admitted ? 1 : 0, g_fpga_init_complete ? 1 : 0,
                g_p2_init_requested ? 1 : 0, g_p3_mode_committed ? 1 : 0);
            return false;
        }
    }
    const int64_t k  = src0->ne[0];
    const int64_t n  = src0->ne[1];
    const int64_t m  = src1->ne[1];
    const int64_t nb = k / VPU_QK8_0;

    std::vector<block_q8_0_t> & act_blocks_all = g_scratch.act_blocks_all;
    std::vector<float> &        act_scales     = g_scratch.act_scales;
    std::vector<float> &        weight_scales  = g_scratch.weight_scales;
    std::vector<int32_t> &      partial        = g_scratch.partial;
    std::vector<float> &        accum          = g_scratch.accum;

    const bool contract_check_active =
        (g_contract_check_limit > 0) && (g_contract_checks_done < (long long) g_contract_check_limit);
    const bool pl_scale_contract_active =
        (g_pl_scale_contract_check_limit > 0) && !g_p2_tile_contract_boundary_reached &&
        !fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit);
    const bool cpu_shadow_dst           = g_contract_cpu_shadow_dst;
    g_contract_source_validation_failed = false;
    const size_t weight_tensor_bytes    = ggml_nbytes(src0);
    const void * weight_data_base       = src0->data;
    if (contract_check_active || pl_scale_contract_active) {
        // Capture the complete immutable tensor before the first tile.  A
        // per-tile snapshot alone can miss corruption caused by an earlier
        // launch when the damaged row is not consumed until a later tile.
        g_scratch.weight_tensor_snapshot.resize(weight_tensor_bytes);
        memcpy(g_scratch.weight_tensor_snapshot.data(), src0->data, weight_tensor_bytes);
        weight_data_base = g_scratch.weight_tensor_snapshot.data();
        if (!fpga_contract_validate_weight_scales(src0, weight_data_base, tensor_name, layer_id)) {
            g_contract_source_validation_failed = true;
            return false;
        }
    } else {
        g_scratch.weight_tensor_snapshot.clear();
    }
    // PL scale uses the canonical single-bank VPU->SPU ABI whenever it is
    // explicitly enabled.  Ping-pong remains a separate scheduler opt-in;
    // it must not decide whether raw RESULT or SPU_OUT is consumed.
    const bool use_pl_scale_path = g_spu_q8_scale_stream_supported && !contract_check_active;

    const long long quant0 = now_us();
    if (!ensure_quantized_activation_matrix(src1, m, k, act_blocks_all, act_scales, !use_pl_scale_path, totals,
                                            tensor_name, layer_id)) {
        return false;
    }
    if (totals) {
        totals->prep_us += now_us() - quant0;
    }

    const long long             weight_cache0 = now_us();
    fpga_weight_cache_entry_t * weight_cache  = get_weight_cache_entry(src0, totals);
    if (totals) {
        totals->prep_us += now_us() - weight_cache0;
    }

    if (use_pl_scale_path) {
        if (!g_p3_split_scale_active && !fpga_set_split_scale_mode(0U, "P2 dispatch route")) {
            return false;
        }
        if (g_pingpong_scheduler_enabled && !pl_scale_contract_active) {
            return fpga_hw_q8_0_matmul_dma_to_ip_pipelined(src0, dst, act_blocks_all, act_scales, weight_cache, totals,
                                                           tensor_name, layer_id, k, n, m, nb, accum);
        }
        const bool pl_ok = fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_single_bank(
            src0, dst, act_blocks_all, weight_cache, weight_data_base, totals, tensor_name, layer_id, k, n, m, nb,
            pl_scale_contract_active, accum);
        if (pl_ok && pl_scale_contract_active &&
            !fpga_p2_cumulative_tile_state_consistent(g_p2_tile_q16_checks, g_p2_tile_limit,
                                                       g_p2_tile_contract_boundary_reached)) {
            LOGE("P3_TILE_LIMIT_INTERNAL_FAIL cumulative_q16=%lld tile_limit=%d boundary=%d expected_boundary=%d "
                 "reason=cumulative_boundary_inconsistent",
                 g_p2_tile_q16_checks, g_p2_tile_limit, g_p2_tile_contract_boundary_reached ? 1 : 0,
                 fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit) ? 1 : 0);
            return false;
        }
        return pl_ok;
    }

    uint32_t tile_id           = 0;
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
            const int group_blocks     = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats      = group_blocks * VPU_BLOCK_BEATS;

            if (should_log_detail_run(tile_id)) {
                LOGSTAGE(
                    "tile tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d group_beats=%d tile_id=%u "
                    "partial_accum=1 transfer=zdma_ddr_to_ip",
                    tensor_name ? tensor_name : "?", layer_id, (long long) row0, rows, (long long) ib0, group_blocks,
                    group_beats, tile_id);
            }

            for (int64_t col = 0; col < m; ++col) {
                const block_q8_0_t * act_group             = &act_blocks_all[(size_t) (col * nb + ib0)];
                float * const        accum_col             = &accum[(size_t) (col * rows)];
                const float * const  act_scales_group      = &act_scales[(size_t) (col * nb + ib0)];
                bool                 accumulated_on_unpack = false;
                if (!fpga_dma_run_q8_group(src0, weight_data_base, act_group, row0, rows, ib0, group_blocks,
                                           weight_tile_index, weight_cache, partial, weight_scales, accum_col,
                                           act_scales_group, &accumulated_on_unpack, totals, tile_id++, tensor_name,
                                           layer_id, k, n, m, contract_check_active)) {
                    return false;
                }

                if (!accumulated_on_unpack) {
                    const long long accum0 = now_us();
                    for (int row = 0; row < rows; ++row) {
                        for (int gb = 0; gb < group_blocks; ++gb) {
                            const int64_t ib  = ib0 + gb;
                            const int32_t raw = partial[(size_t) row * (size_t) group_blocks + (size_t) gb];
                            accum_col[(size_t) row] +=
                                (float) raw * act_scales[(size_t) (col * nb + ib)] *
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
                    g_scratch.contract_actual[(size_t) col * (size_t) n + (size_t) (row0 + row)] =
                        accum_col[(size_t) row];
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
            const uint8_t * const live      = (const uint8_t *) src0->data;
            const uint8_t * const baseline  = (const uint8_t *) weight_data_base;
            size_t                first_bad = 0U;
            while (first_bad < weight_tensor_bytes && live[first_bad] == baseline[first_bad]) {
                first_bad++;
            }
            LOGE(
                "CONTRACT_WEIGHT_TENSOR_MUTATION tensor=%s layer=%d byte=%zu baseline=%u live=%u bytes=%zu; immutable "
                "GGUF tensor changed during FPGA matmul",
                tensor_name ? tensor_name : "?", layer_id, first_bad,
                first_bad < weight_tensor_bytes ? baseline[first_bad] : 0U,
                first_bad < weight_tensor_bytes ? live[first_bad] : 0U, weight_tensor_bytes);
            return false;
        }
        g_contract_checks_done++;
        if (!fpga_contract_check_output_values(src0, dst, act_blocks_all, weight_data_base, tensor_name, layer_id)) {
            return false;
        }
    }
    return true;
}

int fpga_init(void) {
    pthread_mutex_lock(&g_mutex);
    const bool weight_path_bench_requested = env_flag_enabled("FPGA_WEIGHT_PATH_BENCH");
    const bool dma_mapped = dma_is_mapped();
    const bool vpu_mapped = vpu_is_mapped();
    const bool ddr_mapped = ddr_is_mapped();
    const bool all_mapped = dma_mapped && vpu_mapped && ddr_mapped;
    const bool any_mapped = dma_mapped || vpu_mapped || ddr_mapped;

    // main.cpp can initialize the host before ggml-cpu's pthread_once path
    // reaches here.  A completed host is immutable on re-entry: do not parse
    // mutable environment policy, reset P3 state/counters, remap, touch MMIO,
    // or change ZDMA ownership.  The marker intentionally uses only retained
    // host state, so re-entry itself performs no board access.
    if (g_fpga_init_complete) {
        if (g_weight_path_bench.passive_mapping) {
            if (!weight_path_bench_requested || dma_mapped || !vpu_mapped || !ddr_mapped) {
                fpga_fatal("WEIGHT_PATH_BENCH_REENTRY_FAIL requested=%d dma_mapped=%d vpu_mapped=%d ddr_mapped=%d "
                           "action=fail_closed_no_mapping_or_mmio_change",
                           weight_path_bench_requested ? 1 : 0, dma_mapped ? 1 : 0, vpu_mapped ? 1 : 0,
                           ddr_mapped ? 1 : 0);
            }
            pthread_mutex_unlock(&g_mutex);
            return 0;
        }
        if (!all_mapped) {
            fpga_fatal(
                "FPGA_INIT_REENTRY_PARTIAL_FAIL init_complete=1 dma_mapped=%d vpu_mapped=%d ddr_mapped=%d "
                "p3_requested=%d p3_active=%d p3_admitted=%d action=fail_closed_no_config_mmio_dma_map_change",
                dma_mapped ? 1 : 0, vpu_mapped ? 1 : 0, ddr_mapped ? 1 : 0, g_p3_split_scale_requested ? 1 : 0,
                g_p3_split_scale_active ? 1 : 0, g_p3_split_scale_admitted ? 1 : 0);
        }
        LOGINIT(
            "P3_INIT_REENTRY version=%s init_complete=1 dma_mapped=1 vpu_mapped=1 ddr_mapped=1 "
            "p3_requested=%d p3_active=%d p3_admitted=%d mode_committed=%d p3_jobs=%lld "
            "action=return_existing_state_no_config_reset_mmio_dma_map_change",
            FPGA_HOST_TRACE_VERSION, g_p3_split_scale_requested ? 1 : 0, g_p3_split_scale_active ? 1 : 0,
            g_p3_split_scale_admitted ? 1 : 0, g_p3_mode_committed ? 1 : 0, g_p3_jobs);
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }
    if (any_mapped) {
        fpga_fatal(
            "FPGA_INIT_REENTRY_PARTIAL_FAIL init_complete=0 dma_mapped=%d vpu_mapped=%d ddr_mapped=%d "
            "action=fail_closed_no_config_mmio_dma_map_change",
            dma_mapped ? 1 : 0, vpu_mapped ? 1 : 0, ddr_mapped ? 1 : 0);
    }
    if (weight_path_bench_requested) {
        if (!fpga_weight_path_bench_environment_allowed() || !fpga_p2_scale_layout_self_test() ||
            !fpga_weight_path_bench_host_self_test() || !map_weight_path_bench_vpu_ddr() ||
            !fpga_weight_path_bench_read_capabilities()) {
            pthread_mutex_unlock(&g_mutex);
            fpga_fatal("FPGA_WEIGHT_PATH_BENCH passive initialization failed before any ZDMA configuration, DMA, or VPU start");
        }
        g_weight_path_bench.enabled         = true;
        g_weight_path_bench.passive_mapping = true;
        g_weight_path_bench.replayed        = false;
        g_weight_path_bench.graph_seq       = INT_MIN;
        g_weight_path_bench.payload_bytes   = 0U;
        g_weight_path_bench.jobs.clear();
        g_weight_path_bench.cached_payload.clear();
        g_cleanup_done       = false;
        g_fpga_init_complete = true;
        g_fpga_start_us      = now_us();
        LOGINIT(
            "WEIGHT_PATH_BENCH_HOST_SELFTEST pass geometry_overflow=1 statistics_median=1 "
            "cached_direct_bytes=identical "
            "stale_sequence_reset=1 same_sequence_aggregation=1");
        LOGINIT("WEIGHT_PATH_BENCH_MODE enabled=1 route=capture_m1_tiler_then_native_cpu replay_boundary=fpga_advance_sequence_position(1) "
                "timing=one_warmup_plus_five_full_replays metrics=median_min_max_mib_s_bytes_jobs "
                "zdma=not_mapped dma_init=not_run register_writes=none dst_access=none");
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }
    g_init_verbose = env_flag_enabled("FPGA_INIT_VERBOSE");
    // This checks the host-only P2 PARAM packing contract before any UIO,
    // /dev/mem, DDR, VPU, or ZDMA mapping can be attempted.
    if (!fpga_p2_scale_layout_self_test()) {
        fpga_fatal("P2 scale layout self-test failed before hardware mapping; refusing FPGA initialization");
    }
    LOGINIT("P2_SCALE_LAYOUT_SELFTEST pass cases=8 modulo=0,1,2,3 odd_even_multirow=1 max_rows=256 max_group_blocks=64 "
            "invalid_shape_rejected_before_mutation=1");
    if (getenv("FPGA_P2_HOST_PREPACK") || getenv("FPGA_P2_HOST_PREPACK_MB")) {
        fpga_fatal(
            "FPGA_P2_HOST_PREPACK and FPGA_P2_HOST_PREPACK_MB are retired after measured regression; remove both variables");
    }
    const bool pl_scale_enable_env  = env_flag_enabled("FPGA_PL_SCALE_ENABLE");
    const bool pl_scale_disable_env = env_flag_enabled("FPGA_PL_SCALE_DISABLE");
    if (pl_scale_enable_env && pl_scale_disable_env) {
        fpga_fatal("FPGA_PL_SCALE_ENABLE=1 conflicts with FPGA_PL_SCALE_DISABLE=1; select exactly one P2 policy");
    }
    // P2 is production-default.  FPGA_PL_SCALE_ENABLE=1 remains compatible;
    // FPGA_PL_SCALE_DISABLE=1 is the explicit production opt-out.
    g_p2_init_requested = !pl_scale_disable_env;
    g_p3_split_scale_requested = env_flag_enabled("FPGA_P3_SPLIT_SCALE");
    g_p3_split_scale_active = false;
    g_p3_split_scale_admitted = false;
    g_p3_mode_committed = false;
    g_committed_stream_mode = -1;
    g_p3_jobs = 0;
    g_p3_param_dma_bytes = 0;
    g_p3_scratch_dma_bytes = 0;
    g_p3_retire_pass_logs = 0;
    g_p3_retire_pass_suppressed = 0;
    g_p3_retire_timing_enabled = env_flag_enabled("FPGA_P3_RETIRE_TIMING");
    g_p3_retire_timing_calls = 0;
    g_p3_retire_timing_passes = 0;
    g_p3_retire_timing_failures = 0;
    g_p3_retire_timing_valid_samples = 0;
    g_p3_retire_timing_clock_errors = 0;
    g_p3_retire_timing_mmio_reads = 0;
    g_p3_retire_timing_core_total_ns = 0;
    g_p3_retire_timing_core_min_ns = 0;
    g_p3_retire_timing_core_max_ns = 0;
    if (g_p3_split_scale_requested && !g_p2_init_requested) {
        fpga_fatal("FPGA_P3_SPLIT_SCALE=1 requires the admitted P2 base path; FPGA_PL_SCALE_DISABLE=1 is incompatible");
    }
    g_p2_first_act_dma_trace_enabled    = env_flag_enabled("FPGA_P2_FIRST_ACT_TRACE");
    g_p2_first_act_dma_trace_active     = false;
    g_p2_first_act_dma_trace_done       = false;
    g_p2_tile_limit                     = 0;
    g_p2_allow_multitile                = false;
    g_p2_tile_contract_boundary_reached = false;
    g_p2_tile_q16_checks                = 0;
    g_p2_matrix_contract_checks         = 0;
    g_p2_tile_trace_enabled             = false;
    g_p6_accum_trace                    = {};
    g_p2_terminal_trace_enabled         = env_flag_enabled("FPGA_P2_TERMINAL_TRACE");
    g_p2_boundary_diagnostics_enabled   = false;
    g_p2_event_trace_enabled             = env_flag_enabled("FPGA_P2_EVENT_TRACE");
    // P1 input preload is a production default for the normal admitted P2
    // ping-pong route.  Preserve FPGA_P2_INPUT_PRELOAD=1 as an explicit
    // compatibility setting, and use FPGA_P2_INPUT_PRELOAD=0 as the opt-out.
    // P3 is structurally serialized, so the default is suppressed there.
    const bool p2_input_preload_enable_env  = env_flag_enabled("FPGA_P2_INPUT_PRELOAD");
    const bool p2_input_preload_disable_env = env_flag_disabled("FPGA_P2_INPUT_PRELOAD");
    g_p2_input_preload_enabled = !p2_input_preload_disable_env && !g_p3_split_scale_requested;
    g_p1_preload_trace_enabled           = env_flag_enabled("FPGA_P1_PRELOAD_TRACE");
    g_p1_sched_summary_enabled           = env_flag_enabled("FPGA_P1_SCHED_SUMMARY");
    g_pingpong_timing_enabled            = env_flag_enabled("FPGA_PINGPONG_TIMING");
    g_bottleneck_summary_enabled         = !env_flag_disabled("FPGA_BOTTLENECK_SUMMARY");
    g_token_timing_enabled               = !env_flag_disabled("FPGA_TOKEN_TIMING");
    g_token_timing_collection_enabled    = g_token_timing_enabled || g_pingpong_timing_enabled ||
                                           g_bottleneck_summary_enabled;
    g_summary_detail_every               = env_int_value("FPGA_SUMMARY_DETAIL_EVERY", FPGA_DEFAULT_SUMMARY_DETAIL_EVERY, 0, 1000000);
    g_p1_sched_summary                   = {};
    fpga_token_timing_reset();
    g_summary_detail_decode_tokens = 0;
    g_summary_detail_after_error = false;
    g_fpga_perf_decode = {};
    if (g_token_timing_collection_enabled) {
        LOGINIT(
            "TOKEN_TIMING_CONFIG enabled=1 pingpong_detail=%d bottleneck_summary=%d output=/tmp/fpga_debug.log "
            "token_boundary=fpga_advance_sequence_position host_to_ip=ACT_plus_WEIGHT_plus_SCALE "
            "token_read=SPU_OUT_DMA_plus_host_result_read token_log=%d detail_every=%d",
            g_pingpong_timing_enabled ? 1 : 0, g_bottleneck_summary_enabled ? 1 : 0,
            g_token_timing_enabled ? 1 : 0, g_summary_detail_every);
    }
    if (g_bottleneck_summary_enabled) {
        LOGINIT(
            "BOTTLENECK_SUMMARY_CONFIG enabled=1 mode=aggregate_per_graph_sequence per_tile_logs=0 "
            "metrics=prep_decomposition+scheduler_handoff+actual_preload_overlap+tensor_category+zdma_descriptors");
    }
    if (g_p3_split_scale_requested && p2_input_preload_enable_env) {
        fpga_fatal("FPGA_P3_SPLIT_SCALE=1 forbids explicit FPGA_P2_INPUT_PRELOAD=1; P3 serializes both scale DMAs before launch");
    }
    const char * const p2_pack_workers_env = getenv("FPGA_P2_PACK_WORKERS");
    int p2_pack_workers_requested = 2;
    if (p2_pack_workers_env && p2_pack_workers_env[0] != '\0') {
        if (strcmp(p2_pack_workers_env, "1") == 0) {
            p2_pack_workers_requested = 1;
        } else if (strcmp(p2_pack_workers_env, "2") == 0) {
            p2_pack_workers_requested = 2;
        } else {
            fpga_fatal(
                "FPGA_P2_PACK_WORKERS=%s is invalid; only 1 (legacy serial) or 2 (caller plus one persistent "
                "WEIGHT-pack helper) is permitted",
                p2_pack_workers_env);
        }
    }
    g_p1_preload_breadcrumbs              = 0U;
    g_p2_trace_job_id                   = 0U;
    g_p2_trace_tile_id                  = 0U;
    g_p2_trace_bank                     = -1;
    g_p2_dma_transfer_sequence          = 0U;
    g_p2_trace_dma_tag.clear();
    g_p2_pack_workers_requested = p2_pack_workers_requested;
    g_p2_pack_workers_active = 1;
    g_p2_pack_parallel_jobs = 0;
    g_p2_pack_parallel_bytes = 0;
    g_p2_pack_serial_threshold_skips = 0;
    g_p2_pack_main_us = 0;
    g_p2_pack_helper_service_us = 0;
    g_p2_pack_caller_wait_us = 0;

    fpga_p2_init_breadcrumb("phase=config_loader_gate begin pl_scale=1");

    const char * path = getenv("FPGA_PATH");
    if (path && strcmp(path, "dma") != 0 && strcmp(path, "auto") != 0 && strcmp(path, "zdma") != 0) {
        fpga_fatal("FPGA_PATH=%s is not allowed in ZDMA DDR-to-IP build; set FPGA_PATH=dma/zdma or leave it unset",
                   path);
    }
    if (env_flag_enabled("FPGA_DISABLE")) {
        fpga_fatal("FPGA_DISABLE is set, but this build must not silently fall back to CPU");
    }

    g_stage_timing_enabled = env_flag_enabled("FPGA_STAGE_TIMING");
    g_dma_timing_enabled   = env_flag_enabled("FPGA_DMA_TIMING");
    g_ip_timing_enabled    = env_flag_enabled("FPGA_IP_TIMING");
    g_status_stderr        = env_flag_enabled("FPGA_STATUS_STDERR");
    g_trace_data_enabled   = env_flag_enabled("FPGA_TRACE_DATA");
    g_weight_cache_enabled = env_flag_enabled("FPGA_WEIGHT_CACHE");
    const bool p2_weight_residency_env_requested = true;
    g_p2_weight_residency_diagnostic = env_flag_enabled("FPGA_P2_WEIGHT_RESIDENCY_DIAGNOSTIC");
    g_p2_weight_residency_env_requested = p2_weight_residency_env_requested;
    // Board-qualified production policy: P2 residency is always requested.
    // FPGA_P2_WEIGHT_RESIDENCY_DIAGNOSTIC remains an independent opt-in for
    // qualification/diagnostic behavior and does not gate residency admission.
    g_p2_weight_residency_requested = true;
    if (g_p3_split_scale_requested && p2_weight_residency_env_requested) {
        fpga_fatal("FPGA_P3_SPLIT_SCALE=1 forbids FPGA_P2_WEIGHT_RESIDENCY; v79 has no immutable PL-bank scale cache");
    }
    g_p2_weight_residency_enabled   = false;
    g_p2_residency_trace_enabled    = env_flag_enabled("FPGA_P2_RESIDENCY_TRACE");
    g_p2_residency_verify_metadata  = env_flag_enabled("FPGA_P2_RESIDENCY_VERIFY_METADATA");
    g_p2_resident_tiles.fill({});
    g_p2_residency_index.fill(P2_WEIGHT_RESIDENCY_NO_SLOT);
    g_p2_resident_tile_count = 0;
    g_p2_residency_next_slot = 0;
    g_p2_residency_next_off = WEIGHT_CACHE_BASE;
    g_p2_residency_builds = 0;
    g_p2_residency_hits = 0;
    g_p2_residency_misses = 0;
    g_p2_residency_build_failures = 0;
    g_p2_residency_logical_bytes = 0;
    g_p2_residency_miss_alignment = 0;
    g_p2_residency_miss_shape = 0;
    g_p2_residency_miss_collision = 0;
    g_p2_residency_miss_poison = 0;
    g_p2_residency_miss_stale = 0;
    g_p2_residency_miss_mismatch = 0;
    g_p2_residency_miss_capacity = 0;
    g_p2_residency_miss_quiescence = 0;
    g_p2_residency_miss_range = 0;
    g_p2_residency_miss_verify = 0;
    g_p2_residency_probe_count = 0;
    g_p2_residency_probe_exhausted = 0;
    g_p2_residency_host_metadata_hits = 0;
    g_p2_residency_host_metadata_invalidations = 0;
    g_p2_residency_volatile_ddr_reads = 0;
    g_p2_residency_build_us = 0;
    g_p2_residency_select_us = 0;
    g_p2_residency_metadata_validate_us = 0;
    g_p2_residency_resident_param_us = 0;
    g_p2_residency_direct_weight_pack_us = 0;
    g_p2_residency_direct_weight_pack_bytes = 0;
    g_p2_residency_avoided_cpu_pack_bytes = 0;
    g_p2_residency_avoided_ddr_to_ip_bytes = 0;
    if (g_p2_weight_residency_requested) {
        if (++g_p2_weight_residency_epoch == 0U) {
            fpga_fatal("P2 residency epoch exhausted; refusing identity reuse");
        }
    }
    if (env_flag_enabled("FPGA_ACTIVATION_CACHE")) {
        fpga_fatal(
            "FPGA_ACTIVATION_CACHE is unsupported: the existing key is pointer/shape based and cannot prove immutable "
            "activation contents; refusing a stale-activation cache");
    }
    g_activation_cache_enabled = false;
    g_allow_devmem_fallback    = env_flag_enabled("FPGA_ALLOW_DEVMEM");
    g_allow_vpu_devmem_compat =
        !env_flag_enabled("FPGA_VPU_UIO_REQUIRED") && !env_flag_disabled("FPGA_VPU_DEVMEM_COMPAT");
    if (g_p2_init_requested) {
        // Prefer exact UIO resources when Linux exposes them. If they are
        // absent, the fixed board contract may use bounded /dev/mem
        // mappings. DDR still passes the /proc/iomem System RAM overlap
        // preflight before any direct mapping or data-plane transfer.
        const bool p2_devmem_compat = !env_flag_disabled("FPGA_P2_DEVMEM_COMPAT");
        g_allow_devmem_fallback = g_allow_devmem_fallback || p2_devmem_compat;
        g_allow_vpu_devmem_compat = false;
        fpga_p2_init_breadcrumb(
            "phase=config_loader_gate policy=%s my_ip_phys=0x%llx my_ip_offset=0 my_ip_required_size=0x%zx",
            g_allow_devmem_fallback ? "p2_uio_preferred_devmem_compat" : "p2_uio_only",
            (unsigned long long) REG_BASE_PHYS, VPU_DEVMEM_COMPAT_MMAP);
        if (g_weight_cache_enabled) {
            fpga_fatal(
                "P2 requires direct bounded uncached WEIGHT staging so each ACT/WEIGHT/SPU_PARAM handoff can use the "
                "physical-UIO DSB/readback contract; FPGA_WEIGHT_CACHE is not permitted");
        }
    }
    g_strict_coherency               = env_flag_enabled("FPGA_STRICT_COHERENCY");
    g_coherency_platform_whitelisted = env_flag_enabled("FPGA_COHERENCY_PLATFORM_VERIFIED");
    g_run_coherency_stress           = env_flag_enabled("FPGA_COHERENCY_STRESS") || g_strict_coherency;
    if (g_p2_init_requested && p2_msync_or_stress_requested()) {
        fpga_fatal(
            "P2 physical-UIO DDR coherency uses bounded DSB/readback, not msync; FPGA_STRICT_COHERENCY, "
            "FPGA_STRICT_MSYNC, and FPGA_COHERENCY_STRESS are incompatible and are rejected before mapping or "
            "data-plane transfer");
    }
    g_contract_check_abort                = env_flag_enabled("FPGA_CONTRACT_ABORT");
    g_contract_forensic_replay            = !env_flag_disabled("FPGA_CONTRACT_FORENSIC_REPLAY");
    g_clear_result_before_run             = env_flag_enabled("FPGA_CLEAR_RESULT");
    // A repair changes the value consumed by the model, so it is a diagnostic
    // opt-in only.  Correctness runs must report mismatches, not hide them.
    g_contract_raw_repair_enabled         = env_flag_enabled("FPGA_CONTRACT_RAW_REPAIR");
    g_fuse_raw_result_accum               = env_flag_enabled("FPGA_FUSE_RAW_RESULT_ACCUM");
    g_weight_cache_crc_verify_each_lookup = env_flag_enabled("FPGA_WEIGHT_CACHE_CRC_EACH_LOOKUP");
    g_vocab_projection_cpu_bypass =
        !env_flag_disabled("FPGA_VOCAB_PROJECTION_CPU") && !env_flag_enabled("FPGA_ACCELERATE_VOCAB");
    if (!g_fuse_raw_result_accum) {
        LOGINIT(
            "v23 correctness policy: raw-result fusion is disabled by default; set FPGA_FUSE_RAW_RESULT_ACCUM=1 only "
            "after end-to-end CPU/FPGA logits A/B passes");
    }
    if (g_vocab_projection_cpu_bypass) {
        LOGINIT(
            "v23 correctness policy: vocabulary projection remains on CPU; set FPGA_ACCELERATE_VOCAB=1 only after the "
            "deployed bitstream reports protocol=2/id=0x56505532/abi=0x50320003 and logits A/B passes");
    }
    LOGINIT(
        "v58 numerical policy: raw C0 and P2 PL-scale qualification validate immutable Q8_0 sources before launch; P2 "
        "performs tile-only canonical SPU Q16 verification, then CPU-shadows and natively recomputes the complete "
        "current MUL_MAT; matrix F32 comparison is not attempted and P2 never consumes RESULT");
    LOGINIT(
        "v38 ZDMA policy: every descriptor is preceded by a W1C ISR clear that is read back until DMA_DONE and all "
        "error bits are zero. The completion loop therefore accepts only a newly generated DONE event. C0 keeps the "
        "same ACT/WEIGHT staging sequence as primary raw GEMV; staging_restages must remain zero before primary FPGA "
        "use.");
    LOGINIT(
        "v53 loader gate: FPGA_CONTRACT_CHECK and FPGA_PL_SCALE_CONTRACT_CHECK require a completed upstream GGUF "
        "tensor-validation handshake before this host maps MY_IP/ZDMA/DDRHIGH or launches a model VPU transfer.");
    g_log_flush_every               = env_int_value("FPGA_LOG_FLUSH_EVERY", 256, 1, 1000000);
    fpga_log_set_flush_every(g_log_flush_every);
    g_profile_every                 = env_int_value("FPGA_PROFILE_EVERY", FPGA_DEFAULT_PROFILE_EVERY, 0, 1000000);
    g_ip_status_every               = env_int_value("FPGA_IP_STATUS_EVERY", FPGA_DEFAULT_STATUS_EVERY, 0, 1000000);
    g_detail_every                  = env_int_value("FPGA_DETAIL_EVERY", FPGA_DEFAULT_DETAIL_EVERY, 0, 1000000);
    g_contract_check_limit          = env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000);
    g_pl_scale_contract_check_limit = env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000);
    // Qualification must remain serialized unless the operator explicitly
    // requests preload.  This keeps the primary command default-on while
    // preserving the existing bounded contract behavior.
    if ((g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0) &&
        !p2_input_preload_enable_env) {
        g_p2_input_preload_enabled = false;
        LOGINIT(
            "P1 input preload default suppressed for qualification raw_contract=%d p2_contract=%d; "
            "set FPGA_P2_INPUT_PRELOAD=1 only for an intentional preload qualification",
            g_contract_check_limit, g_pl_scale_contract_check_limit);
    }
    if (g_pl_scale_contract_check_limit > 1) {
        fpga_fatal(
            "FPGA_PL_SCALE_CONTRACT_CHECK=%d is unsupported by v58 tile qualification; set it to 1 and select the "
            "exact hardware scope with FPGA_P2_TILE_LIMIT",
            g_pl_scale_contract_check_limit);
    }
    g_p2_allow_multitile              = env_flag_enabled("FPGA_P2_ALLOW_MULTITILE");
    g_p2_boundary_diagnostics_enabled = env_flag_enabled("FPGA_P2_BOUNDARY_DIAGNOSTICS");
    if (g_pl_scale_contract_check_limit > 0) {
        g_p2_tile_limit = env_int_value("FPGA_P2_TILE_LIMIT", 1, 1, 1000000);
        // The prior owner trace completed tile 0 and stopped during tile 1.
        // Refuse to cross that known-unproven boundary unless the owner
        // explicitly asks for it with both variables set.
        if (g_p2_tile_limit > 1 && !g_p2_allow_multitile) {
            fpga_fatal(
                "P2 contract tile limit=%d requires FPGA_P2_ALLOW_MULTITILE=1; default qualification is exactly one "
                "fully retired tile",
                g_p2_tile_limit);
        }
        g_p2_tile_trace_enabled = true;
        LOGINIT(
            "P2 tile qualification policy tile_limit=%d allow_multitile=%d matrix_value_contract=not_attempted; after "
            "the final verified tile, native GGML computes the complete current MUL_MAT and all later eligible GEMVs",
            g_p2_tile_limit, g_p2_allow_multitile ? 1 : 0);
    } else {
        g_p2_tile_trace_enabled = g_p2_boundary_diagnostics_enabled;
    }
    g_q8_source_audit_only = env_flag_enabled("FPGA_SOURCE_AUDIT_ONLY");
    if (g_contract_check_limit > 0 && g_pl_scale_contract_check_limit > 0) {
        fpga_fatal(
            "FPGA_CONTRACT_CHECK and FPGA_PL_SCALE_CONTRACT_CHECK are mutually exclusive; raw RESULT C0 and P2 SPU_OUT "
            "qualification have different result contracts");
    }
    if (g_q8_source_audit_only && (g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0)) {
        fpga_fatal(
            "FPGA_SOURCE_AUDIT_ONLY cannot coexist with FPGA_CONTRACT_CHECK or FPGA_PL_SCALE_CONTRACT_CHECK; audit "
            "must not launch model GEMVs through ZDMA/VPU");
    }
    if ((g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0) && !fpga_model_tensor_validation_passed()) {
        fpga_fatal(
            "qualification loader-validation handshake is missing; no board MMIO was mapped. Rebuild and deploy the "
            "coupled frontend files ggml/src/ggml-cpu/fpga_host.{cpp,h} and src/llama-model-loader.cpp from the same "
            "source tree, then rerun the requested qualification");
    }
    if (g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0) {
        LOGINIT(
            "qualification loader-validation handshake=pass; full GGUF tensor validation completed before FPGA "
            "initialization raw_c0=%d p2_pl_scale=%d",
            g_contract_check_limit, g_pl_scale_contract_check_limit);
    }
    if (g_pl_scale_contract_check_limit > 0) {
        if (pl_scale_disable_env) {
            fpga_fatal("FPGA_PL_SCALE_CONTRACT_CHECK requires P2, but FPGA_PL_SCALE_DISABLE=1 opted it out");
        }
        if (env_flag_enabled("FPGA_PIPELINE_ENABLE")) {
            fpga_fatal(
                "P2 requires single-bank execution; FPGA_PIPELINE_ENABLE is not permitted during "
                "FPGA_PL_SCALE_CONTRACT_CHECK");
        }
        if (g_weight_cache_enabled) {
            fpga_fatal(
                "P2 requires direct bounded weight staging; FPGA_WEIGHT_CACHE is not permitted during "
                "FPGA_PL_SCALE_CONTRACT_CHECK");
        }
    }
    fpga_p2_init_breadcrumb("phase=config_loader_gate pass p2_contract_limit=%d loader_validation=%s",
                            g_pl_scale_contract_check_limit,
                            g_pl_scale_contract_check_limit > 0 ? "pass" : "not_required");
    g_contract_deep_staging = g_contract_check_limit > 0 && !env_flag_disabled("FPGA_CONTRACT_DEEP_STAGING");
    // C0/C1 validates hardware first, but must not overwrite dst while the
    // GGML worker pool owns it.  The ordinary contract route therefore stores
    // the verified hardware result privately and lets the native kernel write
    // dst.  Raw propagation remains an explicit forensic opt-in.
    g_contract_raw_propagation_diagnostic =
        g_contract_check_limit > 0 && env_flag_enabled("FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC");
    if (g_pl_scale_contract_check_limit > 0 && env_flag_enabled("FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC")) {
        fpga_fatal(
            "P2 never propagates raw RESULT; FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC is incompatible with "
            "FPGA_PL_SCALE_CONTRACT_CHECK");
    }
    const char * const legacy_canonical_override = getenv("FPGA_CONTRACT_CANONICAL_DST");
    g_contract_legacy_canonical_override_ignored = g_contract_check_limit > 0 && legacy_canonical_override != nullptr &&
                                                   env_flag_disabled("FPGA_CONTRACT_CANONICAL_DST");
    g_contract_cpu_shadow_dst =
        (g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0) && !g_contract_raw_propagation_diagnostic;
    if (g_contract_cpu_shadow_dst) {
        LOGINIT(
            "v50 C0/C1 shadow policy: every eligible GEMV still runs on ZDMA/VPU and is raw/value checked, but its "
            "result is retained privately while upstream GGML writes dst. This avoids a contract-mode dst race and is "
            "not a CPU fallback. Use FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC=1 only for raw-F32 propagation.");
        if (g_contract_legacy_canonical_override_ignored) {
            LOGINIT(
                "v50 compatibility: FPGA_CONTRACT_CANONICAL_DST=0 is ignored for C0/C1. Use "
                "FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC=1 only for a forensic raw-propagation run.");
        }
    } else if (g_contract_check_limit > 0) {
        LOGINIT(
            "v49 forensic policy: FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC=1; accepted raw-F32 FPGA results will "
            "propagate into CPU attention/KV. This mode diagnoses end-to-end sensitivity and is not a C0/C1 pass.");
    }
    if (env_flag_enabled("FPGA_P6_ACCUM_TRACE")) {
        const char * const tensor = getenv("FPGA_P6_TENSOR");
        int64_t layer = -1;
        int64_t tile_row0 = -1;
        int64_t row_first = -1;
        int64_t row_count = -1;
        int64_t expected_chunks = -1;
        int64_t record_limit = -1;
        if (!tensor || tensor[0] == '\0' || !fpga_p6_parse_required_i64("FPGA_P6_LAYER", 0, &layer) ||
            !fpga_p6_parse_required_i64("FPGA_P6_TILE_ROW0", 0, &tile_row0) ||
            !fpga_p6_parse_required_i64("FPGA_P6_ROW_FIRST", 0, &row_first) ||
            !fpga_p6_parse_required_i64("FPGA_P6_ROW_COUNT", 1, &row_count) ||
            !fpga_p6_parse_required_i64("FPGA_P6_EXPECTED_K_CHUNKS", 2, &expected_chunks) ||
            !fpga_p6_parse_required_i64("FPGA_P6_RECORD_LIMIT", 1, &record_limit) ||
            layer > INT_MAX || tile_row0 > INT64_MAX - row_count || row_first > INT64_MAX - row_count ||
            row_count > INT_MAX || expected_chunks > INT_MAX) {
            fpga_fatal("FPGA_P6_ACCUM_TRACE requires valid explicit selectors: FPGA_P6_TENSOR, FPGA_P6_LAYER, "
                       "FPGA_P6_TILE_ROW0, FPGA_P6_ROW_FIRST, FPGA_P6_ROW_COUNT, FPGA_P6_EXPECTED_K_CHUNKS>=2, "
                       "and FPGA_P6_RECORD_LIMIT");
        }
        // Reject before any P6 log line or board mapping.  The quotient form
        // proves the product is both representable and within the documented
        // finite diagnostic budget without relying on SIZE_MAX as a policy.
        if ((uint64_t) record_limit > (uint64_t) FPGA_P6_RECORD_LIMIT_HARD_MAX ||
            (uint64_t) row_count > (uint64_t) FPGA_P6_RECORD_LIMIT_HARD_MAX / (uint64_t) expected_chunks) {
            fpga_fatal(
                "P6_ARM_GOLDEN_CAPTURE_REJECT reason=record_budget_exceeds_hard_max record_limit=%lld row_count=%lld "
                "expected_k_chunks=%lld hard_max=%zu action=abort_before_mapping",
                (long long) record_limit, (long long) row_count, (long long) expected_chunks,
                FPGA_P6_RECORD_LIMIT_HARD_MAX);
        }
        const size_t required_records = (size_t) row_count * (size_t) expected_chunks;
        if ((size_t) record_limit < required_records ||
            g_pl_scale_contract_check_limit != 1 || !g_p2_allow_multitile ||
            g_p2_tile_limit != (int) expected_chunks || g_p3_split_scale_requested ||
            env_flag_enabled("FPGA_PIPELINE_ENABLE") || g_p2_input_preload_enabled ||
            env_flag_enabled("FPGA_CONTRACT_RAW_PROPAGATION_DIAGNOSTIC") || g_contract_raw_repair_enabled ||
            getenv("FPGA_CONTRACT_CANONICAL_DST") != nullptr || env_flag_enabled("FPGA_SOURCE_AUDIT_ONLY") ||
            env_flag_enabled("FPGA_WEIGHT_PATH_BENCH") ||
            !g_contract_cpu_shadow_dst) {
            fpga_fatal(
                "FPGA_P6_ACCUM_TRACE rejects this mode: requires FPGA_PL_SCALE_CONTRACT_CHECK=1, "
                "FPGA_P2_ALLOW_MULTITILE=1, FPGA_P2_TILE_LIMIT==FPGA_P6_EXPECTED_K_CHUNKS, P2 single-bank, "
                "CPU shadow, no P3/pipeline/preload/raw propagation/raw repair/canonical override/source audit/weight "
                "benchmark, RECORD_LIMIT <= P6 hard maximum, and RECORD_LIMIT >= ROW_COUNT*EXPECTED_K_CHUNKS");
        }
        g_p6_accum_trace.enabled           = true;
        g_p6_accum_trace.tensor            = tensor;
        g_p6_accum_trace.layer             = (int) layer;
        g_p6_accum_trace.tile_row0         = tile_row0;
        g_p6_accum_trace.row_first         = row_first;
        g_p6_accum_trace.row_count         = (int) row_count;
        g_p6_accum_trace.expected_k_chunks = (int) expected_chunks;
        g_p6_accum_trace.record_limit      = (size_t) record_limit;
        LOGINIT(
            "P6_ARM_GOLDEN_CONFIG tensor=%s layer=%d tile_row0=%lld row_first=%lld row_count=%d expected_k_chunks=%d "
            "record_limit=%zu required_records=%zu hard_max=%zu route=p2_single_bank_cpu_shadow pl_bits=unavailable "
            "bit_match=unavailable",
            g_p6_accum_trace.tensor.c_str(), g_p6_accum_trace.layer, (long long) g_p6_accum_trace.tile_row0,
            (long long) g_p6_accum_trace.row_first, g_p6_accum_trace.row_count,
            g_p6_accum_trace.expected_k_chunks, g_p6_accum_trace.record_limit, required_records,
            FPGA_P6_RECORD_LIMIT_HARD_MAX);
    }
    g_dma_trace_enabled =
        env_flag_enabled("FPGA_DMA_AUDIT") || g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0;
    memset(g_dma_trace, 0, sizeof(g_dma_trace));
    g_dma_trace_sequence       = 0;
    g_contract_raw_retry_limit = env_int_value("FPGA_CONTRACT_RAW_RETRY", 1, 0, 8);
    g_runtime_max_rows         = env_int_value("FPGA_RUNTIME_MAX_ROWS", VPU_SAFE_RUNTIME_ROWS, 1, VPU_DEFAULT_ROWS);
    g_vocab_projection_min_n   = env_int64_value("FPGA_VOCAB_PROJECTION_MIN_N", 65536, 1024, LLONG_MAX);
    if (env_flag_enabled("FPGA_TILE_TIMING") && g_detail_every == 0) {
        g_detail_every = 1;
    }
    g_dma_timeout_us          = env_int64_value("FPGA_DMA_TIMEOUT_US", FPGA_DEFAULT_DMA_TIMEOUT_US, 1000, LLONG_MAX);
    g_ip_timeout_us           = env_int64_value("FPGA_IP_TIMEOUT_US", FPGA_DEFAULT_IP_TIMEOUT_US, 1000, LLONG_MAX);
    g_zdma_max_transfer_bytes = (size_t) env_int64_value(
        "FPGA_ZDMA_MAX_TRANSFER_BYTES", (long long) FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES, 16, (long long) UINT32_MAX);
    g_zdma_max_transfer_bytes &= ~(size_t) 0xFU;
    if (g_zdma_max_transfer_bytes == 0U) {
        fpga_fatal("FPGA_ZDMA_MAX_TRANSFER_BYTES must be at least 16 and 16-byte aligned");
    }
    g_large_matrix_min_macs =
        env_int64_value("FPGA_LARGE_MATRIX_MIN_MACS", FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS, 0, LLONG_MAX);
    const double fpga_clock_hz = env_double_value("FPGA_CLOCK_HZ", 0.0, 0.0, 1.0e12);
    if (fpga_clock_hz > 0.0) {
        g_fpga_clock_mhz = fpga_clock_hz / 1.0e6;
    } else {
        g_fpga_clock_mhz = env_double_value("FPGA_CLOCK_MHZ", 0.0, 0.0, 1.0e12);
        if (g_fpga_clock_mhz > 100000.0) {
            g_fpga_clock_mhz /= 1.0e6;
        }
    }
    g_contract_atol                    = env_double_value("FPGA_CONTRACT_ATOL", 1.0e-3, 0.0, 1.0e9);
    g_contract_rtol                    = env_double_value("FPGA_CONTRACT_RTOL", 1.0e-4, 0.0, 1.0e9);
    g_abort_on_cpu_fallback            = !env_flag_disabled("FPGA_ABORT_ON_CPU_FALLBACK");
    g_activation_input_integrity_check = env_flag_enabled("FPGA_INPUT_INTEGRITY_CHECK");
    if ((g_contract_check_limit > 0 || g_pl_scale_contract_check_limit > 0) && !g_activation_input_integrity_check) {
        // Qualification deliberately exercises M>1 layouts and must prove
        // that the host/VPU path leaves each live F32 source intact.  Ordinary
        // primary P2 remains snapshot-free unless explicitly requested.
        g_activation_input_integrity_check = true;
        LOGINIT(
            "qualification input-integrity policy: raw C0/P2 automatically enable the F32 src1 snapshot/verify guard; "
            "FPGA_INPUT_INTEGRITY_CHECK=1 is implied for this qualification run");
    }

    fpga_p2_init_breadcrumb("phase=mapping begin");
    if (!configure_ddr_mapping_policy()) {
        fpga_fatal("DDR cache/mapping policy is invalid; refusing FPGA initialization");
    }
    if (!map_registers_dma_ddr()) {
        fpga_fatal("ZDMA DDR-to-IP FPGA init failed; refusing CPU fallback");
    }
    if (g_p2_init_requested &&
        g_ddr_mapping_kind != fpga_mapping_kind::UIO_PHYSICAL &&
        !(g_allow_devmem_fallback && g_ddr_mapping_kind == fpga_mapping_kind::DEVMEM_PHYSICAL)) {
        fpga_fatal(
            "P2 requires a verified physical UIO mapping or admitted bounded /dev/mem mapping; map_kind=%s. "
            "No ZDMA initialization or data-plane transfer was issued",
            fpga_mapping_kind_name(g_ddr_mapping_kind));
    }
    fpga_p2_init_breadcrumb("phase=mapping pass my_ip_source=%s my_ip_size=0x%zx", g_vpu_map_source.c_str(),
                            g_vpu_map_size);
    configure_weight_cache();
    fpga_p2_init_breadcrumb("phase=zdma_init begin");
    if (g_p2_init_requested && !p2_zdma_preinit_passive_gate()) {
        fpga_fatal(
            "P2 pre-init ZDMA ownership gate failed; CTRL2.EN was set or passive registers were unavailable. No ZDMA "
            "initialization write was issued");
    }
    if (!fpga_dma_init()) {
        fpga_fatal("ZDMA init failed; refusing CPU fallback");
    }
    fpga_p2_init_breadcrumb("phase=zdma_init pass");
    if (g_run_coherency_stress && !fpga_ddr_coherency_stress_test()) {
        fpga_fatal("DDR coherency stress test failed; refusing FPGA execution");
    }

    g_fpga_start_us                  = now_us();
    g_cleanup_done                   = false;
    g_scratch.activation_cache_valid = false;

    fpga_p2_init_breadcrumb("phase=identity_reads begin");
    const uint32_t limits     = vpu_rd32(REG_LIMITS);
    const uint32_t caps       = vpu_rd32(REG_CAPS);
    const uint32_t spu_caps   = vpu_rd32(REG_SPU_CAPS);
    g_stream_protocol_version = vpu_rd32(REG_STREAM_PROTOCOL_VERSION);
    g_bitstream_id            = vpu_rd32(REG_BITSTREAM_ID);
    g_p2_stream_abi_signature = vpu_rd32(REG_P2_STREAM_ABI);
    g_p3_split_scale_abi_signature = vpu_rd32(REG_P3_SPLIT_SCALE_ABI);
    g_spu_stream_status       = vpu_rd32(REG_SPU_STREAM_STATUS);
    g_spu_word_capacity       = (spu_caps >> 16) & 0xFFFFU;
    fpga_p2_init_breadcrumb(
        "phase=identity_reads pass limits=0x%08x caps=0x%08x spu_caps=0x%08x protocol=0x%08x bitstream_id=0x%08x "
        "p2_abi=0x%08x p3_abi=0x%08x stream_status=0x%08x",
        limits, caps, spu_caps, g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature,
        g_p3_split_scale_abi_signature, g_spu_stream_status);
    const int limit_rows  = (int) (limits & 0xFFFFU);
    const int limit_beats = (int) ((limits >> 16) & 0xFFFFU);
    if (limit_rows > 0 && limit_rows <= VPU_DEFAULT_ROWS) {
        g_vpu_max_rows = limit_rows;
    }
    if (limit_beats > 0 && limit_beats <= VPU_DEFAULT_BEATS) {
        g_vpu_max_beats = limit_beats;
        g_vpu_max_cols  = g_vpu_max_beats * VPU_NUM_LANES;
    }

    const bool caps_valid = caps != 0U && caps != 0xFFFFFFFFU;
    if (caps_valid && ((caps & VPU_CAP_PACKED_Q8) != 0U)) {
        const bool pair_interleaved_weight_layout = (caps & VPU_CAP_COMPACT_WEIGHT_LAYOUT) != 0U;
        const int  cap_blocks            = (int) ((caps >> 8) & 0xFFU);
        const int  cap_result_words      = (int) ((caps >> 16) & 0xFFFFU);
        if (pair_interleaved_weight_layout && cap_blocks > 0 && cap_result_words > 0) {
            g_packed_q8_supported    = 1;
            g_packed_q8_max_blocks   = std::min(cap_blocks, g_vpu_max_beats / VPU_BLOCK_BEATS);
            g_packed_q8_result_words = cap_result_words;
        } else if (!pair_interleaved_weight_layout) {
            LOGE(
                "REG_CAPS=0x%08x exposes packed_q8 without protocol-2/VPU2 pair-interleaved padded WEIGHT layout",
                caps);
        }
    }
    if (!caps_valid) {
        fpga_fatal("REG_CAPS=0x%08x is invalid; refusing legacy capability assumptions", caps);
    }
    if (env_flag_enabled("FPGA_FORCE_PACKED_Q8")) {
        fpga_fatal(
            "FPGA_FORCE_PACKED_Q8 is not permitted in the production host; the bitstream must advertise packed-Q8 "
            "capability");
    }
    const bool bitstream_id_compatible        = g_bitstream_id == FPGA_EXPECTED_BITSTREAM_ID;
    const bool stream_protocol_compatible     = g_stream_protocol_version == FPGA_REQUIRED_STREAM_PROTOCOL_VERSION;
    const bool p2_abi_compatible              = g_p2_stream_abi_signature == FPGA_REQUIRED_P2_STREAM_ABI;
    const bool raw_fpga_compatible            = bitstream_id_compatible && stream_protocol_compatible && p2_abi_compatible;
    const bool legacy_raw_diagnostic_override = env_flag_enabled("FPGA_ALLOW_UNVERIFIED_LEGACY_RAW");
    if (legacy_raw_diagnostic_override) {
        fpga_fatal("FPGA_ALLOW_UNVERIFIED_LEGACY_RAW is forbidden by protocol-2/VPU2 ABI v3; no unverified launch or CPU fallback is permitted");
    }
    if (!raw_fpga_compatible) {
        fpga_fatal(
            "FPGA ABI v3 admission failed bitstream_id=0x%08x expected_id=0x%08x id_ok=%d stream_protocol=0x%08x "
            "required_protocol=%u protocol_ok=%d p2_abi=0x%08x required_p2_abi=0x%08x abi_ok=%d; refusing all "
            "self-test and GEMV launches without CPU fallback",
            g_bitstream_id, FPGA_EXPECTED_BITSTREAM_ID, bitstream_id_compatible ? 1 : 0, g_stream_protocol_version,
            FPGA_REQUIRED_STREAM_PROTOCOL_VERSION, stream_protocol_compatible ? 1 : 0, g_p2_stream_abi_signature,
            FPGA_REQUIRED_P2_STREAM_ABI, p2_abi_compatible ? 1 : 0);
    }
    g_legacy_raw_cpu_bypass = false;
    if (g_runtime_max_rows > 0 && g_runtime_max_rows < g_vpu_max_rows) {
        g_vpu_max_rows = g_runtime_max_rows;
    }
    g_packed_q8_result_words =
        (g_vpu_max_rows * g_packed_q8_max_blocks + VPU_RESULT_PACK_LANES - 1) / VPU_RESULT_PACK_LANES;
    g_vpu_pingpong_supported                = caps_valid && ((caps & VPU_CAP_PINGPONG_BANKS) != 0U);
    g_vpu_descriptor_supported              = caps_valid && ((caps & VPU_CAP_JOB_DESCRIPTOR) != 0U);
    const bool spu_caps_valid               = spu_caps != 0U && spu_caps != 0xFFFFFFFFU;
    const bool vpu_p2_two_row_transport_cap = caps_valid && ((caps & VPU_CAP_P2_TWO_ROW_TRANSPORT) != 0U);
    // Raw self-tests and raw GEMVs also program VPU_MODE_P2_TWO_ROW because
    // VPU2 consumes the same pair-interleaved padded WEIGHT ABI.  Do not let
    // the explicit PL-scale opt-out weaken that VPU-side transport contract.
    if (!vpu_p2_two_row_transport_cap) {
        fpga_fatal(
            "REG_CAPS=0x%08x lacks VPU_CAP_P2_TWO_ROW_TRANSPORT; refusing every raw/P2 self-test and GEMV before "
            "VPU_MODE_P2_TWO_ROW can be configured",
            caps);
    }
    const bool spu_q8_scale_pair_stream_cap =
        spu_caps_valid && ((spu_caps & SPU_CAP_VPU_Q8_SCALE_PAIR_STREAM) != 0U);
    const bool p2_two_row_transport_compatible = vpu_p2_two_row_transport_cap && spu_q8_scale_pair_stream_cap;
    const bool pl_scale_requested           = g_p2_init_requested;
    const bool spu_stream_quiescent         = (g_spu_stream_status & SPU_STREAM_STATUS_QUIESCENT) != 0U;
    const bool spu_word_capacity_compatible = g_spu_word_capacity >= P2_REQUIRED_SPU_WORD_CAPACITY;
    // This admission is intentionally P2-only.  Raw/C0 remains compatible
    // with deployed VPU1/protocol1 images that predate SPU_PARAM/SPU_OUT v1.
    const bool p2_admission_compatible = p2_abi_compatible && spu_stream_quiescent && spu_word_capacity_compatible &&
                                          p2_two_row_transport_compatible;
    const bool p3_cap_compatible = spu_caps_valid && ((spu_caps & SPU_CAP_P3_SPLIT_SCALE) != 0U);
    const bool p3_abi_compatible = g_p3_split_scale_abi_signature == FPGA_REQUIRED_P3_SPLIT_SCALE_ABI;
    const uint32_t p3_bank_words = (g_spu_word_capacity % 2U) == 0U ? g_spu_word_capacity / 2U : 0U;
    const uint32_t p3_required_bank_words =
        ((uint32_t) P3_MAX_ROWS * (uint32_t) P3_MAX_GROUP_BLOCKS + 7U) / 8U;
    const bool p3_capacity_compatible = p3_bank_words >= p3_required_bank_words;
    if (g_p3_split_scale_requested) {
        LOGINIT(
            "P3_ADMISSION requested=1 p2_admission=%d p3_cap=%d p3_abi=0x%08x required_p3_abi=0x%08x "
            "spu_words=%u bank_words=%u bank_bytes=0x%zx required_bank_words=%u stream_status=0x%08x",
            p2_admission_compatible ? 1 : 0, p3_cap_compatible ? 1 : 0, g_p3_split_scale_abi_signature,
            FPGA_REQUIRED_P3_SPLIT_SCALE_ABI, g_spu_word_capacity, p3_bank_words, (size_t) p3_bank_words * 16U,
            p3_required_bank_words,
            g_spu_stream_status);
        if (!p2_admission_compatible || !p3_cap_compatible || !p3_abi_compatible || !p3_capacity_compatible) {
            fpga_fatal(
                "FPGA_P3_SPLIT_SCALE=1 admission failed p2=%d p3_cap=%d p3_abi=0x%08x required=0x%08x "
                "bank_words=%u required=%u; no P3 mode/data-plane write and no fallback was issued",
                p2_admission_compatible ? 1 : 0, p3_cap_compatible ? 1 : 0, g_p3_split_scale_abi_signature,
                FPGA_REQUIRED_P3_SPLIT_SCALE_ABI, p3_bank_words, p3_required_bank_words);
        }
        g_p3_split_scale_active = true;
        g_p3_split_scale_admitted = true;
    }
    if (pl_scale_requested) {
        LOGINIT(
            "P2 admission abi=0x%08x required=0x%08x abi_ok=%d stream_status=0x%08x quiescent=%d spu_word_capacity=%u "
            "required_words=%u capacity_ok=%d vpu_two_row_transport_cap=%d spu_q8_scale_pair_stream_cap=%d "
            "two_row_transport_ok=%d layout=weight_pair_interleaved_padded_v2,param128x4_fp16pairs,out128_q16row",
            g_p2_stream_abi_signature, FPGA_REQUIRED_P2_STREAM_ABI, p2_abi_compatible ? 1 : 0, g_spu_stream_status,
            spu_stream_quiescent ? 1 : 0, g_spu_word_capacity, P2_REQUIRED_SPU_WORD_CAPACITY,
            spu_word_capacity_compatible ? 1 : 0, vpu_p2_two_row_transport_cap ? 1 : 0,
            spu_q8_scale_pair_stream_cap ? 1 : 0, p2_two_row_transport_compatible ? 1 : 0);
    }
    g_spu_silu_supported    = spu_caps_valid && ((spu_caps & SPU_CAP_SILU_MUL) != 0U);
    g_spu_rmsnorm_supported = spu_caps_valid && ((spu_caps & SPU_CAP_RMSNORM) != 0U);
    g_spu_rope_supported    = spu_caps_valid && ((spu_caps & SPU_CAP_ROPE) != 0U);
    g_spu_softmax_supported = spu_caps_valid && ((spu_caps & SPU_CAP_SOFTMAX) != 0U);
    g_spu_q8_scale_stream_supported =
        caps_valid && spu_caps_valid && g_vpu_pingpong_supported && g_vpu_descriptor_supported &&
        bitstream_id_compatible && (g_stream_protocol_version == FPGA_REQUIRED_STREAM_PROTOCOL_VERSION) &&
        ((caps & VPU_CAP_SPU_RAW_STREAM) != 0U) && ((caps & VPU_CAP_SPU_Q8_SCALE_STREAM) != 0U) &&
        ((spu_caps & SPU_CAP_VPU_RAW_STREAM) != 0U) && ((spu_caps & SPU_CAP_VPU_Q8_SCALE_STREAM) != 0U) &&
        p2_admission_compatible && pl_scale_requested;
    const bool pipeline_enable_env  = env_flag_enabled("FPGA_PIPELINE_ENABLE");
    const bool pipeline_disable_env = env_flag_enabled("FPGA_PIPELINE_DISABLE");
    if (pipeline_enable_env && pipeline_disable_env) {
        fpga_fatal("FPGA_PIPELINE_ENABLE=1 conflicts with FPGA_PIPELINE_DISABLE=1; select exactly one scheduler policy");
    }
    // Once P2 admission has passed, the descriptor/bank scheduler is the
    // production default.  FPGA_PIPELINE_ENABLE=1 remains compatible;
    // FPGA_PIPELINE_DISABLE=1 is the explicit serialized-P2 opt-out.
    g_pingpong_scheduler_enabled = !g_p3_split_scale_active && g_p2_init_requested && g_vpu_pingpong_supported && g_vpu_descriptor_supported &&
                                   raw_fpga_compatible && p2_admission_compatible && !pipeline_disable_env;
    if (g_p2_input_preload_enabled &&
        (!g_pingpong_scheduler_enabled || !g_spu_q8_scale_stream_supported || g_contract_check_limit > 0 ||
         g_pl_scale_contract_check_limit > 0)) {
        fpga_fatal(
            "FPGA_P2_INPUT_PRELOAD=1 requires admitted P2 ping-pong production (not qualification): scheduler=%d "
            "stream_supported=%d raw_contract_limit=%d p2_contract_limit=%d",
            g_pingpong_scheduler_enabled ? 1 : 0, g_spu_q8_scale_stream_supported ? 1 : 0,
            g_contract_check_limit, g_pl_scale_contract_check_limit);
    }
    if (pl_scale_requested && !g_spu_q8_scale_stream_supported) {
        fpga_fatal(
            "FPGA_PL_SCALE_ENABLE requested but P2 admission or stream/descriptors/protocol are incompatible: "
            "caps=0x%08x spu_caps=0x%08x protocol=0x%08x required_protocol=%u p2_abi=0x%08x required_p2_abi=0x%08x "
            "stream_status=0x%08x required_quiescent=0x%08x spu_words=%u required_words=%u "
            "vpu_two_row_transport_cap=%d spu_q8_scale_pair_stream_cap=%d two_row_transport_ok=%d",
            caps, spu_caps, g_stream_protocol_version, FPGA_REQUIRED_STREAM_PROTOCOL_VERSION, g_p2_stream_abi_signature,
            FPGA_REQUIRED_P2_STREAM_ABI, g_spu_stream_status, SPU_STREAM_STATUS_QUIESCENT, g_spu_word_capacity,
            P2_REQUIRED_SPU_WORD_CAPACITY, vpu_p2_two_row_transport_cap ? 1 : 0,
            spu_q8_scale_pair_stream_cap ? 1 : 0, p2_two_row_transport_compatible ? 1 : 0);
    }
    if (g_p2_weight_residency_requested) {
        const size_t expected_map = (size_t) WEIGHT_CACHE_BASE +
                                    (size_t) g_p2_weight_residency_budget_mb * 1024U * 1024U;
        const bool residency_admitted = g_p2_init_requested && raw_fpga_compatible && p2_admission_compatible &&
                                        g_spu_q8_scale_stream_supported &&
                                        g_ddr_mapping_kind == fpga_mapping_kind::UIO_PHYSICAL &&
                                        g_ddr_map_size == expected_map && g_ddr_advertised_size >= expected_map;
        if (!residency_admitted) {
            fpga_fatal(
                "P2 residency admission failed requested=1 map_kind=%s map_size=0x%zx expected_map=0x%zx "
                "advertised=0x%zx raw_abi=%d p2_abi=%d stream=%d; refusing cache selection or CPU fallback",
                fpga_mapping_kind_name(g_ddr_mapping_kind), g_ddr_map_size, expected_map, g_ddr_advertised_size,
                raw_fpga_compatible ? 1 : 0, p2_admission_compatible ? 1 : 0,
                g_spu_q8_scale_stream_supported ? 1 : 0);
        }
        g_p2_weight_residency_enabled = true;
        LOGINIT(
            "P2_RESIDENCY_ADMISSION pass diagnostic=%d verify_metadata=%d epoch=%llu base_off=0x%08x range=[0x%llx,0x%llx) map_kind=%s slots=%zu index_buckets=%zu "
            "protocol=0x%08x bitstream_id=0x%08x p2_abi=0x%08x",
            g_p2_weight_residency_diagnostic ? 1 : 0, g_p2_residency_verify_metadata ? 1 : 0,
            (unsigned long long) g_p2_weight_residency_epoch, WEIGHT_CACHE_BASE,
            (unsigned long long) (DDR_BASE_PHYS + WEIGHT_CACHE_BASE),
            (unsigned long long) (DDR_BASE_PHYS + expected_map), fpga_mapping_kind_name(g_ddr_mapping_kind),
            P2_WEIGHT_RESIDENCY_SLOT_CAPACITY, P2_WEIGHT_RESIDENCY_INDEX_BUCKETS,
            g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature);
    }
    fpga_p2_init_breadcrumb(
        "phase=admission pass p2_abi_ok=%d quiescent=%d capacity_ok=%d vpu_two_row_transport_cap=%d "
        "spu_q8_scale_pair_stream_cap=%d two_row_transport_ok=%d stream_supported=%d",
                            p2_abi_compatible ? 1 : 0, spu_stream_quiescent ? 1 : 0,
                            spu_word_capacity_compatible ? 1 : 0, vpu_p2_two_row_transport_cap ? 1 : 0,
                            spu_q8_scale_pair_stream_cap ? 1 : 0, p2_two_row_transport_compatible ? 1 : 0,
                            g_spu_q8_scale_stream_supported ? 1 : 0);
    if (g_p2_init_requested && !g_p3_split_scale_active && !pipeline_disable_env && !g_pingpong_scheduler_enabled) {
        fpga_fatal(
            "P2 default ping-pong scheduler requires compatible bank/descriptor/identity/stream admission; "
            "caps=0x%08x spu_caps=0x%08x protocol=0x%08x p2_abi=0x%08x stream_status=0x%08x "
            "vpu_two_row_transport_cap=%d spu_q8_scale_pair_stream_cap=%d two_row_transport_ok=%d",
            caps, spu_caps, g_stream_protocol_version, g_p2_stream_abi_signature, g_spu_stream_status,
            vpu_p2_two_row_transport_cap ? 1 : 0, spu_q8_scale_pair_stream_cap ? 1 : 0,
            p2_two_row_transport_compatible ? 1 : 0);
    }
    LOGINIT(
        "raw FPGA compatibility gate id_ok=%d protocol_ok=%d raw_compatible=%d legacy_diagnostic_override=%d route=%s",
        bitstream_id_compatible ? 1 : 0, stream_protocol_compatible ? 1 : 0, raw_fpga_compatible ? 1 : 0,
        legacy_raw_diagnostic_override ? 1 : 0,
        g_legacy_raw_cpu_bypass ?
            "cpu_quarantine" :
            (g_pl_scale_contract_check_limit > 0 ? "p2_pl_scale_contract" :
                                                   (g_contract_check_limit > 0 ? "contract_diagnostic" : "raw_fpga")));
    LOGINIT(
        "P2_CONFIG enabled=%d contract_limit=%d pipeline_default_on=%d pipeline_disable=%d scheduler=%d "
        "input_preload=%d input_preload_policy=default_on_opt_out input_preload_disable_env=%d route=%s identity "
        "protocol=0x%08x bitstream_id=0x%08x p2_abi=0x%08x stream_status=0x%08x spu_words=%u descriptor_cap=%d "
        "pingpong_cap=%d vpu_two_row_transport_cap=%d spu_q8_scale_pair_stream_cap=%d two_row_transport_ok=%d "
        "windows spu_param=0x%08x spu_out=0x%08x",
        g_spu_q8_scale_stream_supported ? 1 : 0, g_pl_scale_contract_check_limit,
        pipeline_disable_env ? 0 : 1, pipeline_disable_env ? 1 : 0, g_pingpong_scheduler_enabled ? 1 : 0,
        g_p2_input_preload_enabled ? 1 : 0, p2_input_preload_disable_env ? 1 : 0,
        g_pl_scale_contract_check_limit > 0 ? "p2_contract_cpu_shadow" :
                                              (g_pingpong_scheduler_enabled ? "pingpong_production" :
                                                                               (g_spu_q8_scale_stream_supported ? "serialized_p2" : "disabled")),
        g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature, g_spu_stream_status, g_spu_word_capacity,
        g_vpu_descriptor_supported ? 1 : 0, g_vpu_pingpong_supported ? 1 : 0,
        vpu_p2_two_row_transport_cap ? 1 : 0, spu_q8_scale_pair_stream_cap ? 1 : 0,
        p2_two_row_transport_compatible ? 1 : 0, SPU_PARAM_BASE, SPU_OUT_BASE);
    LOGINIT(
        "P3_CONFIG requested=%d active=%d p3_abi=0x%08x required=0x%08x cap=%d bank_words=%u bank_bytes=0x%zx max_rows=%d "
        "max_blocks=%d scheduler=serialized scale_preload_overlap=0 residency=disabled",
        g_p3_split_scale_requested ? 1 : 0, g_p3_split_scale_active ? 1 : 0, g_p3_split_scale_abi_signature,
        FPGA_REQUIRED_P3_SPLIT_SCALE_ABI, p3_cap_compatible ? 1 : 0, p3_bank_words, (size_t) p3_bank_words * 16U, P3_MAX_ROWS,
        P3_MAX_GROUP_BLOCKS);
    // Every non-P3 route establishes and reads back P2 mode 0 before any
    // self-test or model transfer.  P3 also starts from mode 0 and switches
    // only immediately before its first dense-table staging operation.
    if (!fpga_set_split_scale_mode(0U, g_p3_split_scale_active ? "P3 baseline" : "P2 route baseline")) {
        fpga_fatal("stream mode-0 baseline was not quiescent/readable; refusing P2/P3 data-plane traffic");
    }
    if (!g_p2_init_requested) {
        // Preserve the established raw-v52 initialization write.  P2 init is
        // passive after admission; its first model tile selects its bank
        // inside the force-flushed CONTROL_DESCRIPTOR trace boundary.
        vpu_select_banks(0, 0);
    }

    const char * mapping_policy = g_allow_devmem_fallback ?
                                      "diagnostic_all_resources" :
                                      (g_allow_vpu_devmem_compat ? "uio_dma_ddr+vpu_devmem_compat" : "uio_required");

    LOGINIT(
        "ready version=%s path=%s rows=%d host_row_limit=%d col_beats=%d cols=%d packed_q8=%d max_group_blocks=%d "
        "result_words=%d zdma_max_transfer_bytes=%zu pingpong_cap=%d descriptor_cap=%d scheduler=%d input_preload=%d "
        "scheduler_policy=default_on_opt_out input_preload_policy=default_on_opt_out pl_scale=%d "
        "pl_scale_policy=default_on_opt_out stream_protocol=0x%08x bitstream_id=0x%08x "
        "spu_silu=%d spu_rms=%d spu_rope=%d spu_softmax=%d weight_cache=%d activation_cache=%d "
        "input_integrity_check=%d vocab_cpu_bypass=%d legacy_raw_cpu_bypass=%d vocab_min_n=%lld contract_check=%d "
        "source_audit_only=%d contract_abort=%d contract_forensics=%d contract_deep_staging=%d "
        "contract_cpu_shadow_dst=%d contract_raw_propagation_diagnostic=%d raw_retry=%d raw_repair=%d "
        "raw_accum_fused=%d result_clear=%d strict_coherency=%d clock_mhz=%.3f profile_log=1 dma_detail=%d "
        "ip_detail=%d detail_every=%d flush_every=%d",
        FPGA_HOST_TRACE_VERSION, path ? path : "dma(default)", g_vpu_max_rows, g_runtime_max_rows, g_vpu_max_beats,
        g_vpu_max_cols, g_packed_q8_supported, g_packed_q8_max_blocks, g_packed_q8_result_words,
        g_zdma_max_transfer_bytes, g_vpu_pingpong_supported ? 1 : 0, g_vpu_descriptor_supported ? 1 : 0,
        g_pingpong_scheduler_enabled ? 1 : 0, g_p2_input_preload_enabled ? 1 : 0,
        g_spu_q8_scale_stream_supported ? 1 : 0, g_stream_protocol_version,
        g_bitstream_id, g_spu_silu_supported ? 1 : 0, g_spu_rmsnorm_supported ? 1 : 0, g_spu_rope_supported ? 1 : 0,
        g_spu_softmax_supported ? 1 : 0, g_weight_cache_enabled ? 1 : 0, g_activation_cache_enabled ? 1 : 0,
        g_activation_input_integrity_check ? 1 : 0, g_vocab_projection_cpu_bypass ? 1 : 0,
        g_legacy_raw_cpu_bypass ? 1 : 0, (long long) g_vocab_projection_min_n, g_contract_check_limit,
        g_q8_source_audit_only ? 1 : 0, g_contract_check_abort ? 1 : 0, g_contract_forensic_replay ? 1 : 0,
        g_contract_deep_staging ? 1 : 0, g_contract_cpu_shadow_dst ? 1 : 0,
        g_contract_raw_propagation_diagnostic ? 1 : 0, g_contract_raw_retry_limit,
        g_contract_raw_repair_enabled ? 1 : 0, g_fuse_raw_result_accum ? 1 : 0, g_clear_result_before_run ? 1 : 0,
        g_strict_coherency ? 1 : 0, g_fpga_clock_mhz, g_dma_timing_enabled ? 1 : 0, g_ip_timing_enabled ? 1 : 0,
        g_detail_every, g_log_flush_every);
    LOGINIT(
        "ZDMA trace policy enabled=%d depth=%zu trigger=qualification_mismatch_or_transfer_failure "
        "raw_contract_check=%d p2_contract_check=%d explicit_env=FPGA_DMA_AUDIT",
        g_dma_trace_enabled ? 1 : 0, FPGA_DMA_TRACE_DEPTH, g_contract_check_limit, g_pl_scale_contract_check_limit);
    if (g_activation_input_integrity_check) {
        LOGINIT(
            "FPGA_INPUT_INTEGRITY_CHECK=1: each raw FPGA matmul snapshots its logical F32 src1 before launch and "
            "verifies it after dst ownership returns; this is a qualification guard for M>1 graph layouts, not a CPU "
            "fallback or numerical replacement");
    }
    LOGINIT(
        "ZDMA byte-counter policy clear=before_every_transfer current=0x%08x error_mask=0x%08x; BYTE_CNT_OVRFL is "
        "fatal and produces a bounded DMA trace",
        g_dma ? g_dma->ZDMA_CH_TOTAL_BYTE : 0U, ZDMA_ISR_ERROR_MASK);
    LOGINIT(
        "ZDMA descriptor policy max_transfer_bytes=%zu; larger ACT/WEIGHT/RESULT copies are submitted as ordered "
        "contiguous chunks and preserve the VPU data layout",
        g_zdma_max_transfer_bytes);
    LOGI(
        "MANIFEST host_version=%s host_build=\"%s %s\" stream_protocol=%u bitstream_id=0x%08x "
        "p2_abi=0x%08x p3_abi=0x%08x ddr_phys=[0x%llx,0x%llx) ddr_map_size=0x%zx "
        "dma_map_source=%s vpu_map_source=%s ddr_map_source=%s zdma_descriptor_bytes=%zu rows=%d workers=%d "
        "pingpong=%d preload=%d p3=%d vocab_cpu_bypass=%d",
        FPGA_HOST_TRACE_VERSION, __DATE__, __TIME__, g_stream_protocol_version, g_bitstream_id,
        g_p2_stream_abi_signature, g_p3_split_scale_abi_signature, (unsigned long long) DDR_BASE_PHYS,
        (unsigned long long) DDR_END_EXCLUSIVE, g_ddr_map_size, g_dma_map_source.c_str(), g_vpu_map_source.c_str(),
        g_ddr_map_source.c_str(), g_zdma_max_transfer_bytes, g_runtime_max_rows, g_p2_pack_workers_requested,
        g_pingpong_scheduler_enabled ? 1 : 0, g_p2_input_preload_enabled ? 1 : 0,
        g_p3_split_scale_active ? 1 : 0, g_vocab_projection_cpu_bypass ? 1 : 0);
    LOGINIT("bases my_ip=0x%llx reg=0x%llx lmm=0x%llx dma=0x%llx ddr=0x%llx", (unsigned long long) MY_IP_BASE_ADDRESS,
            (unsigned long long) REG_BASE_PHYS, (unsigned long long) LMM_BASE_PHYS, (unsigned long long) DMA_BASE_PHYS,
            (unsigned long long) DDR_BASE_PHYS);
    LOGINIT(
        "mappings policy=%s dma=%s virt=0x%llx size=0x%zx vpu=%s virt=0x%llx size=0x%zx ddr=%s ddr_map_kind=%s "
        "virt=0x%llx mapped_size=0x%zx advertised_size=0x%zx",
        mapping_policy, g_dma_map_source.c_str(), fpga_ptr_addr(g_dma), g_dma_map_size, g_vpu_map_source.c_str(),
        fpga_ptr_addr(g_vpu), g_vpu_map_size, g_ddr_map_source.c_str(), fpga_mapping_kind_name(g_ddr_mapping_kind),
        fpga_ptr_addr(g_ddr), g_ddr_map_size, g_ddr_advertised_size);
    LOGINIT(
        "VPU windows act=0x%08x weight=0x%08x result=0x%08x spu_out=0x%08x spu_param=0x%08x "
        "data_movement=ZDMA_bulk_copy no_axi_stream_main=1",
        ACT_BASE, WEIGHT_BASE, RESULT_BASE, SPU_OUT_BASE, SPU_PARAM_BASE);
    LOGINIT("VPU raw_limits=0x%08x caps=0x%08x spu_caps=0x%08x", limits, caps, spu_caps);
    if (g_vpu_max_beats == 256) {
        LOGE(
            "MAX_COL_BEATS=256 detected; DMA-to-IP path will run, but this large BRAM setting is still suspicious for "
            "timing/resource use");
    }
    LOGINIT(
        "cache coherency source=%s map_kind=%s strict=%d whitelist=%d stress=%d; P2 physical-UIO DDR uses no_msync "
        "with DSB/readback before device and DSB before CPU reads, while raw-v52/C0 retains its legacy msync path",
        g_ddr_map_source.c_str(), fpga_mapping_kind_name(g_ddr_mapping_kind), g_strict_coherency ? 1 : 0,
        g_coherency_platform_whitelisted ? 1 : 0, g_run_coherency_stress ? 1 : 0);
    LOGINIT("fallback policy: FPGA_ABORT_ON_CPU_FALLBACK=%d default_no_cpu_matmul_fallback=1",
            g_abort_on_cpu_fallback ? 1 : 0);

    if (!g_packed_q8_supported) {
        fpga_fatal("REG_CAPS=0x%08x does not expose packed_q8 capability; refusing CPU fallback", caps);
    }
    if (!fpga_weight_layout_host_self_test()) {
        fpga_fatal("P2 pair-interleaved padded WEIGHT host self-test failed; refusing all launches");
    }
    if (g_p3_split_scale_active && !fpga_p3_split_scale_host_self_test()) {
        fpga_fatal("P3 dense split-scale host self-test failed; refusing P3 data-plane traffic");
    }
    if (g_pl_scale_contract_check_limit > 0 && !fpga_p2_cumulative_tile_limit_host_self_test()) {
        fpga_fatal("P3 cumulative tile-limit host self-test failed; refusing qualification traffic");
    }
    if (g_p6_accum_trace.enabled && !fpga_p6_accum_trace_host_self_test()) {
        fpga_fatal("P6 ARM golden trace host self-test failed; refusing qualification traffic");
    }
    if (g_p6_accum_trace.enabled) {
        LOGINIT(
            "P6_ARM_GOLDEN_HOST_SELFTEST pass selectors=1 m2_defer=1 m1_eligible=1 "
            "cleanup_incomplete_rejected=1 cleanup_complete_accepted=1 cap=1 plus_zero_bits=1 minus_zero_bits=1 "
            "four_contiguous_chunks=1 duplicate_gap_premature_missing_final_extra_rejected=1");
    }
    if (g_p2_init_requested) {
        // P2 initialization is deliberately passive after UIO, ZDMA, and
        // identity admission.  The first bounded P2 tile supplies its exact
        // scale table and owns all VPU-to-SPU stream traffic.
        fpga_p2_init_breadcrumb("phase=data_plane_selftests skipped reason=p2_passive_init");
    } else {
        // Preserve the established v52 raw-GEMV validation sequence exactly.
        if (!fpga_dma_basic_self_test()) {
            fpga_fatal("basic ZDMA-to-IP self-test failed; refusing CPU fallback");
        }
        if (!fpga_dma_packed_self_test()) {
            fpga_fatal("packed Q8 ZDMA-to-IP self-test failed; refusing CPU fallback");
        }
        if (!fpga_dma_row_limit_self_test()) {
            fpga_fatal("row-limit packed Q8 self-test failed; refusing CPU fallback");
        }
    }

    // The helper is admitted only after all mappings, identity checks, and
    // host layout self-tests have passed.  A creation failure is fail-closed:
    // release the outer lock, tear down the just-created mappings, then abort
    // without advertising a ready hardware route.
    if (g_p2_pack_workers_requested == 2) {
        if (!fpga_p2_pack_worker_start()) {
            pthread_mutex_unlock(&g_mutex);
            fpga_cleanup();
            fpga_fatal(
                "FPGA_P2_PACK_WORKERS=2 could not create its persistent WEIGHT-pack helper; mappings were torn down "
                "before ready and no DMA/IP job was started");
        }
        g_p2_pack_workers_active = 2;
    }
    LOGINIT(
        "P2_PACK_CONFIG requested_workers=%d active_workers=%d parallel_min_bytes=%zu policy=pair_disjoint_before_dma "
        "helper_owns=no_dma_no_vpu_no_spu_no_descriptor",
        g_p2_pack_workers_requested, g_p2_pack_workers_active, FPGA_P2_PACK_PARALLEL_MIN_BYTES);

    g_fpga_init_complete = true;
    fpga_p2_init_breadcrumb("phase=ready route=p2_single_bank init_data_plane=none pack_workers_requested=%d active=%d",
                            g_p2_pack_workers_requested, g_p2_pack_workers_active);
    pthread_mutex_unlock(&g_mutex);
    return 0;
}

void fpga_cleanup(void) {
    pthread_mutex_lock(&g_mutex);
    if (g_cleanup_done) {
        pthread_mutex_unlock(&g_mutex);
        return;
    }
    g_cleanup_done = true;
    const bool fpga_initialized_before_cleanup = g_fpga_init_complete;
    LOGPROOF(
        "cleanup begin lifecycle=explicit-before-backend-free ddr_mapped=%d vpu_mapped=%d dma_mapped=%d "
        "weight_cache_entries=%zu weight_path_bench_passive=%d",
        ddr_is_mapped() ? 1 : 0, vpu_is_mapped() ? 1 : 0, dma_is_mapped() ? 1 : 0, g_weight_cache.size(),
        g_weight_path_bench.passive_mapping ? 1 : 0);
    if (g_p1_sched_summary_enabled) {
        fpga_p1_sched_summary_emit("cleanup");
    }
    if (g_token_timing_collection_enabled && g_token_timing.active) {
        fpga_token_timing_emit_final(monotonic_now_us());
    }

    // No local source establishes a safe software abort/reset sequence for an
    // in-flight ZDMA descriptor.  Therefore cleanup must observe the
    // documented hardware-cleared CTRL2.EN state before invalidating DDR
    // metadata or unmapping any participant in the transfer.
    if (dma_is_mapped() && !zdma_wait_channel_disabled("cleanup", "before_unmap")) {
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal(
            "ZDMA EN remained set during cleanup; descriptors, DDR/VPU/DMA mappings, and /dev/mem fd were "
            "intentionally left owned by the process. No undocumented reset/abort write was issued");
    }

    // A normal P3 tile has already proved its mode-retained postcondition.
    // Cleanup is the only permitted reset point: after ZDMA EN is clear, it
    // returns the quiescent stream to P2 mode 0 and reads it back.  If an
    // abnormal P3 termination still owns a bank, fail closed and keep every
    // mapping alive rather than writing mode through a live FIFO.
    if (g_p3_mode_committed && vpu_is_mapped()) {
        if (!fpga_set_split_scale_mode(0U, "cleanup P3 recovery")) {
            pthread_mutex_unlock(&g_mutex);
            fpga_fatal(
                "P3 cleanup recovery could not prove quiescent unlocked mode-0 transition; mappings remain owned "
                "and no unsafe mode reset/unmap was issued");
        }
        g_p3_mode_committed = false;
    }

    // The helper has no DMA ownership, but it can still hold a volatile DDR
    // pointer.  Join it before touching any DDR cache metadata or unmapping
    // the aperture.  A failed join deliberately leaves all mappings intact.
    if (!fpga_p2_pack_worker_stop()) {
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal(
            "P2 WEIGHT-pack helper did not stop cleanly during cleanup; DDR/VPU/DMA mappings were intentionally "
            "left owned by the process and no unmap was issued");
    }

    // Trace records contain only in-process references.  Release them before
    // unmapping DDR/VPU so passive diagnostic cleanup has the same single,
    // idempotent mapping owner as normal initialization.
    g_weight_path_bench.jobs.clear();
    g_weight_path_bench.cached_payload.clear();
    g_weight_path_bench.payload_bytes   = 0U;
    g_weight_path_bench.graph_seq       = INT_MIN;
    g_weight_path_bench.enabled         = false;
    g_weight_path_bench.passive_mapping = false;

    if (ddr_is_mapped()) {
        for (fpga_weight_cache_entry_t & entry : g_weight_cache) {
            if (entry.valid && ddr_range_fits(entry.header_off, sizeof(fpga_weight_cache_header_t))) {
                fpga_weight_cache_header_t header = ddr_read_weight_cache_header(entry.header_off);
                header.valid                      = 0U;
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
    g_ddr          = nullptr;

    if (g_vpu_map_base && g_vpu_map_base != MAP_FAILED) {
        munmap(g_vpu_map_base, g_vpu_map_size);
    }
    g_vpu_map_base = nullptr;
    g_vpu          = nullptr;

    if (g_dma_map_base && g_dma_map_base != MAP_FAILED) {
        munmap(g_dma_map_base, g_dma_map_size);
    }
    g_dma_map_base = nullptr;
    g_dma          = nullptr;

    if (g_mem_fd >= 0) {
        close(g_mem_fd);
        g_mem_fd = -1;
    }
    g_committed_stream_mode = -1;
    g_fpga_init_complete = false;

    // P6 has no safe early failure path here: ZDMA quiescence, the optional
    // P3 mode recovery, helper join, and every mapping/fd release above must
    // complete first.  Only then turn incomplete golden evidence into the
    // requested fail-closed result.  `selected_m1_matmul_seen` distinguishes
    // a run that never reached decode from one that started but did not finish.
    if (fpga_p6_cleanup_requires_failure(fpga_initialized_before_cleanup, g_p6_accum_trace)) {
        const bool selected_m1_matmul_seen = g_p6_accum_trace.selected_m1_matmul_seen;
        const char * const reason = selected_m1_matmul_seen ?
                                       "selected_m1_capture_incomplete" :
                                       "selected_m1_not_seen";
        const int chunks_seen = g_p6_accum_trace.chunks_seen;
        const int expected_chunks = g_p6_accum_trace.expected_k_chunks;
        const size_t records_emitted = g_p6_accum_trace.records_emitted;
        const size_t record_limit = g_p6_accum_trace.record_limit;
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal(
            "P6_ARM_GOLDEN_CAPTURE_REJECT reason=%s selected_m1_seen=%d capture_complete=0 chunks_seen=%d "
            "expected_k_chunks=%d records=%zu record_limit=%zu action=abort_after_safe_teardown",
            reason, selected_m1_matmul_seen ? 1 : 0, chunks_seen, expected_chunks, records_emitted,
            record_limit);
    }

    const long long elapsed_us         = g_fpga_start_us > 0 ? now_us() - g_fpga_start_us : 0;
    const long long hook_calls         = g_matmul_hook_calls.load(std::memory_order_relaxed);
    const long long q8_candidates      = g_q8_candidate_calls.load(std::memory_order_relaxed);
    const long long q8_intentional_cpu = g_q8_intentional_cpu_bypass_calls.load(std::memory_order_relaxed);
    const long long q8_unavailable_cpu = g_q8_unavailable_cpu_fallback_calls.load(std::memory_order_relaxed);
    const long long q8_expected_fpga   = q8_candidates >= q8_intentional_cpu + q8_unavailable_cpu ?
                                             q8_candidates - q8_intentional_cpu - q8_unavailable_cpu :
                                             -1;
    const bool      q8_coverage_complete =
        q8_expected_fpga >= 0 && q8_unavailable_cpu == 0 && g_fpga_count == q8_expected_fpga;
    const char * fpga_block_gemv_mode = q8_candidates == 0                ? "not_applicable" :
                                        g_legacy_raw_cpu_bypass_count > 0 ? "cpu_quarantine" :
                                        q8_coverage_complete              ? "active" :
                                                                            "incomplete";
    const uint64_t p2_residency_budget_bytes = g_p2_weight_residency_budget_mb > 0 ?
                                                  (uint64_t) g_p2_weight_residency_budget_mb * 1024ULL * 1024ULL :
                                                  0ULL;
    const uint64_t p2_residency_allocated_bytes =
        g_p2_residency_next_off >= WEIGHT_CACHE_BASE ? (uint64_t) g_p2_residency_next_off - WEIGHT_CACHE_BASE : 0ULL;
    const uint64_t p2_residency_remaining_bytes = p2_residency_allocated_bytes <= p2_residency_budget_bytes ?
                                                     p2_residency_budget_bytes - p2_residency_allocated_bytes :
                                                     0ULL;
    const double p2_direct_weight_pack_mib_s =
        g_p2_residency_direct_weight_pack_us > 0 ?
            (double) g_p2_residency_direct_weight_pack_bytes * 1000000.0 /
                ((double) g_p2_residency_direct_weight_pack_us * 1024.0 * 1024.0) :
            0.0;
    LOGPROOF(
        "FPGA_GEMV_COVERAGE hook_calls=%lld q8_candidates=%lld q8_expected_fpga=%lld q8_hw_completed=%lld "
        "q8_intentional_cpu_bypass=%lld q8_unavailable_cpu_fallback=%lld routing_verdict=%s fpga_block_gemv=%s",
        hook_calls, q8_candidates, q8_expected_fpga, g_fpga_count, q8_intentional_cpu, q8_unavailable_cpu,
        q8_coverage_complete ? "complete" : "incomplete", fpga_block_gemv_mode);
    LOGPROOF(
        "P2_PACK_SUMMARY requested_workers=%d active_workers=%d helper_stopped=%d parallel_min_bytes=%zu "
        "parallel_jobs=%lld parallel_bytes=%llu serial_threshold_skips=%lld main_pack_us=%lld "
        "helper_service_us=%lld caller_wait_us=%lld",
        g_p2_pack_workers_requested, g_p2_pack_workers_active, g_p2_pack_worker_created ? 0 : 1,
        FPGA_P2_PACK_PARALLEL_MIN_BYTES, g_p2_pack_parallel_jobs, (unsigned long long) g_p2_pack_parallel_bytes,
        g_p2_pack_serial_threshold_skips, g_p2_pack_main_us, g_p2_pack_helper_service_us, g_p2_pack_caller_wait_us);
    LOGPROOF(
        "P3_SPLIT_SCALE_SUMMARY requested=%d active=%d mode_committed=%d jobs=%lld param_dma_bytes=%lld "
        "scratch_dma_bytes=%lld retire_pass_logs=%lld retire_pass_suppressed=%lld "
        "retire_log_interval=%lld scheduler=serialized scale_preload_overlap=0",
        g_p3_split_scale_requested ? 1 : 0, g_p3_split_scale_active ? 1 : 0, g_p3_mode_committed ? 1 : 0,
        g_p3_jobs, g_p3_param_dma_bytes, g_p3_scratch_dma_bytes, g_p3_retire_pass_logs,
        g_p3_retire_pass_suppressed, P3_PRODUCTION_RETIRE_LOG_INTERVAL);
    if (g_p3_retire_timing_enabled) {
        const uint64_t retire_timing_core_avg_ns =
            g_p3_retire_timing_valid_samples == 0U ? 0U :
                                                       g_p3_retire_timing_core_total_ns /
                                                           g_p3_retire_timing_valid_samples;
        LOGPROOF(
            "P3_RETIRE_TIMING_SUMMARY enabled=1 scope=host_monotonic_elapsed_before_nine_retirement_mmio_reads_through_exact_predicate "
            "not_device_internal_latency calls=%llu passes=%llu failures=%llu valid_samples=%llu clock_errors=%llu "
            "mmio_reads=%llu expected_mmio_reads_per_call=9 core_total_ns=%llu core_min_ns=%llu core_max_ns=%llu core_avg_ns=%llu",
            (unsigned long long) g_p3_retire_timing_calls, (unsigned long long) g_p3_retire_timing_passes,
            (unsigned long long) g_p3_retire_timing_failures, (unsigned long long) g_p3_retire_timing_valid_samples,
            (unsigned long long) g_p3_retire_timing_clock_errors, (unsigned long long) g_p3_retire_timing_mmio_reads,
            (unsigned long long) g_p3_retire_timing_core_total_ns, (unsigned long long) g_p3_retire_timing_core_min_ns,
            (unsigned long long) g_p3_retire_timing_core_max_ns, (unsigned long long) retire_timing_core_avg_ns);
    }
    if (fpga_p2_residency_has_reportable_activity()) {
        LOGPROOF(
            "P2_RESIDENCY_SUMMARY forced=1 enabled=%d diagnostic=%d trace=%d verify_metadata=%d slots=%zu/%zu index_buckets=%zu max_probes=%zu probes=%lld "
            "probe_exhausted_direct_stage=%lld hits=%lld misses=%lld host_metadata_hits=%lld host_metadata_invalidations=%lld "
            "volatile_ddr_reads=%lld build_us=%lld select_us=%lld metadata_validate_us=%lld resident_param_us=%lld "
            "direct_weight_pack_us=%lld direct_weight_pack_bytes=%llu direct_weight_pack_MiB_s=%.3f "
            "avoided_cpu_pack_bytes=%lld avoided_ddr_to_ip_bytes=%lld "
            "miss_alignment=%lld miss_shape=%lld miss_collision=%lld miss_poison=%lld miss_stale=%lld miss_mismatch=%lld "
            "miss_capacity=%lld miss_quiescence=%lld miss_range=%lld miss_verify=%lld builds=%lld build_failures=%lld logical_bytes=%lld "
            "allocated_bytes=%llu remaining_bytes=%llu budget_bytes=%llu",
            g_p2_weight_residency_enabled ? 1 : 0, g_p2_weight_residency_diagnostic ? 1 : 0,
            g_p2_residency_trace_enabled ? 1 : 0, g_p2_residency_verify_metadata ? 1 : 0, g_p2_resident_tile_count,
            g_p2_resident_tiles.size(), g_p2_residency_index.size(), P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES,
            g_p2_residency_probe_count, g_p2_residency_probe_exhausted, g_p2_residency_hits, g_p2_residency_misses,
            g_p2_residency_host_metadata_hits, g_p2_residency_host_metadata_invalidations,
            g_p2_residency_volatile_ddr_reads, g_p2_residency_build_us, g_p2_residency_select_us,
            g_p2_residency_metadata_validate_us, g_p2_residency_resident_param_us,
            g_p2_residency_direct_weight_pack_us, (unsigned long long) g_p2_residency_direct_weight_pack_bytes,
            p2_direct_weight_pack_mib_s, g_p2_residency_avoided_cpu_pack_bytes,
            g_p2_residency_avoided_ddr_to_ip_bytes,
            g_p2_residency_miss_alignment, g_p2_residency_miss_shape, g_p2_residency_miss_collision,
            g_p2_residency_miss_poison, g_p2_residency_miss_stale, g_p2_residency_miss_mismatch,
            g_p2_residency_miss_capacity, g_p2_residency_miss_quiescence, g_p2_residency_miss_range,
            g_p2_residency_miss_verify,
            g_p2_residency_builds, g_p2_residency_build_failures, g_p2_residency_logical_bytes,
            (unsigned long long) p2_residency_allocated_bytes, (unsigned long long) p2_residency_remaining_bytes,
            (unsigned long long) p2_residency_budget_bytes);
    }
    LOGPROOF(
        "cleanup complete fpga_calls=%lld vpu_runs=%lld rejects=%lld attention_cpu_bypass=%lld "
        "vocab_projection_cpu_bypass=%lld legacy_raw_cpu_bypass=%lld elapsed_s=%.3f pingpong_cap=%d descriptor_cap=%d "
        "scheduler=%d activation_cache_enabled=%d activation_cache_hits=%lld misses=%lld weight_cache_builds=%lld "
        "hits=%lld misses=%lld bytes=%lld cache_lookup_ms=%.3f cache_crc_ms=%.3f weight_pack_ms=%.3f "
        "activation_scale_fp16_overflows=%lld input_integrity_checks=%lld input_integrity_failures=%lld "
        "contract_checks=%lld contract_limit_cpu_bypass=%lld raw_mismatches=%lld raw_repairs=%lld "
        "value_mismatches=%lld contract_cpu_shadow_dst_values=%lld staging_restages=%lld q8_source_audit_checks=%lld "
        "q8_source_audit_failures=%lld p2_matrix_contract_checks=%lld p2_tile_q16_checks=%lld p2_tile_limit=%d "
        "p2_tile_boundary=%d matrix_value_contract=%s p2_contract_jobs=%lld p2_contract_banks=%lld p2_stream_drops=%lld "
        "p2_stream_errors=%lld p2_residency_enabled=%d p2_residency_slots=%zu/%zu p2_residency_builds=%lld "
        "p2_residency_hits=%lld p2_residency_direct_stage_misses=%lld p2_residency_build_failures=%lld "
        "p2_residency_logical_bytes=%lld p2_residency_allocated_bytes=%llu p2_residency_remaining_bytes=%llu "
        "p2_direct_weight_pack_bytes=%llu p2_direct_weight_pack_ms=%.3f p2_direct_weight_pack_MiB_s=%.3f",
        g_fpga_count, g_fpga_vpu_runs, g_reject_count, g_attention_bypass_count, g_vocab_projection_bypass_count,
        g_legacy_raw_cpu_bypass_count, elapsed_us > 0 ? (double) elapsed_us / 1000000.0 : 0.0,
        g_vpu_pingpong_supported ? 1 : 0, g_vpu_descriptor_supported ? 1 : 0, g_pingpong_scheduler_enabled ? 1 : 0,
        g_activation_cache_enabled ? 1 : 0, g_activation_cache_hits, g_activation_cache_misses, g_weight_cache_builds,
        g_weight_cache_hits, g_weight_cache_misses, g_weight_cache_bytes, (double) g_weight_cache_lookup_us / 1000.0,
        (double) g_weight_cache_crc_us / 1000.0, (double) g_weight_pack_us / 1000.0, g_activation_scale_fp16_overflows,
        g_activation_input_integrity_checks, g_activation_input_integrity_failures, g_contract_checks_done,
        g_contract_limit_cpu_bypass_count, g_contract_raw_mismatches, g_contract_raw_repairs,
        g_contract_value_mismatches, g_contract_cpu_shadow_dst_values, g_contract_staging_restage_count,
        g_q8_source_audit_checks, g_q8_source_audit_failures, g_p2_matrix_contract_checks, g_p2_tile_q16_checks,
        g_p2_tile_limit, g_p2_tile_contract_boundary_reached ? 1 : 0,
        g_pl_scale_contract_check_limit > 0 ? "not_attempted" : "not_applicable", g_pl_scale_jobs, g_pl_scale_banks,
        g_pl_scale_stream_drops, g_pl_scale_stream_errors, g_p2_weight_residency_enabled ? 1 : 0,
        g_p2_resident_tile_count, g_p2_resident_tiles.size(), g_p2_residency_builds, g_p2_residency_hits,
        g_p2_residency_misses, g_p2_residency_build_failures, g_p2_residency_logical_bytes,
        (unsigned long long) p2_residency_allocated_bytes, (unsigned long long) p2_residency_remaining_bytes,
        (unsigned long long) g_p2_residency_direct_weight_pack_bytes,
        (double) g_p2_residency_direct_weight_pack_us / 1000.0, p2_direct_weight_pack_mib_s);
    fflush(fpga_log_fp());
    pthread_mutex_unlock(&g_mutex);
}


void fpga_set_context(int layer_id, int seq_pos, int is_attn) {
    g_current_layer_id = layer_id;
    g_current_seq_pos  = seq_pos;
    g_is_attention_op  = is_attn;
}

extern "C" int fpga_get_sequence_position(void) {
    return g_current_seq_pos;
}

extern "C" int fpga_perf_decode_get(fpga_perf_decode_data * data) {
    if (!data) {
        return 0;
    }

    *data = g_fpga_perf_decode;
    data->run_fpga_gemvs = g_fpga_count;
    data->run_q8_unavailable_cpu_fallbacks = g_q8_unavailable_cpu_fallback_calls.load(std::memory_order_relaxed);
    data->run_rejects = g_reject_count;
    data->run_stream_drops = g_pl_scale_stream_drops;
    data->run_stream_errors = g_pl_scale_stream_errors;
    data->residency_slots_used = (int64_t) g_p2_resident_tile_count;
    data->residency_slots_total = (int64_t) g_p2_resident_tiles.size();
    data->residency_hits = g_p2_residency_hits;
    data->residency_misses = g_p2_residency_misses;
    return data->decode_tokens > 0 &&
           (data->ip_compute_us + data->h2ip_dma_us + data->output_transfer_us) > 0 ? 1 : 0;
}

extern "C" void fpga_advance_sequence_position(int n_tokens) {
    if (n_tokens < 0 || g_current_seq_pos > INT_MAX - n_tokens) {
        fpga_fatal("invalid FPGA sequence advance current=%d delta=%d", g_current_seq_pos, n_tokens);
    }

    const int previous_graph_seq = g_current_seq_pos;
    if (g_weight_path_bench.enabled) {
        pthread_mutex_lock(&g_mutex);
        bool benchmark_ok = true;
        if (fpga_weight_path_bench_reset_required(g_weight_path_bench, previous_graph_seq, n_tokens)) {
            fpga_weight_path_bench_reset_unreplayed_trace(&g_weight_path_bench, previous_graph_seq + n_tokens,
                                                          "sequence_advanced_before_single_token_boundary", true);
        } else if (!g_weight_path_bench.replayed && n_tokens == 1 &&
                   g_weight_path_bench.graph_seq == previous_graph_seq) {
            benchmark_ok = fpga_weight_path_bench_replay_at_boundary(previous_graph_seq, n_tokens);
        }
        pthread_mutex_unlock(&g_mutex);
        if (!benchmark_ok) {
            fpga_fatal("WEIGHT_PATH_BENCH_REPLAY_FAIL graph_seq=%d action=abort_after_bounded_ddr_weight_staging "
                       "zdma=not_mapped vpu_start=not_run dst_access=none",
                       previous_graph_seq);
        }
    }
    if (g_token_timing_collection_enabled && g_token_timing.active && g_token_timing.graph_seq == previous_graph_seq) {
        fpga_token_timing_emit(previous_graph_seq + n_tokens, n_tokens, "sequence_advance", monotonic_now_us());
    }
    if (g_p2_event_trace_enabled && g_last_token_seq == previous_graph_seq && g_last_token_us > 0) {
        const long long graph_eval_done_us = p2_event_now_us();
        fpga_log_line(
            true, "P2_GRAPH_EVAL_DONE", false,
            "prev_graph_seq=%d new_graph_seq=%d ubatch_tokens=%d first_fpga_hook_to_graph_eval_done_us=%lld "
            "decode_step=%d semantics=graph_evaluation_completion_not_generated_token",
            previous_graph_seq, previous_graph_seq + n_tokens, n_tokens, graph_eval_done_us - g_last_token_us,
            n_tokens == 1 && g_last_epoch_first_hook_m == 1 ? 1 : 0);
    }
    g_current_seq_pos += n_tokens;
    fpga_log_line(g_p2_event_trace_enabled, "P2_SEQ_ADVANCE", false,
                  "prev_graph_seq=%d delta=%d graph_seq=%d semantics=graph_sequence_position", previous_graph_seq,
                  n_tokens, g_current_seq_pos);
}

extern "C" int fpga_try_matmul(const struct ggml_tensor * src0,
                               const struct ggml_tensor * src1,
                               const struct ggml_tensor * dst,
                               int                        ith) {
    return fpga_try_matmul_extended(src0, src1, dst, ith, 0, g_current_seq_pos, 0);
}

static void log_sequence_epoch_close_if_needed(int seq_pos, int64_t m);

static bool should_bypass_vocab_projection_to_cpu(const char * tensor_name, int64_t k, int64_t n, int64_t m) {
    (void) tensor_name;
    return g_vocab_projection_cpu_bypass && m == 1 && k > 0 && n >= g_vocab_projection_min_n;
}

extern "C" int fpga_try_matmul_extended(const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src1,
                                        const struct ggml_tensor * dst,
                                        int                        ith,
                                        int                        layer_id,
                                        int                        seq_pos,
                                        int                        is_attention) {
    const char *  tensor_name        = tensor_name_or_unknown(src0);
    const int     effective_layer_id = infer_layer_id_from_name(tensor_name, layer_id);
    const int64_t probe_k            = src0 ? src0->ne[0] : 0;
    const int64_t probe_n            = src0 ? src0->ne[1] : 0;
    const int64_t probe_m            = src1 ? src1->ne[1] : 0;
    // Resolve the P6 selector before any native-CPU route.  The selected
    // warm-up M>1 call is deferred; the selected decode M=1 call must fail
    // closed rather than being hidden by an audit, attention, vocabulary,
    // legacy, or validation bypass.
    const bool p6_selected_name_layer = g_p6_accum_trace.enabled && tensor_name &&
                                        g_p6_accum_trace.tensor == tensor_name &&
                                        g_p6_accum_trace.layer == effective_layer_id;
    const bool p6_selected_m1 = fpga_p6_selected_m1_eligible(p6_selected_name_layer, probe_m);
    if (p6_selected_name_layer && !p6_selected_m1) {
        // Gemma's empty/warm-up graph legitimately evaluates the selected
        // tensor with M>1.  That is not the requested decode measurement, so
        // leave it entirely to native GGML before validation, counters,
        // staging, DMA/VPU submission, or destination ownership can change.
        pthread_mutex_lock(&g_mutex);
        if (!g_p6_accum_trace.selected_non_m1_defer_logged) {
            LOGPROOF(
                "P6_ARM_GOLDEN_DEFER tensor=%s layer=%d M=%lld reason=selector_qualifies_decode_M1_only "
                "action=native_cpu_before_validation_accounting_staging_hardware_dst",
                tensor_name, effective_layer_id, (long long) probe_m);
            g_p6_accum_trace.selected_non_m1_defer_logged = true;
        }
        pthread_mutex_unlock(&g_mutex);
        return FPGA_MATMUL_NOT_HANDLED;
    }
    if (p6_selected_m1) {
        // This state is intentionally set before the M=1 fail-closed checks
        // below.  If any such check rejects, cleanup can distinguish it from
        // a graph that never reached the requested decode call at all.
        pthread_mutex_lock(&g_mutex);
        g_p6_accum_trace.selected_m1_matmul_seen = true;
        pthread_mutex_unlock(&g_mutex);
    }
    if (p6_selected_m1 && is_attention) {
        fpga_fatal(
            "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=selected_target_is_attention "
            "action=abort_no_hardware",
            tensor_name, effective_layer_id, (long long) probe_m);
    }

    // This path deliberately runs before every board-facing policy branch.
    // It must validate the same Q8 tensors the normal graph would consume,
    // including the vocabulary projection, while leaving all work to CPU.
    const bool source_audit_only = fpga_source_audit_only_requested() != 0;
    if (source_audit_only) {
        if (p6_selected_m1) {
            fpga_fatal(
                "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=source_audit_cpu_route "
                "action=abort_no_hardware",
                tensor_name, effective_layer_id, (long long) probe_m);
        }
        if (env_int_value("FPGA_CONTRACT_CHECK", 0, 0, 1000000) > 0 ||
            env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000) > 0) {
            fpga_fatal(
                "FPGA_SOURCE_AUDIT_ONLY cannot coexist with FPGA_CONTRACT_CHECK or FPGA_PL_SCALE_CONTRACT_CHECK; "
                "source audit must not launch model GEMVs through ZDMA/VPU");
        }
        if (ith != 0) {
            return 0;
        }
        const char * reason = nullptr;
        if (!fpga_validate_tensors(src0, src1, dst, false, &reason)) {
            LOGI("Q8_SOURCE_AUDIT_SKIP tensor=%s layer=%d reason=%s", tensor_name, effective_layer_id,
                 reason ? reason : "unsupported tensor");
            return 0;
        }
        pthread_mutex_lock(&g_mutex);
        if (!g_q8_source_audit_mode_logged) {
            g_q8_source_audit_only = true;
            g_log_flush_every      = env_int_value("FPGA_LOG_FLUSH_EVERY", 256, 1, 1000000);
            fpga_log_set_flush_every(g_log_flush_every);
            LOGI(
                "Q8_SOURCE_AUDIT_MODE version=%s host_hardware_init=skipped board_mmio=not_mapped "
                "zdma_selftests=not_run action=validate_q8_source_then_cpu_matmul",
                FPGA_HOST_TRACE_VERSION);
            g_q8_source_audit_mode_logged = true;
        }
        const bool source_ok = fpga_audit_q8_source_only(src0, tensor_name, effective_layer_id);
        pthread_mutex_unlock(&g_mutex);
        if (!source_ok) {
            fpga_fatal(
                "Q8 source audit failed tensor=%s layer=%d; refusing to continue with a numerically invalid source "
                "tensor",
                tensor_name, effective_layer_id);
        }
        return 0;
    }

    if (is_attention) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_attention_bypass_count++;
            if (g_attention_bypass_count == 1) {
                LOGI(
                    "attention path is currently bypassed to CPU; FPGA timing log below only covers Q8_0 matmul/GEMV "
                    "hooks");
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return 0;
    }

    // Only count the single thread that owns this matmul result.  The GGML
    // CPU scheduler may call the hook from multiple workers, but non-zero
    // workers return success below without launching their own VPU job.
    const bool q8_f32_f32_candidate = src0 && src1 && dst && src0->type == GGML_TYPE_Q8_0 &&
                                      src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    if (g_p6_accum_trace.enabled) {
        if (p6_selected_m1 && !q8_f32_f32_candidate) {
            fpga_fatal(
                "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=selected_target_requires_Q8_0_F32_F32 "
                "action=abort_no_hardware",
                tensor_name, effective_layer_id, (long long) probe_m);
        }
        // P6 has one explicit hardware target.  Every other Q8 candidate is
        // intentionally returned to native GGML before validation, staging,
        // fallback accounting, or FPGA-completion accounting.
        if (!p6_selected_m1) {
            return FPGA_MATMUL_NOT_HANDLED;
        }
        // Other GGML workers can encounter the already-completed selected
        // matrix.  Preserve its CPU-shadow ownership explicitly; do not let
        // it re-enter a policy branch that could be mistaken for a fallback.
        if (g_p6_accum_trace.capture_complete) {
            return FPGA_MATMUL_CONTRACT_CPU_SHADOW;
        }
    }
    if (ith == 0) {
        g_matmul_hook_calls.fetch_add(1, std::memory_order_relaxed);
        if (q8_f32_f32_candidate) {
            g_q8_candidate_calls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // This explicit diagnostic route is not an FPGA-unavailability fallback:
    // it captures the same geometry the admitted normal tiler would use, then
    // deliberately leaves dst to native GGML.  It has no prepare/submit/DMA/
    // VPU operation and never observes or writes destination storage.
    if (g_weight_path_bench.enabled) {
        if (p6_selected_m1) {
            fpga_fatal(
                "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=weight_path_benchmark_cpu_route "
                "action=abort_no_hardware",
                tensor_name, effective_layer_id, (long long) probe_m);
        }
        if (ith != 0 || g_weight_path_bench.replayed) {
            return FPGA_MATMUL_NOT_HANDLED;
        }
        if (q8_f32_f32_candidate && probe_m == 1 &&
            !should_bypass_vocab_projection_to_cpu(tensor_name, probe_k, probe_n, probe_m)) {
            const char * bench_reason = nullptr;
            if (fpga_validate_tensors(src0, src1, dst, true, &bench_reason)) {
                pthread_mutex_lock(&g_mutex);
                const char * capture_reason = nullptr;
                const bool captured = fpga_weight_path_bench_capture(src0, seq_pos, &capture_reason);
                const size_t jobs = g_weight_path_bench.jobs.size();
                const uint64_t bytes = g_weight_path_bench.payload_bytes;
                pthread_mutex_unlock(&g_mutex);
                if (!captured) {
                    fpga_fatal(
                        "WEIGHT_PATH_BENCH_CAPTURE_FAIL reason=%s tensor=%s layer=%d seq=%d jobs=%zu bytes=%llu "
                        "action=abort_no_ddr_write_no_zdma_no_vpu_start",
                        capture_reason ? capture_reason : "state_unknown", tensor_name, effective_layer_id, seq_pos,
                        jobs, (unsigned long long) bytes);
                }
            } else {
                LOGI("WEIGHT_PATH_BENCH_SKIP tensor=%s layer=%d seq=%d reason=%s route=native_cpu",
                     tensor_name, effective_layer_id, seq_pos, bench_reason ? bench_reason : "unsupported_tensor");
            }
        }
        return FPGA_MATMUL_NOT_HANDLED;
    }

    // A qualification limit must stop all board launches once it is reached.
    // This applies to both raw C0 and P2; later eligible GEMVs remain native
    // CPU work and are not an FPGA-unavailability fallback.
    // `FPGA_CONTRACT_CHECK=N` used to stop only the comparison loop. The host
    // still launched later raw VPU jobs, but returned CPU shadow for their
    // destinations. That is neither a complete hardware qualification nor a
    // CPU baseline, and it obscures the first source of a later non-finite
    // activation. Make N a strict qualification boundary: the first N eligible
    // GEMVs execute, stage, and validate on FPGA; all later eligible GEMVs use
    // the upstream CPU kernel without being counted as FPGA unavailability or
    // a production fallback.
    const bool raw_contract_boundary =
        g_contract_check_limit > 0 && g_contract_checks_done >= (long long) g_contract_check_limit;
    const bool p2_contract_boundary =
        g_pl_scale_contract_check_limit > 0 &&
        (g_p2_tile_contract_boundary_reached ||
         fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit));
    if (p6_selected_m1 && (raw_contract_boundary || p2_contract_boundary)) {
        fpga_fatal(
            "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=qualification_boundary_before_selected "
            "action=abort_no_hardware",
            tensor_name, effective_layer_id, (long long) probe_m);
    }
    if (q8_f32_f32_candidate && g_contract_cpu_shadow_dst && (raw_contract_boundary || p2_contract_boundary)) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_contract_limit_cpu_bypass_count++;
            if (!g_contract_limit_cpu_bypass_logged) {
                LOGI(
                    "qualification boundary route=cpu_native mode=%s checked=%lld limit=%d p2_tile_q16_checks=%lld "
                    "p2_matrix_contract_checks=%lld matrix_value_contract=%s hw_completed=%lld "
                    "input_integrity_failures=%lld contract_limit_cpu_bypass=%lld; subsequent eligible Q8 GEMVs use "
                    "the native CPU kernel without ZDMA/VPU launch. This is an explicit bounded qualification policy, "
                    "not FPGA unavailability or a production fallback.",
                    p2_contract_boundary ? "p2_spu_scale" : "raw_c0",
                    p2_contract_boundary ? g_p2_tile_q16_checks : g_contract_checks_done,
                    p2_contract_boundary ? g_p2_tile_limit : g_contract_check_limit, g_p2_tile_q16_checks,
                    g_p2_matrix_contract_checks, p2_contract_boundary ? "not_attempted" : "n/a", g_fpga_count,
                    g_activation_input_integrity_failures, g_contract_limit_cpu_bypass_count);
                g_contract_limit_cpu_bypass_logged = true;
                // C0 runs are often terminated at the first anomaly.  Make
                // the one-time transition evidence durable immediately rather
                // than relying on normal cleanup after the CPU continuation.
                fflush(fpga_log_fp());
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return FPGA_MATMUL_NOT_HANDLED;
    }

    if (!p6_selected_m1 && should_bypass_vocab_projection_to_cpu(tensor_name, probe_k, probe_n, probe_m)) {
        if (ith == 0) {
            if (q8_f32_f32_candidate) {
                g_q8_intentional_cpu_bypass_calls.fetch_add(1, std::memory_order_relaxed);
            }
            pthread_mutex_lock(&g_mutex);
            g_vocab_projection_bypass_count++;
            if (g_vocab_projection_bypass_count == 1) {
                LOGI(
                    "vocab projection bypassed to CPU tensor=%s shape=K%lld_N%lld_M%lld threshold_n=%lld; unset "
                    "FPGA_VOCAB_PROJECTION_CPU or set FPGA_ACCELERATE_VOCAB=1 to force FPGA",
                    tensor_name, (long long) probe_k, (long long) probe_n, (long long) probe_m,
                    (long long) g_vocab_projection_min_n);
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return 0;
    }

    if (!p6_selected_m1 && g_legacy_raw_cpu_bypass) {
        if (ith == 0) {
            if (q8_f32_f32_candidate) {
                g_q8_intentional_cpu_bypass_calls.fetch_add(1, std::memory_order_relaxed);
            }
            pthread_mutex_lock(&g_mutex);
            g_legacy_raw_cpu_bypass_count++;
            if (g_legacy_raw_cpu_bypass_count == 1) {
                LOGI(
                    "legacy raw FPGA path bypassed to CPU tensor=%s shape=K%lld_N%lld_M%lld; this preserves language "
                    "correctness while contract diagnostics repair the unverified bitstream/transfer path",
                    tensor_name, (long long) probe_k, (long long) probe_n, (long long) probe_m);
            }
            pthread_mutex_unlock(&g_mutex);
        }
        return 0;
    }

    const char * reason = nullptr;
    if (!fpga_validate_tensors(src0, src1, dst, true, &reason)) {
        if (p6_selected_m1) {
            fpga_fatal(
                "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=selected_target_validation_%s "
                "action=abort_no_hardware",
                tensor_name, effective_layer_id, (long long) probe_m, reason ? reason : "unknown");
        }
        if (ith == 0) {
            if (q8_f32_f32_candidate) {
                g_q8_unavailable_cpu_fallback_calls.fetch_add(1, std::memory_order_relaxed);
            }
            const int64_t k = src0 ? src0->ne[0] : 0;
            const int64_t n = src0 ? src0->ne[1] : 0;
            const int64_t m = src1 ? src1->ne[1] : 0;
            pthread_mutex_lock(&g_mutex);
            g_reject_count++;
            LOGE("matmul rejected tensor=%s layer=%d shape=K%lld_N%lld_M%lld reason=%s action=%s", tensor_name,
                 effective_layer_id, (long long) k, (long long) n, (long long) m, reason ? reason : "unknown",
                 g_abort_on_cpu_fallback ? "abort_no_cpu_fallback" : "return_to_cpu");
            pthread_mutex_unlock(&g_mutex);
            if (g_abort_on_cpu_fallback) {
                fpga_fatal("CPU fallback matmul blocked tensor=%s layer=%d reason=%s", tensor_name, effective_layer_id,
                           reason ? reason : "unknown");
            }
        }
        return 0;
    }

    if (!dma_is_mapped() || !vpu_is_mapped() || !ddr_is_mapped()) {
        if (p6_selected_m1) {
            fpga_fatal(
                "P6_ARM_GOLDEN_CAPTURE_REJECT tensor=%s layer=%d M=%lld reason=required_mapping_unavailable "
                "action=abort_no_hardware",
                tensor_name, effective_layer_id, (long long) probe_m);
        }
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
        return g_contract_cpu_shadow_dst ? FPGA_MATMUL_CONTRACT_CPU_SHADOW : FPGA_MATMUL_FPGA_DST;
    }

    pthread_mutex_lock(&g_mutex);

    const int64_t       k                       = src0->ne[0];
    const int64_t       n                       = src0->ne[1];
    const int64_t       m                       = src1->ne[1];
    log_sequence_epoch_close_if_needed(seq_pos, m);
    fpga_token_timing_begin(seq_pos, m);
    const int64_t       q8_blocks               = k / VPU_QK8_0;
    const long long     macs                    = matrix_mac_count(k, n, m);
    const long long     row_tiles               = (n + g_vpu_max_rows - 1) / g_vpu_max_rows;
    const int           first_tile_rows         = (int) std::min<int64_t>(n, g_vpu_max_rows);
    const int           max_group_blocks        = packed_q8_group_blocks_for_rows(first_tile_rows, (int) q8_blocks);
    const long long     estimated_runs          = estimate_vpu_runs(k, n, m);
    fpga_stage_totals_t totals                  = {};
    const bool          verify_activation_input = g_activation_input_integrity_check;
    if (g_next_matmul_call_id == 0U) {
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal("P2 matmul_call_id exhausted; refusing to reuse a telemetry identity");
    }
    g_active_matmul_call_id    = g_next_matmul_call_id++;
    g_active_matmul_graph_seq  = seq_pos;
    g_active_matmul_layer_id   = effective_layer_id;
    g_active_matmul_shape_k    = k;
    g_active_matmul_shape_n    = n;
    g_active_matmul_shape_m    = m;
    g_active_matmul_cpu_shadow = g_contract_cpu_shadow_dst;
    g_active_matmul_pingpong   = g_pingpong_scheduler_enabled;
    g_active_matmul_tensor_name = tensor_name;
    fpga_p1_sched_summary_begin_graph(g_active_matmul_graph_seq);
    if (verify_activation_input) {
        if (!fpga_capture_activation_input_snapshot(src1, k, m, g_scratch.activation_input_snapshot)) {
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
        LOGSTAGE(
            "start tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld path=zdma_ddr_to_ip row_tiles=%lld "
            "group_tiles_per_rowtile~=%lld q8_blocks=%lld max_group_blocks=%d vpu_runs=%lld",
            tensor_name, effective_layer_id, seq_pos, decode_or_prefill(m), (long long) k, (long long) n, (long long) m,
            row_tiles, (q8_blocks + max_group_blocks - 1) / max_group_blocks, (long long) q8_blocks, max_group_blocks,
            estimated_runs);
    }

    const long long event_matmul_start_us = p2_event_now_us();
    bool hw_ok = fpga_hw_q8_0_matmul_dma_to_ip(src0, src1, dst, &totals, tensor_name, effective_layer_id);
    if (hw_ok && verify_activation_input &&
        !fpga_verify_activation_input_snapshot(src1, dst, k, m, g_scratch.activation_input_snapshot, tensor_name,
                                               effective_layer_id)) {
        g_activation_input_integrity_failures++;
        hw_ok = false;
    }
    const long long event_matmul_done_us = hw_ok ? p2_event_now_us() : 0;
    const long long t1 = now_us();
    if (!hw_ok) {
        const bool source_validation_failed = g_contract_source_validation_failed;
        pthread_mutex_unlock(&g_mutex);
        if (source_validation_failed) {
            fpga_fatal(
                "C0 source validation failed before any model ZDMA/VPU launch for tensor=%s layer=%d; the active GGUF "
                "contains a non-finite Q8_0 scale. Replace or re-copy the model, then rerun C0. No CPU fallback or "
                "scale repair was applied",
                tensor_name, effective_layer_id);
        }
        fpga_fatal("ZDMA-to-IP/VPU matmul failed tensor=%s layer=%d shape=K%lld_N%lld_M%lld; refusing CPU fallback",
                   tensor_name, effective_layer_id, (long long) k, (long long) n, (long long) m);
    }

    p2_matmul_finish_trace(event_matmul_done_us, event_matmul_start_us, g_p2_tile_contract_boundary_reached);
    if (g_p2_tile_contract_boundary_reached) {
        // A P2 tile contract deliberately does not own a complete dst matrix.
        // The GGML kernel must compute the current MUL_MAT natively, while the
        // boundary above keeps every following eligible GEMV off the board.
        pthread_mutex_unlock(&g_mutex);
        return FPGA_MATMUL_CONTRACT_CPU_SHADOW;
    }

    g_fpga_count++;
    g_token_matmuls++;
    g_fpga_vpu_runs += totals.vpu_runs;
    g_weight_cache_hits += totals.weight_cache_hits;
    g_weight_cache_misses += totals.weight_cache_misses;
    if (g_p1_sched_summary_enabled) {
        g_p1_sched_summary.matmuls++;
        g_p1_sched_summary.vpu_runs += totals.vpu_runs;
        g_p1_sched_summary.input_preload_us += totals.input_preload_us;
        g_p1_sched_summary.preload_launch_bubble_us += totals.preload_launch_bubble_us;
        g_p1_sched_summary.ip_compute_us += totals.ip_compute_us;
        g_p1_sched_summary.dma_act_us += totals.dma_act_us;
        g_p1_sched_summary.dma_weight_us += totals.dma_weight_us;
        g_p1_sched_summary.matrix_wall_us += t1 - t0;
    }
    fpga_token_timing_accumulate(totals, t1 - t0, tensor_name);
    const double total_ms        = (double) (t1 - t0) / 1000.0;
    const double prep_ms         = (double) totals.prep_us / 1000.0;
    const double dma_act_ms      = (double) totals.dma_act_us / 1000.0;
    const double dma_weight_ms   = (double) totals.dma_weight_us / 1000.0;
    const double dma_scale_ms    = (double) totals.dma_scale_us / 1000.0;
    const double dma_in_ms       = dma_act_ms + dma_weight_ms + dma_scale_ms;
    const double ip_ms           = (double) totals.ip_compute_us / 1000.0;
    const double dma_out_ms      = (double) totals.dma_result_us / 1000.0;
    const double host_result_ms  = (double) totals.host_result_us / 1000.0;
    const double host_accum_ms   = (double) totals.host_accum_us / 1000.0;
    const double cache_lookup_ms = (double) totals.weight_cache_lookup_us / 1000.0;
    const double cache_crc_ms    = (double) totals.weight_cache_crc_us / 1000.0;
    const double weight_pack_ms  = (double) totals.weight_pack_us / 1000.0;
    const double preload_ms      = (double) totals.input_preload_us / 1000.0;
    const double preload_bubble_ms = (double) totals.preload_launch_bubble_us / 1000.0;

    const char * dominant    = "prep";
    double       dominant_ms = prep_ms;
    if (dma_in_ms > dominant_ms) {
        dominant    = "dma_input";
        dominant_ms = dma_in_ms;
    }
    if (ip_ms > dominant_ms) {
        dominant    = "ip_compute";
        dominant_ms = ip_ms;
    }
    if (dma_out_ms > dominant_ms) {
        dominant    = "dma_output";
        dominant_ms = dma_out_ms;
    }
    if (host_result_ms > dominant_ms) {
        dominant    = g_spu_q8_scale_stream_supported ? "host_scaled_row_read" : "host_result_unpack";
        dominant_ms = host_result_ms;
    }
    if (host_accum_ms > dominant_ms) {
        dominant    = "host_accum";
        dominant_ms = host_accum_ms;
    }

    const size_t effective_bytes =
        totals.activation_bytes + totals.weight_bytes + totals.scale_bytes + totals.result_bytes;
    const double gmac_s = total_ms > 0.0 ? (double) macs / (total_ms * 1000000.0) : 0.0;
    const double mib_s  = total_ms > 0.0 ? (double) effective_bytes * 1000.0 / (total_ms * 1024.0 * 1024.0) : 0.0;
    const double cycles_per_run = (g_fpga_clock_mhz > 0.0 && totals.vpu_runs > 0) ?
                                      ((double) totals.ip_compute_us * g_fpga_clock_mhz / (double) totals.vpu_runs) :
                                      0.0;

    if (g_profile_every > 0 && (g_fpga_count == 1 || (g_fpga_count % g_profile_every) == 0)) {
        LOGSTAGE(
            "tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld row_tiles=%lld group_tiles=%lld q8_blocks=%lld "
            "vpu_runs=%lld prep_ms=%.3f cache_lookup_ms=%.3f cache_crc_ms=%.3f weight_pack_ms=%.3f "
            "activation_scale_fp16_overflows=%lld dma_input_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f "
            "scale_dma_ms=%.3f ip_compute_ms=%.3f dma_output_ms=%.3f host_result_ms=%.3f host_accum_ms=%.3f "
            "p1_input_preload=%d p1_preload_jobs=%lld p1_preload_ms=%.3f p1_free_to_start_ms=%.3f "
            "total_ms=%.3f dominant=%s pl_scale=%d raw_accum_fused=%d effective_GMAC/s=%.3f effective_MiB/s=%.1f "
            "act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu weight_cache_hits=%lld "
            "weight_cache_misses=%lld cycles_per_run=%.1f",
            tensor_name, effective_layer_id, seq_pos, decode_or_prefill(m), (long long) k, (long long) n, (long long) m,
            row_tiles, (q8_blocks + max_group_blocks - 1) / max_group_blocks, (long long) q8_blocks, totals.vpu_runs,
            prep_ms, cache_lookup_ms, cache_crc_ms, weight_pack_ms, totals.activation_scale_fp16_overflows, dma_in_ms,
            dma_act_ms, dma_weight_ms, dma_scale_ms, ip_ms, dma_out_ms, host_result_ms, host_accum_ms,
            g_p2_input_preload_enabled ? 1 : 0, totals.input_preload_jobs, preload_ms, preload_bubble_ms, total_ms,
            dominant, g_spu_q8_scale_stream_supported ? 1 : 0, g_fuse_raw_result_accum ? 1 : 0, gmac_s, mib_s,
            totals.activation_bytes, totals.weight_bytes, totals.scale_bytes, totals.result_bytes,
            totals.weight_cache_hits, totals.weight_cache_misses, cycles_per_run);
    }
    LOGIP("summary vpu_runs=%lld ip_ms=%.3f cycles_per_run=%.1f clock_mhz=%.3f", totals.vpu_runs, ip_ms, cycles_per_run,
          g_fpga_clock_mhz);

    if (g_status_stderr && (g_fpga_count == 1 || (g_profile_every > 0 && (g_fpga_count % g_profile_every) == 0))) {
        fprintf(stderr,
                "[FPGA][STAGE] tensor=%s layer=%d K=%lld N=%lld M=%lld total_ms=%.3f dma_in_ms=%.3f ip_ms=%.3f "
                "dma_out_ms=%.3f\n",
                tensor_name, effective_layer_id, (long long) k, (long long) n, (long long) m, total_ms, dma_in_ms,
                ip_ms, dma_out_ms);
        fflush(stderr);
    }

    if (macs >= g_large_matrix_min_macs && should_log_detail_run(g_fpga_count)) {
        LOGSTAGE("large tensor=%s layer=%d macs=%lld row_tiles=%lld q8_blocks=%lld vpu_runs=%lld no_cpu_fallback=1",
                 tensor_name, effective_layer_id, macs, row_tiles, (long long) q8_blocks, totals.vpu_runs);
    }

    (void) dominant_ms;
    (void) g_current_layer_id;
    (void) g_is_attention_op;
    pthread_mutex_unlock(&g_mutex);
    return g_contract_cpu_shadow_dst ? FPGA_MATMUL_CONTRACT_CPU_SHADOW : FPGA_MATMUL_FPGA_DST;
}

extern "C" void fpga_reset_kv_cache(void) {
    if (g_token_timing_collection_enabled && g_token_timing.active) {
        fpga_token_timing_emit(g_token_timing.graph_seq, 0, "kv_cache_reset_incomplete", monotonic_now_us());
    }
    g_current_seq_pos                = 0;
    g_scratch.activation_cache_valid = false;
}
