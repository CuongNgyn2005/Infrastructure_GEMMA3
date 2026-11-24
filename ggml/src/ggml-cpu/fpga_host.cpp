/*
 * KR260 Linux host bridge: Q8_0 model weights x F32 activations -> F32 output.
 *
 * P2: FPGA scale-and-accumulate path. The VPU (vector processing unit) computes
 * integer dot products; the SPU scaling unit applies paired FP16
 * weight/activation scales and returns Q16.16 (16 fractional bits) row sums.
 * The host converts those sums to F32 and accumulates across column tiles.
 *
 * P1: legacy scheduler-summary label (P1_SCHED_SUMMARY), not a compute mode.
 * Production P2 reports pingpong. P2 single_bank is retained only for bounded
 * tile-verification diagnostics; it never writes the model's final output.
 *
 * Mechanisms: validate/map hardware, quantize activations, pack/reuse tiles,
 * transfer through ZDMA (direct memory access), launch/wait/drain FPGA jobs,
 * and assemble results. Production P2 requires ping-pong across two banks;
 * single-bank execution is only for tile qualification. Optional DMA-poll
 * packing prepares the next tile in another DDR staging slot. Ownership,
 * synchronization, completion checks and error cleanup stay with these steps.
 * Shared log output and token-report formatting live in fpga_log.cpp.
 */

#include "fpga_host.h"
#include "fpga_log.h"
#include "fpga_q8_layout.h"
#include "fpga_q8_pack.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
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

__extension__ typedef unsigned __int128 fpga_uint128_t;
__extension__ typedef __int128          fpga_int128_t;

#define FPGA_HOST_TRACE_VERSION "kr260-gemma3-q8-v97-host-stream"

#define MY_IP_BASE_ADDRESS 0x00000000A0000000LL
#define REG_BASE_PHYS      0x00000000A0000000LL
#define LMM_BASE_PHYS      0x00000000A0000000LL

#define DMA_BASE_PHYS 0x00000000fd500000LL
// The verified ZDMA UIO resource is exactly one page:
// [0xFD500000, 0xFD501000).  Mapping a speculative 64 KiB aperture can cross
// into an unrelated peripheral even when current register offsets are small.
#define DMA_MMAP_SIZE 0x0000000000001000LL

// FPGA staging DDR must be real, installed PS DDR and excluded from Linux
// during boot.  The KR260 address report reserves the 2 GiB high-DDR
// window [0x800000000, 0x880000000), exposed by the ddr_high UIO resource.
static constexpr uint64_t DDR_BASE_PHYS     = 0x0000000800000000ULL;
static constexpr size_t   DDR_REGION_SIZE   = 0x0000000080000000ULL;  // 2 GiB
static constexpr uint64_t DDR_END_EXCLUSIVE = DDR_BASE_PHYS + (uint64_t) DDR_REGION_SIZE;

static int                g_log_flush_every                   = 256;
// P2 has a separate MY_IP/SPU ABI and its initialization must be observable
// on the terminal even when the file log is unavailable or buffered.
static bool               g_p2_init_requested                 = false;
static bool               g_init_verbose                      = false;
static bool               g_summary_detail_after_error       = false;
// P2 contract qualification is deliberately tile-bounded across eligible
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
// P2 scaling uses a distinct ABI. Do not infer it from the raw
// VPU1/protocol1 identity: previous raw-compatible images did not guarantee
// the SPU_PARAM/SPU_OUT packing or stream-finality contract.
static constexpr uint32_t REG_P2_STREAM_ABI           = 0x000000FC;
static constexpr uint32_t REG_P3_SPLIT_SCALE_ABI      = 0x00000200;
static constexpr uint32_t REG_P3_STREAM_STATUS        = 0x0000021C;
static constexpr uint32_t REG_P3_REJECT_COUNT         = 0x00000218;
static constexpr uint32_t P3_SPLIT_SCALE_ABI_ID = 0x50330001U;
static constexpr uint32_t P3_BANK_BYTES = 0x00008000U;
static constexpr uint32_t P3_SCALES_PER_WORD = 8U;
static constexpr uint32_t P3_STATUS_LOCK_VALID = 1U << 4;
static constexpr uint32_t P3_STATUS_SPLIT_ENABLE = 1U << 7;
static constexpr long long P3_LOCK_WAIT_TIMEOUT_US = 10000;
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
// The opt-in DMA/packing pipeline uses only the already mapped scratch
// windows.  Each source slot is disjoint from the other slot and from the
// fixed PL destination windows.  No model-sized allocation is introduced.
static constexpr uint32_t P2_PIPELINE_WEIGHT_SLOT_BYTES = 512U * 1024U;
static constexpr uint32_t P2_PIPELINE_SCALE_SLOT_BYTES  = 64U * 1024U;
static constexpr uint32_t P2_PIPELINE_WEIGHT_SLOT_COUNT = 2U;
static constexpr uint32_t P2_PIPELINE_SCALE_SLOT_COUNT  = 2U;
static constexpr size_t   P2_PIPELINE_PACK_BATCH_PAIRS  = 8U;
static_assert(WEIGHT_BASE + P2_PIPELINE_WEIGHT_SLOT_BYTES * P2_PIPELINE_WEIGHT_SLOT_COUNT == WEIGHT_END,
              "P2 pipeline weight slots must fill only the existing weight staging window");
static_assert(SPU_PARAM_BASE + P2_PIPELINE_SCALE_SLOT_BYTES * P2_PIPELINE_SCALE_SLOT_COUNT <= SPU_PARAM_END,
              "P2 pipeline scale slots must stay inside the existing SPU_PARAM staging window");
static_assert(P2_PIPELINE_PACK_BATCH_PAIRS * 4096U <= 32U * 1024U,
              "P2 pipeline callback pack batches must stay bounded");
static_assert((WEIGHT_BASE % alignof(uint32_t)) == 0U, "WEIGHT_BASE must be 32-bit aligned");
// All host-visible MY_IP registers and local-memory windows used by this
// driver lie below SPU_SCRATCH_END. The Vivado segment is 256 MiB, but the
// UIO mapping only needs the bounded register/local-memory aperture.
static_assert(SPU_PARAM_BASE + 2U * P3_BANK_BYTES <= SPU_SCRATCH_BASE, "P3 PARAM banks overlap SCRATCH");
static_assert(SPU_SCRATCH_BASE + 2U * P3_BANK_BYTES <= 0x00400000U, "P3 SCRATCH banks exceed window");
static_assert(16384U * sizeof(uint16_t) == P3_BANK_BYTES, "P3 PARAM bank capacity");
static_assert(P3_BANK_BYTES <= P2_PIPELINE_SCALE_SLOT_BYTES, "P3 scale staging capacity");
static constexpr size_t   DDR_REQUIRED_BYTES     = SPU_SCRATCH_END;
static constexpr uint32_t WEIGHT_CACHE_BASE      = 0x01000000;
static constexpr uint32_t P2_WEIGHT_RESIDENCY_END = 0x2D000000;
static constexpr size_t   WEIGHT_CACHE_ALIGN     = 4096;
static_assert((DDR_BASE_PHYS & 0xFFFULL) == 0ULL, "DDR_BASE_PHYS must be page aligned");
static_assert((DDR_REGION_SIZE & 0xFFFULL) == 0ULL, "DDR_REGION_SIZE must be page aligned");
static_assert(DDR_BASE_PHYS >= 0x0000000800000000ULL && DDR_END_EXCLUSIVE <= 0x0000000880000000ULL,
              "KR260 FPGA DDR carveout must stay inside reserved high DDR");
static_assert(DDR_REQUIRED_BYTES <= DDR_REGION_SIZE, "FPGA scratch windows exceed the reserved DDR carveout");
static_assert((uint64_t) WEIGHT_CACHE_BASE < (uint64_t) DDR_REGION_SIZE,
              "weight-cache base lies outside the reserved DDR carveout");
// P2 residency owns a fixed, non-evicting directory of sealed tiles and never
// calls msync(). The default remains 32 MiB; 704 MiB is an experimental ceiling.
// The 2 GiB window froze the KR260 (3b8304d) and is forbidden.
static constexpr long long P2_WEIGHT_RESIDENCY_MAX_MB     = 704;
static constexpr long long P2_WEIGHT_RESIDENCY_DEFAULT_MB = 32;
static constexpr long long P2_WEIGHT_RESIDENCY_MIN_MB     = 32;
static constexpr long long P2_WEIGHT_RESIDENCY_STEP_MB    = 16;
// Protocol-2/VPU2 stores each beat as an adjacent even/odd row pair.  This
// version is deliberately part of the sealed-residency identity: a v1 tile
// is byte-valid but semantically row-major, and must never be reused by VPU2.
static constexpr uint32_t  P2_WEIGHT_RESIDENCY_LAYOUT_V2  = 2U;
static constexpr size_t    P2_WEIGHT_RESIDENCY_SLOT_CAPACITY = 4096U;
// Residency lookups are bounded without walking the slot directory.  Linear
// probing is deterministic, capped, and never overwrites or evicts a seal.
static constexpr size_t    P2_WEIGHT_RESIDENCY_INDEX_BUCKETS = 16384U;
static constexpr size_t    P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES = 8U;
static constexpr uint32_t  P2_WEIGHT_RESIDENCY_NO_SLOT       = UINT32_MAX;
static constexpr uint32_t  P2_WEIGHT_RESIDENCY_INVALID_SLOT  = UINT32_MAX - 1U;
static_assert(P2_WEIGHT_RESIDENCY_INDEX_BUCKETS >= 4U * P2_WEIGHT_RESIDENCY_SLOT_CAPACITY,
              "P2 residency index must have at least four buckets per sealed slot");
static_assert((uint64_t) P2_WEIGHT_RESIDENCY_END - (uint64_t) WEIGHT_CACHE_BASE ==
                  (uint64_t) P2_WEIGHT_RESIDENCY_MAX_MB * 1024ULL * 1024ULL,
              "P2 residency budget must remain inside [0x01000000,0x2D000000)");

static constexpr int      VPU_NUM_LANES                         = 16;
static constexpr int      VPU_QK8_0                             = 32;
static constexpr int      VPU_BLOCK_BEATS                       = VPU_QK8_0 / VPU_NUM_LANES;
static constexpr int      VPU_RESULT_PACK_LANES                 = 4;
static constexpr int      VPU_PACKED_Q8_MAX_BLOCKS              = 64;
static constexpr int      VPU_DEFAULT_ROWS                      = 256;
static constexpr int      VPU_SAFE_RUNTIME_ROWS                 = 256;
static constexpr int      VPU_DEFAULT_BEATS                     = 128;
static constexpr int      VPU_DEFAULT_COLS                      = VPU_DEFAULT_BEATS * VPU_NUM_LANES;
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
static constexpr uint32_t SPU_STREAM_STATUS_QUIESCENT           = 0x00000010U;
static constexpr uint32_t SPU_CTRL_SOFT_RESET                   = 0x00000004U;
static constexpr uint32_t P2_REQUIRED_SPU_WORD_CAPACITY = (VPU_DEFAULT_ROWS * VPU_PACKED_Q8_MAX_BLOCKS + 3U) / 4U;
// The deployed PL consumes only the first 4096 128-bit SPU_PARAM words even
// though the host mapping reserves a larger PARAM window.  Keep the source
// and fixed PL destination limits separate so a two-slot source cannot admit
// a tile that would overrun the actual parameter BRAM.
static constexpr size_t P2_PIPELINE_PL_SCALE_BYTES =
    (size_t) P2_REQUIRED_SPU_WORD_CAPACITY * 16U;
static_assert(P2_PIPELINE_PL_SCALE_BYTES <= P2_PIPELINE_SCALE_SLOT_BYTES,
              "P2 pipeline scale source slots must hold the deployed PL parameter capacity");

static constexpr long long FPGA_DEFAULT_DMA_TIMEOUT_US          = 5000000LL;
static constexpr long long FPGA_DEFAULT_IP_TIMEOUT_US           = 5000000LL;
// The ZDMA hardware accepts much larger descriptors, but the legacy board
// path has shown a normal-run-only staging corruption on 512 KiB WEIGHT
// copies.  Keep each submitted descriptor bounded to 64 KiB.  Consecutive
// chunks retain the same contiguous destination window and do not alter the
// VPU's tile, Q8 layout, or arithmetic contract.
static constexpr size_t    FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES = 256U * 1024U;
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
    long long weight_cache_misses;
    long long weight_pack_us;
    // Low-overhead preparation decomposition. These counters distinguish
    // source lookup/residency, direct WEIGHT packing, scale-table packing,
    // and ACT packing without emitting one log line per tile.
    long long prep_weight_select_us;
    long long prep_direct_weight_pack_us;
    long long prep_scale_pack_us;
    long long prep_act_pack_us;
    long long activation_scale_fp16_overflows;
    // Scheduler handoff telemetry. All timestamps use CLOCK_MONOTONIC.
    // "late" means preparation completed after the running job output was
    // already ready; "headroom" means it completed before that boundary.
    long long scheduler_prepare_overlap_us;
    long long scheduler_prepare_late_us;
    long long scheduler_prepare_headroom_us;
    long long scheduler_output_to_launch_us;
    long long scheduler_retire_to_launch_us;
    long long scheduler_handoffs;
    long long scheduler_prepare_late_jobs;
    long long result_overlap_jobs;
    long long result_overlap_host_us;
    long long p3_jobs;
    long long drain_wait_us;
    long long drain_wait_jobs;
    long long scratch_dma_us;
    long long scratch_bytes;
    long long scratch_desc;
    long long p3_reject_delta;
    long long p3_mode_switches;

    // Bank-aware telemetry. H2IP includes ACT + WEIGHT + SCALE.
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
    size_t                            spu_result_bytes;
    size_t                            result_bytes;
    uint32_t                          result_values;
    uint32_t                          result_words;
    uint32_t                          scale_words;
    uint32_t                          weight_src_off;
    uint32_t                          scale_src_off;
    bool                              p2_residency_hit;
    uint32_t                          p2_residency_slot;
    uint64_t                          p2_residency_epoch;
    uint32_t                          p2_residency_seal;
    const block_q8_0_t *              act_group;
    const struct ggml_tensor *        src0;
    long long                         dma_act_us;
    long long                         dma_weight_us;
    long long                         dma_scale_us;
    long long                         dma_result_us;
    long long                         ip_start_us;
    long long                         measured_compute_us;
    long long                         ip_compute_us;
    long long                         host_result_us;
    bool                              output_dma_ready;
    bool                              hardware_released;
    bool                              result_consumed;
    long long                         event_prep_begin_us;
    long long                         event_prep_done_us;
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
    // Valid only within the pipelined scheduler's current weight-tile column loop.
    bool                              weight_bank_reused;
    bool                              act_bank_reused;
    bool                              p3_split_scale;
    size_t                            scratch_bytes;
    const block_q8_0_t *               p3_act_group;
    bool *                            p3_scratch_valid;
} fpga_tile_job_t;

// A bounded lookahead owns only the next tile's immutable identity and the
// two existing DDR source slots.  It has no descriptor, bank, or device
// ownership.  The DMA poll callback advances the main producer in small
// disjoint pair ranges while the persistent helper emits the fixed suffix.
typedef struct {
    bool                  valid;
    bool                  worker_submitted;
    bool                  consumed;
    uint64_t              generation;
    uint64_t              source_epoch;
    size_t                pair_count;
    size_t                helper_pair_begin;
    size_t                main_pair_next;
    size_t                words_per_pair;
    size_t                expected_words;
    size_t                expected_scale_entries;
    size_t                callback_bytes;
    size_t                main_written_words;
    size_t                main_scale_entries;
    long long             helper_service_us;
    long long             residual_us;
    long long             callback_us;
    fpga_stage_totals_t * totals;
    const void *          weight_data_base;
    volatile uint32_t *   weight_words;
    volatile uint32_t *   scale_words;
    fpga_tile_job_t       job;
} fpga_p2_dma_pipeline_lookahead_t;

typedef struct {
    std::vector<block_q8_0_t>  act_blocks_all;
    // Used only by FPGA_INPUT_INTEGRITY_CHECK.  It records the logical F32
    // activation columns before a raw-FPGA matmul, independent of padding in
    // src1->nb[1], so a host output store cannot silently damage the next
    // consumer in a multi-column graph.
    std::vector<uint8_t>       activation_input_snapshot;
    std::vector<uint8_t>       weight_tensor_snapshot;
    std::vector<float>         contract_actual;
    std::vector<float>         act_scales;
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

static volatile dma_ctrl * g_dma          = nullptr;
static volatile uint8_t *  g_vpu          = nullptr;
static uint8_t *           g_ddr          = nullptr;
static void *              g_dma_map_base = nullptr;
static void *              g_vpu_map_base = nullptr;
enum class fpga_mapping_kind : uint8_t {
    UNKNOWN = 0,
    UIO_PHYSICAL,
};

static const char * fpga_mapping_kind_name(fpga_mapping_kind kind) {
    switch (kind) {
        case fpga_mapping_kind::UIO_PHYSICAL:
            return "uio_physical";
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
// This is set by the mapper that created g_ddr, not inferred later from a
// display string. P2 admits only the bounded UIO DDR mapping.
static fpga_mapping_kind                      g_ddr_mapping_kind = fpga_mapping_kind::UNKNOWN;
static pthread_mutex_t                        g_mutex            = PTHREAD_MUTEX_INITIALIZER;

static fpga_scratch_t                         g_scratch;

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
static long long              g_weight_cache_misses           = 0;
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
static uint32_t g_spu_stream_status             = 0;
static uint32_t g_spu_word_capacity             = 0;
// Set only after mappings, identity/capability admission, and host self-tests
// have all completed.  A second fpga_init() must observe this before changing
// any route, counters, mapping, MMIO, or DMA state.
static bool     g_fpga_init_complete             = false;
static int      g_committed_stream_mode        = -1;
static bool     g_spu_silu_supported            = false;
static bool     g_spu_rmsnorm_supported         = false;
static bool     g_spu_rope_supported            = false;
static bool     g_spu_softmax_supported         = false;
static bool     g_pingpong_scheduler_enabled    = false;
static bool     g_p2_result_overlap_enabled     = true;
static bool     g_p2_act_bank_reuse_enabled     = true;
static uint64_t g_p2_act_bank_reused_jobs       = 0;
static uint64_t g_p2_act_bank_avoided_bytes     = 0;
// File-only aggregate telemetry for a completed graph sequence.  This is
// opt-in because even a host timestamp belongs outside the production fast
// path unless the owner is actively measuring scheduler behavior.
static bool     g_p1_sched_summary_enabled      = false;
// FPGA_TOKEN_TIMING controls TIMING records. Aggregate collection stays active
// when BREAKDOWN or PINGPONG_TIMING needs the same counters.
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
static uint32_t          g_p2_residency_extent_marked                 = 0;
static long long         g_p2_residency_graph_builds_start            = 0;
static long long         g_p2_residency_graph_build_us_start          = 0;
static long long         g_p2_residency_decode_builds                 = 0;
static long long         g_p2_residency_decode_build_us               = 0;
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
// V91 fuses direct WEIGHT and SCALE staging only for large, non-reused P2
// tiles.  The worker never owns a DMA, descriptor, IP, or SPU action; g_mutex
// still serializes every hardware job around this local work.
static int               g_p2_pack_workers_requested                  = 2;
static int               g_p2_pack_workers_active                     = 2;
static long long         g_p2_pack_parallel_jobs                       = 0;
static uint64_t          g_p2_pack_parallel_bytes                      = 0;
static long long         g_p2_pack_serial_threshold_skips              = 0;
static long long         g_p2_pack_main_us                             = 0;
static long long         g_p2_pack_helper_service_us                   = 0;
static long long         g_p2_pack_caller_wait_us                      = 0;
static long long         g_p2_pack_fused_jobs                          = 0;
static long long         g_p2_pack_fused_us                             = 0;
static long long         g_p2_pack_fused_scale_entries                  = 0;
// This second-stage pipeline is the production default after compatible P2
// admission and remains decode-only.  An explicit zero keeps the established
// prepare/submit order for diagnostics or rollback; an explicit one remains a
// fail-closed request when the admission gate is incompatible.
static bool               g_p2_pack_dma_pipeline_requested               = false;
static bool               g_p2_pack_dma_pipeline_enabled                 = false;
static long long          g_p2_pack_dma_pipeline_eligible_jobs            = 0;
static long long          g_p2_pack_dma_pipeline_consumed_jobs            = 0;
static uint64_t           g_p2_pack_dma_pipeline_callback_bytes           = 0;
static long long          g_p2_pack_dma_pipeline_callback_us              = 0;
static long long          g_p2_pack_dma_pipeline_helper_us                = 0;
static long long          g_p2_pack_dma_pipeline_helper_wait_us            = 0;
static long long          g_p2_pack_dma_pipeline_residual_us              = 0;
static long long          g_p2_pack_dma_pipeline_residual_bytes           = 0;
static long long          g_p2_pack_dma_pipeline_rejected_jobs            = 0;
static long long          g_p2_pack_dma_pipeline_helper_publication_lead_us = 0;
static long long          g_p2_pack_dma_pipeline_helper_early_completions  = 0;
enum fpga_pack_detail_field {
    PACK_TOTAL, PACK_SERIAL, PACK_DISPATCH, PACK_MAIN, PACK_FENCE, PACK_WAIT,
    PACK_HELPER, PACK_SERIAL_JOBS, PACK_PARALLEL_JOBS,
    PACK_RESERVE_LOCK, PACK_RESERVE_BODY, PACK_RESERVE_UNLOCK,
    PACK_SUBMIT_LOCK, PACK_SUBMIT_BODY, PACK_SUBMIT_SIGNAL, PACK_SUBMIT_UNLOCK, PACK_DETAIL_COUNT
};
// Caller-owned counters; helper service is overlapping, not an additive phase.
static std::array<long long, PACK_DETAIL_COUNT> g_pack_detail = {};
static std::array<long long, PACK_DETAIL_COUNT> g_pack_detail_snapshot = {};
static std::array<long long, PACK_DETAIL_COUNT> g_pack_detail_decode = {};
static fpga_pack_breakdown_log_t fpga_pack_detail_record(
        const std::array<long long, PACK_DETAIL_COUNT> & d) {
    return {d[PACK_TOTAL], d[PACK_SERIAL], d[PACK_DISPATCH], d[PACK_MAIN], d[PACK_FENCE], d[PACK_WAIT],
            d[PACK_HELPER], d[PACK_SERIAL_JOBS], d[PACK_PARALLEL_JOBS], d[PACK_RESERVE_LOCK],
            d[PACK_RESERVE_BODY], d[PACK_RESERVE_UNLOCK], d[PACK_SUBMIT_LOCK], d[PACK_SUBMIT_BODY],
            d[PACK_SUBMIT_SIGNAL], d[PACK_SUBMIT_UNLOCK]};
}
static constexpr size_t  FPGA_P2_PACK_PARALLEL_MIN_BYTES               = 256U * 1024U;

// A task has no device ownership: it is a prevalidated, disjoint range of
// volatile WEIGHT words and, for the direct fused path, the corresponding
// row-major SCALE entries in the already-admitted DDR mapping.  The caller
// owns all range checks and does not start DMA until it observes completion
// for the matching generation.
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
    const uint32_t *           cached_words = nullptr; // Diagnostic copy only; production packs.
    const void *               activation_data_base = nullptr;
    volatile uint32_t *        scale_words = nullptr;
    size_t                     expected_scale_entries = 0;
    bool                       fused_scale = false;
    bool                       p3_scale = false;
    volatile uint16_t *         p3_scale_halves = nullptr;
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
static size_t                 g_p2_pack_worker_completed_scale_entries = 0;
static long long              g_p2_pack_worker_completed_service_us = 0;
static long long              g_p2_pack_worker_completed_mono_us = 0;
static long long         g_p2_residency_avoided_cpu_pack_bytes        = 0;
// Residency still transfers each selected tile from DDR to the IP.  Keep this
// explicit so a diagnostic log never claims a traffic reduction it cannot
// prove from the current RTL/descriptor contract.
static long long         g_p2_residency_avoided_ddr_to_ip_bytes       = 0;
static bool              g_activation_cache_enabled                   = false;
// Qualification-only input ownership proof.  This is intentionally opt-in:
// it copies each logical F32 activation matrix before/after a raw FPGA
// matmul, which is appropriate for validating M>1 graph layouts but not for
// production throughput measurement.
static bool              g_activation_input_integrity_check           = false;
// Contract mode must prove the raw FPGA path without taking ownership of the
// GGML destination tensor.  In this explicit shadow mode, the host retains
// its hardware result in contract_actual for checking while the upstream
// threaded kernel writes dst.  It is not a hardware-unavailable CPU fallback
// and is never the production-acceleration route.
static bool              g_contract_cpu_shadow_dst                    = false;
// The tied embedding / vocabulary projection has 262144 rows for Gemma3-1B.
// It is the final, top-token-sensitive operation and the current legacy
// bitstream advertises neither the required stream protocol nor bitstream ID.
// Keep it on the established CPU GGML path by default until FPGA logits pass
// an end-to-end A/B comparison.  All block GEMVs remain eligible for FPGA.
static bool              g_vocab_projection_cpu_bypass                = true;
static int               g_profile_every                              = FPGA_DEFAULT_PROFILE_EVERY;
static int               g_ip_status_every                            = FPGA_DEFAULT_STATUS_EVERY;
static int               g_detail_every                               = FPGA_DEFAULT_DETAIL_EVERY;
// P2 qualification validates the canonical
// VPU->SPU scale/accumulate ABI and never consumes the raw RESULT window.
static int               g_pl_scale_contract_check_limit              = 0;
static int               g_runtime_max_rows                           = VPU_SAFE_RUNTIME_ROWS;
static int64_t           g_vocab_projection_min_n                     = 65536;
static long long         g_dma_timeout_us                             = FPGA_DEFAULT_DMA_TIMEOUT_US;
static long long         g_ip_timeout_us                              = FPGA_DEFAULT_IP_TIMEOUT_US;
static size_t            g_zdma_max_transfer_bytes                    = FPGA_DEFAULT_ZDMA_MAX_TRANSFER_BYTES;
static long long         g_large_matrix_min_macs                      = FPGA_DEFAULT_LARGE_MATRIX_MIN_MACS;
static double            g_fpga_clock_mhz                             = 0.0;
static long long         g_pl_scale_jobs                              = 0;
static long long         g_pl_scale_banks                             = 0;
static long long         g_pl_scale_stream_drops                      = 0;
static long long         g_pl_scale_stream_errors                     = 0;
static long long         g_contract_value_mismatches                  = 0;
static long long         g_contract_cpu_shadow_dst_values             = 0;
// P2 tile checking is a bounded hardware qualification. Once its requested number of
// checked tiles has completed, later graph GEMVs must execute only in the
// native CPU backend. Launching unverified VPU work after that boundary can
// turn a later CPU-side failure into misleading qualification evidence.
static long long         g_contract_limit_cpu_bypass_count            = 0;
static bool              g_contract_limit_cpu_bypass_logged           = false;
static long long         g_activation_input_integrity_checks          = 0;
static long long         g_activation_input_integrity_failures        = 0;
static long long         g_activation_scale_fp16_overflows            = 0;
// Set only while a P2 source preflight rejects an immutable Q8 tensor.  This
// lets the outer hook report the real cause instead of mislabelling it as a
// ZDMA/VPU transfer failure.
static bool              g_contract_source_validation_failed          = false;
// Set by llama-model-loader only after its complete upstream tensor validation
// succeeds. P2 qualification starts before normal FPGA initialization, so this handshake
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

typedef bool (*fpga_dma_poll_hook_fn)(void * context, long long elapsed_us);

typedef struct {
    fpga_dma_poll_hook_fn fn;
    void *                 context;
} fpga_dma_poll_hook_t;

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



static uint32_t  g_stream_protocol_version = 0;
static uint32_t  g_bitstream_id            = 0;

static bool g_p3_split_scale_enabled = false;
static bool g_p3_matmul_active = false;
static long long g_p3_mode_switches = 0;
static long long g_p3_graph_mode_switches_start = 0;

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
    long long ip_compute_us;
    long long ip2host_dma_us;
    long long host_read_us;
    long long host_accum_us;
    long long scheduler_prepare_overlap_us;
    long long scheduler_prepare_late_us;
    long long scheduler_prepare_headroom_us;
    long long scheduler_output_to_launch_us;
    long long scheduler_retire_to_launch_us;
    long long scheduler_handoffs;
    long long scheduler_prepare_late_jobs;
    long long result_overlap_jobs;
    long long result_overlap_host_us;
    long long p3_jobs;
    long long drain_wait_us;
    long long drain_wait_jobs;
    long long scratch_dma_us;
    long long scratch_bytes;
    long long scratch_desc;
    long long p3_reject_delta;
    long long p3_mode_switches;

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
struct fpga_zdma_split_t {
    long long desc = 0, bytes = 0, setup_us = 0, wait_us = 0;
    long long hook_us = 0, hook_calls = 0, done_after_hook = 0, polls = 0;
};
// Same successful descriptors and classification as the existing ZDMA summary.
static std::array<fpga_zdma_split_t, 5> g_zdma_split = {};
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

static double fpga_compute_report_us(long long us);

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
    fprintf(fp, "[ERROR] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fflush(fp);

    fprintf(stderr, "[ERROR] ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    g_summary_detail_after_error = true;
    if (g_token_timing_collection_enabled && g_token_timing.active) {
        fpga_token_timing_emit(g_token_timing.graph_seq, 0, "error", monotonic_now_us());
    }
    fpga_dma_profile_log_config();
    fpga_log_startup_begin();
    fflush(fpga_log_fp());
    fpga_p2_init_breadcrumb("phase=failure reason=fpga_fatal");
    abort();
}

static inline void mmio_fence(void) {
#if defined(__aarch64__) || defined(__arm__)
    // The Zynq UltraScale+ (KR260) host is Arm. A compiler/CPU fence alone does
    // not guarantee that a posted device write reached ZDMA before the next MMIO poll.
    // DSB SY orders completion of the descriptor and CTRL2 writes with the
    // subsequent read of the same peripheral.
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

extern "C" int fpga_p2_contract_check_requested(void) {
    // This query is called before fpga_init() by llama-cli. It deliberately
    // covers P2 qualification because the frontend uses this
    // board-free gate to defer initialization until loader validation passes.
    return env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000) > 0 ? 1 : 0;
}

extern "C" void fpga_mark_model_tensor_validation_passed(void) {
    // This function is intentionally board-free: it can run while main has
    // deferred fpga_init() for P2 qualification. Do not treat a normal model load as qualification
    // evidence; the loader calls it only for P2 qualification validation.
    if (fpga_p2_contract_check_requested()) {
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
                                           bool                    final_free_free,
                                           bool                    force_readback = false) {
    // Preserve vpu_write_tile_descriptor()'s capability gate exactly: an
    // unsupported descriptor block must not receive diagnostic MMIO writes.
    if (!g_vpu_descriptor_supported) {
        return true;
    }
    // Keep the established quiet fast path except for explicit boundary
    // diagnostics and the final FREE/FREE readback requested by the caller.
    if (!g_p2_boundary_diagnostics_enabled &&
        !(final_free_free && force_readback)) {
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

static inline size_t fpga_p3_param_bytes(size_t entries) {
    return ((entries + P3_SCALES_PER_WORD - 1U) / P3_SCALES_PER_WORD) * 16U;
}
static inline size_t fpga_p3_scratch_bytes(int G) { return (((size_t) G + 7U) / 8U) * 16U; }

static bool fpga_p3_checked_scale_shape(int rows, int G, size_t * entries, size_t * bytes) {
    if (!entries || !bytes || rows <= 0 || rows > 256 || G <= 0 || G > 64) return false;
    const size_t count = (size_t) rows * (size_t) G;
    const size_t size = fpga_p3_param_bytes(count);
    if (size > P3_BANK_BYTES) return false;
    *entries = count;
    *bytes = size;
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





static int64_t ddr_read_spu_q16_row(uint32_t off, uint16_t * row_id, bool trace_q16_verifier_row0 = false) {
    if ((off & 0x3U) != 0) {
        fpga_fatal("DDR SPU row read requires 32-bit alignment off=0x%08x", off);
    }
    const volatile uint32_t * src = (volatile const uint32_t *) ddr_ptr(off, 16U);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // Normal SPU rows are 16-byte aligned. Keep the scalar reader for tracing
    // and callers that satisfy only the existing 32-bit alignment contract.
    if (!trace_q16_verifier_row0 && (reinterpret_cast<uintptr_t>(src) & 7U) == 0U) {
        const volatile uint64_t * wide = reinterpret_cast<const volatile uint64_t *>(src);
        const uint64_t low = wide[0];
        const uint64_t high = wide[1];
        if (row_id) {
            *row_id = static_cast<uint16_t>(low & 0xffffU);
        }
        // Bits 15:0 hold the row ID; bits 79:16 hold the signed accumulator.
        // Unsigned shifts discard padding bits 127:80 without sign extension.
        const uint64_t accum_bits = (low >> 16U) | (high << 48U);
        int64_t accum;
        memcpy(&accum, &accum_bits, sizeof(accum));
        return accum;
    }
#endif
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
                                    const char *  wanted_name,
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

        if (!read_text_file("/sys/class/uio/" + uio + "/name", resolved_name) ||
            *resolved_name != wanted_name) {
            continue;
        }
        *dev_path = "/dev/" + uio;
        *map_size = size;
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
        // Multiple DMA channels share a name; select the exact physical channel.
        if (!find_uio_device_by_addr(phys, uio_name, &dev_path, &uio_size, &resolved_name)) {
            return false;
        }
        LOGINIT("using UIO name/address match for %s expected_name=%s phys=0x%llx resolved_dev=%s resolved_name=%s", tag,
                uio_name, (unsigned long long) phys, dev_path.c_str(), resolved_name.c_str());
    }
    // The KR260 reserved DDR is a distinct production resource.  Do not
    // accept an address-only alias: require the named ddr_high UIO node.
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

static bool map_registers_dma_ddr(void) {
    fpga_p2_init_breadcrumb("phase=mapping_zdma begin policy=uio_required");
    if (!map_uio_region("dma-controller", "FPGA_DMA_UIO", DMA_BASE_PHYS, sizeof(dma_ctrl), sizeof(dma_ctrl), "ZDMA",
                               0, &g_dma_map_base, &g_dma_map_size, nullptr,
                               &g_dma_map_source, nullptr)) {
        fpga_fatal("Required UIO dma-controller mapping failed; refusing initialization");
    }
    g_dma = (volatile dma_ctrl *) g_dma_map_base;
    fpga_p2_init_breadcrumb("phase=mapping_zdma pass source=%s size=0x%zx", g_dma_map_source.c_str(), g_dma_map_size);

    // A MY_IP UIO resource can advertise the entire Vivado segment.  This
    // driver uses only the established 4 MiB register/local-memory ABI, so
    // never turn an advertised UIO size into a larger process mapping.
    fpga_p2_init_breadcrumb(
        "phase=mapping_my_ip begin required_phys=0x%llx required_offset=0 required_size=0x%zx policy=uio_required",
        (unsigned long long) REG_BASE_PHYS, (size_t) SPU_SCRATCH_END);
    if (!map_uio_region("MY_IP", "FPGA_VPU_UIO", REG_BASE_PHYS, SPU_SCRATCH_END, SPU_SCRATCH_END,
                               "MY_IP/VPU", SPU_SCRATCH_END,
                               &g_vpu_map_base, &g_vpu_map_size, nullptr, &g_vpu_map_source, nullptr)) {
        fpga_fatal("Required UIO MY_IP mapping failed; refusing initialization");
    }
    g_vpu = (volatile uint8_t *) g_vpu_map_base;
    fpga_p2_init_breadcrumb("phase=mapping_my_ip pass source=%s phys=0x%llx offset=0 size=0x%zx",
                            g_vpu_map_source.c_str(), (unsigned long long) REG_BASE_PHYS, g_vpu_map_size);

    fpga_p2_init_breadcrumb(
        "phase=mapping_ddr begin policy=uio_required "
        "phys=[0x%llx,0x%llx) required_size=0x%zx",
        (unsigned long long) DDR_BASE_PHYS, (unsigned long long) DDR_END_EXCLUSIVE, DDR_REQUIRED_BYTES);
    if (!fpga_ddr_iomem_preflight()) {
        return false;
    }
    g_ddr_mapping_kind = fpga_mapping_kind::UNKNOWN;
    if (!map_uio_region("ddr_high", "FPGA_DDR_UIO", DDR_BASE_PHYS, DDR_REGION_SIZE, DDR_REQUIRED_BYTES,
                               "ddr_high", g_ddr_requested_map_size,
                               &g_ddr_map_base, &g_ddr_map_size, &g_ddr_advertised_size, &g_ddr_map_source,
                               &g_ddr_mapping_kind)) {
        fpga_fatal("Required UIO ddr_high mapping failed; refusing initialization");
    }
    g_ddr = (uint8_t *) g_ddr_map_base;
    fpga_p2_init_breadcrumb("phase=mapping_ddr pass source=%s map_kind=%s size=0x%zx", g_ddr_map_source.c_str(),
                            fpga_mapping_kind_name(g_ddr_mapping_kind), g_ddr_map_size);
    return true;
}

static void fpga_uncached_read_probe(void) {
#if defined(__aarch64__)
    constexpr size_t rows = 16384;
    static_assert(SPU_OUT_END - SPU_OUT_BASE == rows * 16U, "probe must stay inside SPU_OUT");
    const uint8_t * base = ddr_ptr(SPU_OUT_BASE, rows * 16U);
    if (!base || (reinterpret_cast<uintptr_t>(base) & 15U) != 0U) {
        fpga_fatal("SPU_OUT read probe requires a mapped, 16-byte aligned range");
    }
    volatile uint64_t accumulator = 0;
    double best_ns[3] = {};
    for (int method = 0; method < 3; ++method) {
        for (int pass = 0; pass < 3; ++pass) {
            const long long start = now_us();
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t * addr = base + row * 16U;
                uint64_t lo, hi;
                if (method == 0) {
                    const volatile uint64_t * pair = reinterpret_cast<const volatile uint64_t *>(addr);
                    lo = pair[0];
                    hi = pair[1];
                } else if (method == 1) {
                    asm volatile("ldp %0, %1, [%2]" : "=&r"(lo), "=&r"(hi) : "r"(addr) : "memory");
                } else {
                    asm volatile("ldr q0, [%2]\n\tumov %0, v0.d[0]\n\tumov %1, v0.d[1]"
                                 : "=&r"(lo), "=&r"(hi) : "r"(addr) : "v0", "memory");
                }
                accumulator = accumulator + lo + hi;
            }
            const double ns_per_row = (double) (now_us() - start) * 1000.0 / (double) rows;
            if (pass == 0 || ns_per_row < best_ns[method]) best_ns[method] = ns_per_row;
        }
    }
    (void) accumulator;
    fpga_log_line(true, "UNCACHED_READ_PROBE", true,
                  "rows=16384 ldr64x2_ns_per_row=%.1f ldp_ns_per_row=%.1f ldrq_ns_per_row=%.1f",
                  best_ns[0], best_ns[1], best_ns[2]);
#else
    fpga_log_line(true, "UNCACHED_READ_PROBE", true, "skipped=non_aarch64");
#endif
}

static bool configure_ddr_mapping_policy(void) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        LOGE("cannot determine page size for ddr_high mapping");
        return false;
    }

    g_p2_weight_residency_budget_mb = 0;
    g_ddr_requested_map_size = align_up_size(DDR_REQUIRED_BYTES, (size_t) page_size);
    if (g_p2_weight_residency_requested) {
        long long residency_mb = P2_WEIGHT_RESIDENCY_DEFAULT_MB;
        const char * source = "default";
        const char * value = getenv("FPGA_P2_RESIDENCY_MB");
        if (value && *value) {
            char * end = nullptr;
            errno = 0;
            residency_mb = strtoll(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' ||
                residency_mb < P2_WEIGHT_RESIDENCY_MIN_MB || residency_mb > P2_WEIGHT_RESIDENCY_MAX_MB ||
                residency_mb % P2_WEIGHT_RESIDENCY_STEP_MB != 0) {
                LOGE("invalid FPGA_P2_RESIDENCY_MB='%s': expected a multiple of %lld in [%lld,%lld] MiB",
                     value, P2_WEIGHT_RESIDENCY_STEP_MB, P2_WEIGHT_RESIDENCY_MIN_MB, P2_WEIGHT_RESIDENCY_MAX_MB);
                return false;
            }
            source = "env";
        }
        const uint64_t required_bytes = (uint64_t) WEIGHT_CACHE_BASE + (uint64_t) residency_mb * 1024ULL * 1024ULL;
        if (required_bytes > DDR_REGION_SIZE) {
            LOGE("P2 residency range exceeds reserved DDR: budget_mb=%lld required=0x%llx region=0x%zx", residency_mb,
                 (unsigned long long) required_bytes, DDR_REGION_SIZE);
            return false;
        }
        g_p2_weight_residency_budget_mb = residency_mb;
        g_ddr_requested_map_size = align_up_size((size_t) required_bytes, (size_t) page_size);
        LOGINIT(
            "P2_RESIDENCY_CONFIG requested=1 diagnostic=%d trace=%d verify_metadata=%d base_off=0x%08x phys=[0x%llx,0x%llx) budget_mb=%lld requested_map=0x%zx "
            "policy=fixed_non_evicting_directory_index_no_msync slots=%zu index_buckets=%zu source=%s",
            g_p2_weight_residency_diagnostic ? 1 : 0, g_p2_residency_trace_enabled ? 1 : 0,
            g_p2_residency_verify_metadata ? 1 : 0, WEIGHT_CACHE_BASE,
            (unsigned long long) (DDR_BASE_PHYS + WEIGHT_CACHE_BASE),
            (unsigned long long) (DDR_BASE_PHYS + required_bytes), residency_mb, g_ddr_requested_map_size,
            P2_WEIGHT_RESIDENCY_SLOT_CAPACITY, P2_WEIGHT_RESIDENCY_INDEX_BUCKETS, source);
        return true;
    }
    LOGINIT("DDR map policy: residency=disabled requested_size=0x%zx", g_ddr_requested_map_size);
    return true;
}




// Defined with the staging helpers below.  P2 uses this exact bounded
// first/last-word readback after CPU writes to an admitted physical DDR aperture.
static void fpga_ddr_staging_readback_commit(uint32_t off, size_t bytes);

static bool p2_physical_ddr_mapping_active(void) {
    return g_p2_init_requested && g_ddr_mapping_kind == fpga_mapping_kind::UIO_PHYSICAL;
}


static void fpga_p2_ddr_sync_breadcrumb(const char * tag,
                                        const char * direction,
                                        uint32_t     off,
                                        size_t       bytes,
                                        const char * ordering);

// Physical UIO DDR uses bounded readback and DSB ordering before DMA reads,
// and DSB ordering before the CPU consumes results written by DMA.
static bool fpga_p2_ddr_sync(uint32_t off, size_t bytes, bool device_to_cpu, const char * tag) {
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
static bool zdma_wait_transfer_complete(const char *                  tag,
                                        zdma_completion_info_t *       info,
                                        const fpga_dma_poll_hook_t *   poll_hook = nullptr,
                                        long long                      launch_us = 0,
                                        long long *                    hook_us = nullptr,
                                        long long *                    hook_calls = nullptr,
                                        long long *                    done_after_hook = nullptr) {
    if (hook_us) *hook_us = 0;
    if (hook_calls) *hook_calls = 0;
    if (done_after_hook) *done_after_hook = 0;
    if (!dma_is_mapped()) {
        LOGE("ZDMA completion gate has no mapped channel tag=%s", tag ? tag : "?");
        return false;
    }

    const long long t0          = launch_us > 0 ? launch_us : now_us();
    long long       polls       = 0;
    bool            saw_enabled = false;
    bool            previous_poll_hook = false;
    while (true) {
        const uint32_t ctrl2    = g_dma->ZDMA_CH_CTRL2;
        const uint32_t isr      = g_dma->ZDMA_CH_ISR;
        const uint32_t status   = g_dma->ZDMA_CH_STATUS;
        const bool     enabled  = (ctrl2 & ZDMA_CTRL2_EN) != 0U;
        const bool     dma_done = (isr & ZDMA_ISR_DMA_DONE) != 0U;

        fpga_dma_profile_observe((isr & ZDMA_ISR_ERROR_MASK) != 0U, dma_done, enabled);

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
            // Observation at the next poll, not an exact hardware completion timestamp.
            if (previous_poll_hook && done_after_hook) ++*done_after_hook;
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
        // The hook is deliberately after error, DONE/disabled, and the
        // launch-anchored timeout checks.  It may only touch a disjoint host
        // DDR source slot; it cannot alter descriptors or device state.
        previous_poll_hook = false;
        bool hook_ok = true;
        if (poll_hook && poll_hook->fn) {
            const long long hook_start = hook_us ? now_us() : 0;
            hook_ok = poll_hook->fn(poll_hook->context, now_us() - t0);
            if (hook_us) *hook_us += now_us() - hook_start;
            if (hook_calls) ++*hook_calls;
            previous_poll_hook = true;
        }
        if (!hook_ok) {
            // A failed host hook must not return while the channel still owns
            // its descriptor.  Drain EN before the caller can reuse either
            // staging slot or report the transfer failure.
            if (!zdma_wait_channel_disabled(tag, "poll_hook_failure")) {
                // No caller may reuse either source slot, descriptor, or
                // mapping while EN remains asserted.  There is no safe
                // software abort/reset sequence in this ABI, so fail closed
                // instead of returning into a possible CPU fallback.
                fpga_fatal("ZDMA EN remained set after DMA poll hook failure; refusing slot reuse");
            }
            LOGE("ZDMA poll hook failed tag=%s elapsed_us=%lld", tag ? tag : "?", now_us() - t0);
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

static bool fpga_dma_copy_one(uint64_t                         src_phys,
                              uint64_t                         dst_phys,
                              size_t                           bytes,
                              const char *                     tag,
                              const fpga_dma_poll_hook_t *     poll_hook = nullptr) {
    const long long setup_start = now_us();
    fpga_dma_profile_transfer profile(tag, bytes);
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
                "step=source_sync edge=before tag=%s direction=%s offset=0x%08x bytes=%zu map_kind=%s "
                "action=no_msync ordering=%s",
                tag ? tag : "?", "cpu_to_device",
                src_ddr_off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind),
                "dsb_readback");
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
                "step=source_sync edge=after tag=%s direction=%s offset=0x%08x bytes=%zu map_kind=%s "
                "action=no_msync ordering=%s",
                tag ? tag : "?", "cpu_to_device",
                src_ddr_off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind),
                "dsb_readback");
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
    profile.launch();
    const long long start_us = now_us();
    g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_START;
    mmio_fence();
    if (p2_transfer_detail) {
        fpga_p2_dma_breadcrumb("step=ctrl2_start edge=after_write_fence ctrl2=0x%08x", g_dma->ZDMA_CH_CTRL2);
    }

    zdma_completion_info_t completion = {};
    long long hook_us = 0, hook_calls = 0, done_after_hook = 0;
    const bool completed = zdma_wait_transfer_complete(tag, &completion, poll_hook, t0,
                                                       &hook_us, &hook_calls, &done_after_hook);
    const long long wait_end = now_us();
    if (!completed) {
        if (poll_hook && !zdma_wait_channel_disabled(tag, "pipeline_completion_failure")) {
            fpga_fatal("ZDMA EN remained set after pipeline completion failure; refusing slot reuse");
        }
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
        size_t split_type = 4;
        if (strstr(effective_tag, "WEIGHT") != nullptr) {
            split_type = 1;
            g_token_timing.zdma_weight_descriptors++;
        } else if (strstr(effective_tag, "ACT") != nullptr) {
            split_type = 0;
            g_token_timing.zdma_act_descriptors++;
        } else if (strstr(effective_tag, "SCALE") != nullptr || strstr(effective_tag, "PARAM") != nullptr) {
            split_type = 2;
            g_token_timing.zdma_scale_descriptors++;
        } else if (strstr(effective_tag, "SPU_OUT") != nullptr || strstr(effective_tag, "RESULT") != nullptr) {
            split_type = 3;
            g_token_timing.zdma_result_descriptors++;
        } else {
            g_token_timing.zdma_other_descriptors++;
        }
        fpga_zdma_split_t & split = g_zdma_split[split_type];
        ++split.desc;
        split.bytes += (long long) bytes;
        split.setup_us += start_us - setup_start;
        split.wait_us += wait_end - start_us;
        split.hook_us += hook_us;
        split.hook_calls += hook_calls;
        split.done_after_hook += done_after_hook;
        split.polls += completion.polls;
    }

    LOGDMA(
        "tag=%s src=0x%llx dst=0x%llx bytes=%zu units=bytes ms=%.3f GB/s=%.3f "
        "completion=isr_ack_then_dma_done_and_ctrl2_en_clear status=0x%08x state=%u isr=0x%08x",
        tag ? tag : "?", (unsigned long long) src_phys, (unsigned long long) dst_phys, bytes,
        (double) (t1 - t0) / 1000.0,
        (t1 > t0) ? (double) bytes / ((double) (t1 - t0) * 1000.0) : 0.0, status, state, isr);
    profile.success();
    return true;
}

static bool fpga_dma_copy(uint64_t                     src_phys,
                          uint64_t                     dst_phys,
                          size_t                       bytes,
                          const char *                 tag,
                          const fpga_dma_poll_hook_t * poll_hook = nullptr);

static bool fpga_p2_dma_pipeline_poll(void * context, long long elapsed_us);

static bool fpga_dma_write_to_ip(uint32_t offset, size_t bytes, const char * tag) {
    return fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) offset, LMM_BASE_PHYS + (uint64_t) offset, bytes, tag);
}

static bool fpga_dma_read_from_ip(uint32_t offset, size_t bytes, const char * tag) {
    return fpga_dma_copy(LMM_BASE_PHYS + (uint64_t) offset, DDR_BASE_PHYS + (uint64_t) offset, bytes, tag);
}

// Submit a long linear copy as ordered, non-overlapping descriptors.  The
// legacy bitstream cannot observe a VPU start until the caller has returned,
// so splitting a copy here does not expose a partially loaded ACT/WEIGHT
// window to the VPU.  It does, however, keep the ZDMA/IP interconnect away
// from the 512 KiB descriptor pattern implicated by the contract log.
static bool fpga_dma_copy(uint64_t                     src_phys,
                          uint64_t                     dst_phys,
                          size_t                       bytes,
                          const char *                 tag,
                          const fpga_dma_poll_hook_t * poll_hook) {
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

    if (bytes <= g_zdma_max_transfer_bytes) {
        return fpga_dma_copy_one(src_phys, dst_phys, bytes, tag ? tag : "?", poll_hook);
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
        if (!fpga_dma_copy_one(src_phys + (uint64_t) offset, dst_phys + (uint64_t) offset, this_bytes, chunk_tag,
                               poll_hook)) {
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

// P2 requires stream mode 0.  The register write is accepted only while the
// stream FIFO and command-mode SPU are idle, so verify both before writing and
// read the mode back before any P2 staging or DMA.
struct fpga_stream_mode_report_t {
    bool recorded = false;
    uint32_t status = 0;
};

static bool fpga_enforce_p2_stream_mode(const char * context,
                                      fpga_stream_mode_report_t * deferred = nullptr,
                                      uint32_t want_mode = 0U) {
    if (deferred) *deferred = {};
    if (want_mode > 1U || (want_mode == 1U && !g_p3_split_scale_enabled)) return false;
    if (g_committed_stream_mode == (int) want_mode) {
        return true;
    }

    // The cached value is invalid while a real transition is being checked.
    // Any failed MMIO readback therefore leaves the next caller forced through
    // the full quiescence, write, and verification sequence.
    const bool p3_transition = g_p3_split_scale_enabled || g_committed_stream_mode == 1 || want_mode == 1U;
    g_committed_stream_mode = -1;
    const uint32_t stream_status = vpu_rd32(REG_SPU_STREAM_STATUS);
    const uint32_t spu_status = vpu_rd32(REG_SPU_CTRL);
    const bool quiescent = (stream_status & SPU_STREAM_STATUS_QUIESCENT) != 0U;
    const bool spu_busy = (spu_status & 1U) != 0U;
    if (!quiescent || spu_busy) {
        LOGE("P2_STREAM_MODE_REJECT context=%s stream_status=0x%08x spu_status=0x%08x "
             "reason=not_quiescent_or_busy action=no_p2_data_plane_write",
             context ? context : "?", stream_status, spu_status);
        return false;
    }
    if (p3_transition && (vpu_rd32(REG_P3_STREAM_STATUS) & P3_STATUS_LOCK_VALID) != 0U) {
        LOGE("P3 stream mode locked context=%s want=%u", context ? context : "?", want_mode);
        return false;
    }
    mmio_fence();
    vpu_wr32(REG_STREAM_MODE, want_mode);
    ++g_p3_mode_switches;
    mmio_fence();
    const uint32_t mode_readback = vpu_rd32(REG_STREAM_MODE);
    const uint32_t stream_after = vpu_rd32(REG_SPU_STREAM_STATUS);
    const uint32_t p3_after = p3_transition ? vpu_rd32(REG_P3_STREAM_STATUS) : 0U;
    const bool split_matches = !p3_transition ||
        ((p3_after & P3_STATUS_SPLIT_ENABLE) != 0U) == (want_mode == 1U);
    const bool mode_matches = (mode_readback & 1U) == want_mode && split_matches;
    const bool still_quiescent = (stream_after & SPU_STREAM_STATUS_QUIESCENT) != 0U;
    if (!mode_matches || !still_quiescent) {
        LOGE("P2_STREAM_MODE_FAIL context=%s readback=0x%08x stream_status=0x%08x "
             "mode_matches=%d quiescent=%d action=no_p2_data_plane_write",
             context ? context : "?", mode_readback, stream_after, mode_matches ? 1 : 0,
             still_quiescent ? 1 : 0);
        return false;
    }
    g_committed_stream_mode = (int) want_mode;
    if (p3_transition) {
        LOGI("P3 STREAM MODE PASS context=%s Chế độ = %u", context ? context : "?", want_mode);
    }
    if (deferred) {
        deferred->recorded = true;
        deferred->status = stream_after;
    } else if (want_mode == 0U) {
        LOGI("P2 STREAM MODE PASS\n  Ngữ cảnh = %s | Chế độ = 0 | Trạng thái stream = 0x%08x",
             context ? context : "?", stream_after);
    }
    return true;
}

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

// Copy a prevalidated pair range using the production destination store width.
static bool fpga_copy_weight_pair_range(const fpga_p2_pack_worker_task_t & task, size_t * written) {
    if (!task.cached_words || !task.dst_words || !written || task.rows <= 0 ||
        task.group_blocks <= 0 || task.group_blocks > INT_MAX / VPU_BLOCK_BEATS ||
        task.group_beats != task.group_blocks * VPU_BLOCK_BEATS ||
        task.pair_begin > task.pair_end || task.pair_end > ((size_t) task.rows + 1U) / 2U ||
        ((uintptr_t) task.dst_words & 7U)) return false;
    const size_t words_per_pair = (size_t) task.group_beats * 8U;
    if (task.pair_end > SIZE_MAX / words_per_pair) return false;
    const size_t begin = task.pair_begin * words_per_pair;
    const size_t end = task.pair_end * words_per_pair;
    for (size_t i = begin; i < end; i += 2) {
        uint64_t value;
        memcpy(&value, task.cached_words + i, sizeof(value));
        *reinterpret_cast<volatile uint64_t *>(task.dst_words + i) = value;
    }
    *written = end - begin;
    return true;
}

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
        size_t written_scale_entries = 0U;
        const bool range_success = task.p3_scale ?
            fpga_pack_direct_weight_p3scale_pair_range(
                task.dst_words, task.p3_scale_halves, task.src0, task.weight_data_base,
                task.row0, task.k_block0, task.rows, task.group_blocks, task.group_beats, task.pair_begin,
                task.pair_end, true, &written_words, &written_scale_entries) : task.fused_scale ?
            fpga_pack_direct_weight_scale_pair_range(
                task.dst_words, task.scale_words, task.src0, task.weight_data_base, task.activation_data_base,
                task.row0, task.k_block0, task.rows, task.group_blocks, task.group_beats, task.pair_begin,
                task.pair_end, true, &written_words, &written_scale_entries) :
            (task.cached_words ? fpga_copy_weight_pair_range(task, &written_words) :
             fpga_pack_direct_weight_pair_range(
                task.dst_words, task.src0, task.weight_data_base, task.row0, task.k_block0, task.rows,
                task.group_blocks, task.group_beats, task.pair_begin, task.pair_end, true, &written_words));
        const bool scale_count_ok = !(task.fused_scale || task.p3_scale) || written_scale_entries == task.expected_scale_entries;
        const bool success = range_success && written_words == task.expected_words && scale_count_ok;
        // Completion is never published before all volatile WEIGHT and SCALE
        // stores are globally observed.  The caller performs its own DSB
        // before DMA.
        mmio_fence();
        pthread_mutex_lock(&g_p2_pack_worker_mutex);
        g_p2_pack_worker_busy = false;
        g_p2_pack_worker_completed_generation = task.generation;
        g_p2_pack_worker_completed_success = success;
        g_p2_pack_worker_completed_words = written_words;
        g_p2_pack_worker_completed_scale_entries = written_scale_entries;
        g_p2_pack_worker_completed_service_us = now_us() - service0;
        g_p2_pack_worker_completed_mono_us = monotonic_now_us();
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
    g_p2_pack_worker_completed_scale_entries = 0U;
    g_p2_pack_worker_completed_service_us = 0;
    g_p2_pack_worker_completed_mono_us = 0;
    const int create_rc = pthread_create(&g_p2_pack_worker_thread, nullptr, fpga_p2_pack_worker_main, nullptr);
    if (create_rc == 0) {
        g_p2_pack_worker_created = true;
    }
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    return create_rc == 0;
}

static bool fpga_p2_pack_worker_next_generation(uint64_t * generation, bool record_dispatch = true) {
    if (!generation) {
        return false;
    }
    const long long lock0 = now_us();
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    const long long locked = now_us();
    const bool ready = g_p2_pack_worker_created && !g_p2_pack_worker_stop_requested &&
                       !g_p2_pack_worker_task_pending && !g_p2_pack_worker_busy &&
                       g_p2_pack_worker_next_generation != UINT64_MAX;
    if (ready) {
        *generation = ++g_p2_pack_worker_next_generation;
    }
    const long long unlock0 = now_us();
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    const long long unlocked = now_us();
    if (ready && record_dispatch) {
        g_pack_detail[PACK_RESERVE_LOCK] += locked - lock0;
        g_pack_detail[PACK_RESERVE_BODY] += unlock0 - locked;
        g_pack_detail[PACK_RESERVE_UNLOCK] += unlocked - unlock0;
    }
    return ready;
}

static bool fpga_p2_pack_worker_submit(const fpga_p2_pack_worker_task_t & task, bool record_dispatch = true) {
    const long long lock0 = now_us();
    pthread_mutex_lock(&g_p2_pack_worker_mutex);
    const long long locked = now_us();
    if (!g_p2_pack_worker_created || g_p2_pack_worker_stop_requested || g_p2_pack_worker_task_pending ||
        g_p2_pack_worker_busy || task.generation == 0U) {
        pthread_mutex_unlock(&g_p2_pack_worker_mutex);
        return false;
    }
    g_p2_pack_worker_task = task;
    g_p2_pack_worker_task_pending = true;
    const long long unlock0 = now_us();
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    const long long unlocked = now_us();
    // Publish the predicate before notifying so the awakened helper need not
    // contend with this caller for the mutex. The helper tests task_pending
    // under the same mutex before waiting, so an early notification cannot
    // lose a task. Submission and cleanup are serialized by the owning g_mutex;
    // the static condition variable remains alive throughout notification.
    pthread_cond_signal(&g_p2_pack_worker_task_cv);
    const long long signaled = now_us();
    // Elapsed intervals may include OS descheduling; these are not CPU-only costs.
    if (record_dispatch) {
        g_pack_detail[PACK_SUBMIT_LOCK] += locked - lock0;
        g_pack_detail[PACK_SUBMIT_BODY] += unlock0 - locked;
        g_pack_detail[PACK_SUBMIT_SIGNAL] += signaled - unlocked;
        g_pack_detail[PACK_SUBMIT_UNLOCK] += unlocked - unlock0;
    }
    return true;
}

static bool fpga_p2_pack_worker_wait(uint64_t generation,
                                     size_t * words,
                                     long long * service_us,
                                     long long * wait_us,
                                     size_t * scale_entries = nullptr,
                                     long long * completion_mono_us = nullptr) {
    if (completion_mono_us) {
        *completion_mono_us = 0;
    }
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
    if (scale_entries) {
        *scale_entries = g_p2_pack_worker_completed_scale_entries;
    }
    *service_us = g_p2_pack_worker_completed_service_us;
    if (completion_mono_us && success) {
        *completion_mono_us = g_p2_pack_worker_completed_mono_us;
    }
    pthread_mutex_unlock(&g_p2_pack_worker_mutex);
    *wait_us = now_us() - wait0;
    return success;
}

static void fpga_p2_dma_pipeline_invalidate(fpga_p2_dma_pipeline_lookahead_t * lookahead) {
    if (!lookahead) {
        return;
    }
    if (lookahead->worker_submitted) {
        size_t helper_words = 0U;
        size_t helper_scales = 0U;
        long long helper_service = 0;
        long long helper_wait = 0;
        const bool helper_done = fpga_p2_pack_worker_wait(lookahead->generation, &helper_words, &helper_service,
                                                          &helper_wait, &helper_scales);
        if (!helper_done) {
            fpga_fatal("P2 DMA pipeline helper did not publish completion; refusing source-slot reuse");
        }
        lookahead->worker_submitted = false;
    }
    lookahead->valid = false;
    lookahead->consumed = false;
}

struct fpga_p2_dma_pipeline_scope_guard {
    fpga_p2_dma_pipeline_lookahead_t * lookahead;
    ~fpga_p2_dma_pipeline_scope_guard() {
        if (lookahead && (lookahead->valid || lookahead->worker_submitted)) {
            fpga_p2_dma_pipeline_invalidate(lookahead);
        }
    }
};

// Pack one bounded prefix slice for a preadmitted future tile.  The helper
// owns the fixed suffix; both ranges are disjoint in the inactive source slot.
static bool fpga_p2_dma_pipeline_pack_batch(fpga_p2_dma_pipeline_lookahead_t * lookahead, size_t pair_end) {
    if (!lookahead || !lookahead->valid || lookahead->consumed || !lookahead->weight_words ||
        !lookahead->scale_words || pair_end < lookahead->main_pair_next || pair_end > lookahead->helper_pair_begin) {
        return false;
    }
    const size_t pair_begin = lookahead->main_pair_next;
    if (pair_begin == pair_end) {
        return true;
    }
    size_t written_words = 0U;
    size_t written_scales = 0U;
    const bool packed = lookahead->job.p3_split_scale ?
        fpga_pack_direct_weight_p3scale_pair_range(
            lookahead->weight_words, reinterpret_cast<volatile uint16_t *>(lookahead->scale_words),
            lookahead->job.src0, lookahead->weight_data_base, lookahead->job.row0, lookahead->job.k_block0,
            lookahead->job.rows, lookahead->job.group_blocks, lookahead->job.group_beats, pair_begin, pair_end,
            true, &written_words, &written_scales) :
        fpga_pack_direct_weight_scale_pair_range(
            lookahead->weight_words, lookahead->scale_words, lookahead->job.src0, lookahead->weight_data_base,
            lookahead->job.act_group, lookahead->job.row0, lookahead->job.k_block0, lookahead->job.rows,
            lookahead->job.group_blocks, lookahead->job.group_beats, pair_begin, pair_end, true, &written_words,
            &written_scales);
    if (!packed) {
        return false;
    }
    const size_t pair_count = pair_end - pair_begin;
    if (pair_count > SIZE_MAX / lookahead->words_per_pair ||
        pair_count * lookahead->words_per_pair > SIZE_MAX / sizeof(uint32_t)) {
        return false;
    }
    if (lookahead->main_written_words > SIZE_MAX - written_words ||
        lookahead->main_scale_entries > SIZE_MAX - written_scales) {
        return false;
    }
    const size_t expected_scales =
        ((std::min(pair_end * 2U, (size_t) lookahead->job.rows) - pair_begin * 2U) *
         (size_t) lookahead->job.group_blocks);
    if (written_words != pair_count * lookahead->words_per_pair || written_scales != expected_scales) {
        return false;
    }
    lookahead->main_pair_next = pair_end;
    lookahead->main_written_words += written_words;
    lookahead->main_scale_entries += written_scales;
    return true;
}

static bool fpga_p2_dma_pipeline_poll(void * context, long long elapsed_us) {
    (void) elapsed_us;
    auto * const lookahead = static_cast<fpga_p2_dma_pipeline_lookahead_t *>(context);
    if (!lookahead || !lookahead->valid || lookahead->consumed) {
        return false;
    }
    if (lookahead->main_pair_next >= lookahead->helper_pair_begin) {
        return true;
    }
    const long long pack0 = now_us();
    const size_t words_before = lookahead->main_written_words;
    const size_t pair_end = std::min(lookahead->main_pair_next + P2_PIPELINE_PACK_BATCH_PAIRS,
                                     lookahead->helper_pair_begin);
    fpga_dma_profile_pack_begin();
    const bool ok = fpga_p2_dma_pipeline_pack_batch(lookahead, pair_end);
    fpga_dma_profile_pack_end(ok, words_before, lookahead->main_written_words);
    const long long pack_us = now_us() - pack0;
    lookahead->callback_us += pack_us;
    g_p2_pack_dma_pipeline_callback_us += pack_us;
    if (ok) {
        const size_t words = lookahead->main_written_words - words_before;
        if (words > SIZE_MAX / sizeof(uint32_t) ||
            lookahead->callback_bytes > SIZE_MAX - words * sizeof(uint32_t) ||
            g_p2_pack_dma_pipeline_callback_bytes > UINT64_MAX - (uint64_t) (words * sizeof(uint32_t))) {
            lookahead->valid = false;
            return false;
        }
        const size_t bytes = words * sizeof(uint32_t);
        lookahead->callback_bytes += bytes;
        g_p2_pack_dma_pipeline_callback_bytes += (uint64_t) bytes;
        return true;
    }
    // The DMA completion loop will first quiesce the channel.  The submitter
    // then joins the helper and invalidates this record before any slot reuse.
    lookahead->valid = false;
    return false;
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
    g_p2_pack_worker_completed_scale_entries = 0U;
    g_p2_pack_worker_completed_mono_us = 0;
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


// When a supposedly immutable GGUF Q8 block is invalid, distinguish an
// invalid file from a process-memory mutation. A qualification failure here is
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


static void fpga_contract_log_q8_nonfinite_provenance(const struct ggml_tensor * src0,
                                                      const block_q8_0_t *       weight,
                                                      const block_q8_0_t *       act,
                                                      int64_t                    row,
                                                      int64_t                    col,
                                                      const char *               tensor_name,
                                                      int                        layer_id,
                                                      float                      kernel_reference);

// A DSB followed by reads from both ends of the written range drains
// posted PS stores before ZDMA becomes the reader. This bounded fence remains
// enabled in normal execution.
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

// Use aligned scalar device accesses; memcpy only moves ordinary local words.
// Prefix/tail bytes keep accesses bounded for unaligned or short spans.
static void fpga_p2_residency_store(volatile uint8_t * dst, const uint8_t * src, size_t bytes) {
    size_t i = 0;
    for (; i < bytes && ((uintptr_t) (dst + i) & 7U); ++i) dst[i] = src[i];
    for (; bytes - i >= sizeof(uint64_t); i += sizeof(uint64_t)) {
        uint64_t word;
        memcpy(&word, src + i, sizeof(word));
        *reinterpret_cast<volatile uint64_t *>(dst + i) = word;
    }
    for (; i < bytes; ++i) dst[i] = src[i];
}

// Return the first mismatching byte, or bytes after exact whole-span readback.
static size_t fpga_p2_residency_mismatch(const volatile uint8_t * src, const uint8_t * expected,
                                        size_t bytes, uint8_t * actual) {
    size_t i = 0;
    for (; i < bytes && ((uintptr_t) (src + i) & 7U); ++i) {
        *actual = src[i];
        if (*actual != expected[i]) return i;
    }
    for (; bytes - i >= sizeof(uint64_t); i += sizeof(uint64_t)) {
        const uint64_t word = *reinterpret_cast<const volatile uint64_t *>(src + i);
        uint64_t wanted;
        memcpy(&wanted, expected + i, sizeof(wanted));
        if (word != wanted) {
            uint8_t observed[sizeof(word)];
            memcpy(observed, &word, sizeof(word));
            for (size_t j = 0; j < sizeof(word); ++j) {
                if (observed[j] != expected[i + j]) {
                    *actual = observed[j];
                    return i + j;
                }
            }
        }
    }
    for (; i < bytes; ++i) {
        *actual = src[i];
        if (*actual != expected[i]) return i;
    }
    return bytes;
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
    // A build is optional. While the ping-pong scheduler owns either bank,
    // decline immediately so residency never blocks waiting for a running
    // descriptor to retire.
    if (g_pingpong_scheduler_enabled) {
        mmio_fence();
        const uint32_t status_before = vpu_rd32(REG_STATUS);
        const uint32_t bank_stat     = vpu_rd32(REG_BANK_STAT);
        const uint32_t slot_state    = vpu_rd32(REG_SLOT_STATE);
        const uint32_t desc_flags    = vpu_rd32(REG_DESC_FLAGS);
        const uint32_t status_after  = vpu_rd32(REG_STATUS);
        mmio_fence();
        const bool scheduler_registers_idle = status_before == status_after && (status_before & STATUS_BUSY) == 0U &&
                                       (bank_stat & BANK_STAT_BUSY) == 0U &&
                                       slot_state == fpga_slot_state_word(0, FPGA_SLOT_FREE, FPGA_SLOT_FREE) &&
                                       desc_flags == 0U;
        if (!scheduler_registers_idle) {
            g_p2_residency_misses++;
            g_p2_residency_miss_quiescence++;
            fpga_p2_residency_log(false, "MISS", "reason=active_scheduler_registers_not_idle action=defer_direct_stage "
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
    const uint32_t chunk = (allocation_end - 1U) >> 24;
    if (chunk + 1U > g_p2_residency_extent_marked) {
        fpga_p2_boundary_marker(
            "P2_RESIDENCY_EXTENT chunk=%u off=[0x%08x,0x%08x) phys_end=0x%llx budget_mb=%lld builds=%lld",
            chunk, g_p2_residency_next_off, allocation_end,
            (unsigned long long) (DDR_BASE_PHYS + allocation_end), g_p2_weight_residency_budget_mb,
            g_p2_residency_builds);
        g_p2_residency_extent_marked = chunk + 1U;
    }
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
    fpga_p2_residency_store(dst_qs, expected_qs.data(), qs_bytes);
    fpga_p2_residency_store(dst_scales, expected_scales.data(), scale_bytes);
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
    {
        uint8_t actual = 0;
        const size_t i = fpga_p2_residency_mismatch(verify_qs, expected_qs.data(), qs_bytes, &actual);
        if (i != qs_bytes) {
            g_p2_residency_build_us += now_us() - build0;
            fpga_p2_residency_log(true, "VERIFY_FAIL", "kind=qs offset=%zu expected=0x%02x actual=0x%02x", i,
                                  expected_qs[i], actual);
            fpga_p2_residency_poison_slot(slot, "qs_readback_mismatch");
            g_p2_residency_build_failures++;
            g_p2_residency_misses++;
            g_p2_residency_miss_verify++;
            fpga_p2_residency_log(false, "MISS", "reason=qs_readback_mismatch action=direct_stage slot=%u", slot);
            return P2_WEIGHT_RESIDENCY_NO_SLOT;
        }
    }
    {
        uint8_t actual = 0;
        const size_t i = fpga_p2_residency_mismatch(verify_scales, expected_scales.data(), scale_bytes, &actual);
        if (i != scale_bytes) {
            g_p2_residency_build_us += now_us() - build0;
            fpga_p2_residency_log(true, "VERIFY_FAIL", "kind=scale offset=%zu expected=0x%02x actual=0x%02x", i,
                                  expected_scales[i], actual);
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

// Read-only admission probe for the DMA packing pipeline.  It never reserves
// a residency slot, writes DDR, polls hardware, or changes residency state.
static bool fpga_p2_residency_contains(const struct ggml_tensor * src0,
                                       int64_t                    row0,
                                       int                        rows,
                                       int64_t                    k_block0,
                                       int                        group_blocks,
                                       int                        group_beats,
                                       size_t                     weight_bytes) {
    if (!g_p2_weight_residency_enabled || !src0 || rows <= 0 || group_blocks <= 0 || group_beats <= 0) {
        return false;
    }
    const size_t scale_bytes = (size_t) rows * (size_t) group_blocks * sizeof(uint16_t);
    const uint64_t key_hash =
        fpga_p2_residency_key_hash(src0, row0, rows, k_block0, group_blocks, group_beats, weight_bytes, scale_bytes);
    const size_t bucket = fpga_p2_residency_bucket(key_hash);
    for (size_t probe = 0; probe < P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES; ++probe) {
        const uint32_t slot = g_p2_residency_index[(bucket + probe) & (P2_WEIGHT_RESIDENCY_INDEX_BUCKETS - 1U)];
        if (slot == P2_WEIGHT_RESIDENCY_NO_SLOT) return false;
        if (slot >= g_p2_resident_tiles.size()) return false;
        const fpga_p2_resident_tile_t & tile = g_p2_resident_tiles[slot];
        if (tile.key_hash != key_hash || !tile.enabled || tile.building || !tile.sealed || tile.poisoned ||
            tile.epoch != g_p2_weight_residency_epoch ||
            tile.stream_protocol != g_stream_protocol_version || tile.bitstream_id != g_bitstream_id ||
            tile.p2_abi != g_p2_stream_abi_signature) {
            continue;
        }
        if (fpga_p2_residency_identity_matches(tile, src0, row0, rows, k_block0, group_blocks, group_beats,
                                               weight_bytes, scale_bytes)) {
            return true;
        }
    }
    return false;
}

static void p2_trace_first_tile(const fpga_tile_job_t & job, const char * stage, const char * edge);

static void p2_trace_set_job_context(const fpga_tile_job_t & job);

static bool p2_trace_this_tile();

// Owned by one row/block tile's prompt-column loop, never by a ping/pong
// job slot. Only the direct WEIGHT_BASE payload is retained; ACT/PARAM and
// all device transfers still belong to each individual job.
struct fpga_prompt_weight_staging_t {
    bool valid = false;
    const ggml_tensor * src0 = nullptr;
    const void * data = nullptr;
    const void * ddr = nullptr;
    int64_t row0 = 0;
    int64_t k_block0 = 0;
    int rows = 0;
    int group_blocks = 0;
    size_t bytes = 0;
    uint64_t epoch = 0;
    uint32_t weight_off = WEIGHT_BASE;
    uint32_t scale_off = SPU_PARAM_BASE;
};

static bool fpga_prepare_q8_tile_job(fpga_tile_job_t &                 job,
                                     const struct ggml_tensor *        src0,
                                     const void *                      weight_data_base,
                                     const block_q8_0_t *              act_group,
                                     int64_t                           row0,
                                     int                               rows,
                                     int64_t                           k_block0,
                                     int                               group_blocks,
                                     int64_t                           col,
                                     uint32_t                          tile_id,
                                     int                               bank,
                                     fpga_stage_totals_t *             totals,
                                     fpga_prompt_weight_staging_t *    staging = nullptr,
                                     uint32_t                          weight_stage_off = WEIGHT_BASE,
                                     uint32_t                          scale_stage_off = SPU_PARAM_BASE);

static bool fpga_p2_dma_pipeline_admit(fpga_p2_dma_pipeline_lookahead_t & lookahead,
                                       const struct ggml_tensor *         src0,
                                       const void *                       weight_data_base,
                                       const block_q8_0_t *               act_group,
                                       int64_t                            row0,
                                       int                                rows,
                                       int64_t                            k_block0,
                                       int                                group_blocks,
                                       int64_t                            col,
                                       uint32_t                           tile_id,
                                       int                                bank,
                                       fpga_stage_totals_t *              totals,
                                       uint32_t                           weight_src_off,
                                       uint32_t                           scale_src_off);

static bool fpga_p2_dma_pipeline_consume(fpga_p2_dma_pipeline_lookahead_t & lookahead,
                                         fpga_tile_job_t &                   job,
                                         fpga_stage_totals_t *                totals,
                                         const struct ggml_tensor *             expected_src0,
                                         const void *                           expected_weight_data_base,
                                         int64_t                                expected_row0,
                                         int                                      expected_rows,
                                         int64_t                                expected_k_block0,
                                         int                                      expected_group_blocks,
                                         int64_t                                expected_col,
                                         uint32_t                               expected_tile_id);

static bool fpga_p2_dma_pipeline_finalize(fpga_p2_dma_pipeline_lookahead_t & lookahead);

static bool fpga_p2_dma_pipeline_geometry(size_t rows,
                                          size_t group_blocks,
                                          int    group_beats,
                                          size_t * weight_bytes,
                                          size_t * scale_bytes,
                                          size_t * pair_count,
                                          size_t * words_per_pair,
                                          size_t * scale_entries,
                                          bool p3 = false) {
    if (!weight_bytes || !scale_bytes || !pair_count || !words_per_pair || !scale_entries || rows == 0U ||
        rows > (size_t) g_vpu_max_rows || group_blocks == 0U || group_blocks > (size_t) INT_MAX ||
        group_blocks > (size_t) INT_MAX / (size_t) VPU_BLOCK_BEATS || group_beats <= 0 ||
        group_beats > g_vpu_max_beats) {
        return false;
    }
    if (rows > (size_t) std::numeric_limits<int>::max() ||
        group_beats != (int) (group_blocks * (size_t) VPU_BLOCK_BEATS)) {
        return false;
    }
    const size_t weight = weight_window_bytes_for_rows((int) rows, group_beats);
    if (rows > std::numeric_limits<size_t>::max() / group_blocks) {
        return false;
    }
    const size_t entries = rows * group_blocks;
    if (entries == 0U || entries > std::numeric_limits<size_t>::max() - ((size_t) VPU_RESULT_PACK_LANES - 1U)) {
        return false;
    }
    const size_t result_words =
        (entries + (size_t) VPU_RESULT_PACK_LANES - 1U) / (size_t) VPU_RESULT_PACK_LANES;
    if (result_words == 0U || result_words > std::numeric_limits<size_t>::max() / 16U) {
        return false;
    }
    const size_t scale = p3 ? fpga_p3_param_bytes(entries) : result_words * 16U;
    if (p3 && (group_blocks > 64U || scale > P3_BANK_BYTES)) return false;
    const size_t pairs = (rows + 1U) / 2U;
    const size_t pair_words = (size_t) group_beats * 8U;
    if (pairs == 0U || pair_words == 0U ||
        pairs > std::numeric_limits<size_t>::max() / pair_words ||
        pairs * pair_words > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        return false;
    }
    const size_t payload = pairs * pair_words * sizeof(uint32_t);
    if (weight == 0U || weight != payload || weight > P2_PIPELINE_WEIGHT_SLOT_BYTES ||
        scale == 0U || scale > P2_PIPELINE_SCALE_SLOT_BYTES || scale > P2_PIPELINE_PL_SCALE_BYTES ||
        !range_fits(WEIGHT_BASE, weight, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(SPU_PARAM_BASE, scale, SPU_PARAM_BASE,
                    SPU_PARAM_BASE + (uint32_t) P2_PIPELINE_PL_SCALE_BYTES)) {
        return false;
    }
    *weight_bytes = weight;
    *scale_bytes = scale;
    *pair_count = pairs;
    *words_per_pair = pair_words;
    *scale_entries = entries;
    return true;
}

// The DMA-pipeline caller also polls DMA and retires the previous result.  For
// large lookahead tiles, leave it a bounded 3/8 prefix and give the persistent
// helper the 5/8 suffix.  Smaller tiles retain the established half split.
static size_t fpga_p2_dma_pipeline_main_pair_count(size_t pair_count, size_t weight_bytes) {
    if (pair_count < 2U || weight_bytes < FPGA_P2_PACK_PARALLEL_MIN_BYTES) {
        return pair_count / 2U;
    }
    if (pair_count > std::numeric_limits<size_t>::max() / 3U) {
        return pair_count / 2U;
    }
    size_t main_pairs = (pair_count * 3U) / 8U;
    main_pairs = std::max<size_t>(1U, main_pairs);
    main_pairs = std::min(main_pairs, pair_count - 1U);
    return main_pairs;
}

static bool fpga_p2_dma_pipeline_slot_ranges(uint32_t weight_src_off,
                                              uint32_t scale_src_off,
                                              size_t   weight_bytes,
                                              size_t   scale_bytes) {
    if (weight_src_off < WEIGHT_BASE || weight_src_off >= WEIGHT_END ||
        scale_src_off < SPU_PARAM_BASE || scale_src_off >= SPU_PARAM_END ||
        (weight_src_off - WEIGHT_BASE) / P2_PIPELINE_WEIGHT_SLOT_BYTES >= P2_PIPELINE_WEIGHT_SLOT_COUNT ||
        (scale_src_off - SPU_PARAM_BASE) / P2_PIPELINE_SCALE_SLOT_BYTES >= P2_PIPELINE_SCALE_SLOT_COUNT) {
        return false;
    }
    const uint32_t weight_slot = (weight_src_off - WEIGHT_BASE) / P2_PIPELINE_WEIGHT_SLOT_BYTES;
    const uint32_t scale_slot = (scale_src_off - SPU_PARAM_BASE) / P2_PIPELINE_SCALE_SLOT_BYTES;
    if (weight_src_off != WEIGHT_BASE + weight_slot * P2_PIPELINE_WEIGHT_SLOT_BYTES ||
        scale_src_off != SPU_PARAM_BASE + scale_slot * P2_PIPELINE_SCALE_SLOT_BYTES ||
        weight_bytes == 0U || scale_bytes == 0U || weight_bytes > P2_PIPELINE_WEIGHT_SLOT_BYTES ||
        scale_bytes > P2_PIPELINE_SCALE_SLOT_BYTES ||
        !range_fits(weight_src_off, weight_bytes, WEIGHT_BASE + weight_slot * P2_PIPELINE_WEIGHT_SLOT_BYTES,
                    WEIGHT_BASE + (weight_slot + 1U) * P2_PIPELINE_WEIGHT_SLOT_BYTES) ||
        !range_fits(scale_src_off, scale_bytes, SPU_PARAM_BASE + scale_slot * P2_PIPELINE_SCALE_SLOT_BYTES,
                    SPU_PARAM_BASE + (scale_slot + 1U) * P2_PIPELINE_SCALE_SLOT_BYTES)) {
        return false;
    }
    return true;
}

// Emit the small ACT window only after the previously launched job has been
// drained.  WEIGHT and PARAM are prepared ahead of time, but ACT_BASE is
// shared by both jobs and must never be overwritten while the old job reads it.
static bool fpga_p2_dma_pipeline_prepare_activation(fpga_tile_job_t & job, fpga_stage_totals_t * totals) {
    if (!job.act_group || job.group_blocks <= 0 || job.group_beats <= 0 ||
        !range_fits(ACT_BASE, job.act_bytes, ACT_BASE, ACT_END) || !ddr_range_fits(ACT_BASE, job.act_bytes)) {
        return false;
    }
    const long long prep0 = now_us();
    if (job.event_prep_begin_us == 0) {
        job.event_prep_begin_us = p2_event_now_us();
    }
    for (int gb = 0; gb < job.group_blocks; ++gb) {
        const block_q8_0_t & act = job.act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act.qs + beat * VPU_NUM_LANES);
        }
    }
    mmio_fence();
    job.event_prep_done_us = p2_event_now_us();
    const long long elapsed = now_us() - prep0;
    if (totals) {
        totals->prep_act_pack_us += elapsed;
        totals->prep_us += elapsed;
    }
    return true;
}

static bool fpga_p2_dma_pipeline_admit(fpga_p2_dma_pipeline_lookahead_t & lookahead,
                                       const struct ggml_tensor *         src0,
                                       const void *                       weight_data_base,
                                       const block_q8_0_t *               act_group,
                                       int64_t                            row0,
                                       int                                rows,
                                       int64_t                            k_block0,
                                       int                                group_blocks,
                                       int64_t                            col,
                                       uint32_t                           tile_id,
                                       int                                bank,
                                       fpga_stage_totals_t *              totals,
                                       uint32_t                           weight_src_off,
                                       uint32_t                           scale_src_off) {
    if (lookahead.valid || lookahead.worker_submitted || !g_p2_pack_dma_pipeline_enabled || !src0 ||
        !weight_data_base || !act_group || rows <= 0 || group_blocks <= 0 || row0 < 0 || k_block0 < 0 || col < 0 ||
        !ddr_is_mapped()) {
        return false;
    }
    if ((size_t) group_blocks > (size_t) INT_MAX / (size_t) VPU_BLOCK_BEATS ||
        (src0->ne[1] < 0 || (uint64_t) row0 > (uint64_t) src0->ne[1] ||
         (uint64_t) rows > (uint64_t) src0->ne[1] - (uint64_t) row0) ||
        (src0->ne[0] < 0 || (src0->ne[0] % VPU_QK8_0) != 0 ||
         (uint64_t) k_block0 > (uint64_t) (src0->ne[0] / VPU_QK8_0) ||
         (uint64_t) group_blocks > (uint64_t) (src0->ne[0] / VPU_QK8_0) - (uint64_t) k_block0)) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }
    size_t weight_bytes = 0U;
    size_t scale_bytes = 0U;
    size_t pair_count = 0U;
    size_t words_per_pair = 0U;
    size_t scale_entries = 0U;
    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    if (!fpga_p2_dma_pipeline_geometry((size_t) rows, (size_t) group_blocks, group_beats,
                                       &weight_bytes, &scale_bytes, &pair_count, &words_per_pair, &scale_entries, g_p3_matmul_active) ||
        pair_count < 2U || !fpga_p2_dma_pipeline_slot_ranges(weight_src_off, scale_src_off, weight_bytes, scale_bytes) ||
        !ddr_range_fits(weight_src_off, weight_bytes) || !ddr_range_fits(scale_src_off, scale_bytes)) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }
    fpga_p2_scale_shape_t scale_shape = {
        scale_entries, (scale_entries + VPU_RESULT_PACK_LANES - 1U) / VPU_RESULT_PACK_LANES, scale_bytes,
        ((scale_entries + VPU_RESULT_PACK_LANES - 1U) / VPU_RESULT_PACK_LANES) * VPU_RESULT_PACK_LANES - scale_entries};
    const bool scale_shape_ok = g_p3_matmul_active ?
        fpga_p3_checked_scale_shape(rows, group_blocks, &scale_shape.entries, &scale_shape.bytes) :
        fpga_p2_checked_scale_shape(rows, group_blocks, (uint32_t) scale_entries, (uint32_t) scale_shape.words,
                                   scale_bytes, scale_src_off, &scale_shape);
    if (!scale_shape_ok || scale_shape.bytes != scale_bytes) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }
    if (fpga_p2_residency_contains(src0, row0, rows, k_block0, group_blocks, group_blocks * VPU_BLOCK_BEATS,
                                   weight_bytes)) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }

    fpga_p2_dma_pipeline_lookahead_t candidate = {};
    candidate.valid                  = true;
    candidate.generation             = 0U;
    candidate.source_epoch           = g_p2_weight_residency_epoch;
    candidate.pair_count             = pair_count;
    candidate.helper_pair_begin      = fpga_p2_dma_pipeline_main_pair_count(pair_count, weight_bytes);
    candidate.main_pair_next         = 0U;
    candidate.words_per_pair         = words_per_pair;
    candidate.expected_words         = pair_count * words_per_pair;
    candidate.expected_scale_entries = scale_entries;
    candidate.totals                 = totals;
    candidate.weight_data_base       = weight_data_base;
    candidate.weight_words           = ddr_checked_u32_ptr(weight_src_off, weight_bytes);
    candidate.scale_words            = ddr_checked_u32_ptr(scale_src_off, scale_bytes);
    candidate.job                    = {};
    candidate.job.bank               = bank & 1;
    candidate.job.job_id             = fpga_next_job_id();
    candidate.job.matmul_call_id     = g_active_matmul_call_id;
    candidate.job.graph_seq          = g_active_matmul_graph_seq;
    candidate.job.layer_id           = g_active_matmul_layer_id;
    candidate.job.shape_k            = g_active_matmul_shape_k;
    candidate.job.shape_n            = g_active_matmul_shape_n;
    candidate.job.shape_m            = g_active_matmul_shape_m;
    candidate.job.cpu_shadow_dst     = g_active_matmul_cpu_shadow;
    candidate.job.pingpong_scheduler  = g_active_matmul_pingpong;
    candidate.job.tensor_name         = g_active_matmul_tensor_name;
    candidate.job.tile_id             = tile_id;
    candidate.job.tensor_id           = fpga_tensor_id_from_ptr(src0);
    candidate.job.row0                = row0;
    candidate.job.rows                = rows;
    candidate.job.k_block0            = k_block0;
    candidate.job.group_blocks        = group_blocks;
    candidate.job.group_beats         = group_blocks * VPU_BLOCK_BEATS;
    candidate.job.col                 = col;
    candidate.job.act_bytes           = (size_t) candidate.job.group_beats * 16U;
    candidate.job.weight_bytes        = weight_bytes;
    candidate.job.scale_bytes         = scale_bytes;
    candidate.job.spu_result_bytes    = (size_t) rows * 16U;
    candidate.job.result_bytes        = (size_t) scale_shape.words * 16U;
    candidate.job.result_values       = (uint32_t) scale_entries;
    candidate.job.result_words        = (uint32_t) scale_shape.words;
    // P3 scale_words counts PARAM words, independently of result_words.
    candidate.job.scale_words         = (uint32_t) (scale_bytes / 16U);
    candidate.job.p3_split_scale      = g_p3_matmul_active;
    candidate.job.scratch_bytes       = g_p3_matmul_active ? fpga_p3_scratch_bytes(group_blocks) : 0U;
    candidate.job.p3_act_group        = g_p3_matmul_active ? act_group : nullptr;
    candidate.job.p3_scratch_valid    = nullptr;
    candidate.job.weight_src_off      = weight_src_off;
    candidate.job.scale_src_off       = scale_src_off;
    candidate.job.p2_residency_slot   = P2_WEIGHT_RESIDENCY_NO_SLOT;
    candidate.job.act_group            = act_group;
    candidate.job.src0                 = src0;
    candidate.job.event_prep_begin_us  = p2_event_now_us();

    if (!range_fits(ACT_BASE, candidate.job.act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(RESULT_BASE, (size_t) candidate.job.result_words * 16U, RESULT_BASE, RESULT_END) ||
        !range_fits(SPU_OUT_BASE, candidate.job.spu_result_bytes, SPU_OUT_BASE, SPU_OUT_END) ||
        !ddr_range_fits(ACT_BASE, candidate.job.act_bytes) ||
        !ddr_range_fits(RESULT_BASE, (size_t) candidate.job.result_words * 16U) ||
        !ddr_range_fits(SPU_OUT_BASE, candidate.job.spu_result_bytes)) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }

    const size_t helper_words = (pair_count - candidate.helper_pair_begin) * words_per_pair;
    const size_t helper_rows = (size_t) rows - candidate.helper_pair_begin * 2U;
    if (helper_rows == 0U || helper_rows > std::numeric_limits<size_t>::max() / (size_t) group_blocks ||
        helper_words == 0U) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }
    uint64_t generation = 0U;
    if (!fpga_p2_pack_worker_next_generation(&generation, false)) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }
    fpga_p2_pack_worker_task_t task = {};
    task.src0                    = src0;
    task.weight_data_base       = weight_data_base;
    task.activation_data_base   = act_group;
    task.row0                   = row0;
    task.k_block0               = k_block0;
    task.rows                   = rows;
    task.group_blocks           = group_blocks;
    task.group_beats            = group_blocks * VPU_BLOCK_BEATS;
    task.pair_begin             = candidate.helper_pair_begin;
    task.pair_end               = pair_count;
    task.dst_words              = candidate.weight_words;
    task.expected_words         = helper_words;
    task.generation             = generation;
    task.scale_words            = candidate.scale_words;
    task.expected_scale_entries = helper_rows * (size_t) group_blocks;
    task.fused_scale            = true;
    task.p3_scale               = candidate.job.p3_split_scale;
    task.p3_scale_halves        = reinterpret_cast<volatile uint16_t *>(candidate.scale_words);
    if (!fpga_p2_pack_worker_submit(task, false)) {
        ++g_p2_pack_dma_pipeline_rejected_jobs;
        return false;
    }
    candidate.generation       = generation;
    candidate.worker_submitted = true;
    lookahead                  = candidate;
    ++g_p2_pack_dma_pipeline_eligible_jobs;
    return true;
}

static bool fpga_p2_dma_pipeline_consume(fpga_p2_dma_pipeline_lookahead_t & lookahead,
                                         fpga_tile_job_t &                   job,
                                         fpga_stage_totals_t *                totals,
                                         const struct ggml_tensor *             expected_src0,
                                         const void *                           expected_weight_data_base,
                                         int64_t                                expected_row0,
                                         int                                      expected_rows,
                                         int64_t                                expected_k_block0,
                                         int                                      expected_group_blocks,
                                         int64_t                                expected_col,
                                         uint32_t                               expected_tile_id) {
    if (!lookahead.valid || lookahead.consumed || lookahead.worker_submitted || lookahead.main_pair_next !=
                                                                            lookahead.helper_pair_begin ||
        lookahead.main_written_words != lookahead.helper_pair_begin * lookahead.words_per_pair ||
        lookahead.main_scale_entries > lookahead.expected_scale_entries || lookahead.totals != totals ||
        lookahead.source_epoch != g_p2_weight_residency_epoch || lookahead.job.src0 != expected_src0 ||
        lookahead.weight_data_base != expected_weight_data_base || lookahead.job.row0 != expected_row0 ||
        lookahead.job.rows != expected_rows || lookahead.job.k_block0 != expected_k_block0 ||
        lookahead.job.group_blocks != expected_group_blocks || lookahead.job.col != expected_col ||
        lookahead.job.tile_id != expected_tile_id || lookahead.job.weight_src_off == 0U ||
        lookahead.job.scale_src_off == 0U || lookahead.job.bank != (int) (expected_tile_id & 1U) ||
        !fpga_p2_dma_pipeline_slot_ranges(lookahead.job.weight_src_off, lookahead.job.scale_src_off,
                                           lookahead.job.weight_bytes, lookahead.job.scale_bytes)) {
        return false;
    }
    job = lookahead.job;
    job.event_prep_begin_us = lookahead.job.event_prep_begin_us;
    lookahead.consumed = true;
    lookahead.valid = false;
    ++g_p2_pack_dma_pipeline_consumed_jobs;
    return true;
}

static bool fpga_p2_dma_pipeline_finalize(fpga_p2_dma_pipeline_lookahead_t & lookahead) {
    if (!lookahead.valid || lookahead.consumed || !lookahead.worker_submitted) {
        return !lookahead.valid || lookahead.consumed;
    }
    const long long finalize0 = now_us();
    fpga_dma_profile_finalize_begin();
    const long long finalize_mono0 = monotonic_now_us();
    const long long residual0 = now_us();
    while (lookahead.main_pair_next < lookahead.helper_pair_begin) {
        const size_t pair_end = std::min(lookahead.main_pair_next + P2_PIPELINE_PACK_BATCH_PAIRS,
                                         lookahead.helper_pair_begin);
        if (!fpga_p2_dma_pipeline_pack_batch(&lookahead, pair_end)) {
            return false;
        }
    }
    lookahead.residual_us += now_us() - residual0;

    size_t helper_words = 0U;
    size_t helper_scales = 0U;
    long long helper_wait = 0;
    long long helper_publication_mono_us = 0;
    if (!fpga_p2_pack_worker_wait(lookahead.generation, &helper_words, &lookahead.helper_service_us, &helper_wait,
                                  &helper_scales, &helper_publication_mono_us)) {
        return false;
    }
    lookahead.worker_submitted = false;
    const size_t helper_expected_words =
        (lookahead.pair_count - lookahead.helper_pair_begin) * lookahead.words_per_pair;
    const size_t helper_rows = lookahead.job.rows > (int) (lookahead.helper_pair_begin * 2U) ?
                                   (size_t) lookahead.job.rows - lookahead.helper_pair_begin * 2U :
                                   0U;
    const size_t helper_expected_scales = helper_rows * (size_t) lookahead.job.group_blocks;
    if (helper_words != helper_expected_words || helper_scales != helper_expected_scales) {
        return false;
    }
    if (lookahead.main_written_words != lookahead.helper_pair_begin * lookahead.words_per_pair ||
        lookahead.main_scale_entries != lookahead.expected_scale_entries - helper_expected_scales) {
        return false;
    }
    // scale_bytes is already the padded byte capacity of the source slot.
    // Each entry occupies one uint32_t in the packed row-major table; do not
    // multiply the byte-derived entry count by the number of 128-bit lanes a
    // result word contains.  That would clear four times the admitted range
    // and could overwrite the adjacent pipeline slot.
    if (lookahead.job.p3_split_scale) {
        const size_t half_capacity = lookahead.job.scale_bytes / sizeof(uint16_t);
        if (lookahead.expected_scale_entries > half_capacity) return false;
        volatile uint16_t * halves = reinterpret_cast<volatile uint16_t *>(lookahead.scale_words);
        for (size_t i = lookahead.expected_scale_entries; i < half_capacity; ++i) halves[i] = 0U;
    } else {
        const size_t scale_capacity_entries = lookahead.job.scale_bytes / sizeof(uint32_t);
        if (lookahead.expected_scale_entries > scale_capacity_entries) {
            return false;
        }
        for (size_t linear = lookahead.expected_scale_entries; linear < scale_capacity_entries; ++linear) {
            lookahead.scale_words[linear] = 0U;
        }
    }
    mmio_fence();
    if (lookahead.main_written_words > SIZE_MAX / sizeof(uint32_t)) {
        return false;
    }
    const size_t total_main_bytes = lookahead.main_written_words * sizeof(uint32_t);
    if (lookahead.callback_bytes > total_main_bytes ||
        g_p2_pack_dma_pipeline_residual_bytes > LLONG_MAX - (long long) (total_main_bytes - lookahead.callback_bytes)) {
        return false;
    }
    g_p2_pack_dma_pipeline_residual_bytes += (long long) (total_main_bytes - lookahead.callback_bytes);
    g_p2_pack_dma_pipeline_helper_us += lookahead.helper_service_us;
    g_p2_pack_dma_pipeline_residual_us += lookahead.residual_us;
    // This is caller-observed mutex/condition/scheduling elapsed time, not
    // helper CPU service time; the latter is tracked separately above.
    if (helper_wait >= 0 && g_p2_pack_dma_pipeline_helper_wait_us <= LLONG_MAX - helper_wait) {
        g_p2_pack_dma_pipeline_helper_wait_us += helper_wait;
    }
    if (finalize_mono0 > 0 && helper_publication_mono_us > 0 && finalize_mono0 > helper_publication_mono_us) {
        const long long publication_lead_us = finalize_mono0 - helper_publication_mono_us;
        if (publication_lead_us > 0 &&
            g_p2_pack_dma_pipeline_helper_publication_lead_us <= LLONG_MAX - publication_lead_us) {
            g_p2_pack_dma_pipeline_helper_publication_lead_us += publication_lead_us;
            ++g_p2_pack_dma_pipeline_helper_early_completions;
        }
    }
    if (lookahead.totals) {
        // Callback and helper service overlap the current DMA.  Count only
        // the host-side residual/finalization wall time in the ordinary prep
        // chart; the separate pipeline counters retain the overlapped work.
        const long long finalize_wall_us = now_us() - finalize0;
        lookahead.totals->prep_direct_weight_pack_us += lookahead.residual_us;
        lookahead.totals->prep_us += finalize_wall_us;
    }
    fpga_dma_profile_finalize_end(lookahead.residual_us, helper_wait, lookahead.helper_service_us,
                                   total_main_bytes, lookahead.callback_bytes);
    return true;
}

static void p2_trace_first_tile(const fpga_tile_job_t & job, const char * stage, const char * edge) {
    if (!p2_trace_this_tile()) {
        return;
    }
    LOGI("P2_TILE_TRACE stage=%s edge=%s job=%u bank=%d tile=%u", stage, edge, job.job_id, job.bank, job.tile_id);
    if (g_p2_terminal_trace_enabled) {
        fprintf(stderr, "[P2] stage=%s edge=%s job=%u bank=%d tile=%u\n", stage, edge, job.job_id, job.bank,
                job.tile_id);
        fflush(stderr);
    }
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

static void fpga_p3_split_scale_configure() {
    const char * value = getenv("FPGA_P3_SPLIT_SCALE");
    if (!value || strcmp(value, "0") == 0) g_p3_split_scale_enabled = false;
    else if (strcmp(value, "1") == 0) g_p3_split_scale_enabled = true;
    else fpga_fatal("FPGA_P3_SPLIT_SCALE must be 0 or 1");
    g_p3_matmul_active = false;
    g_token_timing_collection_enabled = g_token_timing_collection_enabled || g_p3_split_scale_enabled;
    g_p3_mode_switches = 0;
    g_p3_graph_mode_switches_start = 0;
}

// Fixed for one initialized host lifetime; fpga_init re-entry keeps the policy.
static bool g_compute_watch_enabled = true;

static bool fpga_compute_watch_configure() {
    const char * value = getenv("FPGA_COMPUTE_WATCH");
    const bool enabled = env_flag_enabled("FPGA_COMPUTE_WATCH");
    const bool disabled = env_flag_disabled("FPGA_COMPUTE_WATCH");
    if (value && value[0] != '\0' && !enabled && !disabled) {
        LOGE("FPGA_COMPUTE_WATCH=%s is invalid; use 1 (default ON) or 0 (OFF)", value);
        return false;
    }
    g_compute_watch_enabled = !disabled;
    LOGINIT("COMPUTE_WATCH enabled=%d independent_ip_timing=%s disabled_compute_fields=nan "
            "main_thread_completion_checks=retained",
            g_compute_watch_enabled ? 1 : 0, g_compute_watch_enabled ? "requested" : "unavailable");
    return true;
}

// Keep absent compute samples out of numeric log analysis, including derived
// utilization/cycle estimates. Internal counters stay additive; the API has a
// validity flag and uses -1 for disabled timing instead.
static double fpga_compute_report_us(long long us) {
    return g_compute_watch_enabled ? (double) us : std::numeric_limits<double>::quiet_NaN();
}

static bool fpga_measure_q8_compute_completion(fpga_tile_job_t & job) {
    job.measured_compute_us = 0;
    uint32_t status = 0;
    // Runs on the read-only completion watcher, independently of preparation.
    // VPU DONE precedes final SPU writes: include both in P2 computation.
    if (!wait_vpu_done(&status) || !wait_spu_stream_outputs(job)) {
        return false;
    }
    const long long elapsed = monotonic_now_us() - job.ip_start_us;
    if (elapsed <= 0) {
        LOGE("IP compute timer returned a non-positive interval job=%u", job.job_id);
        return false;
    }
    job.measured_compute_us = elapsed;
    return true;
}

// One outstanding compute job, matching the hardware protocol. The watcher
// owns a metadata snapshot (never a scheduler slot), and only reads MMIO.
// Main remains the sole owner of DMA, descriptors, START and result retirement.
static pthread_t g_compute_watch_thread;
static pthread_mutex_t g_compute_watch_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_compute_watch_cond = PTHREAD_COND_INITIALIZER;
static std::atomic<int> g_compute_watch_phase{0}; // 0 idle, 1 ready, 2 launched, 3 captured
static std::atomic<bool> g_compute_watch_stop{false};
static bool g_compute_watch_created = false;
static bool g_compute_watch_pending = false;
static bool g_compute_watch_ok = false;
static fpga_tile_job_t g_compute_watch_job = {};

static void * fpga_compute_watch_main(void *) {
    for (;;) {
        pthread_mutex_lock(&g_compute_watch_mutex);
        while (!g_compute_watch_pending && !g_compute_watch_stop.load()) {
            pthread_cond_wait(&g_compute_watch_cond, &g_compute_watch_mutex);
        }
        const bool stopping = g_compute_watch_stop.load();
        pthread_mutex_unlock(&g_compute_watch_mutex);
        if (stopping) {
            return nullptr;
        }
        // Ready handshake happens BEFORE START, excluding thread wake-up time.
        g_compute_watch_phase.store(1, std::memory_order_release);
        while (g_compute_watch_phase.load(std::memory_order_acquire) != 2) {
            if (g_compute_watch_stop.load()) {
                return nullptr;
            }
            sched_yield();
        }
        g_compute_watch_ok = fpga_measure_q8_compute_completion(g_compute_watch_job);
        pthread_mutex_lock(&g_compute_watch_mutex);
        g_compute_watch_pending = false;
        pthread_mutex_unlock(&g_compute_watch_mutex);
        g_compute_watch_phase.store(3, std::memory_order_release);
    }
}

static bool fpga_compute_watch_arm(const fpga_tile_job_t & job) {
    if (!g_compute_watch_enabled) {
        return true;
    }
    if (!g_compute_watch_created) {
        g_compute_watch_stop.store(false);
        if (pthread_create(&g_compute_watch_thread, nullptr, fpga_compute_watch_main, nullptr) != 0) {
            LOGE("Cannot create automatic IP completion watcher");
            return false;
        }
        g_compute_watch_created = true;
    }
    if (g_compute_watch_phase.load(std::memory_order_acquire) != 0) {
        LOGE("IP completion watcher still owns the preceding job");
        return false;
    }
    pthread_mutex_lock(&g_compute_watch_mutex);
    // Copy only the fields used by wait_spu_stream_outputs; no model data or
    // vectors are copied, and early scheduler errors cannot dangle a pointer.
    g_compute_watch_job.job_id = job.job_id;
    g_compute_watch_job.bank = job.bank;
    g_compute_watch_job.rows = job.rows;
    g_compute_watch_job.group_blocks = job.group_blocks;
    g_compute_watch_job.spu_stream_count_before = job.spu_stream_count_before;
    g_compute_watch_job.spu_stream_done_before = job.spu_stream_done_before;
    g_compute_watch_job.spu_stream_out_before = job.spu_stream_out_before;
    g_compute_watch_job.spu_stream_drop_before = job.spu_stream_drop_before;
    g_compute_watch_job.spu_stream_error_before = job.spu_stream_error_before;
    g_compute_watch_pending = true;
    pthread_cond_signal(&g_compute_watch_cond);
    pthread_mutex_unlock(&g_compute_watch_mutex);
    const long long begin = monotonic_now_us();
    while (g_compute_watch_phase.load(std::memory_order_acquire) != 1) {
        if (monotonic_now_us() - begin > g_ip_timeout_us) {
            LOGE("IP completion watcher readiness timeout");
            return false;
        }
        sched_yield();
    }
    return true;
}

static void fpga_compute_watch_launch(long long start_us) {
    if (!g_compute_watch_enabled) {
        return;
    }
    g_compute_watch_job.ip_start_us = start_us;
    g_compute_watch_phase.store(2, std::memory_order_release);
}

static bool fpga_compute_watch_collect(fpga_tile_job_t & job) {
    if (!g_compute_watch_enabled) {
        job.measured_compute_us = 0;
        return true;
    }
    while (g_compute_watch_phase.load(std::memory_order_acquire) != 3) {
        sched_yield(); // watcher uses the existing bounded hardware waits
    }
    const bool ok = g_compute_watch_ok && g_compute_watch_job.job_id == job.job_id;
    job.measured_compute_us = ok ? g_compute_watch_job.measured_compute_us : 0;
    g_compute_watch_phase.store(0, std::memory_order_release);
    return ok;
}

static void fpga_compute_watch_shutdown() {
    if (!g_compute_watch_created) {
        return;
    }
    pthread_mutex_lock(&g_compute_watch_mutex);
    g_compute_watch_stop.store(true);
    pthread_cond_signal(&g_compute_watch_cond);
    pthread_mutex_unlock(&g_compute_watch_mutex);
    pthread_join(g_compute_watch_thread, nullptr); // before any MMIO unmap
    g_compute_watch_created = false;
    g_compute_watch_pending = false;
    g_compute_watch_phase.store(0);
}

static bool fpga_submit_q8_tile_job(fpga_tile_job_t &     job,
                                    fpga_stage_totals_t * totals,
                                    const char *          tensor_name,
                                    int                   layer_id,
                                    int64_t               k,
                                    int64_t               n,
                                    int64_t               m,
                                    int                   attempt,
                                    const fpga_dma_poll_hook_t * poll_hook = nullptr) {
    job.event_submit_begin_us = p2_event_now_us();
    if (!fpga_p2_residency_job_still_valid(job)) {
        LOGE("P2_RESIDENCY_POST_SELECTION_FAIL job=%u tile=%u action=no_start_no_cpu_fallback", job.job_id,
             job.tile_id);
        return false;
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

    const long long dma_act0       = now_us();
    const long long event_dma_act0 = p2_event_now_us();
    job.event_input_transfer_begin_us = event_dma_act0;
    p2_trace_first_tile(job, "ACT_DMA", "before");
    const bool first_p2_act_detail =
        g_p2_first_act_dma_trace_enabled && g_p2_init_requested && job.tile_id == 0U &&
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
    const bool act_dma_ok =
        job.act_bank_reused ? true :
        poll_hook ? fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) ACT_BASE, LMM_BASE_PHYS + (uint64_t) ACT_BASE,
                                  job.act_bytes, "ACT", poll_hook) :
                    fpga_dma_write_to_ip(ACT_BASE, job.act_bytes, "ACT");
    if (first_p2_act_detail) {
        g_p2_first_act_dma_trace_active = false;
        g_p2_first_act_dma_trace_done   = true;
    }
    if (!act_dma_ok) {
        return false;
    }
    const long long event_dma_act_done = p2_event_now_us();
    p2_trace_first_tile(job, "ACT_DMA", "after");
    p2_event_trace(job, "ACT_DMA_DONE", event_dma_act_done, "act_dma_us",
                   job.act_bank_reused ? 0 : event_dma_act_done - event_dma_act0);
    const long long dma_act1 = now_us();

    const long long dma_weight0       = now_us();
    const long long event_dma_weight0 = p2_event_now_us();
    if (!job.weight_bank_reused) {
        p2_trace_first_tile(job, "WEIGHT_DMA", "before");
    }
    vpu_select_banks(job.bank, job.bank);
    if (!job.weight_bank_reused &&
        !fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) job.weight_src_off, LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE,
                       job.weight_bytes, "WEIGHT", poll_hook)) {
        if (poll_hook && poll_hook->context) {
            fpga_p2_dma_pipeline_invalidate(
                static_cast<fpga_p2_dma_pipeline_lookahead_t *>(poll_hook->context));
        }
        return false;
    }
    const long long event_dma_weight_done = p2_event_now_us();
    if (!job.weight_bank_reused) {
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
    long long scratch_stage_us = 0;
    long long scratch_dma_us = 0;
    if (job.p3_split_scale) {
        const long long scratch_stage_start = now_us();
        const int b = job.bank & 1;
        if (!job.p3_scratch_valid || !job.p3_act_group || job.group_blocks <= 0 || job.group_blocks > 64 ||
            job.scratch_bytes != fpga_p3_scratch_bytes(job.group_blocks) || job.scale_bytes > P3_BANK_BYTES) {
            LOGE("P3 submit invalid scale/scratch geometry job=%u tile=%u bank=%d", job.job_id, job.tile_id, b);
            return false;
        }
        if (!(job.act_bank_reused && job.p3_scratch_valid[b])) {
            const uint32_t scratch_off = SPU_SCRATCH_BASE + (uint32_t) b * P3_BANK_BYTES;
            volatile uint16_t * scratch = reinterpret_cast<volatile uint16_t *>(ddr_ptr(scratch_off, job.scratch_bytes));
            for (int gb = 0; gb < job.group_blocks; ++gb) scratch[gb] = (uint16_t) job.p3_act_group[gb].d;
            for (size_t gb = (size_t) job.group_blocks; gb < job.scratch_bytes / sizeof(uint16_t); ++gb) scratch[gb] = 0U;
            mmio_fence();
            const long long scratch_dma_start = now_us();
            if (!fpga_dma_copy(DDR_BASE_PHYS + scratch_off, LMM_BASE_PHYS + scratch_off,
                               job.scratch_bytes, "SPU_SCRATCH", poll_hook)) return false;
            scratch_dma_us = now_us() - scratch_dma_start;
            job.p3_scratch_valid[b] = true;
            if (totals) {
                totals->scratch_dma_us += scratch_dma_us;
                totals->scratch_bytes += (long long) job.scratch_bytes;
                totals->scratch_desc += (long long) (job.scratch_bytes / g_zdma_max_transfer_bytes +
                                                    (job.scratch_bytes % g_zdma_max_transfer_bytes != 0U));
            }
        }
        scratch_stage_us = now_us() - scratch_stage_start;
    }
    const long long event_dma_scale0 = p2_event_now_us();
    p2_trace_first_tile(job, "SPU_PARAM_DMA", "before");
    if (job.p3_split_scale) {
        const uint32_t param_off = SPU_PARAM_BASE + (uint32_t) (job.bank & 1) * P3_BANK_BYTES;
        if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) job.scale_src_off, LMM_BASE_PHYS + param_off,
                           job.scale_bytes, "SPU_SCALE", poll_hook)) return false;
    } else if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) job.scale_src_off, LMM_BASE_PHYS + (uint64_t) SPU_PARAM_BASE,
                       job.scale_bytes, "SPU_SCALE", poll_hook)) {
        return false;
    }
    const long long event_dma_scale_done = p2_event_now_us();
    p2_trace_first_tile(job, "SPU_PARAM_DMA", "after");
    p2_event_trace(job, "SPU_PARAM_DMA_DONE", event_dma_scale_done, "spu_param_dma_us",
                   event_dma_scale_done - event_dma_scale0);
    p2_event_trace(job, "WRITE_DONE", event_dma_scale_done, "write_phase_elapsed_us",
                   event_dma_scale_done - job.event_input_transfer_begin_us);
    const long long dma_scale1 = now_us();

    job.dma_act_us              = job.act_bank_reused ? 0 : dma_act1 - dma_act0;
    job.dma_weight_us           = job.weight_bank_reused ? 0 : dma_weight1 - dma_weight0;
    job.dma_scale_us            = dma_scale1 - dma_scale0 - scratch_stage_us;
    job.spu_stream_count_before = vpu_rd32(REG_SPU_STREAM_COUNT);
    job.spu_stream_done_before  = vpu_rd32(REG_SPU_STREAM_DONE);
    job.spu_stream_out_before   = vpu_rd32(REG_SPU_STREAM_OUT);
    job.spu_stream_drop_before  = vpu_rd32(REG_SPU_STREAM_DROP);
    job.spu_stream_error_before = vpu_rd32(REG_SPU_STREAM_ERROR);
    const int bank_index = job.bank & 1;
    const long long h2ip_us = job.dma_act_us + job.dma_weight_us + job.dma_scale_us + scratch_dma_us;
    if (totals) {
        totals->dma_act_us += job.dma_act_us;
        totals->dma_weight_us += job.dma_weight_us;
        totals->dma_scale_us += job.dma_scale_us;
        totals->bank_h2ip_us[bank_index] += h2ip_us;
        totals->bank_jobs[bank_index]++;
        totals->activation_bytes += job.act_bank_reused ? 0 : job.act_bytes;
        totals->weight_bytes += job.weight_bank_reused ? 0 : job.weight_bytes;
        totals->scale_bytes += job.scale_bytes;
        totals->result_bytes += job.spu_result_bytes;
        totals->vpu_runs++;
    }
    if (g_pingpong_timing_enabled) {
        fpga_log_line(
            true, "PINGPONG_TIMING", false,
            "phase=host_to_ip graph_seq=%d layer=%d tensor=%s tile=%u job=%u bank=%d bank_role=%s "
            "h2ip_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f "
            "act_bytes=%zu weight_bytes=%zu scale_bytes=%zu",
            job.graph_seq, layer_id, tensor_name ? tensor_name : "?", job.tile_id, job.job_id, bank_index,
            p2_bank_label(bank_index), (double) h2ip_us / 1000.0,
            (double) job.dma_act_us / 1000.0, (double) job.dma_weight_us / 1000.0,
            (double) job.dma_scale_us / 1000.0, job.act_bank_reused ? 0 : job.act_bytes,
            job.weight_bank_reused ? 0 : job.weight_bytes, job.scale_bytes);
    }

    vpu_write_tile_descriptor(job, FPGA_SLOT_COMPUTING, FPGA_SLOT_FREE, 0x00000101U);
    mmio_fence();
    p2_trace_first_tile(job, "VPU_LAUNCH", "before");
    if (!fpga_compute_watch_arm(job)) {
        return false;
    }
    job.ip_start_us = now_us();
    const long long compute_start_us = g_compute_watch_enabled ? monotonic_now_us() : 0;
    vpu_wr32(REG_CTRL, CTRL_START);
    mmio_fence();
    fpga_compute_watch_launch(compute_start_us);
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
    p2_trace_first_tile(job, "VPU_LAUNCH", "after");
    p2_event_trace(job, "START", job.event_launch_us, "none_us", 0);

    if (g_ip_timing_enabled && should_log_detail_run(job.tile_id)) {
        LOGIP(
            "submit tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u job=%u bank=%d attempt=%d rows=%d col_beats=%d "
            "mode=0x%x act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f",
            tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, job.tile_id,
            job.job_id, job.bank, attempt, job.rows, job.group_beats, launch_mode,
            (double) job.dma_act_us / 1000.0,
            (double) job.dma_weight_us / 1000.0, (double) job.dma_scale_us / 1000.0);
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
    const long long drain_start = now_us();
    if (job.output_dma_ready || job.hardware_released || job.result_consumed) {
        LOGE("P2 result ownership violation before drain job=%u", job.job_id);
        return false;
    }
    if (!fpga_compute_watch_collect(job)) {
        LOGE("IP completion measurement failed job=%u", job.job_id);
        return false;
    }
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
    job.event_vpu_done_us = p2_event_now_us();
    p2_trace_first_tile(job, "VPU_DONE_WAIT", "after");
    p2_event_trace(job, "VPU_DONE", job.event_vpu_done_us, "launch_to_vpu_done_observed_us",
                   job.event_vpu_done_us - job.event_launch_us);
    job.vpu_status    = vpu_status;
    job.ip_compute_us = job.measured_compute_us;
    if (!vpu_verify_done_job(job, vpu_status)) {
        return false;
    }
    p2_trace_first_tile(job, "SPU_FINALITY_WAIT", "before");
    if (!wait_spu_stream_outputs(job)) {
        g_pl_scale_stream_drops += (long long) (vpu_rd32(REG_SPU_STREAM_DROP) - job.spu_stream_drop_before);
        g_pl_scale_stream_errors += (long long) (vpu_rd32(REG_SPU_STREAM_ERROR) - job.spu_stream_error_before);
        return false;
    }
    const long long drain_wait_us = now_us() - drain_start;
    if (totals) {
        totals->drain_wait_us += drain_wait_us;
        if (drain_wait_us > 5) ++totals->drain_wait_jobs;
    }
    // drain_wait excludes the following P3 lock-release polling interval.
    if (job.p3_split_scale) {
        const long long lock_start = now_us();
        while (true) {
            const uint32_t status = vpu_rd32(REG_P3_STREAM_STATUS);
            if ((status & P3_STATUS_SPLIT_ENABLE) == 0U || now_us() - lock_start > P3_LOCK_WAIT_TIMEOUT_US) {
                LOGE("P3 drain lock/mode failure job=%u tile=%u bank=%d status=0x%08x",
                     job.job_id, job.tile_id, job.bank, status);
                return false;
            }
            if ((status & P3_STATUS_LOCK_VALID) == 0U) break;
        }
        if (totals) ++totals->p3_jobs;
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
            fpga_compute_report_us(job.ip_compute_us) / 1000.0, (double) job.dma_result_us / 1000.0,
            job.spu_result_bytes);
    }

    if (g_ip_timing_enabled && should_log_detail_run(job.tile_id)) {
        LOGIP(
            "complete tensor=%s layer=%d shape=K%lldxN%lldxM%lld tile=%u job=%u bank=%d attempt=%d rows=%d "
            "col_beats=%d ip_ms=%.3f spu_out_dma_ms=%.3f status=0x%08x progress=0x%08x spu_out=%u",
            tensor_name ? tensor_name : "?", layer_id, (long long) k, (long long) n, (long long) m, job.tile_id,
            job.job_id, job.bank, attempt, job.rows, job.group_beats, fpga_compute_report_us(job.ip_compute_us) / 1000.0,
            (double) job.dma_result_us / 1000.0, vpu_status, vpu_rd32(REG_PROGRESS), vpu_rd32(REG_SPU_STREAM_OUT));
    }
    job.output_dma_ready = true;
    return true;
}

static bool fpga_release_pl_scaled_q8_tile_job(fpga_tile_job_t & job, bool pending_host_read = false) {
    if (!job.output_dma_ready || job.hardware_released || (!job.result_consumed && !pending_host_read)) {
        LOGE("P2 result ownership violation before release job=%u", job.job_id);
        return false;
    }
    p2_trace_first_tile(job, "DESCRIPTOR_FREE", "before");
    // A pending DDR read may outlive hardware ownership only after an exact
    // FREE/FREE readback. It must not issue any descriptor writes after B starts.
    if (!fpga_write_post_spu_descriptor(job, FPGA_SLOT_FREE, FPGA_SLOT_FREE, 0x00000000U,
                                       "final_free_free", true, pending_host_read)) {
        return false;
    }
    job.hardware_released = true;
    job.event_retire_us = p2_event_now_us();
    p2_trace_first_tile(job, "DESCRIPTOR_FREE", "after");
    p2_event_trace(job, "DESCRIPTOR_RETIRED", job.event_retire_us, "submit_to_retire_us",
                   job.event_retire_us - job.event_submit_begin_us);
    return true;
}

static bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t & job,
                                                  std::vector<float> &    accum,
                                                  fpga_stage_totals_t *   totals,
                                                  bool release_after_read = true) {
    if (!job.output_dma_ready || job.result_consumed || job.hardware_released == release_after_read) {
        LOGE("P2 result ownership violation before consume job=%u", job.job_id);
        return false;
    }
    if (job.rows <= 0 || (size_t) job.rows > (SPU_OUT_END - SPU_OUT_BASE) / 16U ||
        (size_t) job.rows * 16U > job.spu_result_bytes || job.col < 0 ||
        (size_t) job.rows > accum.size() ||
        (uint64_t) job.col > (accum.size() - (size_t) job.rows) / (size_t) job.rows) {
        LOGE("P2 result span rejected job=%u", job.job_id);
        return false;
    }
    const long long result0   = now_us();
    const long long event_host_read_accum0 = p2_event_now_us();
    float *         accum_col = &accum[(size_t) (job.col * job.rows)];
    const volatile uint32_t * result_src = reinterpret_cast<const volatile uint32_t *>(
        ddr_ptr(SPU_OUT_BASE, (size_t) job.rows * 16U));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    const bool wide_result = (reinterpret_cast<uintptr_t>(result_src) & 7U) == 0U;
#else
    (void) result_src;
#endif
    for (int row = 0; row < job.rows; ++row) {
        uint16_t row_id = 0xffffU;
        int64_t q16;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        if (wide_result) {
            const volatile uint64_t * wide = reinterpret_cast<const volatile uint64_t *>(result_src) + (size_t) row * 2U;
            const uint64_t low = wide[0];
            const uint64_t high = wide[1];
            row_id = (uint16_t) low;
            const uint64_t bits = (low >> 16U) | (high << 48U);
            memcpy(&q16, &bits, sizeof(q16));
        } else
#endif
        {
            q16 = ddr_read_spu_q16_row(SPU_OUT_BASE + (uint32_t) row * 16U, &row_id);
        }
        if (row_id != (uint16_t) row) {
            LOGE("SPU_OUT row mismatch job=%u tile=%u bank=%d row=%d got_row=%u q16=%lld", job.job_id, job.tile_id,
                 job.bank, row, (unsigned) row_id, (long long) q16);
            return false;
        }
        accum_col[(size_t) row] += (float) q16 * (1.0f / 65536.0f);
    }
    const long long result1 = now_us();
    const long long event_host_read_accum_done = p2_event_now_us();
    const long long host_read_us = result1 - result0;
    job.host_result_us = host_read_us;
    job.result_consumed = true;
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
    if (release_after_read && !fpga_release_pl_scaled_q8_tile_job(job)) {
        return false;
    }
    const long long finish_us = p2_event_now_us();
    p2_event_trace(job, "TILE_FINISH", finish_us, "submit_to_finish_us", finish_us - job.event_submit_begin_us);
    return true;
}

// The P2 contract is a tile-level transport/numerical qualification, not a
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
    fpga_tile_job_t slots[2]          = {};
    fpga_tile_job_t * running        = nullptr;
    long long reused_jobs = 0;
    uint64_t avoided_weight_bytes = 0;
    const bool overlap_results = g_p2_result_overlap_enabled && g_vpu_descriptor_supported;
    // Both banks retain the same complete decode vector across row tiles.
    // The largest row tile sets the tightest single-group result limit.
    const bool reuse_act = g_p2_act_bank_reuse_enabled && m == 1 &&
        n > 0 && nb > 0 && nb <= INT_MAX && g_vpu_max_rows > 0 &&
        packed_q8_group_blocks_for_rows((int) std::min<int64_t>(g_vpu_max_rows, n), (int) nb) == nb;
    bool act_bank_valid[2] = { false, false };
    bool p3_scratch_valid[2] = { false, false };
    const uint32_t p3_reject_before = g_p3_matmul_active ? vpu_rd32(REG_P3_REJECT_COUNT) : 0U;
    // B's input transfers do not touch A's completed DDR result. The single
    // host thread consumes A before returning to any drain of B. No new DMA
    // buffer, worker, or concurrent register owner is introduced.
    static_assert(ACT_END <= SPU_OUT_BASE && WEIGHT_END <= SPU_OUT_BASE &&
                      SPU_OUT_END <= SPU_PARAM_BASE,
                  "result overlap requires disjoint input/output DDR windows");

    const auto store_rows = [&](int64_t completed_row0, int completed_rows) {
        const long long store0 = now_us();
        for (int64_t col = 0; col < m; ++col) {
            const float * accum_col = &accum[(size_t) (col * completed_rows)];
            for (int row = 0; row < completed_rows; ++row) {
                store_dst_value(dst, completed_row0 + row, col, accum_col[(size_t) row]);
            }
        }
        if (totals) {
            totals->host_accum_us += now_us() - store0;
        }
    };
    const auto can_carry_to_row = [&](int next_rows) {
        // Prompt columns retain their established staging/bank-reuse loop.
        // The existing off switch also restores the original row drains.
        if (!overlap_results || m != 1) {
            return false;
        }
        if (!g_p2_weight_residency_enabled || g_p2_residency_next_slot >= g_p2_resident_tiles.size()) {
            return true;
        }
        // Previously the first tile of this row could build a resident entry
        // with hardware idle. Keep that opportunity whenever its payload fits,
        // even if it might already be resident. Do not change cache admission.
        const int blocks = packed_q8_group_blocks_for_rows(next_rows, (int) nb);
        size_t qs_padded = 0;
        size_t scales_padded = 0;
        if (!fpga_p2_residency_align_up(weight_window_bytes_for_rows(next_rows, blocks * VPU_BLOCK_BEATS),
                                        WEIGHT_CACHE_ALIGN, &qs_padded) ||
            !fpga_p2_residency_align_up((size_t) next_rows * (size_t) blocks * sizeof(uint16_t),
                                        WEIGHT_CACHE_ALIGN, &scales_padded) ||
            qs_padded > std::numeric_limits<size_t>::max() - scales_padded ||
            g_p2_weight_residency_budget_mb < 0) {
            return false;
        }
        const uint64_t budget_end = (uint64_t) WEIGHT_CACHE_BASE +
                                   (uint64_t) g_p2_weight_residency_budget_mb * 1024ULL * 1024ULL;
        if (budget_end > P2_WEIGHT_RESIDENCY_END) {
            return false;
        }
        uint32_t allocation_end = 0;
        return !fpga_p2_residency_range_end(g_p2_residency_next_off, qs_padded + scales_padded,
                                           (uint32_t) budget_end, &allocation_end);
    };

    const bool pipeline_route = g_p2_pack_dma_pipeline_enabled && m == 1;
    // Keep the packed lookahead alive across both K and row loops.  The
    // admission lambda may select the first tile of the next row when the
    // existing overlap policy carries the current result forward.
    fpga_p2_dma_pipeline_lookahead_t lookahead = {};
    fpga_p2_dma_pipeline_scope_guard pipeline_guard{ &lookahead };

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        if (!running) {
            accum.assign((size_t) (m * rows), 0.0f);
        }

        for (int64_t ib0 = 0; ib0 < nb;) {
            const int remaining_blocks = (int) (nb - ib0);
            const int group_blocks     = packed_q8_group_blocks_for_rows(rows, remaining_blocks);
            const int group_beats      = group_blocks * VPU_BLOCK_BEATS;

            // No staging identity survives a tile/tensor transition or an
            // error return. Submit completes source DMA before next prepare;
            // drain only writes the disjoint result window. Contract/restage
            // diagnostics use the separate single-bank path, not this loop.
            fpga_prompt_weight_staging_t staging;
            const auto admit_next_tile = [&](const fpga_tile_job_t & current) {
                if (!pipeline_route || lookahead.valid || current.weight_bank_reused ||
                    current.p2_residency_hit ||
                    current.weight_bytes > P2_PIPELINE_WEIGHT_SLOT_BYTES ||
                    current.scale_bytes > P2_PIPELINE_SCALE_SLOT_BYTES || current.scale_bytes > P2_PIPELINE_PL_SCALE_BYTES ||
                    current.weight_src_off != WEIGHT_BASE +
                                                  (current.tile_id & 1U) * P2_PIPELINE_WEIGHT_SLOT_BYTES) {
                    return true;
                }
                int64_t next_col = current.col + 1;
                int64_t next_k   = current.k_block0;
                int64_t next_row0 = current.row0;
                int     next_rows  = current.rows;
                if (next_col >= m) {
                    next_col = 0;
                    next_k += current.group_blocks;
                }
                if (next_k >= nb) {
                    const int64_t candidate_row0 = current.row0 + current.rows;
                    if (candidate_row0 >= n) {
                        return true;
                    }
                    next_row0 = candidate_row0;
                    next_rows = (int) std::min<int64_t>(g_vpu_max_rows, n - candidate_row0);
                    if (!can_carry_to_row(next_rows)) {
                        return true;
                    }
                    next_k = 0;
                }
                const int next_group_blocks = packed_q8_group_blocks_for_rows(next_rows, (int) (nb - next_k));
                const int next_group_beats  = next_group_blocks * VPU_BLOCK_BEATS;
                const uint32_t next_tile_id = current.tile_id + 1U;
                const uint32_t next_slot = next_tile_id & 1U;
                const uint32_t next_weight_off = WEIGHT_BASE + next_slot * P2_PIPELINE_WEIGHT_SLOT_BYTES;
                const uint32_t next_scale_off  = SPU_PARAM_BASE + next_slot * P2_PIPELINE_SCALE_SLOT_BYTES;
                const block_q8_0_t * next_act = &act_blocks_all[(size_t) (next_col * nb + next_k)];
                size_t next_weight_bytes = 0U;
                size_t next_scale_bytes = 0U;
                size_t next_pairs = 0U;
                size_t next_words_per_pair = 0U;
                size_t next_scale_entries = 0U;
                if (!fpga_p2_dma_pipeline_geometry((size_t) next_rows, (size_t) next_group_blocks, next_group_beats,
                                                   &next_weight_bytes, &next_scale_bytes, &next_pairs,
                                                   &next_words_per_pair, &next_scale_entries, g_p3_matmul_active) ||
                    next_pairs < 2U ||
                    fpga_p2_residency_contains(src0, next_row0, next_rows, next_k, next_group_blocks, next_group_beats,
                                               next_weight_bytes)) {
                    return true;
                }
                (void) next_scale_bytes;
                (void) next_words_per_pair;
                (void) next_scale_entries;
                return fpga_p2_dma_pipeline_admit(
                    lookahead, src0, src0->data, next_act, next_row0, next_rows, next_k, next_group_blocks,
                    next_col, next_tile_id, (int) next_slot, totals, next_weight_off, next_scale_off);
            };
            // Both banks start unknown for every row/K tile. Only this column
            // loop writes their weights; ACT, scales, and result retirement
            // do not modify weight RAM. No validity survives a return/retry.
            bool weight_bank_loaded[2] = { false, false };
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

                bool pipeline_prepared = false;
                if (lookahead.valid) {
                    if (lookahead.job.tile_id != tile_id ||
                        !fpga_p2_dma_pipeline_consume(lookahead, prepared, totals, src0, src0->data, row0, rows, ib0,
                                                      group_blocks, col, tile_id)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    pipeline_prepared = true;
                } else {
                    size_t pipeline_weight_bytes = 0U;
                    size_t pipeline_scale_bytes = 0U;
                    size_t pipeline_pairs = 0U;
                    size_t pipeline_words_per_pair = 0U;
                    size_t pipeline_scale_entries = 0U;
                    const bool use_pipeline_slots =
                        pipeline_route &&
                        fpga_p2_dma_pipeline_geometry((size_t) rows, (size_t) group_blocks, group_beats,
                                                      &pipeline_weight_bytes, &pipeline_scale_bytes, &pipeline_pairs,
                                                      &pipeline_words_per_pair, &pipeline_scale_entries, g_p3_matmul_active) &&
                        pipeline_pairs >= 2U;
                    const uint32_t stage_slot = tile_id & 1U;
                    if (!fpga_prepare_q8_tile_job(
                            prepared, src0, src0->data, act_group, row0, rows, ib0, group_blocks, col,
                            tile_id, bank, totals, m > 1 ? &staging : nullptr,
                            use_pipeline_slots ? WEIGHT_BASE + stage_slot * P2_PIPELINE_WEIGHT_SLOT_BYTES : WEIGHT_BASE,
                            use_pipeline_slots ? SPU_PARAM_BASE + stage_slot * P2_PIPELINE_SCALE_SLOT_BYTES :
                                                  SPU_PARAM_BASE)) {
                        return false;
                    }
                }

                prepared.weight_bank_reused = m > 1 && weight_bank_loaded[bank];
                prepared.act_bank_reused = reuse_act && act_bank_valid[bank];
                prepared.p3_scratch_valid = p3_scratch_valid;

                // Consumed B is finalized; C may pack into A's old input slot
                // while A drains to disjoint SPU output. Keep C metadata in lookahead.
                const bool admission_attempted_early = pipeline_prepared && running != nullptr;
                if (admission_attempted_early && !admit_next_tile(prepared)) {
                    fpga_p2_dma_pipeline_invalidate(&lookahead);
                    return false;
                }

                if (running) {
                    if (g_p1_sched_summary_enabled) {
                        // A pair is a scheduler handoff opportunity, whether
                        // it overlaps or must submit serially after drain.
                        g_p1_sched_summary.pingpong_pairs++;
                    }
                    if (!fpga_wait_and_drain_q8_tile_job(*running, totals, tensor_name, layer_id, k, n, m, 0)) {
                        return false;
                    }
                    if (overlap_results) {
                        if (!fpga_release_pl_scaled_q8_tile_job(*running, true)) {
                            return false;
                        }
                    } else {
                        if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals)) {
                            return false;
                        }
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
                    prepared.handoff_prev_output_ready_us = running->event_spu_finality_us;
                    prepared.handoff_prev_retire_us = running->event_retire_us;
                    if (pipeline_prepared && !fpga_p2_dma_pipeline_prepare_activation(prepared, totals)) {
                        return false;
                    }
                    if (!admission_attempted_early && !admit_next_tile(prepared)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    const fpga_dma_poll_hook_t poll_hook = {
                        fpga_p2_dma_pipeline_poll, lookahead.valid ? static_cast<void *>(&lookahead) : nullptr };
                    if (!fpga_submit_q8_tile_job(prepared, totals, tensor_name, layer_id, k, n, m, 0,
                                                 lookahead.valid ? &poll_hook : nullptr)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    if (overlap_results) {
                        // No MMIO writes for A after B starts; no B result DMA
                        // before this read completes. Preserve accumulation order.
                        if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals, false)) {
                            return false;
                        }
                        if (totals) {
                            ++totals->result_overlap_jobs;
                            totals->result_overlap_host_us += running->host_result_us;
                        }
                    }
                    if (running->row0 != row0) {
                        // B is now executing, but accum still belongs to A.
                        // Store A with its own row count (including tails)
                        // before reusing the buffer. B cannot drain until the
                        // next iteration, after this transition is complete.
                        store_rows(running->row0, running->rows);
                        accum.assign((size_t) (m * rows), 0.0f);
                    }
                    // Finalize the next tile only after A's pending result has
                    // been accumulated and, at a row transition, stored.  The
                    // helper still owns C's inactive DDR source slot, and the
                    // scope guard joins it if this finalization fails.
                    if (lookahead.valid && !fpga_p2_dma_pipeline_finalize(lookahead)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    running = &prepared;
                } else {
                    if (pipeline_prepared && !fpga_p2_dma_pipeline_prepare_activation(prepared, totals)) {
                        return false;
                    }
                    if (!admit_next_tile(prepared)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    const fpga_dma_poll_hook_t poll_hook = {
                        fpga_p2_dma_pipeline_poll, lookahead.valid ? static_cast<void *>(&lookahead) : nullptr };
                    if (!fpga_submit_q8_tile_job(prepared, totals, tensor_name, layer_id, k, n, m, 0,
                                                 lookahead.valid ? &poll_hook : nullptr)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    if (lookahead.valid && !fpga_p2_dma_pipeline_finalize(lookahead)) {
                        fpga_p2_dma_pipeline_invalidate(&lookahead);
                        return false;
                    }
                    running = &prepared;
                }

                // Submit has completed all required DMA and launched this job.
                // A failed drain/submit returns before publishing validity.
                weight_bank_loaded[bank] = true;
                act_bank_valid[bank] = reuse_act;
                if (!reuse_act) p3_scratch_valid[bank] = false;
                if (prepared.act_bank_reused) {
                    ++g_p2_act_bank_reused_jobs;
                    g_p2_act_bank_avoided_bytes += prepared.act_bytes;
                }
                if (prepared.weight_bank_reused) {
                    ++reused_jobs;
                    avoided_weight_bytes += prepared.weight_bytes;
                }
                tile_id++;
            }

            ib0 += group_blocks;
        }

        const int64_t next_row0 = row0 + rows;
        if (running && next_row0 < n &&
            can_carry_to_row((int) std::min<int64_t>(g_vpu_max_rows, n - next_row0))) {
            continue;
        }
        if (running) {
            if (!fpga_wait_and_drain_q8_tile_job(*running, totals, tensor_name, layer_id, k, n, m, 0)) {
                return false;
            }
            if (!fpga_accumulate_pl_scaled_q8_tile_job(*running, accum, totals)) {
                return false;
            }
            running = nullptr;
        }

        store_rows(row0, rows);
    }
    if (g_p3_matmul_active) {
        const uint32_t delta = vpu_rd32(REG_P3_REJECT_COUNT) - p3_reject_before;
        if (totals) totals->p3_reject_delta += delta;
        if (delta != 0U) fpga_fatal("P3 reject count changed");
    }
    if (m > 1) {
        fpga_log_prompt_weight_reuse(tensor_name, m, reused_jobs, avoided_weight_bytes);
    }
    return true;
}

static bool fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_qualify_tiles(const struct ggml_tensor *        src0,
                                                               const std::vector<block_q8_0_t> & act_blocks_all,
                                                               const void *                      weight_data_base,
                                                               fpga_stage_totals_t *             totals,
                                                               const char *                      tensor_name,
                                                               int                               layer_id,
                                                               int64_t                           k,
                                                               int64_t                           n,
                                                               int64_t                           m,
                                                               int64_t                           nb,
                                                               std::vector<float> & accum) {
    if (g_pl_scale_contract_check_limit <= 0 || g_p2_tile_limit <= 0 ||
        g_p2_tile_contract_boundary_reached ||
        fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit)) {
        LOGE("P2 tile qualification is not active; refusing tile launch");
        return false;
    }
    g_p3_matmul_active = false;
    if (!fpga_enforce_p2_stream_mode("P2 tile qualification route")) {
        return false;
    }
    uint32_t tile_id           = 0;
    // Whole-tile qualification never fabricates a partial matrix result.
    g_scratch.contract_actual.clear();

    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        accum.assign((size_t) m * (size_t) rows, 0.0f);
        for (int64_t ib0 = 0; ib0 < nb;) {
            const int group_blocks = packed_q8_group_blocks_for_rows(rows, (int) (nb - ib0));
            for (int64_t col = 0; col < m; ++col) {
                fpga_tile_job_t            job       = {};
                const block_q8_0_t * const act_group = &act_blocks_all[(size_t) (col * nb + ib0)];
                if (should_log_detail_run(tile_id)) {
                    LOGSTAGE(
                        "P2_SINGLE_BANK_TILE tensor=%s layer=%d row0=%lld rows=%d k_block0=%lld group_blocks=%d "
                        "tile=%u job_policy=one_outstanding bank=0 route=vpu_to_spu scope=qualification",
                        tensor_name ? tensor_name : "?", layer_id, (long long) row0, rows, (long long) ib0,
                        group_blocks, tile_id);
                }
                if (!fpga_prepare_q8_tile_job(job, src0, weight_data_base, act_group, row0, rows, ib0, group_blocks,
                                              col, tile_id, 0, totals) ||
                    !fpga_submit_q8_tile_job(job, totals, tensor_name, layer_id, k, n, m, 0) ||
                    !fpga_wait_and_drain_q8_tile_job(job, totals, tensor_name, layer_id, k, n, m, 0)) {
                    return false;
                }
                g_pl_scale_jobs++;
                g_pl_scale_banks++;
                if (!fpga_pl_scale_contract_verify_q16_tile(job, weight_data_base, tensor_name, layer_id)) {
                    return false;
                }
                ++g_p2_tile_q16_checks;
                if (!fpga_accumulate_pl_scaled_q8_tile_job(job, accum, totals)) {
                    return false;
                }
                if (fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit)) {
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
        }
    }
    if (g_p2_tile_contract_boundary_reached ||
        fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit)) {
        LOGE("P2_TILE_LIMIT_INTERNAL_FAIL tensor=%s layer=%d matrix_tiles=%u cumulative_q16=%lld tile_limit=%d "
             "boundary=%d reason=matrix_returned_without_exact_boundary",
             tensor_name ? tensor_name : "?", layer_id, tile_id, g_p2_tile_q16_checks, g_p2_tile_limit,
             g_p2_tile_contract_boundary_reached ? 1 : 0);
        return false;
    }
    LOGINIT(
        "P2_TILE_MATRIX_CONTINUE tensor=%s layer=%d matrix_tiles=%u cumulative_q16=%lld tile_limit=%d "
        "remaining=%lld matrix_value_contract=not_attempted "
        "action=cpu_shadow_current_matmul_then_continue_qualification",
        tensor_name ? tensor_name : "?", layer_id, tile_id, g_p2_tile_q16_checks, g_p2_tile_limit,
        (long long) g_p2_tile_limit - g_p2_tile_q16_checks);
    return true;
}

static bool fpga_hw_q8_0_matmul_dma_to_ip(const struct ggml_tensor * src0,
                                          const struct ggml_tensor * src1,
                                          const struct ggml_tensor * dst,
                                          fpga_stage_totals_t *      totals,
                                          const char *               tensor_name,
                                          int                        layer_id) {
    const int64_t k  = src0->ne[0];
    const int64_t n  = src0->ne[1];
    const int64_t m  = src1->ne[1];
    const int64_t nb = k / VPU_QK8_0;

    std::vector<block_q8_0_t> & act_blocks_all = g_scratch.act_blocks_all;
    std::vector<float> &        act_scales     = g_scratch.act_scales;
    std::vector<float> &        accum          = g_scratch.accum;

    const bool pl_scale_contract_active =
        (g_pl_scale_contract_check_limit > 0) && !g_p2_tile_contract_boundary_reached &&
        !fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit);
    g_contract_source_validation_failed = false;
    const size_t weight_tensor_bytes    = ggml_nbytes(src0);
    const void * weight_data_base       = src0->data;
    if (pl_scale_contract_active) {
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
    // P2 production requires ping-pong; tile qualification uses its dedicated
    // one-outstanding runner.
    const bool use_pl_scale_path = g_spu_q8_scale_stream_supported;

    const long long quant0 = now_us();
    if (!ensure_quantized_activation_matrix(src1, m, k, act_blocks_all, act_scales, !use_pl_scale_path, totals,
                                            tensor_name, layer_id)) {
        return false;
    }
    if (totals) {
        totals->prep_us += now_us() - quant0;
    }

    if (use_pl_scale_path) {
        const uint32_t want_mode = g_p3_split_scale_enabled && m == 1 && !pl_scale_contract_active ? 1U : 0U;
        const long long mode_switches_before = g_p3_mode_switches;
        if (!fpga_enforce_p2_stream_mode("P2 dispatch route", nullptr, want_mode)) {
            return false;
        }
        g_p3_matmul_active = want_mode == 1U;
        if (totals) totals->p3_mode_switches += g_p3_mode_switches - mode_switches_before;
        if (!pl_scale_contract_active) {
            if (!g_pingpong_scheduler_enabled) {
                LOGE("ordinary P2 requires admitted ping-pong scheduling; no serialized fallback");
                return false;
            }
            return fpga_hw_q8_0_matmul_dma_to_ip_pipelined(src0, dst, act_blocks_all, act_scales, totals,
                                                           tensor_name, layer_id, k, n, m, nb, accum);
        }
        const bool pl_ok = fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_qualify_tiles(
            src0, act_blocks_all, weight_data_base, totals, tensor_name, layer_id, k, n, m, nb, accum);
        if (pl_ok && pl_scale_contract_active &&
            !fpga_p2_cumulative_tile_state_consistent(g_p2_tile_q16_checks, g_p2_tile_limit,
                                                       g_p2_tile_contract_boundary_reached)) {
            LOGE("P2_TILE_LIMIT_INTERNAL_FAIL cumulative_q16=%lld tile_limit=%d boundary=%d expected_boundary=%d "
                 "reason=cumulative_boundary_inconsistent",
                 g_p2_tile_q16_checks, g_p2_tile_limit, g_p2_tile_contract_boundary_reached ? 1 : 0,
                 fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit) ? 1 : 0);
            return false;
        }
        return pl_ok;
    }

    LOGE("P2 scale stream is unavailable; refusing raw execution");
    return false;
}

int fpga_init(void) {
    pthread_mutex_lock(&g_mutex);
    if (env_flag_enabled("FPGA_PL_SCALE_DISABLE")) {
        fpga_fatal("FPGA_PL_SCALE_DISABLE is no longer supported: raw-v52 path removed");
    }
    const bool dma_mapped = dma_is_mapped();
    const bool vpu_mapped = vpu_is_mapped();
    const bool ddr_mapped = ddr_is_mapped();
    const bool all_mapped = dma_mapped && vpu_mapped && ddr_mapped;
    const bool any_mapped = dma_mapped || vpu_mapped || ddr_mapped;

    // main.cpp can initialize the host before ggml-cpu's pthread_once path
    // reaches here.  A completed host is immutable on re-entry: do not parse
    // mutable environment policy, reset counters, remap, touch MMIO,
    // or change ZDMA ownership.  The marker intentionally uses only retained
    // host state, so re-entry itself performs no board access.
    if (g_fpga_init_complete) {
        if (!all_mapped) {
            fpga_fatal(
                "FPGA_INIT_REENTRY_PARTIAL_FAIL init_complete=1 dma_mapped=%d vpu_mapped=%d ddr_mapped=%d "
                "action=fail_closed_no_config_mmio_dma_map_change",
                dma_mapped ? 1 : 0, vpu_mapped ? 1 : 0, ddr_mapped ? 1 : 0);
        }
        LOGINIT(
            "FPGA_INIT_REENTRY version=%s init_complete=1 dma_mapped=1 vpu_mapped=1 ddr_mapped=1 "
            "action=return_existing_state_no_config_reset_mmio_dma_map_change",
            FPGA_HOST_TRACE_VERSION);
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }
    if (any_mapped) {
        fpga_fatal(
            "FPGA_INIT_REENTRY_PARTIAL_FAIL init_complete=0 dma_mapped=%d vpu_mapped=%d ddr_mapped=%d "
            "action=fail_closed_no_config_mmio_dma_map_change",
            dma_mapped ? 1 : 0, vpu_mapped ? 1 : 0, ddr_mapped ? 1 : 0);
    }
    if (!fpga_compute_watch_configure()) {
        fpga_fatal("Invalid FPGA compute watcher policy; initialization stopped before hardware mapping");
    }
    if (!fpga_dma_profile_configure()) {
        fpga_fatal("Invalid FPGA DMA profiling policy; initialization stopped before hardware mapping");
    }
    g_init_verbose = env_flag_enabled("FPGA_INIT_VERBOSE");
    // This checks the host-only P2 PARAM packing contract before any UIO
    // mapping of DDR, VPU, or ZDMA can be attempted.
    if (!fpga_p2_scale_layout_self_test()) {
        fpga_fatal("P2 scale layout self-test failed before hardware mapping; refusing FPGA initialization");
    }
    LOGINIT("P2_SCALE_LAYOUT_SELFTEST pass cases=8 modulo=0,1,2,3 odd_even_multirow=1 max_rows=256 max_group_blocks=64 "
            "invalid_shape_rejected_before_mutation=1");
    if (getenv("FPGA_P2_HOST_PREPACK") || getenv("FPGA_P2_HOST_PREPACK_MB")) {
        fpga_fatal(
            "FPGA_P2_HOST_PREPACK and FPGA_P2_HOST_PREPACK_MB are retired after measured regression; remove both variables");
    }
    g_p2_init_requested = true;
    g_committed_stream_mode = -1;
    g_p2_first_act_dma_trace_enabled    = env_flag_enabled("FPGA_P2_FIRST_ACT_TRACE");
    g_p2_first_act_dma_trace_active     = false;
    g_p2_first_act_dma_trace_done       = false;
    g_p2_tile_limit                     = 0;
    g_p2_allow_multitile                = false;
    g_p2_tile_contract_boundary_reached = false;
    g_p2_tile_q16_checks                = 0;
    g_p2_matrix_contract_checks         = 0;
    g_p2_tile_trace_enabled             = false;
    g_p2_terminal_trace_enabled         = env_flag_enabled("FPGA_P2_TERMINAL_TRACE");
    g_p2_boundary_diagnostics_enabled   = false;
    g_p2_event_trace_enabled             = env_flag_enabled("FPGA_P2_EVENT_TRACE");
    const char * const p2_pack_dma_pipeline_env = getenv("FPGA_P2_PACK_DMA_PIPELINE");
    const bool p2_pack_dma_pipeline_enable_env = env_flag_enabled("FPGA_P2_PACK_DMA_PIPELINE");
    const bool p2_pack_dma_pipeline_disable_env = env_flag_disabled("FPGA_P2_PACK_DMA_PIPELINE");
    if (p2_pack_dma_pipeline_enable_env && p2_pack_dma_pipeline_disable_env) {
        fpga_fatal(
            "FPGA_P2_PACK_DMA_PIPELINE=1 conflicts with FPGA_P2_PACK_DMA_PIPELINE=0; select exactly one pipeline policy");
    }
    if (p2_pack_dma_pipeline_env && p2_pack_dma_pipeline_env[0] != '\0' &&
        !p2_pack_dma_pipeline_enable_env && !p2_pack_dma_pipeline_disable_env) {
        fpga_fatal(
            "FPGA_P2_PACK_DMA_PIPELINE=%s is invalid; use 1 to request the pipeline or 0 to disable its default policy",
            p2_pack_dma_pipeline_env);
    }
    // Keep the explicit request until the hardware capability gate below is
    // known.  An unset variable becomes the production default only when the
    // admitted P2 route can safely provide the two source slots.
    g_p2_pack_dma_pipeline_requested = p2_pack_dma_pipeline_enable_env;
    g_p2_pack_dma_pipeline_enabled = false;
    g_p2_result_overlap_enabled = !env_flag_disabled("FPGA_P2_RESULT_OVERLAP");
    g_p2_act_bank_reuse_enabled = !env_flag_disabled("FPGA_P2_ACT_BANK_REUSE");
    g_p2_act_bank_reused_jobs = 0;
    g_p2_act_bank_avoided_bytes = 0;
    g_p1_sched_summary_enabled           = env_flag_enabled("FPGA_P1_SCHED_SUMMARY");
    g_pingpong_timing_enabled            = env_flag_enabled("FPGA_PINGPONG_TIMING");
    g_bottleneck_summary_enabled         = !env_flag_disabled("FPGA_BOTTLENECK_SUMMARY");
    g_token_timing_enabled               = !env_flag_disabled("FPGA_TOKEN_TIMING");
    g_token_timing_collection_enabled    = g_token_timing_enabled || g_pingpong_timing_enabled ||
                                           g_bottleneck_summary_enabled || fpga_dma_profile_enabled();
    g_summary_detail_every               = env_int_value("FPGA_SUMMARY_DETAIL_EVERY", FPGA_DEFAULT_SUMMARY_DETAIL_EVERY, 0, 1000000);
    g_p1_sched_summary                   = {};
    fpga_token_timing_reset();
    g_summary_detail_decode_tokens = 0;
    g_summary_detail_after_error = false;
    g_fpga_perf_decode = {};
    if (g_token_timing_collection_enabled) {
        LOGINIT(
            "TIMING_CONFIG enabled=1 pingpong_detail=%d breakdown=%d output=/tmp/fpga_debug.log "
            "graph_boundary=fpga_advance_sequence_position host_to_ip=ACT_plus_WEIGHT_plus_SCALE "
            "result_read=SPU_OUT_DMA_plus_host_result_read graph_log=%d detail_every=%d",
            g_pingpong_timing_enabled ? 1 : 0, g_bottleneck_summary_enabled ? 1 : 0,
            g_token_timing_enabled ? 1 : 0, g_summary_detail_every);
    }
    if (g_bottleneck_summary_enabled) {
        LOGINIT(
            "BREAKDOWN_CONFIG enabled=1 mode=aggregate_per_sequence per_tile_logs=0 "
            "metrics=prep_decomposition+scheduler_handoff+tensor_category+zdma_descriptors");
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
    g_p2_pack_fused_jobs = 0;
    g_p2_pack_fused_us = 0;
    g_p2_pack_fused_scale_entries = 0;
    g_p2_pack_dma_pipeline_eligible_jobs = 0;
    g_p2_pack_dma_pipeline_consumed_jobs = 0;
    g_p2_pack_dma_pipeline_callback_bytes = 0;
    g_p2_pack_dma_pipeline_callback_us = 0;
    g_p2_pack_dma_pipeline_helper_us = 0;
    g_p2_pack_dma_pipeline_helper_wait_us = 0;
    g_p2_pack_dma_pipeline_residual_us = 0;
    g_p2_pack_dma_pipeline_residual_bytes = 0;
    g_p2_pack_dma_pipeline_rejected_jobs = 0;
    g_p2_pack_dma_pipeline_helper_publication_lead_us = 0;
    g_p2_pack_dma_pipeline_helper_early_completions = 0;
    g_pack_detail.fill(0);
    g_pack_detail_snapshot.fill(0);
    g_pack_detail_decode.fill(0);

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
    const bool p2_weight_residency_env_requested = true;
    g_p2_weight_residency_diagnostic = env_flag_enabled("FPGA_P2_WEIGHT_RESIDENCY_DIAGNOSTIC");
    g_p2_weight_residency_env_requested = p2_weight_residency_env_requested;
    // Board-qualified production policy: P2 residency is always requested.
    // FPGA_P2_WEIGHT_RESIDENCY_DIAGNOSTIC remains an independent opt-in for
    // qualification/diagnostic behavior and does not gate residency admission.
    g_p2_weight_residency_requested = true;
    g_p2_weight_residency_enabled   = false;
    g_p2_residency_trace_enabled    = env_flag_enabled("FPGA_P2_RESIDENCY_TRACE");
    g_p2_residency_verify_metadata  = env_flag_enabled("FPGA_P2_RESIDENCY_VERIFY_METADATA");
    g_p2_resident_tiles.fill({});
    g_p2_residency_index.fill(P2_WEIGHT_RESIDENCY_NO_SLOT);
    g_p2_resident_tile_count = 0;
    g_p2_residency_next_slot = 0;
    g_p2_residency_next_off = WEIGHT_CACHE_BASE;
    g_p2_residency_extent_marked = 0;
    g_p2_residency_graph_builds_start = 0;
    g_p2_residency_graph_build_us_start = 0;
    g_p2_residency_decode_builds = 0;
    g_p2_residency_decode_build_us = 0;
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
    g_vocab_projection_cpu_bypass =
        !env_flag_disabled("FPGA_VOCAB_PROJECTION_CPU") && !env_flag_enabled("FPGA_ACCELERATE_VOCAB");
    g_log_flush_every               = env_int_value("FPGA_LOG_FLUSH_EVERY", 256, 1, 1000000);
    fpga_log_set_flush_every(g_log_flush_every);
    g_profile_every                 = env_int_value("FPGA_PROFILE_EVERY", FPGA_DEFAULT_PROFILE_EVERY, 0, 1000000);
    g_ip_status_every               = env_int_value("FPGA_IP_STATUS_EVERY", FPGA_DEFAULT_STATUS_EVERY, 0, 1000000);
    g_detail_every                  = env_int_value("FPGA_DETAIL_EVERY", FPGA_DEFAULT_DETAIL_EVERY, 0, 1000000);
    g_pl_scale_contract_check_limit = env_int_value("FPGA_PL_SCALE_CONTRACT_CHECK", 0, 0, 1000000);
    if (g_pl_scale_contract_check_limit > 1) {
        fpga_fatal(
            "FPGA_PL_SCALE_CONTRACT_CHECK=%d is unsupported; set it to 1 and select the "
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
            "P2 qualification tile_limit=%d allow_multitile=%d matrix_f32_check=not_attempted after_limit=cpu",
            g_p2_tile_limit, g_p2_allow_multitile ? 1 : 0);
    } else {
        g_p2_tile_trace_enabled = g_p2_boundary_diagnostics_enabled;
    }
    const bool pipeline_enable_env  = env_flag_enabled("FPGA_PIPELINE_ENABLE");
    const bool pipeline_disable_env = env_flag_enabled("FPGA_PIPELINE_DISABLE");
    if (pipeline_enable_env && pipeline_disable_env) {
        fpga_fatal("FPGA_PIPELINE_ENABLE=1 conflicts with FPGA_PIPELINE_DISABLE=1; select exactly one scheduler policy");
    }
    if (pipeline_disable_env && g_p2_init_requested &&
        g_pl_scale_contract_check_limit == 0) {
        fpga_fatal("ordinary P2 requires ping-pong; FPGA_PIPELINE_DISABLE=1 is supported only for P2 qualification diagnostics");
    }
    if ((g_pl_scale_contract_check_limit > 0) && !fpga_model_tensor_validation_passed()) {
        fpga_fatal(
            "qualification loader-validation handshake is missing; no board MMIO was mapped. Rebuild and deploy the "
            "coupled frontend files ggml/src/ggml-cpu/fpga_host.{cpp,h} and src/llama-model-loader.cpp from the same "
            "source tree, then rerun the requested qualification");
    }
    if (g_pl_scale_contract_check_limit > 0) {
        LOGINIT(
            "Model tensor validation passed p2_pl_scale=%d",
            g_pl_scale_contract_check_limit);
    }
    if (g_pl_scale_contract_check_limit > 0) {
        if (pipeline_enable_env) {
            fpga_fatal(
                "P2 tile qualification requires one-outstanding execution; FPGA_PIPELINE_ENABLE is not permitted during "
                "FPGA_PL_SCALE_CONTRACT_CHECK");
        }
    }
    fpga_p2_init_breadcrumb("phase=config_loader_gate pass p2_contract_limit=%d loader_validation=%s",
                            g_pl_scale_contract_check_limit,
                            g_pl_scale_contract_check_limit > 0 ? "pass" : "not_required");
    g_contract_cpu_shadow_dst = g_pl_scale_contract_check_limit > 0;
    if (g_contract_cpu_shadow_dst) {
        LOGINIT("Qualification output: CPU; hardware checks limited by configured scope");
    }
    g_dma_trace_enabled =
        env_flag_enabled("FPGA_DMA_AUDIT") || g_pl_scale_contract_check_limit > 0;
    memset(g_dma_trace, 0, sizeof(g_dma_trace));
    g_dma_trace_sequence       = 0;
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
    g_abort_on_cpu_fallback            = !env_flag_disabled("FPGA_ABORT_ON_CPU_FALLBACK");
    g_activation_input_integrity_check = env_flag_enabled("FPGA_INPUT_INTEGRITY_CHECK");
    if ((g_pl_scale_contract_check_limit > 0) && !g_activation_input_integrity_check) {
        // Qualification deliberately exercises M>1 layouts and must prove
        // that the host/VPU path leaves each live F32 source intact.  Ordinary
        // primary P2 remains snapshot-free unless explicitly requested.
        g_activation_input_integrity_check = true;
        LOGINIT("Input integrity check enabled for qualification");
    }

    fpga_p2_init_breadcrumb("phase=mapping begin");
    if (!configure_ddr_mapping_policy()) {
        fpga_fatal("DDR cache/mapping policy is invalid; refusing FPGA initialization");
    }
    if (!map_registers_dma_ddr()) {
        fpga_fatal("ZDMA DDR-to-IP FPGA init failed; refusing CPU fallback");
    }
    if (g_p2_init_requested && g_ddr_mapping_kind != fpga_mapping_kind::UIO_PHYSICAL) {
        fpga_fatal(
            "P2 requires a verified physical UIO ddr_high mapping; map_kind=%s. "
            "No ZDMA initialization or data-plane transfer was issued",
            fpga_mapping_kind_name(g_ddr_mapping_kind));
    }
    fpga_p2_init_breadcrumb("phase=mapping pass my_ip_source=%s my_ip_size=0x%zx", g_vpu_map_source.c_str(),
                            g_vpu_map_size);
    fpga_uncached_read_probe();
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
    const uint32_t p3_sig = vpu_rd32(REG_P3_SPLIT_SCALE_ABI);
    const uint32_t p3_mode = vpu_rd32(REG_STREAM_MODE);
    const uint32_t p3_status = vpu_rd32(REG_P3_STREAM_STATUS);
    const uint32_t p3_reject = vpu_rd32(REG_P3_REJECT_COUNT);
    fpga_log_line(true, "P3_PROBE", true,
                  "p3_sig=0x%08x stream_mode=0x%08x p3_status=0x%08x p3_reject=0x%08x",
                  p3_sig, p3_mode, p3_status, p3_reject);
    fpga_p3_split_scale_configure();
    if (g_p3_split_scale_enabled && p3_sig != P3_SPLIT_SCALE_ABI_ID) {
        fpga_fatal("P3 split-scale ABI mismatch sig=0x%08x expected=0x%08x", p3_sig, P3_SPLIT_SCALE_ABI_ID);
    }
    LOGINIT("P3_CONFIG enabled=%d sig=0x%08x", g_p3_split_scale_enabled ? 1 : 0, p3_sig);
    g_spu_stream_status       = vpu_rd32(REG_SPU_STREAM_STATUS);
    g_spu_word_capacity       = (spu_caps >> 16) & 0xFFFFU;
    fpga_p2_init_breadcrumb(
        "phase=identity_reads pass limits=0x%08x caps=0x%08x spu_caps=0x%08x protocol=0x%08x bitstream_id=0x%08x "
        "p2_abi=0x%08x stream_status=0x%08x",
        limits, caps, spu_caps, g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature,
        g_spu_stream_status);
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
    const bool fpga_identity_compatible            = bitstream_id_compatible && stream_protocol_compatible && p2_abi_compatible;
    if (!fpga_identity_compatible) {
        fpga_fatal(
            "FPGA ABI v3 admission failed bitstream_id=0x%08x expected_id=0x%08x id_ok=%d stream_protocol=0x%08x "
            "required_protocol=%u protocol_ok=%d p2_abi=0x%08x required_p2_abi=0x%08x abi_ok=%d; refusing all "
            "self-test and GEMV launches without CPU fallback",
            g_bitstream_id, FPGA_EXPECTED_BITSTREAM_ID, bitstream_id_compatible ? 1 : 0, g_stream_protocol_version,
            FPGA_REQUIRED_STREAM_PROTOCOL_VERSION, stream_protocol_compatible ? 1 : 0, g_p2_stream_abi_signature,
            FPGA_REQUIRED_P2_STREAM_ABI, p2_abi_compatible ? 1 : 0);
    }
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
    const bool p2_admission_compatible = p2_abi_compatible && spu_stream_quiescent && spu_word_capacity_compatible &&
                                          p2_two_row_transport_compatible;
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
    if (g_p3_split_scale_enabled && (!g_spu_q8_scale_stream_supported || !g_vpu_pingpong_supported)) {
        fpga_fatal("P3 split-scale requires admitted SPU scale stream and ping-pong banks");
    }

    // Ordinary P2 requires the admitted descriptor/bank scheduler. Explicit
    // disable remains available for qualification and raw diagnostic routes.
    g_pingpong_scheduler_enabled = g_p2_init_requested && g_vpu_pingpong_supported && g_vpu_descriptor_supported &&
                                   fpga_identity_compatible && p2_admission_compatible && !pipeline_disable_env;
    const bool p2_pack_dma_pipeline_compatible =
        g_pl_scale_contract_check_limit == 0 && g_pingpong_scheduler_enabled &&
        g_spu_q8_scale_stream_supported && g_p2_pack_workers_requested == 2;
    if (p2_pack_dma_pipeline_enable_env && !p2_pack_dma_pipeline_compatible) {
        fpga_fatal(
            "FPGA_P2_PACK_DMA_PIPELINE=1 requires admitted P2 ping-pong, no contract qualification, and "
            "FPGA_P2_PACK_WORKERS=2; refusing an ambiguous shared-staging policy");
    }
    // The pipeline is now the default only for the same production route that
    // already proved its P2 scheduler, scale stream, and two-worker contract.
    // Explicit zero is the only normal way to retain the serialized pack path.
    g_p2_pack_dma_pipeline_requested = !p2_pack_dma_pipeline_disable_env && p2_pack_dma_pipeline_compatible;
    if (g_p2_pack_dma_pipeline_requested) {
        LOGINIT(
            "P2_PACK_DMA_PIPELINE requested=1 source=%s admission=waiting_for_persistent_helper "
            "mode=decode_direct_large_tiles "
            "slots=2 weight_slot_bytes=%u scale_slot_bytes=%u callback_batch_pairs=%zu residency_builds=deferred",
            p2_pack_dma_pipeline_enable_env ? "explicit" : "default",
            P2_PIPELINE_WEIGHT_SLOT_BYTES, P2_PIPELINE_SCALE_SLOT_BYTES, P2_PIPELINE_PACK_BATCH_PAIRS);
    } else {
        g_p2_pack_dma_pipeline_enabled = false;
        LOGINIT(
            "P2_PACK_DMA_PIPELINE enabled=0 reason=%s",
            p2_pack_dma_pipeline_disable_env ? "explicit_opt_out" : "incompatible_admission");
    }
    if (pl_scale_requested && !g_spu_q8_scale_stream_supported) {
        fpga_fatal(
            "Required P2 admission or stream/descriptors/protocol are incompatible: "
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
        const bool residency_mapping_fits = g_ddr_map_size == expected_map;
        const bool residency_admitted = g_p2_init_requested && fpga_identity_compatible && p2_admission_compatible &&
                                        g_spu_q8_scale_stream_supported &&
                                        g_ddr_mapping_kind == fpga_mapping_kind::UIO_PHYSICAL &&
                                        residency_mapping_fits && g_ddr_advertised_size >= expected_map;
        if (!residency_admitted) {
            fpga_fatal(
                "P2 residency admission failed requested=1 map_kind=%s map_size=0x%zx expected_map=0x%zx "
                "advertised=0x%zx raw_abi=%d p2_abi=%d stream=%d; refusing cache selection or CPU fallback",
                fpga_mapping_kind_name(g_ddr_mapping_kind), g_ddr_map_size, expected_map, g_ddr_advertised_size,
                fpga_identity_compatible ? 1 : 0, p2_admission_compatible ? 1 : 0,
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
    if (g_p2_init_requested && !pipeline_disable_env && !g_pingpong_scheduler_enabled) {
        fpga_fatal(
            "P2 default ping-pong scheduler requires compatible bank/descriptor/identity/stream admission; "
            "caps=0x%08x spu_caps=0x%08x protocol=0x%08x p2_abi=0x%08x stream_status=0x%08x "
            "vpu_two_row_transport_cap=%d spu_q8_scale_pair_stream_cap=%d two_row_transport_ok=%d",
            caps, spu_caps, g_stream_protocol_version, g_p2_stream_abi_signature, g_spu_stream_status,
            vpu_p2_two_row_transport_cap ? 1 : 0, spu_q8_scale_pair_stream_cap ? 1 : 0,
            p2_two_row_transport_compatible ? 1 : 0);
    }
    LOGINIT(
        "P2_CONFIG enabled=%d contract_limit=%d pipeline_default_on=%d pipeline_disable=%d scheduler=%d "
        "route=%s identity "
        "protocol=0x%08x bitstream_id=0x%08x p2_abi=0x%08x stream_status=0x%08x spu_words=%u descriptor_cap=%d "
        "pingpong_cap=%d vpu_two_row_transport_cap=%d spu_q8_scale_pair_stream_cap=%d two_row_transport_ok=%d "
        "windows spu_param=0x%08x spu_out=0x%08x pack_dma_pipeline_requested=%d pack_dma_pipeline_enabled=%d",
        g_spu_q8_scale_stream_supported ? 1 : 0, g_pl_scale_contract_check_limit,
        pipeline_disable_env ? 0 : 1, pipeline_disable_env ? 1 : 0, g_pingpong_scheduler_enabled ? 1 : 0,
        g_pl_scale_contract_check_limit > 0 ? "p2_contract_cpu_shadow" :
                                              (g_pingpong_scheduler_enabled ? "pingpong_production" :
                                               (g_spu_q8_scale_stream_supported ? "unavailable" : "disabled")),
        g_stream_protocol_version, g_bitstream_id, g_p2_stream_abi_signature, g_spu_stream_status, g_spu_word_capacity,
        g_vpu_descriptor_supported ? 1 : 0, g_vpu_pingpong_supported ? 1 : 0,
        vpu_p2_two_row_transport_cap ? 1 : 0, spu_q8_scale_pair_stream_cap ? 1 : 0,
         p2_two_row_transport_compatible ? 1 : 0, SPU_PARAM_BASE, SPU_OUT_BASE,
         g_p2_pack_dma_pipeline_requested ? 1 : 0, g_p2_pack_dma_pipeline_enabled ? 1 : 0);
    // Establish and read back P2 mode 0 before any self-test or model transfer.
    fpga_stream_mode_report_t baseline_stream_report;
    if (!fpga_enforce_p2_stream_mode("P2 route baseline", &baseline_stream_report)) {
        fpga_fatal("stream mode-0 baseline was not quiescent/readable; refusing P2 data-plane traffic");
    }
    const char * mapping_policy = "uio_required";

    LOGINIT(
        "ready version=%s path=%s rows=%d host_row_limit=%d col_beats=%d cols=%d packed_q8=%d max_group_blocks=%d "
        "result_words=%d zdma_max_transfer_bytes=%zu pingpong_cap=%d descriptor_cap=%d scheduler=%d "
        "scheduler_policy=required_for_p2_production pl_scale=%d "
        "pl_scale_policy=required stream_protocol=0x%08x bitstream_id=0x%08x "
        "spu_silu=%d spu_rms=%d spu_rope=%d spu_softmax=%d activation_cache=%d "
        "input_integrity_check=%d vocab_cpu_bypass=%d vocab_min_n=%lld "
        "contract_cpu_shadow_dst=%d "
        "clock_mhz=%.3f profile_log=1 dma_detail=%d "
        "ip_detail=%d detail_every=%d flush_every=%d",
        FPGA_HOST_TRACE_VERSION, path ? path : "dma(default)", g_vpu_max_rows, g_runtime_max_rows, g_vpu_max_beats,
        g_vpu_max_cols, g_packed_q8_supported, g_packed_q8_max_blocks, g_packed_q8_result_words,
        g_zdma_max_transfer_bytes, g_vpu_pingpong_supported ? 1 : 0, g_vpu_descriptor_supported ? 1 : 0,
        g_pingpong_scheduler_enabled ? 1 : 0,
        g_spu_q8_scale_stream_supported ? 1 : 0, g_stream_protocol_version,
        g_bitstream_id, g_spu_silu_supported ? 1 : 0, g_spu_rmsnorm_supported ? 1 : 0, g_spu_rope_supported ? 1 : 0,
        g_spu_softmax_supported ? 1 : 0, g_activation_cache_enabled ? 1 : 0,
        g_activation_input_integrity_check ? 1 : 0, g_vocab_projection_cpu_bypass ? 1 : 0,
        (long long) g_vocab_projection_min_n, g_contract_cpu_shadow_dst ? 1 : 0,
        g_fpga_clock_mhz, g_dma_timing_enabled ? 1 : 0, g_ip_timing_enabled ? 1 : 0,
        g_detail_every, g_log_flush_every);
    LOGINIT(
        "ZDMA trace policy enabled=%d depth=%zu trigger=qualification_mismatch_or_transfer_failure "
        "p2_contract_check=%d explicit_env=FPGA_DMA_AUDIT",
        g_dma_trace_enabled ? 1 : 0, FPGA_DMA_TRACE_DEPTH, g_pl_scale_contract_check_limit);
    if (g_activation_input_integrity_check) {
        LOGINIT("Input integrity check: compare F32 activations before and after FPGA execution");
    }
    LOGINIT(
        "DMA byte_counter_clear=before_transfer current=0x%08x error_mask=0x%08x overflow=fatal",
        g_dma ? g_dma->ZDMA_CH_TOTAL_BYTE : 0U, ZDMA_ISR_ERROR_MASK);
    LOGINIT(
        "DMA max_transfer_bytes=%zu larger_transfers=chunked",
        g_zdma_max_transfer_bytes);
    fpga_log_section("Thông tin chung");
    LOGI(
        "MANIFEST\n"
        "Thông tin bản build\n"
        "  Phiên bản host = %s | Bản build host = \"%s %s\"\n"
        "Phần cứng\n"
        "  Phiên bản giao thức stream = %u | Mã bitstream = 0x%08x | ABI P2 = 0x%08x | Dải địa chỉ vật lý DDR = [0x%llx,0x%llx)\n"
        "  Kích thước ánh xạ DDR = 0x%zx | Nguồn ánh xạ DMA = %s | Nguồn ánh xạ VPU = %s | Nguồn ánh xạ DDR = %s\n"
        "Cấu hình thực thi\n"
        "  Số byte mỗi descriptor ZDMA = %zu | Số hàng = %d | Số worker = %d | Chế độ ping pong = %d | Chuyển vocab sang CPU = %d",
        FPGA_HOST_TRACE_VERSION, __DATE__, __TIME__, g_stream_protocol_version, g_bitstream_id,
        g_p2_stream_abi_signature, (unsigned long long) DDR_BASE_PHYS,
        (unsigned long long) DDR_END_EXCLUSIVE, g_ddr_map_size, g_dma_map_source.c_str(), g_vpu_map_source.c_str(),
        g_ddr_map_source.c_str(), g_zdma_max_transfer_bytes, g_runtime_max_rows, g_p2_pack_workers_requested,
        g_pingpong_scheduler_enabled ? 1 : 0,
        g_vocab_projection_cpu_bypass ? 1 : 0);
    if (baseline_stream_report.recorded) {
        LOGI("P2 STREAM MODE PASS\n  Ngữ cảnh = Nhánh P2 cơ sở | Chế độ = 0 | Trạng thái stream = 0x%08x",
             baseline_stream_report.status);
    }
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
    LOGINIT("DDR synchronization source=%s map_kind=%s ordering=dsb_readback_before_device_dsb_before_cpu",
            g_ddr_map_source.c_str(), fpga_mapping_kind_name(g_ddr_mapping_kind));
    LOGINIT("FPGA_ABORT_ON_CPU_FALLBACK=%d",
            g_abort_on_cpu_fallback ? 1 : 0);

    if (!g_packed_q8_supported) {
        fpga_fatal("REG_CAPS=0x%08x does not expose packed_q8 capability; refusing CPU fallback", caps);
    }
    if (!fpga_weight_layout_host_self_test()) {
        fpga_fatal("P2 pair-interleaved padded WEIGHT host self-test failed; refusing all launches");
    }
    fpga_dma_profile_log_config();
    if (g_pl_scale_contract_check_limit > 0 && !fpga_p2_cumulative_tile_limit_host_self_test()) {
        fpga_fatal("P2 cumulative tile-limit host self-test failed; refusing qualification traffic");
    }
    // The first bounded P2 tile owns all VPU-to-SPU data-plane traffic.
    fpga_p2_init_breadcrumb("phase=data_plane_selftests skipped reason=p2_passive_init");

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
    if (g_p2_pack_dma_pipeline_requested) {
        // The pipeline submits its suffix to this already-created helper during
        // admission. Publish the route only after the helper exists, so a
        // worker policy mismatch cannot reach the first eligible tile.
        if (!g_p2_pack_worker_created || g_p2_pack_workers_active != 2) {
            pthread_mutex_unlock(&g_mutex);
            fpga_cleanup();
            fpga_fatal(
                "FPGA_P2_PACK_DMA_PIPELINE requested but the persistent WEIGHT-pack helper was not admitted; "
                "no pipeline tile was started");
        }
        g_p2_pack_dma_pipeline_enabled = true;
        LOGINIT(
            "P2_PACK_DMA_PIPELINE enabled=1 mode=decode_direct_large_tiles slots=2 weight_slot_bytes=%u "
            "scale_slot_bytes=%u callback_batch_pairs=%zu residency_builds=deferred helper=ready",
            P2_PIPELINE_WEIGHT_SLOT_BYTES, P2_PIPELINE_SCALE_SLOT_BYTES, P2_PIPELINE_PACK_BATCH_PAIRS);
    }
    LOGINIT(
        "P2_PACK_CONFIG requested_workers=%d active_workers=%d parallel_min_bytes=%zu policy=pair_disjoint_before_dma "
        "helper_owns=no_dma_no_vpu_no_spu_no_descriptor",
        g_p2_pack_workers_requested, g_p2_pack_workers_active, FPGA_P2_PACK_PARALLEL_MIN_BYTES);

    g_fpga_init_complete = true;
    fpga_p2_init_breadcrumb("phase=ready route=%s init_data_plane=none pack_workers_requested=%d active=%d",
                            g_pl_scale_contract_check_limit > 0 ? "p2_tile_qualification" :
                            (g_pingpong_scheduler_enabled ? "p2_pingpong" : "unavailable"),
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
    fpga_log_line(true, "P2_ACT_BANK_REUSE", true,
                  "enabled=%d reused_jobs=%llu avoided_bytes=%llu",
                  g_p2_act_bank_reuse_enabled ? 1 : 0,
                  (unsigned long long) g_p2_act_bank_reused_jobs,
                  (unsigned long long) g_p2_act_bank_avoided_bytes);
    fpga_compute_watch_shutdown();
    LOGPROOF(
        "Bắt đầu dọn dẹp tài nguyên\n"
        "  Thời điểm trong vòng đời = trước khi giải phóng backend | DDR đã ánh xạ = %d | VPU đã ánh xạ = %d | DMA đã ánh xạ = %d",
        ddr_is_mapped() ? 1 : 0, vpu_is_mapped() ? 1 : 0, dma_is_mapped() ? 1 : 0);
    if (g_p1_sched_summary_enabled) {
        fpga_p1_sched_summary_emit("cleanup");
    }
    if (g_token_timing_collection_enabled && g_token_timing.active) {
        fpga_token_timing_emit_final(monotonic_now_us());
    }
    fpga_log_section("Dọn dẹp tài nguyên và số liệu tổng kết cuối cùng");
    fpga_dma_profile_finish();

    // No local source establishes a safe software abort/reset sequence for an
    // in-flight ZDMA descriptor.  Therefore cleanup must observe the
    // documented hardware-cleared CTRL2.EN state before invalidating DDR
    // metadata or unmapping any participant in the transfer.
    if (dma_is_mapped() && !zdma_wait_channel_disabled("cleanup", "before_unmap")) {
        pthread_mutex_unlock(&g_mutex);
        fpga_fatal(
            "ZDMA EN remained set during cleanup; descriptors, DDR/VPU/DMA mappings were "
            "intentionally left owned by the process. No undocumented reset/abort write was issued");
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

    g_committed_stream_mode = -1;
    g_fpga_init_complete = false;

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
    const char * fpga_block_gemv_mode = q8_candidates == 0 ? "not_applicable" :
                                      q8_coverage_complete ? "active" : "incomplete";
    const uint64_t p2_residency_budget_bytes = g_p2_weight_residency_budget_mb > 0 ?
                                                  (uint64_t) g_p2_weight_residency_budget_mb * 1024ULL * 1024ULL :
                                                  0ULL;
    const uint64_t p2_residency_allocated_bytes =
        g_p2_residency_next_off >= WEIGHT_CACHE_BASE ? (uint64_t) g_p2_residency_next_off - WEIGHT_CACHE_BASE : 0ULL;
    const uint64_t p2_residency_remaining_bytes = p2_residency_allocated_bytes <= p2_residency_budget_bytes ?
                                                     p2_residency_budget_bytes - p2_residency_allocated_bytes :
                                                     0ULL;
    const double p2_direct_weight_pack_gb_s =
        g_p2_residency_direct_weight_pack_us > 0 ?
            (double) g_p2_residency_direct_weight_pack_bytes /
                ((double) g_p2_residency_direct_weight_pack_us * 1000.0) :
            0.0;
    LOGPROOF(
        "FPGA GEMV COVERAGE\n"
        "  Số lần gọi hook = %lld | Số ứng viên Q8 = %lld | Số phép Q8 dự kiến chạy FPGA = %lld\n"
        "  Số phép Q8 phần cứng đã hoàn tất = %lld | Số phép Q8 chủ động chuyển sang CPU = %lld\n"
        "  Số phép Q8 fallback về CPU do FPGA không khả dụng = %lld | Kết luận định tuyến = %s | Trạng thái block GEMV FPGA = %s",
        hook_calls, q8_candidates, q8_expected_fpga, g_fpga_count, q8_intentional_cpu, q8_unavailable_cpu,
        q8_coverage_complete ? "đầy đủ" : "chưa đầy đủ",
        strcmp(fpga_block_gemv_mode, "active") == 0 ? "đang hoạt động" :
        strcmp(fpga_block_gemv_mode, "not_applicable") == 0 ? "không áp dụng" : "chưa đầy đủ");
    LOGPROOF(
        "P2 PACK SUMMARY\n"
        "  Số worker yêu cầu = %d | Số worker hoạt động = %d | Helper đã dừng = %d | Ngưỡng byte tối thiểu để xử lý song song = %zu\n"
        "  Số công việc song song = %lld | Số byte xử lý song song = %llu | Số lần bỏ qua xử lý song song do ngưỡng = %lld\n"
        "  Thời gian đóng gói trên luồng chính(us) = %lld | Thời gian phục vụ của helper(us) = %lld\n"
        "  Thời gian chờ của luồng gọi(us) = %lld | Số công việc hợp nhất = %lld | Thời gian xử lý hợp nhất(us) = %lld\n"
        "  Số mục scale hợp nhất = %lld | Đóng gói trực tiếp hợp nhất có bao gồm scale = 1 | Đã yêu cầu DMA pipeline = %d\n"
        "  Đã bật DMA pipeline = %d | Thời gian DMA phía host trong pipeline có bao gồm callback = 1\n"
        "  Số công việc đủ điều kiện DMA pipeline = %lld | Số công việc đã xử lý trong DMA pipeline = %lld\n"
        "  Thời gian callback trong DMA pipeline(us) = %lld | Số byte callback trong DMA pipeline = %llu\n"
        "  Thời gian helper trong DMA pipeline(us) = %lld | Thời gian xử lý phần còn lại trong DMA pipeline(us) = %lld\n"
        "  Số byte còn lại trong DMA pipeline = %lld | Số công việc DMA pipeline bị từ chối = %lld\n"
        "  Thời gian chờ helper trong DMA pipeline(us) = %lld | Thời gian helper công bố trước trong DMA pipeline(us) = %lld\n"
        "  Số lần helper hoàn tất sớm trong DMA pipeline = %lld",
        g_p2_pack_workers_requested, g_p2_pack_workers_active, g_p2_pack_worker_created ? 0 : 1,
        FPGA_P2_PACK_PARALLEL_MIN_BYTES, g_p2_pack_parallel_jobs, (unsigned long long) g_p2_pack_parallel_bytes,
        g_p2_pack_serial_threshold_skips, g_p2_pack_main_us, g_p2_pack_helper_service_us, g_p2_pack_caller_wait_us,
         g_p2_pack_fused_jobs, g_p2_pack_fused_us, g_p2_pack_fused_scale_entries,
         g_p2_pack_dma_pipeline_requested ? 1 : 0, g_p2_pack_dma_pipeline_enabled ? 1 : 0,
         g_p2_pack_dma_pipeline_eligible_jobs, g_p2_pack_dma_pipeline_consumed_jobs,
         g_p2_pack_dma_pipeline_callback_us, (unsigned long long) g_p2_pack_dma_pipeline_callback_bytes,
         g_p2_pack_dma_pipeline_helper_us, g_p2_pack_dma_pipeline_residual_us,
         g_p2_pack_dma_pipeline_residual_bytes, g_p2_pack_dma_pipeline_rejected_jobs,
         g_p2_pack_dma_pipeline_helper_wait_us,
         g_p2_pack_dma_pipeline_helper_publication_lead_us, g_p2_pack_dma_pipeline_helper_early_completions);
    fpga_log_pack_breakdown("decode_total", -1, g_fpga_perf_decode.decode_tokens,
                            fpga_pack_detail_record(g_pack_detail_decode));
    fpga_log_pack_breakdown("process_total_including_startup", -1, -1,
                            fpga_pack_detail_record(g_pack_detail));
    if (fpga_p2_residency_has_reportable_activity()) {
        fpga_log_line(true, "P2 RESIDENCY SUMMARY", true,
            "scope=decode_total builds=%lld build_us_including_failed_attempts=%lld",
            g_p2_residency_decode_builds, g_p2_residency_decode_build_us);
        fpga_log_line(true, "P2 RESIDENCY SUMMARY", true,
            "Cấu hình\n"
        "  Bắt buộc bật = 1 | Được bật = %d | Chế độ chẩn đoán = %d | Theo dõi chi tiết = %d | Kiểm tra metadata = %d\n"
        "  Số slot = %zu/%zu | Số bucket chỉ mục = %zu | Số lần dò tối đa = %zu\n"
        "Tra cứu\n"
        "  Số lần dò = %lld | Số lần hết lượt dò và chuyển sang staging trực tiếp = %lld | Số lần trúng = %lld | Số lần trượt = %lld\n"
        "  Số lần trúng metadata trên host = %lld | Số lần vô hiệu hóa metadata trên host = %lld | Số lần đọc DDR volatile = %lld\n"
        "Thời gian\n"
        "  Thời gian xây dựng(us) = %lld | Thời gian lựa chọn(us) = %lld | Thời gian kiểm tra metadata(us) = %lld\n"
        "  Thời gian xử lý tham số thường trú(us) = %lld | Thời gian đóng gói weight trực tiếp(us) = %lld\n"
        "  Số byte đóng gói weight trực tiếp = %llu | Tốc độ đóng gói weight trực tiếp suy ra(GB/s) = %.3f\n"
        "  Đóng gói trực tiếp có bao gồm scale hợp nhất = 1\n"
        "Dữ liệu tránh xử lý\n"
        "  Số byte tránh phải đóng gói trên CPU = %lld | Số byte tránh phải truyền từ DDR đến IP = %lld\n"
        "Nguyên nhân trượt\n"
        "  Số lần trượt do căn chỉnh = %lld | Số lần trượt do kích thước = %lld | Số lần trượt do xung đột = %lld\n"
        "  Số lần trượt do đánh dấu không hợp lệ = %lld | Số lần trượt do dữ liệu cũ = %lld | Số lần trượt do không khớp = %lld\n"
        "  Số lần trượt do dung lượng = %lld | Số lần trượt do chưa ở trạng thái yên = %lld | Số lần trượt do phạm vi = %lld\n"
        "  Số lần trượt do kiểm tra = %lld\n"
        "Dung lượng và xây dựng\n"
        "  Số lần xây dựng = %lld | Số lần xây dựng thất bại = %lld | Số byte logic = %lld | Số byte đã cấp phát = %llu\n"
        "  Số byte còn lại = %llu | Ngân sách bộ nhớ(byte) = %llu",
            g_p2_weight_residency_enabled ? 1 : 0, g_p2_weight_residency_diagnostic ? 1 : 0,
            g_p2_residency_trace_enabled ? 1 : 0, g_p2_residency_verify_metadata ? 1 : 0, g_p2_resident_tile_count,
            g_p2_resident_tiles.size(), g_p2_residency_index.size(), P2_WEIGHT_RESIDENCY_INDEX_MAX_PROBES,
            g_p2_residency_probe_count, g_p2_residency_probe_exhausted, g_p2_residency_hits, g_p2_residency_misses,
            g_p2_residency_host_metadata_hits, g_p2_residency_host_metadata_invalidations,
            g_p2_residency_volatile_ddr_reads, g_p2_residency_build_us, g_p2_residency_select_us,
            g_p2_residency_metadata_validate_us, g_p2_residency_resident_param_us,
            g_p2_residency_direct_weight_pack_us, (unsigned long long) g_p2_residency_direct_weight_pack_bytes,
            p2_direct_weight_pack_gb_s, g_p2_residency_avoided_cpu_pack_bytes,
            g_p2_residency_avoided_ddr_to_ip_bytes,
            g_p2_residency_miss_alignment, g_p2_residency_miss_shape, g_p2_residency_miss_collision,
            g_p2_residency_miss_poison, g_p2_residency_miss_stale, g_p2_residency_miss_mismatch,
            g_p2_residency_miss_capacity, g_p2_residency_miss_quiescence, g_p2_residency_miss_range,
            g_p2_residency_miss_verify,
            g_p2_residency_builds, g_p2_residency_build_failures, g_p2_residency_logical_bytes,
            (unsigned long long) p2_residency_allocated_bytes, (unsigned long long) p2_residency_remaining_bytes,
            (unsigned long long) p2_residency_budget_bytes);
    }
    fpga_log_line(true, "CLEANUP", true,
        "Hoàn tất\n"
        "Thực thi\n"
        "  Số lần gọi FPGA = %lld | Số lần chạy VPU = %lld | Số lần từ chối = %lld | Số lần chuyển attention sang CPU = %lld\n"
        "  Số lần chuyển vocab projection sang CPU = %lld\n"
        "  Thời gian đã trôi qua(s) = %.3f | Khả năng hỗ trợ ping pong = %d | Khả năng hỗ trợ descriptor = %d | Bộ lập lịch = %d\n"
        "Cache\n"
        "  Đã bật activation cache = %d | Số lần trúng trong activation cache = %lld | Số lần trượt = %lld\n"
        "  Số lần staging weight trực tiếp = %lld\n"
        "  Thời gian đóng gói weight(ms) = %.3f\n"
        "Kiểm tra dữ liệu\n"
        "  Số lần tràn scale FP16 của activation = %lld | Số lần kiểm tra tính toàn vẹn đầu vào = %lld\n"
        "  Số lần kiểm tra tính toàn vẹn đầu vào thất bại = %lld\n"
        "  Số lần chuyển sang CPU do giới hạn contract = %lld\n"
        "  Số giá trị không khớp = %lld | Số giá trị đầu ra do CPU shadow ghi trong chế độ contract = %lld\n"
        "Kiểm tra P2\n"
        "  Số lần kiểm tra contract ma trận P2 = %lld | Số lần kiểm tra Q16 của tile P2 = %lld | Giới hạn tile P2 = %d\n"
        "  Ranh giới tile P2 = %d | Contract giá trị ma trận = %s | Số công việc contract P2 = %lld | Số bank contract P2 = %lld\n"
        "  Số lần mất dữ liệu stream P2 = %lld | Số lỗi stream P2 = %lld\n"
        "Lưu trú P2\n"
        "  Đã bật lưu trú P2 = %d | Số slot lưu trú P2 = %zu/%zu | Số lần xây dựng dữ liệu lưu trú P2 = %lld\n"
        "  Số lần trúng dữ liệu lưu trú P2 = %lld | Số lần trượt dữ liệu lưu trú P2 và staging trực tiếp = %lld\n"
        "  Số lần xây dựng dữ liệu lưu trú P2 thất bại = %lld | Số byte logic lưu trú P2 = %lld\n"
        "  Số byte đã cấp phát lưu trú P2 = %llu | Số byte còn lại lưu trú P2 = %llu | Số byte đóng gói weight trực tiếp P2 = %llu\n"
        "  Thời gian đóng gói weight trực tiếp P2(ms) = %.3f | Tốc độ đóng gói weight trực tiếp P2 suy ra(GB/s) = %.3f",
        g_fpga_count, g_fpga_vpu_runs, g_reject_count, g_attention_bypass_count, g_vocab_projection_bypass_count,
        elapsed_us > 0 ? (double) elapsed_us / 1000000.0 : 0.0,
        g_vpu_pingpong_supported ? 1 : 0, g_vpu_descriptor_supported ? 1 : 0, g_pingpong_scheduler_enabled ? 1 : 0,
        g_activation_cache_enabled ? 1 : 0, g_activation_cache_hits, g_activation_cache_misses, g_weight_cache_misses, (double) g_weight_pack_us / 1000.0, g_activation_scale_fp16_overflows,
        g_activation_input_integrity_checks, g_activation_input_integrity_failures,
        g_contract_limit_cpu_bypass_count,
        g_contract_value_mismatches, g_contract_cpu_shadow_dst_values,
        g_p2_matrix_contract_checks, g_p2_tile_q16_checks,
        g_p2_tile_limit, g_p2_tile_contract_boundary_reached ? 1 : 0,
        g_pl_scale_contract_check_limit > 0 ? "chưa thực hiện" : "không áp dụng", g_pl_scale_jobs, g_pl_scale_banks,
        g_pl_scale_stream_drops, g_pl_scale_stream_errors, g_p2_weight_residency_enabled ? 1 : 0,
        g_p2_resident_tile_count, g_p2_resident_tiles.size(), g_p2_residency_builds, g_p2_residency_hits,
        g_p2_residency_misses, g_p2_residency_build_failures, g_p2_residency_logical_bytes,
        (unsigned long long) p2_residency_allocated_bytes, (unsigned long long) p2_residency_remaining_bytes,
        (unsigned long long) g_p2_residency_direct_weight_pack_bytes,
        (double) g_p2_residency_direct_weight_pack_us / 1000.0, p2_direct_weight_pack_gb_s);
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
    data->ip_compute_timing_valid = g_compute_watch_enabled && g_spu_q8_scale_stream_supported;
    if (!g_compute_watch_enabled) {
        data->ip_compute_us = -1;
    }
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
           (data->h2ip_dma_us > 0 || data->ip2host_dma_us > 0 ||
            (data->ip_compute_timing_valid && data->ip_compute_us > 0)) ? 1 : 0;
}

extern "C" void fpga_advance_sequence_position(int n_tokens) {
    if (n_tokens < 0 || g_current_seq_pos > INT_MAX - n_tokens) {
        fpga_fatal("invalid FPGA sequence advance current=%d delta=%d", g_current_seq_pos, n_tokens);
    }

    const int previous_graph_seq = g_current_seq_pos;
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
    if (is_attention) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_attention_bypass_count++;
            if (g_attention_bypass_count == 1) {
                LOGI("Attention do CPU xử lý; thời gian FPGA chỉ bao gồm các phép tính ma trận Q8_0");
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
    if (ith == 0) {
        g_matmul_hook_calls.fetch_add(1, std::memory_order_relaxed);
        if (q8_f32_f32_candidate) {
            g_q8_candidate_calls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // A qualification limit must stop all board launches once it is reached.
    // After P2 qualification completes, later eligible GEMVs remain native
    // CPU work and are not an FPGA-unavailability fallback.
    // After the P2 tile boundary, continue through the upstream CPU kernel.
    const bool p2_contract_boundary =
        g_pl_scale_contract_check_limit > 0 &&
        (g_p2_tile_contract_boundary_reached ||
         fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit));
    if (q8_f32_f32_candidate && g_contract_cpu_shadow_dst && p2_contract_boundary) {
        if (ith == 0) {
            pthread_mutex_lock(&g_mutex);
            g_contract_limit_cpu_bypass_count++;
            if (!g_contract_limit_cpu_bypass_logged) {
                LOGI(
                    "qualification boundary route=cpu_native mode=%s checked=%lld limit=%d p2_tile_q16_checks=%lld "
                    "p2_matrix_contract_checks=%lld matrix_value_contract=%s hw_completed=%lld "
                    "input_integrity_failures=%lld contract_limit_cpu_bypass=%lld reason=qualification_limit",
                    "p2_spu_scale",
                    g_p2_tile_q16_checks,
                    g_p2_tile_limit, g_p2_tile_q16_checks,
                    g_p2_matrix_contract_checks, "not_attempted", g_fpga_count,
                    g_activation_input_integrity_failures, g_contract_limit_cpu_bypass_count);
                g_contract_limit_cpu_bypass_logged = true;
                // Qualification runs may stop at the first anomaly.  Make
                // the one-time transition evidence durable immediately rather
                // than relying on normal cleanup after the CPU continuation.
                fflush(fpga_log_fp());
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
                LOGI(
                    "Vocab projection do CPU xử lý | tensor = %s | Kích thước = K%lld_N%lld_M%lld | Ngưỡng N = %lld",
                    tensor_name, (long long) probe_k, (long long) probe_n, (long long) probe_m,
                    (long long) g_vocab_projection_min_n);
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
    g_active_matmul_pingpong   = g_pingpong_scheduler_enabled &&
                                g_pl_scale_contract_check_limit == 0;
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
                "P2 source validation failed before any model ZDMA/VPU launch for tensor=%s layer=%d; the active GGUF "
                "contains a non-finite Q8_0 scale. Replace or re-copy the model, then rerun P2 qualification. No CPU fallback or "
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
    g_weight_cache_misses += totals.weight_cache_misses;
    if (g_p1_sched_summary_enabled) {
        g_p1_sched_summary.matmuls++;
        g_p1_sched_summary.vpu_runs += totals.vpu_runs;
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
    const double dma_in_ms       = dma_act_ms + dma_weight_ms + dma_scale_ms + (double) totals.scratch_dma_us / 1000.0;
    const double ip_ms           = fpga_compute_report_us(totals.ip_compute_us) / 1000.0;
    const double dma_out_ms      = (double) totals.dma_result_us / 1000.0;
    const double host_result_ms  = (double) totals.host_result_us / 1000.0;
    const double host_accum_ms   = (double) totals.host_accum_us / 1000.0;
    const double weight_pack_ms  = (double) totals.weight_pack_us / 1000.0;
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
    const double gb_s   = total_ms > 0.0 ? (double) effective_bytes / (total_ms * 1000000.0) : 0.0;
    if (!g_compute_watch_enabled) {
        dominant = "unknown_compute_unmeasured";
    }
    const double cycles_per_run = !g_compute_watch_enabled ? std::numeric_limits<double>::quiet_NaN() :
                                  (g_fpga_clock_mhz > 0.0 && totals.vpu_runs > 0) ?
                                      ((double) totals.ip_compute_us * g_fpga_clock_mhz / (double) totals.vpu_runs) :
                                      0.0;

    if (g_profile_every > 0 && (g_fpga_count == 1 || (g_fpga_count % g_profile_every) == 0)) {
        LOGSTAGE(
            "tensor=%s layer=%d seq=%d phase=%s shape=K%lld_N%lld_M%lld row_tiles=%lld group_tiles=%lld q8_blocks=%lld "
            "vpu_runs=%lld prep_ms=%.3f weight_pack_ms=%.3f "
            "activation_scale_fp16_overflows=%lld dma_input_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f "
            "scale_dma_ms=%.3f ip_compute_ms=%.3f dma_output_ms=%.3f host_result_ms=%.3f host_accum_ms=%.3f "
            "total_ms=%.3f dominant=%s pl_scale=%d effective_GMAC/s=%.3f effective_GB/s=%.3f "
            "act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu "
            "weight_cache_misses=%lld cycles_per_run=%.1f",
            tensor_name, effective_layer_id, seq_pos, decode_or_prefill(m), (long long) k, (long long) n, (long long) m,
            row_tiles, (q8_blocks + max_group_blocks - 1) / max_group_blocks, (long long) q8_blocks, totals.vpu_runs,
            prep_ms, weight_pack_ms, totals.activation_scale_fp16_overflows, dma_in_ms,
            dma_act_ms, dma_weight_ms, dma_scale_ms, ip_ms, dma_out_ms, host_result_ms, host_accum_ms,
            total_ms, dominant, g_spu_q8_scale_stream_supported ? 1 : 0, gmac_s, gb_s,
            totals.activation_bytes, totals.weight_bytes, totals.scale_bytes, totals.result_bytes,
            totals.weight_cache_misses, cycles_per_run);
    }
    LOGIP("summary vpu_runs=%lld ip_ms=%.3f cycles_per_run=%.1f clock_mhz=%.3f", totals.vpu_runs, ip_ms, cycles_per_run,
          g_fpga_clock_mhz);

    if (g_status_stderr && (g_fpga_count == 1 || (g_profile_every > 0 && (g_fpga_count % g_profile_every) == 0))) {
        fprintf(stderr,
                "[STAGE] tensor=%s layer=%d K=%lld N=%lld M=%lld total_ms=%.3f dma_in_ms=%.3f ip_ms=%.3f "
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
    pthread_mutex_lock(&g_mutex);
    if (g_token_timing_collection_enabled && g_token_timing.active) {
        fpga_token_timing_emit(g_token_timing.graph_seq, 0, "kv_cache_reset_incomplete", monotonic_now_us());
    }
    g_current_seq_pos                = 0;
    g_scratch.activation_cache_valid = false;
    pthread_mutex_unlock(&g_mutex);
}

// Private implementation body for fpga_host.cpp. This file is deliberately
// included by the coordinator instead of being compiled as a translation unit.

static void fpga_p2_init_breadcrumb(const char * fmt, ...) {
    if (!g_p2_init_requested ||
        (!g_init_verbose && (!fmt || strstr(fmt, "phase=failure") == nullptr))) {
        return;
    }

    fprintf(stderr, "[P2_INIT] version=%s ", FPGA_HOST_TRACE_VERSION);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
}

static void fpga_p2_boundary_marker(const char * fmt, ...) {
    if (!g_p2_boundary_diagnostics_enabled) {
        return;
    }

    FILE * fp = fpga_log_fp();
    fprintf(fp, "[INFO] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fflush(fp);

    fprintf(stderr, "[P2_BOUNDARY] ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
}

static void fpga_p2_dma_breadcrumb(const char * fmt, ...) {
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[INFO] P2_ACT_DMA_TRACE ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    if (g_p2_trace_job_id != 0U) {
        fprintf(fp, " ctx_job=%u ctx_tile=%u ctx_bank=%d ctx_tag=%s ctx_dma_seq=%llu", g_p2_trace_job_id,
                g_p2_trace_tile_id, g_p2_trace_bank, g_p2_trace_dma_tag.empty() ? "none" : g_p2_trace_dma_tag.c_str(),
                g_p2_dma_transfer_sequence);
    }
    fprintf(fp, "\n");
    fflush(fp);

    if (g_p2_terminal_trace_enabled) {
        fprintf(stderr, "[P2_ACT_DMA] ");
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        if (g_p2_trace_job_id != 0U) {
            fprintf(stderr, " ctx_job=%u ctx_tile=%u ctx_bank=%d ctx_tag=%s ctx_dma_seq=%llu", g_p2_trace_job_id,
                    g_p2_trace_tile_id, g_p2_trace_bank,
                    g_p2_trace_dma_tag.empty() ? "none" : g_p2_trace_dma_tag.c_str(), g_p2_dma_transfer_sequence);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }
}

static void fpga_log_line(bool enabled, const char * tag, bool force_flush, const char * fmt, ...) {
    if (!enabled) {
        return;
    }
    if (tag && strcmp(tag, "ERROR") == 0) {
        g_summary_detail_after_error = true;
    }
    va_list ap;
    va_start(ap, fmt);
    fpga_log_vline(tag, force_flush, fmt, ap);
    va_end(ap);
}

static void log_uio_inventory_once(void) {
    if (g_uio_inventory_logged) {
        return;
    }
    g_uio_inventory_logged = true;

    DIR * dir = opendir("/sys/class/uio");
    if (!dir) {
        LOGINIT("UIO inventory unavailable: /sys/class/uio cannot be opened errno=%d (%s)", errno, strerror(errno));
        return;
    }

    bool            any = false;
    struct dirent * ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }

        const std::string uio = ent->d_name;
        std::string       name;
        std::string       addr;
        std::string       size;
        read_text_file("/sys/class/uio/" + uio + "/name", &name);
        read_text_file("/sys/class/uio/" + uio + "/maps/map0/addr", &addr);
        read_text_file("/sys/class/uio/" + uio + "/maps/map0/size", &size);
        LOGINIT("UIO inventory dev=/dev/%s name=%s addr=%s size=%s", uio.c_str(), name.empty() ? "?" : name.c_str(),
                addr.empty() ? "?" : addr.c_str(), size.empty() ? "?" : size.c_str());
        any = true;
    }
    closedir(dir);

    if (!any) {
        LOGINIT("UIO inventory: /sys/class/uio exists but contains no uio devices");
    }
}

static void fpga_p2_ddr_sync_breadcrumb(const char * tag,
                                        const char * direction,
                                        uint32_t     off,
                                        size_t       bytes,
                                        const char * ordering) {
    // DSB/readback remains mandatory for every P2 handoff.  Routine success
    // records are bounded so primary latency logging cannot be flooded.
    const bool emit_success = g_p2_boundary_diagnostics_enabled || g_p2_first_act_dma_trace_active ||
                              g_pl_scale_contract_check_limit > 0;
    if (emit_success) {
        LOGI("P2_DDR_SYNC tag=%s direction=%s offset=0x%08x bytes=%zu map_kind=%s action=no_msync ordering=%s",
                 tag ? tag : "?", direction, off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind), ordering);
    }
    // Qualification is deliberately bounded.  Mirror its handoffs to stderr
    // so a board lockup cannot hide the final ordering edge.
    if (g_p2_first_act_dma_trace_active || g_pl_scale_contract_check_limit > 0) {
        fpga_p2_dma_breadcrumb(
            "step=ddr_sync tag=%s direction=%s offset=0x%08x bytes=%zu map_kind=%s action=no_msync ordering=%s",
            tag ? tag : "?", direction, off, bytes, fpga_mapping_kind_name(g_ddr_mapping_kind), ordering);
    }
}

static void zdma_format_error_mask(uint32_t isr, char * out, size_t out_size) {
    struct zdma_error_name_t {
        uint32_t     mask;
        const char * name;
    };

    static constexpr zdma_error_name_t kErrors[] = {
        { 0x00000001U, "INV_APB"            },
        { 0x00000008U, "BYTE_CNT_OVRFL"     },
        { 0x00000010U, "SRC_IRQ_ACCT_OVRFL" },
        { 0x00000020U, "DST_IRQ_ACCT_OVRFL" },
        { 0x00000040U, "AXI_RD_SRC_DSCR"    },
        { 0x00000080U, "AXI_RD_DST_DSCR"    },
        { 0x00000100U, "AXI_RD_DATA"        },
        { 0x00000200U, "AXI_WR_DATA"        },
        { 0x00000800U, "DMA_PAUSE"          },
    };

    if (out_size == 0U) {
        return;
    }
    out[0]                = '\0';
    size_t         used   = 0;
    const uint32_t errors = isr & ZDMA_ISR_ERROR_MASK;
    if (errors == 0U) {
        snprintf(out, out_size, "none");
        return;
    }
    for (const zdma_error_name_t & entry : kErrors) {
        if ((errors & entry.mask) == 0U || used >= out_size) {
            continue;
        }
        const int written = snprintf(out + used, out_size - used, "%s%s", used == 0U ? "" : "|", entry.name);
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
    char           errors[160];
    zdma_format_error_mask(isr, errors, sizeof(errors));
    LOGE(
        "ZDMA dump tag=%s status=0x%08x isr=0x%08x errors=%s ctrl0=0x%08x ctrl1=0x%08x ctrl2=0x%08x total_bytes=0x%08x "
        "data_attr=0x%08x cur_src=0x%llx cur_dst=0x%llx src_desc=[0x%08x,0x%08x,0x%08x,0x%08x] "
        "dst_desc=[0x%08x,0x%08x,0x%08x,0x%08x]",
        tag ? tag : "?", g_dma ? g_dma->ZDMA_CH_STATUS : 0xFFFFFFFFU, isr, errors,
        g_dma ? g_dma->ZDMA_CH_CTRL0 : 0xFFFFFFFFU, g_dma ? g_dma->ZDMA_CH_CTRL1 : 0xFFFFFFFFU,
        g_dma ? g_dma->ZDMA_CH_CTRL2 : 0xFFFFFFFFU, g_dma ? g_dma->ZDMA_CH_TOTAL_BYTE : 0xFFFFFFFFU,
        g_dma ? g_dma->ZDMA_CH_DATA_ATTR : 0xFFFFFFFFU,
        g_dma ?
            (unsigned long long) zdma_read_addr(&g_dma->ZDMA_CH_SRC_CUR_PYLD_LSB, &g_dma->ZDMA_CH_SRC_CUR_PYLD_MSB) :
            0ULL,
        g_dma ?
            (unsigned long long) zdma_read_addr(&g_dma->ZDMA_CH_DST_CUR_PYLD_LSB, &g_dma->ZDMA_CH_DST_CUR_PYLD_MSB) :
            0ULL,
        g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD0 : 0xFFFFFFFFU, g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD1 : 0xFFFFFFFFU,
        g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD2 : 0xFFFFFFFFU, g_dma ? g_dma->ZDMA_CH_SRC_DSCR_WORD3 : 0xFFFFFFFFU,
        g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD0 : 0xFFFFFFFFU, g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD1 : 0xFFFFFFFFU,
        g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD2 : 0xFFFFFFFFU, g_dma ? g_dma->ZDMA_CH_DST_DSCR_WORD3 : 0xFFFFFFFFU);
}

static void fpga_log_source_file_provenance(const void * source, size_t bytes) {
    constexpr size_t MAX_PROBE_BYTES = sizeof(block_q8_0_t);
    if (!source || bytes == 0U) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=%p bytes=%zu result=invalid_request", source, bytes);
        return;
    }

    fpga_proc_map_info_t map     = {};
    const uintptr_t      address = (uintptr_t) source;
    if (!fpga_find_process_mapping(address, &map)) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu result=map_not_found errno=%d (%s)",
             (unsigned long long) address, bytes, errno, strerror(errno));
        return;
    }

    const size_t bytes_in_mapping = (size_t) (map.end - address);
    const size_t probe_bytes      = std::min(std::min(bytes, MAX_PROBE_BYTES), bytes_in_mapping);
    if (probe_bytes == 0U) {
        LOGE("Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu map=[0x%llx,0x%llx) result=empty_map_probe",
             (unsigned long long) address, bytes, (unsigned long long) map.start, (unsigned long long) map.end);
        return;
    }
    const bool file_backed = map.path[0] == '/';
    if (!file_backed) {
        LOGE(
            "Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu map=[0x%llx,0x%llx) perms=%s file_offset=0x%llx path=%s "
            "result=not_file_backed",
            (unsigned long long) address, bytes, (unsigned long long) map.start, (unsigned long long) map.end,
            map.perms, (unsigned long long) map.file_offset, map.path[0] ? map.path : "[anonymous]");
        return;
    }

    const int fd = open(map.path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        LOGE(
            "Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu map=[0x%llx,0x%llx) perms=%s file_offset=0x%llx path=%s "
            "result=file_open_failed errno=%d (%s)",
            (unsigned long long) address, bytes, (unsigned long long) map.start, (unsigned long long) map.end,
            map.perms, (unsigned long long) map.file_offset, map.path, errno, strerror(errno));
        return;
    }

    uint8_t        file_bytes[MAX_PROBE_BYTES] = {};
    const uint64_t mapped_file_offset          = map.file_offset + (uint64_t) (address - map.start);
    const ssize_t  read_bytes                  = pread(fd, file_bytes, probe_bytes, (off_t) mapped_file_offset);
    const int      saved_errno                 = errno;
    close(fd);

    const bool complete_read       = read_bytes == (ssize_t) probe_bytes;
    const bool source_matches_file = complete_read && memcmp(file_bytes, source, probe_bytes) == 0;
    LOGE(
        "Q8_SOURCE_MAP_PROVENANCE source=0x%llx bytes=%zu probe_bytes=%zu map=[0x%llx,0x%llx) perms=%s "
        "map_file_offset=0x%llx source_file_offset=0x%llx path=%s pread=%zd source_matches_file=%d "
        "file_bytes=[%02x,%02x,%02x,%02x] source_bytes=[%02x,%02x,%02x,%02x]%s",
        (unsigned long long) address, bytes, probe_bytes, (unsigned long long) map.start, (unsigned long long) map.end,
        map.perms, (unsigned long long) map.file_offset, (unsigned long long) mapped_file_offset, map.path, read_bytes,
        source_matches_file ? 1 : 0, file_bytes[0], file_bytes[1], file_bytes[2], file_bytes[3],
        ((const uint8_t *) source)[0], ((const uint8_t *) source)[1], ((const uint8_t *) source)[2],
        ((const uint8_t *) source)[3], complete_read ? "" : strerror(saved_errno));
}

static void fpga_p2_residency_log(bool force_flush, const char * event, const char * fmt, ...) {
    if (!force_flush && !g_p2_residency_trace_enabled) {
        return;
    }
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[P2_RESIDENCY] event=%s ", event ? event : "?");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fpga_log_finish_line(fp, force_flush);
}

static void p2_trace_set_job_context(const fpga_tile_job_t & job) {
    g_p2_trace_job_id  = job.job_id;
    g_p2_trace_tile_id = job.tile_id;
    g_p2_trace_bank    = job.bank;
}

static bool p2_trace_this_tile() {
    // Success breadcrumbs are diagnostic/qualification evidence only.  A
    // normal P2 GEMV keeps STAGE/TOKEN and error logs, but emits no per-tile
    // trace records into the primary latency log.
    return g_p2_init_requested &&
           (g_p2_boundary_diagnostics_enabled || g_p2_terminal_trace_enabled || g_p2_tile_trace_enabled ||
            g_pl_scale_contract_check_limit > 0);
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
                                     uint32_t                          tile_id,
                                     int                               bank,
                                     fpga_stage_totals_t *             totals,
                                     fpga_prompt_weight_staging_t *    staging,
                                     uint32_t                          weight_stage_off,
                                     uint32_t                          scale_stage_off) {
    const bool reuse_direct_weight = staging && staging->valid &&
        staging->src0 == src0 && staging->data == weight_data_base && staging->ddr == g_ddr &&
        staging->row0 == row0 && staging->rows == rows && staging->k_block0 == k_block0 &&
        staging->group_blocks == group_blocks && staging->epoch == g_p2_weight_residency_epoch &&
        staging->weight_off == weight_stage_off && staging->scale_off == scale_stage_off;
    if (staging) {
        // Every early return leaves reuse invalid. Publish only after the
        // pack helper and the existing final publication fence complete.
        staging->valid = false;
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

    const size_t act_bytes        = (size_t) group_beats * 16U;
    const size_t weight_bytes     = weight_window_bytes_for_rows(rows, group_beats);
    const size_t result_bytes     = (size_t) result_words * 16U;
    const size_t scale_bytes      = g_p3_matmul_active ? fpga_p3_param_bytes(result_values) : (size_t) result_words * 16U;
    const size_t spu_result_bytes = (size_t) rows * 16U;
    if (!range_fits(ACT_BASE, act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(WEIGHT_BASE, weight_bytes, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(RESULT_BASE, result_bytes, RESULT_BASE, RESULT_END) ||
        !range_fits(SPU_OUT_BASE, spu_result_bytes, SPU_OUT_BASE, SPU_OUT_END) ||
        !ddr_range_fits(ACT_BASE, act_bytes) || !ddr_range_fits(WEIGHT_BASE, weight_bytes) ||
        !ddr_range_fits(RESULT_BASE, result_bytes) ||
        !ddr_range_fits(SPU_OUT_BASE, spu_result_bytes)) {
        LOGE(
            "unsupported DMA-to-IP tiling case: window overflow act=%zu weight=%zu result=%zu scale=%zu spu_out=%zu "
            "ddr_size=0x%zx",
            act_bytes, weight_bytes, result_bytes, scale_bytes, spu_result_bytes, g_ddr_map_size);
        return false;
    }

    job = {};

    job.bank             = bank & 1;
    job.job_id           = fpga_next_job_id();
    job.matmul_call_id   = g_active_matmul_call_id;
    job.graph_seq        = g_active_matmul_graph_seq;
    job.layer_id         = g_active_matmul_layer_id;
    job.shape_k          = g_active_matmul_shape_k;
    job.shape_n          = g_active_matmul_shape_n;
    job.shape_m          = g_active_matmul_shape_m;
    job.cpu_shadow_dst   = g_active_matmul_cpu_shadow;
    job.pingpong_scheduler = g_active_matmul_pingpong;
    job.tensor_name      = g_active_matmul_tensor_name;
    job.tile_id          = tile_id;
    job.tensor_id        = fpga_tensor_id_from_ptr(src0);
    job.row0             = row0;
    job.rows             = rows;
    job.k_block0         = k_block0;
    job.group_blocks     = group_blocks;
    job.group_beats      = group_beats;
    job.col              = col;
    job.act_bytes        = act_bytes;
    job.weight_bytes     = weight_bytes;
    job.scale_bytes      = scale_bytes;
    job.spu_result_bytes = spu_result_bytes;
    job.result_bytes     = result_bytes;
    job.result_values    = result_values;
    job.result_words     = result_words;
    // P3 scale_words counts PARAM words; result_words keeps the raw result geometry.
    job.scale_words      = (uint32_t) (scale_bytes / 16U);
    job.p3_split_scale   = g_p3_matmul_active;
    job.scratch_bytes    = g_p3_matmul_active ? fpga_p3_scratch_bytes(group_blocks) : 0U;
    job.p3_act_group     = g_p3_matmul_active ? act_group : nullptr;
    job.p3_scratch_valid = nullptr;
    job.weight_src_off   = weight_stage_off;
    job.scale_src_off    = scale_stage_off;
    job.p2_residency_slot = P2_WEIGHT_RESIDENCY_NO_SLOT;
    job.act_group        = act_group;
    job.src0             = src0;

    if (!range_fits(scale_stage_off, scale_bytes, SPU_PARAM_BASE, SPU_PARAM_END) ||
        !ddr_range_fits(scale_stage_off, scale_bytes)) {
        LOGE("P2 scale source window rejected off=0x%08x bytes=%zu action=no_dma", scale_stage_off, scale_bytes);
        return false;
    }

    fpga_p2_scale_shape_t scale_shape = {};
    const bool scale_shape_ok = job.p3_split_scale ?
        fpga_p3_checked_scale_shape(rows, group_blocks, &scale_shape.entries, &scale_shape.bytes) :
        fpga_p2_checked_scale_shape(rows, group_blocks, job.result_values, job.result_words, job.scale_bytes,
                                   scale_stage_off, &scale_shape);
    if (job.p3_split_scale) scale_shape.words = scale_shape.bytes / 16U;
    if (!scale_shape_ok || scale_shape.bytes != job.scale_bytes) {
        LOGE(
            "P2 scale shape rejected job=%u tile=%u rows=%d group_blocks=%d result_values=%u result_words=%u "
            "scale_bytes=%zu spu_param=0x%08x action=no_write_no_dma_no_start",
            job.job_id, job.tile_id, rows, group_blocks, job.result_values, job.result_words, job.scale_bytes,
            scale_stage_off);
        return false;
    }
    // Admit the complete scale range before either direct producer can write.
    volatile uint32_t * const scale_words = ddr_checked_u32_ptr(scale_stage_off, job.scale_bytes);

    if (g_p2_init_requested) {
        p2_trace_set_job_context(job);
    }

    const long long prep0       = now_us();
    const long long event_prep0 = p2_event_now_us();
    job.event_prep_begin_us     = event_prep0;
    // Preparation owns only CPU-visible DDR staging.  In particular, never
    // write the global descriptor register file here: the ping-pong scheduler
    // may prepare N+1 while N is still executing, and descriptor metadata is
    // not banked.  Submit performs the deferred descriptor/config commit only
    // after the prior job has been drained, accumulated, and retired FREE.

    const long long weight_select0 = now_us();
    const uint32_t residency_slot =
        g_p2_weight_residency_enabled ?
            fpga_p2_residency_select_or_build(src0, weight_data_base, row0, rows, k_block0, group_blocks, group_beats) :
            P2_WEIGHT_RESIDENCY_NO_SLOT;
    if (residency_slot == P2_WEIGHT_RESIDENCY_INVALID_SLOT) {
        LOGE("P2_RESIDENCY_HOST_METADATA_FAIL job=%u tile=%u action=no_dma_no_start_no_direct_stage", job.job_id,
             job.tile_id);
        return false;
    }
    if (residency_slot != P2_WEIGHT_RESIDENCY_NO_SLOT) {
        const fpga_p2_resident_tile_t & resident = g_p2_resident_tiles[residency_slot];
        const size_t resident_scale_bytes = (size_t) rows * (size_t) group_blocks * sizeof(uint16_t);
        if (!fpga_p2_residency_identity_matches(resident, src0, row0, rows, k_block0, group_blocks, group_beats,
                                                weight_bytes, resident_scale_bytes)) {
            fpga_p2_residency_poison_slot(residency_slot, "prepare_host_metadata_mismatch");
            LOGE("P2_RESIDENCY_HOST_METADATA_FAIL job=%u tile=%u slot=%u action=no_dma_no_start", job.job_id,
                 job.tile_id, residency_slot);
            return false;
        }
        job.weight_src_off      = resident.qs_off;
        job.p2_residency_hit    = true;
        job.p2_residency_slot   = residency_slot;
        job.p2_residency_epoch  = resident.epoch;
        job.p2_residency_seal   = resident.seal;
    }

    const long long weight_select_us = now_us() - weight_select0;
    if (totals) {
        totals->prep_weight_select_us += weight_select_us;
    }

    const bool direct_source = !job.p2_residency_hit;
    const bool direct_reused = direct_source && reuse_direct_weight && staging->bytes == weight_bytes;
    bool fused_direct_pack = false;
    const long long direct_weight_pack0 = direct_source && !direct_reused ? now_us() : 0;
    if (direct_source && !direct_reused) {
        size_t direct_payload_bytes = 0;
        const size_t pair_count       = ((size_t) rows + 1U) / 2U;
        const size_t group_beats_size = (size_t) group_beats;
        if (pair_count > (size_t) INT_MAX / 2U || group_beats_size > std::numeric_limits<size_t>::max() / pair_count ||
            pair_count * group_beats_size > std::numeric_limits<size_t>::max() / 32U ||
            row0 < 0 || row0 > INT64_MAX - (int64_t) rows || k_block0 < 0 || k_block0 > INT64_MAX - (int64_t) group_blocks ||
            !fpga_weight_layout_payload_bytes(rows, group_beats, &direct_payload_bytes) ||
            direct_payload_bytes != weight_bytes || direct_payload_bytes != job.weight_bytes ||
            direct_payload_bytes != pair_count * group_beats_size * 32U ||
            direct_payload_bytes > (size_t) UINT32_MAX - (size_t) weight_stage_off ||
            !range_fits(weight_stage_off, direct_payload_bytes, WEIGHT_BASE, WEIGHT_END) ||
            !ddr_range_fits(weight_stage_off, direct_payload_bytes) ||
            direct_payload_bytes > UINT64_MAX - g_p2_residency_direct_weight_pack_bytes) {
            LOGE(
                "P2 direct WEIGHT pack precondition failed job=%u tile=%u rows=%d group_beats=%d payload=%zu "
                "weight_bytes=%zu job_weight_bytes=%zu pair_count=%zu ddr_size=0x%zx action=no_write_no_dma_no_start",
                job.job_id, job.tile_id, rows, group_beats, direct_payload_bytes, weight_bytes, job.weight_bytes,
                pair_count, g_ddr_map_size);
            return false;
        }

        volatile uint32_t * direct_weight_words = ddr_checked_u32_ptr(weight_stage_off, direct_payload_bytes);
        const size_t words_per_pair = group_beats_size * 8U;
        const size_t expected_words = direct_payload_bytes / sizeof(uint32_t);
        if (words_per_pair == 0U || pair_count > std::numeric_limits<size_t>::max() / words_per_pair ||
            pair_count * words_per_pair != expected_words) {
            fpga_fatal(
                "P2 direct WEIGHT pair-range arithmetic failed job=%u tile=%u pair_count=%zu group_beats=%d "
                "words_per_pair=%zu expected_words=%zu action=no_dma_no_start",
                job.job_id, job.tile_id, pair_count, group_beats, words_per_pair, expected_words);
        }

        size_t written_words = 0U;
        const bool use_parallel_pack = g_p2_pack_workers_requested == 2 && pair_count >= 2U &&
                                       direct_payload_bytes >= FPGA_P2_PACK_PARALLEL_MIN_BYTES;
        if (use_parallel_pack) {
            const long long dispatch0 = now_us();
            const size_t main_pair_end = pair_count / 2U;
            const size_t helper_words = (pair_count - main_pair_end) * words_per_pair;
            const size_t main_scale_rows = main_pair_end * 2U;
            const size_t helper_scale_rows = (size_t) rows - main_scale_rows;
            if (main_scale_rows > (size_t) rows ||
                main_scale_rows > std::numeric_limits<size_t>::max() / (size_t) group_blocks ||
                helper_scale_rows > std::numeric_limits<size_t>::max() / (size_t) group_blocks) {
                fpga_fatal(
                    "P2 direct WEIGHT+SCALE parallel split arithmetic failed job=%u tile=%u rows=%d pairs=%zu "
                    "split=%zu action=no_write_no_dma_no_start",
                    job.job_id, job.tile_id, rows, pair_count, main_pair_end);
            }
            const size_t expected_main_scale_entries = main_scale_rows * (size_t) group_blocks;
            const size_t expected_helper_scale_entries = helper_scale_rows * (size_t) group_blocks;
            if (main_pair_end == 0U || helper_words == 0U || direct_payload_bytes > UINT64_MAX - g_p2_pack_parallel_bytes) {
                fpga_fatal(
                    "P2 direct WEIGHT parallel split precondition failed job=%u tile=%u pairs=%zu split=%zu "
                    "payload=%zu action=no_dma_no_start",
                    job.job_id, job.tile_id, pair_count, main_pair_end, direct_payload_bytes);
            }
            uint64_t generation = 0U;
            if (!fpga_p2_pack_worker_next_generation(&generation)) {
                fpga_fatal(
                    "P2 direct WEIGHT helper is unavailable before pack job=%u tile=%u action=no_write_no_dma_no_start",
                    job.job_id, job.tile_id);
            }
            fpga_p2_pack_worker_task_t task = {
                src0, weight_data_base, row0, k_block0, rows, group_blocks, group_beats,
                main_pair_end, pair_count, direct_weight_words, helper_words, generation,
            };
            task.activation_data_base = act_group;
            task.scale_words = scale_words;
            task.expected_scale_entries = expected_helper_scale_entries;
            task.fused_scale = direct_source && !direct_reused;
            task.p3_scale = task.fused_scale && job.p3_split_scale;
            task.p3_scale_halves = reinterpret_cast<volatile uint16_t *>(scale_words);
            fused_direct_pack = task.fused_scale;
            if (!fpga_p2_pack_worker_submit(task)) {
                fpga_fatal(
                    "P2 direct WEIGHT helper submit failed job=%u tile=%u generation=%llu "
                    "action=no_write_no_dma_no_start",
                    job.job_id, job.tile_id, (unsigned long long) generation);
            }

            const long long main_pack0 = now_us();
            g_pack_detail[PACK_DISPATCH] += main_pack0 - dispatch0;
            size_t main_words = 0U;
            size_t main_scale_entries = 0U;
            const bool main_ok = fused_direct_pack && job.p3_split_scale ?
                fpga_pack_direct_weight_p3scale_pair_range(
                    direct_weight_words, reinterpret_cast<volatile uint16_t *>(scale_words), src0, weight_data_base,
                    row0, k_block0, rows, group_blocks, group_beats, 0U, main_pair_end, true,
                    &main_words, &main_scale_entries) : fused_direct_pack ?
                fpga_pack_direct_weight_scale_pair_range(
                    direct_weight_words, scale_words, src0, weight_data_base, act_group, row0, k_block0, rows,
                    group_blocks, group_beats, 0U, main_pair_end, true, &main_words, &main_scale_entries) :
                fpga_pack_direct_weight_pair_range(
                    direct_weight_words, src0, weight_data_base, row0, k_block0, rows, group_blocks, group_beats,
                    0U, main_pair_end, true, &main_words);
            const long long main_pack_end = now_us();
            const long long main_pack_us = main_pack_end - main_pack0;
            // The caller publishes its prefix before it waits for the helper
            // and performs the existing full-WEIGHT coherency sequence.
            mmio_fence();
            const long long caller_fence_end = now_us();
            g_pack_detail[PACK_FENCE] += caller_fence_end - main_pack_end;
            size_t helper_written_words = 0U;
            size_t helper_scale_entries = 0U;
            long long helper_service_us = 0;
            long long caller_wait_us = 0;
            const bool helper_ok = fpga_p2_pack_worker_wait(generation, &helper_written_words, &helper_service_us,
                                                              &caller_wait_us, fused_direct_pack ?
                                                                                                  &helper_scale_entries :
                                                                                                  nullptr);
            if (!main_ok || !helper_ok || main_words != main_pair_end * words_per_pair ||
                helper_written_words != helper_words || main_words > expected_words ||
                helper_written_words > expected_words - main_words ||
                (fused_direct_pack &&
                 (main_scale_entries != expected_main_scale_entries ||
                  helper_scale_entries != expected_helper_scale_entries))) {
                fpga_fatal(
                    "P2 direct WEIGHT+SCALE parallel pack completion mismatch job=%u tile=%u generation=%llu "
                    "fused=%d main_ok=%d helper_ok=%d main_words=%zu helper_words=%zu expected_main=%zu "
                    "expected_helper=%zu main_scale_entries=%zu helper_scale_entries=%zu expected_main_scale=%zu "
                    "expected_helper_scale=%zu "
                    "action=no_dma_no_start",
                    job.job_id, job.tile_id, (unsigned long long) generation, fused_direct_pack ? 1 : 0,
                    main_ok ? 1 : 0, helper_ok ? 1 : 0, main_words, helper_written_words,
                    main_pair_end * words_per_pair, helper_words, main_scale_entries, helper_scale_entries,
                    expected_main_scale_entries, expected_helper_scale_entries);
            }
            written_words = main_words + helper_written_words;
            g_p2_pack_parallel_jobs++;
            g_p2_pack_parallel_bytes += (uint64_t) direct_payload_bytes;
            g_p2_pack_main_us += main_pack_us;
            g_p2_pack_helper_service_us += helper_service_us;
            g_p2_pack_caller_wait_us += caller_wait_us;
            g_pack_detail[PACK_MAIN] += main_pack_us;
            g_pack_detail[PACK_WAIT] += caller_wait_us;
            g_pack_detail[PACK_HELPER] += helper_service_us;
            g_pack_detail[PACK_PARALLEL_JOBS]++;
            if (fused_direct_pack) {
                g_p2_pack_fused_jobs++;
                g_p2_pack_fused_scale_entries += (long long) (main_scale_entries + helper_scale_entries);
            }
        } else {
            if (g_p2_pack_workers_requested == 2) {
                g_p2_pack_serial_threshold_skips++;
            }
            const long long serial0 = now_us();
            if (!fpga_pack_direct_weight_pair_range(direct_weight_words, src0, weight_data_base, row0, k_block0,
                                                     rows, group_blocks, group_beats, 0U, pair_count, true,
                                                     &written_words)) {
                fpga_fatal(
                    "P2 direct WEIGHT serial pair-range pack failed job=%u tile=%u action=no_dma_no_start",
                    job.job_id, job.tile_id);
            }
            g_pack_detail[PACK_SERIAL] += now_us() - serial0;
            g_pack_detail[PACK_SERIAL_JOBS]++;
        }
        if (written_words != expected_words) {
            fpga_fatal(
                "P2 direct WEIGHT pack word count mismatch job=%u tile=%u written_words=%zu expected_words=%zu "
                "action=no_dma_no_start",
                job.job_id, job.tile_id, written_words, expected_words);
        }
        g_p2_residency_direct_weight_pack_bytes += (uint64_t) direct_payload_bytes;
    }
    if (direct_weight_pack0 != 0) {
        const long long direct_weight_pack_us = now_us() - direct_weight_pack0;
        g_p2_residency_direct_weight_pack_us += direct_weight_pack_us;
        g_pack_detail[PACK_TOTAL] += direct_weight_pack_us;
        if (totals) {
            totals->prep_direct_weight_pack_us += direct_weight_pack_us;
        }
        if (fused_direct_pack) {
            g_p2_pack_fused_us += direct_weight_pack_us;
        }
    } else if (job.p2_residency_hit) {
        g_p2_residency_avoided_cpu_pack_bytes += (long long) job.weight_bytes;
    }

    // Fused direct jobs emitted every live scale entry in the two producer
    // ranges above.  The scale timer therefore measures only residual padding
    // and fence work for those jobs; direct-pack timing explicitly includes
    // the fused live scale emission.
    const long long scale_pack0 = now_us();
    // Check the complete P2 PARAM range once before the first volatile
    // store.  Every live entry is written exactly once; only the 0..3
    // unused lanes in the final 128-bit word are cleared.
    volatile uint32_t * scale_out = scale_words;
    const auto store_scale_pair = [](volatile uint32_t * dst, uint32_t first, uint32_t second) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // Callers align the pair before entering their two-entry loop.
        *reinterpret_cast<volatile uint64_t *>(dst) =
            static_cast<uint64_t>(first) | (static_cast<uint64_t>(second) << 32U);
#else
        dst[0] = first;
        dst[1] = second;
#endif
    };
    if (!fused_direct_pack && job.p2_residency_hit) {
        // The matmul lock keeps resident metadata stable throughout preparation.
        // Validate the entire source before writing any scale entries.
        if (job.p2_residency_slot >= g_p2_resident_tiles.size() ||
            !fpga_p2_residency_host_metadata_shape_valid(g_p2_resident_tiles[job.p2_residency_slot]) ||
            g_p2_resident_tiles[job.p2_residency_slot].scale_count != scale_shape.entries) {
            fpga_p2_residency_poison_slot(job.p2_residency_slot, "param_host_scale_read_invalid");
            LOGE("P2_RESIDENCY_HOST_METADATA_FAIL job=%u tile=%u slot=%u action=no_dma_no_start", job.job_id,
                 job.tile_id, job.p2_residency_slot);
            return false;
        }
        const uint16_t * weight_scale = g_p2_resident_tiles[job.p2_residency_slot].scale_bits.data();
        for (int row = 0; row < rows; ++row) {
            if (job.p3_split_scale) {
                fpga_p3_store_scale_row(reinterpret_cast<volatile uint16_t *>(scale_words) + (size_t) row * group_blocks,
                                        weight_scale + (size_t) row * group_blocks, group_blocks);
                continue;
            }
            int gb = 0;
            // An odd preceding row can leave the next row only 4-byte aligned.
            if ((reinterpret_cast<uintptr_t>(scale_out) & 7U) != 0U) {
                *scale_out++ = fpga_p2_pack_scale_entry((uint16_t) act_group[gb++].d, *weight_scale++);
            }
            for (; gb + 1 < group_blocks; gb += 2) {
                const uint32_t first = fpga_p2_pack_scale_entry((uint16_t) act_group[gb].d, weight_scale[0]);
                const uint32_t second = fpga_p2_pack_scale_entry((uint16_t) act_group[gb + 1].d, weight_scale[1]);
                store_scale_pair(scale_out, first, second);
                scale_out += 2;
                weight_scale += 2;
            }
            if (gb < group_blocks) {
                *scale_out++ = fpga_p2_pack_scale_entry((uint16_t) act_group[gb].d, *weight_scale++);
            }
        }
    } else if (!fused_direct_pack) {
        for (int row = 0; row < rows; ++row) {
            const block_q8_0_t * weight_row =
                weight_block_from_base(src0, weight_data_base, row0 + row, k_block0);
            if (job.p3_split_scale) {
                uint16_t row_buf[64];
                for (int gb = 0; gb < group_blocks; ++gb) row_buf[gb] = (uint16_t) weight_row[gb].d;
                fpga_p3_store_scale_row(reinterpret_cast<volatile uint16_t *>(scale_words) + (size_t) row * group_blocks,
                                        row_buf, group_blocks);
                continue;
            }
            int gb = 0;
            if ((reinterpret_cast<uintptr_t>(scale_out) & 7U) != 0U) {
                *scale_out++ = fpga_p2_pack_scale_entry((uint16_t) act_group[gb].d, (uint16_t) weight_row[gb].d);
                ++gb;
            }
            for (; gb + 1 < group_blocks; gb += 2) {
                const uint32_t first = fpga_p2_pack_scale_entry((uint16_t) act_group[gb].d, (uint16_t) weight_row[gb].d);
                const uint32_t second = fpga_p2_pack_scale_entry((uint16_t) act_group[gb + 1].d, (uint16_t) weight_row[gb + 1].d);
                store_scale_pair(scale_out, first, second);
                scale_out += 2;
            }
            if (gb < group_blocks) {
                *scale_out++ = fpga_p2_pack_scale_entry((uint16_t) act_group[gb].d, (uint16_t) weight_row[gb].d);
            }
        }
    }
    if (job.p3_split_scale) {
        volatile uint16_t * halves = reinterpret_cast<volatile uint16_t *>(scale_words);
        for (size_t i = scale_shape.entries; i < job.scale_bytes / sizeof(uint16_t); ++i) halves[i] = 0U;
    } else {
        for (size_t linear = scale_shape.entries;
             linear < scale_shape.words * (size_t) VPU_RESULT_PACK_LANES;
             ++linear) {
            scale_words[linear] = 0U;
        }
    }
    mmio_fence();

    const long long scale_pack_us = now_us() - scale_pack0;
    if (totals) {
        totals->prep_scale_pack_us += scale_pack_us;
    }
    if (job.p2_residency_hit) {
        g_p2_residency_resident_param_us += scale_pack_us;
    }

    const long long act_pack0 = now_us();
    for (int gb = 0; gb < group_blocks; ++gb) {
        const block_q8_0_t & act = act_group[gb];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            const uint32_t word_index = (uint32_t) gb * (uint32_t) VPU_BLOCK_BEATS + (uint32_t) beat;
            write_i8x16_to_ddr(ACT_BASE + word_index * 16U, act.qs + beat * VPU_NUM_LANES);
        }
    }

    mmio_fence();
    const long long act_pack_us = now_us() - act_pack0;
    const long long event_prep_done = p2_event_now_us();
    job.event_prep_done_us = event_prep_done;

    if (totals) {
        totals->prep_act_pack_us += act_pack_us;
        totals->prep_us += now_us() - prep0;
        if (!job.p2_residency_hit) {
            totals->weight_cache_misses++;
        }
    }
    p2_event_trace(job, "PREP_DONE", event_prep_done, "prep_us", event_prep_done - event_prep0);
    if (staging && direct_source) {
        staging->src0 = src0;
        staging->data = weight_data_base;
        staging->ddr = g_ddr;
        staging->row0 = row0;
        staging->rows = rows;
        staging->k_block0 = k_block0;
        staging->group_blocks = group_blocks;
        staging->bytes = weight_bytes;
        staging->epoch = g_p2_weight_residency_epoch;
        staging->weight_off = job.weight_src_off;
        staging->scale_off = job.scale_src_off;
        staging->valid = true;
    }
    return true;
}

static void log_sequence_epoch_close_if_needed(int seq_pos, int64_t m) {
    const long long now = monotonic_now_us();
    if (g_last_token_seq == INT_MIN) {
        g_last_token_seq = seq_pos;
        g_last_token_us  = now;
        g_token_matmuls  = 0;
        g_last_epoch_first_hook_m = m;
        return;
    }
    if (seq_pos != g_last_token_seq) {
        const double epoch_ms = (double) (now - g_last_token_us) / 1000.0;
        fpga_log_line(g_stage_timing_enabled, "P2_SEQ_EPOCH_CLOSE", false,
                      "graph_seq=%d next_graph_seq=%d matmuls=%lld epoch_ms=%.3f est_graph_seq_s=%.3f "
                      "semantics=observed_graph_sequence_change",
                      g_last_token_seq, seq_pos, g_token_matmuls, epoch_ms,
                      epoch_ms > 0.0 ? 1000.0 / epoch_ms : 0.0);
        g_last_token_seq = seq_pos;
        g_last_token_us  = now;
        g_token_matmuls  = 0;
        g_last_epoch_first_hook_m = m;
    }
}


// ============================================================================
// TOKEN, BOTTLENECK AND SCHEDULER TELEMETRY
// ============================================================================

static void fpga_p1_sched_summary_emit(const char * reason) {
    if (!g_p1_sched_summary_enabled || !g_p1_sched_summary.active) {
        return;
    }
    // Buffered and file-only: no terminal emission, forced flush, MMIO, or
    // fence is associated with a scheduler summary.
    fpga_log_line(
        true, "P1_SCHED_SUMMARY", false,
        "scope=graph_sequence reason=%s graph_seq=%d scheduler=%s matmuls=%lld vpu_runs=%lld pingpong_pairs=%lld "
        "ip_compute_us=%.0f dma_act_us=%lld dma_weight_us=%lld matrix_wall_us=%lld "
        "overlap_duration=not_measured",
        reason ? reason : "?", g_p1_sched_summary.graph_seq,
        g_active_matmul_pingpong ? "pingpong" : "single_bank", g_p1_sched_summary.matmuls,
        g_p1_sched_summary.vpu_runs, g_p1_sched_summary.pingpong_pairs,
        fpga_compute_report_us(g_p1_sched_summary.ip_compute_us), g_p1_sched_summary.dma_act_us, g_p1_sched_summary.dma_weight_us,
        g_p1_sched_summary.matrix_wall_us);
    g_p1_sched_summary = {};
}

static void fpga_p1_sched_summary_begin_graph(int graph_seq) {
    if (!g_p1_sched_summary_enabled) {
        return;
    }
    if (g_p1_sched_summary.active && g_p1_sched_summary.graph_seq != graph_seq) {
        fpga_p1_sched_summary_emit("graph_sequence_change");
    }
    if (!g_p1_sched_summary.active) {
        g_p1_sched_summary.active    = true;
        g_p1_sched_summary.graph_seq = graph_seq;
    }
}

static long long p2_event_now_us(void) {
    return (g_p2_event_trace_enabled || g_token_timing_collection_enabled) ?
               monotonic_now_us() :
               0;
}

static fpga_bottleneck_category_t fpga_bottleneck_category(const char * tensor_name) {
    if (!tensor_name) {
        return FPGA_BOTTLENECK_OTHER;
    }
    if (strstr(tensor_name, ".ffn_gate.") != nullptr) {
        return FPGA_BOTTLENECK_FFN_GATE;
    }
    if (strstr(tensor_name, ".ffn_up.") != nullptr) {
        return FPGA_BOTTLENECK_FFN_UP;
    }
    if (strstr(tensor_name, ".ffn_down.") != nullptr) {
        return FPGA_BOTTLENECK_FFN_DOWN;
    }
    if (strstr(tensor_name, ".attn_") != nullptr || strstr(tensor_name, ".attention.") != nullptr) {
        return FPGA_BOTTLENECK_ATTN;
    }
    return FPGA_BOTTLENECK_OTHER;
}

static void fpga_token_timing_reset(void) {
    g_token_timing = {};
    g_zdma_split = {};
    g_p3_graph_mode_switches_start = g_p3_mode_switches;
    g_p2_residency_graph_builds_start = 0;
    g_p2_residency_graph_build_us_start = 0;
}

static bool fpga_token_timing_emit(int next_graph_seq,
                                   int64_t     ubatch_tokens,
                                   const char * reason,
                                   long long    end_mono_us,
                                   bool         force_flush) {
    if (!g_token_timing_collection_enabled || !g_token_timing.active) {
        return false;
    }
    if (end_mono_us <= 0) {
        end_mono_us = monotonic_now_us();
    }
    const long long token_wall_us =
        end_mono_us >= g_token_timing.start_mono_us ? end_mono_us - g_token_timing.start_mono_us : 0;
    const long long h2ip_dma_us = g_token_timing.act_dma_us + g_token_timing.weight_dma_us +
                                  g_token_timing.scale_dma_us + g_token_timing.scratch_dma_us;
    const long long token_read_us = g_token_timing.ip2host_dma_us + g_token_timing.host_read_us;
    const long long device_span_us =
        g_token_timing.first_ip_launch_mono_us > 0 &&
                g_token_timing.last_ip_output_ready_mono_us >= g_token_timing.first_ip_launch_mono_us ?
            g_token_timing.last_ip_output_ready_mono_us - g_token_timing.first_ip_launch_mono_us :
            0;
    const bool decode_token = ubatch_tokens == 1 && g_token_timing.first_m == 1;
    const char * scope = decode_token ? "decode_token" : (ubatch_tokens > 0 ? "prefill_or_ubatch" : "incomplete");
    fpga_dma_profile_graph_end(scope, decode_token, true);
    fpga_log_graph_header(g_token_timing.graph_seq, scope);
    std::array<long long, PACK_DETAIL_COUNT> pack_delta = {};
    for (size_t i = 0; i < pack_delta.size(); ++i) {
        pack_delta[i] = g_pack_detail[i] - g_pack_detail_snapshot[i];
        if (decode_token) {
            g_pack_detail_decode[i] += pack_delta[i];
        }
    }
    g_pack_detail_snapshot = g_pack_detail;
    if (decode_token) {
        ++g_summary_detail_decode_tokens;
        ++g_fpga_perf_decode.decode_tokens;
        g_fpga_perf_decode.decode_wall_us += token_wall_us;
        g_fpga_perf_decode.fpga_matmuls += g_token_timing.matmuls;
        g_fpga_perf_decode.vpu_runs += g_token_timing.vpu_runs;
        g_fpga_perf_decode.ip_compute_us += g_token_timing.ip_compute_us;
        g_fpga_perf_decode.h2ip_dma_us += h2ip_dma_us;
        g_fpga_perf_decode.ip2host_dma_us += g_token_timing.ip2host_dma_us;
        g_fpga_perf_decode.host_result_us += g_token_timing.host_read_us;
        g_fpga_perf_decode.preparation_us += g_token_timing.prep_us;
        g_fpga_perf_decode.direct_weight_pack_us += g_token_timing.prep_direct_weight_pack_us;
        g_fpga_perf_decode.scale_pack_us += g_token_timing.prep_scale_pack_us;
        g_fpga_perf_decode.zdma_descriptors += g_token_timing.zdma_descriptors;
        g_fpga_perf_decode.zdma_bytes += g_token_timing.zdma_bytes;
        g_fpga_perf_decode.zdma_elapsed_us += g_token_timing.zdma_elapsed_us;
        g_fpga_perf_decode.pingpong_handoffs += g_token_timing.scheduler_handoffs;
        g_fpga_perf_decode.pingpong_bank_jobs[0] += g_token_timing.bank_jobs[0];
        g_fpga_perf_decode.pingpong_bank_jobs[1] += g_token_timing.bank_jobs[1];
        g_fpga_perf_decode.pingpong_prepare_overlap_us += g_token_timing.scheduler_prepare_overlap_us;
        g_fpga_perf_decode.pingpong_prepare_late_us += g_token_timing.scheduler_prepare_late_us;
        g_fpga_perf_decode.pingpong_prepare_late_jobs += g_token_timing.scheduler_prepare_late_jobs;
    }
    const bool sampled_detail = g_summary_detail_after_error ||
                                (decode_token &&
                                 (g_summary_detail_decode_tokens == 1 ||
                                  (g_summary_detail_every > 0 &&
                                   (g_summary_detail_decode_tokens % g_summary_detail_every) == 0)));
    const long long graph_builds = g_p2_residency_builds - g_p2_residency_graph_builds_start;
    // build_us includes time spent in failed build attempts, as in the lifetime counter.
    const long long graph_build_us = g_p2_residency_build_us - g_p2_residency_graph_build_us_start;
    if (decode_token) {
        g_p2_residency_decode_builds += graph_builds;
        g_p2_residency_decode_build_us += graph_build_us;
    }
    fpga_log_line(true, "P2 RESIDENCY SUMMARY", force_flush,
                  "scope=graph Graph=%d decode=%d builds=%lld build_us_including_failed_attempts=%lld "
                  "decode_builds=%lld decode_build_us_including_failed_attempts=%lld",
                  g_token_timing.graph_seq, decode_token ? 1 : 0, graph_builds, graph_build_us,
                  g_p2_residency_decode_builds, g_p2_residency_decode_build_us);
    g_token_timing.p3_mode_switches = g_p3_mode_switches - g_p3_graph_mode_switches_start;
    g_summary_detail_after_error = false;

    fpga_token_timing_log_t report = {};
    report.graph_seq = g_token_timing.graph_seq;
    report.force_flush = force_flush;
    report.p3_jobs = g_token_timing.p3_jobs;
    report.drain_wait_us = g_token_timing.drain_wait_us;
    report.drain_wait_jobs = g_token_timing.drain_wait_jobs;
    report.scratch_dma_us = g_token_timing.scratch_dma_us;
    report.scratch_bytes = g_token_timing.scratch_bytes;
    report.scratch_desc = g_token_timing.scratch_desc;
    report.p3_reject_delta = g_token_timing.p3_reject_delta;
    report.p3_mode_switches = g_token_timing.p3_mode_switches;

    if (g_token_timing_enabled || g_bottleneck_summary_enabled) {
        report.timing_enabled = g_token_timing_enabled;
        report.bottleneck_summary_enabled = g_bottleneck_summary_enabled;
        report.compute_watch_enabled = g_compute_watch_enabled;
        report.sampled_detail = sampled_detail;
        report.force_flush = force_flush;
        report.next_graph_seq = next_graph_seq;
        report.ubatch_tokens = (long long) ubatch_tokens;
        report.scope = scope;
        report.reason = reason;
        report.token_wall_us = token_wall_us;
        report.h2ip_dma_us = h2ip_dma_us;
        report.token_read_us = token_read_us;
        report.device_span_us = device_span_us;
        report.max_descriptor_bytes = g_zdma_max_transfer_bytes;
        report.graph_seq = g_token_timing.graph_seq;
        report.matmuls = g_token_timing.matmuls;
        report.vpu_runs = g_token_timing.vpu_runs;
        report.prep_us = g_token_timing.prep_us;
        report.prep_weight_select_us = g_token_timing.prep_weight_select_us;
        report.prep_direct_weight_pack_us = g_token_timing.prep_direct_weight_pack_us;
        report.prep_scale_pack_us = g_token_timing.prep_scale_pack_us;
        report.prep_act_pack_us = g_token_timing.prep_act_pack_us;
        report.matmul_wall_us = g_token_timing.matmul_wall_us;
        report.act_dma_us = g_token_timing.act_dma_us;
        report.weight_dma_us = g_token_timing.weight_dma_us;
        report.scale_dma_us = g_token_timing.scale_dma_us;

        report.ip_compute_us = g_token_timing.ip_compute_us;
        report.ip2host_dma_us = g_token_timing.ip2host_dma_us;
        report.host_read_us = g_token_timing.host_read_us;
        report.host_accum_us = g_token_timing.host_accum_us;
        report.scheduler_prepare_overlap_us = g_token_timing.scheduler_prepare_overlap_us;
        report.scheduler_prepare_late_us = g_token_timing.scheduler_prepare_late_us;
        report.scheduler_prepare_headroom_us = g_token_timing.scheduler_prepare_headroom_us;
        report.scheduler_output_to_launch_us = g_token_timing.scheduler_output_to_launch_us;
        report.scheduler_retire_to_launch_us = g_token_timing.scheduler_retire_to_launch_us;
        report.scheduler_handoffs = g_token_timing.scheduler_handoffs;
        report.scheduler_prepare_late_jobs = g_token_timing.scheduler_prepare_late_jobs;
        report.zdma_descriptors = g_token_timing.zdma_descriptors;
        report.zdma_polls = g_token_timing.zdma_polls;
        report.zdma_zero_poll_descriptors = g_token_timing.zdma_zero_poll_descriptors;
        report.zdma_saw_enabled_descriptors = g_token_timing.zdma_saw_enabled_descriptors;
        report.zdma_elapsed_us = g_token_timing.zdma_elapsed_us;
        report.zdma_bytes = g_token_timing.zdma_bytes;
        report.zdma_act_descriptors = g_token_timing.zdma_act_descriptors;
        report.zdma_weight_descriptors = g_token_timing.zdma_weight_descriptors;
        report.zdma_scale_descriptors = g_token_timing.zdma_scale_descriptors;
        report.zdma_result_descriptors = g_token_timing.zdma_result_descriptors;
        report.zdma_other_descriptors = g_token_timing.zdma_other_descriptors;
        report.activation_bytes = g_token_timing.activation_bytes;
        report.weight_bytes = g_token_timing.weight_bytes;
        report.scale_bytes = g_token_timing.scale_bytes;
        report.result_bytes = g_token_timing.result_bytes;
        const int category_order[5] = { FPGA_BOTTLENECK_ATTN, FPGA_BOTTLENECK_FFN_GATE,
                                        FPGA_BOTTLENECK_FFN_UP, FPGA_BOTTLENECK_FFN_DOWN,
                                        FPGA_BOTTLENECK_OTHER };
        static_assert(FPGA_BOTTLENECK_CATEGORY_COUNT == 5, "token log category count changed");
        for (int category = 0; category < 5; ++category) {
            report.category_matmuls[category] = g_token_timing.category_matmuls[category_order[category]];
            report.category_runs[category] = g_token_timing.category_runs[category_order[category]];
            report.category_wall_us[category] = g_token_timing.category_wall_us[category_order[category]];
            report.category_prep_us[category] = g_token_timing.category_prep_us[category_order[category]];
            report.category_compute_us[category] = g_token_timing.category_compute_us[category_order[category]];
            report.category_dma_us[category] = g_token_timing.category_dma_us[category_order[category]];
        }
        for (int bank = 0; bank < 2; ++bank) {
            report.bank_h2ip_us[bank] = g_token_timing.bank_h2ip_us[bank];
            report.bank_compute_us[bank] = g_token_timing.bank_compute_us[bank];
            report.bank_ip2host_us[bank] = g_token_timing.bank_ip2host_us[bank];
            report.bank_host_read_us[bank] = g_token_timing.bank_host_read_us[bank];
            report.bank_jobs[bank] = g_token_timing.bank_jobs[bank];
        }
        fpga_log_token_timing(report);
        if (g_bottleneck_summary_enabled) {
            static const char * const types[] = { "ACT", "WEIGHT", "SPU_SCALE", "RESULT", "OTHER" };
            for (size_t type = 0; type < g_zdma_split.size(); ++type) {
                const fpga_zdma_split_t & split = g_zdma_split[type];
                fpga_log_line(true, "ZDMA_SPLIT", force_flush,
                              "Graph=%d type=%s desc=%lld bytes=%lld setup_us=%lld wait_us=%lld hook_us=%lld "
                              "hook_calls=%lld done_after_hook=%lld polls=%lld",
                              g_token_timing.graph_seq, types[type], split.desc, split.bytes, split.setup_us,
                              split.wait_us, split.hook_us, split.hook_calls, split.done_after_hook, split.polls);
            }
        }
    }
    fpga_dma_profile_graph_emit();
    if (!decode_token) {
        fpga_log_pack_breakdown(scope, g_token_timing.graph_seq, ubatch_tokens,
                                fpga_pack_detail_record(pack_delta));
    }
    if (decode_token) {
        fpga_log_result_overlap(g_token_timing.graph_seq, g_p2_result_overlap_enabled,
                                g_token_timing.result_overlap_jobs, g_token_timing.result_overlap_host_us);
        fpga_log_p3_split(report, g_p3_split_scale_enabled);
    }
    fpga_log_token_diagnostics(report);
    fpga_token_timing_reset();
    return true;
}

static bool fpga_token_timing_emit_final(long long end_mono_us) {
    if (!g_token_timing_collection_enabled || !g_token_timing.active) {
        return false;
    }
    return fpga_token_timing_emit(g_token_timing.graph_seq, g_token_timing.first_m, "cleanup_final", end_mono_us,
                                  true);
}

static void fpga_token_timing_begin(int graph_seq, int64_t m) {
    if (!g_token_timing_collection_enabled) {
        return;
    }
    const long long now = monotonic_now_us();
    if (g_token_timing.active && g_token_timing.graph_seq != graph_seq) {
        const int inferred_delta = graph_seq > g_token_timing.graph_seq ? graph_seq - g_token_timing.graph_seq : 0;
        fpga_token_timing_emit(graph_seq, inferred_delta, "observed_sequence_change_without_advance_hook", now);
    }
    if (!g_token_timing.active) {
        fpga_dma_profile_graph_begin(graph_seq);
        g_p3_graph_mode_switches_start = g_p3_mode_switches;
        g_p2_residency_graph_builds_start = g_p2_residency_builds;
        g_p2_residency_graph_build_us_start = g_p2_residency_build_us;
        g_token_timing.active              = true;
        g_token_timing.graph_seq           = graph_seq;
        g_token_timing.first_m             = m;
        g_token_timing.start_mono_us       = now;
        g_token_timing.last_update_mono_us = now;
    }
}

static void fpga_token_timing_accumulate(const fpga_stage_totals_t & totals, long long matmul_wall_us,
                                                 const char * tensor_name) {
    if (!g_token_timing_collection_enabled || !g_token_timing.active) {
        return;
    }
    g_token_timing.matmuls++;
    g_token_timing.vpu_runs += totals.vpu_runs;
    g_token_timing.prep_us += totals.prep_us;
    g_token_timing.prep_weight_select_us += totals.prep_weight_select_us;
    g_token_timing.prep_direct_weight_pack_us += totals.prep_direct_weight_pack_us;
    g_token_timing.prep_scale_pack_us += totals.prep_scale_pack_us;
    g_token_timing.prep_act_pack_us += totals.prep_act_pack_us;
    g_token_timing.matmul_wall_us += matmul_wall_us;
    g_token_timing.act_dma_us += totals.dma_act_us;
    g_token_timing.weight_dma_us += totals.dma_weight_us;
    g_token_timing.scale_dma_us += totals.dma_scale_us;
    g_token_timing.p3_jobs += totals.p3_jobs;
    g_token_timing.drain_wait_us += totals.drain_wait_us;
    g_token_timing.drain_wait_jobs += totals.drain_wait_jobs;
    g_token_timing.scratch_dma_us += totals.scratch_dma_us;
    g_token_timing.scratch_bytes += totals.scratch_bytes;
    g_token_timing.scratch_desc += totals.scratch_desc;
    g_token_timing.p3_reject_delta += totals.p3_reject_delta;
    g_token_timing.p3_mode_switches += totals.p3_mode_switches;

    g_token_timing.ip_compute_us += totals.ip_compute_us;
    g_token_timing.ip2host_dma_us += totals.dma_result_us;
    g_token_timing.host_read_us += totals.host_result_us;
    g_token_timing.host_accum_us += totals.host_accum_us;
    g_token_timing.scheduler_prepare_overlap_us += totals.scheduler_prepare_overlap_us;
    g_token_timing.scheduler_prepare_late_us += totals.scheduler_prepare_late_us;
    g_token_timing.scheduler_prepare_headroom_us += totals.scheduler_prepare_headroom_us;
    g_token_timing.scheduler_output_to_launch_us += totals.scheduler_output_to_launch_us;
    g_token_timing.scheduler_retire_to_launch_us += totals.scheduler_retire_to_launch_us;
    g_token_timing.scheduler_handoffs += totals.scheduler_handoffs;
    g_token_timing.scheduler_prepare_late_jobs += totals.scheduler_prepare_late_jobs;
    g_token_timing.result_overlap_jobs += totals.result_overlap_jobs;
    g_token_timing.result_overlap_host_us += totals.result_overlap_host_us;
    const int category = (int) fpga_bottleneck_category(tensor_name);
    g_token_timing.category_matmuls[category]++;
    g_token_timing.category_runs[category] += totals.vpu_runs;
    g_token_timing.category_wall_us[category] += matmul_wall_us;
    g_token_timing.category_prep_us[category] += totals.prep_us;
    g_token_timing.category_compute_us[category] += totals.ip_compute_us;
    g_token_timing.category_dma_us[category] += totals.dma_act_us + totals.dma_weight_us + totals.dma_scale_us + totals.scratch_dma_us +
                                                 totals.dma_result_us;
    for (int bank = 0; bank < 2; ++bank) {
        g_token_timing.bank_h2ip_us[bank] += totals.bank_h2ip_us[bank];
        g_token_timing.bank_compute_us[bank] += totals.bank_compute_us[bank];
        g_token_timing.bank_ip2host_us[bank] += totals.bank_ip2host_us[bank];
        g_token_timing.bank_host_read_us[bank] += totals.bank_host_read_us[bank];
        g_token_timing.bank_jobs[bank] += totals.bank_jobs[bank];
    }
    if (totals.first_ip_launch_mono_us > 0 &&
        (g_token_timing.first_ip_launch_mono_us == 0 ||
         totals.first_ip_launch_mono_us < g_token_timing.first_ip_launch_mono_us)) {
        g_token_timing.first_ip_launch_mono_us = totals.first_ip_launch_mono_us;
    }
    if (totals.last_ip_output_ready_mono_us > g_token_timing.last_ip_output_ready_mono_us) {
        g_token_timing.last_ip_output_ready_mono_us = totals.last_ip_output_ready_mono_us;
    }
    g_token_timing.activation_bytes += totals.activation_bytes;
    g_token_timing.weight_bytes += totals.weight_bytes;
    g_token_timing.scale_bytes += totals.scale_bytes;
    g_token_timing.result_bytes += totals.result_bytes;
    g_token_timing.last_update_mono_us = monotonic_now_us();
}

static void p2_event_trace(const fpga_tile_job_t & job,
                           const char *            event,
                           long long               event_mono_us,
                           const char *            duration_name,
                           long long               duration_us) {
    if (!g_p2_event_trace_enabled) {
        return;
    }

    fpga_log_line(
        true, "P2_EVT", false,
        "event=%s mono_us=%lld matmul_call_id=%llu graph_seq=%d layer=%d tensor=%s tensor_id=%u job=%u tile=%u "
        "bank=%d bank_role=%s row0=%lld rows=%d col=%lld k_block0=%lld group_blocks=%d shape=K%lld_N%lld_M%lld "
        "bytes_act=%zu bytes_weight=%zu bytes_spu_param=%zu bytes_spu_out=%zu scheduler=%s path=zdma_ddr_to_ip "
        "route=%s dst_owner=%s finish_scope=%s dst_value_ready=%d duration_name=%s duration_us=%lld",
        event ? event : "?", event_mono_us, job.matmul_call_id, job.graph_seq, job.layer_id,
        job.tensor_name ? job.tensor_name : "?", job.tensor_id, job.job_id, job.tile_id, job.bank,
        p2_bank_label(job.bank),
        (long long) job.row0, job.rows, (long long) job.col, (long long) job.k_block0, job.group_blocks,
        (long long) job.shape_k, (long long) job.shape_n, (long long) job.shape_m, job.act_bytes, job.weight_bytes,
        job.scale_bytes, job.spu_result_bytes, job.pingpong_scheduler ? "pingpong" : "single_bank",
        job.pingpong_scheduler ? "p2_pingpong" : "p2_single_bank", job.cpu_shadow_dst ? "ggml_cpu" : "fpga_host",
        event && strcmp(event, "TILE_FINISH") == 0 ? "tile_retired_partial_accum" : "not_applicable",
        event && strcmp(event, "TILE_FINISH") == 0 ? 0 : -1, duration_name ? duration_name : "none_us", duration_us);
}

static void p2_matmul_finish_trace(long long event_mono_us, long long matmul_start_us, bool contract_tile_only) {
    if (!g_p2_event_trace_enabled || !g_spu_q8_scale_stream_supported) {
        return;
    }

    fpga_log_line(
        true, "P2_MATMUL_FINISH", false,
        "mono_us=%lld matmul_call_id=%llu graph_seq=%d tensor=%s layer=%d shape=K%lld_N%lld_M%lld scheduler=%s "
        "path=zdma_ddr_to_ip route=%s dst_owner=%s finish_scope=%s dst_value_ready=%d matmul_wall_us=%lld "
        "semantics=matmul_graph_op_completion_not_generated_token",
        event_mono_us, g_active_matmul_call_id, g_active_matmul_graph_seq,
        g_active_matmul_tensor_name ? g_active_matmul_tensor_name : "?", g_active_matmul_layer_id,
        (long long) g_active_matmul_shape_k, (long long) g_active_matmul_shape_n, (long long) g_active_matmul_shape_m,
        g_active_matmul_pingpong ? "pingpong" : "single_bank",
        g_active_matmul_pingpong ? "p2_pingpong" : "p2_single_bank",
        g_active_matmul_cpu_shadow ? "ggml_cpu" : "fpga_host",
        contract_tile_only ? "contract_tile_only" : "matmul_graph_op_complete", contract_tile_only ? 0 : 1,
        event_mono_us - matmul_start_us);
}


// ============================================================================
// DESCRIPTOR AND DMA FORENSICS
// ============================================================================

static void fpga_p2_descriptor_commit_breadcrumb_before(const fpga_tile_job_t &            job,
                                                        const fpga_p2_descriptor_words_t & expected) {
    const bool emit_success = g_p2_boundary_diagnostics_enabled || g_p2_terminal_trace_enabled ||
                              g_p2_tile_trace_enabled || g_pl_scale_contract_check_limit > 0;
    if (!g_p2_init_requested || job.tile_id != 0U || !emit_success) {
        return;
    }

    FILE * fp = fpga_log_fp();
    fprintf(fp,
            "[INFO] P2_DESCRIPTOR_COMMIT edge=before job=%u bank=%d tile=%u expected bank_bits=0x%08x "
            "bank_stat_bits=0x%08x job_id=0x%08x slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x "
            "group_blocks=0x%08x token_id=0x%08x desc_flags=0x%08x actual=pending match=pending\n",
            job.job_id, job.bank, job.tile_id, expected.bank, expected.bank_stat, expected.job_id, expected.slot_state,
            expected.tensor_id, expected.row0, expected.k_block0, expected.group_blocks, expected.token_id,
            expected.desc_flags);
    fflush(fp);

    if (g_p2_terminal_trace_enabled) {
        fprintf(stderr,
                "[P2_DESCRIPTOR] edge=before job=%u bank=%d tile=%u expected bank_bits=0x%08x bank_stat_bits=0x%08x "
                "job_id=0x%08x slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x group_blocks=0x%08x "
                "token_id=0x%08x desc_flags=0x%08x actual=pending match=pending\n",
                job.job_id, job.bank, job.tile_id, expected.bank, expected.bank_stat, expected.job_id,
                expected.slot_state, expected.tensor_id, expected.row0, expected.k_block0, expected.group_blocks,
                expected.token_id, expected.desc_flags);
        fflush(stderr);
    }
}

static void fpga_p2_descriptor_commit_breadcrumb_after(const fpga_tile_job_t &            job,
                                                       const fpga_p2_descriptor_words_t & expected,
                                                       const fpga_p2_descriptor_words_t & actual,
                                                       bool                               match) {
    const bool emit_success = g_p2_boundary_diagnostics_enabled || g_p2_terminal_trace_enabled ||
                              g_p2_tile_trace_enabled || g_pl_scale_contract_check_limit > 0;
    if (!g_p2_init_requested || job.tile_id != 0U || !emit_success) {
        return;
    }

    FILE * fp = fpga_log_fp();
    fprintf(fp,
            "[INFO] P2_DESCRIPTOR_COMMIT edge=after job=%u bank=%d tile=%u expected bank_bits=0x%08x "
            "bank_stat_bits=0x%08x job_id=0x%08x slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x "
            "group_blocks=0x%08x token_id=0x%08x desc_flags=0x%08x actual bank=0x%08x bank_stat=0x%08x job_id=0x%08x "
            "slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x group_blocks=0x%08x token_id=0x%08x "
            "desc_flags=0x%08x match=%d\n",
            job.job_id, job.bank, job.tile_id, expected.bank, expected.bank_stat, expected.job_id, expected.slot_state,
            expected.tensor_id, expected.row0, expected.k_block0, expected.group_blocks, expected.token_id,
            expected.desc_flags, actual.bank, actual.bank_stat, actual.job_id, actual.slot_state, actual.tensor_id,
            actual.row0, actual.k_block0, actual.group_blocks, actual.token_id, actual.desc_flags, match ? 1 : 0);
    fflush(fp);

    if (g_p2_terminal_trace_enabled) {
        fprintf(
            stderr,
            "[P2_DESCRIPTOR] edge=after job=%u bank=%d tile=%u expected bank_bits=0x%08x bank_stat_bits=0x%08x "
            "job_id=0x%08x slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x group_blocks=0x%08x "
            "token_id=0x%08x desc_flags=0x%08x actual bank=0x%08x bank_stat=0x%08x job_id=0x%08x slot_state=0x%08x "
            "tensor_id=0x%08x row0=0x%08x k_block0=0x%08x group_blocks=0x%08x token_id=0x%08x desc_flags=0x%08x match=%d\n",
            job.job_id, job.bank, job.tile_id, expected.bank, expected.bank_stat, expected.job_id, expected.slot_state,
            expected.tensor_id, expected.row0, expected.k_block0, expected.group_blocks, expected.token_id,
            expected.desc_flags, actual.bank, actual.bank_stat, actual.job_id, actual.slot_state, actual.tensor_id,
            actual.row0, actual.k_block0, actual.group_blocks, actual.token_id, actual.desc_flags, match ? 1 : 0);
        fflush(stderr);
    }
}

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
                                  const zdma_completion_info_t & completion) {
    if (!g_dma_trace_enabled) {
        return;
    }
    const unsigned long long  sequence = ++g_dma_trace_sequence;
    fpga_dma_trace_record_t & record   = g_dma_trace[(sequence - 1U) % FPGA_DMA_TRACE_DEPTH];
    record                             = {};
    record.valid                       = true;
    record.sequence                    = sequence;
    snprintf(record.tag, sizeof(record.tag), "%s", tag ? tag : "?");
    record.src_phys                   = src_phys;
    record.dst_phys                   = dst_phys;
    record.bytes                      = bytes;
    record.pre_status                 = pre_status;
    record.pre_isr                    = pre_isr;
    record.pre_ctrl2                  = pre_ctrl2;
    record.total_bytes_before_clear   = total_bytes_before_clear;
    record.pre_vpu_status             = pre_vpu_status;
    record.pre_vpu_progress           = pre_vpu_progress;
    record.post_status                = completion.status;
    record.post_isr                   = completion.isr;
    record.post_ctrl2                 = completion.ctrl2;
    record.total_bytes_after_transfer = total_bytes_after_transfer;
    record.post_vpu_status            = post_vpu_status;
    record.post_vpu_progress          = post_vpu_progress;
    record.elapsed_us                 = elapsed_us;
    record.polls                      = completion.polls;
    record.saw_enabled                = completion.saw_enabled;
}

static void fpga_dma_trace_dump(const char * reason,
                                const char * tensor_name,
                                int          layer_id,
                                uint32_t     tile_id,
                                const char * failed_transfer_tag) {
    if (!g_dma_trace_enabled || g_dma_trace_sequence == 0U) {
        return;
    }
    const unsigned long long first =
        g_dma_trace_sequence > FPGA_DMA_TRACE_DEPTH ? g_dma_trace_sequence - FPGA_DMA_TRACE_DEPTH + 1U : 1U;
    LOGE(
        "DMA_TRACE_BEGIN reason=%s tensor=%s layer=%d tile=%u failed_transfer=%s first_seq=%llu last_seq=%llu "
        "depth=%zu",
        reason ? reason : "?", tensor_name ? tensor_name : "?", layer_id, tile_id,
        failed_transfer_tag ? failed_transfer_tag : "none", first, g_dma_trace_sequence, FPGA_DMA_TRACE_DEPTH);
    for (unsigned long long sequence = first; sequence <= g_dma_trace_sequence; ++sequence) {
        const fpga_dma_trace_record_t & record = g_dma_trace[(sequence - 1U) % FPGA_DMA_TRACE_DEPTH];
        if (!record.valid || record.sequence != sequence) {
            continue;
        }
        LOGE(
            "DMA_TRACE seq=%llu tag=%s src=0x%llx dst=0x%llx bytes=%zu elapsed_us=%lld polls=%lld saw_enabled=%d "
            "total_before_clear=0x%08x pre_status=0x%08x pre_isr=0x%08x pre_ctrl2=0x%08x pre_vpu_status=0x%08x "
            "pre_vpu_progress=0x%08x post_status=0x%08x post_isr=0x%08x post_ctrl2=0x%08x total_after=0x%08x "
            "post_vpu_status=0x%08x post_vpu_progress=0x%08x dma_done=%d",
            record.sequence, record.tag, (unsigned long long) record.src_phys, (unsigned long long) record.dst_phys,
            record.bytes, record.elapsed_us, record.polls, record.saw_enabled ? 1 : 0, record.total_bytes_before_clear,
            record.pre_status, record.pre_isr, record.pre_ctrl2, record.pre_vpu_status, record.pre_vpu_progress,
            record.post_status, record.post_isr, record.post_ctrl2, record.total_bytes_after_transfer,
            record.post_vpu_status, record.post_vpu_progress, (record.post_isr & ZDMA_ISR_DMA_DONE) != 0U ? 1 : 0);
    }
    LOGE("DMA_TRACE_END reason=%s tensor=%s layer=%d tile=%u failed_transfer=%s", reason ? reason : "?",
         tensor_name ? tensor_name : "?", layer_id, tile_id, failed_transfer_tag ? failed_transfer_tag : "none");
}


// ============================================================================
// HOST SELF TESTS
// ============================================================================

static bool fpga_p2_scale_layout_case(int rows, int group_blocks) {
    const size_t entries = (size_t) rows * (size_t) group_blocks;
    const size_t words = (entries + (size_t) VPU_RESULT_PACK_LANES - 1U) / (size_t) VPU_RESULT_PACK_LANES;
    const size_t bytes = words * 16U;
    fpga_p2_scale_shape_t shape = {};
    if (!fpga_p2_checked_scale_shape(rows, group_blocks, (uint32_t) entries, (uint32_t) words, bytes, SPU_PARAM_BASE,
                                     &shape)) {
        return false;
    }

    static constexpr size_t guard_words = 4U;
    static constexpr uint32_t sentinel = 0xA55AA55AU;
    std::vector<uint32_t> words_with_guards(guard_words + shape.words * (size_t) VPU_RESULT_PACK_LANES + guard_words,
                                            sentinel);
    uint32_t * const packed = words_with_guards.data() + guard_words;
    for (size_t linear = 0; linear < shape.entries; ++linear) {
        const uint16_t activation_scale = (uint16_t) (0x1000U | ((uint32_t) linear & 0x0FFFU));
        const uint16_t weight_scale     = (uint16_t) (0xA000U | ((uint32_t) linear & 0x0FFFU));
        const size_t word                = linear / (size_t) VPU_RESULT_PACK_LANES;
        const size_t lane                = linear % (size_t) VPU_RESULT_PACK_LANES;
        packed[word * (size_t) VPU_RESULT_PACK_LANES + lane] =
            fpga_p2_pack_scale_entry(activation_scale, weight_scale);
    }
    for (size_t linear = shape.entries; linear < shape.words * (size_t) VPU_RESULT_PACK_LANES; ++linear) {
        packed[linear] = 0U;
    }

    for (size_t i = 0; i < guard_words; ++i) {
        if (words_with_guards[i] != sentinel ||
            words_with_guards[guard_words + shape.words * (size_t) VPU_RESULT_PACK_LANES + i] != sentinel) {
            return false;
        }
    }
    for (size_t linear = 0; linear < shape.entries; ++linear) {
        const uint16_t activation_scale = (uint16_t) (0x1000U | ((uint32_t) linear & 0x0FFFU));
        const uint16_t weight_scale     = (uint16_t) (0xA000U | ((uint32_t) linear & 0x0FFFU));
        const uint32_t expected          = fpga_p2_pack_scale_entry(activation_scale, weight_scale);
        const size_t word                = linear / (size_t) VPU_RESULT_PACK_LANES;
        const size_t lane                = linear % (size_t) VPU_RESULT_PACK_LANES;
        const uint32_t actual            = packed[word * (size_t) VPU_RESULT_PACK_LANES + lane];
        if (actual != expected || (uint16_t) actual != activation_scale ||
            (uint16_t) (actual >> 16) != weight_scale) {
            return false;
        }
    }
    for (size_t linear = shape.entries; linear < shape.words * (size_t) VPU_RESULT_PACK_LANES; ++linear) {
        if (packed[linear] != 0U) {
            return false;
        }
    }
    return true;
}

static bool fpga_p2_scale_layout_self_test(void) {
    // Modulo coverage, odd/even rows, a nontrivial multi-row shape, and the
    // deployed maximum P2 tile all exercise the same host-only layout path.
    static constexpr int cases[][2] = {
        {1, 4}, {1, 1}, {1, 2}, {1, 3}, {3, 5}, {2, 6}, {7, 5}, {256, 64},
    };
    for (const auto & test_case : cases) {
        if (!fpga_p2_scale_layout_case(test_case[0], test_case[1])) {
            return false;
        }
    }

    uint32_t sentinels[8];
    for (uint32_t & sentinel : sentinels) {
        sentinel = 0xC33CC33CU;
    }
    fpga_p2_scale_shape_t rejected = {17U, 19U, 23U, 29U};
    const bool rejected_shape =
        !fpga_p2_checked_scale_shape(0, 4, 0U, 0U, 0U, SPU_PARAM_BASE, &rejected) &&
        !fpga_p2_checked_scale_shape(1, 3, 3U, 1U, 32U, SPU_PARAM_BASE, &rejected) &&
        !fpga_p2_checked_scale_shape(1, 4, 4U, 2U, 16U, SPU_PARAM_BASE, &rejected) &&
        !fpga_p2_checked_scale_shape(1, 4, 3U, 1U, 16U, SPU_PARAM_BASE, &rejected) &&
        !fpga_p2_checked_scale_shape(1, 4, 4U, 1U, 16U, SPU_PARAM_BASE + 4U, &rejected);
    for (uint32_t sentinel : sentinels) {
        if (sentinel != 0xC33CC33CU) {
            return false;
        }
    }
    return rejected_shape && rejected.entries == 17U && rejected.words == 19U && rejected.bytes == 23U &&
           rejected.tail_entries == 29U;
}

static bool fpga_i8x16_le_pack_self_test(void) {
    static const int8_t lanes[VPU_NUM_LANES] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    static constexpr uint32_t expected_words[4] = {
        0x03020100U,
        0x07060504U,
        0x0b0a0908U,
        0x0f0e0d0cU,
    };
    for (size_t word = 0; word < 4U; ++word) {
        const uint32_t actual = ddr_pack_i8x4_le(lanes + word * 4U);
        if (actual != expected_words[word]) {
            LOGE("P2_LAYOUT_SELFTEST_FAIL reason=i8x16_le_pack word=%zu actual=0x%08x expected=0x%08x", word,
                 actual, expected_words[word]);
            return false;
        }
    }
    return true;
}

static bool fpga_weight_layout_host_self_test(void) {
    if (!fpga_i8x16_le_pack_self_test()) {
        return false;
    }
    static const int k_rows[]  = {1, 2, 3, 255, 256};
    static const int k_beats[] = {1, 2, 64, 128};
    for (const int rows : k_rows) {
        for (const int group_beats : k_beats) {
            size_t payload_words = 0;
            size_t payload_bytes = 0;
            if (!fpga_weight_layout_payload_words(rows, group_beats, &payload_words) ||
                !fpga_weight_layout_payload_bytes(rows, group_beats, &payload_bytes) ||
                payload_words > std::numeric_limits<size_t>::max() / 16U ||
                payload_bytes != payload_words * 16U) {
                LOGE("P2_LAYOUT_SELFTEST_FAIL reason=payload_size rows=%d group_beats=%d", rows, group_beats);
                return false;
            }
            std::vector<uint8_t> seen(payload_words, 0U);
            for (int row = 0; row < rows; ++row) {
                for (int beat = 0; beat < group_beats; ++beat) {
                    size_t index = 0;
                    const size_t expected = (((size_t) row >> 1U) * (size_t) group_beats + (size_t) beat) * 2U +
                                            ((size_t) row & 1U);
                    if (!fpga_weight_layout_word_index(rows, group_beats, row, beat, &index) || index != expected ||
                        index >= payload_words || seen[index] != 0U) {
                        LOGE("P2_LAYOUT_SELFTEST_FAIL reason=logical_mapping rows=%d group_beats=%d row=%d beat=%d index=%zu expected=%zu",
                             rows, group_beats, row, beat, index, expected);
                        return false;
                    }
                    seen[index] = 1U;
                }
            }
            if ((rows & 1) != 0) {
                for (int beat = 0; beat < group_beats; ++beat) {
                    size_t index = 0;
                    if (!fpga_weight_layout_word_index(rows, group_beats, rows, beat, &index) || index >= payload_words ||
                        seen[index] != 0U) {
                        LOGE("P2_LAYOUT_SELFTEST_FAIL reason=padded_companion rows=%d group_beats=%d beat=%d index=%zu",
                             rows, group_beats, beat, index);
                        return false;
                    }
                    seen[index] = 2U;
                }
            }
            for (size_t index = 0; index < payload_words; ++index) {
                if (seen[index] == 0U) {
                    LOGE("P2_LAYOUT_SELFTEST_FAIL reason=hole rows=%d group_beats=%d index=%zu words=%zu", rows,
                         group_beats, index, payload_words);
                    return false;
                }
            }

            // Compare the pre-V76 logical-row producer with V76's
            // pair-major producer without touching DDR/MMIO.  Each logical
            // source word is distinct, so this proves the exact byte layout
            // as well as the final odd-row zero companion.
            const size_t logical_words = (size_t) rows * (size_t) group_beats;
            if (logical_words > std::numeric_limits<size_t>::max() / 16U) {
                LOGE("P2_LAYOUT_SELFTEST_FAIL reason=logical_source_overflow rows=%d group_beats=%d", rows,
                     group_beats);
                return false;
            }
            std::vector<uint8_t> logical_source(logical_words * 16U);
            std::vector<uint8_t> legacy_emission(payload_bytes, 0xA5U);
            std::vector<uint8_t> pair_major_emission(payload_bytes, 0xA5U);
            for (size_t word = 0; word < logical_words; ++word) {
                for (size_t lane = 0; lane < 16U; ++lane) {
                    logical_source[word * 16U + lane] = (uint8_t) ((word * 37U + lane * 13U + 1U) & 0xffU);
                }
            }
            for (int row = 0; row < rows; ++row) {
                for (int beat = 0; beat < group_beats; ++beat) {
                    size_t word = 0;
                    if (!fpga_weight_layout_word_index(rows, group_beats, row, beat, &word)) {
                        LOGE("P2_LAYOUT_SELFTEST_FAIL reason=legacy_emission_offset rows=%d group_beats=%d row=%d beat=%d",
                             rows, group_beats, row, beat);
                        return false;
                    }
                    memcpy(legacy_emission.data() + word * 16U,
                           logical_source.data() + ((size_t) row * (size_t) group_beats + (size_t) beat) * 16U,
                           16U);
                }
            }
            if ((rows & 1) != 0) {
                for (int beat = 0; beat < group_beats; ++beat) {
                    size_t word = 0;
                    if (!fpga_weight_layout_word_index(rows, group_beats, rows, beat, &word)) {
                        LOGE("P2_LAYOUT_SELFTEST_FAIL reason=legacy_zero_companion_offset rows=%d group_beats=%d beat=%d",
                             rows, group_beats, beat);
                        return false;
                    }
                    memset(legacy_emission.data() + word * 16U, 0, 16U);
                }
            }
            const size_t pair_count = ((size_t) rows + 1U) / 2U;
            for (size_t pair = 0; pair < pair_count; ++pair) {
                const size_t even_row = pair * 2U;
                const size_t odd_row  = even_row + 1U;
                for (int beat = 0; beat < group_beats; ++beat) {
                    const size_t logical_beat = (size_t) beat;
                    const size_t off = (pair * (size_t) group_beats + logical_beat) * 32U;
                    memcpy(pair_major_emission.data() + off,
                           logical_source.data() + (even_row * (size_t) group_beats + logical_beat) * 16U, 16U);
                    if (odd_row < (size_t) rows) {
                        memcpy(pair_major_emission.data() + off + 16U,
                               logical_source.data() + (odd_row * (size_t) group_beats + logical_beat) * 16U,
                               16U);
                    } else {
                        memset(pair_major_emission.data() + off + 16U, 0, 16U);
                    }
                }
            }
            if (legacy_emission != pair_major_emission) {
                LOGE("P2_LAYOUT_SELFTEST_FAIL reason=pair_major_byte_mismatch rows=%d group_beats=%d", rows,
                     group_beats);
                return false;
            }
        }
    }
    LOGI(
        "P2 LAYOUT SELFTEST PASS\n"
        "  Bố trí dữ liệu = xen kẽ theo cặp có đệm | Số trường hợp = 20\n"
        "  Các word i8x16 little endian = 03020100,07060504,0b0a0908,0f0e0d0c\n"
        "  Quy tắc xuất dữ liệu = hàng logic theo pair major, đệm zero cho hàng lẻ\n"
        "  Công thức ánh xạ = index=((row>>1)*group_beats+beat)*2+(row&1)");
    return true;
}

static bool fpga_p2_cumulative_tile_limit_host_self_test(void) {
    static constexpr int matrix_tiles[] = {52, 4, 52, 4, 52, 4, 52, 4, 40};
    static constexpr int test_limit = 256;
    long long cumulative = 0;
    bool boundary_reached = false;
    int completed_matrices = 0;
    bool continuation_52_seen = false;
    bool terminal_256_seen = false;
    bool admission_of_257_prevented = false;

    for (int matrix_tile_count : matrix_tiles) {
        if (boundary_reached) {
            LOGE("P2_TILE_LIMIT_SELFTEST_FAIL reason=matrix_admitted_after_boundary cumulative=%lld limit=%d",
                 cumulative, test_limit);
            return false;
        }
        for (int tile = 0; tile < matrix_tile_count; ++tile) {
            if (fpga_p2_cumulative_tile_limit_reached(cumulative, test_limit)) {
                // Do not increment the simulated counter to 257.  This is
                // the same pre-admission boundary that protects real tiles.
                admission_of_257_prevented = true;
                break;
            }
            ++cumulative;
            boundary_reached = fpga_p2_cumulative_tile_limit_reached(cumulative, test_limit);
            if (!fpga_p2_cumulative_tile_state_consistent(cumulative, test_limit, boundary_reached)) {
                LOGE("P2_TILE_LIMIT_SELFTEST_FAIL reason=incorrect_boundary cumulative=%lld limit=%d reached=%d",
                     cumulative, test_limit, boundary_reached ? 1 : 0);
                return false;
            }
            if (cumulative == 52) {
                continuation_52_seen = !boundary_reached;
            }
            if (cumulative == test_limit) {
                terminal_256_seen = boundary_reached;
            }
        }
        ++completed_matrices;
    }
    if (!continuation_52_seen || !terminal_256_seen || !admission_of_257_prevented || !boundary_reached ||
        cumulative != test_limit || completed_matrices != 9 ||
        !fpga_p2_cumulative_tile_limit_reached(cumulative, test_limit) ||
        fpga_p2_cumulative_tile_limit_reached(test_limit - 1LL, test_limit) ||
        !fpga_p2_cumulative_tile_state_consistent(52, test_limit, false) ||
        !fpga_p2_cumulative_tile_state_consistent(test_limit, test_limit, true) ||
        fpga_p2_cumulative_tile_state_consistent(52, test_limit, true) ||
        fpga_p2_cumulative_tile_state_consistent(test_limit, test_limit, false)) {
        LOGE("P2_TILE_LIMIT_SELFTEST_FAIL reason=final_state cumulative=%lld limit=%d matrices=%d reached=%d",
             cumulative, test_limit, completed_matrices, boundary_reached ? 1 : 0);
        return false;
    }
    LOGINIT(
        "P2_TILE_LIMIT_SELFTEST_PASS scope=cumulative_across_matrices limit=%d matrices=%d continuation=52/256 "
        "terminal=256/256 inconsistent_states=rejected simulated_257=not_admitted exact_stop=%lld",
        test_limit, completed_matrices, cumulative);
    return true;
}




// ============================================================================
// SOURCE, CONTRACT AND QUALIFICATION DIAGNOSTICS
// ============================================================================

static bool fpga_contract_validate_weight_scales(const struct ggml_tensor * src0,
                                                 const void *               weight_data_base,
                                                 const char *               tensor_name,
                                                 int                        layer_id) {
    const int64_t k             = src0->ne[0];
    const int64_t n             = src0->ne[1];
    const int64_t nb            = k / VPU_QK8_0;
    long long     zero_scales   = 0;
    float         min_abs_scale = INFINITY;
    float         max_abs_scale = 0.0f;

    for (int64_t row = 0; row < n; ++row) {
        for (int64_t block = 0; block < nb; ++block) {
            const block_q8_0_t * const snapshot = weight_block_from_base(src0, weight_data_base, row, block);
            const float                scale    = fp16_to_fp32(snapshot->d);
            if (!std::isfinite(scale)) {
                const block_q8_0_t * const live                  = weight_block(src0, row, block);
                const bool                 live_matches_snapshot = memcmp(live, snapshot, sizeof(*snapshot)) == 0;
                block_q8_0_t               read_a                = {};
                block_q8_0_t               read_b                = {};
                memcpy(&read_a, live, sizeof(read_a));
                std::atomic_thread_fence(std::memory_order_seq_cst);
                memcpy(&read_b, live, sizeof(read_b));
                const bool   source_reads_stable = memcmp(&read_a, &read_b, sizeof(read_a)) == 0;
                const bool   upstream_row_valid  = ggml_validate_row_data(src0->type, src0->data, ggml_nbytes(src0));
                const size_t byte_offset = (size_t) row * (size_t) src0->nb[1] + (size_t) block * (size_t) src0->nb[0];
                LOGE(
                    "CONTRACT_WEIGHT_SCALE_NONFINITE tensor=%s layer=%d row=%lld block=%lld byte_offset=%zu "
                    "d_bits=0x%04x scale=%.9g live_d_bits=0x%04x live_scale=%.9g live_matches_snapshot=%d "
                    "source_reads_stable=%d upstream_q8_validate=%s src0_type=%d src0_nb=[%lld,%lld,%lld,%lld] "
                    "snapshot_bytes=%zu raw_bytes=[%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x] "
                    "qs_first8=[%d,%d,%d,%d,%d,%d,%d,%d]; refusing VPU launch because finite raw dots cannot yield a "
                    "finite F32 result",
                    tensor_name ? tensor_name : "?", layer_id, (long long) row, (long long) block, byte_offset,
                    (unsigned) snapshot->d, scale, (unsigned) live->d, fp16_to_fp32(live->d),
                    live_matches_snapshot ? 1 : 0, source_reads_stable ? 1 : 0, upstream_row_valid ? "pass" : "fail",
                    (int) src0->type, (long long) src0->nb[0], (long long) src0->nb[1], (long long) src0->nb[2],
                    (long long) src0->nb[3], ggml_nbytes(src0), ((const uint8_t *) snapshot)[0],
                    ((const uint8_t *) snapshot)[1], ((const uint8_t *) snapshot)[2], ((const uint8_t *) snapshot)[3],
                    ((const uint8_t *) snapshot)[4], ((const uint8_t *) snapshot)[5], ((const uint8_t *) snapshot)[6],
                    ((const uint8_t *) snapshot)[7], ((const uint8_t *) snapshot)[8], ((const uint8_t *) snapshot)[9],
                    (int) snapshot->qs[0], (int) snapshot->qs[1], (int) snapshot->qs[2], (int) snapshot->qs[3],
                    (int) snapshot->qs[4], (int) snapshot->qs[5], (int) snapshot->qs[6], (int) snapshot->qs[7]);
                fpga_log_source_file_provenance(live, sizeof(*live));
                return false;
            }
            const float abs_scale = std::fabs(scale);
            min_abs_scale         = std::min(min_abs_scale, abs_scale);
            max_abs_scale         = std::max(max_abs_scale, abs_scale);
            if (scale == 0.0f) {
                zero_scales++;
            }
        }
    }

    if (!std::isfinite(min_abs_scale)) {
        min_abs_scale = 0.0f;
    }
    LOGI(
        "CONTRACT_WEIGHT_SCALE_AUDIT tensor=%s layer=%d blocks=%lld zero_scales=%lld min_abs=%.9g max_abs=%.9g "
        "snapshot_bytes=%zu result=pass",
        tensor_name ? tensor_name : "?", layer_id, (long long) (n * nb), zero_scales, min_abs_scale, max_abs_scale,
        ggml_nbytes(src0));
    return true;
}


static void fpga_contract_log_q8_nonfinite_provenance(const struct ggml_tensor * src0,
                                                      const block_q8_0_t *       weight,
                                                      const block_q8_0_t *       act,
                                                      int64_t                    row,
                                                      int64_t                    col,
                                                      const char *               tensor_name,
                                                      int                        layer_id,
                                                      float                      kernel_reference) {
    const int64_t nb                     = src0->ne[0] / VPU_QK8_0;
    float         scalar_reference       = 0.0f;
    int64_t       first_bad_block        = -1;
    int32_t       first_bad_raw          = 0;
    float         first_bad_act_scale    = 0.0f;
    float         first_bad_weight_scale = 0.0f;
    float         first_bad_term         = 0.0f;
    const char *  first_bad_kind         = "scalar_accumulator";

    for (int64_t block = 0; block < nb; ++block) {
        const float   act_scale    = fp16_to_fp32(act[block].d);
        const float   weight_scale = fp16_to_fp32(weight[block].d);
        const int32_t raw          = q8_0_raw_dot(act[block].qs, weight[block].qs);
        const float   term         = (float) raw * act_scale * weight_scale;
        if (first_bad_block < 0 && (!std::isfinite(act_scale) || !std::isfinite(weight_scale) || !std::isfinite(term) ||
                                    !std::isfinite(scalar_reference + term))) {
            first_bad_block        = block;
            first_bad_raw          = raw;
            first_bad_act_scale    = act_scale;
            first_bad_weight_scale = weight_scale;
            first_bad_term         = term;
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

    LOGE(
        "CONTRACT_Q8_NONFINITE_PROVENANCE tensor=%s layer=%d row=%lld col=%lld kernel_reference=%.9g "
        "scalar_reference=%.9g first_bad_kind=%s first_bad_block=%lld raw=%d act_d_bits=0x%04x act_scale=%.9g "
        "weight_d_bits=0x%04x weight_scale=%.9g term=%.9g",
        tensor_name ? tensor_name : "?", layer_id, (long long) row, (long long) col, kernel_reference, scalar_reference,
        first_bad_kind, (long long) first_bad_block, first_bad_raw,
        first_bad_block >= 0 ? (unsigned) act[first_bad_block].d : 0U, first_bad_act_scale,
        first_bad_block >= 0 ? (unsigned) weight[first_bad_block].d : 0U, first_bad_weight_scale, first_bad_term);
}









static bool fpga_p2_complete_tile_contract_boundary(const fpga_tile_job_t & job,
                                                    const char *            tensor_name,
                                                    int                     layer_id) {
    if (g_pl_scale_contract_check_limit <= 0 ||
        !fpga_p2_cumulative_tile_limit_reached(g_p2_tile_q16_checks, g_p2_tile_limit)) {
        return true;
    }

    p2_trace_first_tile(job, "TILE_BOUNDARY", "before_free_readback");
    // Do not merely infer retirement from VPU/SPU completion.  A second
    // descriptor must never be admitted until ZDMA has dropped EN and the
    // stream controller has returned to its quiescent state.
    if (!zdma_wait_channel_disabled("p2_tile_boundary", "after_tile")) {
        LOGE(
            "P2_TILE_BOUNDARY_FAIL tensor=%s layer=%d job=%u tile=%u reason=zdma_channel_still_enabled; refusing "
            "another tile",
            tensor_name ? tensor_name : "?", layer_id, job.job_id, job.tile_id);
        return false;
    }
    if (!wait_spu_stream_quiescent("P2 tile boundary", false)) {
        LOGE(
            "P2_TILE_BOUNDARY_FAIL tensor=%s layer=%d job=%u tile=%u reason=spu_stream_not_quiescent; refusing another "
            "tile",
            tensor_name ? tensor_name : "?", layer_id, job.job_id, job.tile_id);
        return false;
    }
    mmio_fence();
    const uint32_t zdma_ctrl2    = g_dma->ZDMA_CH_CTRL2;
    const uint32_t slot_state    = vpu_rd32(REG_SLOT_STATE);
    const uint32_t desc_flags    = vpu_rd32(REG_DESC_FLAGS);
    const uint32_t bank          = vpu_rd32(REG_BANK);
    const uint32_t bank_stat     = vpu_rd32(REG_BANK_STAT);
    const uint32_t active_job    = vpu_rd32(REG_ACTIVE_JOB);
    const uint32_t done_job      = vpu_rd32(REG_DONE_JOB);
    const uint32_t stream_status = vpu_rd32(REG_SPU_STREAM_STATUS);
    const bool     descriptor_released =
        slot_state == fpga_slot_state_word(job.bank, FPGA_SLOT_FREE, FPGA_SLOT_FREE) && desc_flags == 0U;
    if (!descriptor_released) {
        LOGE(
            "P2_TILE_BOUNDARY_FAIL tensor=%s layer=%d job=%u tile=%u bank=%d slot_state=0x%08x desc_flags=0x%08x "
            "expected_slot_state=0x%08x expected_desc_flags=0; refusing another tile",
            tensor_name ? tensor_name : "?", layer_id, job.job_id, job.tile_id, job.bank, slot_state, desc_flags,
            fpga_slot_state_word(job.bank, FPGA_SLOT_FREE, FPGA_SLOT_FREE));
        return false;
    }

    g_p2_tile_contract_boundary_reached = true;
    FILE * fp                           = fpga_log_fp();
    fprintf(fp,
            "[INFO] P2_TILE_BOUNDARY status=pass tensor=%s layer=%d job=%u tile=%u bank=%d tile_limit=%d "
            "p2_tile_q16_checks=%lld p2_matrix_contract_checks=%lld matrix_value_contract=not_attempted "
            "zdma_ctrl2=0x%08x stream_status=0x%08x slot_state=0x%08x desc_flags=0x%08x reg_bank=0x%08x "
            "bank_stat=0x%08x active_job=0x%08x done_job=0x%08x action=cpu_shadow_current_matmul_then_cpu_native\n",
            tensor_name ? tensor_name : "?", layer_id, job.job_id, job.tile_id, job.bank, g_p2_tile_limit,
            g_p2_tile_q16_checks, g_p2_matrix_contract_checks, zdma_ctrl2, stream_status, slot_state, desc_flags, bank,
            bank_stat, active_job, done_job);
    fflush(fp);
    fprintf(stderr,
            "[P2_TILE_BOUNDARY] status=pass tensor=%s layer=%d job=%u tile=%u bank=%d tile_limit=%d "
            "p2_tile_q16_checks=%lld p2_matrix_contract_checks=%lld matrix_value_contract=not_attempted "
            "zdma_ctrl2=0x%08x stream_status=0x%08x slot_state=0x%08x desc_flags=0x%08x reg_bank=0x%08x "
            "bank_stat=0x%08x active_job=0x%08x done_job=0x%08x action=cpu_shadow_current_matmul_then_cpu_native\n",
            tensor_name ? tensor_name : "?", layer_id, job.job_id, job.tile_id, job.bank, g_p2_tile_limit,
            g_p2_tile_q16_checks, g_p2_matrix_contract_checks, zdma_ctrl2, stream_status, slot_state, desc_flags, bank,
            bank_stat, active_job, done_job);
    fflush(stderr);
    p2_trace_first_tile(job, "TILE_BOUNDARY", "after_free_readback");
    return true;
}

static bool fpga_spu_q16_contribution(int32_t   raw,
                                      uint16_t  act_scale,
                                      uint16_t  weight_scale,
                                      int64_t * contribution_q16) {
    if (!contribution_q16 || (act_scale & 0x8000U) != 0U || (weight_scale & 0x8000U) != 0U ||
        (act_scale & 0x7c00U) == 0x7c00U || (weight_scale & 0x7c00U) == 0x7c00U) {
        return false;
    }
    const auto fp16_to_q0_32 = [](uint16_t h) -> uint64_t {
        const uint64_t exp  = (h >> 10) & 0x1fU;
        const uint64_t frac = h & 0x03ffU;
        return exp == 0U ? (frac << 8) : ((0x400U | frac) << (exp + 7U));
    };
    const uint64_t          act_q32      = fp16_to_q0_32(act_scale);
    const uint64_t       weight_q32   = fp16_to_q0_32(weight_scale);
    const fpga_uint128_t product_full = (fpga_uint128_t) act_q32 * (fpga_uint128_t) weight_q32;
    const uint64_t       product_q32  = (product_full >> 96U) != 0U ? UINT64_MAX : (uint64_t) (product_full >> 32U);
    const fpga_int128_t  contribution = ((fpga_int128_t) raw * (fpga_int128_t) product_q32) >> 16U;
    *contribution_q16                 = (int64_t) contribution;
    return true;
}

// Called only after a failed serialized P2 check, before retirement or reuse.
// Read DDR staging only: do not replay DMA, read IP memories, or repair results.
static void fpga_p2_audit_failed_row(const fpga_tile_job_t & job, const void * weight_data_base,
                                     const char * tensor_name, int row, int64_t actual, uint64_t expected_bits) {
    if (job.weight_bank_reused) {
        fpga_log_q16_audit("status=unavailable reason=inputs_reused job=%u", job.job_id);
        return;
    }
    const uint32_t result_off = SPU_OUT_BASE + (uint32_t) row * 16U;
    const volatile uint32_t * result = ddr_checked_u32_ptr(result_off, 16U);
    const uint32_t r0 = result[0], r1 = result[1], r2 = result[2], r3 = result[3];
    const uint64_t scalar_bits = ((uint64_t) r0 >> 16U) | ((uint64_t) r1 << 16U) |
                                 ((uint64_t) (r2 & 0xffffU) << 48U);
    fpga_log_q16_audit(
        "status=begin tensor=%s job=%u tile=%u row=%d row0=%lld col=%lld k_block0=%lld groups=%d "
        "weight_src_off=0x%08x resident=%d actual=%lld expected_bits=0x%016llx "
        "result_words=%08x,%08x,%08x,%08x scalar_bits=0x%016llx scalar_matches_actual=%d",
        tensor_name ? tensor_name : "?", job.job_id, job.tile_id, row, (long long) job.row0,
        (long long) job.col, (long long) job.k_block0, job.group_blocks, job.weight_src_off,
        job.p2_residency_hit ? 1 : 0, (long long) actual, (unsigned long long) expected_bits,
        r0, r1, r2, r3, (unsigned long long) scalar_bits, scalar_bits == (uint64_t) actual ? 1 : 0);
    uint64_t staged_sum = 0;
    unsigned act_bad = 0, weight_bad = 0, scale_bad = 0;
    bool staged_valid = true;
    for (int gb = 0; gb < job.group_blocks; ++gb) {
        const block_q8_0_t & act = job.act_group[gb];
        const block_q8_0_t * weight =
            weight_block_from_base(job.src0, weight_data_base, job.row0 + row, job.k_block0 + gb);
        int8_t staged_act[VPU_QK8_0], staged_weight[VPU_QK8_0];
        for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
            uint32_t weight_off = 0;
            if (!fpga_weight_layout_word_offset(job.weight_src_off, job.rows, job.group_beats,
                                                row, gb * VPU_BLOCK_BEATS + beat, &weight_off)) {
                fpga_log_q16_audit("status=unavailable reason=weight_offset job=%u block=%d", job.job_id, gb);
                return;
            }
            const uint32_t act_off = ACT_BASE + (uint32_t) (gb * VPU_BLOCK_BEATS + beat) * 16U;
            const volatile uint32_t * a = ddr_checked_u32_ptr(act_off, 16U);
            const volatile uint32_t * w = ddr_checked_u32_ptr(weight_off, 16U);
            uint32_t aw[4], ww[4], ref_a[4] = {}, ref_w[4] = {};
            for (int word = 0; word < 4; ++word) {
                aw[word] = a[word];
                ww[word] = w[word];
                for (int byte = 0; byte < 4; ++byte) {
                    const int lane = beat * VPU_NUM_LANES + word * 4 + byte;
                    const uint8_t ab = (uint8_t) (aw[word] >> (byte * 8));
                    const uint8_t wb = (uint8_t) (ww[word] >> (byte * 8));
                    memcpy(&staged_act[lane], &ab, 1);
                    memcpy(&staged_weight[lane], &wb, 1);
                    act_bad += ab != (uint8_t) act.qs[lane];
                    weight_bad += wb != (uint8_t) weight->qs[lane];
                    ref_a[word] |= (uint32_t) (uint8_t) act.qs[lane] << (byte * 8);
                    ref_w[word] |= (uint32_t) (uint8_t) weight->qs[lane] << (byte * 8);
                }
            }
            fpga_log_q16_audit(
                "job=%u block=%d beat=%d act_off=0x%08x weight_off=0x%08x "
                "source_act=%08x,%08x,%08x,%08x staged_act=%08x,%08x,%08x,%08x "
                "source_weight=%08x,%08x,%08x,%08x staged_weight=%08x,%08x,%08x,%08x",
                job.job_id, gb, beat, act_off, weight_off,
                ref_a[0], ref_a[1], ref_a[2], ref_a[3], aw[0], aw[1], aw[2], aw[3],
                ref_w[0], ref_w[1], ref_w[2], ref_w[3], ww[0], ww[1], ww[2], ww[3]);
        }
        const uint32_t scale_off = SPU_PARAM_BASE + ((uint32_t) row * job.group_blocks + gb) * 4U;
        const uint32_t packed_scale = *ddr_checked_u32_ptr(scale_off, 4U);
        const uint32_t source_scale = fpga_p2_pack_scale_entry(act.d, weight->d);
        scale_bad += packed_scale != source_scale;
        const int32_t source_raw = q8_0_raw_dot(act.qs, weight->qs);
        const int32_t staged_raw = q8_0_raw_dot(staged_act, staged_weight);
        int64_t source_q16 = 0, staged_q16 = 0;
        const bool source_ok = fpga_spu_q16_contribution(source_raw, act.d, weight->d, &source_q16);
        const bool staged_ok = fpga_spu_q16_contribution(staged_raw, (uint16_t) packed_scale,
                                                        (uint16_t) (packed_scale >> 16U), &staged_q16);
        staged_valid = staged_valid && staged_ok;
        staged_sum += (uint64_t) staged_q16;
        fpga_log_q16_audit(
            "job=%u block=%d scale_off=0x%08x source_scale=%08x staged_scale=%08x source_raw=%d "
            "staged_raw=%d source_q16=%lld staged_q16=%lld source_valid=%d staged_valid=%d",
            job.job_id, gb, scale_off, source_scale, packed_scale, source_raw, staged_raw,
            (long long) source_q16, (long long) staged_q16, source_ok ? 1 : 0, staged_ok ? 1 : 0);
    }
    fpga_log_q16_audit(
        "status=end job=%u act_byte_mismatches=%u weight_byte_mismatches=%u scale_mismatches=%u "
        "staged_valid=%d staged_sum_bits=0x%016llx staged_matches_actual=%d "
        "scope=post_failure_ddr_snapshot ip_inputs_not_observed=1",
        job.job_id, act_bad, weight_bad, scale_bad, staged_valid ? 1 : 0,
        (unsigned long long) staged_sum, staged_valid && staged_sum == (uint64_t) actual ? 1 : 0);
}

static bool fpga_pl_scale_contract_verify_q16_tile(const fpga_tile_job_t & job,
                                                   const void *            weight_data_base,
                                                   const char *            tensor_name,
                                                   int                     layer_id) {
    long long checked = 0;
    fpga_p2_boundary_marker(
        "P2_Q16_VERIFY edge=entry tensor=%s layer=%d job=%u bank=%d tile=%u rows=%d group_blocks=%d",
        tensor_name ? tensor_name : "?", layer_id, job.job_id, job.bank, job.tile_id, job.rows, job.group_blocks);
    for (int row = 0; row < job.rows; ++row) {
        uint64_t expected_bits = 0U;
        for (int gb = 0; gb < job.group_blocks; ++gb) {
            const block_q8_0_t * const weight =
                weight_block_from_base(job.src0, weight_data_base, job.row0 + row, job.k_block0 + gb);
            const block_q8_0_t & act          = job.act_group[gb];
            int64_t              contribution = 0;
            if (!fpga_spu_q16_contribution(q8_0_raw_dot(act.qs, weight->qs), act.d, weight->d, &contribution)) {
                LOGE(
                    "SPU_SCALE_CONTRACT_Q16_BAD_SCALE tensor=%s layer=%d job=%u bank=%d tile=%u row=%d block=%d "
                    "act_d=0x%04x weight_d=0x%04x",
                    tensor_name ? tensor_name : "?", layer_id, job.job_id, job.bank, job.tile_id, row, gb, act.d,
                    weight->d);
                fpga_p2_boundary_marker(
                    "P2_Q16_VERIFY edge=complete status=fail reason=bad_scale job=%u bank=%d tile=%u row=%d block=%d",
                    job.job_id, job.bank, job.tile_id, row, gb);
                return false;
            }
            expected_bits += (uint64_t) contribution;
        }
        uint16_t      row_id    = 0xffffU;
        const int64_t actual    = ddr_read_spu_q16_row(SPU_OUT_BASE + (uint32_t) row * 16U, &row_id, row == 0);
        const bool    row_match = row_id == (uint16_t) row && (uint64_t) actual == expected_bits;
        if (row == 0) {
            fpga_p2_boundary_marker(
                "P2_Q16_VERIFY edge=after_row0_compare job=%u bank=%d tile=%u off=0x%08x got_row=%u got_q16=%lld "
                "expected_q16=%lld match=%d",
                job.job_id, job.bank, job.tile_id, SPU_OUT_BASE, (unsigned) row_id, (long long) actual,
                (long long) (int64_t) expected_bits, row_match ? 1 : 0);
        }
        if (!row_match) {
            LOGE(
                "SPU_SCALE_CONTRACT_Q16_FAIL tensor=%s layer=%d job=%u bank=%d tile=%u row=%d got_row=%u got_q16=%lld "
                "expected_q16=%lld",
                tensor_name ? tensor_name : "?", layer_id, job.job_id, job.bank, job.tile_id, row, (unsigned) row_id,
                (long long) actual, (long long) (int64_t) expected_bits);
            fpga_p2_audit_failed_row(job, weight_data_base, tensor_name, row, actual, expected_bits);
            fpga_p2_boundary_marker(
                "P2_Q16_VERIFY edge=complete status=fail reason=row_mismatch job=%u bank=%d tile=%u row=%d", job.job_id,
                job.bank, job.tile_id, row);
            return false;
        }
        checked++;
    }
    LOGI(
        "SPU_SCALE_CONTRACT_Q16_PASS tensor=%s layer=%d job=%u bank=%d tile=%u rows=%lld expected_raw=%u "
        "expected_out=%u",
        tensor_name ? tensor_name : "?", layer_id, job.job_id, job.bank, job.tile_id, checked,
        (unsigned) (job.rows * job.group_blocks), (unsigned) job.rows);
    fpga_p2_boundary_marker("P2_Q16_VERIFY edge=complete status=pass job=%u bank=%d tile=%u rows=%lld", job.job_id,
                            job.bank, job.tile_id, checked);
    return true;
}

static bool fpga_pl_scale_contract_check_output_values(const struct ggml_tensor *        src0,
                                                       const struct ggml_tensor *        dst,
                                                       const std::vector<block_q8_0_t> & act_blocks_all,
                                                       const void *                      weight_data_base,
                                                       const char *                      tensor_name,
                                                       int                               layer_id) {
    const int64_t k           = src0->ne[0];
    const int64_t n           = src0->ne[1];
    const int64_t m           = dst->ne[1];
    const int64_t nb          = k / VPU_QK8_0;
    const size_t  value_count = (size_t) n * (size_t) m;
    if (g_scratch.contract_actual.size() != value_count) {
        LOGE("SPU_SCALE_CONTRACT_VALUE_LAYOUT tensor=%s layer=%d actual_values=%zu expected_values=%zu",
             tensor_name ? tensor_name : "?", layer_id, g_scratch.contract_actual.size(), value_count);
        return false;
    }
    long long bad     = 0;
    double    max_abs = 0.0;
    double    max_rel = 0.0;
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < n; ++row) {
            float ref = 0.0f;
            ggml_vec_dot_q8_0_q8_0((int) k, &ref, 0, weight_block_from_base(src0, weight_data_base, row, 0), 0,
                                   &act_blocks_all[(size_t) col * (size_t) nb], 0, 1);
            const double actual   = (double) g_scratch.contract_actual[(size_t) col * (size_t) n + (size_t) row];
            const double expected = (double) ref;
            const double abs_err  = std::fabs(actual - expected);
            const double rel_err  = abs_err / (std::fabs(expected) + 1.0e-12);
            max_abs               = std::max(max_abs, abs_err);
            max_rel               = std::max(max_rel, rel_err);
            if (!std::isfinite(actual) || !std::isfinite(expected) || abs_err > P2_PL_SCALE_VALUE_ATOL) {
                if (bad < 4) {
                    LOGE(
                        "SPU_SCALE_CONTRACT_VALUE_FAIL tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g "
                        "abs=%.9g rel=%.9g",
                        tensor_name ? tensor_name : "?", layer_id, (long long) row, (long long) col, actual, expected,
                        abs_err, rel_err);
                }
                bad++;
            }
        }
    }
    if (bad != 0) {
        g_contract_value_mismatches += bad;
        LOGE(
            "SPU_SCALE_CONTRACT_VALUE_SUMMARY tensor=%s layer=%d checked=%lld bad=%lld max_abs=%.9g max_rel=%.9g "
            "p2_abs_atol=%.9g rel=informational action=abort_no_partial_dst",
            tensor_name ? tensor_name : "?", layer_id, (long long) value_count, bad, max_abs, max_rel,
            P2_PL_SCALE_VALUE_ATOL);
        return false;
    }
    g_contract_cpu_shadow_dst_values += (long long) value_count;
    LOGI(
        "SPU_SCALE_CONTRACT_VALUE_PASS tensor=%s layer=%d checked=%lld max_abs=%.9g max_rel=%.9g p2_abs_atol=%.9g "
        "rel=informational dst=native_cpu_shadow",
        tensor_name ? tensor_name : "?", layer_id, (long long) value_count, max_abs, max_rel, P2_PL_SCALE_VALUE_ATOL);
    return true;
}
