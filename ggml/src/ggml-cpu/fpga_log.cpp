#include "fpga_log.h"
#include "fpga_host.h"
#include "fpga_q8_layout.h"
#include "fpga_q8_pack.h"

#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
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
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

__extension__ typedef unsigned __int128 fpga_uint128_t;
__extension__ typedef __int128          fpga_int128_t;

namespace {

constexpr const char * FPGA_LOG_FILE = "/tmp/fpga_debug.log";

int g_flush_every   = 256;
int g_pending_lines = 0;

} // namespace

FILE * fpga_log_fp() {
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

void fpga_log_set_flush_every(int flush_every) {
    g_flush_every = std::max(1, flush_every);
}

void fpga_log_finish_line(FILE * fp, bool force_flush) {
    ++g_pending_lines;
    if (force_flush || g_flush_every <= 1 || g_pending_lines >= g_flush_every) {
        fflush(fp);
        g_pending_lines = 0;
    }
}

void fpga_log_vline(const char * tag, bool force_flush, const char * fmt, va_list ap) {
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[FPGA][%s] ", tag ? tag : "INFO");
    vfprintf(fp, fmt, ap);
    fputc('\n', fp);
    fpga_log_finish_line(fp, force_flush);
}

// ============================================================================
// LOGGING AND TERMINAL OUTPUT
// ============================================================================

static void fpga_p2_init_breadcrumb(const char * fmt, ...) {
    if (!g_p2_init_requested ||
        (!g_init_verbose && (!fmt || strstr(fmt, "phase=failure") == nullptr))) {
        return;
    }

    fprintf(stderr, "[FPGA][P2_INIT] version=%s ", FPGA_HOST_TRACE_VERSION);
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
    fprintf(fp, "[FPGA][INFO] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fflush(fp);

    fprintf(stderr, "[FPGA][P2_BOUNDARY] ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
}

static void fpga_p2_dma_breadcrumb(const char * fmt, ...) {
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[FPGA][INFO] P2_ACT_DMA_TRACE ");
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
        fprintf(stderr, "[FPGA][P2_ACT_DMA] ");
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

static void fpga_p1_preload_breadcrumb(bool force, const char * fmt, ...) {
    if (!force && (!g_p1_preload_trace_enabled || g_p1_preload_breadcrumbs >= FPGA_P1_PRELOAD_BREADCRUMB_LIMIT)) {
        return;
    }
    ++g_p1_preload_breadcrumbs;

    FILE * fp = fpga_log_fp();
    fprintf(fp, "[FPGA][P1_PRELOAD] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\n");
    fpga_log_finish_line(fp, force);
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
    fprintf(fp, "[FPGA][P2_RESIDENCY] event=%s ", event ? event : "?");
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
                                     uint32_t                          weight_tile_index,
                                     const fpga_weight_cache_entry_t * weight_cache,
                                     uint32_t                          tile_id,
                                     int                               bank,
                                     fpga_stage_totals_t *             totals) {
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
    const bool p3_split_scale     = g_p3_split_scale_active;
    const size_t scale_bytes      = (size_t) result_words * 16U;
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

    std::vector<int32_t> partial_storage;
    std::vector<float>   weight_scale_storage;
    partial_storage.swap(job.partial);
    weight_scale_storage.swap(job.weight_scales);
    job = {};
    partial_storage.clear();
    weight_scale_storage.clear();
    job.partial.swap(partial_storage);
    job.weight_scales.swap(weight_scale_storage);

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
    job.p3_split_scale   = p3_split_scale;
    job.spu_result_bytes = spu_result_bytes;
    job.result_bytes     = result_bytes;
    job.result_values    = result_values;
    job.result_words     = result_words;
    job.scale_words      = result_words;
    job.weight_src_off   = WEIGHT_BASE;
    job.p2_residency_slot = P2_WEIGHT_RESIDENCY_NO_SLOT;
    job.act_group        = act_group;
    job.src0             = src0;
    job.weight_cache     = weight_cache;

    if (p3_split_scale) {
        if (rows > P3_MAX_ROWS || group_blocks > P3_MAX_GROUP_BLOCKS || g_spu_word_capacity == 0U ||
            (g_spu_word_capacity & 1U) != 0U) {
            LOGE("P3 tile bounds rejected rows=%d blocks=%d spu_words=%u max_rows=%d max_blocks=%d action=no_p3_dma",
                 rows, group_blocks, g_spu_word_capacity, P3_MAX_ROWS, P3_MAX_GROUP_BLOCKS);
            return false;
        }
        const size_t entries = (size_t) rows * (size_t) group_blocks;
        const size_t weight_words = (entries + 7U) / 8U;
        const size_t act_words = ((size_t) group_blocks + 7U) / 8U;
        const size_t bank_words = (size_t) g_spu_word_capacity / 2U;
        const size_t bank_bytes = bank_words * 16U;
        if (entries == 0U || weight_words == 0U || act_words == 0U || weight_words > bank_words || act_words > bank_words ||
            bank_bytes > UINT32_MAX || weight_words > std::numeric_limits<size_t>::max() / 16U ||
            act_words > std::numeric_limits<size_t>::max() / 16U) {
            LOGE("P3 scale capacity rejected rows=%d blocks=%d entries=%zu weight_words=%zu act_words=%zu bank_words=%zu "
                 "action=no_p3_dma",
                 rows, group_blocks, entries, weight_words, act_words, bank_words);
            return false;
        }
        const size_t bank_offset = (size_t) (job.bank & 1) * bank_bytes;
        const size_t param_off64 = (size_t) SPU_PARAM_BASE + bank_offset;
        const size_t scratch_off64 = (size_t) SPU_SCRATCH_BASE + bank_offset;
        job.p3_weight_scale_bytes = weight_words * 16U;
        job.p3_activation_scale_bytes = act_words * 16U;
        if (param_off64 > UINT32_MAX || scratch_off64 > UINT32_MAX ||
            !range_fits((uint32_t) param_off64, job.p3_weight_scale_bytes, SPU_PARAM_BASE, SPU_PARAM_END) ||
            !range_fits((uint32_t) scratch_off64, job.p3_activation_scale_bytes, SPU_SCRATCH_BASE, SPU_SCRATCH_END) ||
            !ddr_range_fits((uint32_t) param_off64, job.p3_weight_scale_bytes) ||
            !ddr_range_fits((uint32_t) scratch_off64, job.p3_activation_scale_bytes) ||
            (uint64_t) DDR_BASE_PHYS + param_off64 + job.p3_weight_scale_bytes > DDR_END_EXCLUSIVE ||
            (uint64_t) DDR_BASE_PHYS + scratch_off64 + job.p3_activation_scale_bytes > DDR_END_EXCLUSIVE) {
            LOGE("P3 bank range rejected bank=%d param_off=0x%zx param_bytes=%zu scratch_off=0x%zx scratch_bytes=%zu "
                 "ddr=[0x%llx,0x%llx) action=no_p3_dma",
                 job.bank, param_off64, job.p3_weight_scale_bytes, scratch_off64, job.p3_activation_scale_bytes,
                 (unsigned long long) DDR_BASE_PHYS, (unsigned long long) DDR_END_EXCLUSIVE);
            return false;
        }
        job.p3_param_off = (uint32_t) param_off64;
        job.p3_scratch_off = (uint32_t) scratch_off64;
        job.scale_bytes = job.p3_weight_scale_bytes + job.p3_activation_scale_bytes;
    } else if (!range_fits(SPU_PARAM_BASE, scale_bytes, SPU_PARAM_BASE, SPU_PARAM_END) ||
               !ddr_range_fits(SPU_PARAM_BASE, scale_bytes)) {
        LOGE("P2 scale window rejected bytes=%zu action=no_dma", scale_bytes);
        return false;
    }

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
    if (weight_cache && weight_tile_index < weight_cache->tiles.size()) {
        const fpga_weight_tile_cache_t & tile = weight_cache->tiles[weight_tile_index];
        if (tile.row0 == row0 && tile.rows == rows && tile.k_block0 == k_block0 && tile.group_blocks == group_blocks &&
            tile.group_beats == group_beats && tile.bytes == weight_bytes) {
            job.weight_src_off   = tile.ddr_off;
            job.weight_cache_hit = true;
        }
    }
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

    const long long direct_weight_pack0 = !job.weight_cache_hit && !job.p2_residency_hit ? now_us() : 0;
    if (!job.weight_cache_hit && !job.p2_residency_hit) {
        size_t direct_payload_bytes = 0;
        const size_t pair_count       = ((size_t) rows + 1U) / 2U;
        const size_t group_beats_size = (size_t) group_beats;
        if (pair_count > (size_t) INT_MAX / 2U || group_beats_size > std::numeric_limits<size_t>::max() / pair_count ||
            pair_count * group_beats_size > std::numeric_limits<size_t>::max() / 32U ||
            row0 < 0 || row0 > INT64_MAX - (int64_t) rows || k_block0 < 0 || k_block0 > INT64_MAX - (int64_t) group_blocks ||
            !fpga_weight_layout_payload_bytes(rows, group_beats, &direct_payload_bytes) ||
            direct_payload_bytes != weight_bytes || direct_payload_bytes != job.weight_bytes ||
            direct_payload_bytes != pair_count * group_beats_size * 32U ||
            direct_payload_bytes > (size_t) UINT32_MAX - (size_t) WEIGHT_BASE ||
            !range_fits(WEIGHT_BASE, direct_payload_bytes, WEIGHT_BASE, WEIGHT_END) ||
            !ddr_range_fits(WEIGHT_BASE, direct_payload_bytes) ||
            direct_payload_bytes > UINT64_MAX - g_p2_residency_direct_weight_pack_bytes) {
            LOGE(
                "P2 direct WEIGHT pack precondition failed job=%u tile=%u rows=%d group_beats=%d payload=%zu "
                "weight_bytes=%zu job_weight_bytes=%zu pair_count=%zu ddr_size=0x%zx action=no_write_no_dma_no_start",
                job.job_id, job.tile_id, rows, group_beats, direct_payload_bytes, weight_bytes, job.weight_bytes,
                pair_count, g_ddr_map_size);
            return false;
        }

        volatile uint32_t * direct_weight_words = ddr_checked_u32_ptr(WEIGHT_BASE, direct_payload_bytes);
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
            const size_t main_pair_end = pair_count / 2U;
            const size_t helper_words = (pair_count - main_pair_end) * words_per_pair;
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
            const fpga_p2_pack_worker_task_t task = {
                src0, weight_data_base, row0, k_block0, rows, group_blocks, group_beats,
                main_pair_end, pair_count, direct_weight_words, helper_words, generation,
            };
            if (!fpga_p2_pack_worker_submit(task)) {
                fpga_fatal(
                    "P2 direct WEIGHT helper submit failed job=%u tile=%u generation=%llu "
                    "action=no_write_no_dma_no_start",
                    job.job_id, job.tile_id, (unsigned long long) generation);
            }

            const long long main_pack0 = now_us();
            size_t main_words = 0U;
            const bool main_ok = fpga_pack_direct_weight_pair_range(
                direct_weight_words, src0, weight_data_base, row0, k_block0, rows, group_blocks, group_beats,
                0U, main_pair_end, &main_words);
            const long long main_pack_us = now_us() - main_pack0;
            // The caller publishes its prefix before it waits for the helper
            // and performs the existing full-WEIGHT coherency sequence.
            mmio_fence();
            size_t helper_written_words = 0U;
            long long helper_service_us = 0;
            long long caller_wait_us = 0;
            const bool helper_ok = fpga_p2_pack_worker_wait(generation, &helper_written_words, &helper_service_us,
                                                              &caller_wait_us);
            if (!main_ok || !helper_ok || main_words != main_pair_end * words_per_pair ||
                helper_written_words != helper_words || main_words > expected_words ||
                helper_written_words > expected_words - main_words) {
                fpga_fatal(
                    "P2 direct WEIGHT parallel pack completion mismatch job=%u tile=%u generation=%llu main_ok=%d "
                    "helper_ok=%d main_words=%zu helper_words=%zu expected_main=%zu expected_helper=%zu "
                    "action=no_dma_no_start",
                    job.job_id, job.tile_id, (unsigned long long) generation, main_ok ? 1 : 0, helper_ok ? 1 : 0,
                    main_words, helper_written_words, main_pair_end * words_per_pair, helper_words);
            }
            written_words = main_words + helper_written_words;
            g_p2_pack_parallel_jobs++;
            g_p2_pack_parallel_bytes += (uint64_t) direct_payload_bytes;
            g_p2_pack_main_us += main_pack_us;
            g_p2_pack_helper_service_us += helper_service_us;
            g_p2_pack_caller_wait_us += caller_wait_us;
        } else {
            if (g_p2_pack_workers_requested == 2) {
                g_p2_pack_serial_threshold_skips++;
            }
            if (!fpga_pack_direct_weight_pair_range(direct_weight_words, src0, weight_data_base, row0, k_block0,
                                                     rows, group_blocks, group_beats, 0U, pair_count, &written_words)) {
                fpga_fatal(
                    "P2 direct WEIGHT serial pair-range pack failed job=%u tile=%u action=no_dma_no_start",
                    job.job_id, job.tile_id);
            }
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
        if (totals) {
            totals->prep_direct_weight_pack_us += direct_weight_pack_us;
        }
    } else if (job.p2_residency_hit) {
        g_p2_residency_avoided_cpu_pack_bytes += (long long) job.weight_bytes;
    }

    const long long scale_pack0 = now_us();
    if (job.p3_split_scale) {
        // Dense P3 words are eight exact GGML FP16 bit patterns.  Do not
        // convert through float: a source scale is data, not a value to
        // canonicalize or repair.  PARAM carries row-major weight scales;
        // SCRATCH carries one activation scale per Q8 block.
        ddr_zero_range32(job.p3_param_off, job.p3_weight_scale_bytes);
        ddr_zero_range32(job.p3_scratch_off, job.p3_activation_scale_bytes);
        const size_t weight_entries = (size_t) rows * (size_t) group_blocks;
        for (size_t word = 0; word < job.p3_weight_scale_bytes / 16U; ++word) {
            for (size_t pair = 0; pair < 4U; ++pair) {
                const size_t index0 = word * 8U + pair * 2U;
                uint16_t scale0 = 0U;
                uint16_t scale1 = 0U;
                if (index0 < weight_entries) {
                    const int row = (int) (index0 / (size_t) group_blocks);
                    const int gb = (int) (index0 % (size_t) group_blocks);
                    scale0 = (uint16_t) weight_block_from_base(src0, weight_data_base, row0 + row, k_block0 + gb)->d;
                }
                if (index0 + 1U < weight_entries) {
                    const int row = (int) ((index0 + 1U) / (size_t) group_blocks);
                    const int gb = (int) ((index0 + 1U) % (size_t) group_blocks);
                    scale1 = (uint16_t) weight_block_from_base(src0, weight_data_base, row0 + row, k_block0 + gb)->d;
                }
                ddr_write_u32(job.p3_param_off + (uint32_t) word * 16U + (uint32_t) pair * 4U,
                              fpga_p3_pack_fp16_pair(scale0, scale1));
            }
        }
        for (size_t word = 0; word < job.p3_activation_scale_bytes / 16U; ++word) {
            for (size_t pair = 0; pair < 4U; ++pair) {
                const size_t index0 = word * 8U + pair * 2U;
                const uint16_t scale0 = index0 < (size_t) group_blocks ? (uint16_t) act_group[index0].d : 0U;
                const uint16_t scale1 = index0 + 1U < (size_t) group_blocks ? (uint16_t) act_group[index0 + 1U].d : 0U;
                ddr_write_u32(job.p3_scratch_off + (uint32_t) word * 16U + (uint32_t) pair * 4U,
                              fpga_p3_pack_fp16_pair(scale0, scale1));
            }
        }
    } else {
        fpga_p2_scale_shape_t scale_shape = {};
        if (!fpga_p2_checked_scale_shape(rows, group_blocks, job.result_values, job.result_words, job.scale_bytes,
                                         SPU_PARAM_BASE, &scale_shape)) {
            LOGE(
                "P2 scale shape rejected job=%u tile=%u rows=%d group_blocks=%d result_values=%u result_words=%u "
                "scale_bytes=%zu spu_param=0x%08x action=no_write_no_dma_no_start",
                job.job_id, job.tile_id, rows, group_blocks, job.result_values, job.result_words, job.scale_bytes,
                SPU_PARAM_BASE);
            return false;
        }
        // Check the complete P2 PARAM range once before the first volatile
        // store.  Every live entry is written exactly once; only the 0..3
        // unused lanes in the final 128-bit word are cleared.
        volatile uint32_t * const scale_words = ddr_checked_u32_ptr(SPU_PARAM_BASE, job.scale_bytes);
        for (int row = 0; row < rows; ++row) {
            for (int gb = 0; gb < group_blocks; ++gb) {
                const size_t         linear = (size_t) row * (size_t) group_blocks + (size_t) gb;
                const size_t         word   = linear / (size_t) VPU_RESULT_PACK_LANES;
                const size_t         lane   = linear % (size_t) VPU_RESULT_PACK_LANES;
                uint16_t weight_d = 0U;
                if (job.p2_residency_hit) {
                    if (!fpga_p2_resident_scale_bits(job.p2_residency_slot,
                                                     (size_t) row * (size_t) group_blocks + (size_t) gb, &weight_d)) {
                        fpga_p2_residency_poison_slot(job.p2_residency_slot, "param_host_scale_read_invalid");
                        LOGE("P2_RESIDENCY_HOST_METADATA_FAIL job=%u tile=%u slot=%u action=no_dma_no_start", job.job_id,
                             job.tile_id, job.p2_residency_slot);
                        return false;
                    }
                } else {
                    const block_q8_0_t * wb =
                        weight_block_from_base(src0, weight_data_base, row0 + row, k_block0 + gb);
                    weight_d = (uint16_t) wb->d;
                }
                const uint32_t packed_scale = fpga_p2_pack_scale_entry((uint16_t) act_group[gb].d, weight_d);
                scale_words[word * (size_t) VPU_RESULT_PACK_LANES + lane] = packed_scale;
            }
        }
        for (size_t linear = scale_shape.entries;
             linear < scale_shape.words * (size_t) VPU_RESULT_PACK_LANES;
             ++linear) {
            scale_words[linear] = 0U;
        }
        mmio_fence();
    }

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
        if (job.weight_cache_hit) {
            totals->weight_cache_hits++;
        } else if (!job.p2_residency_hit) {
            totals->weight_cache_misses++;
        }
    }
    p2_event_trace(job, "PREP_DONE", event_prep_done, "prep_us", event_prep_done - event_prep0);
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
        "scope=graph_sequence reason=%s graph_seq=%d scheduler=%s preload_config=%d matmuls=%lld vpu_runs=%lld pingpong_pairs=%lld "
        "preload_attempts=%lld preload_admitted_while_active=%lld preload_terminal_skip=%lld "
        "serial_submit_after_no_preload=%lld input_preload_us=%lld preload_launch_bubble_us=%lld "
        "ip_compute_us=%lld dma_act_us=%lld dma_weight_us=%lld matrix_wall_us=%lld "
        "overlap_duration=not_measured",
        reason ? reason : "?", g_p1_sched_summary.graph_seq,
        g_pingpong_scheduler_enabled ? "pingpong" : "single_bank", g_p2_input_preload_enabled ? 1 : 0,
        g_p1_sched_summary.matmuls,
        g_p1_sched_summary.vpu_runs, g_p1_sched_summary.pingpong_pairs,
        g_p1_sched_summary.preload_attempts, g_p1_sched_summary.preload_admitted_while_active,
        g_p1_sched_summary.preload_terminal_skip, g_p1_sched_summary.serial_submit_after_no_preload,
        g_p1_sched_summary.input_preload_us, g_p1_sched_summary.preload_launch_bubble_us,
        g_p1_sched_summary.ip_compute_us, g_p1_sched_summary.dma_act_us, g_p1_sched_summary.dma_weight_us,
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

static bool fpga_p3_retire_timing_now_ns(uint64_t * out_ns) {
    struct timespec ts = {};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0 || ts.tv_sec < 0 || ts.tv_nsec < 0 ||
        ts.tv_nsec >= 1000000000L) {
        return false;
    }
    const uint64_t seconds = (uint64_t) ts.tv_sec;
    const uint64_t nanoseconds = (uint64_t) ts.tv_nsec;
    if (seconds > (UINT64_MAX - nanoseconds) / 1000000000ULL) {
        return false;
    }
    *out_ns = seconds * 1000000000ULL + nanoseconds;
    return true;
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
                                  g_token_timing.scale_dma_us + g_token_timing.preload_us;
    const long long token_read_us = g_token_timing.ip2host_dma_us + g_token_timing.host_read_us;
    const long long device_span_us =
        g_token_timing.first_ip_launch_mono_us > 0 &&
                g_token_timing.last_ip_output_ready_mono_us >= g_token_timing.first_ip_launch_mono_us ?
            g_token_timing.last_ip_output_ready_mono_us - g_token_timing.first_ip_launch_mono_us :
            0;
    const bool decode_token = ubatch_tokens == 1 && g_token_timing.first_m == 1;
    const char * scope = decode_token ? "decode_token" : (ubatch_tokens > 0 ? "prefill_or_ubatch" : "incomplete");
    if (decode_token) {
        ++g_summary_detail_decode_tokens;
        ++g_fpga_perf_decode.decode_tokens;
        g_fpga_perf_decode.decode_wall_us += token_wall_us;
        g_fpga_perf_decode.fpga_matmuls += g_token_timing.matmuls;
        g_fpga_perf_decode.vpu_runs += g_token_timing.vpu_runs;
        g_fpga_perf_decode.ip_compute_us += g_token_timing.ip_compute_us;
        g_fpga_perf_decode.h2ip_dma_us += h2ip_dma_us;
        g_fpga_perf_decode.output_transfer_us += token_read_us;
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
        g_fpga_perf_decode.preload_dma_us += g_token_timing.preload_us;
        g_fpga_perf_decode.preload_overlap_us += g_token_timing.scheduler_preload_overlap_us;
        g_fpga_perf_decode.preload_overlap_jobs += g_token_timing.scheduler_preload_overlap_jobs;
    }
    const bool sampled_detail = g_summary_detail_after_error ||
                                (decode_token &&
                                 (g_summary_detail_decode_tokens == 1 ||
                                  (g_summary_detail_every > 0 &&
                                   (g_summary_detail_decode_tokens % g_summary_detail_every) == 0)));
    g_summary_detail_after_error = false;

    if (g_token_timing_enabled) {
        fpga_log_line(
            true, "TOKEN_TIMING", force_flush,
            "graph_seq=%d next_graph_seq=%d ubatch_tokens=%lld scope=%s reason=%s matmuls=%lld vpu_runs=%lld "
            "token_wall_ms=%.3f device_first_start_to_last_output_ready_ms=%.3f fpga_matmul_wall_sum_ms=%.3f "
            "host_to_ip_dma_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f preload_dma_ms=%.3f "
            "ping_h2ip_ms=%.3f pong_h2ip_ms=%.3f ping_jobs=%lld pong_jobs=%lld "
            "ip_compute_sum_ms=%.3f ping_compute_ms=%.3f pong_compute_ms=%.3f "
            "ip_to_host_dma_ms=%.3f ping_ip2host_ms=%.3f pong_ip2host_ms=%.3f "
            "host_result_read_ms=%.3f ping_host_read_ms=%.3f pong_host_read_ms=%.3f token_read_ms=%.3f "
            "prep_ms=%.3f host_accum_ms=%.3f act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu "
            "measurement_scope=host_ddr_to_ip_ip_compute_ip_to_host_ddr internal_pl_interconnect=not_observable_from_host",
            g_token_timing.graph_seq, next_graph_seq, (long long) ubatch_tokens, scope, reason ? reason : "unknown",
            g_token_timing.matmuls, g_token_timing.vpu_runs, (double) token_wall_us / 1000.0,
            (double) device_span_us / 1000.0, (double) g_token_timing.matmul_wall_us / 1000.0,
            (double) h2ip_dma_us / 1000.0, (double) g_token_timing.act_dma_us / 1000.0,
            (double) g_token_timing.weight_dma_us / 1000.0, (double) g_token_timing.scale_dma_us / 1000.0,
            (double) g_token_timing.preload_us / 1000.0, (double) g_token_timing.bank_h2ip_us[0] / 1000.0,
            (double) g_token_timing.bank_h2ip_us[1] / 1000.0, g_token_timing.bank_jobs[0],
            g_token_timing.bank_jobs[1], (double) g_token_timing.ip_compute_us / 1000.0,
            (double) g_token_timing.bank_compute_us[0] / 1000.0,
            (double) g_token_timing.bank_compute_us[1] / 1000.0,
            (double) g_token_timing.ip2host_dma_us / 1000.0,
            (double) g_token_timing.bank_ip2host_us[0] / 1000.0,
            (double) g_token_timing.bank_ip2host_us[1] / 1000.0,
            (double) g_token_timing.host_read_us / 1000.0,
            (double) g_token_timing.bank_host_read_us[0] / 1000.0,
            (double) g_token_timing.bank_host_read_us[1] / 1000.0, (double) token_read_us / 1000.0,
            (double) g_token_timing.prep_us / 1000.0, (double) g_token_timing.host_accum_us / 1000.0,
            g_token_timing.activation_bytes, g_token_timing.weight_bytes, g_token_timing.scale_bytes,
            g_token_timing.result_bytes);
    }

    if (g_bottleneck_summary_enabled) {
        const long long prep_known_us = g_token_timing.prep_weight_select_us +
                                        g_token_timing.prep_direct_weight_pack_us +
                                        g_token_timing.prep_scale_pack_us + g_token_timing.prep_act_pack_us;
        const long long prep_other_us = std::max(0LL, g_token_timing.prep_us - prep_known_us);
        const long long outside_matmul_us = std::max(0LL, token_wall_us - g_token_timing.matmul_wall_us);
        const long long device_noncompute_us = std::max(0LL, device_span_us - g_token_timing.ip_compute_us);
        const double compute_util_pct = device_span_us > 0 ?
                                            100.0 * (double) g_token_timing.ip_compute_us / (double) device_span_us :
                                            0.0;
        const double preload_admission_overlap_pct = g_token_timing.preload_us > 0 ?
            std::clamp(100.0 * (double) g_token_timing.scheduler_preload_overlap_us /
                           (double) g_token_timing.preload_us,
                       0.0, 100.0) :
            0.0;
        fpga_log_line(
            true, "BOTTLENECK_SUMMARY", force_flush,
            "graph_seq=%d next_graph_seq=%d scope=%s token_wall_ms=%.3f matmul_wall_ms=%.3f "
            "outside_matmul_ms=%.3f device_span_ms=%.3f ip_compute_ms=%.3f device_noncompute_ms=%.3f "
            "compute_util_pct=%.2f prep_total_ms=%.3f prep_weight_select_ms=%.3f "
            "prep_direct_weight_pack_ms=%.3f prep_scale_pack_ms=%.3f prep_act_pack_ms=%.3f prep_other_ms=%.3f "
            "handoffs=%lld prep_overlap_ms=%.3f prep_late_jobs=%lld prep_late_ms=%.3f prep_headroom_ms=%.3f "
            "preload_overlap_jobs=%lld preload_overlap_ms=%.3f preload_recorded_ms=%.3f preload_overlap_pct=%.2f "
            "output_ready_to_next_launch_ms=%.3f retire_to_next_launch_ms=%.3f",
            g_token_timing.graph_seq, next_graph_seq, scope, (double) token_wall_us / 1000.0,
            (double) g_token_timing.matmul_wall_us / 1000.0, (double) outside_matmul_us / 1000.0,
            (double) device_span_us / 1000.0, (double) g_token_timing.ip_compute_us / 1000.0,
            (double) device_noncompute_us / 1000.0, compute_util_pct, (double) g_token_timing.prep_us / 1000.0,
            (double) g_token_timing.prep_weight_select_us / 1000.0,
            (double) g_token_timing.prep_direct_weight_pack_us / 1000.0,
            (double) g_token_timing.prep_scale_pack_us / 1000.0,
            (double) g_token_timing.prep_act_pack_us / 1000.0, (double) prep_other_us / 1000.0,
            g_token_timing.scheduler_handoffs, (double) g_token_timing.scheduler_prepare_overlap_us / 1000.0,
            g_token_timing.scheduler_prepare_late_jobs, (double) g_token_timing.scheduler_prepare_late_us / 1000.0,
            (double) g_token_timing.scheduler_prepare_headroom_us / 1000.0,
            g_token_timing.scheduler_preload_overlap_jobs,
            (double) g_token_timing.scheduler_preload_overlap_us / 1000.0,
            (double) g_token_timing.preload_us / 1000.0, preload_admission_overlap_pct,
            (double) g_token_timing.scheduler_output_to_launch_us / 1000.0,
            (double) g_token_timing.scheduler_retire_to_launch_us / 1000.0);

        if (sampled_detail) {
            fpga_log_line(
                true, "BOTTLENECK_CATEGORY", force_flush,
                "graph_seq=%d scope=%s "
                "attn_matmuls=%lld attn_runs=%lld attn_wall_ms=%.3f attn_prep_ms=%.3f attn_compute_ms=%.3f attn_dma_ms=%.3f "
                "gate_matmuls=%lld gate_runs=%lld gate_wall_ms=%.3f gate_prep_ms=%.3f gate_compute_ms=%.3f gate_dma_ms=%.3f "
                "up_matmuls=%lld up_runs=%lld up_wall_ms=%.3f up_prep_ms=%.3f up_compute_ms=%.3f up_dma_ms=%.3f "
                "down_matmuls=%lld down_runs=%lld down_wall_ms=%.3f down_prep_ms=%.3f down_compute_ms=%.3f down_dma_ms=%.3f "
                "other_matmuls=%lld other_runs=%lld other_wall_ms=%.3f other_prep_ms=%.3f other_compute_ms=%.3f other_dma_ms=%.3f",
                g_token_timing.graph_seq, scope,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_ATTN], g_token_timing.category_runs[FPGA_BOTTLENECK_ATTN],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                (double) g_token_timing.category_compute_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_ATTN] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_FFN_GATE], g_token_timing.category_runs[FPGA_BOTTLENECK_FFN_GATE],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                (double) g_token_timing.category_compute_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_FFN_GATE] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_FFN_UP], g_token_timing.category_runs[FPGA_BOTTLENECK_FFN_UP],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                (double) g_token_timing.category_compute_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_FFN_UP] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_FFN_DOWN], g_token_timing.category_runs[FPGA_BOTTLENECK_FFN_DOWN],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                (double) g_token_timing.category_compute_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_FFN_DOWN] / 1000.0,
                g_token_timing.category_matmuls[FPGA_BOTTLENECK_OTHER], g_token_timing.category_runs[FPGA_BOTTLENECK_OTHER],
                (double) g_token_timing.category_wall_us[FPGA_BOTTLENECK_OTHER] / 1000.0,
                (double) g_token_timing.category_prep_us[FPGA_BOTTLENECK_OTHER] / 1000.0,
                (double) g_token_timing.category_compute_us[FPGA_BOTTLENECK_OTHER] / 1000.0,
                (double) g_token_timing.category_dma_us[FPGA_BOTTLENECK_OTHER] / 1000.0);

            const double zdma_avg_us = g_token_timing.zdma_descriptors > 0 ?
                                           (double) g_token_timing.zdma_elapsed_us / (double) g_token_timing.zdma_descriptors :
                                           0.0;
            fpga_log_line(
                true, "BOTTLENECK_ZDMA", force_flush,
                "graph_seq=%d scope=%s descriptors=%lld bytes=%zu elapsed_ms=%.3f avg_descriptor_us=%.3f polls=%lld "
                "zero_poll_descriptors=%lld saw_enabled_descriptors=%lld act_desc=%lld weight_desc=%lld scale_desc=%lld "
                "result_desc=%lld other_desc=%lld max_descriptor_bytes=%zu",
                g_token_timing.graph_seq, scope, g_token_timing.zdma_descriptors, g_token_timing.zdma_bytes,
                (double) g_token_timing.zdma_elapsed_us / 1000.0, zdma_avg_us, g_token_timing.zdma_polls,
                g_token_timing.zdma_zero_poll_descriptors, g_token_timing.zdma_saw_enabled_descriptors,
                g_token_timing.zdma_act_descriptors, g_token_timing.zdma_weight_descriptors,
                g_token_timing.zdma_scale_descriptors, g_token_timing.zdma_result_descriptors,
                g_token_timing.zdma_other_descriptors, g_zdma_max_transfer_bytes);
        }
    }
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
    g_token_timing.preload_us += totals.input_preload_us;
    g_token_timing.ip_compute_us += totals.ip_compute_us;
    g_token_timing.ip2host_dma_us += totals.dma_result_us;
    g_token_timing.host_read_us += totals.host_result_us;
    g_token_timing.host_accum_us += totals.host_accum_us;
    g_token_timing.scheduler_prepare_overlap_us += totals.scheduler_prepare_overlap_us;
    g_token_timing.scheduler_prepare_late_us += totals.scheduler_prepare_late_us;
    g_token_timing.scheduler_prepare_headroom_us += totals.scheduler_prepare_headroom_us;
    g_token_timing.scheduler_preload_overlap_us += totals.scheduler_preload_overlap_us;
    g_token_timing.scheduler_output_to_launch_us += totals.scheduler_output_to_launch_us;
    g_token_timing.scheduler_retire_to_launch_us += totals.scheduler_retire_to_launch_us;
    g_token_timing.scheduler_handoffs += totals.scheduler_handoffs;
    g_token_timing.scheduler_prepare_late_jobs += totals.scheduler_prepare_late_jobs;
    g_token_timing.scheduler_preload_overlap_jobs += totals.scheduler_preload_overlap_jobs;
    const int category = (int) fpga_bottleneck_category(tensor_name);
    g_token_timing.category_matmuls[category]++;
    g_token_timing.category_runs[category] += totals.vpu_runs;
    g_token_timing.category_wall_us[category] += matmul_wall_us;
    g_token_timing.category_prep_us[category] += totals.prep_us;
    g_token_timing.category_compute_us[category] += totals.ip_compute_us;
    g_token_timing.category_dma_us[category] += totals.dma_act_us + totals.dma_weight_us + totals.dma_scale_us +
                                                 totals.input_preload_us + totals.dma_result_us;
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
    if (!g_p2_event_trace_enabled || !g_spu_q8_scale_stream_supported || g_contract_check_limit > 0) {
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
            "[FPGA][INFO] P2_DESCRIPTOR_COMMIT edge=before job=%u bank=%d tile=%u expected bank_bits=0x%08x "
            "bank_stat_bits=0x%08x job_id=0x%08x slot_state=0x%08x tensor_id=0x%08x row0=0x%08x k_block0=0x%08x "
            "group_blocks=0x%08x token_id=0x%08x desc_flags=0x%08x actual=pending match=pending\n",
            job.job_id, job.bank, job.tile_id, expected.bank, expected.bank_stat, expected.job_id, expected.slot_state,
            expected.tensor_id, expected.row0, expected.k_block0, expected.group_blocks, expected.token_id,
            expected.desc_flags);
    fflush(fp);

    if (g_p2_terminal_trace_enabled) {
        fprintf(stderr,
                "[FPGA][P2_DESCRIPTOR] edge=before job=%u bank=%d tile=%u expected bank_bits=0x%08x bank_stat_bits=0x%08x "
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
            "[FPGA][INFO] P2_DESCRIPTOR_COMMIT edge=after job=%u bank=%d tile=%u expected bank_bits=0x%08x "
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
            "[FPGA][P2_DESCRIPTOR] edge=after job=%u bank=%d tile=%u expected bank_bits=0x%08x bank_stat_bits=0x%08x "
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
        "P2_LAYOUT_SELFTEST_PASS layout=pair_interleaved_padded cases=20 "
        "i8x16_le_words=03020100,07060504,0b0a0908,0f0e0d0c "
        "emission=logical_row_equals_pair_major_zero_odd mapping=index=((row>>1)*group_beats+beat)*2+(row&1)");
    return true;
}

static bool fpga_p3_split_scale_host_self_test(void) {
    static constexpr uint16_t lanes[8] = {0x3c00U, 0x7e01U, 0x0001U, 0xbc00U,
                                          0x3555U, 0x8000U, 0x7bffU, 0x0000U};
    static constexpr uint32_t expected[4] = {0x7e013c00U, 0xbc000001U, 0x80003555U, 0x00007bffU};
    for (size_t pair = 0; pair < 4U; ++pair) {
        if (fpga_p3_pack_fp16_pair(lanes[pair * 2U], lanes[pair * 2U + 1U]) != expected[pair]) {
            LOGE("P3_HOST_SELFTEST_FAIL reason=fp16_pair_bits pair=%zu", pair);
            return false;
        }
    }
    const size_t max_entries = (size_t) P3_MAX_ROWS * (size_t) P3_MAX_GROUP_BLOCKS;
    const size_t max_words = (max_entries + 7U) / 8U;
    if (max_entries != 16384U || max_words != 2048U || ((size_t) P3_MAX_GROUP_BLOCKS + 7U) / 8U != 8U) {
        LOGE("P3_HOST_SELFTEST_FAIL reason=dense_scale_capacity entries=%zu words=%zu", max_entries, max_words);
        return false;
    }
    LOGINIT("P3_HOST_SELFTEST_PASS dense_fp16_words=8lanes max_entries=%zu max_weight_words=%zu", max_entries,
             max_words);
    return true;
}

static bool fpga_p6_accum_trace_host_self_test(void) {
    fpga_p6_accum_trace_t saved = g_p6_accum_trace;
    fpga_p6_accum_trace_t trace = {};
    trace.enabled = true;
    trace.tensor = "p6-self-test";
    trace.layer = 3;
    trace.tile_row0 = 0;
    trace.row_first = 0;
    trace.row_count = 1;
    trace.expected_k_chunks = 4;
    trace.record_limit = 4;
    g_p6_accum_trace = trace;
    const bool m2_deferred = !fpga_p6_selected_m1_eligible(true, 2);
    const bool m1_eligible = fpga_p6_selected_m1_eligible(true, 1) &&
                             !fpga_p6_selected_m1_eligible(false, 1);
    fpga_p6_accum_trace_t cleanup_incomplete = trace;
    cleanup_incomplete.selected_m1_matmul_seen = true;
    const bool cleanup_incomplete_rejected = fpga_p6_cleanup_requires_failure(true, cleanup_incomplete);
    cleanup_incomplete.capture_complete = true;
    const bool cleanup_complete_accepted = !fpga_p6_cleanup_requires_failure(true, cleanup_incomplete);
    const bool cleanup_before_init_accepted = !fpga_p6_cleanup_requires_failure(false, trace);
    fpga_tile_job_t job = {};
    job.tensor_name = "p6-self-test";
    job.layer_id = 3;
    job.row0 = 0;
    job.rows = 1;
    job.col = 0;
    job.shape_k = 128;
    job.shape_m = 1;
    job.k_block0 = 0;
    job.group_blocks = 1;
    int ordinal = -1;
    bool final_chunk = true;
    const bool first_ok = fpga_p6_accept_chunk(job, &ordinal, &final_chunk) && ordinal == 0 && !final_chunk;
    const bool duplicate_rejected = !fpga_p6_accept_chunk(job, &ordinal, &final_chunk);
    job.k_block0 = 2;
    const bool gap_rejected = !fpga_p6_accept_chunk(job, &ordinal, &final_chunk);
    job.k_block0 = 1;
    const bool second_ok = fpga_p6_accept_chunk(job, &ordinal, &final_chunk) && ordinal == 1 && !final_chunk;
    job.k_block0 = 2;
    const bool third_ok = fpga_p6_accept_chunk(job, &ordinal, &final_chunk) && ordinal == 2 && !final_chunk;
    job.k_block0 = 3;
    const bool fourth_ok = fpga_p6_accept_chunk(job, &ordinal, &final_chunk) && ordinal == 3 && final_chunk;
    const bool extra_chunk_rejected = !fpga_p6_accept_chunk(job, &ordinal, &final_chunk);
    job.layer_id = 4;
    const bool selector_rejected = !fpga_p6_job_selected(job);
    job.layer_id = 3;
    uint32_t plus_zero_bits = 0U;
    uint32_t minus_zero_bits = 0U;
    const float plus_zero = 0.0f;
    const float minus_zero = -0.0f;
    memcpy(&plus_zero_bits, &plus_zero, sizeof(plus_zero_bits));
    memcpy(&minus_zero_bits, &minus_zero, sizeof(minus_zero_bits));
    const bool zero_bits_ok = plus_zero_bits == 0x00000000U && minus_zero_bits == 0x80000000U;

    // Exercise the deployed accumulation ordering in memory over four chunks.
    // This covers positive, negative, and exact cancellation, while keeping
    // the production accumulation statement below singular and unchanged.
    static constexpr int64_t q16_values[] = {65536, -65536, -32768, 32768};
    static constexpr uint32_t before_expected[] = {0x00000000U, 0x3f800000U, 0x00000000U, 0xbf000000U};
    static constexpr uint32_t after_expected[]  = {0x3f800000U, 0x00000000U, 0xbf000000U, 0x00000000U};
    float p6_accum = 0.0f;
    size_t p6_records = 0U;
    bool accumulation_bits_ok = true;
    for (size_t chunk = 0U; chunk < sizeof(q16_values) / sizeof(q16_values[0]); ++chunk) {
        uint32_t before_bits = 0U;
        memcpy(&before_bits, &p6_accum, sizeof(before_bits));
        const float contribution = (float) q16_values[chunk] * (1.0f / 65536.0f);
        p6_accum = p6_accum + contribution;
        uint32_t after_bits = 0U;
        memcpy(&after_bits, &p6_accum, sizeof(after_bits));
        accumulation_bits_ok = accumulation_bits_ok && before_bits == before_expected[chunk] &&
                               after_bits == after_expected[chunk];
        ++p6_records;
    }
    const bool final_record_count_ok = p6_records == (size_t) trace.row_count * (size_t) trace.expected_k_chunks &&
                                       p6_records == trace.record_limit;

    // Premature-final and record-cap failures are tested from clean states.
    g_p6_accum_trace = trace;
    job.k_block0 = 0;
    job.shape_k = 64;
    job.group_blocks = 2;
    const bool premature_final_rejected = !fpga_p6_accept_chunk(job, &ordinal, &final_chunk);
    g_p6_accum_trace = trace;
    g_p6_accum_trace.expected_k_chunks = 2;
    g_p6_accum_trace.record_limit = 2;
    job.shape_k = 96;
    job.group_blocks = 1;
    const bool missing_final_first_ok = fpga_p6_accept_chunk(job, &ordinal, &final_chunk) && !final_chunk;
    job.k_block0 = 1;
    const bool missing_final_rejected = !fpga_p6_accept_chunk(job, &ordinal, &final_chunk);
    g_p6_accum_trace = trace;
    g_p6_accum_trace.row_count = 2;
    g_p6_accum_trace.record_limit = 1;
    job.shape_k = 128;
    job.k_block0 = 0;
    job.group_blocks = 1;
    job.rows = 2;
    const bool cap_rejected = !fpga_p6_accept_chunk(job, &ordinal, &final_chunk);
    g_p6_accum_trace = saved;
    return m2_deferred && m1_eligible && cleanup_incomplete_rejected && cleanup_complete_accepted &&
           cleanup_before_init_accepted && first_ok && duplicate_rejected && gap_rejected && second_ok && third_ok && fourth_ok &&
           extra_chunk_rejected && selector_rejected && zero_bits_ok && accumulation_bits_ok && final_record_count_ok &&
           premature_final_rejected && missing_final_first_ok && missing_final_rejected && cap_rejected;
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
            LOGE("P3_TILE_LIMIT_SELFTEST_FAIL reason=matrix_admitted_after_boundary cumulative=%lld limit=%d",
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
                LOGE("P3_TILE_LIMIT_SELFTEST_FAIL reason=incorrect_boundary cumulative=%lld limit=%d reached=%d",
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
        LOGE("P3_TILE_LIMIT_SELFTEST_FAIL reason=final_state cumulative=%lld limit=%d matrices=%d reached=%d",
             cumulative, test_limit, completed_matrices, boundary_reached ? 1 : 0);
        return false;
    }
    LOGINIT(
        "P3_TILE_LIMIT_SELFTEST_PASS scope=cumulative_across_matrices limit=%d matrices=%d continuation=52/256 "
        "terminal=256/256 inconsistent_states=rejected simulated_257=not_admitted exact_stop=%lld",
        test_limit, completed_matrices, cumulative);
    return true;
}

static bool fpga_dma_basic_self_test(void) {
    int8_t ones[VPU_QK8_0];
    for (int i = 0; i < VPU_QK8_0; ++i) {
        ones[i] = 1;
    }
    for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) beat * 16U, ones + beat * VPU_NUM_LANES);
        uint32_t weight_off = 0;
        if (!fpga_weight_layout_word_offset(WEIGHT_BASE, 1, VPU_BLOCK_BEATS, 0, beat, &weight_off)) {
            return false;
        }
        write_i8x16_to_ddr(weight_off, ones + beat * VPU_NUM_LANES);
    }
    const size_t weight_bytes = weight_window_bytes_for_rows(1, VPU_BLOCK_BEATS);
    fpga_weight_layout_zero_padded_companion(WEIGHT_BASE, 1, VPU_BLOCK_BEATS);

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(1, VPU_BLOCK_BEATS, 0, VPU_QK8_0, weight_bytes, 16U, "selftest.basic", -1, 32, 1, 1, 0,
                                 &totals)) {
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
        act0[i]          = 1;
        act1[i]          = 2;
        w_row0_block0[i] = 1;
        w_row0_block1[i] = 1;
        w_row1_block0[i] = -1;
        w_row1_block1[i] = 3;
    }

    const int    packed_rows         = 2;
    const int    packed_group_beats  = 4;
    const size_t packed_weight_bytes = weight_window_bytes_for_rows(packed_rows, packed_group_beats);
    ddr_zero_range32(WEIGHT_BASE, packed_weight_bytes);

    for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) beat * 16U, act0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(ACT_BASE + (uint32_t) (VPU_BLOCK_BEATS + beat) * 16U, act1 + beat * VPU_NUM_LANES);
        uint32_t row0_block0_off = 0, row0_block1_off = 0, row1_block0_off = 0, row1_block1_off = 0;
        if (!fpga_weight_layout_word_offset(WEIGHT_BASE, packed_rows, packed_group_beats, 0, beat,
                                            &row0_block0_off) ||
            !fpga_weight_layout_word_offset(WEIGHT_BASE, packed_rows, packed_group_beats, 0,
                                            VPU_BLOCK_BEATS + beat, &row0_block1_off) ||
            !fpga_weight_layout_word_offset(WEIGHT_BASE, packed_rows, packed_group_beats, 1, beat,
                                            &row1_block0_off) ||
            !fpga_weight_layout_word_offset(WEIGHT_BASE, packed_rows, packed_group_beats, 1,
                                            VPU_BLOCK_BEATS + beat, &row1_block1_off)) {
            return false;
        }
        write_i8x16_to_ddr(row0_block0_off, w_row0_block0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(row0_block1_off, w_row0_block1 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(row1_block0_off, w_row1_block0 + beat * VPU_NUM_LANES);
        write_i8x16_to_ddr(row1_block1_off, w_row1_block1 + beat * VPU_NUM_LANES);
    }

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(packed_rows, packed_group_beats, VPU_MODE_PACKED_Q8, 4U * 16U, packed_weight_bytes,
                                 16U, "selftest.packed", -1, 64, 2, 1, 1, &totals)) {
        return false;
    }

    int32_t lanes[4] = {};
    read_result_i32x4_from_ddr(0, lanes);
    LOGI("packed ZDMA-to-IP self-test results=[%d,%d,%d,%d] expected=[32,64,-32,192]", lanes[0], lanes[1], lanes[2],
         lanes[3]);
    return lanes[0] == 32 && lanes[1] == 64 && lanes[2] == -32 && lanes[3] == 192;
}

static bool fpga_dma_row_limit_self_test(void) {
    const int      rows          = g_vpu_max_rows;
    const int      group_blocks  = std::min(2, g_packed_q8_max_blocks);
    const int      group_beats   = group_blocks * VPU_BLOCK_BEATS;
    const uint32_t result_values = (uint32_t) rows * (uint32_t) group_blocks;
    const uint32_t result_words =
        (result_values + (uint32_t) VPU_RESULT_PACK_LANES - 1U) / (uint32_t) VPU_RESULT_PACK_LANES;

    if (rows <= 2 || group_blocks <= 0 || result_words > (uint32_t) g_packed_q8_result_words) {
        LOGI("row-limit self-test skipped rows=%d group_blocks=%d result_words=%u cap=%d", rows, group_blocks,
             result_words, g_packed_q8_result_words);
        return true;
    }

    const size_t act_bytes    = (size_t) group_beats * 16U;
    const size_t weight_bytes = weight_window_bytes_for_rows(rows, group_beats);
    const size_t result_bytes = (size_t) result_words * 16U;
    if (!range_fits(ACT_BASE, act_bytes, ACT_BASE, ACT_END) ||
        !range_fits(WEIGHT_BASE, weight_bytes, WEIGHT_BASE, WEIGHT_END) ||
        !range_fits(RESULT_BASE, result_bytes, RESULT_BASE, RESULT_END)) {
        LOGE("row-limit self-test window overflow rows=%d group_beats=%d act=%zu weight=%zu result=%zu", rows,
             group_beats, act_bytes, weight_bytes, result_bytes);
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
            int8_t       weight[VPU_QK8_0];
            for (int i = 0; i < VPU_QK8_0; ++i) {
                weight[i] = weight_value;
            }
            for (int beat = 0; beat < VPU_BLOCK_BEATS; ++beat) {
                uint32_t word_off = 0;
                if (!fpga_weight_layout_word_offset(WEIGHT_BASE, rows, group_beats, row,
                                                    gb * VPU_BLOCK_BEATS + beat, &word_off)) {
                    return false;
                }
                write_i8x16_to_ddr(word_off, weight + beat * VPU_NUM_LANES);
            }
        }
    }
    fpga_weight_layout_zero_padded_companion(WEIGHT_BASE, rows, group_beats);
    mmio_fence();

    fpga_stage_totals_t totals = {};
    if (!run_vpu_window_transfer(rows, group_beats, VPU_MODE_PACKED_Q8, act_bytes, weight_bytes, result_bytes,
                                 "selftest.row_limit", -1, VPU_QK8_0 * group_blocks, rows, 1, 2, &totals)) {
        return false;
    }

    const int probe_rows[3] = { 0, rows / 2, rows - 1 };
    for (int probe = 0; probe < 3; ++probe) {
        const int row = probe_rows[probe];
        for (int gb = 0; gb < group_blocks; ++gb) {
            const int32_t got      = read_result_i32_flat((uint32_t) row * (uint32_t) group_blocks + (uint32_t) gb);
            const int32_t expected = (int32_t) VPU_QK8_0 * (int32_t) (gb + 1) * (int32_t) (((row + gb) % 5) - 2);
            if (got != expected) {
                LOGE("row-limit self-test mismatch rows=%d row=%d block=%d got=%d expected=%d", rows, row, gb, got,
                     expected);
                return false;
            }
        }
    }

    LOGI("row-limit self-test passed rows=%d group_blocks=%d result_words=%u", rows, group_blocks, result_words);
    return true;
}

static bool fpga_weight_path_bench_host_self_test(void);

// The weight-path benchmark intentionally has no ZDMA ownership.  It maps
// only the verified UIO resources needed to read identity/capability words
// and to make bounded CPU stores into the existing WEIGHT staging window.
static bool map_weight_path_bench_vpu_ddr(void) {
    g_ddr_requested_map_size = DDR_REQUIRED_BYTES;
    g_ddr_advertised_size    = 0U;
    g_ddr_mapping_kind       = fpga_mapping_kind::UNKNOWN;

    if (!map_uio_region("MY_IP", "FPGA_VPU_UIO", REG_BASE_PHYS, VPU_DEVMEM_COMPAT_MMAP,
                        VPU_DEVMEM_COMPAT_MMAP, "MY_IP/VPU passive benchmark", VPU_DEVMEM_COMPAT_MMAP,
                        &g_vpu_map_base, &g_vpu_map_size, nullptr, &g_vpu_map_source, nullptr)) {
        return false;
    }
    g_vpu = (volatile uint8_t *) g_vpu_map_base;

    if (!fpga_ddr_iomem_preflight() ||
        !map_uio_region("fpga_ddr_low", "FPGA_DDR_UIO", DDR_BASE_PHYS, DDR_REGION_SIZE, DDR_REQUIRED_BYTES,
                        "fpga_ddr_low passive benchmark", DDR_REQUIRED_BYTES, &g_ddr_map_base, &g_ddr_map_size,
                        &g_ddr_advertised_size, &g_ddr_map_source, &g_ddr_mapping_kind)) {
        if (g_vpu_map_base && g_vpu_map_base != MAP_FAILED) {
            munmap(g_vpu_map_base, g_vpu_map_size);
        }
        g_vpu_map_base = nullptr;
        g_vpu          = nullptr;
        g_vpu_map_size = 0U;
        return false;
    }
    g_ddr = (uint8_t *) g_ddr_map_base;
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

static bool fpga_audit_q8_source_only(const struct ggml_tensor * src0, const char * tensor_name, int layer_id) {
    if (!src0 || src0->type != GGML_TYPE_Q8_0 || !src0->data) {
        return true;
    }

    g_q8_source_audit_checks++;
    const bool valid = fpga_contract_validate_weight_scales(src0, src0->data, tensor_name, layer_id);
    if (!valid) {
        g_q8_source_audit_failures++;
        LOGE(
            "Q8_SOURCE_AUDIT_FAIL tensor=%s layer=%d check=%lld action=stop_before_model_zdma_vpu_gemv; this run did "
            "not submit this tensor to ZDMA/VPU",
            tensor_name ? tensor_name : "?", layer_id, g_q8_source_audit_checks);
        return false;
    }

    LOGI("Q8_SOURCE_AUDIT_PASS tensor=%s layer=%d check=%lld action=cpu_matmul_only", tensor_name ? tensor_name : "?",
         layer_id, g_q8_source_audit_checks);
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

static bool fpga_contract_verify_staged_q8_group(const block_q8_0_t * weight_snapshot,
                                                 const block_q8_0_t * act_group,
                                                 int                  rows,
                                                 int                  group_blocks,
                                                 uint32_t             weight_src_off,
                                                 const char *         tensor_name,
                                                 int                  layer_id,
                                                 uint32_t             tile_id,
                                                 const char *         phase) {
    const int group_beats = group_blocks * VPU_BLOCK_BEATS;
    for (int gb = 0; gb < group_blocks; ++gb) {
        const volatile int8_t * const staged =
            (volatile const int8_t *) ddr_ptr(ACT_BASE + (uint32_t) gb * VPU_QK8_0, VPU_QK8_0);
        for (int lane = 0; lane < VPU_QK8_0; ++lane) {
            if (staged[lane] != act_group[gb].qs[lane]) {
                LOGE(
                    "CONTRACT_STAGING_BOUNDARY_FAIL phase=%s kind=ACT tensor=%s layer=%d tile=%u block=%d lane=%d "
                    "expected=%d actual=%d",
                    phase ? phase : "?", tensor_name ? tensor_name : "?", layer_id, tile_id, gb, lane,
                    (int) act_group[gb].qs[lane], (int) staged[lane]);
                return false;
            }
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t & expected = weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
            uint32_t off = 0;
            if (!fpga_weight_layout_word_offset(weight_src_off, rows, group_beats, row, gb * VPU_BLOCK_BEATS,
                                                &off)) {
                LOGE("CONTRACT_STAGING_BOUNDARY_FAIL phase=%s kind=WEIGHT reason=layout_offset_overflow rows=%d group_beats=%d row=%d block=%d",
                     phase ? phase : "?", rows, group_beats, row, gb);
                return false;
            }
            const volatile int8_t * const staged = (volatile const int8_t *) ddr_ptr(off, VPU_QK8_0);
            for (int lane = 0; lane < VPU_QK8_0; ++lane) {
                if (staged[lane] != expected.qs[lane]) {
                    LOGE(
                        "CONTRACT_STAGING_BOUNDARY_FAIL phase=%s kind=WEIGHT tensor=%s layer=%d tile=%u row=%d "
                        "block=%d lane=%d off=0x%08x expected=%d actual=%d",
                        phase ? phase : "?", tensor_name ? tensor_name : "?", layer_id, tile_id, row, gb, lane,
                        off + (uint32_t) lane, (int) expected.qs[lane], (int) staged[lane]);
                    return false;
                }
            }
        }
    }
    mmio_fence();
    return true;
}

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
                                                    uint32_t             tile_id) {
    const int attempts = guard_enabled ? 2 : 1;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        fpga_stage_q8_group_payload(weight_snapshot, act_group, rows, group_blocks, write_weight_payload,
                                    weight_src_off);
        fpga_ddr_staging_readback_commit(ACT_BASE, act_bytes);
        fpga_ddr_staging_readback_commit(weight_src_off, weight_bytes);

        if (!guard_enabled || fpga_contract_verify_staged_q8_group(weight_snapshot, act_group, rows, group_blocks,
                                                                   weight_src_off, tensor_name, layer_id, tile_id,
                                                                   attempt == 0 ? "after_stage" : "after_restage")) {
            if (attempt > 0) {
                g_contract_staging_restage_count++;
                LOGI(
                    "CONTRACT_STAGING_RESTAGE_RECOVERED tensor=%s layer=%d tile=%u attempts=%d; the corrected source "
                    "was verified before VPU start",
                    tensor_name ? tensor_name : "?", layer_id, tile_id, attempt + 1);
            }
            return true;
        }

        LOGE("CONTRACT_STAGING_RESTAGE tensor=%s layer=%d tile=%u attempt=%d reason=pre_vpu_ddr_source_mismatch",
             tensor_name ? tensor_name : "?", layer_id, tile_id, attempt + 1);
    }

    LOGE(
        "CONTRACT_STAGING_RESTAGE_FAILED tensor=%s layer=%d tile=%u attempts=%d; refusing to launch VPU with an "
        "unverified ACT/WEIGHT DDR source",
        tensor_name ? tensor_name : "?", layer_id, tile_id, attempts);
    return false;
}

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
                                                uint32_t             tile_id) {
    const uint64_t act_src_begin          = DDR_BASE_PHYS + (uint64_t) ACT_BASE;
    const uint64_t act_src_end            = act_src_begin + (uint64_t) act_bytes;
    const uint64_t act_dst_begin          = LMM_BASE_PHYS + (uint64_t) ACT_BASE;
    const uint64_t act_dst_end            = act_dst_begin + (uint64_t) act_bytes;
    const uint64_t weight_src_begin       = DDR_BASE_PHYS + (uint64_t) weight_src_off;
    const uint64_t weight_src_end         = weight_src_begin + (uint64_t) weight_bytes;
    const bool     source_ranges_disjoint = act_src_end <= weight_src_begin || weight_src_end <= act_src_begin;

    LOGE(
        "CONTRACT_STAGING_ACT_DMA_CONTEXT tensor=%s layer=%d tile=%u act_src=[0x%llx,0x%llx) act_dst=[0x%llx,0x%llx) "
        "weight_src=[0x%llx,0x%llx) source_ranges_disjoint=%d write_weight_payload=%d",
        tensor_name ? tensor_name : "?", layer_id, tile_id, (unsigned long long) act_src_begin,
        (unsigned long long) act_src_end, (unsigned long long) act_dst_begin, (unsigned long long) act_dst_end,
        (unsigned long long) weight_src_begin, (unsigned long long) weight_src_end, source_ranges_disjoint ? 1 : 0,
        write_weight_payload ? 1 : 0);
    zdma_dump("contract_staging_changed_after_act_dma");
    fpga_dma_trace_dump("staging_changed_after_act_dma", tensor_name, layer_id, tile_id, "ACT");

    if (!write_weight_payload) {
        LOGE(
            "CONTRACT_STAGING_RESTAGE_FAILED tensor=%s layer=%d tile=%u "
            "reason=weight_cache_source_changed_after_act_dma",
            tensor_name ? tensor_name : "?", layer_id, tile_id);
        return false;
    }

    fpga_stage_q8_group_payload(weight_snapshot, act_group, rows, group_blocks, true, weight_src_off);
    fpga_ddr_staging_readback_commit(ACT_BASE, act_bytes);
    fpga_ddr_staging_readback_commit(weight_src_off, weight_bytes);
    if (!fpga_contract_verify_staged_q8_group(weight_snapshot, act_group, rows, group_blocks, weight_src_off,
                                              tensor_name, layer_id, tile_id, "after_act_dma_restage")) {
        LOGE("CONTRACT_STAGING_RESTAGE_FAILED tensor=%s layer=%d tile=%u reason=post_restage_source_mismatch",
             tensor_name ? tensor_name : "?", layer_id, tile_id);
        return false;
    }

    g_contract_staging_restage_count++;
    LOGE(
        "CONTRACT_STAGING_RESTAGE_RECOVERED tensor=%s layer=%d tile=%u phase=after_act_dma; raw contract will "
        "continue, but this run is ineligible for primary-FPGA admission",
        tensor_name ? tensor_name : "?", layer_id, tile_id);
    return true;
}

static bool fpga_contract_verify_weight_source_snapshot(const struct ggml_tensor * src0,
                                                        const block_q8_0_t *       weight_snapshot,
                                                        int64_t                    row0,
                                                        int                        rows,
                                                        int64_t                    k_block0,
                                                        int                        group_blocks,
                                                        const char *               tensor_name,
                                                        int                        layer_id,
                                                        uint32_t                   tile_id) {
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * const live     = weight_block(src0, row0 + row, k_block0 + gb);
            const block_q8_0_t * const snapshot = &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
            if (memcmp(live, snapshot, sizeof(*snapshot)) != 0) {
                int                   first_bad      = 0;
                const uint8_t * const live_bytes     = (const uint8_t *) live;
                const uint8_t * const snapshot_bytes = (const uint8_t *) snapshot;
                while (first_bad < (int) sizeof(*snapshot) && live_bytes[first_bad] == snapshot_bytes[first_bad]) {
                    first_bad++;
                }
                LOGE(
                    "CONTRACT_WEIGHT_SOURCE_MUTATION tensor=%s layer=%d tile=%u row=%lld block=%lld byte=%d "
                    "snapshot=%u live=%u snapshot_d=0x%04x live_d=0x%04x; immutable GGUF weight changed during one VPU "
                    "launch",
                    tensor_name ? tensor_name : "?", layer_id, tile_id, (long long) (row0 + row),
                    (long long) (k_block0 + gb), first_bad,
                    first_bad < (int) sizeof(*snapshot) ? snapshot_bytes[first_bad] : 0U,
                    first_bad < (int) sizeof(*snapshot) ? live_bytes[first_bad] : 0U, (unsigned) snapshot->d,
                    (unsigned) live->d);
                return false;
            }
        }
    }
    return true;
}

static bool fpga_contract_log_staging_audit(const block_q8_0_t * act,
                                            const block_q8_0_t * weight,
                                            int                  local_row,
                                            int                  group_block,
                                            int                  group_beats,
                                            uint32_t             weight_src_off,
                                            const char *         tensor_name,
                                            int                  layer_id,
                                            uint32_t             tile_id,
                                            const char *         phase) {
    const uint32_t act_off = ACT_BASE + (uint32_t) group_block * VPU_BLOCK_BEATS * 16U;
    uint32_t weight_off = 0;
    if (!fpga_weight_layout_word_offset(weight_src_off, local_row + 1, group_beats, local_row,
                                        group_block * VPU_BLOCK_BEATS, &weight_off)) {
        LOGE("CONTRACT_STAGING_AUDIT phase=%s integrity=fail reason=layout_offset_overflow local_row=%d group_block=%d group_beats=%d",
             phase ? phase : "post_result", local_row, group_block, group_beats);
        return false;
    }
    const volatile int8_t * const staged_act       = (volatile const int8_t *) ddr_ptr(act_off, VPU_QK8_0);
    const volatile int8_t * const staged_weight    = (volatile const int8_t *) ddr_ptr(weight_off, VPU_QK8_0);
    int                           act_first_bad    = -1;
    int                           weight_first_bad = -1;
    int                           act_expected     = 0;
    int                           act_actual       = 0;
    int                           weight_expected  = 0;
    int                           weight_actual    = 0;
    for (int i = 0; i < VPU_QK8_0; ++i) {
        const int got_act    = (int) staged_act[i];
        const int got_weight = (int) staged_weight[i];
        if (act_first_bad < 0 && got_act != (int) act->qs[i]) {
            act_first_bad = i;
            act_expected  = (int) act->qs[i];
            act_actual    = got_act;
        }
        if (weight_first_bad < 0 && got_weight != (int) weight->qs[i]) {
            weight_first_bad = i;
            weight_expected  = (int) weight->qs[i];
            weight_actual    = got_weight;
        }
    }
    const bool intact = act_first_bad < 0 && weight_first_bad < 0;
    fpga_log_line(true, intact ? "INFO" : "ERROR", !intact,
                  "CONTRACT_STAGING_AUDIT phase=%s integrity=%s tensor=%s layer=%d tile=%u local_row=%d group_block=%d "
                  "group_beats=%d act_off=0x%08x weight_off=0x%08x weight_src_off=0x%08x act_first_bad=%d "
                  "act_expected=%d act_actual=%d weight_first_bad=%d weight_expected=%d weight_actual=%d "
                  "act_d_bits=0x%04x weight_d_bits=0x%04x status=0x%08x progress=0x%08x",
                  phase ? phase : "post_result", intact ? "pass" : "fail", tensor_name ? tensor_name : "?", layer_id,
                  tile_id, local_row, group_block, group_beats, act_off, weight_off, weight_src_off, act_first_bad,
                  act_expected, act_actual, weight_first_bad, weight_expected, weight_actual, (unsigned) act->d,
                  (unsigned) weight->d, vpu_rd32(REG_STATUS), vpu_rd32(REG_PROGRESS));
    return intact;
}

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
                                                    fpga_raw_mismatch_location_t * first_mismatch) {
    long long mismatches = 0;
    for (int row = 0; row < rows; ++row) {
        for (int gb = 0; gb < group_blocks; ++gb) {
            const block_q8_0_t * wb          = &weight_snapshot[(size_t) row * (size_t) group_blocks + (size_t) gb];
            const int32_t        expected    = q8_0_raw_dot(act_group[gb].qs, wb->qs);
            const size_t         partial_idx = (size_t) row * (size_t) group_blocks + (size_t) gb;
            const int32_t        got         = partial[partial_idx];
            if (got != expected) {
                if (first_mismatch && !first_mismatch->valid) {
                    first_mismatch->valid       = true;
                    first_mismatch->local_row   = row;
                    first_mismatch->group_block = gb;
                    first_mismatch->global_row  = row0 + row;
                    first_mismatch->k_block     = k_block0 + gb;
                }
                if (log_mismatches && mismatches < 4) {
                    LOGE(
                        "CONTRACT_RAW_MISMATCH tensor=%s layer=%d tile=%u attempt=%d row=%lld block=%lld got=%d "
                        "expected=%d act_d=%.9g weight_d=%.9g",
                        tensor_name ? tensor_name : "?", layer_id, tile_id, attempt, (long long) (row0 + row),
                        (long long) (k_block0 + gb), got, expected, fp16_to_fp32(act_group[gb].d), fp16_to_fp32(wb->d));
                    fpga_contract_log_staging_audit(&act_group[gb], wb, row, gb, group_blocks * VPU_BLOCK_BEATS,
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

static void fpga_contract_forensic_replay(const block_q8_0_t *                 weight_snapshot,
                                          const block_q8_0_t *                 act_group,
                                          int                                  rows,
                                          int                                  group_blocks,
                                          uint32_t                             weight_src_off,
                                          bool                                 weight_cache_hit,
                                          uint32_t                             tile_id,
                                          const char *                         tensor_name,
                                          int                                  layer_id,
                                          const fpga_raw_mismatch_location_t & mismatch) {
    if (!mismatch.valid) {
        return;
    }

    const int      group_beats   = group_blocks * VPU_BLOCK_BEATS;
    const size_t   act_bytes     = (size_t) group_beats * 16U;
    const size_t   weight_bytes  = weight_window_bytes_for_rows(rows, group_beats);
    const uint32_t result_values = (uint32_t) rows * (uint32_t) group_blocks;
    const uint32_t result_words =
        (result_values + (uint32_t) VPU_RESULT_PACK_LANES - 1U) / (uint32_t) VPU_RESULT_PACK_LANES;
    const size_t               result_bytes = (size_t) result_words * 16U;
    const block_q8_0_t * const weight =
        &weight_snapshot[(size_t) mismatch.local_row * (size_t) group_blocks + (size_t) mismatch.group_block];
    const block_q8_0_t * const act = &act_group[mismatch.group_block];

    LOGE(
        "CONTRACT_FORENSIC_BEGIN tensor=%s layer=%d tile=%u row=%lld local_row=%d block=%lld group_block=%d "
        "cache_hit=%d",
        tensor_name ? tensor_name : "?", layer_id, tile_id, (long long) mismatch.global_row, mismatch.local_row,
        (long long) mismatch.k_block, mismatch.group_block, weight_cache_hit ? 1 : 0);

    // Scratch weights are overwritten from the immutable GGUF source.  A
    // cache hit remains read-only by design; probing it still identifies a
    // corrupt cache payload without touching the large cache range.
    fpga_stage_q8_group_payload(weight_snapshot, act_group, rows, group_blocks, !weight_cache_hit, weight_src_off);
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row, mismatch.group_block, group_beats, weight_src_off,
                                    tensor_name, layer_id, tile_id, "forensic_after_restage");

    vpu_select_banks(0, 0);
    vpu_wr32(REG_CTRL, CTRL_CLEAR_DONE);
    configure_vpu(rows, group_beats, VPU_MODE_PACKED_Q8 | VPU_MODE_P2_TWO_ROW);

    if (!fpga_dma_write_to_ip(ACT_BASE, act_bytes, "FORENSIC_ACT")) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=act_dma tensor=%s layer=%d tile=%u", tensor_name ? tensor_name : "?",
             layer_id, tile_id);
        return;
    }
    fpga_ip_dma_readback_fence();
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row, mismatch.group_block, group_beats, weight_src_off,
                                    tensor_name, layer_id, tile_id, "forensic_after_act_dma");

    if (!fpga_dma_copy(DDR_BASE_PHYS + (uint64_t) weight_src_off, LMM_BASE_PHYS + (uint64_t) WEIGHT_BASE, weight_bytes,
                       "FORENSIC_WEIGHT")) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=weight_dma tensor=%s layer=%d tile=%u", tensor_name ? tensor_name : "?",
             layer_id, tile_id);
        return;
    }
    fpga_ip_dma_readback_fence();
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row, mismatch.group_block, group_beats, weight_src_off,
                                    tensor_name, layer_id, tile_id, "forensic_after_weight_dma");

    vpu_wr32(REG_CTRL, CTRL_START);
    mmio_fence();
    uint32_t vpu_status = 0;
    if (!wait_vpu_done(&vpu_status)) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=vpu_wait tensor=%s layer=%d tile=%u status=0x%08x progress=0x%08x",
             tensor_name ? tensor_name : "?", layer_id, tile_id, vpu_status, vpu_rd32(REG_PROGRESS));
        return;
    }
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row, mismatch.group_block, group_beats, weight_src_off,
                                    tensor_name, layer_id, tile_id, "forensic_after_vpu");

    vpu_select_banks(0, 0);
    if (!fpga_dma_read_from_ip(RESULT_BASE, result_bytes, "FORENSIC_RESULT")) {
        LOGE("CONTRACT_FORENSIC_FAIL stage=result_dma tensor=%s layer=%d tile=%u", tensor_name ? tensor_name : "?",
             layer_id, tile_id);
        return;
    }
    fpga_contract_log_staging_audit(act, weight, mismatch.local_row, mismatch.group_block, group_beats, weight_src_off,
                                    tensor_name, layer_id, tile_id, "forensic_after_result_dma");

    const uint32_t raw_index =
        (uint32_t) mismatch.local_row * (uint32_t) group_blocks + (uint32_t) mismatch.group_block;
    int32_t lanes[VPU_RESULT_PACK_LANES] = {};
    read_result_i32x4_from_ddr(raw_index / (uint32_t) VPU_RESULT_PACK_LANES, lanes);
    const int32_t got      = lanes[raw_index % (uint32_t) VPU_RESULT_PACK_LANES];
    const int32_t expected = q8_0_raw_dot(act->qs, weight->qs);
    LOGE(
        "CONTRACT_FORENSIC_RAW tensor=%s layer=%d tile=%u row=%lld block=%lld got=%d expected=%d status=0x%08x "
        "progress=0x%08x",
        tensor_name ? tensor_name : "?", layer_id, tile_id, (long long) mismatch.global_row,
        (long long) mismatch.k_block, got, expected, vpu_status, vpu_rd32(REG_PROGRESS));
}

static bool fpga_contract_check_output_values(const struct ggml_tensor *        src0,
                                              const struct ggml_tensor *        dst,
                                              const std::vector<block_q8_0_t> & act_blocks_all,
                                              const void *                      weight_data_base,
                                              const char *                      tensor_name,
                                              int                               layer_id) {
    const int64_t k              = src0->ne[0];
    const int64_t n              = src0->ne[1];
    const int64_t m              = dst->ne[1];
    const int64_t nb             = k / VPU_QK8_0;
    long long     bad            = 0;
    long long     nonfinite      = 0;
    double        max_abs        = 0.0;
    double        max_rel        = 0.0;
    const size_t  value_count    = (size_t) n * (size_t) m;
    const bool    cpu_shadow_dst = g_contract_cpu_shadow_dst;
    if (cpu_shadow_dst && g_scratch.contract_actual.size() != value_count) {
        LOGE("CONTRACT_CPU_SHADOW_LAYOUT tensor=%s layer=%d actual_values=%zu expected_values=%zu",
             tensor_name ? tensor_name : "?", layer_id, g_scratch.contract_actual.size(), value_count);
        return false;
    }

    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < n; ++row) {
            float                ref    = 0.0f;
            const block_q8_0_t * act    = &act_blocks_all[(size_t) (col * nb)];
            const block_q8_0_t * weight = weight_block_from_base(src0, weight_data_base, row, 0);
            ggml_vec_dot_q8_0_q8_0((int) k, &ref, 0, weight, 0, act, 0, 1);
            const size_t value_index = (size_t) col * (size_t) n + (size_t) row;
            const double got         = cpu_shadow_dst ? (double) g_scratch.contract_actual[value_index] :
                                                        (double) load_dst_value(dst, row, col);
            const double expected    = (double) ref;

            if (!std::isfinite(got) || !std::isfinite(expected)) {
                if (bad < 4) {
                    LOGE(
                        "CONTRACT_VALUE_NONFINITE tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g; "
                        "matching NaN/Inf is a correctness failure",
                        tensor_name ? tensor_name : "?", layer_id, (long long) row, (long long) col, got, expected);
                    fpga_contract_log_q8_nonfinite_provenance(src0, weight, act, row, col, tensor_name, layer_id, ref);
                }
                nonfinite++;
                bad++;
                continue;
            }

            const double abs_err = std::fabs(got - expected);
            const double rel_err = abs_err / (std::fabs(expected) + 1.0e-12);
            max_abs              = std::max(max_abs, abs_err);
            max_rel              = std::max(max_rel, rel_err);
            if (abs_err > g_contract_atol && rel_err > g_contract_rtol) {
                if (bad < 4) {
                    LOGE(
                        "CONTRACT_VALUE_MISMATCH tensor=%s layer=%d row=%lld col=%lld got=%.9g expected=%.9g abs=%.9g "
                        "rel=%.9g",
                        tensor_name ? tensor_name : "?", layer_id, (long long) row, (long long) col, got, expected,
                        abs_err, rel_err);
                }
                bad++;
            }
        }
    }

    if (bad > 0) {
        g_contract_value_mismatches += bad;
        LOGE(
            "CONTRACT_VALUE_SUMMARY tensor=%s layer=%d checked=%lld bad=%lld nonfinite=%lld max_abs=%.9g max_rel=%.9g "
            "atol=%.9g rtol=%.9g action=%s",
            tensor_name ? tensor_name : "?", layer_id, (long long) (n * m), bad, nonfinite, max_abs, max_rel,
            g_contract_atol, g_contract_rtol, g_contract_check_abort ? "abort" : "log_only");
        return !g_contract_check_abort;
    }

    LOGI(
        "CONTRACT_VALUE_PASS tensor=%s layer=%d checked=%lld nonfinite=0 max_abs=%.9g max_rel=%.9g "
        "reference=ggml_vec_dot_q8_0_q8_0",
        tensor_name ? tensor_name : "?", layer_id, (long long) (n * m), max_abs, max_rel);
    if (cpu_shadow_dst) {
        // ggml-cpu.c receives FPGA_MATMUL_CONTRACT_CPU_SHADOW and continues
        // into its upstream threaded kernel.  Do not write dst here: doing so
        // would race that kernel and would replace its output with a second
        // implementation during a contract run.
        g_contract_cpu_shadow_dst_values += (long long) value_count;
        LOGI(
            "CONTRACT_CPU_SHADOW_DST tensor=%s layer=%d values=%zu hardware_result=validated native_ggml_dst=deferred "
            "purpose=contract_isolation_not_cpu_fallback",
            tensor_name ? tensor_name : "?", layer_id, value_count);
    }
    return true;
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
            "[FPGA][INFO] P2_TILE_BOUNDARY status=pass tensor=%s layer=%d job=%u tile=%u bank=%d tile_limit=%d "
            "p2_tile_q16_checks=%lld p2_matrix_contract_checks=%lld matrix_value_contract=not_attempted "
            "zdma_ctrl2=0x%08x stream_status=0x%08x slot_state=0x%08x desc_flags=0x%08x reg_bank=0x%08x "
            "bank_stat=0x%08x active_job=0x%08x done_job=0x%08x action=cpu_shadow_current_matmul_then_cpu_native\n",
            tensor_name ? tensor_name : "?", layer_id, job.job_id, job.tile_id, job.bank, g_p2_tile_limit,
            g_p2_tile_q16_checks, g_p2_matrix_contract_checks, zdma_ctrl2, stream_status, slot_state, desc_flags, bank,
            bank_stat, active_job, done_job);
    fflush(fp);
    fprintf(stderr,
            "[FPGA][P2_TILE_BOUNDARY] status=pass tensor=%s layer=%d job=%u tile=%u bank=%d tile_limit=%d "
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


// ============================================================================
// WEIGHT PATH BENCHMARK
// ============================================================================

static bool fpga_weight_path_bench_pack_cached(const fpga_weight_path_bench_job_t & job,
                                                std::vector<uint32_t> &            cached_words) {
    if (!job.src0 || !job.weight_data_base || job.payload_bytes == 0U || (job.payload_bytes & 0xFU) != 0U ||
        cached_words.size() < job.payload_bytes / sizeof(uint32_t)) {
        return false;
    }
    size_t written_words = 0U;
    return fpga_pack_direct_weight_pair_range((volatile uint32_t *) cached_words.data(), job.src0, job.weight_data_base,
                                              job.row0, job.k_block0, job.rows, job.group_blocks, job.group_beats, 0U,
                                              ((size_t) job.rows + 1U) / 2U, &written_words) &&
           written_words == job.payload_bytes / sizeof(uint32_t);
}

static bool fpga_weight_path_bench_revalidate_job(const fpga_weight_path_bench_job_t & job) {
    if (!job.src0 || job.src0->type != GGML_TYPE_Q8_0 || job.src0->data != job.weight_data_base ||
        job.src0->ne[0] != job.source_k || job.src0->ne[1] != job.source_n || job.src0->nb[0] != job.source_nb0 ||
        job.src0->nb[1] != job.source_nb1 || ggml_nbytes(job.src0) != job.source_span_bytes || job.source_k <= 0 ||
        job.source_n <= 0 || job.source_k % VPU_QK8_0 != 0 || job.row0 < 0 || job.rows <= 0 ||
        job.row0 > job.source_n - job.rows || job.k_block0 < 0 || job.group_blocks <= 0 ||
        job.k_block0 > job.source_k / VPU_QK8_0 - job.group_blocks || job.group_beats != job.group_blocks * VPU_BLOCK_BEATS) {
        return false;
    }
    size_t payload_bytes = 0U;
    return fpga_weight_layout_payload_bytes(job.rows, job.group_beats, &payload_bytes) &&
           payload_bytes == job.payload_bytes && range_fits(WEIGHT_BASE, payload_bytes, WEIGHT_BASE, WEIGHT_END) &&
           ddr_range_fits(WEIGHT_BASE, payload_bytes);
}

static bool fpga_weight_path_bench_store_uio(const std::vector<uint32_t> & cached_words, size_t bytes) {
    if (bytes == 0U || (bytes & 0xFU) != 0U || cached_words.size() < bytes / sizeof(uint32_t) ||
        !range_fits(WEIGHT_BASE, bytes, WEIGHT_BASE, WEIGHT_END) || !ddr_range_fits(WEIGHT_BASE, bytes)) {
        return false;
    }
    volatile uint32_t * const dst = ddr_checked_u32_ptr(WEIGHT_BASE, bytes);
    for (size_t word = 0; word < bytes / sizeof(uint32_t); ++word) {
        dst[word] = cached_words[word];
    }
    // This is the same bounded volatile-write/readback commit used by normal
    // staging, but benchmark mode has no following ZDMA descriptor.
    mmio_fence();
    fpga_ddr_staging_readback_commit(WEIGHT_BASE, bytes);
    return true;
}

static bool fpga_weight_path_bench_pack_uio(const fpga_weight_path_bench_job_t & job) {
    if (!fpga_weight_path_bench_revalidate_job(job)) {
        return false;
    }
    volatile uint32_t * const dst = ddr_checked_u32_ptr(WEIGHT_BASE, job.payload_bytes);
    size_t written_words = 0U;
    if (!fpga_pack_direct_weight_pair_range(dst, job.src0, job.weight_data_base, job.row0, job.k_block0, job.rows,
                                             job.group_blocks, job.group_beats, 0U, ((size_t) job.rows + 1U) / 2U,
                                             &written_words) ||
        written_words != job.payload_bytes / sizeof(uint32_t)) {
        return false;
    }
    mmio_fence();
    fpga_ddr_staging_readback_commit(WEIGHT_BASE, job.payload_bytes);
    return true;
}

static bool fpga_weight_path_bench_reset_required(const fpga_weight_path_bench_trace_t & trace,
                                                  int                                    previous_graph_seq,
                                                  int                                    n_tokens) {
    return trace.enabled && !trace.replayed && trace.graph_seq != INT_MIN &&
           (n_tokens != 1 || trace.graph_seq != previous_graph_seq);
}

static void fpga_weight_path_bench_reset_unreplayed_trace(fpga_weight_path_bench_trace_t * trace,
                                                          int                              new_graph_seq,
                                                          const char *                     reason,
                                                          bool                             emit_log) {
    if (!trace || trace->replayed || trace->graph_seq == INT_MIN) {
        return;
    }
    const int      old_graph_seq = trace->graph_seq;
    const size_t   stale_jobs    = trace->jobs.size();
    const uint64_t stale_bytes   = trace->payload_bytes;
    trace->cached_payload.clear();
    trace->jobs.clear();
    trace->payload_bytes = 0U;
    trace->graph_seq     = new_graph_seq;
    if (emit_log) {
        LOGPROOF("WEIGHT_PATH_BENCH_CAPTURE_RESET old_seq=%d new_seq=%d stale_jobs=%zu stale_bytes=%llu reason=%s",
                 old_graph_seq, new_graph_seq, stale_jobs, (unsigned long long) stale_bytes,
                 reason ? reason : "unspecified");
    }
}

static bool fpga_weight_path_bench_prepare_capture(fpga_weight_path_bench_trace_t * trace,
                                                   int                              graph_seq,
                                                   const char **                    reason,
                                                   bool                             emit_log) {
    if (reason) {
        *reason = "state_unknown";
    }
    if (!trace || !trace->enabled) {
        if (reason) {
            *reason = "state_benchmark_disabled";
        }
        return false;
    }
    if (trace->replayed) {
        if (reason) {
            *reason = "state_trace_replayed";
        }
        return false;
    }
    if (trace->graph_seq == INT_MIN) {
        trace->graph_seq = graph_seq;
        return true;
    }
    if (trace->graph_seq != graph_seq) {
        fpga_weight_path_bench_reset_unreplayed_trace(trace, graph_seq,
                                                      "sequence_advanced_before_single_token_boundary", emit_log);
    }
    return true;
}

static bool fpga_weight_path_bench_capture(const struct ggml_tensor * src0, int graph_seq, const char ** reason) {
    if (reason) {
        *reason = "source_unknown";
    }
    if (!src0 || src0->type != GGML_TYPE_Q8_0 || !src0->data || src0->ne[0] <= 0 || src0->ne[1] <= 0 ||
        src0->ne[0] % VPU_QK8_0 != 0) {
        if (reason) {
            *reason = "source_invalid_q8_tensor";
        }
        return false;
    }
    if (!fpga_weight_path_bench_prepare_capture(&g_weight_path_bench, graph_seq, reason, true)) {
        return false;
    }
    const int64_t k           = src0->ne[0];
    const int64_t n           = src0->ne[1];
    const int64_t nb          = k / VPU_QK8_0;
    const size_t  source_span = ggml_nbytes(src0);
    for (int64_t row0 = 0; row0 < n; row0 += g_vpu_max_rows) {
        const int rows = (int) std::min<int64_t>(g_vpu_max_rows, n - row0);
        for (int64_t k_block0 = 0; k_block0 < nb;) {
            const int group_blocks  = packed_q8_group_blocks_for_rows(rows, (int) (nb - k_block0));
            const int group_beats   = group_blocks * VPU_BLOCK_BEATS;
            size_t    payload_bytes = 0U;
            if (group_blocks <= 0 || group_beats <= 0 ||
                !fpga_weight_layout_payload_bytes(rows, group_beats, &payload_bytes)) {
                if (reason) {
                    *reason = "geometry_invalid_tile";
                }
                return false;
            }
            if (!range_fits(WEIGHT_BASE, payload_bytes, WEIGHT_BASE, WEIGHT_END) ||
                !ddr_range_fits(WEIGHT_BASE, payload_bytes)) {
                if (reason) {
                    *reason = "range_staging_window";
                }
                return false;
            }
            if (payload_bytes > UINT64_MAX - g_weight_path_bench.payload_bytes) {
                if (reason) {
                    *reason = "state_payload_counter_overflow";
                }
                return false;
            }
            try {
                g_weight_path_bench.jobs.push_back({ src0, src0->data, k, n, src0->nb[0], src0->nb[1], source_span,
                                                     row0, rows, k_block0, group_blocks, group_beats, payload_bytes });
            } catch (...) {
                if (reason) {
                    *reason = "state_job_allocation_failure";
                }
                return false;
            }
            g_weight_path_bench.payload_bytes += payload_bytes;
            k_block0 += group_blocks;
        }
    }
    return true;
}

static bool fpga_weight_path_bench_stats(const std::array<long long, 5> & samples,
                                         long long *                       min_us,
                                         long long *                       median_us,
                                         long long *                       max_us) {
    if (!min_us || !median_us || !max_us) {
        return false;
    }
    std::array<long long, 5> sorted = samples;
    for (long long sample : sorted) {
        if (sample < 0) return false;
    }
    std::sort(sorted.begin(), sorted.end());
    *min_us = sorted.front();
    *median_us = sorted[sorted.size() / 2U];
    *max_us = sorted.back();
    return true;
}

static bool fpga_weight_path_bench_replay_phase(fpga_weight_path_bench_phase phase,
                                                 bool                         store_timing_only,
                                                 long long *                  store_elapsed_us) {
    if (store_timing_only && (phase != fpga_weight_path_bench_phase::STORE_UIO || !store_elapsed_us)) {
        return false;
    }
    if (store_elapsed_us) {
        *store_elapsed_us = 0;
    }
    for (size_t i = 0; i < g_weight_path_bench.jobs.size(); ++i) {
        const fpga_weight_path_bench_job_t & job = g_weight_path_bench.jobs[i];
        if (!fpga_weight_path_bench_revalidate_job(job)) {
            return false;
        }
        bool ok = false;
        if (phase == fpga_weight_path_bench_phase::PACK_CACHED) {
            ok = fpga_weight_path_bench_pack_cached(job, g_weight_path_bench.cached_payload);
        } else if (phase == fpga_weight_path_bench_phase::STORE_UIO) {
            // T_store_uio must not include transform time.  Repack this exact
            // job into the bounded reusable cache before starting its store
            // interval; sum those intervals into one full-workload replay.
            if (!fpga_weight_path_bench_pack_cached(job, g_weight_path_bench.cached_payload)) {
                return false;
            }
            const long long store_start = store_timing_only ? monotonic_now_us() : 0;
            ok = fpga_weight_path_bench_store_uio(g_weight_path_bench.cached_payload, job.payload_bytes);
            if (store_timing_only) {
                const long long store_end = monotonic_now_us();
                if (store_end < store_start || *store_elapsed_us > LLONG_MAX - (store_end - store_start)) {
                    return false;
                }
                *store_elapsed_us += store_end - store_start;
            }
        } else {
            ok = fpga_weight_path_bench_pack_uio(job);
        }
        if (!ok) return false;
    }
    return true;
}

static bool fpga_weight_path_bench_emit_phase(const char * name, fpga_weight_path_bench_phase phase) {
    if (!fpga_weight_path_bench_replay_phase(phase, false, nullptr)) {
        return false; // Untimed warm-up.
    }
    std::array<long long, 5> samples = {};
    for (size_t replay = 0; replay < samples.size(); ++replay) {
        if (phase == fpga_weight_path_bench_phase::STORE_UIO) {
            if (!fpga_weight_path_bench_replay_phase(phase, true, &samples[replay])) {
                return false;
            }
            continue;
        }
        const long long start = monotonic_now_us();
        if (!fpga_weight_path_bench_replay_phase(phase, false, nullptr)) {
            return false;
        }
        const long long end = monotonic_now_us();
        if (end < start) return false;
        samples[replay] = end - start;
    }
    long long min_us = 0;
    long long median_us = 0;
    long long max_us = 0;
    if (!fpga_weight_path_bench_stats(samples, &min_us, &median_us, &max_us)) {
        return false;
    }
    const double mib_s = median_us > 0 ? (double) g_weight_path_bench.payload_bytes * 1000000.0 /
                                             ((double) median_us * 1024.0 * 1024.0) :
                                         0.0;
    LOGPROOF("WEIGHT_PATH_BENCH name=%s warmup=1 timed_replays=5 timing_scope=%s median_us=%lld min_us=%lld max_us=%lld "
             "MiB_s=%.3f bytes=%llu jobs=%zu route=host_only_no_zdma_no_vpu_no_dst",
             name, phase == fpga_weight_path_bench_phase::STORE_UIO ? "sum_per_job_store_intervals_prepack_excluded" :
                                                                       "full_replay_elapsed",
             median_us, min_us, max_us, mib_s, (unsigned long long) g_weight_path_bench.payload_bytes,
             g_weight_path_bench.jobs.size());
    return true;
}

static bool fpga_weight_path_bench_replay_at_boundary(int previous_graph_seq, int n_tokens) {
    if (!g_weight_path_bench.enabled || g_weight_path_bench.replayed || n_tokens != 1 ||
        g_weight_path_bench.graph_seq != previous_graph_seq) {
        return true;
    }
    if (g_weight_path_bench.jobs.empty()) {
        return false;
    }
    try {
        size_t max_payload_bytes = 0U;
        for (const fpga_weight_path_bench_job_t & job : g_weight_path_bench.jobs) {
            max_payload_bytes = std::max(max_payload_bytes, job.payload_bytes);
        }
        if (max_payload_bytes == 0U || (max_payload_bytes & 0xFU) != 0U ||
            max_payload_bytes > WEIGHT_END - WEIGHT_BASE) {
            return false;
        }
        // One tile-sized cache avoids retaining every replay payload (which
        // would scale with the complete model and can exhaust host RAM).
        g_weight_path_bench.cached_payload.assign(max_payload_bytes / sizeof(uint32_t), 0U);
        LOGPROOF("WEIGHT_PATH_BENCH_REPLAY_BUFFER bytes=%zu scope=max_single_captured_tile aggregate_payload_bytes=%llu "
                 "jobs=%zu allocation=bounded_reusable",
                 max_payload_bytes, (unsigned long long) g_weight_path_bench.payload_bytes,
                 g_weight_path_bench.jobs.size());
    } catch (...) {
        return false;
    }
    const bool ok = fpga_weight_path_bench_emit_phase("T_pack_cached", fpga_weight_path_bench_phase::PACK_CACHED) &&
                    fpga_weight_path_bench_emit_phase("T_store_uio", fpga_weight_path_bench_phase::STORE_UIO) &&
                    fpga_weight_path_bench_emit_phase("T_pack_uio", fpga_weight_path_bench_phase::PACK_UIO);
    g_weight_path_bench.cached_payload.clear();
    g_weight_path_bench.jobs.clear();
    g_weight_path_bench.payload_bytes = 0U;
    g_weight_path_bench.replayed      = true;
    return ok;
}
