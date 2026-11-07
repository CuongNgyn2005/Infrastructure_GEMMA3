#include "fpga_log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ctime>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>

namespace {

constexpr const char * FPGA_LOG_FILE = "/tmp/fpga_debug.log";

int g_flush_every   = 256;
int g_pending_lines = 0;

} // namespace

FILE * fpga_log_fp() {
    static FILE * fp = nullptr;
    if (!fp) {
        // Opening an existing file with fopen("a") includes O_CREAT. Linux
        // fs.protected_regular can reject that operation in sticky /tmp when
        // root runs llama-cli against a log owned by the debian user. Open the
        // existing file without O_CREAT first, then create it only if absent.
        int fd = open(FPGA_LOG_FILE, O_WRONLY | O_APPEND | O_CLOEXEC);
        if (fd < 0 && errno == ENOENT) {
            fd = open(FPGA_LOG_FILE, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
        }
        if (fd >= 0) {
            fp = fdopen(fd, "a");
            if (!fp) {
                close(fd);
            }
        }
        if (!fp) {
            const int open_errno = errno;
            fprintf(stderr,
                    "[ERROR] cannot open %s; detailed FPGA telemetry is disabled: errno=%d (%s)\n",
                    FPGA_LOG_FILE, open_errno, strerror(open_errno));
            fp = fopen("/dev/null", "w");
            if (!fp) {
                fp = stderr;
            }
        }

        const time_t now = time(nullptr);
        fprintf(fp, "\n============================================================\n");
        fprintf(fp, "ZDMA DDR-to-IP log started at %ld\n", (long) now);
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
    fprintf(fp, "[%s] ", tag ? tag : "INFO");
    vfprintf(fp, fmt, ap);
    fputc('\n', fp);
    fpga_log_finish_line(fp, force_flush);
}

void fpga_log_latency(const char * fmt, ...) {
    if (fpga_log_fp() == stderr) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fpga_log_vline("LATENCY", true, fmt, ap);
    va_end(ap);
}

void fpga_log_result_overlap(int graph_seq, bool enabled, long long jobs, long long host_us) {
    if (fpga_log_fp() == stderr) {
        return;
    }
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[RESULT_OVERLAP] graph_seq=%d enabled=%d jobs=%lld host_after_next_launch_ms=%.3f "
                "scope=decode_token actual_hidden_time=not_measured\n",
            graph_seq, enabled ? 1 : 0, jobs, (double) host_us / 1000.0);
    fpga_log_finish_line(fp, false);
}

void fpga_log_q16_audit(const char * fmt, ...) {
    if (fpga_log_fp() == stderr) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fpga_log_vline("Q16_AUDIT", true, fmt, ap);
    va_end(ap);
}

void fpga_log_pack_breakdown(const char * scope, int graph_seq, long long tokens,
                             const fpga_pack_breakdown_log_t & data) {
    const long long other_us = data.total_us - data.serial_pack_us - data.dispatch_us -
        data.main_pack_us - data.caller_fence_us - data.caller_wait_us;
    const long long dispatch_other_us = data.dispatch_us - data.reserve_lock_us - data.reserve_body_us -
        data.reserve_unlock_us - data.submit_lock_us - data.submit_publish_us -
        data.submit_signal_us - data.submit_unlock_us;
    fpga_log_latency(
        "P2_PACK_BREAKDOWN scope=%s graph_seq=%d tokens=%lld total_us=%lld serial_pack_us=%lld "
        "dispatch_us=%lld main_pack_us=%lld caller_fence_us=%lld caller_wait_us=%lld "
        "other_us=%lld helper_service_overlap_us=%lld serial_jobs=%lld parallel_jobs=%lld "
        "reserve_lock_us=%lld reserve_body_us=%lld reserve_unlock_us=%lld "
        "submit_lock_us=%lld submit_publish_us=%lld submit_signal_us=%lld submit_unlock_us=%lld "
        "dispatch_other_us=%lld dispatch_detail=subsets_of_dispatch_wall_time submit_notify_order=unlock_then_signal "
        "measurement=instrumented_successful_direct_preparation",
        scope, graph_seq, tokens, data.total_us, data.serial_pack_us, data.dispatch_us,
        data.main_pack_us, data.caller_fence_us, data.caller_wait_us, other_us, data.helper_service_us,
        data.serial_jobs, data.parallel_jobs, data.reserve_lock_us, data.reserve_body_us,
        data.reserve_unlock_us, data.submit_lock_us, data.submit_publish_us, data.submit_signal_us,
        data.submit_unlock_us, dispatch_other_us);
}

#ifdef USE_FPGA
void fpga_log_ram_bandwidth() {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) {
        return;
    }
    // Two 32 MiB buffers exceed the ZCU104 CPU cache capacity. Touch every
    // page before timing, so allocation and initial page faults are excluded.
    // This is logical memcpy throughput, not a DDR-controller traffic counter.
    constexpr size_t bytes = 32u * 1024u * 1024u;
    constexpr unsigned passes = 8;
    auto * src = static_cast<unsigned char *>(std::malloc(bytes));
    auto * dst = static_cast<unsigned char *>(std::malloc(bytes));
    if (!src || !dst) {
        std::free(src);
        std::free(dst);
        fprintf(fp, "[RAM_BANDWIDTH] status=skipped reason=allocation_failed allocation_bytes=%zu\n", 2 * bytes);
        fpga_log_finish_line(fp, true);
        return;
    }
    for (size_t i = 0; i < bytes; ++i) {
        src[i] = static_cast<unsigned char>((i * 31u + (i >> 8)) & 255u);
    }
    std::memset(dst, 0, bytes);
    // A volatile function pointer preserves every copy even with optimization
    // and LTO. Use the platform's libc memcpy, on one calling CPU thread.
    void * (*volatile copy_fn)(void *, const void *, size_t) = std::memcpy;
    copy_fn(dst, src, bytes); // Untimed warm-up, including destination faults.
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < passes; ++i) {
        copy_fn(dst, src, bytes);
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const bool verified = std::memcmp(src, dst, bytes) == 0;
    std::free(src);
    std::free(dst);
    if (!verified || seconds <= 0.0) {
        fprintf(fp, "[RAM_BANDWIDTH] status=failed reason=%s\n",
                verified ? "invalid_duration" : "copy_mismatch");
    } else {
        const size_t copied_bytes = bytes * passes;
        const double copy_gb_s = static_cast<double>(copied_bytes) / seconds / 1e9;
        fprintf(fp, "[RAM_BANDWIDTH] status=ok method=libc_memcpy threads=1 "
                    "buffer_bytes=%zu passes=%u copied_bytes=%zu elapsed_ms=%.6f "
                    "copy_GB_s=%.6f logical_read_write_GB_s=%.6f verified=1 "
                    "scope=startup_cpu_cached_memory physical_ddr_traffic=not_measured "
                    "allocation_warmup_verification=excluded units=decimal_GB\n",
                bytes, passes, copied_bytes, seconds * 1000.0, copy_gb_s, 2.0 * copy_gb_s);
    }
    fpga_log_finish_line(fp, true);
}

void fpga_log_load_checkpoint(const char * phase) {
    // Startup-only snapshots. Counters are process cumulative; subtract
    // adjacent records for a phase. No per-tensor/per-token instrumentation.
    const auto now = std::chrono::steady_clock::now();
    static const auto origin = now;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(now - origin).count();
    struct rusage usage = {};
    const bool usage_ok = getrusage(RUSAGE_SELF, &usage) == 0;
    const double user_ms = usage_ok ? usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0 : -1;
    const double system_ms = usage_ok ? usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0 : -1;
    long long read_bytes = -1;
    FILE * io = fopen("/proc/self/io", "r");
    if (io) {
        char line[128];
        while (fgets(line, sizeof(line), io)) {
            if (sscanf(line, "read_bytes: %lld", &read_bytes) == 1) break;
        }
        fclose(io);
    }
    fpga_log_latency("[MODEL_LOAD_DETAIL] phase=%s elapsed_ms=%.3f user_cpu_ms=%.3f system_cpu_ms=%.3f "
                     "minor_faults=%ld major_faults=%ld block_inputs=%ld read_bytes=%lld "
                     "max_rss_kib=%ld counters=process_cumulative unavailable=-1",
                     phase, elapsed_ms, user_ms, system_ms,
                     usage_ok ? usage.ru_minflt : -1L, usage_ok ? usage.ru_majflt : -1L,
                     usage_ok ? usage.ru_inblock : -1L, read_bytes, usage_ok ? usage.ru_maxrss : -1L);
}
#endif

#ifdef USE_FPGA

void fpga_log_runtime_summary(double load_ms, double prompt_ms, int prompt_tokens,
                              double decode_ms, int decode_runs) {
    fpga_log_latency("RUNTIME_SUMMARY load_ms=%.2f prompt_tokens=%d prompt_ms=%.2f prompt_tokens_s=%.2f "
                     "decode_runs=%d decode_ms=%.2f decode_tokens_s=%.2f",
                     load_ms, prompt_tokens, prompt_ms, prompt_ms > 0 ? 1000.0 * prompt_tokens / prompt_ms : 0.0,
                     decode_runs, decode_ms, decode_ms > 0 ? 1000.0 * decode_runs / decode_ms : 0.0);
}
#endif

void fpga_log_prompt_weight_reuse(const char * tensor, long long columns, long long jobs,
                                  unsigned long long avoided_bytes) {
    fpga_log_latency("PROMPT_WEIGHT_BANK_REUSE tensor=%s columns=%lld reused_jobs=%lld "
                     "avoided_weight_dma_bytes=%llu scope=completed_matmul",
                     tensor ? tensor : "?", columns, jobs, avoided_bytes);
}

namespace {
// Serialized by the host's existing submission/lifecycle mutex. No worker,
// watcher, allocation, lock, MMIO access, or per-descriptor output is added.
struct dma_profile_row {
    unsigned long long descriptors = 0, bytes = 0, failures = 0, failed_bytes = 0;
    long long preparation_us = 0, launch_to_observed_us = 0, callback_pack_us = 0;
    unsigned long long callback_pack_count = 0, callback_pack_bytes = 0;
    long long noncallback_launch_interval_us = 0, cleanup_us = 0;
    long long failed_elapsed_us = 0, failed_callback_pack_us = 0;
    unsigned long long timing_invalid = 0, completion_observed = 0, channel_enabled = 0;
    unsigned long long incomplete_en_clear = 0, error = 0;
    long long post_pack_observation_gap_us = 0;
};
struct dma_profile_bucket {
    dma_profile_row rows[5];
    unsigned long long finalize_count = 0, residual_bytes = 0, finalize_timing_invalid = 0;
    long long residual_us = 0, helper_wait_us = 0, helper_service_overlap_us = 0, finalize_us = 0;
};
struct dma_profile_transfer_state {
    bool active = false, launched = false, observed = false, pending_pack = false, invalid = false;
    int tag = 4;
    size_t bytes = 0;
    long long entry = -1, launch = -1, observed_at = -1, pack_begin = -1, pack_end = -1;
    long long attempted_pack_us = 0;
    dma_profile_row row;
};
bool dma_profile_on = false, dma_profile_finished = true, dma_profile_graph_active = false;
int dma_profile_graph_seq = -1;
unsigned long long dma_profile_decode_graphs = 0;
long long dma_profile_finalize_start = -1;
dma_profile_bucket dma_profile_process, dma_profile_graph, dma_profile_decode;
dma_profile_transfer_state dma_profile_transfer_state_current;

// A graph-local partition of software observations, not hardware bus activity.
// Descriptor hooks reuse their existing timestamps; only graph boundaries read
// an additional clock. Aggregates sum windows, never the gaps between graphs.
struct dma_profile_timeline {
    long long begin = -1, end = -1, last_observed = -1, last_exit = -1;
    long long window_us = 0, head_us = 0, service_us = 0, gap_us = 0, tail_us = 0, no_dma_us = 0;
    long long descriptors = 0, gap_count = 0, max_gap_us = 0;
    long long valid_graphs = 0, excluded_graphs = 0, no_dma_graphs = 0;
    bool invalid = false;
};
dma_profile_timeline dma_timeline_graph, dma_timeline_decode, dma_timeline_all_graphs;

bool dma_timeline_add_value(long long & dst, long long value) {
    if (value < 0 || dst < 0 || value > std::numeric_limits<long long>::max() - dst) return false;
    dst += value;
    return true;
}

long long dma_profile_clock() {
    timespec ts = {};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0 || ts.tv_sec < 0 ||
        ts.tv_nsec < 0 || ts.tv_nsec >= 1000000000L ||
        static_cast<unsigned long long>(ts.tv_sec) >
            static_cast<unsigned long long>(std::numeric_limits<long long>::max() / 1000000LL)) {
        return -1;
    }
    const long long seconds_us = static_cast<long long>(ts.tv_sec) * 1000000LL;
    const long long fraction_us = ts.tv_nsec / 1000;
    if (fraction_us > std::numeric_limits<long long>::max() - seconds_us) return -1;
    return seconds_us + fraction_us;
}
bool dma_profile_interval(long long start, long long end) { return start >= 0 && end >= start; }

void dma_timeline_transfer(const dma_profile_transfer_state & t, long long end, bool success) {
    if (!dma_profile_graph_active) return;
    auto & g = dma_timeline_graph;
    if (g.invalid) return;
    if (!success || t.invalid || !t.active || !t.launched || !t.observed ||
        !dma_profile_interval(g.begin, t.entry) || !dma_profile_interval(t.entry, t.launch) ||
        !dma_profile_interval(t.launch, t.observed_at) || !dma_profile_interval(t.observed_at, end) ||
        (g.descriptors && !dma_profile_interval(g.last_exit, t.entry))) {
        g.invalid = true;
        return;
    }
    if (!g.descriptors) {
        g.head_us = t.launch - g.begin;
    } else {
        const long long gap = t.launch - g.last_observed;
        g.invalid |= !dma_timeline_add_value(g.gap_us, gap) || !dma_timeline_add_value(g.gap_count, 1);
        g.max_gap_us = std::max(g.max_gap_us, gap);
    }
    g.invalid |= !dma_timeline_add_value(g.service_us, t.observed_at - t.launch) ||
                 !dma_timeline_add_value(g.descriptors, 1);
    g.last_observed = t.observed_at;
    g.last_exit = end;
}

void dma_timeline_add(dma_profile_timeline & dst, const dma_profile_timeline & src) {
    if (dst.invalid) return;
    auto next = dst;
#define DMA_TIMELINE_ADD(field) if (!dma_timeline_add_value(next.field, src.field)) { dst.invalid = true; return; }
    DMA_TIMELINE_ADD(valid_graphs); DMA_TIMELINE_ADD(excluded_graphs); DMA_TIMELINE_ADD(no_dma_graphs);
    DMA_TIMELINE_ADD(window_us); DMA_TIMELINE_ADD(head_us); DMA_TIMELINE_ADD(service_us);
    DMA_TIMELINE_ADD(gap_us); DMA_TIMELINE_ADD(tail_us); DMA_TIMELINE_ADD(no_dma_us);
    DMA_TIMELINE_ADD(descriptors); DMA_TIMELINE_ADD(gap_count);
#undef DMA_TIMELINE_ADD
    next.max_gap_us = std::max(next.max_gap_us, src.max_gap_us);
    if (src.valid_graphs) {
        if (next.begin < 0) next.begin = src.begin;
        next.end = src.end;
    }
    dst = next;
}

void dma_timeline_emit(const dma_profile_timeline & g, const char * scope, int graph_seq, bool aggregate) {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    const bool valid = !g.invalid && g.excluded_graphs == 0;
    // Aggregate exclusions do not invalidate the sum of explicitly valid windows.
    const bool usable = !g.invalid && g.valid_graphs > 0 && (aggregate || valid);
    const auto duration = [usable](long long value) { return usable ? value : -1LL; };
    const double service_fraction = usable && g.window_us > 0 ? (double) g.service_us / g.window_us : -1.0;
    const double gap_fraction = usable && g.window_us > 0 ? (double) g.gap_us / g.window_us : -1.0;
    fprintf(fp, "[DMA_PROFILE_TIMELINE] scope=%s graph_seq=%d valid_graphs=%lld excluded_graphs=%lld "
                "no_dma_graphs=%lld aggregate_valid=%d begin_mono_us=%lld end_mono_us=%lld "
                "profile_window_us=%lld head_to_first_launch_us=%lld launch_to_done_observed_us=%lld "
                "done_observed_to_next_launch_us=%lld tail_after_last_done_observed_us=%lld no_dma_window_us=%lld "
                "descriptors=%lld gap_count=%lld max_gap_us=%lld observed_service_fraction=%.6f observed_gap_fraction=%.6f "
                "window_scope=profiler_graph_hooks cross_graph_gaps=excluded endpoints=%s "
                "hardware_completion_time=not_measured service=software_upper_envelope "
                "gap=owned_channel_idle_lower_bound sole_channel_ownership=required clock_resolution_us=1 "
                "ddr_utilization=not_measured callback_prep_cleanup=overlapping_subsets fatal_inflight=excluded\n",
            scope, graph_seq, g.valid_graphs, g.excluded_graphs, g.no_dma_graphs, usable ? 1 : 0,
            g.begin, g.end, duration(g.window_us), duration(g.head_us), duration(g.service_us), duration(g.gap_us),
            duration(g.tail_us), duration(g.no_dma_us), duration(g.descriptors), duration(g.gap_count),
            duration(g.max_gap_us), service_fraction, gap_fraction,
            aggregate ? "valid_window_envelope_not_denominator" : "window_boundaries");
    fpga_log_finish_line(fp, false);
}

void dma_profile_add(dma_profile_row & dst, const dma_profile_row & src) {
#define DMA_PROFILE_ADD(field) dst.field += src.field
    DMA_PROFILE_ADD(descriptors); DMA_PROFILE_ADD(bytes); DMA_PROFILE_ADD(failures); DMA_PROFILE_ADD(failed_bytes);
    DMA_PROFILE_ADD(preparation_us); DMA_PROFILE_ADD(launch_to_observed_us); DMA_PROFILE_ADD(callback_pack_us);
    DMA_PROFILE_ADD(callback_pack_count); DMA_PROFILE_ADD(callback_pack_bytes);
    DMA_PROFILE_ADD(noncallback_launch_interval_us); DMA_PROFILE_ADD(cleanup_us);
    DMA_PROFILE_ADD(failed_elapsed_us); DMA_PROFILE_ADD(failed_callback_pack_us); DMA_PROFILE_ADD(timing_invalid);
    DMA_PROFILE_ADD(completion_observed); DMA_PROFILE_ADD(channel_enabled); DMA_PROFILE_ADD(incomplete_en_clear);
    DMA_PROFILE_ADD(error); DMA_PROFILE_ADD(post_pack_observation_gap_us);
#undef DMA_PROFILE_ADD
}
void dma_profile_add(dma_profile_bucket & dst, const dma_profile_bucket & src) {
    for (int i = 0; i < 5; ++i) dma_profile_add(dst.rows[i], src.rows[i]);
    dst.finalize_count += src.finalize_count;
    dst.residual_bytes += src.residual_bytes;
    dst.finalize_timing_invalid += src.finalize_timing_invalid;
    dst.residual_us += src.residual_us;
    dst.helper_wait_us += src.helper_wait_us;
    dst.helper_service_overlap_us += src.helper_service_overlap_us;
    dst.finalize_us += src.finalize_us;
}
void dma_profile_emit(const dma_profile_bucket & bucket, const char * scope, int graph_seq,
                      unsigned long long graphs) {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    const char * const tags[] = {"ACT", "WEIGHT", "SPU_SCALE", "RESULT", "OTHER"};
    for (int i = 0; i < 5; ++i) {
        const auto & r = bucket.rows[i];
        if (!r.descriptors && !r.failures && !r.timing_invalid) continue;
        fprintf(fp, "[DMA_PROFILE] scope=%s graph_seq=%d decode_graphs=%llu tag=%s "
                    "descriptors=%llu bytes=%llu preparation_us=%lld launch_to_observed_us=%lld "
                    "callback_pack_us=%lld callback_pack_count=%llu callback_pack_bytes=%llu "
                    "noncallback_launch_interval_us=%lld cleanup_us=%lld failures=%llu failed_bytes=%llu "
                    "failed_elapsed_us=%lld failed_callback_pack_us=%lld timing_invalid=%llu "
                    "post_pack_completion_observed=%llu post_pack_channel_enabled=%llu "
                    "post_pack_incomplete_en_clear=%llu post_pack_error=%llu post_pack_observation_gap_us=%lld "
                    "successful_timing=valid_samples_only failed_callback_scope=all_attempted_batches "
                    "post_pack_scope=all_valid_samples callback_bytes_scope=weights_only "
                    "failure_scope=returned_failures fatal_inflight=excluded "
                    "noncallback_meaning=host_launch_interval_minus_measured_callback_pack "
                    "hardware_completion_time=not_measured contention=not_measured\n",
                scope, graph_seq, graphs, tags[i], r.descriptors, r.bytes, r.preparation_us,
                r.launch_to_observed_us, r.callback_pack_us, r.callback_pack_count, r.callback_pack_bytes,
                r.noncallback_launch_interval_us, r.cleanup_us, r.failures, r.failed_bytes,
                r.failed_elapsed_us, r.failed_callback_pack_us, r.timing_invalid, r.completion_observed,
                r.channel_enabled, r.incomplete_en_clear, r.error, r.post_pack_observation_gap_us);
        fpga_log_finish_line(fp, false);
    }
    fprintf(fp, "[DMA_PROFILE_PACK] scope=%s graph_seq=%d decode_graphs=%llu success_only=1 "
                "finalize_count=%llu residual_bytes=%llu residual_us=%lld helper_wait_us=%lld "
                "helper_service_overlap_us=%lld finalize_us=%lld timing_invalid=%llu "
                "helper_service_is_overlapping=1 residual_bytes_scope=weights_only failure_invalidation_waits=excluded "
                "finalize_wall_clock=monotonic imported_component_clock=realtime_existing "
                "components_not_guaranteed_partition=1\n",
            scope, graph_seq, graphs, bucket.finalize_count, bucket.residual_bytes, bucket.residual_us,
            bucket.helper_wait_us, bucket.helper_service_overlap_us, bucket.finalize_us,
            bucket.finalize_timing_invalid);
    fpga_log_finish_line(fp, true);
}
} // namespace

bool fpga_dma_profile_enabled() { return dma_profile_on && !dma_profile_finished; }

bool fpga_dma_profile_configure() {
    const char * value = std::getenv("FPGA_DMA_PROFILE");
    const bool valid = !value || !std::strcmp(value, "0") || !std::strcmp(value, "1");
    dma_profile_on = valid && value && !std::strcmp(value, "1");
    dma_profile_finished = false;
    dma_profile_graph_active = false;
    dma_profile_graph_seq = -1;
    dma_profile_decode_graphs = 0;
    dma_profile_process = {}; dma_profile_graph = {}; dma_profile_decode = {};
    dma_profile_transfer_state_current = {};
    dma_profile_finalize_start = -1;
    dma_timeline_graph = {}; dma_timeline_decode = {}; dma_timeline_all_graphs = {};
    FILE * fp = fpga_log_fp();
    if (fp != stderr) {
        fprintf(fp, "[DMA_PROFILE_CONFIG] enabled=%d valid=%d default=off descriptor_clock=monotonic "
                    "finalize_wall_clock=monotonic imported_component_clock=realtime_existing "
                    "output=file_only graph_collection=enabled_when_profiled final_totals_require=normal_cleanup\n",
                dma_profile_on ? 1 : 0, valid ? 1 : 0);
        fpga_log_finish_line(fp, true);
    }
    return valid;
}

fpga_dma_profile_transfer::fpga_dma_profile_transfer(const char * tag, size_t bytes) :
    enabled_(fpga_dma_profile_enabled()) {
    if (!enabled_) return;
    auto & t = dma_profile_transfer_state_current;
    if (t.active && dma_profile_graph_active) dma_timeline_graph.invalid = true;
    t = {};
    t.active = true;
    t.bytes = bytes;
    // Split descriptors retain the base tag followed by a chunk suffix.
    if (tag && !std::strncmp(tag, "ACT", 3)) t.tag = 0;
    else if (tag && !std::strncmp(tag, "WEIGHT", 6)) t.tag = 1;
    else if (tag && !std::strncmp(tag, "SPU_SCALE", 9)) t.tag = 2;
    else if (tag && (!std::strncmp(tag, "RESULT", 6) || !std::strncmp(tag, "SPU_OUT", 7))) t.tag = 3;
    t.entry = dma_profile_clock();
    t.invalid = t.entry < 0;
}
void fpga_dma_profile_transfer::launch() {
    if (!enabled_) return;
    auto & t = dma_profile_transfer_state_current;
    if ((!t.active || t.launched) && dma_profile_graph_active) dma_timeline_graph.invalid = true;
    t.launched = true;
    t.launch = dma_profile_clock();
    t.invalid |= !dma_profile_interval(t.entry, t.launch);
}
void fpga_dma_profile_transfer::success() { success_ = enabled_; }

fpga_dma_profile_transfer::~fpga_dma_profile_transfer() {
    if (!enabled_) return;
    auto & t = dma_profile_transfer_state_current;
    const long long end = dma_profile_clock();
    t.invalid |= !dma_profile_interval(t.entry, end);
    if (success_) {
        t.invalid |= !t.launched || !t.observed || !dma_profile_interval(t.launch, t.observed_at) ||
                     !dma_profile_interval(t.observed_at, end) ||
                     t.row.callback_pack_us > t.observed_at - t.launch;
    }
    dma_profile_row row = t.row;
    if (t.invalid) {
        row = {};
        row.timing_invalid = 1;
        // Invalid timing never contributes fabricated successful durations.
    } else if (success_) {
        row.descriptors = 1; row.bytes = t.bytes;
        row.preparation_us = t.launch - t.entry;
        row.launch_to_observed_us = t.observed_at - t.launch;
        row.noncallback_launch_interval_us = row.launch_to_observed_us - row.callback_pack_us;
        row.cleanup_us = end - t.observed_at;
    } else {
        row.failed_elapsed_us = end - t.entry;
        row.failed_callback_pack_us = t.attempted_pack_us;
        row.callback_pack_us = 0; row.callback_pack_count = 0; row.callback_pack_bytes = 0;
    }
    if (!success_) { row.failures = 1; row.failed_bytes = t.bytes; }
    dma_timeline_transfer(t, end, success_);
    dma_profile_add(dma_profile_process.rows[t.tag], row);
    if (dma_profile_graph_active) dma_profile_add(dma_profile_graph.rows[t.tag], row);
    t = {};
}

void fpga_dma_profile_observe(bool error, bool done, bool enabled) {
    if (!fpga_dma_profile_enabled()) return;
    auto & t = dma_profile_transfer_state_current;
    if (dma_profile_graph_active && (error || !t.active || !t.launched)) dma_timeline_graph.invalid = true;
    if (!t.active || !t.launched) return;
    const bool completion = !error && done && !enabled;
    // No new per-poll clock unless a productive batch awaits classification
    // or this sample is the terminal successful observation.
    if (!t.pending_pack && (!completion || t.observed)) return;
    const long long now = dma_profile_clock();
    t.invalid |= !dma_profile_interval(t.launch, now);
    if (t.pending_pack) {
        t.invalid |= !dma_profile_interval(t.pack_end, now);
        if (!t.invalid) {
            if (error) ++t.row.error;
            else if (completion) ++t.row.completion_observed;
            else if (enabled) ++t.row.channel_enabled;
            else ++t.row.incomplete_en_clear;
            t.row.post_pack_observation_gap_us += now - t.pack_end;
        }
        t.pending_pack = false;
    }
    if (completion && !t.observed) { t.observed = true; t.observed_at = now; }
}
void fpga_dma_profile_pack_begin() {
    if (!fpga_dma_profile_enabled() || !dma_profile_transfer_state_current.active) return;
    dma_profile_transfer_state_current.pack_begin = dma_profile_clock();
}
void fpga_dma_profile_pack_end(bool ok, size_t words_before, size_t words_after) {
    if (!fpga_dma_profile_enabled() || !dma_profile_transfer_state_current.active) return;
    auto & t = dma_profile_transfer_state_current;
    const long long end = dma_profile_clock();
    t.invalid |= !dma_profile_interval(t.pack_begin, end) || !dma_profile_interval(t.launch, t.pack_begin);
    if (!t.invalid) t.attempted_pack_us += end - t.pack_begin;
    if (!ok || words_after <= words_before) return;
    const size_t words = words_after - words_before;
    if (words > static_cast<size_t>(-1) / sizeof(uint32_t)) { t.invalid = true; return; }
    if (!t.invalid) {
        t.row.callback_pack_us += end - t.pack_begin;
        ++t.row.callback_pack_count;
        t.row.callback_pack_bytes += words * sizeof(uint32_t);
    }
    t.pack_end = end;
    t.pending_pack = true;
}
void fpga_dma_profile_finalize_begin() {
    if (!fpga_dma_profile_enabled()) return;
    if (dma_profile_graph_active && dma_profile_finalize_start >= 0) dma_timeline_graph.invalid = true;
    dma_profile_finalize_start = dma_profile_clock();
    if (dma_profile_graph_active && dma_profile_finalize_start < 0) dma_timeline_graph.invalid = true;
}
void fpga_dma_profile_finalize_end(long long residual_us, long long helper_wait_us,
                                   long long helper_service_us, size_t main_bytes, size_t callback_bytes) {
    if (!fpga_dma_profile_enabled()) return;
    const long long end = dma_profile_clock();
    dma_profile_bucket sample;
    if (!dma_profile_interval(dma_profile_finalize_start, end) || residual_us < 0 || helper_wait_us < 0 ||
        helper_service_us < 0 || callback_bytes > main_bytes) {
        sample.finalize_timing_invalid = 1;
        if (dma_profile_graph_active) dma_timeline_graph.invalid = true;
    } else {
        sample.finalize_count = 1;
        sample.residual_bytes = main_bytes - callback_bytes;
        sample.residual_us = residual_us; sample.helper_wait_us = helper_wait_us;
        sample.helper_service_overlap_us = helper_service_us;
        sample.finalize_us = end - dma_profile_finalize_start;
    }
    dma_profile_finalize_start = -1;
    dma_profile_add(dma_profile_process, sample);
    if (dma_profile_graph_active) dma_profile_add(dma_profile_graph, sample);
}
void fpga_dma_profile_graph_begin(int graph_seq) {
    if (!fpga_dma_profile_enabled()) return;
    if (dma_profile_graph_active) fpga_dma_profile_graph_end("incomplete", false);
    dma_profile_graph = {};
    dma_timeline_graph = {};
    dma_timeline_graph.begin = dma_profile_clock();
    dma_timeline_graph.invalid = dma_timeline_graph.begin < 0 || dma_profile_transfer_state_current.active ||
                                 dma_profile_finalize_start >= 0;
    dma_profile_graph_seq = graph_seq;
    dma_profile_graph_active = true;
}
void fpga_dma_profile_graph_end(const char * scope, bool decode_token) {
    if (!fpga_dma_profile_enabled() || !dma_profile_graph_active) return;
    auto & g = dma_timeline_graph;
    g.end = dma_profile_clock(); // Freeze the window before any report output.
    const bool complete_scope = scope &&
        ((decode_token && !std::strcmp(scope, "decode_token")) ||
         (!decode_token && !std::strcmp(scope, "prefill_or_ubatch")));
    g.invalid |= !complete_scope || dma_profile_transfer_state_current.active ||
                 dma_profile_finalize_start >= 0 || !dma_profile_interval(g.begin, g.end) ||
                 (g.descriptors && !dma_profile_interval(g.last_exit, g.end));
    if (!g.invalid) {
        g.window_us = g.end - g.begin;
        if (g.descriptors) g.tail_us = g.end - g.last_observed;
        else { g.no_dma_us = g.window_us; g.no_dma_graphs = 1; }
        long long partition = 0;
        g.invalid = !dma_timeline_add_value(partition, g.head_us) ||
                    !dma_timeline_add_value(partition, g.service_us) ||
                    !dma_timeline_add_value(partition, g.gap_us) ||
                    !dma_timeline_add_value(partition, g.tail_us) ||
                    !dma_timeline_add_value(partition, g.no_dma_us) || partition != g.window_us;
    }
    dma_profile_timeline sample;
    if (g.invalid) {
        g.excluded_graphs = 1;
        sample.excluded_graphs = 1; // Do not aggregate any partial failed window.
    } else {
        g.valid_graphs = 1;
        sample = g;
    }
    dma_timeline_add(dma_timeline_all_graphs, sample);
    if (decode_token) dma_timeline_add(dma_timeline_decode, sample);
    dma_timeline_emit(g, scope ? scope : "incomplete", dma_profile_graph_seq, false);
    dma_profile_emit(dma_profile_graph, scope, dma_profile_graph_seq, decode_token ? 1 : 0);
    if (decode_token) { dma_profile_add(dma_profile_decode, dma_profile_graph); ++dma_profile_decode_graphs; }
    dma_profile_graph = {};
    dma_profile_graph_active = false;
}
void fpga_dma_profile_finish() {
    if (!fpga_dma_profile_enabled()) return;
    fpga_dma_profile_graph_end("incomplete", false);
    dma_timeline_emit(dma_timeline_decode, "decode_total", -1, true);
    dma_timeline_emit(dma_timeline_all_graphs, "all_graphs_total", -1, true);
    dma_profile_emit(dma_profile_decode, "decode_total", -1, dma_profile_decode_graphs);
    dma_profile_emit(dma_profile_process, "process_total_including_startup", -1, dma_profile_decode_graphs);
    dma_profile_finished = true;
}

namespace {
void fpga_log_token_line(bool enabled, const char * tag, bool force_flush, const char * fmt, ...) {
    if (!enabled) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fpga_log_vline(tag, force_flush, fmt, ap);
    va_end(ap);
}
} // namespace

void fpga_log_token_timing(const fpga_token_timing_log_t & data) {
    const auto compute_report_us = [&data](long long us) {
        return data.compute_watch_enabled ? (double) us : std::numeric_limits<double>::quiet_NaN();
    };
    if (data.timing_enabled) {
        fpga_log_token_line(
            true, "TIMING", data.force_flush,
            "graph_seq=%d next_graph_seq=%d ubatch_tokens=%lld scope=%s reason=%s matmuls=%lld vpu_runs=%lld "
            "graph_wall_ms=%.3f device_first_start_to_last_output_ready_ms=%.3f fpga_matmul_wall_sum_ms=%.3f "
            "host_to_ip_dma_ms=%.3f act_dma_ms=%.3f weight_dma_ms=%.3f scale_dma_ms=%.3f "
            "ping_h2ip_ms=%.3f pong_h2ip_ms=%.3f ping_jobs=%lld pong_jobs=%lld "
            "ip_compute_sum_ms=%.3f ping_compute_ms=%.3f pong_compute_ms=%.3f "
            "ip_to_host_dma_ms=%.3f ping_ip2host_ms=%.3f pong_ip2host_ms=%.3f "
            "host_result_read_ms=%.3f ping_host_read_ms=%.3f pong_host_read_ms=%.3f token_read_ms=%.3f "
            "prep_ms=%.3f host_accum_ms=%.3f act_bytes=%zu weight_bytes=%zu scale_bytes=%zu result_bytes=%zu "
            "measurement_scope=host_ddr_to_ip_ip_compute_ip_to_host_ddr internal_pl_interconnect=not_observable_from_host",
            data.graph_seq, data.next_graph_seq, (long long) data.ubatch_tokens, data.scope, data.reason ? data.reason : "unknown",
            data.matmuls, data.vpu_runs, (double) data.token_wall_us / 1000.0,
            (double) data.device_span_us / 1000.0, (double) data.matmul_wall_us / 1000.0,
            (double) data.h2ip_dma_us / 1000.0, (double) data.act_dma_us / 1000.0,
            (double) data.weight_dma_us / 1000.0, (double) data.scale_dma_us / 1000.0,
            (double) data.bank_h2ip_us[0] / 1000.0,
            (double) data.bank_h2ip_us[1] / 1000.0, data.bank_jobs[0],
            data.bank_jobs[1], compute_report_us(data.ip_compute_us) / 1000.0,
            compute_report_us(data.bank_compute_us[0]) / 1000.0,
            compute_report_us(data.bank_compute_us[1]) / 1000.0,
            (double) data.ip2host_dma_us / 1000.0,
            (double) data.bank_ip2host_us[0] / 1000.0,
            (double) data.bank_ip2host_us[1] / 1000.0,
            (double) data.host_read_us / 1000.0,
            (double) data.bank_host_read_us[0] / 1000.0,
            (double) data.bank_host_read_us[1] / 1000.0, (double) data.token_read_us / 1000.0,
            (double) data.prep_us / 1000.0, (double) data.host_accum_us / 1000.0,
            data.activation_bytes, data.weight_bytes, data.scale_bytes,
            data.result_bytes);
    }

    if (data.bottleneck_summary_enabled) {
        const long long prep_known_us = data.prep_weight_select_us +
                                        data.prep_direct_weight_pack_us +
                                        data.prep_scale_pack_us + data.prep_act_pack_us;
        const long long prep_other_us = std::max(0LL, data.prep_us - prep_known_us);
        const long long outside_matmul_us = std::max(0LL, data.token_wall_us - data.matmul_wall_us);
        const double device_noncompute_us = compute_report_us(std::max(0LL, data.device_span_us - data.ip_compute_us));
        const double compute_util_pct = !data.compute_watch_enabled ? std::numeric_limits<double>::quiet_NaN() :
                                        data.device_span_us > 0 ?
                                            100.0 * (double) data.ip_compute_us / (double) data.device_span_us :
                                            0.0;
        fpga_log_token_line(
            true, "BREAKDOWN", data.force_flush,
            "graph_seq=%d next_graph_seq=%d scope=%s graph_wall_ms=%.3f matmul_wall_ms=%.3f "
            "outside_matmul_ms=%.3f device_span_ms=%.3f ip_compute_ms=%.3f device_noncompute_ms=%.3f "
            "compute_util_pct=%.2f prep_total_ms=%.3f prep_weight_select_ms=%.3f "
            "prep_direct_weight_pack_ms=%.3f prep_scale_pack_ms=%.3f prep_act_pack_ms=%.3f prep_other_ms=%.3f "
            "handoffs=%lld prep_overlap_ms=%.3f prep_late_jobs=%lld prep_late_ms=%.3f prep_headroom_ms=%.3f "
            "output_ready_to_next_launch_ms=%.3f retire_to_next_launch_ms=%.3f",
            data.graph_seq, data.next_graph_seq, data.scope, (double) data.token_wall_us / 1000.0,
            (double) data.matmul_wall_us / 1000.0, (double) outside_matmul_us / 1000.0,
            (double) data.device_span_us / 1000.0, compute_report_us(data.ip_compute_us) / 1000.0,
            device_noncompute_us / 1000.0, compute_util_pct, (double) data.prep_us / 1000.0,
            (double) data.prep_weight_select_us / 1000.0,
            (double) data.prep_direct_weight_pack_us / 1000.0,
            (double) data.prep_scale_pack_us / 1000.0,
            (double) data.prep_act_pack_us / 1000.0, (double) prep_other_us / 1000.0,
            data.scheduler_handoffs, (double) data.scheduler_prepare_overlap_us / 1000.0,
            data.scheduler_prepare_late_jobs, (double) data.scheduler_prepare_late_us / 1000.0,
            (double) data.scheduler_prepare_headroom_us / 1000.0,
            (double) data.scheduler_output_to_launch_us / 1000.0,
            (double) data.scheduler_retire_to_launch_us / 1000.0);

        if (data.sampled_detail) {
            fpga_log_token_line(
                true, "GEMV_TIME_BY_LAYER_TYPE", data.force_flush,
                "graph_seq=%d scope=%s "
                "attention_matmuls=%lld attention_tile_jobs=%lld attention_wall_ms=%.3f attention_host_prepare_ms=%.3f attention_fpga_compute_ms=%.3f attention_dma_ms=%.3f "
                "ffn_gate_matmuls=%lld ffn_gate_tile_jobs=%lld ffn_gate_wall_ms=%.3f ffn_gate_host_prepare_ms=%.3f ffn_gate_fpga_compute_ms=%.3f ffn_gate_dma_ms=%.3f "
                "ffn_up_matmuls=%lld ffn_up_tile_jobs=%lld ffn_up_wall_ms=%.3f ffn_up_host_prepare_ms=%.3f ffn_up_fpga_compute_ms=%.3f ffn_up_dma_ms=%.3f "
                "ffn_down_matmuls=%lld ffn_down_tile_jobs=%lld ffn_down_wall_ms=%.3f ffn_down_host_prepare_ms=%.3f ffn_down_fpga_compute_ms=%.3f ffn_down_dma_ms=%.3f "
                "other_matmuls=%lld other_tile_jobs=%lld other_wall_ms=%.3f other_host_prepare_ms=%.3f other_fpga_compute_ms=%.3f other_dma_ms=%.3f",
                data.graph_seq, data.scope,
                data.category_matmuls[0], data.category_runs[0],
                (double) data.category_wall_us[0] / 1000.0,
                (double) data.category_prep_us[0] / 1000.0,
                compute_report_us(data.category_compute_us[0]) / 1000.0,
                (double) data.category_dma_us[0] / 1000.0,
                data.category_matmuls[1], data.category_runs[1],
                (double) data.category_wall_us[1] / 1000.0,
                (double) data.category_prep_us[1] / 1000.0,
                compute_report_us(data.category_compute_us[1]) / 1000.0,
                (double) data.category_dma_us[1] / 1000.0,
                data.category_matmuls[2], data.category_runs[2],
                (double) data.category_wall_us[2] / 1000.0,
                (double) data.category_prep_us[2] / 1000.0,
                compute_report_us(data.category_compute_us[2]) / 1000.0,
                (double) data.category_dma_us[2] / 1000.0,
                data.category_matmuls[3], data.category_runs[3],
                (double) data.category_wall_us[3] / 1000.0,
                (double) data.category_prep_us[3] / 1000.0,
                compute_report_us(data.category_compute_us[3]) / 1000.0,
                (double) data.category_dma_us[3] / 1000.0,
                data.category_matmuls[4], data.category_runs[4],
                (double) data.category_wall_us[4] / 1000.0,
                (double) data.category_prep_us[4] / 1000.0,
                compute_report_us(data.category_compute_us[4]) / 1000.0,
                (double) data.category_dma_us[4] / 1000.0);

            const double zdma_avg_us = data.zdma_descriptors > 0 ?
                                           (double) data.zdma_elapsed_us / (double) data.zdma_descriptors :
                                           0.0;
            fpga_log_token_line(
                true, "ZDMA", data.force_flush,
                "graph_seq=%d scope=%s descriptors=%lld bytes=%zu elapsed_ms=%.3f avg_descriptor_us=%.3f polls=%lld "
                "zero_poll_descriptors=%lld saw_enabled_descriptors=%lld act_desc=%lld weight_desc=%lld scale_desc=%lld "
                "result_desc=%lld other_desc=%lld max_descriptor_bytes=%zu",
                data.graph_seq, data.scope, data.zdma_descriptors, data.zdma_bytes,
                (double) data.zdma_elapsed_us / 1000.0, zdma_avg_us, data.zdma_polls,
                data.zdma_zero_poll_descriptors, data.zdma_saw_enabled_descriptors,
                data.zdma_act_descriptors, data.zdma_weight_descriptors,
                data.zdma_scale_descriptors, data.zdma_result_descriptors,
                data.zdma_other_descriptors, data.max_descriptor_bytes);
        }
    }
}
