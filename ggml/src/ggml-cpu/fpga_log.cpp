#include "fpga_log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
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
                    "[FPGA][ERROR] cannot open %s; detailed FPGA telemetry is disabled: errno=%d (%s)\n",
                    FPGA_LOG_FILE, open_errno, strerror(open_errno));
            fp = fopen("/dev/null", "w");
            if (!fp) {
                fp = stderr;
            }
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

void fpga_log_latency(const char * fmt, ...) {
    if (fpga_log_fp() == stderr) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fpga_log_vline("LATENCY", true, fmt, ap);
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
