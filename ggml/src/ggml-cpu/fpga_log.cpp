#include "fpga_log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ctime>
#include <cstdint>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>

namespace {

constexpr const char * FPGA_LOG_FILE = "/tmp/fpga_debug.log";

int g_flush_every   = 256;
int g_pending_lines = 0;


constexpr const char * LOG_BORDER = "==============================================================================================================";
const char * log_scope(const char * scope) {
    if (!scope) return "chưa hoàn tất";
    if (!std::strcmp(scope, "decode_token")) return "token decode";
    if (!std::strcmp(scope, "prefill_or_ubatch")) return "prefill hoặc ubatch";
    if (!std::strcmp(scope, "decode_total")) return "tổng decode";
    if (!std::strcmp(scope, "process_total_including_startup")) return "toàn bộ tiến trình, bao gồm khởi động";
    if (!std::strcmp(scope, "all_graphs_total")) return "tổng các graph";
    if (!std::strcmp(scope, "incomplete")) return "chưa hoàn tất";
    return scope;
}
// Labels use precomposed Vietnamese UTF-8; count characters instead of bytes.
size_t log_width(const std::string & text) {
    size_t width = 0;
    for (unsigned char c : text) if ((c & 0xc0) != 0x80) ++width;
    return width;
}
using log_table_t = std::vector<std::vector<std::string>>;
void log_table(FILE * fp, const log_table_t & rows) {
    if (rows.empty()) return;
    std::vector<size_t> widths(rows.front().size());
    for (const auto & row : rows)
        for (size_t i = 0; i < row.size(); ++i) widths[i] = std::max(widths[i], log_width(row[i]));
    const auto border = [&]() {
        fputs("  +", fp);
        for (size_t width : widths) { fputs(std::string(width + 2, '-').c_str(), fp); fputc('+', fp); }
        fputc('\n', fp);
    };
    border();
    for (size_t r = 0; r < rows.size(); ++r) {
        fputs("  |", fp);
        for (size_t i = 0; i < rows[r].size(); ++i) {
            const auto & cell = rows[r][i];
            const std::string padding(widths[i] - log_width(cell), ' ');
            const bool numeric = r > 0 && i > 0;
            fprintf(fp, " %s%s%s |", numeric ? padding.c_str() : "", cell.c_str(), numeric ? "" : padding.c_str());
        }
        fputc('\n', fp);
        border();
    }
}
std::string log_ms(double us) {
    char text[64];
    snprintf(text, sizeof(text), "%.3f", us / 1000.0);
    return text;
}
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
        fprintf(fp, "Log ZDMA DDR-to-IP bắt đầu tại mốc thời gian %ld\n", (long) now);
        fprintf(fp, "============================================================\n");
        fflush(fp);
    }
    return fp;
}

void fpga_log_section(const char * title) {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    fprintf(fp, "\n%s\n%s\n%s\n", LOG_BORDER, title ? title : "", LOG_BORDER);
}
void fpga_log_graph_header(int graph_seq, const char * scope) {
    char title[160];
    if (scope && !std::strcmp(scope, "decode_token")) snprintf(title, sizeof(title), "Graph = %d", graph_seq);
    else snprintf(title, sizeof(title), "Graph = %d | Phạm vi = %s", graph_seq, log_scope(scope));
    fpga_log_section(title);
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
    fpga_log_vline("ĐỘ TRỄ", true, fmt, ap);
    va_end(ap);
}

void fpga_log_result_overlap(int graph_seq, bool enabled, long long jobs, long long host_us) {
    if (fpga_log_fp() == stderr) {
        return;
    }
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[RESULT OVERLAP]\n  Graph = %d | Được bật = %d | Số công việc = %lld\n"
                "  Host xử lý sau lần chạy tiếp(ms) = %.3f | Thời gian thực sự được che giấu = chưa đo\n",
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
        "P2 PACK BREAKDOWN\n  Phạm vi = %s | Graph = %d | Số token = %lld\n"
        "  Tổng thời gian(us) = %lld | Đóng gói tuần tự(us) = %lld | Điều phối(us) = %lld\n"
        "  Đóng gói luồng chính(us) = %lld | Fence luồng gọi(us) = %lld | Chờ luồng gọi(us) = %lld\n"
        "  Thời gian khác suy ra(us) = %lld | Helper overlap(us) = %lld | Công việc tuần tự = %lld | Công việc song song = %lld\n"
        "  Đặt chỗ: lấy khóa(us) = %lld | xử lý(us) = %lld | nhả khóa(us) = %lld\n"
        "  Giao việc: lấy khóa(us) = %lld | công bố(us) = %lld | phát tín hiệu(us) = %lld | nhả khóa(us) = %lld\n"
        "  Điều phối khác suy ra(us) = %lld\n"
        "  Chi tiết điều phối = các phần trong thời gian điều phối | Thứ tự thông báo = nhả khóa rồi phát tín hiệu\n"
        "  Phạm vi đo = chuẩn bị trực tiếp thành công có đo đạc | Thời gian helper = overlap, không cộng độc lập",
        log_scope(scope), graph_seq, tokens, data.total_us, data.serial_pack_us, data.dispatch_us,
        data.main_pack_us, data.caller_fence_us, data.caller_wait_us, other_us, data.helper_service_us,
        data.serial_jobs, data.parallel_jobs, data.reserve_lock_us, data.reserve_body_us,
        data.reserve_unlock_us, data.submit_lock_us, data.submit_publish_us, data.submit_signal_us,
        data.submit_unlock_us, dispatch_other_us);
}

#ifdef USE_FPGA
namespace {
struct ram_report_t {
    bool pending = false;
    int status = 0; // 0 success, 1 allocation failure, 2 invalid duration, 3 mismatch
    size_t allocation_bytes = 0, buffer_bytes = 0, copied_bytes = 0;
    unsigned passes = 0;
    double seconds = 0.0, copy_gb_s = 0.0;
};
ram_report_t ram_report;
bool startup_started = false;
}
void fpga_log_startup_begin() {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    if (!startup_started) { fpga_log_section("Khởi động"); startup_started = true; }
    if (!ram_report.pending) return;
    const auto r = ram_report;
    ram_report.pending = false;
    if (r.status == 1) {
        fprintf(fp, "[RAM BANDWIDTH]\n  Trạng thái = bỏ qua | Lý do = cấp phát thất bại | Số byte cấp phát = %zu\n", r.allocation_bytes);
    } else if (r.status) {
        fprintf(fp, "[RAM BANDWIDTH]\n  Trạng thái = thất bại | Lý do = %s\n",
                r.status == 2 ? "thời gian không hợp lệ" : "dữ liệu sao chép không khớp");
    } else {
        fprintf(fp, "[RAM BANDWIDTH]\n  Phạm vi = bộ nhớ có cache của CPU khi khởi động\n"
                    "  Trạng thái = thành công | Phương pháp = libc_memcpy | Số luồng = 1\n"
                    "  Buffer(byte) = %zu | Số lượt = %u | Số byte sao chép = %zu | Thời gian(ms) = %.6f\n"
                    "  Tốc độ sao chép suy ra(GB/s) = %.6f | Tốc độ đọc và ghi logic suy ra(GB/s) = %.6f | Đã kiểm tra = 1\n"
                    "  Lưu lượng DDR vật lý = chưa đo | Cấp phát, warmup và kiểm tra = không tính | Đơn vị = GB thập phân\n",
                r.buffer_bytes, r.passes, r.copied_bytes, r.seconds * 1000.0, r.copy_gb_s, 2.0 * r.copy_gb_s);
    }
    fpga_log_finish_line(fp, true);
}
void fpga_log_ram_bandwidth(bool defer_output) {
    if (fpga_log_fp() == stderr) return;
    if (ram_report.pending) fpga_log_startup_begin();
    ram_report = {};
    ram_report.pending = true;
    // Two 32 MiB buffers exceed the CPU cache capacity. Touch every
    // page before timing, so allocation and initial page faults are excluded.
    // This is logical memcpy throughput, not a DDR-controller traffic counter.
    constexpr size_t bytes = 32u * 1024u * 1024u;
    constexpr unsigned passes = 8;
    auto * src = static_cast<unsigned char *>(std::malloc(bytes));
    auto * dst = static_cast<unsigned char *>(std::malloc(bytes));
    if (!src || !dst) {
        std::free(src);
        std::free(dst);
        ram_report.status = 1;
        ram_report.allocation_bytes = 2 * bytes;
        if (!defer_output) fpga_log_startup_begin();
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
    ram_report.status = !verified ? 3 : seconds <= 0.0 ? 2 : 0;
    if (!ram_report.status) {
        ram_report.buffer_bytes = bytes;
        ram_report.passes = passes;
        ram_report.copied_bytes = bytes * passes;
        ram_report.seconds = seconds;
        ram_report.copy_gb_s = static_cast<double>(ram_report.copied_bytes) / seconds / 1e9;
    }
    if (!defer_output) fpga_log_startup_begin();
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
    const char * phase_text = phase ? phase : "không rõ";
    const struct { const char * key; const char * label; } phases[] = {
        {"metadata_begin", "bắt đầu metadata"}, {"metadata_done", "hoàn tất metadata"},
        {"vocab_begin", "bắt đầu vocab"}, {"vocab_done", "hoàn tất vocab"},
        {"tensors_begin", "bắt đầu tensor"}, {"tensors_done", "hoàn tất tensor"},
        {"mapping_begin", "bắt đầu ánh xạ"}, {"mapping_done", "hoàn tất ánh xạ"},
        {"buffers_done", "hoàn tất buffer"}, {"tensor_data_begin", "bắt đầu dữ liệu tensor"},
        {"tensor_data_done", "hoàn tất dữ liệu tensor"}, {"tensors_cancelled", "hủy nạp tensor"}, {"load_error", "lỗi nạp"}
    };
    for (const auto & item : phases) if (phase && !std::strcmp(phase, item.key)) { phase_text = item.label; break; }
    fpga_log_latency("[MODEL LOAD DETAIL]\n  Giai đoạn = %s\n"
                     "  Thời gian(ms) = %.3f | CPU người dùng(ms) = %.3f | CPU hệ thống(ms) = %.3f\n"
                     "  Minor page fault = %ld | Major page fault = %ld | Nhập block = %ld | Số byte đọc = %lld\n"
                     "  RSS tối đa(KiB) = %ld | Bộ đếm = tích lũy theo tiến trình | Không có dữ liệu = -1",
                     phase_text, elapsed_ms, user_ms, system_ms,
                     usage_ok ? usage.ru_minflt : -1L, usage_ok ? usage.ru_majflt : -1L,
                     usage_ok ? usage.ru_inblock : -1L, read_bytes, usage_ok ? usage.ru_maxrss : -1L);
}
#endif

#ifdef USE_FPGA

void fpga_log_runtime_summary(double load_ms, double prompt_ms, int prompt_tokens,
                              double decode_ms, int decode_runs) {
    fpga_log_section("Tổng kết thời gian chạy");
    fpga_log_latency("RUNTIME SUMMARY\n  Nạp mô hình(ms) = %.2f | Token prompt = %d | Prompt(ms) = %.2f | Tốc độ prompt suy ra(token/s) = %.2f\n"
                     "  Số lượt decode = %d | Decode(ms) = %.2f | Tốc độ decode suy ra(token/s) = %.2f",
                     load_ms, prompt_tokens, prompt_ms, prompt_ms > 0 ? 1000.0 * prompt_tokens / prompt_ms : 0.0,
                     decode_runs, decode_ms, decode_ms > 0 ? 1000.0 * decode_runs / decode_ms : 0.0);
}
#endif

void fpga_log_prompt_weight_reuse(const char * tensor, long long columns, long long jobs,
                                  unsigned long long avoided_bytes) {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    fprintf(fp, "[ĐỘ TRỄ] PROMPT WEIGHT BANK REUSE\n  Số cột = %lld | Phạm vi = phép matmul đã hoàn tất\n", columns);
    log_table(fp, {{"tensor", "Số công việc tái sử dụng", "Số byte DMA weight tránh được"},
                   {tensor ? tensor : "?", std::to_string(jobs), std::to_string(avoided_bytes)}});
    fpga_log_finish_line(fp, true);
}

namespace {
// Serialized by the host's existing submission/lifecycle mutex. No worker,
// watcher, per-descriptor allocation, lock, MMIO access, or per-descriptor output is added.
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

struct dma_pending_report {
    bool ready = false;
    dma_profile_bucket profile;
    dma_profile_timeline timeline;
    std::string scope;
    int graph_seq = -1;
    unsigned long long graphs = 0;
};
dma_pending_report dma_pending;
bool dma_config_valid = true;
bool dma_configured = false;
bool dma_config_emitted = false;

void dma_timeline_emit(const dma_profile_timeline & g, const char * scope, int graph_seq, bool aggregate) {
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    const bool valid = !g.invalid && g.excluded_graphs == 0;
    // Aggregate exclusions do not invalidate the sum of explicitly valid windows.
    const bool usable = !g.invalid && g.valid_graphs > 0 && (aggregate || valid);
    const auto duration = [usable](long long value) { return usable ? value : -1LL; };
    const double service_fraction = usable && g.window_us > 0 ? (double) g.service_us / g.window_us : -1.0;
    const double gap_fraction = usable && g.window_us > 0 ? (double) g.gap_us / g.window_us : -1.0;
    fprintf(fp, "[DMA_PROFILE_TIMELINE]\n  Phạm vi = %s | Graph = %d | Graph hợp lệ = %lld | Graph loại trừ = %lld | Graph không DMA = %lld | Tổng hợp hợp lệ = %d\n"
                "  Bắt đầu monotonic(us) = %lld | Kết thúc monotonic(us) = %lld\n"
                "  Cửa sổ profiler(us) = %lld | Đầu cửa sổ đến khởi chạy đầu(us) = %lld | Khởi chạy đến quan sát hoàn tất(us) = %lld\n"
                "  Quan sát hoàn tất đến khởi chạy tiếp(us) = %lld | Sau quan sát hoàn tất cuối(us) = %lld | Cửa sổ không DMA(us) = %lld\n"
                "  Descriptor = %lld | Số khoảng giữa descriptor = %lld | Khoảng lớn nhất(us) = %lld\n"
                "  Tỷ lệ khoảng khởi chạy đến quan sát hoàn tất, suy ra = %.6f | Tỷ lệ khoảng giữa descriptor, suy ra = %.6f\n"
                "Phạm vi đo\n  Cửa sổ = hook profiler của graph | Khoảng giữa các graph = loại trừ | Hai đầu = %s\n"
                "  Thời điểm phần cứng hoàn tất = chưa đo | Khoảng phục vụ = bao trên quan sát từ phần mềm\n"
                "  Khoảng giữa descriptor = chỉ là cận dưới thời gian kênh nghỉ nếu host sở hữu độc quyền kênh; không khẳng định DDR ngừng truyền\n"
                "  Sở hữu độc quyền kênh = điều kiện bắt buộc | Độ phân giải đồng hồ(us) = 1\n"
                "  Mức sử dụng DDR = chưa đo | Callback, chuẩn bị, dọn dẹp = các phần có overlap | Lỗi nghiêm trọng khi đang truyền = loại trừ\n",
            log_scope(scope), graph_seq, g.valid_graphs, g.excluded_graphs, g.no_dma_graphs, usable ? 1 : 0,
            g.begin, g.end, duration(g.window_us), duration(g.head_us), duration(g.service_us), duration(g.gap_us),
            duration(g.tail_us), duration(g.no_dma_us), duration(g.descriptors), duration(g.gap_count),
            duration(g.max_gap_us), service_fraction, gap_fraction,
            aggregate ? "bao các cửa sổ hợp lệ, không phải mẫu số" : "biên cửa sổ");
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
    fprintf(fp, "[DMA_PROFILE]\n  Phạm vi = %s | Graph = %d | Số graph decode = %llu\n", log_scope(scope), graph_seq, graphs);
    const auto & other = bucket.rows[4];
    const int columns = other.descriptors || other.failures || other.timing_invalid ? 5 : 4;
    const char * tags[] = {"ACT", "WEIGHT", "SPU SCALE", "RESULT", "OTHER"};
    log_table_t table = {{"Thông số"}};
    for (int i = 0; i < columns; ++i) table[0].push_back(tags[i]);
    { std::vector<std::string> row = {"Số descriptor"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].descriptors)); table.push_back(row); }
    { std::vector<std::string> row = {"Số byte"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].bytes)); table.push_back(row); }
    { std::vector<std::string> row = {"Chuẩn bị(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].preparation_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Khởi chạy đến quan sát hoàn tất(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].launch_to_observed_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Pack trong callback(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].callback_pack_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Số lần pack trong callback"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].callback_pack_count)); table.push_back(row); }
    { std::vector<std::string> row = {"Số byte pack trong callback"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].callback_pack_bytes)); table.push_back(row); }
    { std::vector<std::string> row = {"Khoảng khởi chạy trừ callback pack, suy ra(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].noncallback_launch_interval_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Dọn dẹp(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].cleanup_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Số lần lỗi"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].failures)); table.push_back(row); }
    { std::vector<std::string> row = {"Số byte lỗi"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].failed_bytes)); table.push_back(row); }
    { std::vector<std::string> row = {"Thời gian lần lỗi(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].failed_elapsed_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Callback pack của lần lỗi(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].failed_callback_pack_us)); table.push_back(row); }
    { std::vector<std::string> row = {"Số mẫu thời gian không hợp lệ"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].timing_invalid)); table.push_back(row); }
    { std::vector<std::string> row = {"Sau pack: đã quan sát hoàn tất"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].completion_observed)); table.push_back(row); }
    { std::vector<std::string> row = {"Sau pack: EN còn bật"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].channel_enabled)); table.push_back(row); }
    { std::vector<std::string> row = {"Sau pack: chưa hoàn tất, EN tắt"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].incomplete_en_clear)); table.push_back(row); }
    { std::vector<std::string> row = {"Sau pack: lỗi"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].error)); table.push_back(row); }
    { std::vector<std::string> row = {"Kết thúc pack đến đọc trạng thái(us)"}; for (int i = 0; i < columns; ++i) row.push_back(std::to_string(bucket.rows[i].post_pack_observation_gap_us)); table.push_back(row); }
    log_table(fp, table);
    fputs("  Thời gian thành công = chỉ mẫu hợp lệ | Callback khi lỗi = tất cả batch đã thử\n"
          "  Sau pack = tất cả mẫu hợp lệ | Byte callback = chỉ WEIGHT\n"
          "  Lỗi = các lần trả về lỗi | Lỗi nghiêm trọng khi đang truyền = loại trừ\n"
          "  Khoảng ngoài callback = khoảng khởi chạy phía host trừ callback pack đã đo\n"
          "  Thời điểm phần cứng hoàn tất = chưa đo | Tranh chấp = chưa đo\n", fp);
    fprintf(fp, "[DMA_PROFILE_PACK]\n  Phạm vi = %s | Graph = %d | Số graph decode = %llu | Chỉ lần thành công = 1\n"
                "Pack còn lại\n  Số lần finalize = %llu | Số byte còn lại = %llu | Pack còn lại(us) = %lld\n"
                "Helper và finalize\n  Chờ helper(us) = %lld | Helper overlap(us) = %lld | Finalize(us) = %lld | Mẫu thời gian không hợp lệ = %llu\n"
                "Phạm vi đo\n  Helper có overlap = 1 | Byte còn lại = chỉ WEIGHT | Chờ vô hiệu hóa khi lỗi = loại trừ\n"
                "  Đồng hồ finalize = monotonic | Đồng hồ thành phần = realtime hiện có | Tổng các thành phần có thể khác finalize = 1\n",
            log_scope(scope), graph_seq, graphs, bucket.finalize_count, bucket.residual_bytes, bucket.residual_us,
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
    dma_pending = {};
    dma_config_valid = valid;
    dma_configured = true;
    dma_config_emitted = false;
    if (!valid) fpga_dma_profile_log_config();
    return valid;
}
void fpga_dma_profile_log_config() {
    if (!dma_configured || dma_config_emitted) return;
    dma_config_emitted = true;
    FILE * fp = fpga_log_fp();
    if (fp == stderr) return;
    fprintf(fp, "[DMA_PROFILE_CONFIG]\n  Bật = %d | Hợp lệ = %d | Mặc định = tắt\n"
                "  Đồng hồ descriptor = monotonic | Đồng hồ finalize = monotonic | Đồng hồ thành phần = realtime hiện có\n"
                "  Đầu ra = chỉ file | Thu thập graph = khi bật profiler | Tổng cuối = cần cleanup bình thường\n",
            dma_profile_on ? 1 : 0, dma_config_valid ? 1 : 0);
    fpga_log_finish_line(fp, true);
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
    fpga_dma_profile_graph_emit();
    if (dma_profile_graph_active) fpga_dma_profile_graph_end("incomplete", false);
    dma_profile_graph = {};
    dma_timeline_graph = {};
    dma_timeline_graph.begin = dma_profile_clock();
    dma_timeline_graph.invalid = dma_timeline_graph.begin < 0 || dma_profile_transfer_state_current.active ||
                                 dma_profile_finalize_start >= 0;
    dma_profile_graph_seq = graph_seq;
    dma_profile_graph_active = true;
}
void fpga_dma_profile_graph_end(const char * scope, bool decode_token, bool defer_output) {
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
    // The window is frozen above. Emission never contributes to that window.
    fpga_dma_profile_graph_emit();
    dma_pending.ready = true;
    dma_pending.profile = dma_profile_graph;
    dma_pending.timeline = g;
    dma_pending.scope = scope ? scope : "incomplete";
    dma_pending.graph_seq = dma_profile_graph_seq;
    dma_pending.graphs = decode_token ? 1 : 0;
    if (decode_token) { dma_profile_add(dma_profile_decode, dma_profile_graph); ++dma_profile_decode_graphs; }
    dma_profile_graph = {};
    dma_profile_graph_active = false;
    if (!defer_output) fpga_dma_profile_graph_emit();
}
void fpga_dma_profile_graph_emit() {
    if (!dma_pending.ready) return;
    dma_profile_emit(dma_pending.profile, dma_pending.scope.c_str(), dma_pending.graph_seq, dma_pending.graphs);
    dma_timeline_emit(dma_pending.timeline, dma_pending.scope.c_str(), dma_pending.graph_seq, false);
    dma_pending = {};
}
void fpga_dma_profile_finish() {
    fpga_dma_profile_graph_emit();
    if (!fpga_dma_profile_enabled()) return;
    fpga_log_section("Tổng DMA");
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
        fpga_log_token_line(true, "TIMING", data.force_flush,
            "\n  Số phép matmul = %lld | Số lần chạy VPU = %lld\n"
            "Thời gian thực tế\n"
            "  Graph(ms) = %.3f | IP: lần chạy đầu đến kết quả cuối(ms) = %.3f | Tổng matmul FPGA(ms) = %.3f\n"
            "Chuẩn bị và cộng dồn\n  Chuẩn bị(ms) = %.3f | Cộng dồn trên host(ms) = %.3f\n"
            "Truyền từ host đến IP\n  DMA host đến IP(ms) = %.3f | ACT(ms) = %.3f | WEIGHT(ms) = %.3f | SPU SCALE(ms) = %.3f\n"
            "  Ping(ms) = %.3f | Pong(ms) = %.3f | Số byte ACT = %zu | Số byte WEIGHT = %zu | Số byte SPU SCALE = %zu\n"
            "Tính toán trên IP\n  Tổng compute(ms) = %.3f | Ping(ms) = %.3f | Pong(ms) = %.3f | Số job ping = %lld | Số job pong = %lld\n"
            "Truyền từ IP về host và đọc kết quả\n  DMA IP về host(ms) = %.3f | Ping(ms) = %.3f | Pong(ms) = %.3f | Số byte RESULT = %zu\n"
            "  Host đọc kết quả(ms) = %.3f | Ping(ms) = %.3f | Pong(ms) = %.3f | Truyền về và đọc token(ms) = %.3f\n"
            "Phạm vi đo\n  DDR phía host đến IP, compute trên IP, IP về DDR phía host\n"
            "  Kết nối nội bộ PL = không thể quan sát từ host",
            data.matmuls, data.vpu_runs, data.token_wall_us / 1000.0,
            data.device_span_us / 1000.0, data.matmul_wall_us / 1000.0,
            data.prep_us / 1000.0, data.host_accum_us / 1000.0,
            data.h2ip_dma_us / 1000.0, data.act_dma_us / 1000.0,
            data.weight_dma_us / 1000.0, data.scale_dma_us / 1000.0,
            data.bank_h2ip_us[0] / 1000.0, data.bank_h2ip_us[1] / 1000.0,
            data.activation_bytes, data.weight_bytes, data.scale_bytes,
            compute_report_us(data.ip_compute_us) / 1000.0,
            compute_report_us(data.bank_compute_us[0]) / 1000.0,
            compute_report_us(data.bank_compute_us[1]) / 1000.0,
            data.bank_jobs[0], data.bank_jobs[1], data.ip2host_dma_us / 1000.0,
            data.bank_ip2host_us[0] / 1000.0, data.bank_ip2host_us[1] / 1000.0, data.result_bytes,
            data.host_read_us / 1000.0, data.bank_host_read_us[0] / 1000.0,
            data.bank_host_read_us[1] / 1000.0, data.token_read_us / 1000.0);
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
            "\n  Thời gian graph(ms) = %.3f | Thời gian matmul(ms) = %.3f | "
            "Ngoài matmul, suy ra(ms) = %.3f | Khoảng theo dõi IP(ms) = %.3f | Thời gian tính toán trên IP(ms) = %.3f | IP ngoài compute, suy ra(ms) = %.3f | "
            "Tỷ lệ compute, suy ra(%%) = %.2f\nChuẩn bị\n  Tổng thời gian chuẩn bị(ms) = %.3f | Thời gian chọn weight(ms) = %.3f | "
            "Thời gian đóng gói weight trực tiếp(ms) = %.3f | Thời gian đóng gói scale(ms) = %.3f | Thời gian đóng gói ACT(ms) = %.3f | Chuẩn bị khác, suy ra(ms) = %.3f "
            "\nChuyển tiếp giữa các tile\n  Số lần chuyển tiếp = %lld | Tổng overlap chuẩn bị(ms) = %.3f | Số tile chuẩn bị muộn = %lld | Tổng chuẩn bị muộn(ms) = %.3f | Tổng chuẩn bị sớm(ms) = %.3f | "
            "Kết quả sẵn sàng đến lần chạy tiếp(ms) = %.3f | Hoàn tất xử lý đến lần chạy tiếp(ms) = %.3f",
            (double) data.token_wall_us / 1000.0,
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
            FILE * fp = fpga_log_fp();
            fprintf(fp, "[GEMV_TIME_BY_LAYER_TYPE]\n  Graph = %d | Phạm vi = %s\n", data.graph_seq, log_scope(data.scope));
            log_table_t table = {{"Loại layer", "Matmul", "Job tile", "Thực tế(ms)", "Chuẩn bị host(ms)", "Compute FPGA(ms)", "DMA(ms)"}};
            const char * categories[] = {"Attention", "FFN gate", "FFN up", "FFN down", "Khác"};
            for (int i = 0; i < 5; ++i) {
                table.push_back({categories[i], std::to_string(data.category_matmuls[i]),
                    std::to_string(data.category_runs[i]), log_ms(data.category_wall_us[i]),
                    log_ms(data.category_prep_us[i]), log_ms(compute_report_us(data.category_compute_us[i])),
                    log_ms(data.category_dma_us[i])});
            }
            log_table(fp, table);
            fpga_log_finish_line(fp, data.force_flush);

            const double zdma_avg_us = data.zdma_descriptors > 0 ?
                                           (double) data.zdma_elapsed_us / (double) data.zdma_descriptors :
                                           0.0;
            fpga_log_token_line(
                true, "ZDMA", data.force_flush,
                "Graph = %d | Phạm vi = %s | Số descriptor = %lld | Số byte = %zu | Thời gian đã trôi qua(ms) = %.3f | Trung bình descriptor, suy ra(us) = %.3f | "
                "Số descriptor ACT = %lld | Số descriptor WEIGHT = %lld | Số descriptor SPU SCALE = %lld | "
                "Số descriptor RESULT = %lld | Số descriptor khác = %lld | Số byte descriptor tối đa = %zu",
                data.graph_seq, log_scope(data.scope), data.zdma_descriptors, data.zdma_bytes,
                (double) data.zdma_elapsed_us / 1000.0, zdma_avg_us,
                data.zdma_act_descriptors, data.zdma_weight_descriptors,
                data.zdma_scale_descriptors, data.zdma_result_descriptors,
                data.zdma_other_descriptors, data.max_descriptor_bytes);
        }
    }
}

void fpga_log_token_diagnostics(const fpga_token_timing_log_t & data) {
    if (!data.bottleneck_summary_enabled || !data.sampled_detail) return;
    fpga_log_section("Chẩn đoán DMA");
    fpga_log_token_line(true, "ZDMA_POLL", data.force_flush,
        "\n  Graph = %d | Phạm vi = %s\n"
        "  Tổng lượt chờ = %lld | Descriptor hoàn tất ở lần đọc đầu = %lld | Descriptor từng thấy EN bật = %lld\n"
        "  Lượt chờ = lượt đọc trạng thái chưa thỏa điều kiện hoàn tất; không phải thời gian chờ",
        data.graph_seq, log_scope(data.scope), data.zdma_polls,
        data.zdma_zero_poll_descriptors, data.zdma_saw_enabled_descriptors);
}
