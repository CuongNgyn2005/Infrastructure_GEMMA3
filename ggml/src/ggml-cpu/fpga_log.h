#pragma once

#include <cstdarg>
#include <cstdio>

// Shared logging sink used by the FPGA host. Driver-specific diagnostics
// remain in fpga_host.cpp and pass only formatted records through this API.
FILE * fpga_log_fp();
void   fpga_log_set_flush_every(int flush_every);
void   fpga_log_finish_line(FILE * fp, bool force_flush);
void   fpga_log_vline(const char * tag, bool force_flush, const char * fmt, va_list ap);
// Coarse CLI latency records are file-only, even if the sink failed to open.
void   fpga_log_latency(const char * fmt, ...);
// Failure-only P2 evidence; never redirect these records to the terminal.
void   fpga_log_q16_audit(const char * fmt, ...);
void   fpga_log_result_overlap(int graph_seq, bool enabled, long long jobs, long long host_us);
struct fpga_pack_breakdown_log_t {
    long long total_us;
    long long serial_pack_us;
    long long dispatch_us;
    long long main_pack_us;
    long long caller_fence_us;
    long long caller_wait_us;
    long long helper_service_us;
    long long serial_jobs;
    long long parallel_jobs;
    long long reserve_lock_us;
    long long reserve_body_us;
    long long reserve_unlock_us;
    long long submit_lock_us;
    long long submit_publish_us;
    long long submit_signal_us;
    long long submit_unlock_us;
};
void fpga_log_pack_breakdown(const char * scope, int graph_seq, long long tokens,
                             const fpga_pack_breakdown_log_t & data);
#ifdef USE_FPGA
void   fpga_log_load_checkpoint(const char * phase);
#else
inline void fpga_log_load_checkpoint(const char *) {}
#endif

#ifdef USE_FPGA
void fpga_log_runtime_summary(double load_ms, double prompt_ms, int prompt_tokens,
                              double decode_ms, int decode_runs);
#endif

void fpga_log_prompt_weight_reuse(const char * tensor, long long columns, long long jobs,
                                  unsigned long long avoided_bytes);
