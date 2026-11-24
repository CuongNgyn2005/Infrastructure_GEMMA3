#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstddef>

// Opt-in DMA measurements. Called only on the serialized host submission path;
// helpers never update this state. Disabled hooks do not read a clock or log.
bool fpga_dma_profile_configure();
bool fpga_dma_profile_enabled();
void fpga_dma_profile_graph_begin(int graph_seq);
void fpga_dma_profile_graph_end(const char * scope, bool decode_token, bool defer_output = false);
void fpga_dma_profile_graph_emit();
void fpga_dma_profile_log_config();
void fpga_dma_profile_finish();
class fpga_dma_profile_transfer {
public:
    fpga_dma_profile_transfer(const char * tag, size_t bytes);
    ~fpga_dma_profile_transfer();
    void launch();
    void success();
    fpga_dma_profile_transfer(const fpga_dma_profile_transfer &) = delete;
    fpga_dma_profile_transfer & operator=(const fpga_dma_profile_transfer &) = delete;
private:
    bool enabled_;
    bool success_ = false;
};
void fpga_dma_profile_observe(bool error, bool done, bool enabled);
void fpga_dma_profile_pack_begin();
void fpga_dma_profile_pack_end(bool ok, size_t words_before, size_t words_after);
void fpga_dma_profile_finalize_begin();
void fpga_dma_profile_finalize_end(long long residual_us, long long helper_wait_us,
                                   long long helper_service_us, size_t main_bytes, size_t callback_bytes);

// Shared logging sink used by the FPGA host. Driver-specific diagnostics
// remain in fpga_host.cpp and pass only formatted records through this API.
FILE * fpga_log_fp();
void fpga_log_section(const char * title);
void fpga_log_graph_header(int graph_seq, const char * scope);
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
// One startup-only CPU RAM copy benchmark; no FPGA/physical-memory access.
void   fpga_log_ram_bandwidth(bool defer_output = false);
void   fpga_log_startup_begin();
#else
inline void fpga_log_load_checkpoint(const char *) {}
inline void fpga_log_startup_begin() {}
#endif

#ifdef USE_FPGA
void fpga_log_runtime_summary(double load_ms, double prompt_ms, int prompt_tokens,
                              double decode_ms, int decode_runs);
#endif

void fpga_log_prompt_weight_reuse(const char * tensor, long long columns, long long jobs,
                                  unsigned long long avoided_bytes);

// Token report values: strings are borrowed only during each synchronous call.
// Deferred DMA reports own their scope string and copied counters; no driver or hardware reads.
struct fpga_token_timing_log_t {
    bool timing_enabled;
    bool bottleneck_summary_enabled;
    bool compute_watch_enabled;
    bool sampled_detail;
    bool force_flush;
    int next_graph_seq;
    long long ubatch_tokens;
    const char * scope;
    const char * reason;
    long long token_wall_us;
    long long h2ip_dma_us;
    long long token_read_us;
    long long device_span_us;
    size_t max_descriptor_bytes;
    int graph_seq;
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
    long long p3_jobs;
    long long drain_wait_us;
    long long drain_wait_jobs;
    long long scratch_dma_us;
    long long scratch_bytes;
    long long scratch_mmio_jobs;
    long long scratch_desc;
    long long p3_reject_delta;
    long long p3_mode_switches;

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
    long long zdma_descriptors;
    long long zdma_polls;
    long long zdma_zero_poll_descriptors;
    long long zdma_saw_enabled_descriptors;
    long long zdma_elapsed_us;
    size_t zdma_bytes;
    long long zdma_act_descriptors;
    long long zdma_weight_descriptors;
    long long zdma_scale_descriptors;
    long long zdma_result_descriptors;
    long long zdma_other_descriptors;
    // Category order: attention, FFN gate, FFN up, FFN down, other.
    long long category_matmuls[5];
    long long category_runs[5];
    long long category_wall_us[5];
    long long category_prep_us[5];
    long long category_compute_us[5];
    long long category_dma_us[5];
    long long bank_h2ip_us[2];
    long long bank_compute_us[2];
    long long bank_ip2host_us[2];
    long long bank_host_read_us[2];
    long long bank_jobs[2];
    size_t activation_bytes;
    size_t weight_bytes;
    size_t scale_bytes;
    size_t result_bytes;
};
void fpga_log_token_timing(const fpga_token_timing_log_t & data);
void fpga_log_token_diagnostics(const fpga_token_timing_log_t & data);

void fpga_log_p3_split(const fpga_token_timing_log_t & data, bool enabled, bool scratch_mmio = false);
