#!/usr/bin/env python3
"""Deterministic RAM-only checks of the production DMA logger; no device access."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "ggml/src/ggml-cpu"
source = (CPU / "fpga_host.cpp").read_text()
# Integration guards complement the actual logger and extracted pipeline tests.
init = source[source.index("int fpga_init(void)"):source.index("void fpga_cleanup(void)")]
assert init.index("if (g_fpga_init_complete)") < init.index("fpga_dma_profile_configure()")
assert init.index("if (any_mapped)") < init.index("fpga_dma_profile_configure()") < init.index("map_weight_path_bench_vpu_ddr()")
assert "g_bottleneck_summary_enabled || fpga_dma_profile_enabled()" in init
copy = source[source.index("static bool fpga_dma_copy_one("):source.index("static bool fpga_dma_copy(", source.index("static bool fpga_dma_copy_one("))]
assert copy.index("fpga_dma_profile_transfer profile(tag, bytes)") < copy.index("if (!dma_is_mapped())")
assert copy.index("profile.launch()") < copy.index("g_dma->ZDMA_CH_CTRL2 = ZDMA_CTRL2_START")
assert copy.count("profile.success()") == 1
assert copy.rfind("profile.success()") < copy.rfind("return true")

harness = r'''
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include <sys/stat.h>
#include <cstdio>
#include <cassert>
#include <string>
static FILE * sink;
static long long fake_us = 0;
static unsigned clock_calls = 0;
static bool clock_failure = false;
static long long fake_seconds = -1;
#ifndef _WIN32
static int test_open(const char *, int, ...) { return dup(fileno(sink)); }
#else
static int setenv(const char * name, const char * value, int) { return _putenv_s(name, value); }
static int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif
static int test_clock_gettime(clockid_t id, timespec * ts) {
    assert(id == CLOCK_MONOTONIC);
    ++clock_calls;
    if (clock_failure) return -1;
    ts->tv_sec = fake_seconds >= 0 ? fake_seconds : fake_us / 1000000;
    ts->tv_nsec = (fake_us % 1000000) * 1000;
    return 0;
}
#define open test_open
#define clock_gettime test_clock_gettime
#include "fpga_log.cpp"
#undef clock_gettime
#undef open

static std::string logs() {
    fflush(fpga_log_fp());
    assert(fseek(sink, 0, SEEK_END) == 0);
    const long size = ftell(sink);
    assert(size >= 0 && fseek(sink, 0, SEEK_SET) == 0);
    std::string result(static_cast<size_t>(size), char(0));
    assert(fread(&result[0], 1, result.size(), sink) == result.size());
    assert(fseek(sink, 0, SEEK_END) == 0);
    return result;
}
static std::string visible_logs_without_flush() {
    const int fd = fileno(sink);
    const auto position = lseek(fd, 0, SEEK_CUR);
    struct stat st = {};
    assert(position >= 0 && fstat(fd, &st) == 0 && lseek(fd, 0, SEEK_SET) == 0);
    std::string result(static_cast<size_t>(st.st_size), char(0));
    assert(read(fd, &result[0], static_cast<unsigned>(result.size())) == static_cast<ssize_t>(result.size()));
    assert(lseek(fd, position, SEEK_SET) == position);
    return result;
}
static void configure(const char * value) {
    if (value) setenv("FPGA_DMA_PROFILE", value, 1); else unsetenv("FPGA_DMA_PROFILE");
    assert(fpga_dma_profile_configure());
}
static void disabled_events() {
    fpga_dma_profile_graph_begin(7);
    {
        fpga_dma_profile_transfer t("WEIGHT", 64);
        t.launch();
        fpga_dma_profile_pack_begin();
        fpga_dma_profile_pack_end(true, 0, 8);
        fpga_dma_profile_observe(false, true, false);
        t.success();
    }
    fpga_dma_profile_finalize_begin();
    fpga_dma_profile_finalize_end(1, 2, 3, 64, 32);
    fpga_dma_profile_graph_end("decode_token", true);
    fpga_dma_profile_finish();
}
static void simple_transfer(const char * tag, size_t bytes) {
    fake_us = 10;
    {
        fpga_dma_profile_transfer t(tag, bytes);
        fake_us = 20; t.launch();
        fake_us = 50; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 60;
    }
}
static void test_disabled() {
    unsigned before = clock_calls;
    disabled_events(); // Before configuration: no sink or clock calls.
    assert(clock_calls == before);
    configure(nullptr);
    auto before_log = logs();
    disabled_events();
    assert(clock_calls == before && logs() == before_log);
    configure("0");
    before_log = logs();
    disabled_events();
    assert(clock_calls == before && logs() == before_log);
    setenv("FPGA_DMA_PROFILE", "yes", 1);
    assert(!fpga_dma_profile_configure());
    assert(!fpga_dma_profile_enabled() && logs().find("enabled=0 valid=0") != std::string::npos);
    before_log = logs();
    disabled_events();
    assert(clock_calls == before && logs() == before_log);
}
static void test_intervals() {
    configure("1");
    fpga_dma_profile_graph_begin(10);
    fake_us = 100;
    {
        fpga_dma_profile_transfer t("WEIGHT[1/2]", 1000);
        fake_us = 110; t.launch();
        const unsigned polls_before = clock_calls;
        fpga_dma_profile_observe(false, false, true);
        assert(clock_calls == polls_before); // No productive pack pending.
        fake_us = 120; fpga_dma_profile_pack_begin();
        fake_us = 130; fpga_dma_profile_pack_end(true, 10, 18);
        fake_us = 140; fpga_dma_profile_observe(false, false, true);
        const unsigned after_observation = clock_calls;
        fake_us = 150; fpga_dma_profile_observe(false, false, true);
        assert(clock_calls == after_observation); // Classify once only.
        fake_us = 160; fpga_dma_profile_pack_begin();
        fake_us = 175; fpga_dma_profile_pack_end(true, 18, 22);
        fake_us = 180; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 200;
    }
    const auto & r = dma_profile_graph.rows[1];
    assert(r.descriptors == 1 && r.bytes == 1000 && r.preparation_us == 10);
    assert(r.launch_to_observed_us == 70 && r.callback_pack_us == 25);
    assert(r.callback_pack_count == 2 && r.callback_pack_bytes == 48);
    assert(r.noncallback_launch_interval_us == 45 && r.cleanup_us == 20);
    assert(r.channel_enabled == 1 && r.completion_observed == 1 && r.post_pack_observation_gap_us == 15);
    assert(r.failures == 0 && r.timing_invalid == 0);
    fake_us = 210; fpga_dma_profile_finalize_begin();
    fake_us = 250; fpga_dma_profile_finalize_end(10, 20, 55, 100, 48);
    assert(dma_profile_graph.finalize_count == 1 && dma_profile_graph.finalize_us == 40);
    assert(dma_profile_graph.residual_us == 10 && dma_profile_graph.residual_bytes == 52);
    assert(dma_profile_graph.helper_wait_us == 20 && dma_profile_graph.helper_service_overlap_us == 55);
    fpga_dma_profile_graph_end("decode_token", true);
    fpga_dma_profile_finish();
    auto out = logs();
    assert(out.find("tag=WEIGHT descriptors=1 bytes=1000 preparation_us=10 launch_to_observed_us=70 "
                    "callback_pack_us=25 callback_pack_count=2 callback_pack_bytes=48 "
                    "noncallback_launch_interval_us=45 cleanup_us=20") != std::string::npos);
    assert(out.find("hardware_completion_time=not_measured contention=not_measured") != std::string::npos);
    assert(out.find("callback_bytes_scope=weights_only") != std::string::npos);
    assert(out.find("residual_bytes_scope=weights_only") != std::string::npos);
    assert(out.find("failure_scope=returned_failures fatal_inflight=excluded") != std::string::npos);
    assert(out.find("imported_component_clock=realtime_existing components_not_guaranteed_partition=1") != std::string::npos);
}
static void test_outcomes_and_failures() {
    configure("1");
    // Complete, EN asserted, DONE+EN, neither, and error+DONE/EN-clear.
    const bool errors[]  = {false, false, false, false, true};
    const bool done[]    = {true, false, true, false, true};
    const bool enabled[]= {false, true, true, false, false};
    for (unsigned i = 0; i < 5; ++i) {
        fake_us = 10;
        {
            fpga_dma_profile_transfer t("ACT", 64);
            fake_us = 20; t.launch();
            fake_us = 30; fpga_dma_profile_pack_begin();
            fake_us = 40; fpga_dma_profile_pack_end(true, 0, 8);
            fake_us = 50; fpga_dma_profile_observe(errors[i], done[i], enabled[i]);
            const unsigned observed_calls = clock_calls;
            fpga_dma_profile_observe(errors[i], done[i], enabled[i]);
            assert(clock_calls == observed_calls);
            if (!errors[i]) {
                fake_us = 55; fpga_dma_profile_observe(false, true, false);
                t.success();
            }
            fake_us = 60;
        }
    }
    const auto & r = dma_profile_process.rows[0];
    assert(r.descriptors == 4 && r.failures == 1 && r.failed_bytes == 64);
    assert(r.completion_observed == 1 && r.channel_enabled == 2 && r.incomplete_en_clear == 1 && r.error == 1);
    assert(r.post_pack_observation_gap_us == 50 && r.callback_pack_count == 4 && r.callback_pack_bytes == 128);
    assert(r.callback_pack_us == 40 && r.failed_callback_pack_us == 10 && r.failed_elapsed_us == 50);
    // Failed batch elapsed time belongs only to failed totals; drain does not
    // masquerade as a next normal observation. Next descriptor starts empty.
    fake_us = 100;
    {
        fpga_dma_profile_transfer t("WEIGHT", 32);
        fake_us = 110; t.launch();
        fake_us = 120; fpga_dma_profile_pack_begin();
        fake_us = 135; fpga_dma_profile_pack_end(false, 0, 0);
        fake_us = 150;
    }
    assert(dma_profile_process.rows[1].failed_callback_pack_us == 15);
    simple_transfer("WEIGHT", 16);
    const auto & w = dma_profile_process.rows[1];
    assert(w.descriptors == 1 && w.failures == 1 && w.failed_elapsed_us == 50);
    assert(w.callback_pack_count == 0 && w.completion_observed == 0 && w.channel_enabled == 0);
    // Early return before launch is also a failed descriptor.
    fake_us = 10;
    { fpga_dma_profile_transfer t("OTHER", 1); fake_us = 12; }
    assert(dma_profile_process.rows[4].failures == 1 && dma_profile_process.rows[4].failed_elapsed_us == 2);
    // An empty batch event cannot create productivity or a post-pack sample.
    fake_us = 10;
    {
        fpga_dma_profile_transfer t("OTHER", 2);
        fake_us = 20; t.launch();
        fake_us = 21; fpga_dma_profile_pack_begin();
        fake_us = 22; fpga_dma_profile_pack_end(true, 8, 8);
        unsigned before = clock_calls;
        fpga_dma_profile_observe(false, false, true);
        assert(clock_calls == before);
        fake_us = 30; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 40;
    }
    assert(dma_profile_process.rows[4].callback_pack_count == 0);
    fpga_dma_profile_finish();
}
static void test_scopes_and_reset() {
    configure("1");
    simple_transfer("ACT", 1); // startup stays process-only
    fpga_dma_profile_graph_begin(1);
    simple_transfer("WEIGHT", 2);
    fpga_dma_profile_graph_end("prefill_or_ubatch", false);
    fpga_dma_profile_graph_begin(2);
    simple_transfer("SPU_OUT[1/2]", 4);
    simple_transfer("SPU_SCALE", 8);
    fake_us = 100; fpga_dma_profile_finalize_begin();
    fake_us = 130; fpga_dma_profile_finalize_end(4, 5, 6, 24, 8);
    fpga_dma_profile_graph_end("decode_token", true);
    fpga_dma_profile_graph_end("decode_token", true); // close once
    fpga_dma_profile_graph_begin(3);
    simple_transfer("ACT", 16);
    fpga_dma_profile_graph_end("incomplete", false);
    fpga_dma_profile_graph_begin(4);
    simple_transfer("OTHER", 32);
    fpga_dma_profile_finish(); // flush unfinished as incomplete
    assert(dma_profile_decode_graphs == 1);
    assert(dma_profile_decode.rows[0].descriptors == 0 && dma_profile_decode.rows[1].descriptors == 0);
    assert(dma_profile_decode.rows[2].bytes == 8 && dma_profile_decode.rows[3].bytes == 4);
    assert(dma_profile_decode.rows[4].descriptors == 0 && dma_profile_decode.finalize_count == 1);
    assert(dma_profile_process.rows[0].bytes == 17 && dma_profile_process.rows[1].bytes == 2);
    assert(dma_profile_process.rows[4].bytes == 32);
    auto out = logs();
    assert(out.find("scope=decode_total graph_seq=-1 decode_graphs=1 tag=RESULT descriptors=1 bytes=4") != std::string::npos);
    assert(out.find("scope=process_total_including_startup graph_seq=-1 decode_graphs=1 tag=ACT descriptors=2 bytes=17") != std::string::npos);
    const unsigned before = clock_calls;
    fpga_dma_profile_finish();
    disabled_events();
    assert(clock_calls == before && logs() == out);
    configure("1"); // valid new init lifetime resets all buckets
    assert(dma_profile_decode_graphs == 0 && dma_profile_process.rows[0].bytes == 0);
    fpga_dma_profile_graph_begin(5);
    simple_transfer("RESULT", 64);
    fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_profile_decode.rows[3].bytes == 64 && dma_profile_decode.rows[2].bytes == 0);
    fpga_dma_profile_finish();
}
static void test_invalid_timing() {
    configure("1");
    fake_us = 100;
    {
        fpga_dma_profile_transfer t("WEIGHT", 100);
        fake_us = 90; t.launch();
        fake_us = 110; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 120;
    }
    assert(dma_profile_process.rows[1].timing_invalid == 1 && dma_profile_process.rows[1].descriptors == 0);
    assert(dma_profile_process.rows[1].preparation_us == 0 && dma_profile_process.rows[1].launch_to_observed_us == 0);
    clock_failure = true;
    { fpga_dma_profile_transfer t("ACT", 4); }
    clock_failure = false;
    assert(dma_profile_process.rows[0].failures == 1 && dma_profile_process.rows[0].timing_invalid == 1);
    assert(dma_profile_process.rows[0].failed_elapsed_us == 0);
    fake_us = 20; fpga_dma_profile_finalize_begin();
    fake_us = 10; fpga_dma_profile_finalize_end(1, 2, 3, 4, 0);
    assert(dma_profile_process.finalize_timing_invalid == 1 && dma_profile_process.finalize_count == 0);
    // Failed/invalidation waits have no success event and cannot be counted.
    fake_us = 100; fpga_dma_profile_finalize_begin();
    assert(dma_profile_process.finalize_count == 0);
    fpga_dma_profile_finish();
}
static void timed_transfer(long long entry, long long launch, long long observed, long long exit) {
    fake_us = entry;
    fpga_dma_profile_transfer t("WEIGHT", 128);
    fake_us = launch; t.launch();
    fake_us = observed; fpga_dma_profile_observe(false, true, false);
    t.success(); fake_us = exit;
}
static void assert_partition(const dma_profile_timeline & g) {
    assert(!g.invalid && g.valid_graphs > 0);
    assert(g.window_us == g.head_us + g.service_us + g.gap_us + g.tail_us + g.no_dma_us);
}
static void test_timeline_partition() {
    configure("1");
    fake_us = 100;
    const unsigned before = clock_calls;
    fpga_dma_profile_graph_begin(20);
    timed_transfer(110, 120, 150, 160);
    timed_transfer(190, 200, 240, 250);
    fake_us = 300; fpga_dma_profile_graph_end("decode_token", true);
    assert(clock_calls == before + 2 + 2 * 4); // Only two new boundary reads.
    const auto first = dma_timeline_decode;
    assert_partition(first);
    assert(first.begin == 100 && first.end == 300 && first.window_us == 200);
    assert(first.head_us == 20 && first.service_us == 70 && first.gap_us == 50 && first.tail_us == 60);
    assert(first.descriptors == 2 && first.gap_count == 1 && first.max_gap_us == 50);
    fake_us = 1000; fpga_dma_profile_graph_begin(21); // Intergraph 700 us is excluded.
    timed_transfer(1005, 1010, 1020, 1030);
    timed_transfer(1040, 1050, 1060, 1070);
    fake_us = 1100; fpga_dma_profile_graph_end("decode_token", true);
    assert_partition(dma_timeline_decode);
    assert(dma_timeline_decode.window_us == 300 && dma_timeline_decode.service_us == 90);
    assert(dma_timeline_decode.gap_us == 80 && dma_timeline_decode.max_gap_us == 50);
    assert(dma_timeline_decode.valid_graphs == 2 && dma_timeline_decode.gap_count == 2);
    fpga_dma_profile_finish();
    const auto out = logs();
    assert(out.find("observed_service_fraction=0.300000 observed_gap_fraction=0.266667") != std::string::npos);
    assert(out.find("endpoints=valid_window_envelope_not_denominator") != std::string::npos);
}
static void test_timeline_delay_and_scopes() {
    configure("1");
    fake_us = 0; fpga_dma_profile_graph_begin(30);
    {
        fake_us = 10; fpga_dma_profile_transfer t("ACT", 64);
        fake_us = 20; t.launch();
        // Hardware may finish while packing; the profiler only knows time 80.
        fake_us = 30; fpga_dma_profile_pack_begin();
        fake_us = 70; fpga_dma_profile_pack_end(true, 0, 8);
        fake_us = 80; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 90;
    }
    timed_transfer(100, 110, 130, 140);
    fake_us = 150; fpga_dma_profile_graph_end("decode_token", true);
    assert_partition(dma_timeline_decode);
    assert(dma_timeline_decode.service_us == 80 && dma_timeline_decode.gap_us == 30);
    assert(dma_profile_decode.rows[0].callback_pack_us == 40); // Already inside service.
    fake_us = 200; fpga_dma_profile_graph_begin(31);
    fake_us = 230; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_decode.no_dma_graphs == 1 && dma_timeline_decode.no_dma_us == 30);
    assert_partition(dma_timeline_decode);
    fake_us = 300; fpga_dma_profile_graph_begin(32);
    timed_transfer(310, 320, 330, 340);
    fake_us = 350; fpga_dma_profile_graph_end("prefill_or_ubatch", false);
    assert(dma_timeline_decode.valid_graphs == 2 && dma_timeline_decode.window_us == 180);
    assert(dma_timeline_all_graphs.valid_graphs == 3 && dma_timeline_all_graphs.window_us == 230);
    timed_transfer(360, 370, 380, 390); // Startup/outside-graph descriptor is not added.
    assert(dma_timeline_all_graphs.descriptors == 3);
    fake_us = 400; fpga_dma_profile_graph_begin(33);
    fake_us = 410; fpga_dma_profile_finish(); // Unknown incomplete graph is not called decode.
    assert(dma_timeline_all_graphs.excluded_graphs == 1 && dma_timeline_decode.excluded_graphs == 0);
    configure("1");
    assert(dma_timeline_decode.window_us == 0 && dma_timeline_all_graphs.excluded_graphs == 0);
    fpga_dma_profile_finish();
}
static void test_timeline_invalid() {
    for (int mode = 0; mode < 9; ++mode) {
        configure("1");
        fake_us = 100; fpga_dma_profile_graph_begin(40 + mode);
        {
            fake_us = 110; fpga_dma_profile_transfer t("ACT", 64);
            fake_us = 120; t.launch();
            if (mode == 0) { // Returned error.
                fpga_dma_profile_observe(true, false, false);
            } else {
                if (mode == 1) t.launch(); // Repeated launch.
                if (mode == 2) { fpga_dma_profile_transfer nested("RESULT", 4); }
                if (mode == 3) fake_us = 115; else fake_us = 130;
                if (mode != 4) fpga_dma_profile_observe(false, true, false); // Missing observation.
                t.success();
            }
            fake_us = 140;
            if (mode == 5) fpga_dma_profile_graph_end("decode_token", true); // In flight at boundary.
        }
        if (mode == 6) { fake_us = 150; fpga_dma_profile_finalize_begin(); }
        fake_us = mode == 7 ? 130 : 200;
        if (mode == 8) clock_failure = true;
        fpga_dma_profile_graph_end("decode_token", true);
        clock_failure = false;
        assert(dma_timeline_decode.valid_graphs == 0 && dma_timeline_decode.excluded_graphs == 1);
        assert(dma_timeline_decode.window_us == 0 && dma_timeline_decode.descriptors == 0);
        fpga_dma_profile_finish();
    }
    configure("1");
    fake_us = 10;
    {
        fpga_dma_profile_transfer t("ACT", 64); // Originates before graph.
        fake_us = 20; t.launch();
        fake_us = 30; fpga_dma_profile_graph_begin(60);
        fake_us = 40; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 50;
    }
    fake_us = 60; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_decode.excluded_graphs == 1);
    fake_us = 70; fpga_dma_profile_graph_begin(61);
    fake_us = 80; fpga_dma_profile_graph_begin(62); // Prior scope closes incomplete.
    fake_us = 90; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_all_graphs.excluded_graphs == 2 && dma_timeline_decode.valid_graphs == 1);
    fpga_dma_profile_finish();
    configure("1");
    clock_failure = true; fpga_dma_profile_graph_begin(63); clock_failure = false;
    fake_us = 100; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_decode.excluded_graphs == 1);
    fpga_dma_profile_finish();
    configure("1");
    fake_us = 0; fpga_dma_profile_graph_begin(64);
    {
        fake_us = 10; fpga_dma_profile_transfer t("ACT", 64);
        fake_us = 20; t.launch();
        fake_us = 30; fpga_dma_profile_graph_end("decode_token", true);
        fake_us = 40; fpga_dma_profile_graph_begin(65);
        fake_us = 50; fpga_dma_profile_observe(false, true, false);
        t.success(); fake_us = 60;
    }
    fake_us = 70; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_decode.excluded_graphs == 2 && dma_timeline_decode.descriptors == 0);
    fake_us = 80; fpga_dma_profile_graph_begin(66);
    fake_us = 90; fpga_dma_profile_graph_end("unknown_scope", true);
    assert(dma_timeline_decode.excluded_graphs == 3);
    fake_us = 100; fpga_dma_profile_graph_begin(67);
    fake_us = 110; fpga_dma_profile_finalize_begin();
    fake_us = 105; fpga_dma_profile_finalize_end(0, 0, 0, 0, 0);
    fake_us = 120; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_decode.excluded_graphs == 4);
    fpga_dma_profile_finish();
}
static void test_timeline_overflow() {
    const auto limit = std::numeric_limits<long long>::max();
    long long sum = limit;
    assert(!dma_timeline_add_value(sum, 1) && sum == limit);
    dma_profile_timeline total, sample;
    total.valid_graphs = 1; total.window_us = limit;
    sample.valid_graphs = 1; sample.window_us = 1;
    dma_timeline_add(total, sample);
    assert(total.invalid && total.valid_graphs == 1 && total.window_us == limit);
    dma_timeline_emit(total, "test_overflow", -1, true);
    assert(logs().find("aggregate_valid=0 begin_mono_us=-1 end_mono_us=-1 profile_window_us=-1") != std::string::npos);
    fake_seconds = limit / 1000000;
    fake_us = limit % 1000000;
    assert(dma_profile_clock() == limit);
    ++fake_us;
    assert(dma_profile_clock() == -1);
    fake_seconds = -1;
}
static void test_timeline_empty_and_visibility() {
    configure("1");
    const auto offset = logs().size();
    fpga_dma_profile_finish();
    auto visible = visible_logs_without_flush().substr(offset);
    assert(visible.find("[DMA_PROFILE_TIMELINE] scope=decode_total graph_seq=-1 valid_graphs=0 excluded_graphs=0 "
                        "no_dma_graphs=0 aggregate_valid=0") != std::string::npos);
    assert(visible.find("[DMA_PROFILE_TIMELINE] scope=all_graphs_total") != std::string::npos);
    assert(visible.find("observed_service_fraction=-1.000000 observed_gap_fraction=-1.000000") != std::string::npos);
    configure("1");
    const auto excluded_offset = logs().size();
    fake_us = 0; fpga_dma_profile_graph_begin(70);
    { fake_us = 10; fpga_dma_profile_transfer t("ACT", 1); }
    fake_us = 20; fpga_dma_profile_graph_end("decode_token", true);
    fpga_dma_profile_finish();
    visible = visible_logs_without_flush().substr(excluded_offset);
    assert(visible.find("scope=decode_total graph_seq=-1 valid_graphs=0 excluded_graphs=1 "
                        "no_dma_graphs=0 aggregate_valid=0") != std::string::npos);
    configure("1");
    fake_us = 0; fpga_dma_profile_graph_begin(71);
    fake_us = 10; fpga_dma_profile_graph_end("decode_token", true);
    assert(dma_timeline_decode.valid_graphs == 1 && dma_timeline_decode.no_dma_us == 10);
    fpga_dma_profile_finish();
}
int main() {
    sink = tmpfile();
    assert(sink);
    static char output_buffer[65536];
    assert(setvbuf(sink, output_buffer, _IOFBF, sizeof(output_buffer)) == 0);
    test_disabled();
    test_intervals();
    test_outcomes_and_failures();
    test_scopes_and_reset();
    test_invalid_timing();
    test_timeline_partition();
    test_timeline_delay_and_scopes();
    test_timeline_invalid();
    test_timeline_overflow();
    test_timeline_empty_and_visibility();
}
'''
if os.name == "nt":
    # Linux compiles the complete logger above. Windows lacks sys/resource.h;
    # compile the exact production profiler with only its file-sink dependency
    # supplied by this harness. No device code or reimplemented timing logic.
    logger = (CPU / "fpga_log.cpp").read_text()
    start = logger.index("namespace {\n// Serialized by the host's existing submission/lifecycle mutex.")
    end = logger.index("\nnamespace {\nvoid fpga_log_token_line", start)
    profiler = '#include "fpga_log.h"\nFILE * fpga_log_fp() { return sink; }\n'
    profiler += 'void fpga_log_finish_line(FILE * fp, bool force) { if (force) fflush(fp); }\n'
    harness = harness.replace('#include "fpga_log.cpp"', profiler + logger[start:end])
with tempfile.TemporaryDirectory(prefix="fpga-dma-profile-") as tmp:
    cpp = Path(tmp) / "test.cpp"
    binary = Path(tmp) / "test"
    cpp.write_text(harness)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-DUSE_FPGA", "-I" + str(CPU), str(cpp), "-o", str(binary)], check=True, timeout=60)
    result = subprocess.run([str(binary)], capture_output=True, timeout=30)
    if result.returncode:
        print(result.stderr.decode(errors="replace"), end="")
        result.check_returncode()
    assert not result.stdout and not result.stderr, "DMA profile must be file-only"
print("PASS: DMA logger intervals, callbacks, failures, scopes, reset, disabled clocks, file-only output; timeline partition, gaps, delayed observation, boundaries, overflow")
