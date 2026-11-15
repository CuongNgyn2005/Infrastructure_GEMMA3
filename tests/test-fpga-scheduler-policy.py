#!/usr/bin/env python3
"""RAM-only production routing and exact tile-qualification boundary regression."""
import os
from pathlib import Path
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / 'ggml/src/ggml-cpu/fpga_host.cpp').read_text()
def between(a, b, start=0):
    i = source.index(a, start)
    return source[i:source.index(b, i)]
runner = between('static bool fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_qualify_tiles(',
                 'static bool fpga_hw_q8_0_matmul_dma_to_ip(')
assert 'store_dst_value' not in runner and ' dst' not in runner
assert 'P2_SINGLE_BANK_TILE' in runner and 'scope=qualification' in runner
policy = between('    const bool pipeline_enable_env ', '    g_q8_source_audit_only =')
assert source.index(policy) < source.index('phase=mapping begin')
reject_enable = between('        if (pipeline_enable_env) {', '        if (g_weight_cache_enabled)')
dispatch_start = source.index('static bool fpga_hw_q8_0_matmul_dma_to_ip(')
dispatch = between('    if (use_pl_scale_path) {', '    uint32_t tile_id', dispatch_start)
assert 'single_bank(' not in source
active = between('    g_active_matmul_pingpong   =', ';', dispatch_start) + ';'
code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
template<class... T> void logline(T...) {}
#define LOGE logline
#define LOGINIT logline
#define LOGSTAGE logline
struct ggml_tensor { float data[4] = {91,92,93,94}; };
struct block_q8_0_t {};
struct fpga_weight_cache_entry_t {};
struct fpga_stage_totals_t {};
struct fpga_tile_job_t {};
struct { std::vector<float> contract_actual; } g_scratch;
int g_pl_scale_contract_check_limit, g_contract_check_limit, g_p2_tile_limit;
int g_vpu_max_rows = 1;
long long g_p2_tile_q16_checks, g_pl_scale_jobs, g_pl_scale_banks;
bool g_p2_tile_contract_boundary_reached, g_pingpong_scheduler_enabled;
bool g_p2_init_requested, g_active_matmul_pingpong, enable_env, disable_env;
bool verify_ok = true;
std::string events;
const void * expected_source;
bool env_flag_enabled(const char * name) {
    return std::strcmp(name,"FPGA_PIPELINE_ENABLE") == 0 ? enable_env : disable_env;
}
template<class... T> void fpga_fatal(T...) { throw std::runtime_error("policy"); }
bool fpga_enforce_p2_stream_mode(const char *) { events += 'E'; return true; }
bool fpga_p2_cumulative_tile_limit_reached(long long count, int limit) { return limit > 0 && count >= limit; }
bool fpga_p2_cumulative_tile_state_consistent(long long count, int limit, bool boundary) {
    return count <= limit && boundary == (count == limit);
}
int packed_q8_group_blocks_for_rows(int, int) { return 1; }
bool should_log_detail_run(uint32_t) { return true; }
template<class... T> bool fpga_prepare_q8_tile_job(fpga_tile_job_t &, const ggml_tensor *, const void * ptr, T...) {
    assert(ptr == expected_source); events += 'P'; return true;
}
template<class... T> bool fpga_submit_q8_tile_job(T...) { events += 'S'; return true; }
template<class... T> bool fpga_wait_and_drain_q8_tile_job(T...) { events += 'D'; return true; }
template<class... T> bool fpga_pl_scale_contract_verify_q16_tile(fpga_tile_job_t &, const void * ptr, T...) {
    assert(ptr == expected_source); events += 'V'; return verify_ok;
}
bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t &, std::vector<float> &, fpga_stage_totals_t *) {
    assert(g_p2_tile_q16_checks == g_pl_scale_jobs); events += 'R'; return true;
}
template<class... T> bool fpga_p2_complete_tile_contract_boundary(T...) {
    assert(events.back() == 'R' && g_p2_tile_q16_checks == g_p2_tile_limit);
    events += 'B'; g_p2_tile_contract_boundary_reached = true; return true;
}
template<class... T> bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(T...) { events += 'G'; return true; }
'''
code += runner
code += '\nvoid configure() {\n' + policy + '\nif (g_pl_scale_contract_check_limit > 0) {\n' + reject_enable + '}\n}\n'
code += '\nvoid active_route() {\n' + active + '\n}\n'
code += r'''
bool route(bool use_pl_scale_path, bool pl_scale_contract_active) {
    ggml_tensor tensor;
    const ggml_tensor * src0 = &tensor, * dst = &tensor;
    std::vector<block_q8_0_t> act_blocks_all(8);
    std::vector<float> act_scales, accum;
    fpga_weight_cache_entry_t * weight_cache = nullptr;
    const void * weight_data_base = expected_source;
    fpga_stage_totals_t * totals = nullptr;
    const char * tensor_name = "test";
    int layer_id = 0;
    int64_t k=32, n=1, m=1, nb=1;
'''
code += dispatch + "    events += 'C'; return true;\n}\n"
code += r'''
void reset(int limit=3) {
    events.clear(); g_pl_scale_contract_check_limit=1; g_contract_check_limit=0;
    g_p2_tile_limit=limit; g_p2_tile_q16_checks=0; g_pl_scale_jobs=0; g_pl_scale_banks=0;
    g_p2_tile_contract_boundary_reached=false; verify_ok=true;
}
int main() {
    for (int p2=0; p2<2; ++p2) for (int raw=0; raw<2; ++raw)
    for (int qual=0; qual<2; ++qual) for (int en=0; en<2; ++en) for (int dis=0; dis<2; ++dis) {
        g_p2_init_requested=p2; g_contract_check_limit=raw; g_pl_scale_contract_check_limit=qual;
        enable_env=en; disable_env=dis;
        bool rejected=false;
        try { configure(); } catch (const std::runtime_error &) { rejected=true; }
        assert(rejected == ((en && dis) || (dis && p2 && !raw && !qual) || (en && qual)));
        for (int admitted=0; admitted<2; ++admitted) {
            g_pingpong_scheduler_enabled=admitted; active_route();
            assert(g_active_matmul_pingpong == (admitted && !raw && !qual));
        }
    }
    int immutable=7; expected_source=&immutable;
    reset(); g_pl_scale_contract_check_limit=0;
    g_pingpong_scheduler_enabled=true; assert(route(true,false)); assert(events=="EG");
    events.clear(); g_pingpong_scheduler_enabled=false; assert(!route(true,false)); assert(events=="E");
    events.clear(); assert(route(false,false)); assert(events=="C");
    for (int admitted=0; admitted<2; ++admitted) {
        reset(1); g_pingpong_scheduler_enabled=admitted;
        assert(route(true,true)); assert(events=="EEPSDVRB");
    }
    ggml_tensor src;
    std::vector<block_q8_0_t> acts(8); std::vector<float> accum;
    auto qualify = [&](int columns) {
        return fpga_hw_q8_0_matmul_dma_to_ip_pl_scale_qualify_tiles(
            &src, acts, nullptr, expected_source, nullptr, "test", 0, 32, 1, columns, 1, accum);
    };
    reset(3);
    assert(qualify(2)); assert(events=="EPSDVRPSDVR" && g_p2_tile_q16_checks==2);
    events.clear(); assert(qualify(4)); assert(events=="EPSDVRB" && g_p2_tile_q16_checks==3);
    events.clear(); assert(!qualify(1)); assert(events.empty());
    reset(); verify_ok=false; assert(!qualify(2)); assert(events=="EPSDV" && g_p2_tile_q16_checks==0);
    reset(0); assert(!qualify(1)); assert(events.empty());
    reset(); g_pl_scale_contract_check_limit=0; assert(!qualify(1)); assert(events.empty());
    std::puts("PASS: production ping-pong policy, pre-mapping rejection, raw compatibility, exact qualification boundary/order");
}
'''
with tempfile.TemporaryDirectory(prefix='fpga-scheduler-policy-') as tmp:
    cpp = Path(tmp) / 'test.cpp'
    exe = Path(tmp) / ('test.exe' if os.name == 'nt' else 'test')
    cpp.write_text(code)
    sanitizer = [] if os.name == 'nt' else ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-Wall', '-Wextra', '-Werror', *sanitizer, str(cpp), '-o', str(exe)], check=True, timeout=60)
    subprocess.run([str(exe)], check=True, timeout=30)
