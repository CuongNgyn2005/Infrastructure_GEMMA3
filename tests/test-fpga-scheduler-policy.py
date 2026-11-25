#!/usr/bin/env python3
# RAM-only production routing regression: mandatory ping-pong and decode/prompt stream mode.
import os
from pathlib import Path
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / 'ggml/src/ggml-cpu/fpga_host.cpp').read_text()
def definition(signature):
    start = source.index(signature)
    depth = 0
    for pos in range(source.index('{', start), len(source)):
        depth += (source[pos] == '{') - (source[pos] == '}')
        if depth == 0:
            return source[start:pos + 1]
    raise AssertionError(signature)
router = definition('static bool fpga_hw_q8_0_matmul_dma_to_ip(const')
dispatch = router[router.index('    if (use_pl_scale_path) {'):]   # ends with the function's closing brace
assert 'single_bank(' not in source
assert 'qualify_tiles' not in source and 'FPGA_PIPELINE_' not in source
# Initialization stops without ping-pong, before the pipeline is published.
init = definition('int fpga_init(void) {')
assert (init.index('if (g_p2_init_requested && !g_pingpong_scheduler_enabled) {')
        < init.index('fpga_fatal("P2 pack DMA pipeline requires admitted ping-pong and the P2 scale stream");')
        < init.index('g_p2_pack_dma_pipeline_enabled = true;'))
code = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
template<class... T> void logline(T...) {}
#define LOGE logline
struct ggml_tensor { float data[4] = {91,92,93,94}; };
struct block_q8_0_t {};
struct fpga_stage_totals_t { long long p3_mode_switches = 0; };
bool g_pingpong_scheduler_enabled = false, g_p3_matmul_active = false;
long long g_p3_mode_switches = 0;
std::string events;
unsigned last_mode = 99;
bool stream_ok = true;
bool fpga_enforce_p2_stream_mode(const char *, void *, uint32_t mode) {
    events += 'E'; last_mode = mode; ++g_p3_mode_switches; return stream_ok;
}
template<class... T> bool fpga_hw_q8_0_matmul_dma_to_ip_pipelined(T...) { events += 'G'; return true; }
bool route(bool use_pl_scale_path, int64_t m, fpga_stage_totals_t * totals) {
    ggml_tensor tensor;
    const ggml_tensor * src0 = &tensor, * dst = &tensor;
    std::vector<block_q8_0_t> act_blocks_all(8);
    std::vector<float> act_scales, accum;
    const char * tensor_name = "test";
    int layer_id = 0;
    int64_t k=32, n=1, nb=1;
'''
code += dispatch
code += r'''
int main() {
    for (int64_t m : {int64_t(1), int64_t(2), int64_t(7)}) {
        for (int admitted = 0; admitted < 2; ++admitted) {
            events.clear(); g_pingpong_scheduler_enabled = admitted; stream_ok = true;
            fpga_stage_totals_t totals;
            assert(route(true, m, &totals) == bool(admitted));
            assert(events == (admitted ? "EG" : "E"));            // no serialized fallback
            assert(last_mode == (m == 1 ? 1U : 0U));              // decode P3, prompt P2
            assert(g_p3_matmul_active == (m == 1));
            assert(totals.p3_mode_switches == 1);
        }
    }
    events.clear(); g_pingpong_scheduler_enabled = true; stream_ok = false;
    assert(!route(true, 1, nullptr) && events == "E");
    events.clear(); stream_ok = true;
    assert(!route(false, 1, nullptr) && events.empty());          // no raw route
    std::puts("PASS: mandatory ping-pong route; decode P3 / prompt P2 stream mode; no raw or serialized fallback");
}
'''
with tempfile.TemporaryDirectory(prefix='fpga-scheduler-policy-') as tmp:
    cpp = Path(tmp) / 'test.cpp'
    exe = Path(tmp) / ('test.exe' if os.name == 'nt' else 'test')
    cpp.write_text(code)
    sanitizer = [] if os.name == 'nt' else ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-Wall', '-Wextra', '-Werror', *sanitizer, str(cpp), '-o', str(exe)], check=True, timeout=60)
    subprocess.run([str(exe)], check=True, timeout=30)
