#!/usr/bin/env python3
"""Offline attention control/ownership tests; no device files or FPGA execution.

Compile extracted production integration functions against a RAM/register/P2
transport. PROT_NONE pages make any CPU read of attention F32/Q8 output fail.
The P2 stub checks transport and ownership, not attention/Wo arithmetic.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "ggml/src/ggml-cpu"
SOURCE = (CPU / "fpga_host.cpp").read_text()
GRAPH = (ROOT / "src/llama-graph.cpp").read_text()


def definition(signature):
    search = 0
    while True:
        start = SOURCE.index(signature, search)
        body = SOURCE.index("{", start)
        if ";" not in SOURCE[start:body]:
            depth = 0
            for end in range(body, len(SOURCE)):
                depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
                if depth == 0:
                    return SOURCE[start:end + 1]
        search = body + 1


PROJECT = definition("static bool fpga_attention_project_retained(")
COMPUTE = definition("static bool fpga_attention_compute(")
DMA = definition("static bool fpga_dma_copy_one(")
PREPARE = definition("static bool fpga_prepare_q8_tile_job(")
SUBMIT = definition("static bool fpga_submit_q8_tile_job(")
CALLBACK = definition("void fpga_attention_wo_callback(")
assert "!job.retained_activation && gb < group_blocks" in PREPARE
assert "retained_act_src_off = 0" in SOURCE
assert '"ATTENTION_ACT", nullptr, true' in SUBMIT
assert "fpga_attention_retained_dma_source(src_phys, bytes)" in DMA
assert "if (device_produced_source)" in DMA
assert "if (!device_produced_source && !fpga_p2_ddr_sync(src_ddr_off, bytes, false, tag))" in DMA
assert "if (ith != 0) { return; }" in CALLBACK
assert "fpga_attention_invalidate_host_state()" in CALLBACK
assert "quantize" not in COMPUTE + PROJECT
assert "Q8_OFFSET" not in PROJECT and "F32_OFFSET" not in PROJECT
assert "ATTENTION_SCALE_METADATA" in PROJECT
assert "fpga_dma_copy(DDR_BASE_PHYS + job.retained_act_src_off" in SUBMIT
assert "args, 5, fpga_attention_wo_callback, 1" in GRAPH
fused = GRAPH.index('cb(cur, "fpga_attention_wo", il)')
cache = GRAPH.rfind("ggml_build_forward_expand(gf, mctx_cur->cpy_v", 0, fused)
assert cache >= 0 and cache < GRAPH.rfind("if (fpga_attention_requested())", 0, fused)
assert "build_attn_mha" in GRAPH[fused:fused + 1000]
assert "build_lora_mm" in GRAPH[fused:fused + 2000]
assert "fpga_attention_invalidate_host_state()" in definition("void fpga_reset_kv_cache(")

# Exercise the production admission expression too: extended total mapping must
# not enlarge the fixed 32-MiB weight residency range or change normal admission.
mapping_expression = re.search(r"const bool residency_mapping_fits = ([\s\S]*?);", SOURCE).group(1)

harness = r'''
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <sys/mman.h>
#include "fpga_attention_contract.h"
#include "fpga_attention_abi.h"
namespace attention_abi = fpga_attention_abi;
#define LOGI(...) ((void)0)
enum { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
struct ggml_tensor {
    int type; void * data; int64_t ne[4]; size_t nb[4]; size_t storage;
    ggml_tensor * src[5] = {}; const char * name = "fixture.wo";
};
struct block_q8_0_t { uint16_t d; int8_t qs[32]; };
struct fpga_stage_totals_t {};
struct fpga_tile_job_t { int rows, blocks; int64_t row0, block0; uint32_t source; };
struct fpga_attention_layer_state { fpga_attention::kv_mirror mirror; };
static const uint64_t DDR_BASE_PHYS = attention_abi::DDR_BASE;
static const uint32_t WEIGHT_BASE = 0x10000, SPU_PARAM_BASE = 0x100000;
static const size_t VPU_QK8_0 = 32;
static size_t g_ddr_map_size = 64U << 20;
static uint8_t * g_ddr;
static bool g_attention_enabled = true, g_attention_admitted = true;
static bool g_spu_q8_scale_stream_supported = true, g_pingpong_scheduler_enabled = true;
static std::vector<attention_abi::layer_layout> g_attention_layouts;
static std::vector<std::unique_ptr<fpga_attention_layer_state> > g_attention_layers;
static fpga_attention::retained_q8_state g_attention_retained;
static fpga_attention::retained_q8_handle g_attention_handle;
static attention_abi::command g_attention_command;
static uint64_t g_attention_next_query = 1;
static unsigned long long g_next_matmul_call_id = 1, g_active_matmul_call_id;
static int g_active_matmul_graph_seq, g_current_seq_pos, g_active_matmul_layer_id;
static int64_t g_active_matmul_shape_k, g_active_matmul_shape_n, g_active_matmul_shape_m;
static bool g_active_matmul_cpu_shadow, g_active_matmul_pingpong;
static const char * g_active_matmul_tensor_name;
static int g_vpu_max_rows = 3;
static int64_t g_ip_timeout_us = 100;
static size_t g_pl_scale_jobs, g_pl_scale_banks;
static size_t kv_bytes, q_bytes, mask_bytes, metadata_bytes, dma_bytes, dma_jobs, starts;
static int publication_failure = 0, p2_failure = 0;
static std::vector<int8_t> device_q8;
struct fake_device {
    std::map<uint32_t,uint32_t> r;
    int64_t clock = 0; int polls = 0, fault = 0, scale_override = -1;
    std::vector<uint32_t> tags, generations;
    void reset() { r.clear(); r[attention_abi::ABI] = attention_abi::SIGNATURE;
        r[attention_abi::CAPS] = 1; polls = fault = 0; scale_override = -1; }
} hw;
struct fpga_attention_transport {
    uint32_t read(uint32_t reg) {
        if (reg == attention_abi::STATUS && (hw.r[reg] & attention_abi::BUSY) && hw.fault != 4 && ++hw.polls == 3) {
            hw.r[reg] = hw.fault == 3 ? attention_abi::FAILED : attention_abi::DONE | attention_abi::RETAINED_VALID;
            if (hw.fault == 5) hw.r[reg] &= ~attention_abi::RETAINED_VALID;
            hw.r[attention_abi::ERROR] = hw.fault == 3;
            hw.r[attention_abi::DONE_TAG] = hw.r[attention_abi::TAG] + (hw.fault == 1);
            hw.r[attention_abi::DONE_GENERATION] = hw.r[attention_abi::GENERATION] + (hw.fault == 2);
        }
        return hw.r[reg];
    }
    void write(uint32_t reg, uint32_t value) {
        using namespace attention_abi;
        if (reg != CTRL) { hw.r[reg] = value; return; }
        assert(!(hw.r[STATUS] & BUSY));
        if (value & ACK) { hw.r[STATUS] &= ~(DONE | FAILED); hw.r[ERROR] = 0; }
        if (value & RELEASE) hw.r[STATUS] &= ~RETAINED_VALID;
        if (value & START) {
            assert(hw.r[STATUS] == 0);
            hw.r[STATUS] = BUSY; hw.polls = 0; ++starts;
            hw.tags.push_back(hw.r[TAG]); hw.generations.push_back(hw.r[GENERATION]);
            device_q8.resize(hw.r[D]*hw.r[HQ]);
            for (size_t i = 0; i < device_q8.size(); ++i) device_q8[i] = int8_t(int(i % 251) - 125);
            // Simulated PL writes only the readable metadata here. Q8/F32 CPU
            // mappings remain PROT_NONE throughout execution.
            for (size_t b = 0; b < device_q8.size()/32; ++b) {
                uint16_t scale = uint16_t(hw.scale_override < 0 ? 0x3000 + b : hw.scale_override);
                std::memcpy(g_ddr + SCALE_OFFSET + b*2, &scale, 2);
            }
        }
    }
    void fence() {}
    int64_t now_us() { return hw.clock++; }
    void yield() { ++hw.clock; }
};
static bool fpga_attention_register_admission(std::string & e) {
    fpga_attention_transport io; return attention_abi::admit(io, e);
}
static bool ddr_range_fits(uint32_t off, size_t n) { return off <= g_ddr_map_size && n <= g_ddr_map_size - off; }
static bool fpga_p2_ddr_sync(uint32_t off, size_t n, bool from_device, const char *) {
    assert(ddr_range_fits(off, n));
    if (from_device) {
        assert(off == attention_abi::SCALE_OFFSET && n <= 64); metadata_bytes += n;
    } else {
        assert(off + n <= attention_abi::F32_OFFSET || off >= attention_abi::KV_POOL_OFFSET);
        volatile uint8_t first = g_ddr[off], last = g_ddr[off+n-1]; (void)first; (void)last;
        if (off >= attention_abi::KV_POOL_OFFSET) kv_bytes += n;
        else if (off == attention_abi::Q_OFFSET) q_bytes += n;
        else if (off == attention_abi::MASK_OFFSET) mask_bytes += n;
        else assert(false);
        if (publication_failure && --publication_failure == 0) return false;
    }
    return true;
}
static bool fpga_attention_tensor_storage(const ggml_tensor * t, size_t & n, std::string &) {
    if (!t || !t->data) { return false; }
    n = t->storage; return true;
}
static bool ggml_is_contiguous(const ggml_tensor * t) {
    const size_t row = t->type == GGML_TYPE_Q8_0 ? size_t(t->ne[0]/32)*34 : size_t(t->ne[0])*t->nb[0];
    return t->nb[1] == row && t->nb[2] == row*size_t(t->ne[1]) && t->nb[3] == t->nb[2]*size_t(t->ne[2]);
}
static bool fpga_contract_validate_weight_scales(const ggml_tensor *, const void *, const char *, int) { return true; }
static bool fpga_enforce_p2_stream_mode(const char *) { return true; }
static int packed_q8_group_blocks_for_rows(int, int remaining) { return std::min(2, remaining); }
@INVALIDATE@
@RETAINED@
@VIEW@
@PUBLISH@
@POOL@
@UPLOAD@
@STORE@
static bool fpga_prepare_q8_tile_job(fpga_tile_job_t & j, const ggml_tensor *, const void *,
    const block_q8_0_t * scales, int64_t row0, int rows, int64_t block0, int blocks, int64_t,
    uint32_t, const void *, uint32_t, int, fpga_stage_totals_t *, void *, uint32_t, uint32_t, uint32_t source) {
    assert(fpga_attention_retained_dma_source(DDR_BASE_PHYS + source, size_t(blocks)*32));
    for (int b = 0; b < blocks; ++b) {
        assert(scales[b].d == uint16_t(0x3000 + block0 + b));
        for (int i = 0; i < 32; ++i) assert(scales[b].qs[i] == 0);
    }
    j.rows = rows; j.blocks = blocks; j.row0 = row0; j.block0 = block0; j.source = source;
    return true;
}
static bool fpga_submit_q8_tile_job(fpga_tile_job_t & j, fpga_stage_totals_t *, const char *, int,
    int64_t, int64_t, int64_t, int) {
    assert(fpga_attention_retained_dma_source(DDR_BASE_PHYS + j.source, size_t(j.blocks)*32));
    // A DMA-side bus master reads the retained bytes, not the protected PS map.
    const size_t offset = j.source - attention_abi::Q8_OFFSET;
    assert(offset + size_t(j.blocks)*32 <= device_q8.size());
    for (size_t i = offset; i < offset + size_t(j.blocks)*32; ++i)
        assert(device_q8[i] == int8_t(int(i % 251) - 125));
    dma_bytes += size_t(j.blocks)*32; ++dma_jobs;
    if (p2_failure && --p2_failure == 0) return false;
    return true;
}
static bool fpga_wait_and_drain_q8_tile_job(fpga_tile_job_t &, fpga_stage_totals_t *, const char *, int,
    int64_t, int64_t, int64_t, int) { return true; }
static bool fpga_accumulate_pl_scaled_q8_tile_job(fpga_tile_job_t & j, std::vector<float> & a, fpga_stage_totals_t *) {
    for (int row = 0; row < j.rows; ++row) a[size_t(row)] += float(j.blocks*(j.row0+row+1));
    return true;
}
@SCALE_VALID@
@PROJECT@
@COMPUTE@
static bool residency_mapping(bool g_attention_enabled, size_t g_ddr_map_size, size_t expected_map) {
    return @MAPPING@;
}
static ggml_tensor tensor(int type, void * data, int64_t d, int64_t n, int64_t h, size_t bytes) {
    ggml_tensor t = {}; t.type = type; t.data = data; t.ne[0]=d; t.ne[1]=n; t.ne[2]=h; t.ne[3]=1;
    t.nb[0] = type == GGML_TYPE_F32 ? 4 : type == GGML_TYPE_F16 ? 2 : 34;
    t.nb[1] = size_t(d)*t.nb[0]/(type == GGML_TYPE_Q8_0 ? 32 : 1);
    t.nb[2] = t.nb[1]*size_t(n); t.nb[3]=t.nb[2]*size_t(h); t.storage=bytes; return t;
}
int main() {
    using namespace attention_abi;
    g_ddr = static_cast<uint8_t *>(mmap(nullptr, g_ddr_map_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
    assert(g_ddr != MAP_FAILED);
    assert(mprotect(g_ddr + F32_OFFSET, 4096, PROT_NONE) == 0);
    assert(mprotect(g_ddr + Q8_OFFSET, 4096, PROT_NONE) == 0);
    hw.reset(); std::string error;
    std::vector<float> q(32*3*4, .25f), mask(5*3, 0), out(7*3, -99);
    mask[2] = -INFINITY; mask[5] = -INFINITY; mask[8] = -INFINITY;
    std::vector<uint16_t> k(32*5*2, 0x3800), v(k.size(), 0x3400);
    std::vector<block_q8_0_t> w(4*7); for (auto & b : w) b.d = 0x3800;
    ggml_tensor qt=tensor(GGML_TYPE_F32,q.data(),32,3,4,q.size()*4);
    ggml_tensor kt=tensor(GGML_TYPE_F16,k.data(),32,5,2,k.size()*2);
    ggml_tensor vt=tensor(GGML_TYPE_F16,v.data(),32,5,2,v.size()*2);
    ggml_tensor mt=tensor(GGML_TYPE_F32,mask.data(),5,3,1,mask.size()*4);
    ggml_tensor wt=tensor(GGML_TYPE_Q8_0,w.data(),128,7,1,w.size()*34);
    ggml_tensor dst=tensor(GGML_TYPE_F32,out.data(),7,3,1,out.size()*4);
    dst.src[0]=&qt; dst.src[1]=&kt; dst.src[2]=&vt; dst.src[3]=&mt; dst.src[4]=&wt;
    const uintptr_t layer0 = (2U<<16)|1, layer1 = (2U<<16)|2;
    assert(fpga_attention_compute(&dst,layer0,error));
    assert(fpga_attention_scale_valid(0) && fpga_attention_scale_valid(1));
    assert(fpga_attention_scale_valid(0x03ff) && fpga_attention_scale_valid(0x7bff));
    const uint16_t bad_scales[] = { 0x7c00, 0x7c01, 0x7fff, 0x8000, 0xbc00, 0xfc00, 0xffff };
    for (uint16_t bits : bad_scales) {
        assert(!fpga_attention_scale_valid(bits));
        const size_t before_dma = dma_jobs;
        hw.scale_override = bits;
        assert(!fpga_attention_compute(&dst,layer0,error));
        assert(error == "invalid hardware Q8 scale metadata" && dma_jobs == before_dma);
        fpga_attention_invalidate_host_state(); hw.reset();
    }
    const size_t after_bad_scales = starts;
    assert(fpga_attention_compute(&dst,layer0,error));
    assert(starts == after_bad_scales + 3);
    assert(dma_jobs==36 && dma_bytes==2*3*3*128 && metadata_bytes==(6+7)*8);
    for (size_t t=0;t<3;++t) for(size_t row=0;row<7;++row) assert(out[t*7+row]==4*(row+1));
    assert(!fpga_attention_retained_dma_source(DDR_BASE_PHYS+Q8_OFFSET,32));
    // Registering the other layer replans and invalidates, then both settle.
    assert(fpga_attention_compute(&dst,layer1,error));
    assert(fpga_attention_compute(&dst,layer0,error));
    const size_t before=kv_bytes;
    assert(fpga_attention_compute(&dst,layer1,error));
    assert(fpga_attention_compute(&dst,layer0,error));
    assert(kv_bytes==before); // graph reuse uploads no unchanged cache bytes
    k[91]=0x3c00; const size_t changed=kv_bytes;
    assert(fpga_attention_compute(&dst,layer0,error));
    assert(kv_bytes-changed==4); // 2-byte change, bounded aligned CPU publication fence
    const size_t layer_changed=kv_bytes;
    assert(fpga_attention_compute(&dst,layer1,error));
    assert(kv_bytes-layer_changed==4); // independent physical mirror
    for (size_t i=1;i<hw.tags.size();++i) {
        assert(hw.tags[i]>hw.tags[i-1] && hw.generations[i]>hw.generations[i-1]);
    }
    // Partial K/V publication must not commit any mirror snapshot.
    fpga_attention_invalidate_host_state(); publication_failure=2;
    assert(!fpga_attention_compute(&dst,layer0,error));
    fpga_attention_invalidate_host_state(); const size_t after_failure=kv_bytes;
    assert(fpga_attention_compute(&dst,layer0,error));
    assert(kv_bytes-after_failure==2*k.size()*2);
    // Completion/tag/generation/error/timeout/retained finality and failed Wo.
    for (int fault=1;fault<=5;++fault) {
        hw.fault=fault; assert(!fpga_attention_compute(&dst,layer0,error));
        fpga_attention_invalidate_host_state();
        assert(!fpga_attention_retained_dma_source(DDR_BASE_PHYS+Q8_OFFSET,32));
        hw.reset();
    }
    p2_failure=2; assert(!fpga_attention_compute(&dst,layer0,error));
    fpga_attention_invalidate_host_state(); hw.reset();
    assert(fpga_attention_compute(&dst,layer0,error));
    hw.r[ABI]=0; const size_t no_start=starts;
    assert(!fpga_attention_compute(&dst,layer0,error) && starts==no_start); hw.reset();
    std::fill(mask.begin(),mask.end(),-INFINITY);
    assert(!fpga_attention_compute(&dst,layer0,error) && starts==no_start);
    std::fill(mask.begin(),mask.end(),0);
    // Mapping bounds and distinct layers, including maximum legal N capacity.
    std::vector<layer_layout> old(2), next;
    assert(plan_pool(old,0,256,4,MAX_N,DDR_BYTES,next,error));
    assert(next[0].capacity==MAX_N);
    assert(!plan_pool(next,1,256,4,MAX_N,DDR_BYTES,old,error));
    assert(!plan_pool(next,0,256,4,MAX_N+1,DDR_BYTES,old,error));
    assert(!plan_pool(next,0,256,4,MAX_N,KV_POOL_OFFSET,old,error));
    assert(residency_mapping(true,DDR_BYTES,KV_POOL_OFFSET));
    assert(!residency_mapping(false,DDR_BYTES,KV_POOL_OFFSET));
    assert(residency_mapping(false,KV_POOL_OFFSET,KV_POOL_OFFSET));
    assert(!residency_mapping(true,DDR_BYTES,KV_POOL_OFFSET+1));
    assert(!residency_mapping(true,KV_POOL_OFFSET-1,KV_POOL_OFFSET));
    // Reset invalidates mirrors and handles; reused graph must republish.
    fpga_attention_invalidate_host_state(); const size_t reset_bytes=kv_bytes;
    assert(fpga_attention_compute(&dst,layer0,error));
    assert(kv_bytes-reset_bytes==2*k.size()*2);
    assert(munmap(g_ddr,g_ddr_map_size)==0);
    std::printf("PASS production attention integration: commands=%zu retained_DMA_jobs=%zu; protected F32/Q8 pages, graph reuse, layers, partial publication, finality, reset, combined residency\n",starts,dma_jobs);
}
'''
replacements = {
    "INVALIDATE": "static void fpga_attention_invalidate_host_state(",
    "RETAINED": "static bool fpga_attention_retained_dma_source(",
    "VIEW": "static bool fpga_attention_view(",
    "PUBLISH": "static bool fpga_attention_publish_bytes(",
    "POOL": "static bool fpga_attention_prepare_pool(",
    "UPLOAD": "static bool fpga_attention_upload_kv(",
    "STORE": "static void store_dst_value(",
    "PROJECT": "static bool fpga_attention_project_retained(",
    "SCALE_VALID": "static bool fpga_attention_scale_valid(",
    "COMPUTE": "static bool fpga_attention_compute(",
}
for key, signature in replacements.items():
    harness = harness.replace("@" + key + "@", definition(signature))
harness = harness.replace("@MAPPING@", mapping_expression)
with tempfile.TemporaryDirectory(prefix="fpga-attention-integration-") as temp:
    source = Path(temp) / "test.cpp"
    binary = Path(temp) / "test"
    source.write_text(harness)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++11", "-Wall", "-Wextra", "-Werror",
                    "-pedantic", "-O2", "-I", str(CPU), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
