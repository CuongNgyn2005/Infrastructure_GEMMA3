#!/usr/bin/env python3
"""Exercise production benchmark functions against RAM, without driver init."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
cpu = root / "ggml/src/ggml-cpu"
harness = r'''
#include "fpga_host.cpp"
#include <cassert>
FILE * fpga_log_fp() { static FILE * f = tmpfile(); return f; }
void fpga_log_set_flush_every(int) {}
void fpga_log_finish_line(FILE *, bool) {}
void fpga_log_vline(const char *, bool, const char *, va_list) {}
void fpga_log_latency(const char *, ...) {}
void fpga_log_pack_breakdown(const char *, int, long long, const fpga_pack_breakdown_log_t &) {}
int main() {
    std::vector<uint64_t> ram(48*1024*1024/8, UINT64_C(0xabababababababab));
    g_ddr = (uint8_t*)ram.data(); g_ddr_map_base=ram.data(); g_ddr_map_size=ram.size()*8;
    assert(!g_vpu && !g_dma);
    assert(fpga_p2_pack_worker_start());
    int cases=0;
    for (int rows : {1,3,255,256}) for (int groups : {1,36,64}) {
        int stride=groups+2;
        std::vector<block_q8_0_t> w((rows+2)*stride);
        for(size_t i=0;i<w.size();++i) for(int q=0;q<32;++q) w[i].qs[q]=(int8_t)(i*31+q*17);
        ggml_tensor t={}; t.type=GGML_TYPE_Q8_0; t.data=w.data();
        t.ne[0]=stride*32;t.ne[1]=rows+2;t.ne[2]=t.ne[3]=1;
        t.nb[0]=sizeof(block_q8_0_t);t.nb[1]=stride*sizeof(block_q8_0_t);
        t.nb[2]=t.nb[3]=t.nb[1]*t.ne[1];
        fpga_weight_path_bench_job_t j={};
        j.src0=&t;j.weight_data_base=t.data;j.source_k=t.ne[0];j.source_n=t.ne[1];
        j.source_nb0=t.nb[0];j.source_nb1=t.nb[1];j.source_span_bytes=ggml_nbytes(&t);
        j.row0=1;j.k_block0=1;j.rows=rows;j.group_blocks=groups;j.group_beats=groups*2;
        assert(fpga_weight_layout_payload_bytes(rows,groups*2,&j.payload_bytes));
        std::vector<uint32_t> cached(j.payload_bytes/4);
        assert(fpga_weight_path_bench_pack_cached(j,cached));
        for(bool copy : {false,true}) {
            memset(g_ddr+WEIGHT_BASE-8,0xab,j.payload_bytes+16);
            assert(fpga_weight_path_bench_write(j,copy?cached.data():nullptr));
            // Independent layout oracle, including padding and offset source.
            for(size_t p=0;p<((size_t)rows+1)/2;++p) for(int b=0;b<groups;++b)
                for(int beat=0;beat<2;++beat) for(int r=0;r<2;++r) for(int q=0;q<16;++q) {
                    size_t row=p*2+r, out=(((p*groups+b)*2+beat)*2+r)*16+q;
                    uint8_t expected=row<(size_t)rows?(uint8_t)w[(row+1)*stride+b+1].qs[beat*16+q]:0;
                    assert(g_ddr[WEIGHT_BASE+out]==expected);
                }
            for(int i=1;i<=8;++i) assert(g_ddr[WEIGHT_BASE-i]==0xab);
            for(int i=0;i<8;++i) assert(g_ddr[WEIGHT_BASE+j.payload_bytes+i]==0xab);
            ++cases;
        }
        auto bad=j;bad.source_nb1++;
        assert(!fpga_weight_path_bench_write(bad,nullptr));
        assert(!fpga_weight_path_bench_write(bad,cached.data()));
        if (rows == 256 && groups == 36) {
            // Exercise the complete verification/timing boundary, including
            // owned worker creation and cleanup in passive diagnostic mode.
            assert(fpga_p2_pack_worker_stop());
            g_weight_path_bench.enabled=true;
            g_weight_path_bench.graph_seq=9;
            // Exceed the 32 MiB trial budget so both cached and uncached
            // paths execute. Repeated geometry keeps this test RAM-bounded.
            g_weight_path_bench.jobs.assign(120,j);
            g_weight_path_bench.payload_bytes=120*j.payload_bytes;
            assert(fpga_weight_path_bench_replay_at_boundary(9,1));
            assert(g_weight_path_bench.replayed && g_weight_path_bench.jobs.empty());
            assert(g_weight_path_bench.cached_payload.empty() && !g_p2_pack_worker_created);
            assert(fpga_p2_pack_worker_start());
        }
    }
    assert(fpga_p2_pack_worker_stop());
    assert(!g_vpu && !g_dma);
    printf("PASS: %d mapped-RAM benchmark cases, serial/parallel, byte oracle, guards, stale metadata; no hardware\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-benchmark-") as directory:
    cpp = Path(directory) / "test.cpp"
    exe = Path(directory) / "test"
    cpp.write_text(harness)
    lib = root / "build_mem_linux/bin"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-pthread",
                    "-ffunction-sections", "-fdata-sections", "-fsanitize=undefined",
                    "-fno-sanitize-recover=all", "-I"+str(cpu), "-I"+str(root/"ggml/include"),
                    "-I"+str(root/"ggml/src"), str(cpp), str(cpu/"fpga_q8_pack.cpp"),
                    str(cpu/"fpga_q8_layout.cpp"), "-L"+str(lib), "-lggml-base",
                    "-Wl,--gc-sections", "-Wl,-rpath,"+str(lib), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
