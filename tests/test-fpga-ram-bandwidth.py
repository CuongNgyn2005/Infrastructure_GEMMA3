#!/usr/bin/env python3
"""Run the RAM logger without hardware; capture its log in an anonymous file."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
harness = r'''
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
static FILE * sink;
static int test_open(const char *, int, ...) { return dup(fileno(sink)); }
#define open test_open
#include "fpga_log.cpp"
#undef open
int main() {
    sink = tmpfile();
    assert(sink);
    fpga_log_ram_bandwidth();
    rewind(sink);
    char line[2048];
    unsigned records = 0;
    while (fgets(line, sizeof(line), sink)) {
        if (!strstr(line, "[RAM_BANDWIDTH]")) continue;
        ++records;
        size_t buffer, copied;
        unsigned passes;
        double ms, copy, read_write;
        assert(strstr(line, "status=ok method=libc_memcpy threads=1"));
        assert(sscanf(strstr(line, "buffer_bytes="),
            "buffer_bytes=%zu passes=%u copied_bytes=%zu elapsed_ms=%lf copy_GB_s=%lf logical_read_write_GB_s=%lf",
            &buffer, &passes, &copied, &ms, &copy, &read_write) == 6);
        assert(buffer == 32u*1024u*1024u && passes == 8 && copied == buffer*passes);
        assert(ms > 0 && copy > 0);
        assert(std::abs(copy - copied/ms/1e6) < 1e-4);
        assert(std::abs(read_write - 2*copy) < 2e-6);
        assert(strstr(line, "verified=1"));
        assert(strstr(line, "physical_ddr_traffic=not_measured"));
    }
    assert(records == 1);
}
'''
with tempfile.TemporaryDirectory(prefix="fpga-ram-log-") as tmp:
    source = Path(tmp) / "test.cpp"
    binary = Path(tmp) / "test"
    source.write_text(harness)
    subprocess.run([
        "g++", "-std=c++17", "-O3", "-flto", "-Wall", "-Wextra", "-Werror",
        "-DUSE_FPGA", "-I" + str(root / "ggml/src/ggml-cpu"),
        str(source), "-o", str(binary),
    ], check=True)
    result = subprocess.run([str(binary)], capture_output=True, check=True)
    assert not result.stdout and not result.stderr, "Benchmark must log to file only"
print("PASS: RAM copy verified; byte/time formulas correct; file-only output; no hardware access")
