#pragma once

#include <cstdarg>
#include <cstdio>

// Shared logging sink used by the FPGA host. Driver-specific diagnostics
// remain in fpga_host.cpp and pass only formatted records through this API.
FILE * fpga_log_fp();
void   fpga_log_set_flush_every(int flush_every);
void   fpga_log_finish_line(FILE * fp, bool force_flush);
void   fpga_log_vline(const char * tag, bool force_flush, const char * fmt, va_list ap);
