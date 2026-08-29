// Compatibility include only.
//
// fpga_log.h pre-dates the host refactor and is the canonical home for all
// FPGA telemetry/logging implementation.  fpga_host.cpp still includes this
// path so the existing control flow and diagnostic call sites remain untouched.
#include "fpga_log.h"
