#!/bin/bash
# ZCU104 host-side preflight for the current Infrastructure_GEMMA3 address map.
# This checklist is observational only; it performs no MMIO or DDR writes.

set -u

DDR_BASE=0x70000000
DDR_END=0x80000000
MY_IP_BASE=0xA0000000
ZDMA_BASE=0xFD500000

section() {
    echo
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

section "1. Linux / board"
uname -a

section "2. Installed RAM"
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
cat /proc/iomem | grep -E 'System RAM|reserved' || true

echo
printf 'Expected FPGA DDR carveout used by host: [0x%08X, 0x%08X)\n' "$((DDR_BASE))" "$((DDR_END))"
printf 'Expected MY_IP/LMM base:             0x%08X\n' "$((MY_IP_BASE))"
printf 'Expected ZDMA page base:             0x%08X\n' "$((ZDMA_BASE))"

section "3. Current address-contract checker"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [[ -x "$SCRIPT_DIR/verify_fpga_addresses.sh" ]]; then
    "$SCRIPT_DIR/verify_fpga_addresses.sh" || {
        echo "Address-contract verification failed. Do not start FPGA inference." >&2
        exit 1
    }
else
    echo "verify_fpga_addresses.sh is missing or not executable" >&2
    exit 1
fi

section "4. UIO inventory"
if [[ -d /sys/class/uio ]]; then
    for uio in /sys/class/uio/uio*; do
        [[ -e "$uio" ]] || continue
        echo "$(basename "$uio"): $(cat "$uio/name" 2>/dev/null || echo '?')"
        for map in "$uio"/maps/map*; do
            [[ -e "$map" ]] || continue
            echo "  $(basename "$map") addr=$(cat "$map/addr" 2>/dev/null || echo '?') size=$(cat "$map/size" 2>/dev/null || echo '?')"
        done
    done
else
    echo "WARN: /sys/class/uio not present"
fi

section "5. FPGA programming utilities"
if command -v fpgautil >/dev/null 2>&1; then
    echo "fpgautil: $(command -v fpgautil)"
    fpgautil -d 2>/dev/null || echo "INFO: fpgautil did not report a device state"
else
    echo "INFO: fpgautil not installed; verify the programmed bitstream using the board's normal flow"
fi

section "6. Memory pressure"
free -h

section "7. Result"
echo "Host-side static preflight passed."
echo "This does NOT prove that the physical ranges are electrically accessible or that the loaded bitstream implements the expected VPU2/P2/P3 ABI."
echo "Those properties must be proven by fpga_init() capability/identity checks and, where required, an on-board run."
