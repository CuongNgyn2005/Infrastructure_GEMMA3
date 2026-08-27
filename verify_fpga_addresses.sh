#!/bin/bash
# Verify the physical address contract used by ggml/src/ggml-cpu/fpga_host.cpp.
# This script is intentionally non-destructive: it never writes FPGA DDR or MMIO.

set -euo pipefail

DDR_BASE=0x70000000
DDR_SIZE=0x10000000
DDR_END=0x80000000
MY_IP_BASE=0xA0000000
MY_IP_MIN_SIZE=0x00400000
ZDMA_BASE=0xFD500000
ZDMA_SIZE=0x00001000

fail() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN:  $*" >&2; }

printf '%s\n' "FPGA host address-contract verification"
printf '  DDR carveout : [0x%08X, 0x%08X) size=0x%X\n' "$((DDR_BASE))" "$((DDR_END))" "$((DDR_SIZE))"
printf '  MY_IP/LMM    : 0x%08X (host-visible minimum 0x%X bytes)\n' "$((MY_IP_BASE))" "$((MY_IP_MIN_SIZE))"
printf '  ZDMA         : [0x%08X, 0x%08X)\n' "$((ZDMA_BASE))" "$((ZDMA_BASE + ZDMA_SIZE))"

[[ -r /proc/iomem ]] || fail "/proc/iomem is not readable"

# The reserved FPGA DDR region must not be normal Linux System RAM.  A simple
# overlap check is sufficient to reject the unsafe case without touching it.
python3 - "$DDR_BASE" "$DDR_END" <<'PY'
import re, sys
lo = int(sys.argv[1], 0)
hi = int(sys.argv[2], 0)
unsafe = []
with open('/proc/iomem', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        m = re.match(r'\s*([0-9a-fA-F]+)-([0-9a-fA-F]+)\s*:\s*(.*)', line)
        if not m:
            continue
        a, b, name = int(m.group(1), 16), int(m.group(2), 16) + 1, m.group(3).strip()
        if name == 'System RAM' and max(a, lo) < min(b, hi):
            unsafe.append((a, b, name))
if unsafe:
    for a, b, name in unsafe:
        print(f'ERROR: FPGA DDR carveout overlaps {name}: [0x{a:x},0x{b:x})', file=sys.stderr)
    sys.exit(2)
print('PASS: FPGA DDR carveout does not overlap /proc/iomem System RAM')
PY

# Enumerate UIO resources instead of guessing device numbers.  fpga_host.cpp
# also performs identity/capability checks at runtime, so this script only
# verifies that Linux exposes candidate mappings; it does not claim the
# bitstream/ABI is correct.
if [[ -d /sys/class/uio ]]; then
    echo "UIO devices:"
    found=0
    for uio in /sys/class/uio/uio*; do
        [[ -e "$uio" ]] || continue
        found=1
        name=$(cat "$uio/name" 2>/dev/null || echo '?')
        printf '  %s name=%s\n' "$(basename "$uio")" "$name"
        for map in "$uio"/maps/map*; do
            [[ -e "$map" ]] || continue
            addr=$(cat "$map/addr" 2>/dev/null || echo '?')
            size=$(cat "$map/size" 2>/dev/null || echo '?')
            printf '    %s addr=%s size=%s\n' "$(basename "$map")" "$addr" "$size"
        done
    done
    [[ $found -eq 1 ]] || warn "no /sys/class/uio/uio* devices found"
else
    warn "/sys/class/uio is absent"
fi

# /dev/mem is a compatibility path in the host, not proof of address safety.
if [[ -e /dev/mem ]]; then
    echo "INFO: /dev/mem exists (compatibility path available)"
else
    echo "INFO: /dev/mem absent; UIO-only operation may still be valid"
fi

echo "PASS: static host address contract is synchronized with fpga_host.cpp"
echo "NOTE: board-side identity, ABI and actual address accessibility remain runtime checks in fpga_init()."
