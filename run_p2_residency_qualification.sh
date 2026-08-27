#!/bin/bash
set -euo pipefail

# Bounded qualification launcher for the existing P2 sealed-tile residency path.
# This does not enable residency by default in production. It exists to make the
# required two-flag diagnostic admission explicit and reproducible.
#
# Usage:
#   sudo ./run_p2_residency_qualification.sh -- ./build_mem/bin/llama-cli ...
#
# Optional:
#   FPGA_P2_WEIGHT_RESIDENCY_MB=8|16|32
#
# Start small. Increase only after the board preflight and a shorter decode run
# complete without address, metadata, ZDMA, stream, or numerical errors.

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "ERROR: run with sudo/root so the same process owns the UIO mappings." >&2
    exit 1
fi

if [[ $# -lt 2 || "$1" != "--" ]]; then
    echo "Usage: sudo $0 -- <llama-cli command and arguments>" >&2
    exit 2
fi
shift

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Keep incompatible experiments out of this qualification run.
for name in FPGA_P3_SPLIT_SCALE FPGA_WEIGHT_CACHE FPGA_WEIGHT_PATH_BENCH; do
    if [[ -n ${!name:-} && ${!name} != "0" ]]; then
        echo "ERROR: $name=${!name} is incompatible with bounded P2 residency qualification." >&2
        exit 3
    fi
done

RESIDENCY_MB=${FPGA_P2_WEIGHT_RESIDENCY_MB:-8}
case "$RESIDENCY_MB" in
    8|16|32) ;;
    *)
        echo "ERROR: FPGA_P2_WEIGHT_RESIDENCY_MB must be 8, 16, or 32 for this qualification launcher." >&2
        exit 4
        ;;
esac

# Non-destructive board/address contract check. This must pass before the host
# is allowed to enlarge the mapped DDR aperture beyond the normal 4 MiB scratch
# window and populate residency payloads.
"$SCRIPT_DIR/verify_fpga_addresses.sh"

export FPGA_P2_WEIGHT_RESIDENCY=1
export FPGA_P2_WEIGHT_RESIDENCY_DIAGNOSTIC=1
export FPGA_P2_WEIGHT_RESIDENCY_MB="$RESIDENCY_MB"

# Keep routine production telemetry, but make residency admission/result easy
# to identify in /tmp/fpga_debug.log without enabling per-hit trace spam.
export FPGA_INIT_VERBOSE=${FPGA_INIT_VERBOSE:-1}
export FPGA_P2_RESIDENCY_TRACE=${FPGA_P2_RESIDENCY_TRACE:-0}
export FPGA_P2_RESIDENCY_VERIFY_METADATA=${FPGA_P2_RESIDENCY_VERIFY_METADATA:-0}

CACHE_BASE_OFF=$((0x01000000))
DDR_BASE=$((0x70000000))
CACHE_START=$((DDR_BASE + CACHE_BASE_OFF))
CACHE_END=$((CACHE_START + RESIDENCY_MB * 1024 * 1024))
MAP_END=$CACHE_END

printf 'P2 residency qualification:\n'
printf '  budget           : %s MiB\n' "$RESIDENCY_MB"
printf '  residency phys   : [0x%08x, 0x%08x)\n' "$CACHE_START" "$CACHE_END"
printf '  host DDR mapping : [0x%08x, 0x%08x)\n' "$DDR_BASE" "$MAP_END"
printf '  mode             : diagnostic opt-in; non-evicting sealed-tile directory\n'
printf '  command          :'
printf ' %q' "$@"
printf '\n'

exec "$@"
