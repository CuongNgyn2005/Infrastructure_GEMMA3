#!/usr/bin/env bash

# Apply the temporary Gemma3 FPGA UIO overlay used by fpga_host.cpp.
#
# This overlay is intentionally temporary: configfs removes it at reboot.  The
# script is idempotent, validates the live address map before applying it, and
# can launch a command with the exact UIO devices pinned through environment
# variables so the legacy high-DDR UIO node cannot be selected accidentally.

set -euo pipefail

readonly OVERLAY_NAME="gemma3_fpga_uio"
readonly CONFIGFS_OVERLAYS="/sys/kernel/config/device-tree/overlays"
readonly OVERLAY_DIR="${CONFIGFS_OVERLAYS}/${OVERLAY_NAME}"
readonly MY_IP_BASE=$((0xA0000000))
readonly MY_IP_SIZE=$((0x10000000))
readonly DDR_BASE=$((0x70000000))
readonly DDR_SIZE=$((0x10000000))
readonly DDR_END=$((DDR_BASE + DDR_SIZE - 1))
readonly ENV_FILE="/run/gemma3-fpga-uio.env"

die() {
    printf '[OVERLAY][ERROR] %s\n' "$*" >&2
    exit 1
}

info() {
    printf '[OVERLAY][INFO] %s\n' "$*"
}

require_root() {
    [ "$(id -u)" -eq 0 ] || die "run this script with sudo"
}

require_idle() {
    if pgrep -x llama-cli >/dev/null 2>&1; then
        pgrep -a -x llama-cli >&2 || true
        die "llama-cli is running; stop it before changing device-tree overlays"
    fi
}

hex_value() {
    local text=${1#0x}
    printf '%u' "$((16#${text}))"
}

find_unique_uio_resource() {
    local wanted_name=$1
    local wanted_addr=$2
    local u addr name match="" matches=0

    for u in /sys/class/uio/uio*; do
        [ -d "$u" ] || continue
        [ -r "$u/maps/map0/addr" ] || continue
        addr=$(cat "$u/maps/map0/addr")
        name=$(cat "$u/name" 2>/dev/null || true)
        if [ "$name" = "$wanted_name" ] && [ "$(hex_value "$addr")" -eq "$wanted_addr" ]; then
            match=$u
            matches=$((matches + 1))
        fi
    done

    if [ "$matches" -eq 1 ]; then
        printf '%s' "$match"
        return 0
    fi
    if [ "$matches" -gt 1 ]; then
        printf '[OVERLAY][ERROR] resource name=%s addr=0x%x is ambiguous (%d UIO nodes)\n' \
            "$wanted_name" "$wanted_addr" "$matches" >&2
        return 2
    fi
    return 1
}

validate_uio() {
    local sysfs=$1
    local expected_name=$2
    local expected_addr=$3
    local minimum_size=$4
    local label=$5
    local name addr offset size

    minimum_size=$((minimum_size))

    name=$(cat "$sysfs/name")
    addr=$(cat "$sysfs/maps/map0/addr")
    offset=$(cat "$sysfs/maps/map0/offset")
    size=$(cat "$sysfs/maps/map0/size")

    [ "$name" = "$expected_name" ] ||
        die "$label name mismatch: got=$name expected=$expected_name"
    [ "$(hex_value "$addr")" -eq "$expected_addr" ] ||
        die "$label address mismatch: got=$addr expected=$(printf '0x%x' "$expected_addr")"
    [ "$(hex_value "$offset")" -eq 0 ] ||
        die "$label map offset must be zero: got=$offset"
    [ "$(hex_value "$size")" -ge "$minimum_size" ] ||
        die "$label map is too small: got=$size required=$(printf '0x%x' "$minimum_size")"

    info "$label verified device=/dev/$(basename "$sysfs") name=$name addr=$addr offset=$offset size=$size"
}

validate_low_ddr_not_system_ram() {
    local range start_hex end_hex start end

    while read -r range; do
        [ -n "$range" ] || continue
        start_hex=${range%-*}
        end_hex=${range#*-}
        start=$(hex_value "$start_hex")
        end=$(hex_value "$end_hex")
        if [ "$start" -le "$DDR_END" ] && [ "$end" -ge "$DDR_BASE" ]; then
            die "low DDR staging range [0x70000000,0x80000000) overlaps Linux System RAM $range"
        fi
    done < <(sed -n 's/^\([0-9a-fA-F][0-9a-fA-F]*-[0-9a-fA-F][0-9a-fA-F]*\) : System RAM$/\1/p' /proc/iomem)

    info "low DDR staging range [0x70000000,0x80000000) has no System RAM overlap"
}

write_overlay_source() {
    local dts=$1
    local need_my_ip=$2
    local need_ddr=$3
    local fragment=0

    cat >"$dts" <<'EOF'
/dts-v1/;
/plugin/;

/ {
EOF

    if [ "$need_my_ip" -eq 1 ]; then
        cat >>"$dts" <<EOF
    fragment@${fragment} {
        target = <&amba_pl>;

        __overlay__ {
            #address-cells = <2>;
            #size-cells = <2>;

            MY_IP@a0000000 {
                compatible = "generic-uio";
                linux,uio-name = "MY_IP";
                reg = <0x0 0xa0000000 0x0 0x10000000>;
                status = "okay";
            };
        };
    };
EOF
        fragment=$((fragment + 1))
    fi

    if [ "$need_ddr" -eq 1 ]; then
        cat >>"$dts" <<EOF
    fragment@${fragment} {
        target-path = "/";

        __overlay__ {
            #address-cells = <2>;
            #size-cells = <2>;

            fpga_ddr_low@70000000 {
                compatible = "generic-uio";
                linux,uio-name = "fpga_ddr_low";
                reg = <0x0 0x70000000 0x0 0x10000000>;
                status = "okay";
            };
        };
    };
EOF
    fi

    cat >>"$dts" <<'EOF'
};
EOF
}

write_env_file() {
    local my_ip_sysfs=$1
    local ddr_sysfs=$2
    local my_ip_dev="/dev/$(basename "$my_ip_sysfs")"
    local ddr_dev="/dev/$(basename "$ddr_sysfs")"

    umask 022
    cat >"$ENV_FILE" <<EOF
export FPGA_VPU_UIO=${my_ip_dev}
export FPGA_DDR_UIO=${ddr_dev}
EOF
    info "runtime UIO environment written to $ENV_FILE"
}

apply_overlay() {
    local state firmware my_ip_sysfs ddr_sysfs need_my_ip=1 need_ddr=1
    local workdir dts dtbo

    require_root
    require_idle
    command -v dtc >/dev/null 2>&1 || die "dtc is required"
    [ -d "$CONFIGFS_OVERLAYS" ] || die "kernel device-tree overlay support is unavailable"
    [ -r /sys/class/fpga_manager/fpga0/state ] || die "FPGA manager state is unavailable"

    state=$(cat /sys/class/fpga_manager/fpga0/state)
    [ "$state" = "operating" ] || die "FPGA manager is not operating: state=$state"
    firmware=$(cat /sys/class/fpga_manager/fpga0/firmware 2>/dev/null || true)
    info "FPGA manager state=$state firmware=${firmware:-unknown}"

    validate_low_ddr_not_system_ram

    if my_ip_sysfs=$(find_unique_uio_resource "MY_IP" "$MY_IP_BASE"); then
        need_my_ip=0
        validate_uio "$my_ip_sysfs" "MY_IP" "$MY_IP_BASE" 0x400000 "MY_IP"
    else
        rc=$?
        [ "$rc" -eq 1 ] || die "MY_IP UIO resource discovery failed"
    fi
    if ddr_sysfs=$(find_unique_uio_resource "fpga_ddr_low" "$DDR_BASE"); then
        need_ddr=0
        validate_uio "$ddr_sysfs" "fpga_ddr_low" "$DDR_BASE" "$DDR_SIZE" "FPGA low DDR"
    else
        rc=$?
        [ "$rc" -eq 1 ] || die "fpga_ddr_low UIO resource discovery failed"
    fi

    if [ "$need_my_ip" -eq 1 ] || [ "$need_ddr" -eq 1 ]; then
        if [ -e "$OVERLAY_DIR" ]; then
            # Only this fixed, named configfs overlay may be removed.  The
            # idle gate above prevents its UIO mappings changing under llama.
            rmdir -- "$OVERLAY_DIR" || die "cannot remove stale overlay $OVERLAY_NAME"
            [ ! -e "$OVERLAY_DIR" ] || die "stale overlay $OVERLAY_NAME remained after removal"
            info "stale overlay removed and verified name=$OVERLAY_NAME"

            # Re-discover after configfs removal: a resource observed above
            # may have belonged to the stale overlay and no longer exist.
            need_my_ip=1
            need_ddr=1
            if my_ip_sysfs=$(find_unique_uio_resource "MY_IP" "$MY_IP_BASE"); then
                need_my_ip=0
                validate_uio "$my_ip_sysfs" "MY_IP" "$MY_IP_BASE" 0x400000 "MY_IP"
            else
                rc=$?
                [ "$rc" -eq 1 ] || die "MY_IP UIO resource rediscovery failed"
            fi
            if ddr_sysfs=$(find_unique_uio_resource "fpga_ddr_low" "$DDR_BASE"); then
                need_ddr=0
                validate_uio "$ddr_sysfs" "fpga_ddr_low" "$DDR_BASE" "$DDR_SIZE" "FPGA low DDR"
            else
                rc=$?
                [ "$rc" -eq 1 ] || die "fpga_ddr_low UIO resource rediscovery failed"
            fi
        fi

        if [ "$need_my_ip" -eq 1 ] || [ "$need_ddr" -eq 1 ]; then

            workdir=$(mktemp -d /tmp/gemma3-overlay.XXXXXX)
            dts="$workdir/${OVERLAY_NAME}.dts"
            dtbo="$workdir/${OVERLAY_NAME}.dtbo"

            write_overlay_source "$dts" "$need_my_ip" "$need_ddr"
            dtc -@ -I dts -O dtb -o "$dtbo" "$dts"
            mkdir "$OVERLAY_DIR"
            cat "$dtbo" >"$OVERLAY_DIR/dtbo"
            [ "$(cat "$OVERLAY_DIR/status")" = "applied" ] || die "overlay did not reach applied status"
            info "overlay applied name=$OVERLAY_NAME"
            rm -f -- "$dts" "$dtbo"
            rmdir -- "$workdir"
        fi
    else
        info "required UIO resources already exist; no overlay change needed"
    fi

    my_ip_sysfs=$(find_unique_uio_resource "MY_IP" "$MY_IP_BASE") || die "exact MY_IP UIO was not created"
    ddr_sysfs=$(find_unique_uio_resource "fpga_ddr_low" "$DDR_BASE") || die "exact fpga_ddr_low UIO was not created"
    validate_uio "$my_ip_sysfs" "MY_IP" "$MY_IP_BASE" 0x400000 "MY_IP"
    validate_uio "$ddr_sysfs" "fpga_ddr_low" "$DDR_BASE" "$DDR_SIZE" "FPGA low DDR"
    write_env_file "$my_ip_sysfs" "$ddr_sysfs"

    printf '[OVERLAY][READY] FPGA_VPU_UIO=/dev/%s FPGA_DDR_UIO=/dev/%s\n' \
        "$(basename "$my_ip_sysfs")" "$(basename "$ddr_sysfs")"
}

show_status() {
    local my_ip_sysfs ddr_sysfs

    require_root
    validate_low_ddr_not_system_ram
    my_ip_sysfs=$(find_unique_uio_resource "MY_IP" "$MY_IP_BASE") || die "exact MY_IP UIO at 0xa0000000 is absent"
    ddr_sysfs=$(find_unique_uio_resource "fpga_ddr_low" "$DDR_BASE") || die "exact fpga_ddr_low UIO at 0x70000000 is absent"
    validate_uio "$my_ip_sysfs" "MY_IP" "$MY_IP_BASE" 0x400000 "MY_IP"
    validate_uio "$ddr_sysfs" "fpga_ddr_low" "$DDR_BASE" "$DDR_SIZE" "FPGA low DDR"
    write_env_file "$my_ip_sysfs" "$ddr_sysfs"
}

remove_overlay() {
    require_root
    require_idle
    if [ -d "$OVERLAY_DIR" ]; then
        rmdir "$OVERLAY_DIR"
        info "overlay removed name=$OVERLAY_NAME"
    else
        info "overlay $OVERLAY_NAME is not present"
    fi
    rm -f -- "$ENV_FILE"
}

run_command() {
    require_root
    apply_overlay
    shift
    if [ "${1:-}" = "--" ]; then
        shift
    fi
    [ "$#" -gt 0 ] || die "run requires a command after '--'"
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    exec env FPGA_VPU_UIO="$FPGA_VPU_UIO" FPGA_DDR_UIO="$FPGA_DDR_UIO" "$@"
}

usage() {
    cat <<'EOF'
Usage:
  sudo ./overlay_procedure.sh apply
  sudo ./overlay_procedure.sh status
  sudo ./overlay_procedure.sh run -- COMMAND [ARG ...]
  sudo ./overlay_procedure.sh remove

The default action is apply.  The run action applies/verifies the overlay and
executes COMMAND with FPGA_VPU_UIO and FPGA_DDR_UIO pinned to the verified UIO
devices.  This avoids ambiguity with the legacy ddr_high UIO at 0x800000000.
EOF
}

case "${1:-apply}" in
    apply)
        apply_overlay
        ;;
    status)
        show_status
        ;;
    run)
        run_command "$@"
        ;;
    remove)
        remove_overlay
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
