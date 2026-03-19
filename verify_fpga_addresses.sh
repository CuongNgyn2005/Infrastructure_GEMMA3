#!/bin/bash
# Script to verify FPGA buffer addresses are accessible
# Simplified version without needing gcc

set -e

echo "════════════════════════════════════════════════════════════"
echo "FPGA Buffer Address Verification (Simple Version)"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if running as root (needed for /dev/mem access)
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (sudo)"
   exit 1
fi

echo "✓ Running as root"
echo ""

# Define buffer addresses (must match fpga_host.cpp)
echo "Testing physical addresses:"
echo "  BUF_A    @ 0x77C00000"
echo "  BUF_BD   @ 0x78C00000"
echo "  BUF_BQS  @ 0x79C00000"
echo "  BUF_C    @ 0x7AC00000"
echo "  CTRL     @ 0x400000000"
echo ""

echo "════════════════════════════════════════════════════════════"

# Check if /dev/mem exists
if [ ! -e /dev/mem ]; then
    echo "❌ /dev/mem not found - cannot access physical memory"
    echo "   Try: sudo modprobe -a uio mem"
    exit 1
fi

echo "✓ /dev/mem exists"
echo ""

# Test using dd command instead of compiled binary
test_address_simple() {
    local addr=$1
    local name=$2

    echo -n "Testing $name @ 0x$addr ... "

    # Try to read 1 byte from address using dd
    # This is faster than compiling C code
    if timeout 2 dd if=/dev/mem bs=1 count=1 skip=$(printf '%d' 0x$addr) 2>/dev/null | head -c 1 > /tmp/test_byte 2>&1; then
        echo "✓ OK"
        return 0
    else
        # Alternative: try using devmem if available
        if command -v devmem &> /dev/null; then
            if devmem 0x$addr 32 2>/dev/null > /dev/null; then
                echo "✓ OK (via devmem)"
                return 0
            fi
        fi
        echo "❌ FAILED"
        return 1
    fi
}

# Test each address (convert hex to decimal for dd)
echo "════════════════════════════════════════════════════════════"
ALL_OK=1

# Test BUF_A
if test_address_simple "77C00000" "BUF_A"; then
    :
else
    ALL_OK=0
fi

# Test BUF_BD
if test_address_simple "78C00000" "BUF_BD"; then
    :
else
    ALL_OK=0
fi

# Test BUF_BQS
if test_address_simple "79C00000" "BUF_BQS"; then
    :
else
    ALL_OK=0
fi

# Test BUF_C
if test_address_simple "7AC00000" "BUF_C"; then
    :
else
    ALL_OK=0
fi

# Test CTRL
if test_address_simple "400000000" "CTRL"; then
    :
else
    ALL_OK=0
fi

echo "════════════════════════════════════════════════════════════"
echo ""

if [ $ALL_OK -eq 1 ]; then
    echo "✅ ADDRESSES APPEAR ACCESSIBLE"
    echo ""
    echo "Status: Ready for FPGA use"
    echo "Next: Uncomment CMakeLists and build"
    exit 0
else
    echo "⚠️  SOME ADDRESSES NOT ACCESSIBLE"
    echo ""
    echo "Debug info:"
    echo "  Check current memory layout:"
    cat /proc/iomem | grep "System RAM"
    echo ""
    echo "  Check free memory:"
    free -h
    echo ""
    echo "  Run: cat /proc/iomem | grep -A5 'System RAM'"
    echo "  to see full DDR mapping"
    exit 1
fi
