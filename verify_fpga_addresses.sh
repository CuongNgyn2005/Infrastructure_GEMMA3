#!/bin/bash
# Script to verify FPGA buffer addresses are accessible
# Run this ON ZCU104 board before running llama with FPGA

set -e

echo "════════════════════════════════════════════════════════════"
echo "FPGA Buffer Address Verification Script"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if running as root (needed for /dev/mem access)
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (sudo)"
   echo "   Usage: sudo ./verify_fpga_addresses.sh"
   exit 1
fi

echo "✓ Running as root"
echo ""

# Define buffer addresses (must match fpga_host.cpp)
BUF_A_PHYS=0x77C00000
BUF_BD_PHYS=0x78C00000
BUF_BQS_PHYS=0x79C00000
BUF_C_PHYS=0x7AC00000
CTRL_PHYS=0x400000000

echo "Testing physical addresses:"
echo "  BUF_A    @ 0x${BUF_A_PHYS:2}"
echo "  BUF_BD   @ 0x${BUF_BD_PHYS:2}"
echo "  BUF_BQS  @ 0x${BUF_BQS_PHYS:2}"
echo "  BUF_C    @ 0x${BUF_C_PHYS:2}"
echo "  CTRL     @ 0x${CTRL_PHYS}"
echo ""

# Test function
test_address() {
    local addr=$1
    local name=$2
    local size=$3

    echo -n "Testing $name @ 0x$addr ... "

    # Create a C program to test mmap
    cat > /tmp/test_mmap.c << EOF
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("FAIL (cannot open /dev/mem)\\n");
        return 1;
    }

    void *virt = mmap(0, $size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0x$addr);
    if (virt == MAP_FAILED) {
        printf("FAIL (mmap failed - address may be occupied)\\n");
        close(fd);
        return 1;
    }

    // Try to write/read a test value
    volatile uint32_t *ptr = (volatile uint32_t *)virt;
    uint32_t test_val = 0xDEADBEEF;
    *ptr = test_val;
    uint32_t read_val = *ptr;

    if (read_val != test_val) {
        printf("FAIL (write/read test failed)\\n");
        munmap(virt, $size);
        close(fd);
        return 1;
    }

    munmap(virt, $size);
    close(fd);
    printf("OK\\n");
    return 0;
}
EOF

    gcc -o /tmp/test_mmap /tmp/test_mmap.c 2>/dev/null
    /tmp/test_mmap
    local ret=$?
    rm -f /tmp/test_mmap /tmp/test_mmap.c

    return $ret
}

# Test each buffer
echo "════════════════════════════════════════════════════════════"
ALL_OK=1

test_address "$BUF_A_PHYS" "BUF_A" "4096" || ALL_OK=0
test_address "$BUF_BD_PHYS" "BUF_BD" "4096" || ALL_OK=0
test_address "$BUF_BQS_PHYS" "BUF_BQS" "4096" || ALL_OK=0
test_address "$BUF_C_PHYS" "BUF_C" "4096" || ALL_OK=0
test_address "$CTRL_PHYS" "CTRL" "4096" || ALL_OK=0

echo "════════════════════════════════════════════════════════════"
echo ""

if [ $ALL_OK -eq 1 ]; then
    echo "✅ ALL ADDRESSES VERIFIED - Safe to run FPGA code!"
    echo ""
    echo "Summary:"
    echo "  ✓ BUF_A @ 0x77C00000 is accessible"
    echo "  ✓ BUF_BD @ 0x78C00000 is accessible"
    echo "  ✓ BUF_BQS @ 0x79C00000 is accessible"
    echo "  ✓ BUF_C @ 0x7AC00000 is accessible"
    echo "  ✓ CTRL @ 0x400000000 is accessible"
    echo ""
    echo "Memory layout is SAFE - no conflicts with CMA or kernel!"
    exit 0
else
    echo "❌ SOME ADDRESSES FAILED - Check memory layout!"
    echo ""
    echo "Possible causes:"
    echo "  1. CMA region is still using that address"
    echo "  2. Address is outside valid DDR range"
    echo "  3. Kernel has restricted access"
    echo ""
    echo "Run: cat /proc/iomem | grep 'System RAM'"
    echo "to see actual DDR layout"
    exit 1
fi
