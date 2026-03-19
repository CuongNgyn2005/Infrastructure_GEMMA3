#!/bin/bash
# Hardware verification - Fixed version

echo "FPGA Hardware Check"
echo "======================================"
echo ""
echo "1. Linux Version:"
uname -a
echo ""

echo "2. Total RAM:"
cat /proc/meminfo | head -1
echo ""

echo "3. DDR Memory Layout:"
cat /proc/iomem | grep "System RAM"
echo ""

echo "4. CMA Region (from dmesg):"
dmesg | grep "cma: Reserved" | tail -1
echo ""

echo "5. Memory Stats:"
free -h
echo ""

echo "6. Check /dev/mem:"
if [ -e /dev/mem ]; then
  echo "✓ /dev/mem exists"
else
  echo "✗ /dev/mem NOT found"
fi
echo ""

echo "7. Bitstream status:"
fpgautil -d 2>&1 | head -10
echo ""

echo "======================================"
echo "Summary:"
echo "- If CMA @ 0x6B800000 and DDR shows System RAM"
echo "- And /dev/mem exists"
echo "- And bitstream shows loaded"
echo "=> OK to proceed with build"
echo ""
