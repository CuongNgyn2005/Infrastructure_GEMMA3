#!/bin/bash
# Script để kiểm tra Linux memory layout trên ZCU104
# Sử dụng khi SSH vào board hoặc qua serial console

echo "═══════════════════════════════════════════════════════════"
echo "1. KIỂM TRA TỔNG RAM VÀ LINUX MEMORY USAGE"
echo "═══════════════════════════════════════════════════════════"
cat /proc/meminfo

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "2. KIỂM TRA KERNEL BOOT MESSAGES (tìm 'Memory:' line)"
echo "═══════════════════════════════════════════════════════════"
dmesg | grep -i "memory\|DDR"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "3. KIỂM TRA ADDRESS MAPPING (IOMEM) - QUAN TRỌNG NHẤT!"
echo "═══════════════════════════════════════════════════════════"
cat /proc/iomem

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "4. KIỂM TRA CMA (Contiguous Memory Allocator) - nếu có"
echo "═══════════════════════════════════════════════════════════"
dmesg | grep -i "cma"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "5. KIỂM TRA MEMORY ZONES"
echo "═══════════════════════════════════════════════════════════"
cat /proc/zoneinfo | head -50

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "6. KIỂM TRA RESERVED MEMORY (Device Tree)"
echo "═══════════════════════════════════════════════════════════"
if [ -f /sys/firmware/devicetree/base/reserved-memory ]; then
    echo "Reserved memory regions found:"
    ls -la /sys/firmware/devicetree/base/reserved-memory/ 2>/dev/null || echo "None"
else
    echo "No reserved memory regions in device tree"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "7. PHYSICAL ADDRESS RANGE CÓ THỂ DÙNG CHO BUFFER"
echo "═══════════════════════════════════════════════════════════"
# Calculate usable range
echo "Dựa trên /proc/iomem, vùng System RAM:"
grep "System RAM" /proc/iomem

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "8. KIỂM TRA CURRENT BUFFER ADDRESSES (nếu đã setup)"
echo "═══════════════════════════════════════════════════════════"
cat /proc/iomem | grep -E "6DC00000|6EC00000|6FC00000|70C00000" || echo "Buffers not yet mapped"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "SUMMARY: Sơ đồ DDR layout dự tính của bạn"
echo "═══════════════════════════════════════════════════════════"
cat << 'EOF'
0x00000000 ┌─────────────────────────────────┐
           │  Linux Kernel/Heap (~1GB)       │ ← Cần xác nhận kích thước thực
           │  (load model, malloc, stack)    │
0x6DC00000 ├─────────────────────────────────┤
           │  BUF_A    64MB                  │ (0x6DC00000 ~ 0x6FBFFFFF)
0x6EC00000 ├─────────────────────────────────┤
           │  BUF_BD   64MB                  │ (0x6EC00000 ~ 0x72BFFFFF)
0x6FC00000 ├─────────────────────────────────┤
           │  BUF_BQS  64MB                  │ (0x6FC00000 ~ 0x73BFFFFF)
0x70C00000 ├─────────────────────────────────┤
           │  BUF_C    64MB                  │ (0x70C00000 ~ 0x74BFFFFF)
0x7FFFFFFF └─────────────────────────────────┘
           Hết 2GB DDR
EOF
