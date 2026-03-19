# BUILD GUIDE - ZCU104-02 (1.7GB RAM)

## 📋 **What Was Updated**

### 1. Buffer Addresses - `fpga_host.cpp` (lines 86-139)

**For Board ZCU104-02 (1.7GB):**
```cpp
// Placed BEFORE CMA (0x60000000-0x70000000)
#define BUF_A_PHYS_B02   0x50000000ULL  // 51MB
#define BUF_BD_PHYS_B02  0x53300000ULL  // 51MB
#define BUF_BQS_PHYS_B02 0x56600000ULL  // 51MB
#define BUF_C_PHYS_B02   0x59900000ULL  // 51MB
#define BUF_SIZE_B02     0x3300000       // 51MB (51,855,360 bytes)
```

### 2. Max Matrix Dimensions - `fpga_host.cpp` (lines 148-151)

**For Board ZCU104-02 (51MB buffers):**
```cpp
#define FPGA_MAX_K_B02   6144   // was 8192
#define FPGA_MAX_N_B02   6144   // was 8192
#define FPGA_MAX_M       512    // unchanged
```

---

## 🎯 **To Build for Board ZCU104-02**

### **Option 1: Custom Build Flag** ✅ RECOMMENDED

```bash
cd ~/soc/GEMMA3.cpp-MODEL-IN-FPGA
rm -rf build && mkdir build && cd build

# Build with board 02 configuration
cmake .. \
  -DUSE_FPGA=ON \
  -DUSE_ZCU104_02=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

After build:
```bash
./bin/llama-run -m model.gguf -n 128 -c 512
```

### **Option 2: Default Build (uses Board 01 config)**

```bash
cd ~/soc/GEMMA3.cpp-MODEL-IN-FPGA
rm -rf build && mkdir build && cd build

cmake .. -DUSE_FPGA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This will try to use Board 01 addresses (might fail on Board 02).

---

## ⚠️ **Important: Compile Flag for Board 02**

The code checks for `USE_ZCU104_02` flag at compile time:

```cpp
#ifdef USE_ZCU104_02
    // Use Board 02 addresses (51MB buffers before CMA)
    *buf_a = BUF_A_PHYS_B02;    // 0x50000000
    ...
#else
    // Use Board 01 addresses (64MB buffers after CMA)
    *buf_a = BUF_A_PHYS_B01;    // 0x77C00000
    ...
#endif
```

**MUST pass `-DUSE_ZCU104_02=ON` when building for Board 02!**

---

## 📊 **Memory Layout Comparison**

### Board ZCU104-01 (2.0 GB RAM)
```
0x00000000 ├─ Kernel + Heap
0x6B800000 ├─ CMA (256MB)
0x77800000 ├─ FREE
0x77C00000 ├─ FPGA BUF_A (64MB)  ← 0x77C00000-0x7BBFFFFF
0x78C00000 ├─ FPGA BUF_BD (64MB) ← 0x78C00000-0x7CBFFFFF
0x79C00000 ├─ FPGA BUF_BQS (64MB)← 0x79C00000-0x7DBFFFFF
0x7AC00000 ├─ FPGA BUF_C (64MB)  ← 0x7AC00000-0x7EBFFFFF
0x7FFFFFFF └─ End of RAM
```

### Board ZCU104-02 (1.7 GB RAM) - NEW
```
0x00000000 ├─ Kernel + Heap
0x50000000 ├─ FPGA BUF_A (51MB)  ← 0x50000000-0x53300000 NEW
0x53300000 ├─ FPGA BUF_BD (51MB) ← 0x53300000-0x56600000 NEW
0x56600000 ├─ FPGA BUF_BQS (51MB)← 0x56600000-0x59900000 NEW
0x59900000 ├─ FPGA BUF_C (51MB)  ← 0x59900000-0x5CCC0000 NEW
0x60000000 ├─ CMA (256MB) [larger on this board]
0x6C000000 └─ End of RAM (~1.7GB)
```

---

## ✅ **Verification Before Build**

On Board ZCU104-02, verify:

```bash
# 1. Check total RAM
cat /proc/meminfo | head -1
# Should show: MemTotal: 1775444 kB (1.7GB)

# 2. Check CMA location
dmesg | grep "cma: Reserved"
# Should show: cma: Reserved 256 MiB at 0x0000000060000000

# 3. Verify /dev/mem exists
ls -la /dev/mem
# Should show: crw-r----- 1 root kmem

# 4. Check free memory
free -h
# Should show ~1.4Gi available
```

All checks should pass before building!

---

## 🔨 **Build Steps for Board ZCU104-02**

```bash
# 1. Enter project directory
cd ~/soc/GEMMA3.cpp-MODEL-IN-FPGA

# 2. Clean old build
rm -rf build

# 3. Create new build directory
mkdir build && cd build

# 4. Configure with board 02 flag ← IMPORTANT!
cmake .. \
  -DUSE_FPGA=ON \
  -DUSE_ZCU104_02=ON \
  -DCMAKE_BUILD_TYPE=Release

# 5. Build (takes 15-20 min)
make -j$(nproc)

# 6. Check success
ls -lah bin/llama-run
# Should show binary exists with good size
```

Expected output:
```
[100%] Built target llama-run
--- Done ---
```

---

## 📈 **Max Matrix Sizes**

### Board ZCU104-01
- Max dimensions: **512 × 8192 × 8192**
- Max B_qs size: 8192 × 8192 × 1 = 64MB ✓

### Board ZCU104-02
- Max dimensions: **512 × 6144 × 6144**
- Max B_qs size: 6144 × 6144 × 1 = 37.7MB ✓
- Trade-off: ~25% smaller matrices but same performance gain

---

## 🚀 **After Build - Test on Hardware**

```bash
# 1. Load bitstream (if not already loaded)
sudo fpgautil -b DATN1_wrapper.bit

# 2. Run inference
cd ~/soc/GEMMA3.cpp-MODEL-IN-FPGA/build
sudo ./bin/llama-run -m model.gguf -n 128 -c 512

# 3. Monitor logs
tail -f /tmp/fpga_debug.log
```

Expected logs:
```
[FPGA] init OK — ctrl@0x400000000, DDR bufs@0x50000000..0x5ccc0000
[FPGA][MATMUL] called M=512 K=6144 N=6144
[FPGA][MATMUL] >>> FPGA #1
[FPGA][MATMUL] <<< FPGA OK
```

---

## ❌ **Troubleshooting**

### Error: "mmap failed @ 0x50000000"
- Address collision with kernel/CMA
- Try reducing buffer size further or relocating
- Run: `cat /proc/iomem` to see actual memory layout

### Error: "FPGA_MAX_K too  large"
- Matrix exceeds 6144 limit for board 02
- Reduce model size or use board 01 (2GB)

### Build fails with DUSE_ZCU104_02 not defined
- Make sure CMakeLists recognizes the flag
- Try: `cmake --help-variable USE_ZCU104_02`
- Or manually edit FPGA_MAX_K/N in source

---

## 📝 **Summary**

| Item | Board 01 | Board 02 |
|------|----------|----------|
| RAM | 2GB | 1.7GB |
| Buffer size | 64MB | 51MB |
| Max K | 8192 | 6144 |
| Max N | 8192 | 6144 |
| Build flag | (none) | `-DUSE_ZCU104_02=ON` |

---

**Ready to build?** Use the command above with `-DUSE_ZCU104_02=ON`! 🚀
