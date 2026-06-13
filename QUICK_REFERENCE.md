# 🚀 Quick Reference: FPGA Integration Cheatsheet

## File Locations Reference

```bash
# FPGA Host Interface & Implementation
./ggml/src/ggml-cpu/fpga_host.h        ← Add extended function signature
./ggml/src/ggml-cpu/fpga_host.cpp      ← Implement extended functions + regs

# GGML Dispatchers (Operation Intercepts)
./ggml/src/ggml-cpu/ggml-cpu.c:1230    ← MUL_MAT dispatcher (matmul)
./ggml/src/ggml-cpu/ggml-cpu.c:1999    ← FLASH_ATTN_EXT dispatcher (attention)
./ggml/src/ggml-cpu/ggml-cpu.c:1911    ← ROPE dispatcher (position encoding)

# Graph Construction
./src/llama-model.cpp:19015             ← build_graph() - where ops are built
./src/llama-graph.cpp:1352              ← flash_attn_ext operation creation
./src/llama-graph.cpp:339+              ← K-V cache handling

# Context & K-V Management
./src/llama-context.cpp:732             ← process_ubatch() main loop
./src/llama-context.cpp:1441            ← graph_compute() execution
./src/llama-kv-cache.cpp:646            ← K-V cache graph compute

# HLS Kernel (Already Updated ✓)
./hls_accelerator/kernel_forward.cpp    ← Your 2-branch kernel (is_attn)
```

---

## Critical Code Snippets

### Register Definitions (fpga_host.cpp)
```cpp
#define REG_CTRL     0x00  // bit0=start, bit1=done, bit2=idle
#define REG_A_LO     0x10  // Buffer addresses (64-bit split)
#define REG_A_HI     0x14
#define REG_BD_LO    0x1C
#define REG_BD_HI    0x20
#define REG_BQS_LO   0x28
#define REG_BQS_HI   0x2C
#define REG_C_LO     0x34
#define REG_C_HI     0x38
#define REG_M        0x40  // Dimensions
#define REG_K        0x48
#define REG_N        0x50
#define REG_LAYER_ID 0x58  // NEW⭐
#define REG_SEQ_POS  0x60  // NEW⭐
#define REG_IS_ATTN  0x68  // NEW⭐
```

### Read/Write Register Helpers (Already exist)
```cpp
static void wr32(uint32_t off, uint32_t val) { g_ctrl[off/4] = val; }
static uint32_t rd32(uint32_t off) { return g_ctrl[off/4]; }

// Usage:
wr32(REG_LAYER_ID, (uint32_t)layer_id);
wr32(REG_SEQ_POS, (uint32_t)seq_pos);
wr32(REG_IS_ATTN, (uint32_t)is_attn);
```

### FPGA Handshake Sequence
```cpp
// 1. Write parameters to control registers
wr32(REG_A_LO, (uint32_t)(addr & 0xFFFFFFFF));
wr32(REG_A_HI, (uint32_t)(addr >> 32));
// ... other buffer addresses ...
wr32(REG_M, (uint32_t)M);
wr32(REG_K, (uint32_t)K);
wr32(REG_N, (uint32_t)N);
wr32(REG_LAYER_ID, (uint32_t)layer_id);  // NEW
wr32(REG_SEQ_POS, (uint32_t)seq_pos);    // NEW
wr32(REG_IS_ATTN, (uint32_t)is_attn);    // NEW

// 2. Memory barrier
__sync_synchronize();

// 3. Trigger kernel start
wr32(REG_CTRL, 0x1);

// 4. Poll for completion
while (!(rd32(REG_CTRL) & 0x2)) { /* busy wait */ }

// 5. Read results
memcpy(host_output, fpga_output_buffer, output_size);
```

---

## Dispatcher Modifications Checklist

### ✅ MUL_MAT Dispatcher (ggml-cpu.c:1230)
```cpp
// BEFORE
if (fpga_try_matmul(src0, src1, dst, ith)) {
    return;
}

// AFTER
int layer_id = extract_layer_from_src(src0);
if (fpga_try_matmul_extended(src0, src1, dst, ith, 
                            layer_id, 0, 0)) {  // is_attn=0
    return;
}
```

### ✅ FLASH_ATTN Dispatcher (ggml-cpu.c:1999)
```cpp
// NEW: Add this case
case GGML_OP_FLASH_ATTN_EXT: {
    int layer_id = extract_layer_from_src(node->src[0]);
    int seq_pos = get_current_seq_pos();
    
    if (fpga_try_matmul_extended(
            node->src[1], node->src[0], node,
            ith, layer_id, seq_pos, 1)) {  // is_attn=1
        *compute_status = GGML_COMPUTE_STATUS_COMPLETE;
        return;
    }
}
```

### ✅ Helper Functions (ggml-cpu.c)
```cpp
// Parse layer ID from tensor name
int extract_layer_from_src(struct ggml_tensor * src) {
    int layer = 0;
    if (src->name) {
        sscanf(src->name, "layers.%d", &layer);
    }
    return layer;
}

// Get current position (implement in context)
int get_current_seq_pos(void) {
    return g_current_seq_pos;  // Global tracker
}

// Check if dimensions work for FPGA
int should_use_fpga_attention(struct ggml_tensor * Q,
                               struct ggml_tensor * K,
                               struct ggml_tensor * V) {
    return (Q->ne[0] == 64 && K->ne[0] == 64 && V->ne[0] == 64);
}
```

---

## Kernel Interface (Already in kernel_forward.cpp ✓)

### Parameters Passed to Kernel
```cpp
void kernel_forward(
    const float* A,       // Query (attention) or Activation (matmul)
    const uint16_t* B_d,  // Scales (fp16)
    const int8_t* B_qs,   // Quantized values or K,V data
    float* C,             // Output
    int M, int K, int N,  // Dimensions
    
    // NEW PARAMETERS (via control registers)
    int layer_id,         // Register @ 0x58
    int pos,              // Register @ 0x60 
    int is_attn)          // Register @ 0x68
{
    if (is_attn == 0) {
        // Regular matmul - uses BRAM weights
        // Load A, B_d, B_qs from DDR → compute with BRAM → store C
    } else {
        // Attention with K-V cache
        // Load Q, K, V from DDR → store in URAM → compute attn → output C
    }
}
```

---

## Data Flow Diagrams

### MUL_MAT Path (Regular Matmul)
```
Host CPU                          FPGA
───────────────────────────────────────
1. prepare A, B_d, B_qs in DDR
   via fpga buffers
                                  
2. wr32(REG_A_LO, ...)    ──────→ Load buffers
   wr32(REG_LAYER_ID, 5)
   wr32(REG_IS_ATTN, 0)
                            
3. wr32(REG_CTRL, 1)      ──────→ Start kernel
                            
4.                         ┌─────→ is_attn==0 branch
                           │       Load A, B_qs
                           │       Use BRAM weights 
                           │       Compute matmul
                           │       Store C
                           
5. wait done      ←──────────────  rd32(REG_CTRL) bit1
   
6. memcpy(output, C)   ←#────────  DMA read output
```

### ATTENTION Path (K-V Cache)
```
Host CPU                          FPGA URAM (on-chip)
───────────────────────────────────────
1. prepare Q, K, V in DDR
   via fpga buffers
                                  
2. wr32(REG_LAYER_ID, 5)   ─────→
   wr32(REG_SEQ_POS, 128)  ─────→
   wr32(REG_IS_ATTN, 1)    ─────→
                            
3. wr32(REG_CTRL, 1)      ─────→ Start kernel
                            
4.                         ┌────→ is_attn==1 branch
                           │      Load Q, K, V
                           │      Cache K,V @ [layer][pos]
                           │      Compute attention
                           │      Store output
                           │      
                           │      URAM k_cache[5][0..128]
                           │      URAM v_cache[5][0..128]
                           │      ↑ Persistent across calls!
                           
5. wait done      ←──────────────  
   
6. seq_pos++      → Next call uses [0..129]
   wr32(REG_SEQ_POS, 129)
```

---

## Compiler & Build Flags

### CMake Configuration
```bash
# Build with FPGA support
cmake .. -DUSE_FPGA=ON -DCMAKE_BUILD_TYPE=Release

# If custom flags needed
cmake .. \
  -DUSE_FPGA=ON \
  -DMAX_LAYERS=32 \
  -DMAX_CTX=512 \
  -DHEAD_DIM=64 \
  -DCMAKE_BUILD_TYPE=Release
```

### Compiler Defines (fpga_host.cpp)
```cpp
#define MAX_LAYERS 32
#define MAX_CTX    512
#define HEAD_DIM   64
#define QK8_0      32

#define FPGA_LOG_LEVEL_INFO    1
#define FPGA_LOG_LEVEL_MATMUL  1
#define FPGA_LOG_LEVEL_REG     0
#define FPGA_LOG_LEVEL_TIMING  1
```

---

## Debugging Techniques

### 1. Check What Operations FPGA Sees
```bash
# Monitor FPGA debug log
tail -f /tmp/fpga_debug.log | grep -E "MATMUL|layer|is_attn|pos"

# Expected output:
# [FPGA][MATMUL] called layer=0 pos=0 attn=0 (MLP)
# [FPGA][MATMUL] called layer=0 pos=0 attn=1 (Attention)
```

### 2. Verify Data Reaches FPGA
```cpp
// Add to fpga_run_matmul_internal():
LOGT("A[0:8]: %f, %f, %f, %f, %f, %f, %f, %f",
     ((float*)A)[0], ((float*)A)[1], ..., ((float*)A)[7]);
LOGT("B_qs[0:8]: %d, %d, %d, %d, %d, %d, %d, %d",
     ((int8_t*)B_qs)[0], ..., ((int8_t*)B_qs)[7]);
LOGT("Output C[0:8]: %f, %f, %f, %f, %f, %f, %f, %f",
     ((float*)C)[0], ..., ((float*)C)[7]);
```

### 3. Compare CPU vs FPGA Output
```cpp
// Enable both paths and compare results
int cpu_result = ggml_compute_forward_flash_attn_ext(params, node);
int fpga_result = fpga_try_matmul_extended(...);

if (memcmp(cpu_output, fpga_output, sizeof_output) != 0) {
    LOGE("Output mismatch at layer %d pos %d", layer, pos);
    // Print first differing element
}
```

### 4. Check Register Values
```cpp
uint32_t ctrl = rd32(REG_CTRL);
LOGI("REG_CTRL = 0x%08X (idle=%d, done=%d, start=%d)",
     ctrl, !!(ctrl & 0x4), !!(ctrl & 0x2), !!(ctrl & 0x1));

uint32_t layer = rd32(REG_LAYER_ID);
uint32_t pos = rd32(REG_SEQ_POS);
uint32_t attn = rd32(REG_IS_ATTN);
LOGI("REG_LAYER=%d, REG_POS=%d, REG_ATTN=%d", layer, pos, attn);
```

---

## Performance Measurement

### Minimal Latency Tracking
```cpp
// In fpga_run_matmul_internal():
double t_start = fpga_now_ms();

// ... FPGA operations ...

double t_end = fpga_now_ms();
LOGT("FPGA kernel: %.2f ms (M=%d K=%d N=%d is_attn=%d)",
     t_end - t_start, M_pad, K, N, g_is_attention_op);
```

### Per-Layer Performance
```cpp
// Track total time per layer
static double g_layer_timing[MAX_LAYERS] = {0};

g_layer_timing[layer_id] += (t_end - t_start);

// Print summary at end
LOGI("=== Per-Layer FPGA Timing ===");
for (int i = 0; i < MAX_LAYERS; i++) {
    if (g_layer_timing[i] > 0) {
        LOGI("Layer %2d: %.2f ms", i, g_layer_timing[i]);
    }
}
```

---

## Testing Progression

```bash
# Step 1: Basic FPGA matmul (MLP only, no attention)
sudo ./llama-run -m model.gguf -n 10 -c 64 <<< "test"
# Expected: Tokens generated, some MLP ops on FPGA

# Step 2: Enable attention
sudo ./llama-run -m model.gguf -n 32 -c 256 <<< "hello"
# Expected: Both MLP and attention on FPGA

# Step 3: Longer context
sudo ./llama-run -m model.gguf -n 128 -c 512 <<< "long story"
# Expected: Cache fills to 512, positions 0-512 all FPGA

# Step 4: Compare with CPU
# Rebuild without FPGA, time inference
sudo ./llama-run -m model.gguf -n 128 -c 512 <<< "long story"
# Compare: FPGA should be faster

# Step 5: Multi-session inference
# Run inference 5 times, verify consistency
for i in {1..5}; do
  echo "Test $i"
  sudo ./llama-run -m model.gguf -n 100 -c 256 <<< "prompt" | head -20
done
# Expected: Identical output across runs
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `mmap() failed @ 0x...` | Address out of range | Check buffer size matches board (51MB vs 64MB) |
| `Segfault in fpga_try_matmul` | NULL pointer | Initialize FPGA with `fpga_init()` first |
| `All NaN output` | Buffer overflow | Verify buffer addresses don't overlap |
| `Kernel timeout` | Infinite loop in HLS | Check loop bounds in kernel_forward.cpp |
| `Wrong results` | K-V cache not cleared | Call `fpga_reset_kv_cache()` on new sequence |
| `Layer ID mismatch` | Tensor name parsing failed | Check tensor names contain "layers.X" pattern |

---

**Print this page! Keep it handy while coding!** 📌

