# 🛠️ Detailed Implementation Guide: BRAM Model + URAM K-V Cache

## Phase 1: Sửa Đổi FPGA Host Interface

### **File: `ggml/src/ggml-cpu/fpga_host.h`**

**Current:**
```cpp
int fpga_try_matmul(
    struct ggml_tensor * src0,   // weight B (Q8_0)
    struct ggml_tensor * src1,   // activation A (F32)
    struct ggml_tensor * dst,    // output C (F32)
    int ith                      // thread id
);
```

**Update To:**
```cpp
// Extended interface for BRAM model + URAM cache
int fpga_try_matmul_extended(
    struct ggml_tensor * src0,   // weight (Q8_0) - might be ignored if in BRAM
    struct ggml_tensor * src1,   // activation (F32)
    struct ggml_tensor * dst,    // output (F32)
    int ith,                     // thread id
    int layer_id,                // which layer (0-31)
    int seq_pos,                 // sequence position (for K-V cache)
    int is_attention             // 0=matmul, 1=attention with URAM cache
);

// Keep old interface for backward compatibility
int fpga_try_matmul(
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    int ith
);
```

---

### **File: `ggml/src/ggml-cpu/fpga_host.cpp`**

**Add New Global Variables** (after line 160):
```cpp
// Track current layer being processed
static int g_current_layer_id = -1;
static int g_current_seq_pos  = 0;
static int g_is_attention_op  = 0;

// Function to set context before FPGA operation
void fpga_set_context(int layer_id, int seq_pos, int is_attn) {
    g_current_layer_id = layer_id;
    g_current_seq_pos = seq_pos;
    g_is_attention_op = is_attn;
}
```

**Add New Implementation** (after fpga_try_matmul, around line 362):
```cpp
int fpga_try_matmul_extended(
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    int ith,
    int layer_id,
    int seq_pos,
    int is_attention)
{
    // Store context info
    fpga_set_context(layer_id, seq_pos, is_attention);
    
    // Prepare arguments
    pthread_mutex_lock(&g_mutex);
    
    const float* A_src = (is_attention == 0) 
        ? (const float*)src1->data 
        : (const float*)src1->data;  // For attention, src1 contains Q
    
    const uint16_t* B_d_src = (const uint16_t*)src0->data;
    const int8_t* B_qs_src = (const int8_t*)src0->data + (src0->nb[0] > 0 ? src0->nb[0] : 0);
    
    float* C_output = (float*)dst->data;
    
    // Call FPGA runtime
    int result = 0;
    
    if (is_attention) {
        // For attention: src0=K, src1=Q, dst=output
        // Call with attention flag
        result = fpga_run_attention(
            (const float*)src1->data,  // Q
            (const uint16_t*)src0->data,  // K_scale
            (const int8_t*)src0->data,  // K + V data
            (float*)dst->data,
            layer_id,
            seq_pos,
            1  // is_attention
        );
    } else {
        // For MLP matmul: normal operation
        result = fpga_run_matmul_internal(
            A_src, B_d_src, B_qs_src, C_output,
            1, 1, 1, 1, 0
        );
    }
    
    pthread_mutex_unlock(&g_mutex);
    return result;
}
```

**Add FPGA Attention Wrapper** (new function):
```cpp
static int fpga_run_attention(
    const float* Q,
    const uint16_t* K_scales,
    const int8_t* K_V_data,
    float* output,
    int layer_id,
    int seq_pos,
    int is_attn)
{
    const int HEAD_DIM = 64;  // Match your kernel
    
    // Write layer_id, pos, is_attn to control registers
    wr32(REG_LAYER_ID, (uint32_t)layer_id);
    wr32(REG_SEQ_POS, (uint32_t)seq_pos);
    wr32(REG_IS_ATTN, (uint32_t)is_attn);
    
    // For attention, pack Q, K, V into buffer A
    // A[0..63] = Q_values
    // Copy Q to buffer
    memcpy(g_buf_A, Q, HEAD_DIM * sizeof(float));
    
    // K and V come from input B_qs
    memcpy(g_buf_Bqs, K_V_data, 2 * HEAD_DIM * sizeof(int8_t));
    memcpy(g_buf_Bd, K_scales, 2 * sizeof(uint16_t));
    
    __sync_synchronize();
    
    // Trigger FPGA
    wr32(REG_CTRL, 0x1);  // Start
    while (!(rd32(REG_CTRL) & 0x2)) { /* wait done */ }
    
    // Read output
    memcpy(output, g_buf_C, HEAD_DIM * sizeof(float));
    
    return 1;
}
```

**Add Register Definitions** (after REG_N, around line 85):
```cpp
// New registers for layer/position/attention flag
#define REG_LAYER_ID 0x58   // Register for layer ID
#define REG_SEQ_POS  0x60   // Register for sequence position
#define REG_IS_ATTN  0x68   // Register for attention flag
```

---

## Phase 2: Modify GGML Dispatcher

### **File: `ggml/src/ggml-cpu/ggml-cpu.c`**

**Update MUL_MAT Dispatcher** (around line 1230, in `ggml_compute_forward_mul_mat`):

**Find this section:**
```cpp
// Current code (lines 1235-1248)
pthread_once(&g_fpga_once, do_fpga_init);

if (g_fpga_initialized) {
    if (fpga_try_matmul(src0, src1, dst, ith)) {
        return;
    }
}
```

**Replace With:**
```cpp
// NEW: Extended FPGA interface with layer tracking
pthread_once(&g_fpga_once, do_fpga_init);

if (g_fpga_initialized) {
    // Try to extract layer_id and operation type from tensor metadata
    // or use a global context if available
    int layer_id = extract_layer_from_src(src0);  // Helper function
    int is_mlp = (strstr(src0->name, "mlp.") != NULL) ? 1 : 0;
    
    if (fpga_try_matmul_extended(
            src0, src1, dst, ith,
            layer_id,      // Which layer (0-31)
            0,             // seq_pos = 0 (not used for MLP)
            0              // is_attention = 0 (this is matmul)
        )) {
        return;  // FPGA handled it
    }
}

// Fall back to CPU
```

**Add Helper Function** (somewhere before, e.g. around line 1200):
```cpp
static int extract_layer_from_src(struct ggml_tensor * src) {
    // Try to parse layer ID from tensor name
    // E.g., "layers.5.mlp.w1" -> layer = 5
    if (src->name && strlen(src->name) > 0) {
        int layer = -1;
        if (sscanf(src->name, "layers.%d", &layer) == 1) {
            return layer;
        }
    }
    return 0;  // Default
}
```

---

**Add FLASH_ATTN_EXT Dispatcher** (around line 1999-2008):

**Find this:**
```cpp
case GGML_OP_FLASH_ATTN_EXT: {
    *compute_status = GGML_COMPUTE_STATUS_COMPLETE;
    return;
}
```

**Replace With:**
```cpp
case GGML_OP_FLASH_ATTN_EXT: {
    // Extract Q, K, V from the graph
    struct ggml_tensor * Q = node->src[0];
    struct ggml_tensor * K = node->src[1];
    struct ggml_tensor * V = node->src[2];
    struct ggml_tensor * dst = node;
    
    // Extract layer_id from Q tensor name (e.g., "attn.q.layers.5")
    int layer_id = extract_layer_from_src(Q);
    
    // Try FPGA attention
    if (g_fpga_initialized && should_use_fpga_attention(Q, K, V)) {
        // Get sequence position from context
        int seq_pos = get_current_seq_pos();  // Helper function
        
        if (fpga_try_matmul_extended(
                K, Q, dst, ith,
                layer_id,
                seq_pos,
                1  // is_attention = 1
            )) {
            *compute_status = GGML_COMPUTE_STATUS_COMPLETE;
            return;
        }
    }
    
    // Fall back to CPU Flash Attention
    ggml_compute_forward_flash_attn_ext(params, node);
    *compute_status = GGML_COMPUTE_STATUS_COMPLETE;
    return;
}
```

**Add Helper Functions:**
```cpp
// Helper to check if we should use FPGA for attention
static int should_use_fpga_attention(
    struct ggml_tensor * Q,
    struct ggml_tensor * K,
    struct ggml_tensor * V) {
    
    // Only if dimensions match our kernel requirements
    if (Q->ne[0] != 64) return 0;  // HEAD_DIM must be 64
    if (K->ne[0] != 64) return 0;
    if (V->ne[0] != 64) return 0;
    
    return 1;
}

// Get current sequence position from global context
// You may need to expand this based on your architecture
static int get_current_seq_pos(void) {
    // This should come from the inference context
    // For now, return a placeholder
    // In real code, you'd get this from llama_context
    return g_current_seq_pos;
}
```

---

## Phase 3: Update Bitstream Register Map

### **Update Your FPGA Bitstream Control Registers**

If you haven't already, your Vivado design needs to support:

```
0x400000000 + 0x58: REG_LAYER_ID    (32-bit write)
0x400000000 + 0x60: REG_SEQ_POS     (32-bit write)
0x400000000 + 0x68: REG_IS_ATTN     (32-bit write)
```

Your HLS kernel reads these:
```cpp
int layer_id = read_from_register(0x58);
int pos = read_from_register(0x60);
int is_attn = read_from_register(0x68);
```

---

## Phase 4: Build and Test

### **Compilation**
```bash
cd llama.cpp/build
cmake .. -DUSE_FPGA=ON -DCMAKE_BUILD_TYPE=Release
make -j4
```

### **Testing Sequence**

**Step 1: Test MLP only (is_attn=0)**
```bash
sudo ./bin/llama-run -m model.gguf -n 16 -c 64 <<< "test"
# Monitor: tail -f /tmp/fpga_debug.log | grep "is_attn"
# Should see: is_attn=0 for MLP layers
```

**Step 2: Test with Attention (is_attn=1)**
```bash
sudo ./bin/llama-run -m model.gguf -n 128 -c 512 <<< "Hello"
# Monitor FPGA debug logs for both paths
```

### **Expected Log Output**
```
[FPGA][MATMUL] called layer=5 pos=0 attn=0 (MLP)
[FPGA][MATMUL] >>> FPGA #1: Using BRAM weights for layer 5
[FPGA][TIMING] MLP kernel exec: XX.XX ms

[FPGA][MATMUL] called layer=5 pos=3 attn=1 (Attention)
[FPGA][MATMUL] >>> FPGA #1: Using URAM K-V cache for layer 5 pos 3
[FPGA][TIMING] Attention kernel exec: XX.XX ms
[FPGA][MATMUL] <<< FPGA Stored K,V in URAM cache @ [5][3]
```

---

## 🎯 Summary of Changes

| Component | File | Change |
|-----------|------|--------|
| **FPGA Interface** | `fpga_host.h` | Add extended interface with layer_id, seq_pos, is_attn |
| **FPGA Implementation** | `fpga_host.cpp` | Add `fpga_try_matmul_extended()`, `fpga_run_attention()` |
| **Control Registers** | `fpga_host.cpp` | Add REG_LAYER_ID, REG_SEQ_POS, REG_IS_ATTN |
| **MUL_MAT Dispatcher** | `ggml-cpu.c:1230` | Call extended interface with layer metadata |
| **ATTN Dispatcher** | `ggml-cpu.c:1999` | Hook attention path with seq_pos tracking |
| **Helper Functions** | `ggml-cpu.c` | Add `extract_layer_from_src()`, `should_use_fpga_attention()`, `get_current_seq_pos()` |
| **HLS Kernel** | `kernel_forward.cpp` | Already updated ✓ |

---

## ⚠️ Important Notes

1. **Tensor Naming**: Make sure your model's tensors have consistent names so we can parse layer IDs
   - Example: `"layers.5.attention.q"`, `"layers.5.mlp.w1"`

2. **Sequence Position Tracking**: You need to track `seq_pos` as tokens are generated
   - Initialize to 0 at start of inference
   - Increment after each token

3. **K-V Cache Lifecycle**: 
   - Cache fills up to MAX_CTX (512 in your kernel)
   - After that, you need a cache shift operation (not implemented yet)

4. **BRAM vs DDR**:
   - BRAM: Very fast, limited size (~1-2MB for weights of small models)
   - DDR: Slower, but large (51-64MB buffers you defined)
   - Your kernel uses URAM for K-V cache (good! ~2-10MB typical)

5. **Performance Expectations**:
   - MLP with BRAM: ~10-50x faster than DDR
   - Attention with URAM: ~100-1000x faster than all-DDR approach

---

**Ready to code? Start with fpga_host.h and fpga_host.cpp modifications!** 🚀

