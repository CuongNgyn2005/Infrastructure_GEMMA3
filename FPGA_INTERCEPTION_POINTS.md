# 🎯 Các Điểm Can Thiệp Phép Toán FPGA trong llama.cpp

## 📊 Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────┐
│ llama_context::process_ubatch()                                │
│ (Main inference pipeline)                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ model.build_graph(params)                                       │
│ (construct computation graph with operations)                   │
│ Location: ./src/llama-model.cpp:19015                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ llama_context::graph_compute(gf)                                │
│ Location: ./src/llama-context.cpp:1441                         │
│                                                                 │
│ Executes graph nodes with scheduler:                           │
│  - ggml_backend_sched_graph_compute_async()                    │
│  - Location: ./src/llama-context.cpp:1460                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ ggml_compute_forward(&params, node)                             │
│ Main dispatcher for ALL operations                              │
│ Location: ./ggml/src/ggml-cpu/ggml-cpu.c:1714                 │
│                                                                 │
│ Large switch statement (lines 1726-2070) dispatches:           │
│  - GGML_OP_MUL_MAT        ← INTERCEPTION POINT #1              │
│  - GGML_OP_FLASH_ATTN_EXT ← INTERCEPTION POINT #2              │
│  - GGML_OP_ROPE           ← INTERCEPTION POINT #3              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔴 INTERCEPTION POINT #1: Matrix Multiplication (MUL_MAT)

### **Location**
- **File**: `./ggml/src/ggml-cpu/ggml-cpu.c`
- **Function**: `ggml_compute_forward_mul_mat()` 
- **Lines**: 1230-1450
- **FPGA Hook**: Lines 1235-1248

### **Current Code Structure**
```cpp
// Line 1230-1248: Function stub
int ggml_compute_forward_mul_mat(
    struct ggml_compute_params * params,
    struct ggml_tensor * src0,  // Weights (Q8_0)
    struct ggml_tensor * src1,  // Activation (F32)
    struct ggml_tensor * dst)   // Output (F32)
{
    //... parameter checks ...
    
    #ifdef USE_FPGA
        pthread_once(&g_fpga_once, do_fpga_init);
        
        if (g_fpga_initialized) {
            if (fpga_try_matmul(src0, src1, dst, ith)) {
                return;  // FPGA handled it
            }
        }
    #endif
    
    // Fall back to CPU computation
    // Lines 1250+: CPU matmul code
}
```

### **What You Need to Do**

**Option A - Use your new BRAM model** (Recommended for performance)
1. Detect when this is a **non-attention matmul** → Regular MLP/FFN layers
2. Load model from BRAM cache (you control this in your kernel)
3. Call FPGA with **is_attn=0**
4. Let your kernel use cached weights from BRAM

**Option B - Add operation metadata tracking**
```cpp
// Add this to distinguish MLP vs Attention layers
if (is_this_mlp_layer(dst)) {
    fpga_call_with_params(src0, src1, dst, layer_id, /*is_attn*/0);
} else if (is_this_attention_layer(dst)) {
    // Don't call - attention goes to FLASH_ATTN_EXT instead
}
```

**In your updated kernel**, when `is_attn=0`:
- ✅ Uses BRAM-cached model weights
- ✅ Fast DDR operations only for activations
- ✅ Output writes back results

---

## 🟠 INTERCEPTION POINT #2: Flash Attention (FLASH_ATTN_EXT)

### **Location - Graph Construction**
- **File**: `./src/llama-graph.cpp`
- **Lines**: 1352-1357
- **Where it builds the operation**: Graph builder creates `ggml_flash_attn_ext()` nodes

### **Location - Execution Dispatch**
- **File**: `./ggml/src/ggml-cpu/ggml-cpu.c`
- **Switch case**: Lines 1999-2008 (GGML_OP_FLASH_ATTN_EXT)
- **Function**: `ggml_compute_forward_flash_attn_ext()`
- **Implementation**: `./ggml/src/ggml-cpu/ops.cpp:8923-8938`

### **Current Graph Construction** (Line 1352-1357)
```cpp
// In llama_graph.cpp - around line 1352
gf->nodes[gf->n_nodes++] = ggml_flash_attn_ext(
    cur,    // Q (query)
    Kcur,   // K (key)  
    Vcur,   // V (value)
    ...
);
```

**This is where K and V are passed!** ← **CRITICAL INTERSECTION POINT**

### **What You Need to Do**

**Step 1: Add layer_id and pos to the attention operation**
```cpp
// Modify the graph building to pass layer/position info
// In llama_graph.cpp around line 1352:

// Create a wrapper or extend the operation to include metadata
struct flash_attn_metadata {
    int layer_id;
    int seq_pos;
} metadata;

// Pack this into the operation somehow (or extend ggml_flash_attn_ext)
```

**Step 2: Intercept in the dispatcher**
```cpp
// In ggml-cpu.c, in the GGML_OP_FLASH_ATTN_EXT case (line 1999):

case GGML_OP_FLASH_ATTN_EXT: {
    // Extract layer_id and pos from the tensor metadata
    int layer_id = extract_layer_from_node(node);
    int pos = extract_pos_from_node(node);
    
    // Call FPGA attention with these parameters
    if (fpga_can_handle_attention(layer_id, pos)) {
        fpga_run_attention(
            Q, K, V,
            layer_id,  // NEW
            pos,       // NEW  
            /*is_attn*/ 1
        );
        return;  // FPGA handled it
    }
    
    // Fall back to CPU attention
    ggml_compute_forward_flash_attn_ext(...);
}
```

**In your FPGA kernel**, when `is_attn=1`:
- ✅ Loads Q, K, V from input
- ✅ Caches K, V in internal URAM
- ✅ Computes attention with URAM K-V cache
- ✅ Returns output in C

---

## 🟡 INTERCEPTION POINT #3: RoPE (Rotary Position Embedding)

### **Location - Graph Construction**
- **File**: `./src/llama-model.cpp`
- **Lines**: 6347, 6353, 6519, 6525, ...
- **Pattern**: Multiple `ggml_rope_ext()` calls

### **Location - Execution Dispatch**
- **File**: `./ggml/src/ggml-cpu/ggml-cpu.c`
- **Switch case**: Lines 1911-1918 (GGML_OP_ROPE)
- **Implementation**: `./ggml/src/ggml-cpu/ops.cpp:6677-6721`

### **Current Usage**
```cpp
// In llama-model.cpp around line 6347:
// Applies RoPE to Q and K before attention

Qcur = ggml_rope_ext(
    Qcur,              // Query matrix
    positions,         // Token positions
    rope_base,         // Frequency base
    ...
);

Kcur = ggml_rope_ext(
    Kcur,              // Key matrix
    positions,
    ...
);
```

### **What You Might Do**

**Option A - Keep RoPE on CPU** (Simpler)
- RoPE is relatively fast
- Leave it on CPU, let FPGA handle post-RoPE attention

**Option B - Move RoPE to FPGA** (Advanced)
```cpp
// In your FPGA kernel, add RoPE computation before attention:
if (is_attn == 1) {
    // Apply RoPE to Q in-place
    apply_rope_to_q(Q, pos, rope_base);
    
    // K is re-computed each time (from B_qs input)
    // But K cache stores raw K before RoPE
    // So re-apply RoPE when retrieving from cache
}
```

---

## 🟢 INTERCEPTION POINT #4: Graph Building (Build Graph Function)

### **Location**
- **Main Entry**: `./src/llama-model.cpp:19015`
- **Function**: `llama_model::build_graph(const llm_graph_params & params)`
- **Called from**: `./src/llama-context.cpp` at lines 758, 1403, 2160

### **What Happens Here**
This is where the **computation graph is constructed**. Each layer calls:
- `ggml_mul_mat()` for MLP linear layers
- `ggml_flash_attn_ext()` for attention layers
- `ggml_rope_ext()` for position embeddings
- etc.

### **How to Detect Which Operation is Which**

```cpp
// In llama-model.cpp, around the graph building section:

// You can identify layers by tracking:
for (int il = 0; il < n_layers; ++il) {
    // MLP block operations
    cur = ggml_mul_mat(mlp_w1[il], cur);  // Attention.w_v @ input
    cur = ggml_mul_mat(mlp_w2[il], cur);  // FFN up projection
    cur = ggml_mul_mat(mlp_w3[il], cur);  // FFN down projection
    
    // Attention block operations
    Qcur = ggml_mul_mat(attn_w_q[il], cur);
    Kcur = ggml_mul_mat(attn_w_k[il], cur);
    Vcur = ggml_mul_mat(attn_w_v[il], cur);
    
    Qcur = ggml_rope_ext(Qcur, ...);  // Position encoding
    Kcur = ggml_rope_ext(Kcur, ...);
    
    cur = ggml_flash_attn_ext(Qcur, Kcur, Vcur, ...);
}
```

### **What You Can Do**

**Add Metadata to Operations**
```cpp
// Extend each operation with layer information:

struct ggml_tensor * ggml_mul_mat_with_meta(
    struct ggml_tensor * a,
    struct ggml_tensor * b,
    int layer_id,           // NEW
    const char * op_type    // "mlp_w1", "mlp_w2", etc. - NEW
);

// Then in your dispatcher, you'll know:
if (op_type == "mlp_w1") {
    // This is first feedforward layer of layer_id
    // Load weights from BRAM bank for layer_id
}
```

---

## 🔵 INTERCEPTION POINT #5: K-V Cache Management

### **Location - KV Cache Graph Building**
- **File**: `./src/llama-kv-cache.cpp`
- **Function**: `llama_kv_cache::build_graph_shift()` (Line 1393)
- **Graph computation**: Line 646

### **Location - KV Cache Context Handling**
- **File**: `./src/llama-context.cpp`
- **Lines**: 302-303 (iterating graph nodes, checking for FLASH_ATTN_EXT)

### **What Happens**
- After each token, K-V cache is shifted to make room for new token
- This is where you'd **update your URAM K-V cache**

### **What You Need to Do**

**Modify after FLASH_ATTN_EXT computation**:
```cpp
// In your FPGA kernel or host code:
// After attention with is_attn=1, your kernel automatically stores K,V in URAM

// On host side, after FPGA returns:
if (is_attn == 1) {
    // The K, V are already cached on FPGA URAM
    // Just track position in software:
    current_pos = seq_length;
    
    // For next token, increment pos and call again
}

// For context shift (when cache fills up):
// Call a special operation to reset URAM cache or shift positions
if (need_cache_shift) {
    fpga_shift_kv_cache(layer_id, shift_amount);
}
```

---

## 📋 Summary: The 5 Main Can-Thiệp Points

| Priority | Point | Location | Action |
|----------|-------|----------|--------|
| **🔴 HIGH** | MUL_MAT dispatcher | `ggml-cpu.c:1230` | Hook FPGA for MLPs with BRAM weights |
| **🔴 HIGH** | FLASH_ATTN_EXT dispatcher | `ggml-cpu.c:1999` | Hook FPGA attention with K-V cache |
| **🟠 MEDIUM** | Graph builder | `llama-model.cpp:19015` | Add layer_id & operation type metadata |
| **🟡 LOW** | RoPE operations | `ggml-cpu.c:1911` | (Optional) Move to FPGA if needed |
| **🟢 LOW** | KV Cache reset | `llama-kv-cache.cpp:1393` | Manage URAM cache lifecycle |

---

## 🚀 Implementation Roadmap

### Phase 1: Get MLP working (is_attn=0)
1. Modify `fpga_host.cpp` to pass `layer_id` and `is_attn=0` to kernel
2. Update dispatch in `ggml-cpu.c:1230` to call with new params
3. Rebuild and test MLP layers with BRAM weights

### Phase 2: Add attention (is_attn=1)
1. Modify attention dispatcher `ggml-cpu.c:1999`
2. Extract `layer_id` and `pos` from tensor metadata
3. Call FPGA attention kernel
4. Verify K-V cache accumulation in URAM

### Phase 3: Optimize graph building
1. Add metadata to operations during graph construction
2. Allow dispatcher to make smarter decisions about which path to take
3. Potentially overlap computation phases

---

## 🔍 Key Code Snippets to Find

Search these in your editor:

```bash
# Find MUL_MAT dispatcher
grep -n "ggml_compute_forward_mul_mat" ggml/src/ggml-cpu/ggml-cpu.c

# Find FLASH_ATTN dispatcher
grep -n "GGML_OP_FLASH_ATTN_EXT" ggml/src/ggml-cpu/ggml-cpu.c

# Find graph building
grep -n "build_graph" src/llama-model.cpp

# Find where K-V is handled
grep -n "flash_attn_ext" src/llama-graph.cpp

# Find RoPE dispatch
grep -n "GGML_OP_ROPE" ggml/src/ggml-cpu/ggml-cpu.c
```

---

**Ready to implement? Start with Phase 1 - modify the MUL_MAT dispatcher!** 🎯

