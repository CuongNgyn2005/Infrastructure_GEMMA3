# 📦 K-V Cache Management: URAM Tracking Strategy

## 🔄 The Complete K-V Cache Flow

```
Token 0: Generate Q[0], compute K[0], V[0] → Store in URAM[0]
Token 1: Generate Q[1], compute K[1], V[1] → Store in URAM[1]
         When computing attn: Use URAM[0..1] for all past tokens
Token 2: Generate Q[2], compute K[2], V[2] → Store in URAM[2]
         When computing attn: Use URAM[0..2] for all past tokens
...
Token 512: Max context - URAM is FULL
          When adding token 513: Need to shift or reset cache
```

**Key Insight**: Your kernel stores K,V automatically on each attention call! You just need to track position.

---

## Where to Track seq_pos in llama.cpp

### **Location 1: Context State Machine**

**File**: `./src/llama-context.cpp`  
**Function**: `llama_context::process_ubatch()` (~line 732)

This is the main inference loop. Add position tracking:

```cpp
// Around line 732 in process_ubatch():
struct llama_context::state {
    // ... existing members ...
    
    int kv_cache_seq_pos;  // NEW: Track where we are in cache
    int kv_cache_layer;    // NEW: Which layer's cache is live
    
    state() : kv_cache_seq_pos(0), kv_cache_layer(-1) {}
};

// When processing ubatch
void llama_context::process_ubatch(...) {
    // At start of ubatch processing
    if (ubatch.n_tokens == 1 && ubatch.pos[0] > 0) {
        // Single token = decode phase (not prefill)
        ctx_state.kv_cache_seq_pos = ubatch.pos[0];
    }
    
    // ... rest of processing ...
}
```

### **Location 2: Graph Computation Points**

**File**: `./src/llama-context.cpp`  
**Lines**: 1441-1462 (in `graph_compute()`)

Track which attention operations are being executed:

```cpp
int llama_context::graph_compute(ggml_cgraph * gf, bool is_prefill) {
    // ... setup code ...
    
    // Before graph execution
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        
        // NEW: Look for attention operations
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            // Extract layer info
            int layer_id = extract_layer_from_name(node->name);
            
            // Set context for FPGA before scheduler runs
            fpga_set_context(layer_id, this->kv_cache_seq_pos, 1);
            
            LLAMA_LOG_DEBUG("Attention @ layer %d, seq_pos %d\n", 
                            layer_id, this->kv_cache_seq_pos);
        }
    }
    
    // Execute graph with scheduler
    ggml_backend_sched_graph_compute_async(sched, gf);
    // ... wait ...
    
    return 0;
}
```

### **Location 3: After Token Generation**

**File**: `./src/llama-context.cpp`  
**Function**: After `graph_compute()` completes (around line 1470)

Update position for next token:

```cpp
// After graph computation in llama_context::decode()
bool llama_context::decode(...) {
    // ... actual decode logic ...
    
    // After successful decode
    if (decode_success) {
        // Update cache position for next token
        if (n_tokens == 1) {
            // Single token prediction
            this->kv_cache_seq_pos++;
        } else if (n_tokens > 1) {
            // Batch processing (prefill)
            this->kv_cache_seq_pos = ubatch.all_pos[ubatch.all_pos.size() - 1];
        }
        
        // Check cache limit
        if (this->kv_cache_seq_pos >= MAX_CTX) {
            LLAMA_LOG_WARN("K-V cache full at position %d\n", 
                           this->kv_cache_seq_pos);
            // TODO: Implement cache shift or reset
        }
    }
    
    return decode_success;
}
```

---

## 🔴 Critical: seq_pos in PREFILL vs DECODE

Your kernel needs to handle TWO phases differently:

### **PREFILL Phase** (First tokens, batch processing)
```
Compute position embeddings for all tokens:
  pos[0] = 0, pos[1] = 1, pos[2] = 2, ...
  
BUT: Store ALL K,V in URAM immediately!

for token in [0..127]:
    compute K[token] → store in k_cache[layer][token]
    compute V[token] → store in v_cache[layer][token]
    
When computing attention at step 127:
    Use all k_cache[layer][0..127], v_cache[layer][0..127]
```

### **DECODE Phase** (One token at a time)
```
Generate single new token:
  pos = 128
  
Compute Q[128], K[128], V[128]
Store K[128] → v_cache[layer][128]
         V[128] → v_cache[layer][128]
         
Attention uses k_cache[0..128], v_cache[0..128]
Total: 129 attention operations (O(n^2) but n is small)

Next token:
  pos = 129
  ...repeat...
```

**In your kernel**, `pos` tells it which URAM slot to write to:
```cpp
// In kernel_forward.cpp, is_attn==1 branch:
k_cache[layer_id][pos][d] = B_qs[d];      // Store key at position
v_cache[layer_id][pos][d] = B_qs[HEAD_DIM + d];  // Store value

// This works for both PREFILL and DECODE!
// Just iterate with pos = 0, 1, 2, ... max(batch)
```

---

## 📊 Prefill vs Decode: Detailed Flow

### **PREFILL (batch size = 128)**
```
Host sends:
  M = 128 (num positions)
  K = 4096 (model_dim)
  N = 64 (head_dim)
  layer_id = 5
  pos = [0, 1, 2, ..., 127]
  is_attn = 1

Kernel processes:
  for m in 0..127:
      pos = m
      load Q[m]
      load K[m]
      load V[m]
      
      // Store in URAM
      k_cache[5][m][:] = K[m]
      v_cache[5][m][:] = V[m]
      
      // Attention with pos as loop limit
      for past in 0..m:
          score[past] = Q[m] • K[past]
      softmax(scores)
      
      C[m][:] output = scores @ V

Output: C[128 x 64] - attention output for all 128 positions
```

### **DECODE (batch size = 1)**
```
Host sends:
  M = 1 (single token)
  K = 4096 (model_dim) - but CPU extracts last position, pos = 128
  N = 64 (head_dim)
  layer_id = 5
  pos = 128  ← CRITICAL: Which slot in URAM to write
  is_attn = 1

Kernel processes:
  pos = 128
  
  load Q[128] (new token)
  load K[128]
  load V[128]
  
  // Store in URAM slot 128
  k_cache[5][128][:] = K[128]
  v_cache[5][128][:] = V[128]
  
  // Attention with cached values
  for past in 0..128:
      score[past] = Q[128] • k_cache[5][past]
  softmax(scores)
  
  C[0][:] = scores @ v_cache[5]

Output: C[1 x 64] - attention output for this token
```

---

## 🛠️ Implementation: Track seq_pos in Host

### **Option A: Simple Global Counter** (Easiest)
```cpp
// In fpga_host.cpp
static int g_kv_cache_seq_pos[MAX_LAYERS] = {0};

void fpga_set_kv_seq_pos(int layer_id, int pos) {
    g_kv_cache_seq_pos[layer_id] = pos;
    LOGI("KV Cache @ layer %d, pos %d", layer_id, pos);
}

void fpga_increment_kv_seq_pos(void) {
    // Called after each DECODE token
    for (int i = 0; i < MAX_LAYERS; i++) {
        g_kv_cache_seq_pos[i]++;
    }
}

void fpga_reset_kv_cache(void) {
    // Called at start of new sequence
    memset(g_kv_cache_seq_pos, 0, sizeof(g_kv_cache_seq_pos));
}
```

### **Option B: Extract from Graph** (More Robust)
```cpp
// In ggml-cpu.c dispatcher
int extract_seq_pos_from_context(struct ggml_tensor * dst) {
    // Try to parse position from llama_context
    // Or infer from batch dimensions
    
    // Heuristic: If this is FLASH_ATTN output with known dimensions,
    // we can infer approximate position
    
    return g_current_seq_pos;  // Fallback
}
```

---

## 🔄 Cache Shifting (When Full)

When `seq_pos >= MAX_CTX (512)`:

### **Simple: Reset Cache**
```cpp
void fpga_reset_kv_cache(void) {
    // Forget old K-V values
    // Kernel will treat next token as position 0
    // Downloads new attention window
    
    for (int l = 0; l < MAX_LAYERS; l++) {
        for (int pos = 0; pos < MAX_CTX; pos++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_cache[l][pos][d] = 0;
                v_cache[l][pos][d] = 0;
            }
            k_scale[l][pos] = 0.0f;
            v_scale[l][pos] = 0.0f;
        }
    }
    
    fpga_reset_kv_cache();  // Reset host state too
}
```

### **Advanced: Sliding Window**
```cpp
void fpga_shift_kv_cache(int layer_id, int shift_amount) {
    // Shift all cached K,V by shift_amount positions
    // E.g., keep last 256 positions, discard first 256
    
    // This requires DMA transfer or FPGA kernel to do:
    for (int pos = 0; pos < MAX_CTX - shift_amount; pos++) {
        memcpy(
            &k_cache[layer_id][pos][0],
            &k_cache[layer_id][pos + shift_amount][0],
            HEAD_DIM * sizeof(int8_t)
        );
        k_scale[layer_id][pos] = k_scale[layer_id][pos + shift_amount];
    }
    
    // Zero out end positions
    for (int pos = MAX_CTX - shift_amount; pos < MAX_CTX; pos++) {
        memset(&k_cache[layer_id][pos][0], 0, HEAD_DIM);
        k_scale[layer_id][pos] = 0.0f;
    }
}
```

---

## 📋 Checklist: seq_pos Tracking

- [ ] Add `seq_pos` parameter to `fpga_try_matmul_extended()`
- [ ] Add `fpga_set_context(layer_id, seq_pos, is_attn)` call in dispatcher
- [ ] Track position in `llama_context` state
- [ ] Update position after each token in decode phase
- [ ] Reset position at start of new sequence
- [ ] Handle cache overflow (512 position limit)
- [ ] Test with different sequence lengths (64, 256, 512)
- [ ] Verify K-V values accumulate correctly in FPGA

---

## 🧪 Testing K-V Cache

### **Test 1: Check Cache Storage**
```cpp
// Debug: Read back cached K values after attention op
void debug_read_kv_cache(int layer_id) {
    // This would require reading URAM back via DMA
    // For now, check logs for consistency
    
    LOGI("K-V cache for layer %d:", layer_id);
    // FPGA would output k_cache[layer_id][0..pos][0:8]
}
```

### **Test 2: Verify Attention Correctness**
```
Compare FPGA attention output with CPU:

for each position:
    FPGA output = kernel_forward(..., is_attn=1)
    CPU output = ggml_compute_forward_flash_attn_ext(...)
    
    if outputs differ:
        FAIL: Cache or computation issue
    else:
        PASS: Cache is working correctly
```

### **Test 3: Long Sequence Inference**
```bash
# Test with very long context (512 tokens)
./llama-run -m model.gguf -c 512 -n 128 <<< "Long prompt here..."

# Verify:
#  1. Tokens 0-511 all use FPGA
#  2. No manual CPU computation
#  3. Output quality good
#  4. Performance ~same from token 0 to 512
```

---

## 📝 Final Integration Points

```cpp
// 1. Initialize on startup
fpga_init();
fpga_reset_kv_cache();

// 2. At start of each sequence
llama_context::decode() {
    fpga_reset_kv_cache();  // Clear URAM
}

// 3. Before each attention operation
if (is_flash_attn) {
    fpga_set_context(layer_id, seq_pos, is_attn=1);
}

// 4. After each token generated
fpga_increment_kv_seq_pos();

// 5. When cache fills
if (seq_pos >= MAX_CTX) {
    fpga_shift_kv_cache(layer_id, 256);  // Keep last 256
}
```

---

**Your kernel already handles K-V storage! Just track seq_pos correctly!** 🎯

