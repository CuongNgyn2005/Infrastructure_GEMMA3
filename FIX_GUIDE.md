# 🔧 HƯỚNG DẪN FIX CHI TIẾT

## FIX #1: DEBUG INPUT LAYOUT CỦA GGML Q8_0 (ĐẠO VỪA MỘT)

Trước khi fix kernel, cần **xác nhận** cách ggml layout dữ liệu Q8_0:

### Bước 1: Thêm debug code vào fpga_host.cpp (dòng 313-327)

```cpp
if (b_changed) {
    const int num_blocks = (int)((K * N) / QK8_0);
    const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;
    
    // ===== DEBUG: XEM LAYOUT BLOCK =====
    fprintf(stderr, "[FPGA DEBUG] ===== BLOCK LAYOUT =====\n");
    fprintf(stderr, "[FPGA DEBUG] num_blocks=%d, K=%ld, N=%ld, K/32=%ld\n", 
            num_blocks, K, N, K/QK8_0);
    
    // Print block structure để thấy pattern
    for (int debug_i = 0; debug_i < 10 && debug_i < num_blocks; debug_i++) {
        fprintf(stderr, "[FPGA DEBUG] blocks[%d].d = 0x%04X, qs[0..7] = [%d,%d,%d,%d,%d,%d,%d,%d]\n",
                debug_i, blocks[debug_i].d,
                blocks[debug_i].qs[0], blocks[debug_i].qs[1], 
                blocks[debug_i].qs[2], blocks[debug_i].qs[3],
                blocks[debug_i].qs[4], blocks[debug_i].qs[5], 
                blocks[debug_i].qs[6], blocks[debug_i].qs[7]);
    }
    // ===== END DEBUG =====
    
    #pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < num_blocks; i++) {
        g_B_d_buf[i] = blocks[i].d;
        const int8_t* src_qs = blocks[i].qs;
        int8_t* dst_qs = &g_B_qs_buf[i * QK8_0];
        for (int j = 0; j < QK8_0; j++)
            dst_qs[j] = src_qs[j];
    }
    // ...
}
```

**Sau khi chạy**, xem debug output để biết:
- Blocks được layout theo thứ tự nào?
- Có pattern nào không?

---

## FIX #2: XỬ LÝ DOUBLE-BUFFER ĐỂ TRÁNH DATA RACE 🔄

**File**: `fpga_host.cpp`

**Vấn đề**: Single shared buffer g_buf_C có thể bị ghi vào lúc CPU đang đọc

**Giải pháp**: Sử dụng **2 output buffers alternating**

```cpp
// Thêm vào global section (dòng 95-115)
static void* g_buf_C_alt = NULL;     // ← Alternate output buffer
static int   g_current_c_buf = 0;    // ← Chỉ số buffer hiện tại (0 hoặc 1)
static uint64_t g_buf_C_alt_phys = 0;

// Sửa fpga_init() (dòng 136-139)
int fpga_init(void) {
    // ... existing code ...
    g_buf_C   = mmap(NULL, g_buf_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, g_buf_C_phys);
    g_buf_C_alt = mmap(NULL, g_buf_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, g_buf_C_phys + g_buf_size);
    g_buf_C_alt_phys = g_buf_C_phys + g_buf_size;
    // ...
}

// Sửa fpga_run_matmul_internal() - dòng 163-243
static int fpga_run_matmul_internal(
    const float* A,
    const uint16_t* B_d,
    const int8_t* B_qs,
    float* C,
    int M_pad, int M_real,
    int K, int N,
    int b_changed)
{
    pthread_mutex_lock(&g_mutex);
    
    // Chọn output buffer
    void* write_buf_C = (g_current_c_buf == 0) ? g_buf_C : g_buf_C_alt;
    uint64_t write_phys = (g_current_c_buf == 0) ? g_buf_C_phys : g_buf_C_alt_phys;
    
    // ... copy dữ liệu vào write_buf_C ...
    
    // Ghi địa chỉ output buffer
    wr32(REG_C_LO, (uint32_t)(write_phys & 0xFFFFFFFF));
    wr32(REG_C_HI, (uint32_t)(write_phys >> 32));
    
    // Trigger và chờ
    wr32(REG_CTRL, 0x1);
    while (!(rd32(REG_CTRL) & 0x2)) { }
    
    // Copy result từ buffer đang sử dụng (READ từ buffer cũ, không ghi)
    void* read_buf_C = (g_current_c_buf == 1) ? g_buf_C : g_buf_C_alt;
    memcpy(C, read_buf_C, sz_C_real);
    
    // Hoán đổi buffer cho lần sau
    g_current_c_buf = 1 - g_current_c_buf;
    
    pthread_mutex_unlock(&g_mutex);
    return 1;
}
```

---

## FIX #3: SỬA KERNEL - EXPLICIT BOUNDS CHECK & BETTER LAYOUT 🎯

**File**: `hls_accelerator/kernel_forward.cpp`

**Vấn đề hiện tại**:
```cpp
// Kernel nhận M (actually M_pad) nhưng không biết M_real
// Nên nó ghi data ra ngoài intended output region
```

**Giải pháp**: Truyền M_real vào kernel!

```cpp
// kernel_forward.h
extern "C" {
void kernel_forward(
    const float*    A,
    const uint16_t* B_d,
    const int8_t*   B_qs,
    float*          C,
    int M, int K, int N,
    int M_real  // ← Thêm parameter này
);
}

// kernel_forward.cpp
extern "C" {
void kernel_forward(
    const float*    A,
    const uint16_t* B_d,
    const int8_t*   B_qs,
    float*          C,
    int M, int K, int N,
    int M_real)
{
    // ... AXI interfaces, pragmas ...
    
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            // ... init, load, compute ...
            
            // STORE - chỉ ghi nếu gr < M_real
            store_C:
            for (int tm = 0; tm < TILE_M; ++tm)
                for (int tn = 0; tn < TILE_N; ++tn) {
                    #pragma HLS PIPELINE II=1
                    int gr = m0 + tm, gc = n0 + tn;
                    if (gr < M_real && gc < N)  // ← THÊM CHECK: gr < M_real
                        C[gr * N + gc] = C_tile[tm][tn];
                }
        }
    }
}
}
```

**Sửa fpga_host.cpp** để truyền M_real:
```cpp
// dòng 222
wr32(REG_M, (uint32_t)M_pad);
wr32(REG_K, (uint32_t)K);
wr32(REG_N, (uint32_t)N);
wr32(REG_M_REAL, (uint32_t)M_real);  // ← Thêm một thanh ghi mới
```

Nhưng nếu không có thanh ghi thêm, có thể **encode M_real vào bit cao của REG_M**:
```cpp
// Sửa kernel để đọc từ REG_M - bit 31-16 là M_real, bit 15-0 là M_pad
wr32(REG_M, ((uint32_t)M_real << 16) | (uint32_t)M_pad);
```

---

## FIX #4: VERIFY MATRIX LAYOUT - ADD GOLDEN REFERENCE 🧪

Tạo file test để so sánh output CPU vs FPGA:

**File**: `test_matmul_golden.cpp` (tạo mới)

```cpp
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "kernel_forward.h"

#define QK8_0 32
typedef struct {
    uint16_t d;
    int8_t   qs[QK8_0];
} block_q8_0_t;

// CPU reference implementation
void matmul_cpu_ref(
    const float* A, const uint16_t* B_d, const int8_t* B_qs,
    float* C, int M, int K, int N) {
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int block_idx = n * (K / QK8_0) + (k / QK8_0);
                int qs_idx = block_idx * QK8_0 + (k % QK8_0);
                
                uint16_t scale_fp16 = B_d[block_idx];
                float scale = f16_to_f32(scale_fp16);
                float b_val = (float)B_qs[qs_idx] * scale;
                
                sum += A[m * K + k] * b_val;
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    // TODO: Tạo test case nhỏ
    // So sánh FPGA output vs CPU output
    return 0;
}
```

---

## FIX #5: ADD COMPREHENSIVE LOGGING 📊

**File**: `fpga_log.h` (mở rộng)

```cpp
#define FPGA_LOG_LEVEL_DATA_DUMP 1  // Thêm

// Khi copy dữ liệu, in ra checksum
static uint32_t compute_checksum(const float* data, int n) {
    uint32_t cs = 0;
    for (int i = 0; i < n; i++) {
        uint32_t* p = (uint32_t*)&data[i];
        cs ^= *p;
    }
    return cs;
}

// Trong fpga_run_matmul_internal, trước khi copy A:
uint32_t a_checksum = compute_checksum(A, M_pad * K);
LOGT("A checksum = 0x%08X, size = %ld bytes", a_checksum, sz_A);
memcpy(g_buf_A, A, sz_A);

// Sau khi kernel, trước khi copy C:
LOGT("C buffer (first 8): [%f, %f, %f, %f, %f, %f, %f, %f]",
     ((float*)write_buf_C)[0], ((float*)write_buf_C)[1], ...);
```

---

## EXECUTION ORDER (Thứ Tự Fix)

1. **TRƯỚC TIÊN**: Chạy FIX #1 (Debug) → Xác nhận layout ggml
2. **SECOND**: Chạy FIX #4 (Golden Test) → Verify logic tính toán
3. **THIRD**: Chạy FIX #3 (Kernel bounds check) → Đảm bảo kernel không ghi overflow
4. **FOURTH**: Chạy FIX #2 (Double buffer) → Tránh race condition
5. **FIFTH**: Chạy FIX #5 (Logging) → Monitor kỹ lưỡng

---

