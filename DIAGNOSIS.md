# 🔴 CHẨN ĐOÁN VẤN ĐỀ FPGA ACCELERATOR

## Triệu Chứng
```
Call 1-3 (K=1152): ✓ Output OK
Call 4 (K=1024, N=1152): ✗ Output C = NaN
Call 5 (K=1152, N=6912): ✗ Input A = NaN (Corruption!)
```

## Vấn Đề #1: BUG TRONG KERNEL - DEQUANTIZE F16→F32 SAI ⚠️

**File**: `hls_accelerator/kernel_forward.cpp` (dòng 112)

```cpp
float scale = f16_safe(B_d_tile[tn][tk / 32]);  // ← BUG!
float b_val = (float)(B_qs_tile[tn][tk]) * scale;
```

**Lỗi**: Khi K=1024, ta có:
- TILE_K = 64
- Tiling loop: k0 = 0, 64, 128, ..., 960, 1024
- Mỗi tile load B_d_tile[TILE_N][TILE_K/32] = [64][2]
- Trong compute loop: `tk/32` = 0 hoặc 1 ✓ OK

Nhưng **dequant logic có sự không khớp**:
- B_d được repacked từ block layout: `B_d[n][k_block]`
- Nhưng kernel access `B_d_tile[tn][tk/32]` - **GIẢ SỬ các blocks LIÊN TỤC**

**Khi nào bug lộ diện?**
- Khi K thay đổi (1024 vs 1152), số blocks/hàng thay đổi
- K=1024: 32 blocks/hàng, K=1152: 36 blocks/hàng
- Nếu dữ liệu B không được layout theo đúng cách, kernel đọc scale NHẦM!

---

## Vấn Đề #2: MEMORY CORRUPTION DO SAI STRIDE 🚨

**File**: `ggml/src/ggml-cpu/fpga_host.cpp` (dòng 173-177)

```cpp
size_t sz_Bd     = (size_t)(K / QK8_0) * N * sizeof(uint16_t);
size_t sz_Bqs    = (size_t)K * N * sizeof(int8_t);
```

**Vấn đề**: Nếu dữ liệu B từ ggml **không phải row-major layout mà column-major**, thì:
- Repacking (dòng 316-323) tính `num_blocks = K*N/32`
- Nhưng index `i*QK8_0` vào `g_B_qs_buf` không khớp với thứ tự block thực tế

**Kết quả**: Dữ liệu B bị scrambled → Kernel nhân sai → Output C = NaN

---

## Vấn Đề #3: RACE CONDITION - KERNEL CHƯA KẾT THÚC 🔄

**File**: `fpga_host.cpp` (dòng 204-229)

```cpp
while (!(rd32(REG_CTRL) & 0x4)) { }  // Chờ ready

memcpy(g_buf_A, A, sz_A);            // ← CPU ghi A
if (b_changed) {
    memcpy(g_buf_Bd,  B_d,  sz_Bd);  // ← CPU ghi B
}

__sync_synchronize();
wr32(REG_CTRL, 0x1);                 // ← Trigger kernel
while (!(rd32(REG_CTRL) & 0x2)) { }  // Chờ done

memcpy(C, g_buf_C, sz_C_real);       // ← Copy kết quả
// ← MỠ: Không clear done flag!
```

**Vấn đề**: 
- Nếu FPGA kernel nhận M=M_pad nhưng không biết M_real, nó sẽ ghi M_pad*N phần tử vào buffer C
- Khi M_pad > M_real, kernel ghi vượt ranh giới buffer
- Dữ liệu vượt ranh có thể ghi vào bộ nhớ liền kề (cache của lần call tiếp theo)

**Ví dụ**:
```
Call N-1: M_pad=16, N=1024 → Kernel ghi 16*1024=16384 phần tử
Call N:   M_pad=32, N=1152 → Kernel chuẩn bị ghi 32*1152=36864 phần tử
          Nhưng nếu kernel lỗi hoặc timeout, dữ liệu cũ vẫn ở trong buffer
          → Dữ liệu A của call N bị contaminate!
```

---

## Vấn Đề #4: LAYOUT MATRIX GGML KHÔNG KHỚP KERNEL EXPECTATION 📐

**File**: `fpga_host.cpp` (dòng 316-323)

```cpp
const int num_blocks = (int)((K * N) / QK8_0);
const block_q8_0_t* blocks = (const block_q8_0_t*)src0->data;

#pragma omp parallel for
for (int i = 0; i < num_blocks; i++) {
    g_B_d_buf[i] = blocks[i].d;           // ← Giả sử blocks[i] = scale của block thứ i
    memcpy(&g_B_qs_buf[i*32], blocks[i].qs, 32);
}
```

**Vấn đề**: Code giả sử `blocks` là mảng **flat** [block_00, block_01, ..., block_ij, ...].

Nhưng **ggml Q8_0 layout** có thể là:
1. **Row-major**: blocks xếp theo hàng n, cột k → đúng
2. **Column-major**: blocks xếp theo cột k, hàng n → **SAI!**
3. **Interleaved layout**: Blocks xếp theo cách khác → **SAI!**

Nếu layout khác, khi kernel access `B_d[n*(K/32)+k]`, nó sẽ đọc scale SAI!

---

## Kết Luận

**Nguyên nhân GỐC là**: 
1. ⚠️ **Matrix layout ggml ↔ FPGA kernel mismatch** (VẤN ĐỀ CHÍNH)
2. 🔴 **Kernel dequant logic không robust khi K thay đổi**
3. 🚨 **Synchronization issue giữa CPU ↔ FPGA buffer**

---

