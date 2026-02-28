// FPGA host implementation.
#include "fpga_host.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include "ggml.h"

// ================= CẤU HÌNH ĐỊA CHỈ =================
constexpr uint64_t KERNEL_CTRL_BASE = 0xA0000000;
constexpr size_t   KERNEL_CTRL_SIZE = 65536;
constexpr uint64_t PHY_MEM_BASE     = 0x40000000; 

// Độ lệch địa chỉ từ góc nhìn của FPGA Master (Vivado)
constexpr uint64_t VIVADO_MASTER_OFFSET = 0x400000000ULL; 
// ====================================================

struct MemBuffer {
    uint64_t phy_addr; 
    void* virt_addr; 
    size_t   size;
    bool     valid;
};

static int g_mem_fd = -1;
static void* g_ctrl_base_virt = nullptr;
static std::vector<MemBuffer> g_buffers;
static uint64_t g_current_phy_offset = 0;
static std::mutex g_mutex;
static bool g_ready = false;

// Đổi về int để map với 1 khối dữ liệu Q8_0 duy nhất
static std::unordered_map<std::string, int> g_tensor_map; 

static int g_bo_A_idx = -1;
static int g_bo_C_idx = -1;

// --- HÀM COPY AN TOÀN CHO BAREMETAL ARM (Chống Bus Error) ---
static void safe_io_memcpy(void* dst, const void* src, size_t n) {
    volatile uint32_t* d32 = (volatile uint32_t*)dst;
    const uint32_t* s32 = (const uint32_t*)src;
    size_t n32 = n / 4;
    for (size_t i = 0; i < n32; i++) {
        d32[i] = s32[i];
    }
    volatile uint8_t* d8 = (volatile uint8_t*)(d32 + n32);
    const uint8_t* s8 = (const uint8_t*)(s32 + n32);
    size_t n8 = n % 4;
    for (size_t i = 0; i < n8; i++) {
        d8[i] = s8[i];
    }
}

static void reg_write(int offset, uint32_t value) {
    if (g_ctrl_base_virt) {
        volatile uint32_t* ptr = (volatile uint32_t*)((char*)g_ctrl_base_virt + offset);
        *ptr = value;
    }
}

static uint32_t reg_read(int offset) {
    if (g_ctrl_base_virt) {
        volatile uint32_t* ptr = (volatile uint32_t*)((char*)g_ctrl_base_virt + offset);
        return *ptr;
    }
    return 0;
}

bool fpga_host_init(const std::string &xclbin_path, const std::string &kernel_name, std::string &err) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    (void)xclbin_path; (void)kernel_name;
    
    if ((g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
        err = "Cannot open /dev/mem"; return false;
    }
    void* map_base = mmap(0, KERNEL_CTRL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, g_mem_fd, KERNEL_CTRL_BASE);
    if (map_base == MAP_FAILED) {
        close(g_mem_fd); return false;
    }
    g_ctrl_base_virt = map_base;
    g_buffers.clear(); g_tensor_map.clear(); g_current_phy_offset = 0;
    
    std::cout << "[FPGA] Init: Control Reg (0x00) = 0x" << std::hex << reg_read(0x00) << std::dec << std::endl;
    g_ready = true; return true;
#else
    return false;
#endif
}

void fpga_host_shutdown() {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    for (auto &b : g_buffers) { if (b.valid && b.virt_addr) munmap(b.virt_addr, b.size); }
    g_buffers.clear();
    if (g_ctrl_base_virt) munmap(g_ctrl_base_virt, KERNEL_CTRL_SIZE);
    if (g_mem_fd != -1) { close(g_mem_fd); g_mem_fd = -1; }
    g_ready = false;
#endif
}

bool fpga_ready() { return g_ready; }

int fpga_alloc_bo(size_t bytes, int bank) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    (void)bank;
    if (!g_ready) return -1;
    uint64_t phy_addr = PHY_MEM_BASE + g_current_phy_offset;
    void* virt_addr = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g_mem_fd, phy_addr);
    if (virt_addr == MAP_FAILED) return -1;

    MemBuffer buf = {phy_addr, virt_addr, bytes, true};
    g_buffers.push_back(buf);
    g_current_phy_offset += bytes;
    if (g_current_phy_offset % 4096 != 0) g_current_phy_offset += (4096 - (g_current_phy_offset % 4096));
    return (int)g_buffers.size() - 1;
#else
    return -1;
#endif
}

bool fpga_bo_write(int idx, const void * src, size_t nbytes) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (idx < 0 || idx >= (int)g_buffers.size()) return false;
    MemBuffer &buf = g_buffers[idx];
    safe_io_memcpy(buf.virt_addr, src, (nbytes > buf.size) ? buf.size : nbytes);
    return true;
#else
    return false;
#endif
}

bool fpga_bo_read(int idx, void * dst, size_t nbytes) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (idx < 0 || idx >= (int)g_buffers.size()) return false;
    MemBuffer &buf = g_buffers[idx];
    safe_io_memcpy(dst, buf.virt_addr, (nbytes > buf.size) ? buf.size : nbytes);
    return true;
#else
    return false;
#endif
}

// [SỬA LỖI CHÍ MẠNG] - Khớp offset 100% với Vitis HLS Kernel (A, B, C)
bool fpga_run_matmul(int bo_A, int bo_B, int bo_C, int M, int K, int N) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (!g_ready) return false;
    
    uint64_t addr_A = g_buffers[bo_A].phy_addr + VIVADO_MASTER_OFFSET;
    uint64_t addr_B = g_buffers[bo_B].phy_addr + VIVADO_MASTER_OFFSET;
    uint64_t addr_C = g_buffers[bo_C].phy_addr + VIVADO_MASTER_OFFSET;

    // Truyền thẳng con trỏ
    reg_write(0x10, (uint32_t)(addr_A & 0xFFFFFFFF)); 
    reg_write(0x14, (uint32_t)(addr_A >> 32));
    
    reg_write(0x1c, (uint32_t)(addr_B & 0xFFFFFFFF)); 
    reg_write(0x20, (uint32_t)(addr_B >> 32));
    
    reg_write(0x28, (uint32_t)(addr_C & 0xFFFFFFFF)); 
    reg_write(0x2c, (uint32_t)(addr_C >> 32));
    
    // Đẩy kích thước vào đúng thanh ghi
    reg_write(0x34, (uint32_t)M); 
    reg_write(0x3c, (uint32_t)K); 
    reg_write(0x44, (uint32_t)N);

    // Kích hoạt kernel
    reg_write(0x00, 1);

    // Chờ kernel hoàn thành
    int timeout = 10000000;
    while (timeout-- > 0) {
        if ((reg_read(0x00) & 0x6)) return true; // Trạng thái Done hoặc Idle
    }
    printf("[FPGA-ERROR] Timeout Kernel!\n");
    return false;
#else
    return false;
#endif
}

void fpga_register_tensor_bo(const std::string &name, int bo_idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_tensor_map[name] = bo_idx;
}

int fpga_get_bo_idx_for_name(const std::string &name) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_tensor_map.find(name);
    if (it != g_tensor_map.end()) return it->second;
    return -1;
}

bool fpga_create_global_buffers(size_t n_ctx, size_t n_ff, std::string &err) {
    (void)n_ctx; (void)n_ff; (void)err; return true; 
}

int fpga_get_global_bo_A_idx() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_bo_A_idx < 0) {
        g_bo_A_idx = fpga_alloc_bo(64 * 1024 * 1024); 
    }
    return g_bo_A_idx;
}

int fpga_get_global_bo_C_idx() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_bo_C_idx < 0) {
        g_bo_C_idx = fpga_alloc_bo(64 * 1024 * 1024);
    }
    return g_bo_C_idx;
}

void* fpga_get_virt_addr(int idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (idx < 0 || idx >= (int)g_buffers.size()) return nullptr;
    return g_buffers[idx].virt_addr;
}

// =========================================================================
// HÀM XỬ LÝ CHÍNH CỰC KỲ ĐƠN GIẢN (ĐÃ BỎ REPACK) VÀ THREAD-SAFE
// =========================================================================
extern "C" int fpga_try_matmul(const struct ggml_tensor * weight, const struct ggml_tensor * activ, struct ggml_tensor * dst, int ith) {
    if (!fpga_ready()) return 0;

    if (weight->type != GGML_TYPE_Q8_0 || activ->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return 0;
    
    const int K = weight->ne[0];
    const int N = weight->ne[1];
    const int M = activ->ne[1]; 
    if (activ->ne[0] != K || dst->ne[0] != N || dst->ne[1] != M) return 0;

    // Chặn luồng 1, 2, 3 tránh phá RAM và ghi đè thanh ghi
    if (ith != 0) { return 1; }

    const std::string s_name = weight->name;
    int bo_B_idx = fpga_get_bo_idx_for_name(s_name);

    // BƯỚC A: Nạp Trọng số NGUYÊN BẢN (Không Repack)
    if (bo_B_idx < 0 && weight->data != nullptr) {
        size_t bytes_B = ggml_nbytes(weight); // Lấy thẳng kích thước
        bo_B_idx = fpga_alloc_bo(bytes_B);
        
        if (bo_B_idx >= 0) {
            fpga_bo_write(bo_B_idx, weight->data, bytes_B); // Copy thẳng 1 cục
            fpga_register_tensor_bo(s_name, bo_B_idx);
            // printf("[FPGA] Offloaded Weight: %s\n", s_name.c_str());
        } else {
            return 0; // Hết RAM
        }
    }

    // BƯỚC B: Chạy nhân ma trận
    if (bo_B_idx >= 0) {
        int bo_A_idx = fpga_get_global_bo_A_idx();
        int bo_C_idx = fpga_get_global_bo_C_idx();

        if (bo_A_idx >= 0 && bo_C_idx >= 0) {
            size_t bytes_A = ggml_nbytes(activ);
            if (fpga_bo_write(bo_A_idx, activ->data, bytes_A)) {
                
                printf("[FPGA] Kich hoat Kernel (M=%d, K=%d, N=%d)...\n", M, K, N);
                if (fpga_run_matmul(bo_A_idx, bo_B_idx, bo_C_idx, M, K, N)) {
                    
                    size_t bytes_C = ggml_nbytes(dst);
                    if (fpga_bo_read(bo_C_idx, dst->data, bytes_C)) {
                        return 1; // FPGA HOÀN THÀNH NHIỆM VỤ!
                    }
                }
            }
        }
    }
    return 0; 
}