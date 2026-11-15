#ifndef GGML_FPGA_ATTENTION_ABI_H
#define GGML_FPGA_ATTENTION_ABI_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fpga_attention_abi {

enum : uint32_t {
    ABI = 0x4000, CTRL = 0x4004, STATUS = 0x4008, ERROR = 0x400c,
    TAG = 0x4010, DONE_TAG = 0x4014, D = 0x4018, HQ = 0x401c,
    HKV = 0x4020, N = 0x4024, Q_ADDR = 0x4028, K_ADDR = 0x402c,
    V_ADDR = 0x4030, MASK_ADDR = 0x4034, F32_ADDR = 0x4038,
    Q8_ADDR = 0x403c, SCALE_ADDR = 0x4040, GENERATION = 0x4044,
    DONE_GENERATION = 0x4048, CAPS = 0x404c,
    SIGNATURE = 0x41540101, REQUIRED_CAPS = 1,
    START = 1, ACK = 2, RELEASE = 4,
    BUSY = 1, DONE = 2, FAILED = 4, RETAINED_VALID = 8,
    DDR_BASE = 0x70000000, DDR_BYTES = 0x10000000, MAX_N = 32768,
    Q_OFFSET = 0x00400000, MASK_OFFSET = 0x00410000,
    F32_OFFSET = 0x00420000, Q8_OFFSET = 0x00421000,
    SCALE_OFFSET = 0x00422000, WORKSPACE_END = 0x01000000,
    KV_POOL_OFFSET = 0x03000000,
};

inline bool reject(std::string & error, const char * reason) { error = reason; return false; }

struct layer_layout {
    uint32_t d = 0, hkv = 0, capacity = 0, k_offset = 0, v_offset = 0;
};

// Recompute all offsets before mutating any live mirror or touching DDR.
inline bool plan_pool(const std::vector<layer_layout> & old, size_t layer, uint32_t d, uint32_t hkv,
                      uint32_t n, size_t mapped_bytes, std::vector<layer_layout> & next, std::string & error) {
    error.clear();
    if (layer >= old.size() || d == 0 || d > 256 || d % 32 || hkv == 0 || hkv > 4 ||
        n == 0 || n > MAX_N || mapped_bytes > DDR_BYTES || mapped_bytes <= KV_POOL_OFFSET) {
        return reject(error, "invalid attention KV pool dimensions or mapped capacity");
    }
    next = old;
    layer_layout & changed = next[layer];
    if (changed.d != d || changed.hkv != hkv) { changed.capacity = 0; }
    changed.d = d; changed.hkv = hkv;
    uint32_t capacity = 1;
    while (capacity < n) { capacity *= 2; }
    if (changed.capacity < capacity) { changed.capacity = capacity; }
    uint64_t cursor = KV_POOL_OFFSET;
    for (layer_layout & entry : next) {
        if (entry.capacity == 0) { continue; }
        if (entry.d == 0 || entry.d > 256 || entry.d % 32 || entry.hkv == 0 || entry.hkv > 4 ||
            entry.capacity > MAX_N || (entry.capacity & (entry.capacity - 1))) {
            return reject(error, "invalid existing KV pool layout");
        }
        const uint64_t bytes = uint64_t(entry.capacity)*entry.hkv*entry.d*2;
        const uint64_t aligned = (bytes + 255) & ~uint64_t(255);
        if (cursor > mapped_bytes || 2*aligned > mapped_bytes - cursor) {
            return reject(error, "attention KV pool exceeds mapped reserved DDR");
        }
        entry.k_offset = uint32_t(cursor);
        entry.v_offset = uint32_t(cursor + aligned);
        cursor += 2*aligned;
    }
    return true;
}

struct command {
    uint32_t tag = 0, generation = 0, d = 0, hq = 0, hkv = 0, n = 0;
    uint32_t q = DDR_BASE + Q_OFFSET, k = 0, v = 0, mask = DDR_BASE + MASK_OFFSET;
    uint32_t f32 = DDR_BASE + F32_OFFSET, q8 = DDR_BASE + Q8_OFFSET, scales = DDR_BASE + SCALE_OFFSET;
};

inline bool physical_span(uint32_t address, uint64_t bytes, size_t mapped_bytes) {
    return mapped_bytes <= DDR_BYTES && address >= DDR_BASE && bytes != 0 &&
           uint64_t(address) - DDR_BASE <= mapped_bytes && bytes <= mapped_bytes - (uint64_t(address) - DDR_BASE);
}

inline bool validate(const command & c, size_t mapped_bytes, std::string & error) {
    if (c.tag == 0 || c.generation == 0 || c.d == 0 || c.d > 256 || c.d % 32 ||
        c.hq == 0 || c.hq > 4 || c.hkv == 0 || c.hkv > 4 || c.hq % c.hkv || c.n == 0 || c.n > MAX_N) {
        return reject(error, "invalid attention command identity or dimensions");
    }
    const uint32_t addresses[] = { c.q, c.k, c.v, c.mask, c.f32, c.q8, c.scales };
    const uint64_t lengths[] = { uint64_t(c.d)*c.hq*4, uint64_t(c.n)*c.hkv*c.d*2,
        uint64_t(c.n)*c.hkv*c.d*2, c.n, uint64_t(c.d)*c.hq*4, uint64_t(c.d)*c.hq, uint64_t(c.d)*c.hq/16 };
    if (c.q != DDR_BASE + Q_OFFSET || c.mask != DDR_BASE + MASK_OFFSET ||
        c.f32 != DDR_BASE + F32_OFFSET || c.q8 != DDR_BASE + Q8_OFFSET || c.scales != DDR_BASE + SCALE_OFFSET ||
        c.k < DDR_BASE + KV_POOL_OFFSET || c.v < DDR_BASE + KV_POOL_OFFSET || (c.k & 255) || (c.v & 255)) {
        return reject(error, "attention command address layout mismatch");
    }
    for (size_t i = 0; i < 7; ++i) {
        if (!physical_span(addresses[i], lengths[i], mapped_bytes)) {
            return reject(error, "attention command exceeds mapped DDR");
        }
        for (size_t j = 0; j < i; ++j) {
            if (uint64_t(addresses[i]) < uint64_t(addresses[j]) + lengths[j] &&
                uint64_t(addresses[j]) < uint64_t(addresses[i]) + lengths[i]) {
                return reject(error, "overlapping attention command buffers");
            }
        }
    }
    return true;
}

// Transport supplies register reads/writes, a device fence, monotonic us, and
// yield(). It exposes no attention-result read method. Tests use a register mock.
template<class Transport> bool admit(Transport & io, std::string & error) {
    if (io.read(ABI) != SIGNATURE || io.read(CAPS) != REQUIRED_CAPS) {
        return reject(error, "FPGA attention ABI/capability mismatch");
    }
    return true;
}

template<class Transport> bool release(Transport & io, std::string & error) {
    if (io.read(STATUS) & BUSY) { return reject(error, "cannot release busy attention engine"); }
    io.write(CTRL, ACK | RELEASE);
    io.fence();
    if (io.read(STATUS) & (BUSY | DONE | FAILED | RETAINED_VALID)) {
        return reject(error, "attention release/acknowledge did not retire state");
    }
    return true;
}

template<class Transport> bool retained(Transport & io, const command & c) {
    const uint32_t status = io.read(STATUS);
    return (status & (BUSY | DONE | FAILED | RETAINED_VALID)) == (DONE | RETAINED_VALID) &&
           io.read(ERROR) == 0 && io.read(DONE_TAG) == c.tag && io.read(DONE_GENERATION) == c.generation;
}

template<class Transport> bool execute(Transport & io, const command & c, size_t mapped_bytes,
                                      int64_t timeout_us, std::string & error) {
    error.clear();
    if (timeout_us <= 0 || !validate(c, mapped_bytes, error) || !admit(io, error)) { return false; }
    if (io.read(STATUS) & (BUSY | RETAINED_VALID)) {
        return reject(error, "attention launch would overwrite an owned command/result");
    }
    io.write(CTRL, ACK);
    io.fence();
    if (io.read(STATUS) & (BUSY | DONE | FAILED | RETAINED_VALID)) {
        return reject(error, "attention acknowledge did not clear terminal state");
    }
    const uint32_t regs[] = { TAG, D, HQ, HKV, N, Q_ADDR, K_ADDR, V_ADDR, MASK_ADDR, F32_ADDR, Q8_ADDR, SCALE_ADDR, GENERATION };
    const uint32_t vals[] = { c.tag, c.d, c.hq, c.hkv, c.n, c.q, c.k, c.v, c.mask, c.f32, c.q8, c.scales, c.generation };
    for (size_t i = 0; i < sizeof(regs)/sizeof(regs[0]); ++i) { io.write(regs[i], vals[i]); }
    io.fence();
    const int64_t started = io.now_us();
    io.write(CTRL, START);
    io.fence();
    for (;;) {
        const uint32_t status = io.read(STATUS);
        if (status & FAILED) { return reject(error, "attention hardware error"); }
        if (status & DONE) {
            if (!retained(io, c)) { return reject(error, "attention completion identity/finality mismatch"); }
            io.fence();
            return true;
        }
        if (io.now_us() - started >= timeout_us) { return reject(error, "attention hardware timeout"); }
        io.yield();
    }
}

} // namespace fpga_attention_abi
#endif
