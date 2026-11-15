#include "../ggml/src/ggml-cpu/fpga_attention_contract.h"

#include <cstdio>
#include <cstdlib>

using namespace fpga_attention;

static size_t checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); std::exit(1); \
} } while (0)

struct owned_tensor {
    tensor_view view;
    std::vector<unsigned char> storage;

    owned_tensor(scalar_type type, size_t a, size_t b, size_t c,
                 size_t s0, size_t s1, size_t s2) {
        const size_t element = type == scalar_type::f32 ? 4 : 2;
        const size_t bytes = element + (a - 1)*s0 + (b - 1)*s1 + (c - 1)*s2;
        storage.assign(bytes + 1, 0);
        // Deliberately unaligned base: the packer must read with memcpy.
        view.data = storage.data() + 1; view.type = type; view.storage_bytes = bytes;
        view.ne[0] = int64_t(a); view.ne[1] = int64_t(b); view.ne[2] = int64_t(c); view.ne[3] = 1;
        view.nb[0] = s0; view.nb[1] = s1; view.nb[2] = s2; view.nb[3] = bytes;
    }

    template<typename T> void set(size_t a, size_t b, size_t c, T value) {
        std::memcpy(storage.data() + 1 + a*view.nb[0] + b*view.nb[1] + c*view.nb[2], &value, sizeof(T));
    }
};

static uint16_t cache_bits(size_t d, size_t n, size_t h, bool values) {
    const uint16_t special[] = { 0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff };
    return special[(d + 3*n + 7*h + (values ? 5 : 0)) % 10];
}

struct fixture {
    size_t d, t, n, hq, hkv;
    owned_tensor q, k, v, vt, mask, mask16;

    fixture(size_t dim = 32, size_t tokens = 3, size_t slots = 7, size_t heads = 4, size_t kvheads = 2)
        : d(dim), t(tokens), n(slots), hq(heads), hkv(kvheads),
          q(scalar_type::f32, d, t, hq, 4, (d*4 + 8)*hq + 16, d*4 + 8),
          k(scalar_type::f16, d, n, hkv, 2, (d*2 + 4)*hkv + 6, d*2 + 4),
          v(scalar_type::f16, d, n, hkv, 2, d*2 + 4, (d*2 + 4)*n + 6),
          vt(scalar_type::f16, n, d, hkv, 2, n*2 + 4, (n*2 + 4)*d + 8),
          mask(scalar_type::f32, n, t + 1, 1, 4, n*4 + 8, (n*4 + 8)*(t + 1)),
          mask16(scalar_type::f16, n, t + 1, 1, 2, n*2 + 4, (n*2 + 4)*(t + 1)) {
        for (size_t h = 0; h < hq; ++h) {
            for (size_t i = 0; i < t; ++i) {
                for (size_t c = 0; c < d; ++c) { q.set(c, i, h, float(1000*i + 100*h + c)/16.0f); }
            }
        }
        for (size_t h = 0; h < hkv; ++h) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t c = 0; c < d; ++c) {
                    k.set(c, i, h, cache_bits(c, i, h, false));
                    v.set(c, i, h, cache_bits(c, i, h, true));
                    vt.set(i, c, h, cache_bits(c, i, h, true));
                }
            }
        }
        for (size_t i = 0; i <= t; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const bool show = visible(i, j);
                const float m = i == t ? std::numeric_limits<float>::quiet_NaN() :
                                show ? -0.0f : -std::numeric_limits<float>::infinity();
                mask.set(j, i, 0, m);
                mask16.set(j, i, 0, uint16_t(i == t ? 0x7e00 : show ? 0x8000 : 0xfc00));
            }
        }
    }

    bool visible(size_t i, size_t j) const {
        // Query-dependent causal frontier + sliding window + a hole at slot 1.
        const size_t frontier = std::min(n - 1, i + 2);
        return j <= frontier && j + 2 >= frontier && j != 1;
    }
};

static packed_attention pack_fixture(fixture & f, bool transpose = false, bool half_mask = false) {
    packed_attention p;
    std::string error;
    CHECK(pack(f.q.view, f.k.view, transpose ? f.vt.view : f.v.view,
               transpose ? value_layout::n_d_h_s : value_layout::d_n_h_s,
               half_mask ? &f.mask16.view : &f.mask.view, p, error));
    CHECK(error.empty());
    return p;
}

static void test_layouts() {
    for (size_t d : { size_t(32), size_t(256) }) {
        for (size_t hkv : { size_t(1), size_t(2), size_t(4) }) {
            fixture f(d, 3, 7, 4, hkv);
            const packed_attention p = pack_fixture(f);
            const packed_attention transposed = pack_fixture(f, true, true);
            CHECK(p.q == transposed.q); CHECK(p.k == transposed.k);
            CHECK(p.v == transposed.v); CHECK(p.visible == transposed.visible);
            for (size_t h = 0; h < f.hq; ++h) {
                CHECK(p.kv_head[h] == h/(f.hq/f.hkv));
                for (size_t t = 0; t < f.t; ++t) {
                    for (size_t c = 0; c < f.d; ++c) {
                        CHECK(p.q[(t*f.hq + h)*f.d + c] == float(1000*t + 100*h + c)/16.0f);
                    }
                }
            }
            for (size_t n = 0; n < f.n; ++n) {
                for (size_t h = 0; h < f.hkv; ++h) {
                    for (size_t c = 0; c < f.d; ++c) {
                        CHECK(p.k[(n*f.hkv + h)*f.d + c] == cache_bits(c, n, h, false));
                        CHECK(p.v[(n*f.hkv + h)*f.d + c] == cache_bits(c, n, h, true));
                    }
                }
                for (size_t t = 0; t < f.t; ++t) { CHECK(p.visible[t*f.n + n] == f.visible(t, n)); }
            }
            packed_attention unmasked;
            std::string error;
            CHECK(pack(f.q.view, f.k.view, f.v.view, value_layout::d_n_h_s, nullptr, unmasked, error));
            for (uint8_t m : unmasked.visible) { CHECK(m == 1); }
        }
    }
}

static void test_rejections() {
    fixture f;
    packed_attention out;
    std::string error;
    auto rejected = [&](tensor_view q, tensor_view k, tensor_view v, const tensor_view * mask) {
        out = pack_fixture(f);
        CHECK(!pack(q, k, v, value_layout::d_n_h_s, mask, out, error));
        CHECK(!error.empty()); CHECK(out.q.empty()); CHECK(out.k.empty());
    };
    tensor_view bad = f.q.view; bad.ne[0] = 31;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.ne[0] = 288;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.ne[2] = 3;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.ne[2] = 5;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.ne[3] = 2; bad.nb[3] = bad.storage_bytes; bad.storage_bytes *= 2;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; --bad.storage_bytes;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.nb[1] = std::numeric_limits<size_t>::max() - 3;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.nb[1] = 4;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.nb[0] = 3;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.data = nullptr;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.q.view; bad.ne[0] = -1;
    rejected(bad, f.k.view, f.v.view, &f.mask.view);
    bad = f.k.view; bad.type = scalar_type::f32;
    rejected(f.q.view, bad, f.v.view, &f.mask.view);
    bad = f.k.view; bad.ne[2] = 0;
    rejected(f.q.view, bad, f.v.view, &f.mask.view);
    bad = f.mask.view; bad.ne[1] = 1;
    rejected(f.q.view, f.k.view, f.v.view, &bad);
    const float saved_q = read_f32(f.q.view, 0);
    for (float x : { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        f.q.set(0, 0, 0, x);
        CHECK(!pack(f.q.view, f.k.view, f.v.view, value_layout::d_n_h_s, &f.mask.view, out, error));
    }
    f.q.set(0, 0, 0, saved_q);
    for (uint16_t x : { uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e00) }) {
        f.k.set(0, 0, 0, x);
        CHECK(!pack(f.q.view, f.k.view, f.v.view, value_layout::d_n_h_s, &f.mask.view, out, error));
    }
    f.k.set(0, 0, 0, cache_bits(0, 0, 0, false));
    for (float x : { 0.25f, -0.25f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        f.mask.set(0, 0, 0, x);
        CHECK(!pack(f.q.view, f.k.view, f.v.view, value_layout::d_n_h_s, &f.mask.view, out, error));
    }
    for (size_t n = 0; n < f.n; ++n) { f.mask.set(n, 0, 0, -std::numeric_limits<float>::infinity()); }
    CHECK(!pack(f.q.view, f.k.view, f.v.view, value_layout::d_n_h_s, &f.mask.view, out, error));
    CHECK(error == "query has no visible cache slots");
}

static size_t transferred(const upload_plan & p) {
    size_t bytes = 0;
    for (const upload_span & s : p.spans()) { bytes += s.bytes; }
    return bytes;
}

static void apply_plan(const upload_plan & p, std::vector<uint16_t> & k, std::vector<uint16_t> & v) {
    for (const upload_span & s : p.spans()) {
        const std::vector<uint16_t> & source = s.values ? p.values() : p.keys();
        std::vector<uint16_t> & device = s.values ? v : k;
        CHECK(s.bytes > 0 && s.bytes <= 16 && s.bytes % 2 == 0 && s.offset % 2 == 0);
        CHECK(s.offset + s.bytes <= source.size()*2);
        CHECK(s.offset + s.bytes <= device.size()*2);
        std::memcpy(device.data() + s.offset/2, source.data() + s.offset/2, s.bytes);
    }
}

static void test_mirror() {
    fixture f;
    packed_attention p = pack_fixture(f);
    const size_t capacity = p.k.size()*2 + 512;
    std::vector<uint16_t> device_k(capacity/2), device_v(capacity/2);
    kv_mirror mirror, other_layer;
    upload_plan plan, stale, other;
    std::string error;
    CHECK(!mirror.synchronized());
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(transferred(plan) == (p.k.size() + p.v.size())*2);
    CHECK(!mirror.synchronized());
    stale = plan;
    CHECK(!mirror.prepare(p, capacity, 16, other, error));
    CHECK(!other_layer.complete(plan, true, error));
    apply_plan(plan, device_k, device_v);
    CHECK(mirror.complete(plan, true, error));
    CHECK(mirror.synchronized());
    CHECK(!mirror.complete(stale, true, error));
    CHECK(std::equal(p.k.begin(), p.k.end(), device_k.begin()));
    CHECK(std::equal(p.v.begin(), p.v.end(), device_v.begin()));
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(plan.spans().empty());
    CHECK(mirror.complete(plan, true, error));

    // Slot replacement and K-shift mutation: compare actual bytes, no append assumption.
    p.k[1] ^= 0x8000u; p.k[2] ^= 0x8000u; p.v[p.v.size() - 1] ^= 0x8000u;
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(plan.spans().size() == 2); CHECK(transferred(plan) == 6);
    const uint16_t snapshot = plan.keys()[1];
    p.k[1] ^= 0x8000u;
    CHECK(plan.keys()[1] == snapshot);
    apply_plan(plan, device_k, device_v);
    CHECK(mirror.complete(plan, true, error));
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(transferred(plan) == 2);
    // A partial DMA write followed by failure poisons the complete mirror.
    apply_plan(plan, device_k, device_v);
    stale = plan;
    CHECK(!mirror.complete(plan, false, error));
    CHECK(!mirror.synchronized());
    CHECK(!mirror.complete(stale, true, error));
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(transferred(plan) == (p.k.size() + p.v.size())*2);
    CHECK(mirror.complete(plan, true, error));
    CHECK(other_layer.prepare(p, capacity, 16, other, error));
    CHECK(transferred(other) == (p.k.size() + p.v.size())*2);
    CHECK(other_layer.complete(other, true, error));

    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    mirror.invalidate();
    CHECK(!mirror.complete(plan, true, error));
    CHECK(!mirror.synchronized());
    CHECK(!mirror.prepare(p, p.k.size()*2 - 1, 16, plan, error));
    CHECK(!mirror.prepare(p, capacity, 1, plan, error));
    CHECK(!mirror.prepare(p, capacity, 3, plan, error));
    CHECK(plan.spans().empty());
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(mirror.complete(plan, true, error));

    // Growing and shrinking a view is independent of logical token positions.
    p.n += 1; p.k.resize(p.n*p.hkv*p.d, 0x3c00); p.v.resize(p.n*p.hkv*p.d, 0xbc00);
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(transferred(plan) == 4*p.hkv*p.d);
    CHECK(mirror.complete(plan, true, error));
    p.n -= 2; p.k.resize(p.n*p.hkv*p.d); p.v.resize(p.n*p.hkv*p.d);
    CHECK(mirror.prepare(p, capacity, 16, plan, error)); CHECK(plan.spans().empty());
    CHECK(mirror.complete(plan, true, error));
    p.n += 1; p.k.resize(p.n*p.hkv*p.d, 0x3c00); p.v.resize(p.n*p.hkv*p.d, 0xbc00);
    CHECK(mirror.prepare(p, capacity, 16, plan, error));
    CHECK(transferred(plan) == 4*p.hkv*p.d);
    CHECK(mirror.complete(plan, true, error));
    CHECK(other_layer.synchronized());
}

static void test_retained_q8() {
    retained_q8_state state;
    retained_q8_handle handle;
    std::string error;
    const uint64_t first = state.begin(123);
    CHECK(!state.lookup(first, 123, handle));
    CHECK(state.publish(first, 123, 256, 4, 0, 1024, 1088, true, error));
    CHECK(state.lookup(first, 123, handle));
    CHECK(handle.data_bytes == 1024 && handle.scales_bytes == 64);
    CHECK(!state.lookup(first, 124, handle));
    CHECK(!state.publish(first, 123, 256, 4, 0, 1024, 1088, true, error));
    const uint64_t second = state.begin(123); // query identity alone cannot permit reuse
    CHECK(second != first);
    CHECK(!state.lookup(first, 123, handle));
    CHECK(!state.publish(first, 123, 256, 4, 0, 1024, 1088, true, error));
    CHECK(!state.publish(second, 123, 256, 4, 0, 1024, 1088, false, error));
    CHECK(!state.lookup(second, 123, handle));
    uint64_t ticket = state.begin(124);
    CHECK(!state.publish(ticket, 124, 32, 1, 0, 30, 64, true, error)); // overlapping scales
    ticket = state.begin(125);
    CHECK(!state.publish(ticket, 125, 32, 1, std::numeric_limits<size_t>::max() - 15, 0,
                         std::numeric_limits<size_t>::max(), true, error));
    ticket = state.begin(126);
    CHECK(!state.publish(ticket, 126, 32, 1, 0, 33, 64, true, error)); // scale alignment
    ticket = state.begin(127);
    CHECK(state.publish(ticket, 127, 32, 1, 0, 32, 34, true, error));
    state.invalidate();
    CHECK(!state.lookup(ticket, 127, handle));
}

int main() {
    test_layouts();
    test_rejections();
    test_mirror();
    test_retained_q8();
    std::printf("PASS FPGA attention host contract: %zu checks (offline layout/lifetime only)\n", checks);
    return 0;
}
