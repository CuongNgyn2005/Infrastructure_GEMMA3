#ifndef GGML_FPGA_ATTENTION_CONTRACT_H
#define GGML_FPGA_ATTENTION_CONTRACT_H

// Pure host-side layout and lifetime rules. This file does not execute attention,
// access a device, quantize cache values, or certify a completed hardware job.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fpga_attention {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "attention staging requires IEEE binary32");

enum class scalar_type { f32, f16 };
enum class value_layout { d_n_h_s, n_d_h_s };

struct tensor_view {
    const void * data;
    scalar_type type;
    int64_t ne[4];
    size_t nb[4];
    // Accessible bytes beginning at data, after any GGML view offset.
    size_t storage_bytes;
};

inline bool fail(std::string & error, const char * reason) {
    error = reason;
    return false;
}

inline bool checked_add(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max()/a) {
        return false;
    }
    out = a*b;
    return true;
}

inline bool validate_view(const tensor_view & v, std::string & error) {
    const size_t element = v.type == scalar_type::f32 ? 4 : v.type == scalar_type::f16 ? 2 : 0;
    if (!element || !v.data) {
        return fail(error, "invalid scalar type or null tensor data");
    }
    size_t extent = element;
    int axes[4] = { 0, 1, 2, 3 };
    for (int i = 0; i < 4; ++i) {
        if (v.ne[i] <= 0 || uint64_t(v.ne[i]) > std::numeric_limits<size_t>::max()) {
            return fail(error, "invalid tensor dimension");
        }
        if (v.ne[i] > 1 && (v.nb[i] < element || v.nb[i] % element != 0)) {
            return fail(error, "invalid element stride");
        }
        size_t tail;
        if (!checked_mul(size_t(v.ne[i] - 1), v.nb[i], tail) || !checked_add(extent, tail, extent)) {
            return fail(error, "tensor extent overflow");
        }
    }
    if (extent > v.storage_bytes) {
        return fail(error, "tensor exceeds accessible storage");
    }
    // Standard dense, padded and permuted GGML views are supported. Reject
    // overlapping/broadcast axes instead of interpreting them as cache storage.
    std::sort(axes, axes + 4, [&v](int a, int b) { return v.nb[a] < v.nb[b]; });
    size_t occupied = element;
    for (int j = 0; j < 4; ++j) {
        const int i = axes[j];
        if (v.ne[i] > 1) {
            if (v.nb[i] < occupied) {
                return fail(error, "overlapping tensor strides");
            }
            occupied += size_t(v.ne[i] - 1)*v.nb[i]; // bounded by extent above
        }
    }
    return true;
}

inline size_t byte_offset(const tensor_view & v, size_t i0, size_t i1, size_t i2) {
    // Only used after validate_view and shape validation; stream index is zero.
    return i0*v.nb[0] + i1*v.nb[1] + i2*v.nb[2];
}

inline uint16_t read_f16_bits(const tensor_view & v, size_t offset) {
    uint16_t value;
    std::memcpy(&value, static_cast<const unsigned char *>(v.data) + offset, sizeof(value));
    return value;
}

inline float read_f32(const tensor_view & v, size_t offset) {
    float value;
    std::memcpy(&value, static_cast<const unsigned char *>(v.data) + offset, sizeof(value));
    return value;
}

inline bool finite_f16(uint16_t bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}

struct packed_attention {
    size_t d = 0, t = 0, n = 0, hq = 0, hkv = 0;
    // Q[(query*Hq + head)*D + channel], exactly the supplied F32 values.
    std::vector<float> q;
    // K/V[(slot*Hkv + head)*D + channel], original finite IEEE FP16 bits.
    std::vector<uint16_t> k;
    std::vector<uint16_t> v;
    // visible[query*N + slot]: one means zero mask, zero means -infinity.
    std::vector<uint8_t> visible;
    std::vector<size_t> kv_head;
};

inline bool pack(const tensor_view & q, const tensor_view & k, const tensor_view & v,
                 value_layout v_layout, const tensor_view * mask,
                 packed_attention & out, std::string & error) {
    out = packed_attention();
    error.clear();
    if (q.type != scalar_type::f32 || k.type != scalar_type::f16 || v.type != scalar_type::f16) {
        return fail(error, "requires F32 Q and F16 K/V; lossy cache conversion is unsupported");
    }
    if (!validate_view(q, error) || !validate_view(k, error) || !validate_view(v, error) ||
        (mask && !validate_view(*mask, error))) {
        return false;
    }
    if (q.ne[3] != 1 || k.ne[3] != 1 || v.ne[3] != 1 || (mask && mask->ne[3] != 1)) {
        return fail(error, "only one stream is supported");
    }
    if (v_layout != value_layout::d_n_h_s && v_layout != value_layout::n_d_h_s) {
        return fail(error, "unknown V layout");
    }
    const bool transposed = v_layout == value_layout::n_d_h_s;
    if (q.ne[0] > 256 || q.ne[0] % 32 != 0 || q.ne[2] > 4 || k.ne[2] > 4 ||
        q.ne[2] % k.ne[2] != 0 || k.ne[0] != q.ne[0] || v.ne[2] != k.ne[2] ||
        v.ne[transposed ? 1 : 0] != q.ne[0] || v.ne[transposed ? 0 : 1] != k.ne[1]) {
        return fail(error, "unsupported attention dimensions or GQA ratio");
    }
    if (mask && (mask->ne[0] != k.ne[1] || mask->ne[1] < q.ne[1] || mask->ne[2] != 1)) {
        return fail(error, "mask must have shape [N,T_padded,1,1]");
    }
    packed_attention p;
    p.d = size_t(q.ne[0]); p.t = size_t(q.ne[1]); p.n = size_t(k.ne[1]);
    p.hq = size_t(q.ne[2]); p.hkv = size_t(k.ne[2]);
    size_t nq, nkv, nm, bytes;
    if (!checked_mul(p.t, p.hq, nq) || !checked_mul(nq, p.d, nq) ||
        !checked_mul(p.n, p.hkv, nkv) || !checked_mul(nkv, p.d, nkv) ||
        !checked_mul(p.t, p.n, nm) || !checked_mul(nq, sizeof(float), bytes) ||
        !checked_mul(nkv, sizeof(uint16_t), bytes)) {
        return fail(error, "packed size overflow");
    }
    try {
        p.q.resize(nq); p.k.resize(nkv); p.v.resize(nkv);
        p.visible.resize(nm); p.kv_head.resize(p.hq);
        for (size_t h = 0; h < p.hq; ++h) {
            p.kv_head[h] = h/(p.hq/p.hkv);
            for (size_t t = 0; t < p.t; ++t) {
                for (size_t d = 0; d < p.d; ++d) {
                    const float value = read_f32(q, byte_offset(q, d, t, h));
                    if (!std::isfinite(value)) {
                        return fail(error, "nonfinite Q value");
                    }
                    p.q[(t*p.hq + h)*p.d + d] = value;
                }
            }
        }
        for (size_t n = 0; n < p.n; ++n) {
            for (size_t h = 0; h < p.hkv; ++h) {
                for (size_t d = 0; d < p.d; ++d) {
                    const size_t dst = (n*p.hkv + h)*p.d + d;
                    const uint16_t kb = read_f16_bits(k, byte_offset(k, d, n, h));
                    const uint16_t vb = read_f16_bits(v, byte_offset(v, transposed ? n : d, transposed ? d : n, h));
                    if (!finite_f16(kb) || !finite_f16(vb)) {
                        return fail(error, "nonfinite K/V value");
                    }
                    p.k[dst] = kb; p.v[dst] = vb;
                }
            }
        }
        for (size_t t = 0; t < p.t; ++t) {
            bool any_visible = false;
            for (size_t n = 0; n < p.n; ++n) {
                bool visible = true;
                if (mask) {
                    const size_t offset = byte_offset(*mask, n, t, 0);
                    if (mask->type == scalar_type::f16) {
                        const uint16_t bits = read_f16_bits(*mask, offset);
                        if ((bits & 0x7fffu) == 0) { visible = true; }
                        else if (bits == 0xfc00u) { visible = false; }
                        else { return fail(error, "mask contains unsupported bias or nonfinite value"); }
                    } else {
                        const float value = read_f32(*mask, offset);
                        if (value == 0.0f) { visible = true; }
                        else if (std::isinf(value) && value < 0.0f) { visible = false; }
                        else { return fail(error, "mask contains unsupported bias or nonfinite value"); }
                    }
                }
                p.visible[t*p.n + n] = visible ? 1 : 0;
                any_visible = any_visible || visible;
            }
            if (!any_visible) {
                return fail(error, "query has no visible cache slots");
            }
        }
    } catch (const std::bad_alloc &) {
        return fail(error, "packing allocation failed");
    } catch (const std::length_error &) {
        return fail(error, "packed vector exceeds size limit");
    }
    out = std::move(p);
    return true;
}

struct upload_span {
    bool values; // false: K buffer; true: V buffer
    size_t offset; // byte offset within the selected device buffer
    size_t bytes;
};

class kv_mirror;

class upload_plan {
public:
    const std::vector<uint16_t> & keys() const { return k_; }
    const std::vector<uint16_t> & values() const { return v_; }
    const std::vector<upload_span> & spans() const { return spans_; }
    uint64_t generation() const { return generation_; }
private:
    friend class kv_mirror;
    const kv_mirror * owner_ = nullptr;
    uint64_t generation_ = 0;
    size_t d_ = 0, hkv_ = 0;
    std::vector<uint16_t> k_, v_;
    std::vector<upload_span> spans_;
};

// One instance per allocated K/V device-buffer pair. External host mutex must
// serialize it with transfers and launches. Never infer cache positions here.
class kv_mirror {
public:
    kv_mirror() = default;
    kv_mirror(const kv_mirror &) = delete;
    kv_mirror & operator=(const kv_mirror &) = delete;

    bool synchronized() const { return valid_ && !pending_; }

    void invalidate() {
        valid_ = false;
        pending_ = false;
        k_.clear(); v_.clear();
        advance();
    }

    // capacity_bytes is the usable capacity of EACH device buffer. Plans own
    // their source snapshots, so later mutations of CPU cache/packed inputs
    // cannot change bytes in an in-flight transaction.
    bool prepare(const packed_attention & p, size_t capacity_bytes, size_t max_span_bytes,
                 upload_plan & plan, std::string & error) {
        error.clear();
        if (pending_) {
            return fail(error, "mirror already has a pending transaction");
        }
        plan = upload_plan();
        size_t count, bytes;
        if (p.d == 0 || p.d > 256 || p.d % 32 || p.hkv == 0 || p.hkv > 4 || p.n == 0 ||
            !checked_mul(p.n, p.hkv, count) || !checked_mul(count, p.d, count) ||
            !checked_mul(count, sizeof(uint16_t), bytes) ||
            p.k.size() != count || p.v.size() != count || bytes > capacity_bytes ||
            max_span_bytes < sizeof(uint16_t) || max_span_bytes % sizeof(uint16_t)) {
            return fail(error, "invalid packed KV shape or upload bounds");
        }
        for (size_t i = 0; i < count; ++i) {
            if (!finite_f16(p.k[i]) || !finite_f16(p.v[i])) {
                return fail(error, "nonfinite packed K/V value");
            }
        }
        if (exhausted_) {
            return fail(error, "mirror generation exhausted");
        }
        upload_plan next;
        try {
            next.k_ = p.k; next.v_ = p.v;
            next.d_ = p.d; next.hkv_ = p.hkv;
            const bool reuse = valid_ && d_ == p.d && hkv_ == p.hkv;
            append_spans(next.k_, k_, reuse, false, max_span_bytes/2, next.spans_);
            append_spans(next.v_, v_, reuse, true, max_span_bytes/2, next.spans_);
        } catch (const std::bad_alloc &) {
            return fail(error, "upload planning allocation failed");
        } catch (const std::length_error &) {
            return fail(error, "upload plan exceeds size limit");
        }
        if (!advance()) {
            return fail(error, "mirror generation exhausted");
        }
        next.owner_ = this; next.generation_ = generation_;
        plan = std::move(next);
        pending_ = true;
        return true;
    }

    // Call only after every span has completed successfully and required device
    // visibility/coherency operations have finished. Any failed or uncertain
    // transfer MUST use false, since partial writes invalidate the old mirror.
    bool complete(upload_plan & plan, bool all_transfers_succeeded, std::string & error) {
        error.clear();
        if (!pending_ || plan.owner_ != this || plan.generation_ != generation_) {
            return fail(error, "stale or foreign mirror transaction");
        }
        if (!all_transfers_succeeded) {
            invalidate();
            plan = upload_plan();
            return fail(error, "KV transfer failed; mirror invalidated");
        }
        k_.swap(plan.k_); v_.swap(plan.v_);
        d_ = plan.d_; hkv_ = plan.hkv_;
        valid_ = true; pending_ = false;
        plan = upload_plan();
        return true;
    }

private:
    static void append_spans(const std::vector<uint16_t> & now, const std::vector<uint16_t> & old,
                             bool reuse, bool values, size_t max_elements, std::vector<upload_span> & spans) {
        size_t i = 0;
        while (i < now.size()) {
            if (reuse && i < old.size() && now[i] == old[i]) {
                ++i;
                continue;
            }
            const size_t begin = i++;
            while (i < now.size() && i - begin < max_elements &&
                   (!reuse || i >= old.size() || now[i] != old[i])) {
                ++i;
            }
            spans.push_back(upload_span{ values, begin*2, (i - begin)*2 });
        }
    }

    bool advance() {
        if (generation_ == std::numeric_limits<uint64_t>::max()) {
            exhausted_ = true; valid_ = false; pending_ = false;
            return false;
        }
        ++generation_;
        return true;
    }

    uint64_t generation_ = 0;
    bool valid_ = false, pending_ = false, exhausted_ = false;
    size_t d_ = 0, hkv_ = 0;
    std::vector<uint16_t> k_, v_;
};

struct retained_q8_handle {
    uint64_t generation = 0, query_id = 0;
    size_t d = 0, heads = 0;
    size_t data_offset = 0, data_bytes = 0, scales_offset = 0, scales_bytes = 0;
};

// Lifetime metadata only, for one completed query's retained Q8 output and FP16
// block scales. It neither produces Q8 values nor changes their quantization.
class retained_q8_state {
public:
    retained_q8_state() = default;
    retained_q8_state(const retained_q8_state &) = delete;
    retained_q8_state & operator=(const retained_q8_state &) = delete;

    void invalidate() {
        valid_ = false; pending_ = false;
        if (generation_ == std::numeric_limits<uint64_t>::max()) { exhausted_ = true; }
        else { ++generation_; }
    }

    uint64_t begin(uint64_t query_id) {
        invalidate();
        query_id_ = query_id;
        pending_ = !exhausted_;
        return pending_ ? generation_ : 0;
    }

    bool publish(uint64_t generation, uint64_t query_id, size_t d, size_t heads,
                 size_t data_offset, size_t scales_offset, size_t storage_bytes,
                 bool producer_succeeded, std::string & error) {
        error.clear();
        if (!pending_ || generation != generation_ || query_id != query_id_) {
            return fail(error, "stale retained-Q8 producer identity");
        }
        size_t data_bytes, scale_bytes, data_end, scale_end;
        if (!producer_succeeded || d == 0 || d > 256 || d % 32 || heads == 0 || heads > 4 ||
            !checked_mul(d, heads, data_bytes) || !checked_mul(data_bytes/32, size_t(2), scale_bytes) ||
            !checked_add(data_offset, data_bytes, data_end) ||
            !checked_add(scales_offset, scale_bytes, scale_end) ||
            data_end > storage_bytes || scale_end > storage_bytes || scales_offset % 2 ||
            !(data_end <= scales_offset || scale_end <= data_offset)) {
            invalidate();
            return fail(error, "retained-Q8 publication failed or exceeds storage bounds");
        }
        handle_.generation = generation; handle_.query_id = query_id;
        handle_.d = d; handle_.heads = heads;
        handle_.data_offset = data_offset; handle_.data_bytes = data_bytes;
        handle_.scales_offset = scales_offset; handle_.scales_bytes = scale_bytes;
        valid_ = true; pending_ = false;
        return true;
    }

    bool lookup(uint64_t generation, uint64_t query_id, retained_q8_handle & out) const {
        out = retained_q8_handle();
        if (!valid_ || generation != generation_ || query_id != query_id_) { return false; }
        out = handle_;
        return true;
    }

private:
    uint64_t generation_ = 0, query_id_ = 0;
    bool valid_ = false, pending_ = false, exhausted_ = false;
    retained_q8_handle handle_;
};

} // namespace fpga_attention

#endif // GGML_FPGA_ATTENTION_CONTRACT_H
