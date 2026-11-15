// Offline arithmetic evidence only. Link a GGML build configured USE_FPGA=OFF.
// Compile this file with -ffp-contract=off; do not enable fast-math.
#include "../ggml/src/ggml-cpu/fpga_attention_contract.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

using fpga_attention::packed_attention;

static void require(bool condition, const char * reason) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", reason);
        std::exit(1);
    }
}

// Explicit binary32 rounding after each primitive, even on extended-precision
// targets. There is no FMA in this scalar arithmetic oracle.
static float add32(float a, float b) { volatile float r = a + b; return r; }
static float sub32(float a, float b) { volatile float r = a - b; return r; }
static float mul32(float a, float b) { volatile float r = a * b; return r; }
static float div32(float a, float b) { volatile float r = a / b; return r; }
static float exp32(float a) { volatile float r = std::exp(a); return r; }

static long double half_reference(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned fraction = bits & 1023;
    require(exponent != 31, "nonfinite FP16 reference input");
    const long double magnitude = exponent ? std::ldexp(static_cast<long double>(1024 + fraction), int(exponent) - 25) :
                                             std::ldexp(static_cast<long double>(fraction), -24);
    return bits & 0x8000 ? -magnitude : magnitude;
}

static uint32_t random_word(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

static float random_value(uint32_t & state, float range) {
    return mul32(float(int32_t(random_word(state) % 8193) - 4096)/4096.0f, range);
}

static packed_attention fixture(size_t d, size_t n, size_t t, size_t hkv,
                                unsigned mask_mode, unsigned pattern, uint32_t seed) {
    packed_attention p;
    p.d = d; p.n = n; p.t = t; p.hq = 4; p.hkv = hkv;
    p.q.resize(d*t*4); p.k.resize(d*n*hkv); p.v.resize(d*n*hkv);
    p.visible.resize(n*t); p.kv_head.resize(4);
    for (size_t h = 0; h < 4; ++h) { p.kv_head[h] = h/(4/hkv); }
    const float q_scale = div32(1.0f, std::sqrt(float(d)));
    for (size_t i = 0; i < p.q.size(); ++i) {
        // Pre-scaled Q is authoritative. All graph calls below use scale=1.
        p.q[i] = pattern == 2 ? ((i % d) % 2 ? -4.0f : 4.0f) :
                 pattern == 1 ? mul32(random_value(seed, 0.015625f), q_scale) :
                                mul32(random_value(seed, 4.0f), q_scale);
    }
    for (size_t i = 0; i < p.k.size(); ++i) {
        float k = random_value(seed, 4.0f);
        if (pattern == 1) { k = add32(0.125f, mul32(k, 0.0009765625f)); }
        if (pattern == 2) {
            const size_t slot = i/(d*hkv);
            k = ((i % d) % 2 ? -1.0f : 1.0f)*(slot % 3 == 0 ? 4.0f : slot % 3 == 1 ? -4.0f : 3.998046875f);
        }
        p.k[i] = ggml_fp32_to_fp16(k);
        p.v[i] = ggml_fp32_to_fp16(random_value(seed, 8.0f));
    }
    for (size_t q = 0; q < t; ++q) {
        const size_t frontier = n - t + q;
        for (size_t slot = 0; slot < n; ++slot) {
            bool visible = slot <= frontier;
            if (mask_mode >= 1) { visible = visible && (slot % 7 != 2); }
            if (mask_mode == 2) { visible = visible && slot + 17 > frontier; }
            p.visible[q*n + slot] = visible ? 1 : 0;
        }
    }
    return p;
}

static std::vector<long double> reference_attention(const packed_attention & p) {
    std::vector<long double> out(p.t*p.hq*p.d, 0);
    std::vector<long double> score(p.n), weight(p.n);
    for (size_t q = 0; q < p.t; ++q) {
        for (size_t h = 0; h < p.hq; ++h) {
            const size_t kh = h/(p.hq/p.hkv);
            long double maximum = -std::numeric_limits<long double>::infinity();
            for (size_t n = 0; n < p.n; ++n) {
                if (!p.visible[q*p.n + n]) { continue; }
                long double dot = 0;
                for (size_t d = 0; d < p.d; ++d) {
                    dot += static_cast<long double>(p.q[(q*p.hq + h)*p.d + d]) *
                           half_reference(p.k[(n*p.hkv + kh)*p.d + d]);
                }
                score[n] = dot;
                maximum = std::max(maximum, dot);
            }
            require(std::isfinite(maximum), "all-masked reference fixture");
            long double denominator = 0;
            for (size_t n = 0; n < p.n; ++n) {
                weight[n] = p.visible[q*p.n + n] ? std::exp(score[n] - maximum) : 0;
                denominator += weight[n];
            }
            for (size_t d = 0; d < p.d; ++d) {
                long double numerator = 0;
                for (size_t n = 0; n < p.n; ++n) {
                    numerator += weight[n]*half_reference(p.v[(n*p.hkv + kh)*p.d + d]);
                }
                out[(q*p.hq + h)*p.d + d] = numerator/denominator;
            }
        }
    }
    return out;
}

static std::vector<float> fp32_attention(const packed_attention & p) {
    std::vector<float> out(p.t*p.hq*p.d, 0);
    std::vector<float> score(p.n), weight(p.n);
    for (size_t q = 0; q < p.t; ++q) {
        for (size_t h = 0; h < p.hq; ++h) {
            const size_t kh = h/(p.hq/p.hkv);
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t n = 0; n < p.n; ++n) {
                if (!p.visible[q*p.n + n]) { continue; }
                float dot = 0;
                for (size_t d = 0; d < p.d; ++d) {
                    dot = add32(dot, mul32(p.q[(q*p.hq + h)*p.d + d],
                                          ggml_fp16_to_fp32(p.k[(n*p.hkv + kh)*p.d + d])));
                }
                score[n] = dot;
                maximum = std::max(maximum, dot);
            }
            float denominator = 0;
            for (size_t n = 0; n < p.n; ++n) {
                weight[n] = p.visible[q*p.n + n] ? exp32(sub32(score[n], maximum)) : 0;
                denominator = add32(denominator, weight[n]);
            }
            for (size_t d = 0; d < p.d; ++d) {
                float numerator = 0;
                for (size_t n = 0; n < p.n; ++n) {
                    numerator = add32(numerator, mul32(weight[n], ggml_fp16_to_fp32(p.v[(n*p.hkv + kh)*p.d + d])));
                }
                out[(q*p.hq + h)*p.d + d] = div32(numerator, denominator);
            }
        }
    }
    return out;
}

// Operation-ordered oracle for the one-key online recurrence. Chunk boundaries
// only group fetches: the running maximum, denominator and numerator survive.
// This is CPU reference evidence, not execution of the RTL arithmetic helper.
static bool online_attention(const packed_attention & p, size_t chunk, bool reverse,
                             std::vector<float> & out) {
    require(chunk != 0, "nonzero online chunk");
    out.assign(p.t*p.hq*p.d, 0);
    for (size_t q = 0; q < p.t; ++q) {
        for (size_t h = 0; h < p.hq; ++h) {
            const size_t kh = h/(p.hq/p.hkv);
            bool seen = false;
            float maximum = 0, denominator = 0;
            std::vector<float> numerator(p.d);
            for (size_t base = 0; base < p.n; base += chunk) {
                for (size_t i = base; i < std::min(p.n, base + chunk); ++i) {
                    const size_t n = reverse ? p.n - 1 - i : i;
                    if (!p.visible[q*p.n + n]) { continue; }
                    float score = 0;
                    for (size_t d = 0; d < p.d; ++d) {
                        score = add32(score, mul32(p.q[(q*p.hq + h)*p.d + d],
                                      ggml_fp16_to_fp32(p.k[(n*p.hkv + kh)*p.d + d])));
                        if (!std::isfinite(score)) { out.clear(); return false; }
                    }
                    const bool larger = seen && score > maximum;
                    const float delta = !seen ? 0 : larger ? sub32(maximum, score) : sub32(score, maximum);
                    if (!std::isfinite(delta)) { out.clear(); return false; }
                    const float weight = exp32(delta);
                    denominator = !seen ? 1 : larger ? add32(mul32(denominator, weight), 1) : add32(denominator, weight);
                    for (size_t d = 0; d < p.d; ++d) {
                        const float value = ggml_fp16_to_fp32(p.v[(n*p.hkv + kh)*p.d + d]);
                        numerator[d] = !seen ? value : larger ? add32(mul32(numerator[d], weight), value) :
                                                               add32(numerator[d], mul32(value, weight));
                        if (!std::isfinite(numerator[d])) { out.clear(); return false; }
                    }
                    if (!seen || larger) { maximum = score; }
                    seen = true;
                }
            }
            if (!seen) { out.clear(); return false; }
            for (size_t d = 0; d < p.d; ++d) {
                out[(q*p.hq + h)*p.d + d] = div32(numerator[d], denominator);
            }
        }
    }
    return true;
}

template<typename T> static void store(ggml_tensor * tensor, size_t d, size_t n, size_t h, T value) {
    std::memcpy(static_cast<unsigned char *>(tensor->data) + d*tensor->nb[0] + n*tensor->nb[1] + h*tensor->nb[2],
                &value, sizeof(value));
}

static fpga_attention::tensor_view view_of(ggml_tensor * tensor) {
    fpga_attention::tensor_view v;
    v.data = tensor->data;
    v.type = tensor->type == GGML_TYPE_F32 ? fpga_attention::scalar_type::f32 : fpga_attention::scalar_type::f16;
    v.storage_bytes = ggml_nbytes(tensor);
    for (int i = 0; i < 4; ++i) { v.ne[i] = tensor->ne[i]; v.nb[i] = tensor->nb[i]; }
    return v;
}

struct graph_results { std::vector<float> flash, nonflash; };

static graph_results ggml_attention(const packed_attention & p) {
    ggml_init_params init = { 16*1024*1024 + p.k.size()*6 + p.n*GGML_KQ_MASK_PAD*6 + p.n*p.t*p.hq*8,
                             nullptr, false };
    ggml_context * ctx = ggml_init(init);
    require(ctx != nullptr, "GGML context allocation");
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, p.d, p.t, p.hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p.d, p.n, p.hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p.d, p.n, p.hkv, 1);
    const size_t padded_t = (p.t + GGML_KQ_MASK_PAD - 1)/GGML_KQ_MASK_PAD*GGML_KQ_MASK_PAD;
    ggml_tensor * mask32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, p.n, padded_t, 1, 1);
    ggml_tensor * mask16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p.n, padded_t, 1, 1);
    for (size_t h = 0; h < p.hq; ++h) {
        for (size_t t = 0; t < p.t; ++t) {
            for (size_t d = 0; d < p.d; ++d) { store(q, d, t, h, p.q[(t*p.hq + h)*p.d + d]); }
        }
    }
    for (size_t h = 0; h < p.hkv; ++h) {
        for (size_t n = 0; n < p.n; ++n) {
            for (size_t d = 0; d < p.d; ++d) {
                store(k, d, n, h, p.k[(n*p.hkv + h)*p.d + d]);
                store(v, d, n, h, p.v[(n*p.hkv + h)*p.d + d]);
            }
        }
    }
    for (size_t t = 0; t < padded_t; ++t) {
        for (size_t n = 0; n < p.n; ++n) {
            const bool show = t < p.t && p.visible[t*p.n + n];
            store(mask32, n, t, 0, show ? 0.0f : -std::numeric_limits<float>::infinity());
            store(mask16, n, t, 0, uint16_t(show ? 0 : 0xfc00));
        }
    }
    // Exercise the production packing helper using real GGML tensor metadata.
    const fpga_attention::tensor_view qv = view_of(q), kv = view_of(k), vv = view_of(v), mv = view_of(mask16);
    packed_attention packed;
    std::string error;
    require(fpga_attention::pack(qv, kv, vv, fpga_attention::value_layout::d_n_h_s, &mv, packed, error), error.c_str());
    require(packed.q == p.q && packed.k == p.k && packed.v == p.v && packed.visible == p.visible,
            "GGML tensor packing changed Q/K/V/mask data");

    ggml_tensor * flash = ggml_flash_attn_ext(ctx, q, k, v, mask16, 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, mask32, 1.0f, 0.0f);
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * pv = ggml_mul_mat(ctx, vt, weights);
    ggml_tensor * nonflash = ggml_cont(ctx, ggml_permute(ctx, pv, 0, 2, 1, 3));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, flash);
    ggml_build_forward_expand(graph, nonflash);
    require(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS, "actual GGML CPU graph execution");
    graph_results result;
    const size_t count = p.t*p.hq*p.d;
    require(ggml_nbytes(flash) == count*sizeof(float) && ggml_nbytes(nonflash) == count*sizeof(float), "GGML output layout");
    result.flash.assign(static_cast<float *>(flash->data), static_cast<float *>(flash->data) + count);
    result.nonflash.assign(static_cast<float *>(nonflash->data), static_cast<float *>(nonflash->data) + count);
    ggml_free(ctx);
    return result;
}

struct metrics { long double maxabs = 0, maxrel = 0, maxrel_nonzero = 0, nmse = 0; };

static metrics compare(const std::vector<long double> & ref, const std::vector<float> & actual, long double vscale) {
    require(ref.size() == actual.size(), "comparison shape");
    metrics m;
    long double squared_error = 0, squared_reference = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        require(std::isfinite(ref[i]) && std::isfinite(actual[i]), "nonfinite numerical result");
        const long double diff = std::fabs(ref[i] - actual[i]);
        m.maxabs = std::max(m.maxabs, diff);
        if (ref[i] != 0) { m.maxrel_nonzero = std::max(m.maxrel_nonzero, diff/std::fabs(ref[i])); }
        if (std::fabs(ref[i]) >= 0.01L*vscale) { m.maxrel = std::max(m.maxrel, diff/std::fabs(ref[i])); }
        squared_error += diff*diff;
        squared_reference += ref[i]*ref[i];
    }
    m.nmse = squared_reference ? squared_error/squared_reference : squared_error == 0 ? 0 :
             std::numeric_limits<long double>::infinity();
    return m;
}

static void accumulate(metrics & worst, const metrics & m) {
    worst.maxabs = std::max(worst.maxabs, m.maxabs);
    worst.maxrel = std::max(worst.maxrel, m.maxrel);
    worst.maxrel_nonzero = std::max(worst.maxrel_nonzero, m.maxrel_nonzero);
    worst.nmse = std::max(worst.nmse, m.nmse);
}

static void print_metrics(const char * label, const metrics & m) {
    std::printf("%s maxabs=%.9Lg maxrel_gated=%.9Lg maxrel_nonzero=%.9Lg NMSE=%.9Lg\n",
                label, m.maxabs, m.maxrel, m.maxrel_nonzero, m.nmse);
}

static bool test_attention() {
    metrics fp32_worst, online_worst, flash_worst, nonflash_worst, flash_fp32_worst, nonflash_fp32_worst;
    size_t cases = 0, failures = 0;
    for (size_t d : { size_t(32), size_t(256) }) {
        for (size_t hkv : { size_t(1), size_t(2), size_t(4) }) {
            for (size_t n : { size_t(31), size_t(32), size_t(33), size_t(127), size_t(128), size_t(129), size_t(257) }) {
                for (size_t t : { size_t(1), size_t(3) }) {
                    for (unsigned mask = 0; mask < 3; ++mask) {
                        for (unsigned identity = 0; identity < 4; ++identity) {
                            const unsigned layer = identity/2, sequence = identity % 2;
                            const unsigned pattern = unsigned((n + mask + layer + sequence) % 3);
                            const uint32_t seed = uint32_t(0x9e3779b9u ^ (d*97 + n*23 + hkv*131 + t*17 + mask*101 + sequence*4099 + layer*65537));
                            const packed_attention p = fixture(d, n, t, hkv, mask, pattern, seed);
                            long double vscale = 1;
                            for (uint16_t x : p.v) { vscale = std::max(vscale, std::fabs(static_cast<long double>(ggml_fp16_to_fp32(x)))); }
                            const std::vector<long double> ref = reference_attention(p);
                            const std::vector<float> rounded = fp32_attention(p);
                            std::vector<float> online;
                            require(online_attention(p, 32, false, online), "accepted online fixture");
                            const graph_results ggml = ggml_attention(p);
                            const metrics a = compare(ref, rounded, vscale);
                            const metrics o = compare(ref, online, vscale);
                            const metrics b = compare(ref, ggml.flash, vscale);
                            const metrics c = compare(ref, ggml.nonflash, vscale);
                            const std::vector<long double> rounded_ref(rounded.begin(), rounded.end());
                            const metrics e = compare(rounded_ref, ggml.flash, vscale);
                            const metrics f = compare(rounded_ref, ggml.nonflash, vscale);
                            accumulate(fp32_worst, a); accumulate(flash_worst, b); accumulate(nonflash_worst, c);
                            accumulate(online_worst, o);
                            accumulate(flash_fp32_worst, e); accumulate(nonflash_fp32_worst, f);
                            ++cases;
                            if (a.nmse > 1e-6L || a.maxabs > 1e-3L*vscale || a.maxrel > 5e-3L ||
                                o.nmse > 1e-6L || o.maxabs > 1e-3L*vscale || o.maxrel > 5e-3L ||
                                b.nmse > 5e-4L || c.nmse > 5e-4L || e.nmse > 5e-4L || f.nmse > 5e-4L) {
                                ++failures;
                                std::printf("FAIL fixture D=%zu Hq=4 Hkv=%zu N=%zu T=%zu mask=%u pattern=%u layer=%u sequence=%u seed=%u\n",
                                            d, hkv, n, t, mask, pattern, layer, sequence, seed);
                                print_metrics("  FP32/reference", a); print_metrics("  GGML-FA/reference", b);
                                print_metrics("  online FP32/reference", o);
                                print_metrics("  GGML-nonFA/reference", c);
                            }
                        }
                    }
                }
            }
        }
    }
    print_metrics("FP32/reference worst", fp32_worst);
    print_metrics("online FP32/reference worst", online_worst);
    print_metrics("GGML-FA/reference worst", flash_worst);
    print_metrics("GGML-nonFA/reference worst", nonflash_worst);
    print_metrics("GGML-FA/FP32 worst", flash_fp32_worst);
    print_metrics("GGML-nonFA/FP32 worst", nonflash_fp32_worst);
    std::printf("attention fixtures=%zu failed=%zu; Q pre-scaled, graph scale=1, no output repair\n", cases, failures);
    return failures == 0;
}

static bool test_online_boundaries() {
    metrics worst, flash_worst, nonflash_worst;
    size_t cases = 0, failures = 0;
    for (size_t n : { size_t(33), size_t(129), size_t(4096), size_t(32768) }) {
        packed_attention p = fixture(32, n, 1, 4, 0, n == 129 ? 1 : 0, 0x41545431u + uint32_t(n));
        if (n == 33) {
            std::fill(p.q.begin(), p.q.end(), 0);
            std::fill(p.k.begin(), p.k.end(), 0);
            for (size_t h = 0; h < p.hq; ++h) {
                p.q[h*p.d] = float(h + 1)/2;
                p.k[((n - 1)*p.hkv + h)*p.d] = ggml_fp32_to_fp16(8);
            }
        }
        const std::vector<long double> ref = reference_attention(p);
        std::vector<float> baseline;
        require(online_attention(p, 1, false, baseline), "online boundary baseline");
        for (size_t chunk : { size_t(1), size_t(7), size_t(32), size_t(128) }) {
            for (bool reverse : { false, true }) {
                std::vector<float> actual;
                require(online_attention(p, chunk, reverse, actual), "online chunk/order fixture");
                if (!reverse) { require(actual == baseline, "fetch boundary reset or reordered arithmetic"); }
                const metrics m = compare(ref, actual, 8);
                accumulate(worst, m); ++cases;
                if (m.nmse > 1e-6L || m.maxabs > 0.008L || m.maxrel > 5e-3L) {
                    ++failures;
                    std::printf("FAIL online boundary N=%zu chunk=%zu reverse=%d\n", n, chunk, int(reverse));
                    print_metrics("  online/reference", m);
                }
            }
        }
        const graph_results ggml = ggml_attention(p);
        const metrics f = compare(ref, ggml.flash, 8), nf = compare(ref, ggml.nonflash, 8);
        accumulate(flash_worst, f); accumulate(nonflash_worst, nf);
        if (f.nmse > 5e-4L || nf.nmse > 5e-4L) {
            ++failures;
            std::printf("FAIL extended GGML criterion N=%zu\n", n);
            print_metrics("  GGML-FA/reference", f); print_metrics("  GGML-nonFA/reference", nf);
        }
    }
    packed_attention rejected = fixture(32, 33, 1, 1, 0, 0, 7);
    std::fill(rejected.visible.begin(), rejected.visible.end(), 0);
    std::vector<float> invalid;
    require(!online_attention(rejected, 32, false, invalid) && invalid.empty(), "all-masked online must reject");
    rejected.visible[0] = 1;
    rejected.q[0] = std::numeric_limits<float>::max();
    rejected.k[0] = ggml_fp32_to_fp16(65504);
    require(!online_attention(rejected, 32, false, invalid) && invalid.empty(), "overflow online must reject");
    print_metrics("extended online/reference worst", worst);
    print_metrics("extended GGML-FA/reference worst", flash_worst);
    print_metrics("extended GGML-nonFA/reference worst", nonflash_worst);
    std::printf("online chunk/order cases=%zu failures=%zu; maximum history=32768; rejection checks=2\n", cases, failures);
    return failures == 0;
}

static int nearest_even(float x) {
    const float absolute = std::fabs(x);
    int value = int(std::floor(absolute));
    const float fractional = absolute - float(value);
    if (fractional > 0.5f || (fractional == 0.5f && value % 2)) { ++value; }
    return x < 0 ? -value : value;
}

static block_q8_0 q8_spec(const float * x, bool arm_neon) {
    float amax = 0;
    for (size_t i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(x[i])); }
    const float d = div32(amax, 127.0f);
    const float inverse = arm_neon ? (d ? div32(1.0f, d) : 0) : (amax ? div32(127.0f, amax) : 0);
    block_q8_0 out = {};
    out.d = ggml_fp32_to_fp16(d);
    for (size_t i = 0; i < 32; ++i) {
        const int quantized = nearest_even(mul32(x[i], inverse));
        require(quantized >= -127 && quantized <= 127, "Q8 specification range, no clamp");
        out.qs[i] = int8_t(quantized);
    }
    return out;
}

static void test_quantization() {
    require(sizeof(block_q8_0) == 34 && offsetof(block_q8_0, qs) == 2, "GGML Q8 block byte layout");
    const ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    require(traits && traits->from_float, "actual GGML Q8 converter");
    std::vector<float> input;
    // Exact signed half ties at d=1; zeros, subnormal scales, non-power-of-two
    // scales and near-half ties expose the x86/ARM reciprocal difference.
    uint32_t seed = 0x6a09e667u;
    for (size_t b = 0; b < 2052; ++b) {
        float amax = b == 0 ? 127.0f : b == 1 ? 0.0f : b == 2 ? 0.00001f : b == 3 ? 65504.0f :
                     add32(0.03125f, std::fabs(random_value(seed, 8.0f)));
        const float d = div32(amax, 127.0f);
        input.push_back(amax); input.push_back(-amax);
        for (size_t i = 2; i < 32; ++i) {
            const float sign = i % 2 ? -1.0f : 1.0f;
            const float half_tie = float((i/2) % 15) + 0.5f;
            input.push_back(mul32(sign, mul32(half_tie, d)));
        }
    }
    std::vector<block_q8_0> actual(input.size()/32);
    traits->from_float(input.data(), actual.data(), int64_t(input.size()));
    size_t arm_byte_differences = 0, arm_scale_differences = 0, local_spec_differences = 0;
    const bool x86_avx = ggml_cpu_has_avx() != 0;
    for (size_t b = 0; b < actual.size(); ++b) {
        const block_q8_0 arm = q8_spec(input.data() + b*32, true);
        const block_q8_0 avx = q8_spec(input.data() + b*32, false);
        if (arm.d != actual[b].d) { ++arm_scale_differences; }
        for (size_t i = 0; i < 32; ++i) {
            if (arm.qs[i] != actual[b].qs[i]) { ++arm_byte_differences; }
#if defined(__x86_64__) || defined(__i386__)
            if (x86_avx && (avx.d != actual[b].d || avx.qs[i] != actual[b].qs[i])) { ++local_spec_differences; }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            if (arm.d != actual[b].d || arm.qs[i] != actual[b].qs[i]) { ++local_spec_differences; }
#else
            (void) avx;
#endif
        }
        if (b == 0) {
            require(arm.d == ggml_fp32_to_fp16(1.0f), "exact unit Q8 scale");
            for (size_t i = 2; i < 32; ++i) {
                require(arm.qs[i] == nearest_even(input[i]), "signed round-to-nearest-even half ties");
            }
        }
        if (b == 1) {
            require(arm.d == 0, "zero Q8 scale");
            for (int8_t q : arm.qs) { require(q == 0, "zero block bytes"); }
        }
        // Deinterleave into the retained-data/scales layout and reconstruct.
        uint16_t scale;
        int8_t bytes[32];
        std::memcpy(&scale, &actual[b], 2);
        std::memcpy(bytes, reinterpret_cast<const unsigned char *>(&actual[b]) + 2, 32);
        require(scale == actual[b].d && std::memcmp(bytes, actual[b].qs, 32) == 0, "retained Q8 block split layout");
    }
    std::printf("Q8 actual GGML: AVX=%d blocks=%zu; local-ISA-spec mismatches=%zu; ARM_NEON-spec differences: bytes=%zu scales=%zu\n",
                int(x86_avx), actual.size(), local_spec_differences, arm_byte_differences, arm_scale_differences);
    require(local_spec_differences == 0, "local GGML Q8 differs from inspected ISA specification");
    require(arm_scale_differences == 0, "Q8 FP16 scales differ across inspected formulas");
    std::puts("ARM_NEON specification comparison is not an ARM or FPGA execution result.");
}

int main() {
    // A USE_FPGA=ON library lazily initializes hardware in MUL_MAT. Refuse it
    // before constructing/executing any graph, even on a machine with no board.
    require(dlsym(RTLD_DEFAULT, "fpga_init") == nullptr && dlsym(RTLD_DEFAULT, "g_fpga_initialized") == nullptr,
            "link only CPU-only GGML libraries (USE_FPGA=OFF)");
#if defined(__FAST_MATH__)
    require(false, "fast-math is not permitted for the arithmetic oracle");
#endif
    require(std::fesetround(FE_TONEAREST) == 0, "round-to-nearest FP environment");
    require(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits,
            "reference requires extended-precision long double");
    ggml_cpu_init();
    const bool attention_ok = test_attention();
    const bool online_ok = test_online_boundaries();
    test_quantization();
    std::printf("%s offline attention numerics; no RTL or board execution\n", attention_ok && online_ok ? "PASS" : "FAIL");
    return attention_ok && online_ok ? 0 : 1;
}
