#include "llama.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct ContextDeleter {
    void operator()(llama_context * ctx) const {
        if (ctx != nullptr) {
            llama_free(ctx);
        }
    }
};

struct SamplerDeleter {
    void operator()(llama_sampler * sampler) const {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
        }
    }
};

using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

void load_backends_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_backend_load_all();
    });
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(256);
    int32_t n = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    if (n < 0) {
        buffer.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    }
    if (n < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }
    return std::string(buffer.data(), static_cast<size_t>(n));
}

class InfrastructureLlama {
public:
    InfrastructureLlama(
            std::string model_path,
            uint32_t n_ctx,
            uint32_t n_batch,
            int32_t n_gpu_layers)
        : model_path_(std::move(model_path)),
          n_ctx_(n_ctx),
          n_batch_(n_batch) {
        if (model_path_.empty()) {
            throw std::invalid_argument("model_path must not be empty");
        }
        if (n_ctx_ == 0 || n_batch_ == 0) {
            throw std::invalid_argument("n_ctx and n_batch must be greater than zero");
        }

        load_backends_once();

        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = n_gpu_layers;

        model_ = llama_model_load_from_file(model_path_.c_str(), params);
        if (model_ == nullptr) {
            throw std::runtime_error("failed to load model: " + model_path_);
        }

        vocab_ = llama_model_get_vocab(model_);
        if (vocab_ == nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
            throw std::runtime_error("model vocabulary is unavailable");
        }
    }

    ~InfrastructureLlama() {
        if (model_ != nullptr) {
            llama_model_free(model_);
        }
    }

    InfrastructureLlama(const InfrastructureLlama &) = delete;
    InfrastructureLlama & operator=(const InfrastructureLlama &) = delete;

    std::vector<llama_token> tokenize(
            const std::string & text,
            bool add_special,
            bool parse_special) const {
        const int32_t needed = -llama_tokenize(
            vocab_, text.data(), static_cast<int32_t>(text.size()),
            nullptr, 0, add_special, parse_special);

        if (needed < 0) {
            throw std::runtime_error("failed to determine token count");
        }
        if (needed == 0) {
            return {};
        }

        std::vector<llama_token> tokens(static_cast<size_t>(needed));
        const int32_t written = llama_tokenize(
            vocab_, text.data(), static_cast<int32_t>(text.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()),
            add_special, parse_special);
        if (written < 0) {
            throw std::runtime_error("llama_tokenize failed");
        }
        tokens.resize(static_cast<size_t>(written));
        return tokens;
    }

    std::string generate(
            const std::string & prompt,
            int32_t max_tokens,
            float temperature,
            uint32_t seed) const {
        if (max_tokens < 0) {
            throw std::invalid_argument("max_tokens must be non-negative");
        }

        std::vector<llama_token> prompt_tokens = tokenize(prompt, true, true);
        if (prompt_tokens.empty()) {
            throw std::runtime_error("prompt produced no tokens");
        }
        if (prompt_tokens.size() + static_cast<size_t>(max_tokens) > n_ctx_) {
            throw std::runtime_error("prompt + max_tokens exceeds configured n_ctx");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx_;
        ctx_params.n_batch = std::min<uint32_t>(
            n_ctx_,
            std::max<uint32_t>(n_batch_, static_cast<uint32_t>(prompt_tokens.size())));
        ctx_params.no_perf = false;

        ContextPtr ctx(llama_init_from_model(model_, ctx_params));
        if (!ctx) {
            throw std::runtime_error("failed to create llama_context");
        }

        SamplerPtr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        if (!sampler) {
            throw std::runtime_error("failed to create sampler");
        }

        if (temperature <= 0.0f) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));
        }

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));

        std::string output;
        llama_token next_token = LLAMA_TOKEN_NULL;

        for (int32_t i = 0; i < max_tokens; ++i) {
            const int decode_status = llama_decode(ctx.get(), batch);
            if (decode_status != 0) {
                throw std::runtime_error("llama_decode failed with code " + std::to_string(decode_status));
            }

            next_token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
            if (llama_vocab_is_eog(vocab_, next_token)) {
                break;
            }

            output += token_to_piece(vocab_, next_token);
            batch = llama_batch_get_one(&next_token, 1);
        }

        return output;
    }

    const std::string & model_path() const {
        return model_path_;
    }

    uint32_t n_ctx() const {
        return n_ctx_;
    }

    uint32_t n_batch() const {
        return n_batch_;
    }

private:
    std::string model_path_;
    uint32_t n_ctx_;
    uint32_t n_batch_;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
};

} // namespace

PYBIND11_MODULE(_infrastructure, m) {
    m.doc() = "Python API for the Infrastructure_GEMMA3 llama.cpp runtime";

    py::class_<InfrastructureLlama>(m, "Llama")
        .def(py::init<std::string, uint32_t, uint32_t, int32_t>(),
             py::arg("model_path"),
             py::arg("n_ctx") = 4096,
             py::arg("n_batch") = 512,
             py::arg("n_gpu_layers") = 0)
        .def("generate", &InfrastructureLlama::generate,
             py::arg("prompt"),
             py::arg("max_tokens") = 128,
             py::arg("temperature") = 0.0f,
             py::arg("seed") = 1,
             py::call_guard<py::gil_scoped_release>())
        .def("tokenize", &InfrastructureLlama::tokenize,
             py::arg("text"),
             py::arg("add_special") = true,
             py::arg("parse_special") = true)
        .def_property_readonly("model_path", &InfrastructureLlama::model_path)
        .def_property_readonly("n_ctx", &InfrastructureLlama::n_ctx)
        .def_property_readonly("n_batch", &InfrastructureLlama::n_batch);
}
