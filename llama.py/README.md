# Infrastructure Python API

This directory is a thin Python binding for the parent `Infrastructure_GEMMA3` runtime.
It does not build or carry a second llama/ggml implementation. Python calls the same
`llama` target used by the rest of Infrastructure, so GGML execution still reaches the
existing FPGA interception and `fpga_host.cpp` path when Infrastructure is built with
`USE_FPGA=ON`.

## Build

From the repository root, reuse the existing build directory and enable the Python API:

```bash
cmake -S . -B build_mem -DUSE_FPGA=ON -DLLAMA_BUILD_PYTHON=ON
cmake --build build_mem --target _infrastructure -j4
```

`pybind11` development files must be discoverable by CMake.

The package is emitted to `build_mem/python/llama`. Run it with:

```bash
PYTHONPATH=build_mem/python python3 - <<'PY'
from llama import Llama

llm = Llama(
    "/home/debian/soc/models/gemma-3-1b-it-Q8_0.gguf",
    n_ctx=4096,
    n_batch=512,
)

print(llm.generate(
    "Write a short overview of Vietnam.",
    max_tokens=32,
    temperature=0.0,
    seed=1,
))
PY
```

`Llama.generate()` keeps the complete decode loop in C/C++; Python only enters at the
high-level API boundary. `n_gpu_layers` defaults to `0` so the CPU GGML backend remains
the execution path where this project intercepts supported matrix multiplications for FPGA.
