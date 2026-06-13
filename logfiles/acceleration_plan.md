# ZCU104 Acceleration Journal

## 2026-06-12 - `zcu104-packed-q8-v11-stage-profile`

### Baseline

- Reported throughput: `0.47 tokens/s` in hybrid FPGA/CPU mode.
- Host trace: `zcu104-packed-q8-v10-profile`.
- Loaded bitstream reports `REG_CAPS=0x00000000`.
- Packed-Q8 self-test fails with `[96,0,0,0]`; expected `[32,64,-32,192]`.
- Active FPGA path is therefore `legacy-q8-block` through MMIO.

### Measured bottlenecks

- `K1024_N1152_M13` attention output projection: about `280.75 ms/call`, `2080` VPU runs.
- `K1152_N1024_M13` attention Q projection: about `280.48 ms/call`, `1872` VPU runs.
- Decode `K1152_N1024_M1` and `K1024_N1152_M1`: about `27.6 ms/call` each.
- K/V decode projections `K1152_N256_M1`: about `6.91 ms/call`.
- The delay is uniform across layers. This points to the shared legacy INT8 GEMV/MMIO path, not one defective transformer layer.
- CPU fallbacks remain for attention, FFN gate/up, FFN down, and token embedding. This run is not FPGA-only.

### Changes planned for `fpga_host.cpp`

1. Add sampled stage timing for weight MMIO writes, activation MMIO writes, VPU wait, result MMIO reads, and remaining host work.
2. Reduce logging overhead by buffering routine INFO/PROFILE lines and throttling repeated attention-fallback messages.
3. Reduce default periodic profile frequency while retaining `SLOW`, `shape_top`, and `fallback_shape_top` diagnostics.
4. Keep the current numerical path unchanged in this revision. A full-K requantized legacy GEMV is a later experimental optimization because it changes quantization accuracy and needs hardware-side validation.

### Implemented

- Updated host trace to `zcu104-packed-q8-v11-stage-profile`.
- Added sampled `[PROFILE] stage` records with:
  - `weight_mmio_ms`
  - `activation_mmio_ms`
  - `vpu_wait_ms`
  - `result_mmio_ms`
  - `host_other_ms`
  - `dominant`
- Stage timing samples the first FPGA call and then every `128` accepted FPGA calls by default. Override with `FPGA_STAGE_EVERY`.
- Added a one-entry activation quantization cache for consecutive projections sharing the same input tensor, data address, shape, and strides. This targets repeated Q/K/V activation quantization without changing FPGA arithmetic.
- Reduced default profile reports from every `128` FPGA calls to every `256` calls.
- Routine log writes are flushed every `16` lines instead of every line. Errors still flush immediately. Override with `FPGA_LOG_FLUSH_EVERY`.
- Repeated attention fallback messages are limited to the first eight and then one message every `128` fallbacks. Aggregate fallback counters remain available in periodic profile records.

### Expected next-log signals

- A line such as `dominant=weight_mmio` means host-to-BRAM weight traffic is the next optimization target.
- `dominant=vpu_wait` means the RTL compute/start-done path dominates and host-only changes will have limited value.
- `dominant=result_mmio` points to result window read bandwidth/transaction overhead.
- `dominant=host_other` includes quantization, accumulation, configuration, and untimed host work.
- `activation quantization cache hits` should become non-zero when Q/K/V share the same source tensor identity.

### Runtime validation required

- Confirm generated tokens remain correct.
- Compare `eval time`, tokens/s, and stage breakdown against the `0.47 tokens/s` baseline.
- Copy the resulting `/tmp/fpga_debug.log` back to `logfiles/debug.log` for the next iteration.

### Next hardware direction

- The primary acceleration target is a working packed-Q8 RTL mode. It reduces VPU launches per projection and preserves per-Q8-block scaling.
- A host-only full-K legacy mode could reduce launches further, but requires per-row weight requantization and accuracy testing before it can become a default path.

### Verification status

- Local static/diff checks only. ZCU104 runtime verification must be performed through the remote board.

## 2026-06-12 - `zcu104-packed-q8-v12-capability-timing`

### Flow decision

- The PS/CPU keeps transformer-layer orchestration and context handling.
- Compatible `Q8_0 x F32 -> F32` matrix-vector operations are submitted to the PL VPU.
- The PL result is read back and accumulated/dequantized by the host before llama.cpp continues.
- Flash-attention and non-Q8_0 operations remain CPU fallbacks because the current RTL exposes only the INT8 GEMV datapath.

### `fpga_host.cpp` changes

- Packed mode is no longer guessed when `REG_CAPS=0`. It is enabled only by a valid capability register or explicit `FPGA_PROBE_PACKED_Q8=1` / `FPGA_FORCE_PACKED_Q8=1`.
- `FPGA_REQUIRE_PACKED=1` now fails FPGA initialization when the loaded bitstream does not expose packed mode. This prevents an unnoticed fallback to the slow legacy launch-per-block flow.
- Added experimental aligned 128-bit data-window accesses with `FPGA_ACCESS128=1`. The existing smoke tests automatically fall back to 64-bit and then 32-bit MMIO when required.
- Added `FPGA_OFFLOAD_ALL=1` coverage mode. It removes N/decode thresholds and enables legacy K tiling, but is not a performance default because large token-head matrices are expensive over CPU-driven MMIO.
- Added periodic runtime status with `FPGA_STATUS_EVERY` (default `128`). Set `FPGA_STATUS_STDERR=1` to also print status in the llama-cli terminal.
- Host trace updated to `zcu104-packed-q8-v12-capability-timing`.

### RTL changes validated in `DATN_VIVADO/manual_sim`

- Packed q8_0 groups stream continuously instead of stopping after every block.
- Configuration is latched and validated in a separate FSM state.
- Activation/weight BRAM reads have an additional registered pipeline stage.
- `MAX_COL_BEATS` is reduced from `256` to `32`, matching the maximum packed group of 16 q8_0 blocks. Larger K values are tiled by the host.
- Removed the unconnected ILA instance from the standalone RTL top used by manual implementation.
- Packed XSim result: `[32,64,-32,192]`, PASS.
- Legacy XSim cases: `4x4`, `3x17`, and `4x64`, all PASS.
- Post-route 300 MHz result: `WNS=+0.238 ns`, `TNS=0`, zero failing endpoints.
- Utilization: `1665 LUT`, `40 BRAM36`, `18 DSP`. The previous standalone build used `264 BRAM36` and failed timing.

### Required bitstream signature

- `REG_LIMITS` must report rows `256`, col-beats `32`: expected value `0x00200100`.
- `REG_CAPS` must report packed mode, 16 blocks, 1024 result words: expected value `0x04001001`.
- A board still reporting `REG_CAPS=0x00000000` is running the stale legacy RTL, even if the host binary is v12.

### Recommended board validation

```bash
export FPGA_REQUIRE_PACKED=1
export FPGA_SELF_TEST=1
export FPGA_STATUS_EVERY=32
export FPGA_STATUS_STDERR=1
export FPGA_STAGE_EVERY=32
unset FPGA_OFFLOAD_ALL
unset FPGA_PROBE_PACKED_Q8
```

Run the normal `llama-cli` command, then collect:

```bash
cp /tmp/fpga_debug.log ./logfiles/debug-v12-board.log
```

Only after the normal packed run is numerically correct, test `FPGA_ACCESS128=1` and compare stage-level MMIO times. Use `FPGA_OFFLOAD_ALL=1` only to audit compatible-matmul coverage, not as the expected fastest configuration.

### Remaining architectural limit

- The VPU is still an AXI4-Full slave with local BRAM windows. The CPU writes model weights for each invocation.
- This host-driven traffic is the main barrier to the `2.5 tokens/s` target. Reaching the target reliably requires a PL AXI master/CDMA path that reads weights directly from DDR, or persistent on-chip weight tiles with layer-aware scheduling.
- The current Vivado project sources outside `manual_sim` were not modified. The optimized RTL must be synchronized/repackaged into the actual Vivado project before generating the board bitstream.

### Local verification

- `git diff --check`: PASS.
- MinGW `g++ -fsyntax-only` with a temporary Linux `mman` declaration stub: PASS.
- ZCU104 numerical and throughput validation remains pending remote board execution.
