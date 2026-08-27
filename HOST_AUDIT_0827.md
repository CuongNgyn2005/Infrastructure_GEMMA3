# Host-side investigation and optimization — 2026-08-27

Branch: `tmp/0827`
Baseline: `00ab1bd05173db18fd89fb02eff8e8d94f422efa`
Target model: Gemma3-1B-INT8 / Q8_0 host path

## Non-negotiable constraints

- Do not change any physical address definition in `ggml/src/ggml-cpu/fpga_host.cpp` without board evidence.
- Board memory is a 2 GiB system resource; the host must not assume the whole model can become permanently resident in the FPGA DDR carveout.
- Residency is a bounded working-set optimization, not a whole-model placement strategy.
- Do not enable a P3 data path unless the loaded bitstream advertises the required split-scale capability and exact ABI.
- Greedy decode (`--temp 0 --seed 1`) must remain token-ID identical to the baseline before a performance change is accepted.

## Verified current behavior

### Decode M=1

`M=1` is not itself a defect. It is the normal decode GEMV shape. It is a performance suspect because the current tiler calls `fpga_prepare_q8_tile_job()` inside the per-column loop. When neither the legacy weight cache nor P2 residency hits, static Q8 weights are pair-packed again and the P2 scale table is rebuilt for that job. With `M=1`, none of this preparation can be amortized over multiple activation columns.

Observed runtime evidence supplied with the project shows approximately 2.59 s wall/token, ~0.349 s IP compute, ~1.81 s host preparation, ~1.486 s direct weight packing and ~0.308 s scale packing. The FPGA compute engine is therefore not the dominant decode cost.

### Static weight scale

P2 residency already stores immutable weight-scale bits in host metadata for a sealed resident tile. However, P2 hardware consumes a 32-bit entry containing both the dynamic activation scale and static weight scale. Therefore `SPU_PARAM` still requires `rows * group_blocks` combined entries for each job. Caching weight-scale bits alone cannot make the P2 parameter table static.

The repository already contains the architectural solution: P3 split-scale. P3 writes the weight-scale table independently from a much smaller activation-scale table. This is the correct direction for removing repeated combined-table construction, but it is a hardware ABI change and cannot be enabled from host code by assumption.

## Existing regressions found

1. `verify_fpga_addresses.sh` described and probed a legacy address map (`0x77C00000`, `0x78C00000`, `0x79C00000`, `0x7AC00000`, `0x400000000`) that no longer matches `fpga_host.cpp`.
2. `verify_hardware.sh` repeated the same obsolete CMA/FPGA map and printed unconditional success statements even when the live board had not proven those assumptions.
3. After P2 residency was expanded from 16 MiB to 32 MiB, `P2_WEIGHT_RESIDENCY_END` became `0x03000000`, but one `static_assert` diagnostic string still describes the old `[0x01000000,0x02000000)` interval. This is diagnostic drift, not a runtime arithmetic error.
4. `fpga_host.cpp` mixes production data-path code, qualification experiments, verbose diagnostics, telemetry, address mapping, residency, scheduler state and contract checks in one translation unit. This makes regression review difficult and increases the chance that diagnostic code changes production sequencing.

## Fixes already committed on this branch

- Replaced the legacy address-probing script with a non-destructive address-contract checker synchronized to the current host map:
  - DDR carveout `[0x70000000, 0x80000000)`
  - MY_IP/LMM `0xA0000000`
  - ZDMA page `0xFD500000`
- Reworked `verify_hardware.sh` so it delegates to the synchronized checker, enumerates UIO mappings, and no longer claims hardware success without evidence.
- Neither change modifies any physical address in `fpga_host.cpp`.

## Required next host refactor

The large `fpga_host.cpp` should be split by responsibility without changing register/address constants or execution ordering:

1. `fpga_host_addr.*` — physical map constants, checked ranges, UIO/devmem mapping.
2. `fpga_host_log.*` — buffered logger, optional diagnostics, event/qualification traces.
3. `fpga_host_q8_pack.*` — Q8_0 activation packing, pair-interleaved weight packing, scale packing.
4. `fpga_host_residency.*` — bounded P2 tile residency and immutable metadata.
5. `fpga_host_dma.*` — ZDMA descriptor submission and waits.
6. `fpga_host_scheduler.*` — ping-pong/preload ownership state.
7. `fpga_host.cpp` — orchestration, public API, routing and lifecycle only.

The refactor must be mechanical first: move code with no semantic changes, compile, then compare host-only self-tests and greedy token output before any optimization is layered on top.

## P3 stop condition / required on-board evidence

Before enabling P3 split-scale in the production decode path, record all of the following from the deployed bitstream:

- `REG_BITSTREAM_ID == FPGA_EXPECTED_BITSTREAM_ID`
- `REG_STREAM_PROTOCOL_VERSION == FPGA_REQUIRED_STREAM_PROTOCOL_VERSION`
- P2 ABI remains the exact required P2 ABI
- SPU capability includes `SPU_CAP_P3_SPLIT_SCALE`
- `REG_P3_SPLIT_SCALE_ABI == FPGA_REQUIRED_P3_SPLIT_SCALE_ABI`
- split-scale mode write/readback is retained
- one bounded tile passes existing P3/P2 numerical contract checks before full inference

If any of these cannot be proven, keep production on P2 and do not claim the ~308 ms/token scale-pack cost has been solved.
