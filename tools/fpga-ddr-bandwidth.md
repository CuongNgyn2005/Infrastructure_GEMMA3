# Zynq DDR bandwidth measurement

`fpga-ddr-bandwidth` reads the DDR-controller AXI Performance Monitor (APM)
exposed by Linux as `/dev/uio1`. `--slot N` measures one port; `--slots 1,2,5`
measures those three ports simultaneously using six hardware counters. The
selected-port aggregate includes system traffic on those ports. It is neither
whole-DDR bandwidth nor llama-exclusive traffic. Slot numbers do not establish
which CPU, ZDMA, or PL master generated traffic; that mapping needs separate
platform evidence. CPU cache hits and FPGA internal arithmetic are not measured.

## Build and run

On the board, from the llama.cpp directory:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror \
    -o fpga-ddr-bandwidth tools/fpga-ddr-bandwidth.c
sudo ./fpga-ddr-bandwidth --uio /dev/uio1 --slots 1,2,5 --seconds 60
```

The existing single-slot command remains supported:

```sh
sudo ./fpga-ddr-bandwidth --uio /dev/uio1 --slot 0 --seconds 60
```

Use exactly one slot selector. Lists contain 1..5 unique single-digit slots
from 0..5; list order controls the counter assignment and output order. Empty,
repeated, conflicting, malformed, out-of-range, and six-port lists are rejected.
There are ten metric counters, requiring two per selected port. The tool still
requires the existing DDR monitor capability contract, including ten counters.

Confirm that map0 is the DDR APM (`name=axi-pmon`, address `0xfd0b0000`, size
at least `0x10000`). Start the tool immediately before the workload, then run
the unchanged model command in a second terminal. The interval includes model
loading if loading happens after the monitor starts; it is not automatically
decode-only. Avoid unrelated workloads when comparing runs. Separate runs are
not simultaneous measurements and must not be summed as if they were.

## Shared hardware window

For slots 1,2,5, counters 0/1 measure slot 1 write/read bytes, counters 2/3
slot 2, and counters 4/5 slot 5. Metric selectors use `(slot << 5) | metric`,
where metric 2 is write bytes and metric 3 read bytes. Counter `n` is at
`0x100 + 0x10*n`; selector words are at `0x44`, `0x48`, and `0x4c`.
Only selected selector bytes are changed. For this list MSR0 is `0x43422322`,
MSR1's low 16 bits are `0xa3a2` with its upper 16 bits preserved, and MSR2 is
unchanged during configuration.

```text
Stopped: save configuration -> select all pairs -> reset ALL metric counters
                                  |
One shared metric-enable write    +--------------------------------+
slot 1 counters                   | write / read accumulation      |
slot 2 counters                   | write / read accumulation      |
slot 5 counters                   | write / read accumulation      |
One shared stop + readback        +--------------------------------+
Stopped: read status and all selected counters -> validate -> print
```

Control bit 0 starts/stops the shared metric bank; bit 1 resets it. Global
cycle-counter enable bit 16 is not used. Requested sleeps are at most 100 ms,
with a shorter final window when appropriate; OS scheduling can extend the
actual active interval. All ports use the same software-measured interval.

The tool requires an idle APM and takes an advisory UIO lock. It rejects active
control, sample enable/load, interrupt enables, and pending status. Sample
interval control bit `0x100` alone is an idle metric-reset policy, so it is
accepted. Another client that ignores the lock must not use the APM concurrently.

Configuration is saved before changes. On normal exit, handled SIGINT/SIGTERM,
and reported errors, cleanup stops the bank and restores selectors (including
neighboring fields), control, sample interval/control, and interrupt settings.
Selector writes are checked by readback. A failed partial setup uses the same
cleanup. Configuration restoration does not restore counter history or status:
the shared reset resets **all ten metric counters**, including unselected ones,
and overflow status is cleared as needed. SIGKILL, process crashes, or bus faults
cannot be guaranteed to run cleanup.

## Output and validity

Each valid window prints one line per selected slot, with the same window number,
timestamp, `active_s`, `window_s`, and `overhead_us`. Existing single-slot keys
are retained, with `combined_MBps` added. Session output has one `AGGREGATE slot=N`
line per port. Multi-port sessions additionally print:

```text
SELECTED_AGGREGATE scope=selected_ports_system_traffic ports=3 active_s=... write_bytes=... read_bytes=... combined_bytes=... write_MBps=... read_MBps=... combined_MBps=...
```

Write, read, and combined rates use decimal MB/s: `bytes / active_s / 1,000,000`.
Each session rate uses that port's 64-bit byte totals divided by the **same sum
of valid active intervals**. The selected aggregate sums selected-port bytes and
uses that same denominator; it does not sum separate wall-clock runs. Combined
bytes are added in 64-bit arithmetic before conversion to a rate.

`active_s` begins immediately before the enable write and ends after the stop
write's readback. `window_s` begins before status preparation/reset and ends
after status/counter reads. `overhead_us = (window_s - active_s) * 1,000,000`.
Session `total_wall_s` includes setup, reads, resets, output, and inter-window
gaps. `dead_s = total_wall_s - active_s` and
`coverage_pct = active_s / total_wall_s * 100`. Traffic during dead time is not
counted or extrapolated. Timestamps use UTC; durations use `CLOCK_MONOTONIC`.
No hardware clock frequency or cycle-accurate timing is assumed.

Any nonzero status invalidates the window and the entire session aggregate.
This deliberately includes unknown status and overflow from unselected counters:
their selector fields are preserved and the shared bank still runs them.
Counter overflow bit `n+3` identifies metric counter `n`. No wrapped counter is
silently repaired. Nonpositive/nonfinite active time, inconsistent intervals,
clock/sleep failures, signals, configuration readback errors, and output errors
also invalidate the session. A nonzero process exit means discard all session
aggregate lines, including any already printed before a later output failure.
The tool does not emit a successful zero-denominator rate.

If ownership is refused, stop the other APM client. Resolve status or timing
errors before repeating. A 100 ms requested window reduces wrap risk but does
not prove no overflow; status checks remain mandatory.

## Register references and local validation

The register offsets, selectors, and control masks follow the Xilinx driver:

- [xaxipmon_hw.h, xilinx_v2021.2](https://raw.githubusercontent.com/Xilinx/embeddedsw/xilinx_v2021.2/XilinxProcessorIPLib/drivers/axipmon/src/xaxipmon_hw.h)
- [xaxipmon.c, xilinx_v2021.2](https://raw.githubusercontent.com/Xilinx/embeddedsw/xilinx_v2021.2/XilinxProcessorIPLib/drivers/axipmon/src/xaxipmon.c)
- [xaxipmon.h, xilinx_v2021.2](https://raw.githubusercontent.com/Xilinx/embeddedsw/xilinx_v2021.2/XilinxProcessorIPLib/drivers/axipmon/src/xaxipmon.h)

Run the RAM-only regression with `python3 tests/test-fpga-ddr-bandwidth.py`.
It compiles extracted production parsing, register configuration, measurement,
and cleanup code against mocked MMIO, clocks, and signals; it honors `CC` and
uses C11 with `-Wall -Wextra -Werror`. This is local software evidence, not proof
of board traffic attribution, real MMIO behavior, or bandwidth. Native Linux
compilation and owner-returned ZCU104 output remain deployment checks.
