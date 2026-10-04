#!/usr/bin/env python3
"""Collect Linux scheduling evidence around a user-supplied command.

This wrapper does not change affinity, priority, governors, environment, or
the command. It samples the direct child and every task visible below its
/proc/<pid>/task directory once per second. It is intended for owner-side
stability runs; it does not access FPGA devices or infer hardware behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Optional


MAX_RUNS = 100
SAMPLE_INTERVAL_S = 1.0
SCRIPT_VERSION = "fpga-stability-profile-v2"


def field(value: Any = None, error: Optional[str] = None, unit: Optional[str] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"value": value, "error": error}
    if unit is not None:
        result["unit"] = unit
    return result


def read_text(path: Path) -> tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:  # proc/sysfs entries may disappear between samples
        return None, f"{type(exc).__name__}: {exc}"


def read_int(path: Path, unit: str) -> dict[str, Any]:
    text, error = read_text(path)
    if error is not None or text is None:
        return field(None, error or "unavailable", unit)
    try:
        return field(int(text.strip()), None, unit)
    except ValueError as exc:
        return field(None, f"ValueError: {exc}", unit)


def missing_fields(names: Iterable[str], error: str, units: Optional[dict[str, str]] = None) -> dict[str, dict[str, Any]]:
    return {name: field(None, error, (units or {}).get(name)) for name in names}


def parse_stat(tid: int, path: Path) -> dict[str, Any]:
    names = ("comm", "state", "utime", "stime", "minflt", "majflt", "last_cpu")
    units = {
        "utime": "clock ticks",
        "stime": "clock ticks",
        "minflt": "faults",
        "majflt": "faults",
        "last_cpu": "cpu index",
    }
    text, error = read_text(path)
    if error is not None or text is None:
        result = missing_fields(names, error or "unavailable", units)
        result["tid"] = field(tid, None, "task id")
        result["last_cpu_note"] = field(None, "unavailable; /proc stat could not be read")
        return result

    open_paren = text.find("(")
    close_paren = text.rfind(")")
    if open_paren <= 0 or close_paren <= open_paren:
        result = missing_fields(names, "malformed /proc stat: missing comm delimiters", units)
        result["tid"] = field(tid, None, "task id")
        result["last_cpu_note"] = field(None, "unavailable; malformed /proc stat")
        return result

    comm = text[open_paren + 1:close_paren]
    fields_after_comm = text[close_paren + 2:].split()
    result: dict[str, Any] = {
        "tid": field(tid, None, "task id"),
        "comm": field(comm),
        "state": field(None, None),
        "utime": field(None, None, "clock ticks"),
        "stime": field(None, None, "clock ticks"),
        "minflt": field(None, None, "faults"),
        "majflt": field(None, None, "faults"),
        "last_cpu": field(None, None, "cpu index"),
        "last_cpu_note": field(
            "last CPU reported by /proc stat; this is not proof of process affinity"
        ),
    }
    indexes = {"state": 0, "minflt": 7, "majflt": 9, "utime": 11, "stime": 12, "last_cpu": 36}
    for name, index in indexes.items():
        if index >= len(fields_after_comm):
            result[name] = field(None, f"/proc stat field {index + 3} is missing", units.get(name))
            continue
        raw = fields_after_comm[index]
        if name == "state":
            result[name] = field(raw)
            continue
        try:
            result[name] = field(int(raw), None, units.get(name))
        except ValueError as exc:
            result[name] = field(None, f"ValueError: {exc}", units.get(name))
    return result


def parse_status(path: Path) -> dict[str, Any]:
    names = ("Cpus_allowed_list", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
    text, error = read_text(path)
    if error is not None or text is None:
        return missing_fields(names, error or "unavailable", {
            "voluntary_ctxt_switches": "context-switch count",
            "nonvoluntary_ctxt_switches": "context-switch count",
        })

    values: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator:
            values[name] = value.strip()
    result: dict[str, Any] = {}
    for name in names:
        raw = values.get(name)
        if raw is None:
            result[name] = field(None, "field not present")
        elif name == "Cpus_allowed_list":
            result[name] = field(raw)
        else:
            try:
                result[name] = field(int(raw), None, "context-switch count")
            except ValueError as exc:
                result[name] = field(None, f"ValueError: {exc}", "context-switch count")
    return result


def parse_schedstat(path: Path) -> dict[str, Any]:
    names = ("runtime", "run_delay", "timeslices")
    units = {"runtime": "nanoseconds", "run_delay": "nanoseconds", "timeslices": "count"}
    text, error = read_text(path)
    if error is not None or text is None:
        return missing_fields(names, error or "unavailable", units)
    parts = text.split()
    if len(parts) < 3:
        return missing_fields(names, "malformed schedstat: fewer than three fields", units)
    result: dict[str, Any] = {}
    for name, raw in zip(names, parts[:3]):
        try:
            result[name] = field(int(raw), None, units[name])
        except ValueError as exc:
            result[name] = field(None, f"ValueError: {exc}", units[name])
    return result


def list_task_ids(task_root: Path) -> tuple[Optional[list[int]], Optional[str]]:
    try:
        tids = sorted(int(entry.name) for entry in task_root.iterdir() if entry.name.isdigit())
        return tids, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def sample_tasks(pid: int) -> dict[str, Any]:
    task_root = Path("/proc") / str(pid) / "task"
    tids, error = list_task_ids(task_root)
    if error is not None or tids is None:
        return {
            "task_ids": field(None, error or "unavailable"),
            "tasks": field(None, error or "unavailable"),
        }

    tasks: dict[str, Any] = {}
    for tid in tids:
        task_root_for_tid = task_root / str(tid)
        tasks[str(tid)] = {
            "stat": parse_stat(tid, task_root_for_tid / "stat"),
            "status": parse_status(task_root_for_tid / "status"),
            "schedstat": parse_schedstat(task_root_for_tid / "schedstat"),
        }
    return {"task_ids": field(tids, None, "task ids"), "tasks": field(tasks)}


def sample_proc_stat() -> dict[str, Any]:
    path = Path("/proc/stat")
    text, error = read_text(path)
    if error is not None or text is None:
        return {"error": error or "unavailable", "cpu": field(None, error or "unavailable", "jiffies")}

    result: dict[str, Any] = {}
    wanted = {"ctxt": "context-switch count", "processes": "process count",
              "procs_running": "process count", "procs_blocked": "process count"}
    cpu_values: Optional[list[int]] = None
    cpu_error: Optional[str] = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        cpu_name = parts[0]
        is_cpu_line = cpu_name == "cpu" or (
            cpu_name.startswith("cpu") and cpu_name[3:].isdigit()
        )
        if is_cpu_line:
            values: Optional[list[int]] = None
            value_error: Optional[str] = None
            if len(parts) <= 1:
                value_error = "malformed /proc/stat cpu line: no counters"
            else:
                try:
                    values = [int(value) for value in parts[1:]]
                except ValueError as exc:
                    value_error = f"ValueError: {exc}"
            if cpu_name == "cpu":
                cpu_values = values
                cpu_error = value_error
            else:
                result[cpu_name] = field(values, value_error, "jiffies")
        elif parts[0] in wanted and len(parts) > 1:
            try:
                result[parts[0]] = field(int(parts[1]), None, wanted[parts[0]])
            except ValueError as exc:
                result[parts[0]] = field(None, f"ValueError: {exc}", wanted[parts[0]])
    result["cpu"] = field(cpu_values, cpu_error or (None if cpu_values is not None else "cpu aggregate not present"), "jiffies")
    for name, unit in wanted.items():
        result.setdefault(name, field(None, "field not present", unit))
    return result


def sample_schedstats_control() -> dict[str, Any]:
    return read_int(
        Path("/proc/sys/kernel/sched_schedstats"),
        "enabled flag (0 or 1)",
    )


def sample_cpufreq() -> dict[str, Any]:
    root = Path("/sys/devices/system/cpu")
    try:
        paths = sorted(root.glob("cpu[0-9]*/cpufreq"))
    except Exception as exc:
        return {"cpus": field(None, f"{type(exc).__name__}: {exc}")}
    if not paths:
        return {"cpus": field({}, "no cpufreq directories found")}
    cpus: dict[str, Any] = {}
    for path in paths:
        cpu_name = path.parent.name
        governor_text, governor_error = read_text(path / "scaling_governor")
        cpus[cpu_name] = {
            "current": read_int(path / "scaling_cur_freq", "kHz"),
            "governor": field(
                governor_text.strip() if governor_text is not None else None,
                governor_error or None,
            ),
        }
    return {"cpus": field(cpus)}


def sample_thermal() -> dict[str, Any]:
    root = Path("/sys/class/thermal")
    try:
        paths = sorted(root.glob("thermal_zone*/temp"))
    except Exception as exc:
        return {"zones": field(None, f"{type(exc).__name__}: {exc}")}
    if not paths:
        return {"zones": field({}, "no thermal zones found")}
    zones: dict[str, Any] = {}
    for temp_path in paths:
        zone = temp_path.parent
        type_text, type_error = read_text(zone / "type")
        zones[zone.name] = {
            "temperature": read_int(temp_path, "millidegrees Celsius"),
            "type": field(type_text.strip() if type_text is not None else None, type_error or None),
        }
    return {"zones": field(zones)}


def sample_process(pid: int, returncode: Optional[int]) -> dict[str, Any]:
    return {
        "pid": field(pid, None, "process id"),
        "alive": field(returncode is None),
        "returncode": field(returncode, None if returncode is not None else "still running"),
        **sample_tasks(pid),
    }


def sample_once(pid: int, returncode: Optional[int]) -> dict[str, Any]:
    return {
        "process": sample_process(pid, returncode),
        "proc_stat": sample_proc_stat(),
        "cpufreq": sample_cpufreq(),
        "thermal": sample_thermal(),
    }


def positive_runs(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--runs must be an integer: {exc}") from exc
    if parsed < 1 or parsed > MAX_RUNS:
        raise argparse.ArgumentTypeError(f"--runs must be between 1 and {MAX_RUNS}")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def terminate_child(process: subprocess.Popen[bytes]) -> Optional[int]:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.returncode


def run_one(command: list[str], output_dir: Path, run_index: int) -> tuple[dict[str, Any], bool]:
    prefix = f"run-{run_index:03d}"
    stdout_path = output_dir / f"{prefix}.stdout.txt"
    stderr_path = output_dir / f"{prefix}.stderr.txt"
    samples_path = output_dir / f"{prefix}.samples.jsonl"
    started_wall = utc_now()
    started_mono = time.monotonic()
    process: Optional[subprocess.Popen[bytes]] = None
    sample_count = 0
    interrupted = False
    spawn_error: Optional[str] = None
    returncode: Optional[int] = None

    print(f"run={run_index} starting={started_wall} command={command}", flush=True)
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file, samples_path.open(
        "w", encoding="utf-8"
    ) as samples_file:
        try:
            process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=False)
            print(f"run={run_index} pid={process.pid} started={started_wall}", flush=True)
            next_sample = time.monotonic()
            while True:
                returncode = process.poll()
                sample = {
                    "kind": "sample",
                    "run": run_index,
                    "sequence": sample_count,
                    "event": "interval" if returncode is None else "exit",
                    "wall_time": utc_now(),
                    "monotonic_s": time.monotonic(),
                    **sample_once(process.pid, returncode),
                }
                samples_file.write(json.dumps(sample, sort_keys=True) + "\n")
                samples_file.flush()
                sample_count += 1
                if returncode is not None:
                    break
                next_sample += SAMPLE_INTERVAL_S
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                returncode = terminate_child(process)
            print(f"run={run_index} interrupted=1 child_reaped=1", flush=True)
        except OSError as exc:
            spawn_error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                returncode = terminate_child(process)
        finally:
            if process is not None and process.poll() is None:
                returncode = terminate_child(process)

    ended_wall = utc_now()
    result = {
        "kind": "run_result",
        "run": run_index,
        "pid": process.pid if process is not None else None,
        "command": command,
        "started_wall_time": started_wall,
        "ended_wall_time": ended_wall,
        "elapsed_s": time.monotonic() - started_mono,
        "returncode": returncode,
        "status": "interrupted" if interrupted else ("spawn_failed" if spawn_error else ("completed" if returncode == 0 else "failed")),
        "error": spawn_error,
        "sample_count": sample_count,
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "samples_file": samples_path.name,
    }
    print(
        f"run={run_index} status={result['status']} returncode={returncode} samples={sample_count} "
        f"ended={ended_wall}",
        flush=True,
    )
    return result, interrupted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new directory for metadata, samples, and command output")
    parser.add_argument("--runs", type=positive_runs, default=1, help=f"positive run count, maximum {MAX_RUNS}")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command and arguments after --")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {output_dir}")
    except OSError as exc:
        parser.error(f"cannot create output directory {output_dir}: {exc}")

    try:
        clk_tck = os.sysconf("SC_CLK_TCK")
        clock = {"value": clk_tck, "error": None, "unit": "clock ticks per second"}
    except (AttributeError, OSError, ValueError) as exc:
        clock = {"value": None, "error": f"{type(exc).__name__}: {exc}", "unit": "clock ticks per second"}
    relevant_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("OMP_", "GOMP_", "KMP_"))
    }
    metadata = {
        "kind": "profile_metadata",
        "script_version": SCRIPT_VERSION,
        "created_wall_time": utc_now(),
        "cwd": os.getcwd(),
        "argv": command,
        "shell": False,
        "runs_requested": args.runs,
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "clock": {"clk_tck": clock},
        "relevant_environment": relevant_env,
        "inherited_parallel_environment": {
            "variables": relevant_env,
            "inherited_unchanged": True,
            "note": (
                "OMP_/GOMP_/KMP_ variables are inherited by the child and passed "
                "unchanged; an empty set means no such variables were present"
            ),
        },
        "sched_schedstats": sample_schedstats_control(),
        "units": {
            "stat_ticks": "clock ticks; convert using clock.clk_tck.value",
            "schedstat_runtime": "nanoseconds",
            "cpufreq_current": "kHz",
            "thermal_temperature": "millidegrees Celsius",
            "proc_stat_cpu": "jiffies; convert using clock.clk_tck.value where applicable",
        },
        "notes": [
            "last_cpu is the last CPU reported by /proc stat, not proof of affinity",
            "missing proc/sysfs reads are represented by value=null and an error",
            "the wrapper does not modify environment, affinity, priority, governors, or the command",
            "the CLI may re-exec with OMP_WAIT_POLICY=PASSIVE when it was absent from the inherited environment",
            "schedstat run_delay is only fully useful when sched_schedstats is enabled; zero or unavailable values do not prove no scheduling delay",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    interrupted = False
    with (output_dir / "runs.jsonl").open("w", encoding="utf-8") as runs_file:
        for run_index in range(1, args.runs + 1):
            result, interrupted = run_one(command, output_dir, run_index)
            runs_file.write(json.dumps(result, sort_keys=True) + "\n")
            runs_file.flush()
            if interrupted or result["status"] != "completed":
                break
    if interrupted:
        return 130
    if result["status"] == "completed":
        return 0
    if result["status"] == "spawn_failed":
        return 127
    if result["returncode"] is not None:
        if result["returncode"] < 0:
            return 128 + (-result["returncode"])
        return result["returncode"]
    return 1


if __name__ == "__main__":
    sys.exit(main())
