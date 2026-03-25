#!/usr/bin/env python3
# Shared CLI and pause-loop helpers for slowed main entry points.

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import psutil


def add_pause_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-seconds", type=int, default=480, help="Run time before a fixed pause. Default: 480s.")
    parser.add_argument("--cooldown-seconds", type=int, default=120, help="Base cooldown duration. Default: 120s.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval. Default: 2.0s.")
    parser.add_argument("--cpu-threshold", type=float, default=90.0, help="CPU threshold for an early pause. Default: 90.0.")
    parser.add_argument("--cpu-check-window", type=int, default=5, help="CPU samples to average. Default: 5.")
    parser.add_argument(
        "--extra-cooldown-seconds",
        type=int,
        default=90,
        help="Base cooldown when CPU exceeds the threshold. Default: 90s.",
    )
    parser.add_argument("--disable-load-trigger", action="store_true", help="Disable CPU-triggered pauses.")
    parser.add_argument(
        "--pause-all-stages",
        action="store_true",
        help="Pause all stages. By default, only model generate stages are paused.",
    )
    parser.add_argument(
        "--min-cooldown-seconds",
        type=int,
        default=20,
        help="Lower bound for adaptive cooldown duration. Default: 20s.",
    )
    parser.add_argument(
        "--max-cooldown-seconds",
        type=int,
        default=240,
        help="Upper bound for adaptive cooldown duration. Default: 240s.",
    )
    parser.add_argument(
        "--inter-section-cooldown-seconds",
        type=int,
        default=90,
        help="Extra cooldown between distinct generate sections. Default: 90s.",
    )
    parser.add_argument(
        "--disable-section-cooldown",
        action="store_true",
        help="Disable the extra cooldown between chained generate sections.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command and settings without running it.")


def resolve_main_script(path_arg: str | None, default_filename: str) -> Path:
    if path_arg:
        main_path = Path(path_arg).expanduser().resolve()
    else:
        main_path = (Path(__file__).resolve().parent.parent / default_filename).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"{default_filename} not found at: {main_path}")
    return main_path


def validate_input_dir(input_dir: str) -> Path:
    path = Path(input_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {path}")
    return path


def stop_process_tree(proc: psutil.Process) -> None:
    try:
        proc.send_signal(signal.SIGSTOP)
    except psutil.NoSuchProcess:
        return
    for child in proc.children(recursive=True):
        try:
            child.send_signal(signal.SIGSTOP)
        except psutil.NoSuchProcess:
            pass


def continue_process_tree(proc: psutil.Process) -> None:
    try:
        proc.send_signal(signal.SIGCONT)
    except psutil.NoSuchProcess:
        return
    for child in proc.children(recursive=True):
        try:
            child.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass


def proc_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline()).lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def is_generate_stage_process(proc: psutil.Process) -> bool:
    cmd = proc_cmdline(proc)
    return "generate.py" in cmd and "/models/" in cmd


def any_generate_stage_active(root_proc: psutil.Process) -> bool:
    if is_generate_stage_process(root_proc):
        return True
    return any(is_generate_stage_process(child) for child in root_proc.children(recursive=True))


def get_generate_stage_signatures(root_proc: psutil.Process) -> set[str]:
    signatures: set[str] = set()
    root_cmd = proc_cmdline(root_proc)
    if root_cmd and is_generate_stage_process(root_proc):
        signatures.add(root_cmd)
    for child in root_proc.children(recursive=True):
        cmd = proc_cmdline(child)
        if cmd and is_generate_stage_process(child):
            signatures.add(cmd)
    return signatures


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def adaptive_cooldown_seconds(args: argparse.Namespace, avg_cpu: float, load_triggered: bool) -> int:
    base = args.extra_cooldown_seconds if load_triggered else args.cooldown_seconds
    cpu_excess = max(0.0, avg_cpu - args.cpu_threshold)
    dynamic = int(base + cpu_excess * 0.8)
    if not load_triggered and avg_cpu < args.cpu_threshold * 0.75:
        dynamic = int(base * 0.6)
    return clamp(dynamic, args.min_cooldown_seconds, args.max_cooldown_seconds)


def sleep_with_progress(seconds: int, label: str) -> None:
    end = time.time() + seconds
    while True:
        remaining = int(round(end - time.time()))
        if remaining <= 0:
            print(f"\r{label}: done{' ' * 20}")
            break
        print(f"\r{label}: {remaining:>4}s remaining", end="", flush=True)
        time.sleep(1)


def prime_cpu_counters(proc: psutil.Process) -> None:
    try:
        proc.cpu_percent(interval=None)
        for child in proc.children(recursive=True):
            child.cpu_percent(interval=None)
    except psutil.Error:
        pass


def print_pause_plan(cmd: list[str], args: argparse.Namespace) -> None:
    print("\n[MODE] Process-level pausing")
    print(f"[INFO] Run slice:        {args.run_seconds}s")
    print(f"[INFO] Cooldown slice:   {args.cooldown_seconds}s")
    print(f"[INFO] Poll interval:    {args.poll_seconds}s")
    print(f"[INFO] CPU trigger:      {not args.disable_load_trigger}")
    print(f"[INFO] CPU threshold:    {args.cpu_threshold:.1f}%")
    print(f"[INFO] Stage-aware pause:{not args.pause_all_stages}")
    print(f"[INFO] Section cooldown: {not args.disable_section_cooldown} ({args.inter_section_cooldown_seconds}s)")
    print(f"[INFO] Command:          {' '.join(cmd)}")


def run_paused_subprocess(cmd: list[str], args: argparse.Namespace, *, main_label: str) -> int:
    print_pause_plan(cmd, args)
    if args.dry_run:
        return 0

    try:
        popen = subprocess.Popen(cmd)
    except Exception as exc:
        print(f"[ERROR] Failed to start {main_label}: {exc}", file=sys.stderr)
        return 1

    try:
        proc = psutil.Process(popen.pid)
    except psutil.Error as exc:
        print(f"[ERROR] Failed to attach to child process: {exc}", file=sys.stderr)
        popen.terminate()
        return 1

    print(f"[INFO] Started {main_label} with PID {popen.pid}")
    prime_cpu_counters(proc)

    run_start = time.time()
    cpu_samples: deque[float] = deque(maxlen=max(1, args.cpu_check_window))
    was_paused = False
    cycle_count = 0
    current_generate_session_key: tuple[str, ...] = tuple()
    seen_generate_sessions = 0

    try:
        while True:
            ret = popen.poll()
            if ret is not None:
                print(f"\n[INFO] {main_label} exited with code {ret}")
                return ret

            time.sleep(args.poll_seconds)

            total_cpu = 0.0
            try:
                total_cpu += proc.cpu_percent(interval=None)
                for child in proc.children(recursive=True):
                    try:
                        total_cpu += child.cpu_percent(interval=None)
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                ret = popen.poll()
                return 0 if ret is None else ret

            cpu_samples.append(total_cpu)
            elapsed_in_run = time.time() - run_start
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            generate_active = any_generate_stage_active(proc)
            generate_signatures = get_generate_stage_signatures(proc) if generate_active else set()
            generate_session_key = tuple(sorted(generate_signatures)) if generate_signatures else tuple()

            print(
                f"\r[MONITOR] cpu={total_cpu:6.1f}% | avg={avg_cpu:6.1f}% | "
                f"elapsed={int(elapsed_in_run):4d}s | cycles={cycle_count} | "
                f"stage={'generate' if generate_active else 'non-generate'}",
                end="",
                flush=True,
            )

            if generate_active and generate_session_key != current_generate_session_key:
                seen_generate_sessions += 1
                current_generate_session_key = generate_session_key

                should_cool_between_sections = (
                    seen_generate_sessions > 1
                    and not args.disable_section_cooldown
                    and args.inter_section_cooldown_seconds > 0
                )
                if should_cool_between_sections:
                    print("\n[INFO] Cooling between heavy generate sections.")
                    stop_process_tree(proc)
                    was_paused = True
                    sleep_with_progress(args.inter_section_cooldown_seconds, "[SECTION-COOLDOWN]")
                    print("[INFO] Resuming after section cooldown.")
                    continue_process_tree(proc)
                    was_paused = False
                    run_start = time.time()
                    cpu_samples.clear()
                    prime_cpu_counters(proc)
                    continue

            if not generate_active:
                current_generate_session_key = tuple()

            if not args.pause_all_stages and not generate_active:
                run_start = time.time()
                cpu_samples.clear()
                continue

            fixed_cycle_due = elapsed_in_run >= args.run_seconds
            load_cycle_due = (
                not args.disable_load_trigger
                and len(cpu_samples) == cpu_samples.maxlen
                and avg_cpu >= args.cpu_threshold
            )
            if not fixed_cycle_due and not load_cycle_due:
                continue

            reason_parts: list[str] = []
            if fixed_cycle_due:
                reason_parts.append("fixed pacing")
            if load_cycle_due:
                reason_parts.append("high CPU load")
            reason = " + ".join(reason_parts)
            cooldown_time = adaptive_cooldown_seconds(args, avg_cpu, load_cycle_due)

            print(f"\n[INFO] Pausing {main_label} due to {reason}.")
            print(f"[INFO] Adaptive cooldown: {cooldown_time}s")
            stop_process_tree(proc)
            was_paused = True
            cycle_count += 1
            sleep_with_progress(cooldown_time, "[COOLDOWN]")
            print(f"[INFO] Resuming {main_label}.")
            continue_process_tree(proc)
            was_paused = False
            run_start = time.time()
            cpu_samples.clear()
            prime_cpu_counters(proc)

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.")
        try:
            if was_paused:
                continue_process_tree(proc)
        except Exception:
            pass
        try:
            popen.terminate()
            popen.wait(timeout=10)
        except Exception:
            try:
                popen.kill()
            except Exception:
                pass
        return 130
