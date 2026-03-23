#!/usr/bin/env python3
"""
main_slowed.py — Thermal-aware orchestrator for FYP verification pipeline.

RECOMMENDED: Use --plugin-throttle to apply in-process throttling via batch_size/batch_delay.
FALLBACK:   Use --pause-generate to only SIGSTOP the embedding generation stages.

Key insight: Embedding generation is ~90% of heat; pairs and evaluate are lightweight.
- --plugin-throttle: Reduce batch_size + add inter-batch delays (no state loss, smooth)
- --pause-generate:  Only SIGSTOP during generate.py runs (process-level, targeted)

Usage examples:
    python3 main_slowed.py /path/to/dataset --plugin-throttle
    python3 main_slowed.py /path/to/dataset --pause-generate --run-seconds 420 --cooldown-seconds 120
    python3 main_slowed.py /path/to/dataset --pause-generate --cpu-threshold 85 --disable-load-trigger

Behavior:
- main.py orchestrates 4 stages: pairs → generate → evaluate → compile CSV
- If --plugin-throttle: passes --throttle to main.py (preferred baseline)
- If --pause-generate: defaults to stage-aware pausing only during generate.py
- If --pause-generate: also passes --throttle unless --disable-main-throttle is set
- If --pause-generate + high CPU: applies extra cooldown early

Why smarter than naive SIGSTOP:
- Pairs generation runs per-dataset once (fast)
- Evaluation is threshold-tuning, not I/O-bound (fast)
- Embedding generation is the GPU/CPU hotspot (needs throttling)
- Pausing layers 2-3 wastes time; layers 1 & 4 don't need it

Notes:
- macOS-specific SIGSTOP/SIGCONT require Unix-like OS
- In-process throttling is preferred: no pause overhead, no cache loss
- For maximum control, use --plugin-throttle (main.py --throttle) before pause mode
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

try:
    import psutil
except ImportError:
    print(
        "[ERROR] psutil is required for main_slowed.py.\n"
        "Install it with:\n"
        "    python3 -m pip install psutil",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main.py with thermal-aware control: plugin throttling or process-level pause.",
        epilog="RECOMMENDED: Use --plugin-throttle. FALLBACK: Use --pause-generate."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input dataset directory.",
    )

    parser.add_argument(
        "--main-script",
        type=str,
        default=None,
        help="Path to main.py. Defaults to main.py in the same folder as this script.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to use. Default: current interpreter.",
    )

    throttle_group = parser.add_mutually_exclusive_group(required=True)
    throttle_group.add_argument(
        "--plugin-throttle",
        action="store_true",
        help="Use in-process throttling: batch_size + inter-batch delays (recommended).",
    )
    throttle_group.add_argument(
        "--pause-generate",
        action="store_true",
        help="Only SIGSTOP during generate.py stages, skip pairs/evaluate.",
    )

    parser.add_argument(
        "--run-seconds",
        type=int,
        default=480,
        help="Duration to let process run before fixed pause. Only used with --pause-generate. Default: 480s.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=120,
        help="Duration to pause at each cycle. Only used with --pause-generate. Default: 120s.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval. Only used with --pause-generate. Default: 2.0s.",
    )

    parser.add_argument(
        "--cpu-threshold",
        type=float,
        default=90.0,
        help="CPU threshold (%) for early pause. Only used with --pause-generate. Default: 90.0.",
    )
    parser.add_argument(
        "--cpu-check-window",
        type=int,
        default=5,
        help="CPU samples to average. Only used with --pause-generate. Default: 5.",
    )
    parser.add_argument(
        "--extra-cooldown-seconds",
        type=int,
        default=90,
        help="Extra pause when CPU exceeds threshold. Only used with --pause-generate. Default: 90s.",
    )

    parser.add_argument(
        "--disable-load-trigger",
        action="store_true",
        help="Disable early pause based on CPU. Only used with --pause-generate.",
    )

    parser.add_argument(
        "--pause-all-stages",
        action="store_true",
        help="Pause regardless of stage. By default, pauses only while generate.py is active.",
    )
    parser.add_argument(
        "--disable-main-throttle",
        action="store_true",
        help="In --pause-generate mode, do not pass --throttle to main.py.",
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
        help="Extra cooldown inserted between distinct generate.py sections. Default: 90s.",
    )
    parser.add_argument(
        "--disable-section-cooldown",
        action="store_true",
        help="Disable extra cooldown between chained generate.py sections.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and settings without running main.py.",
    )

    return parser.parse_args()


def resolve_main_script(path_arg: str | None) -> Path:
    if path_arg:
        main_path = Path(path_arg).expanduser().resolve()
    else:
        main_path = (Path(__file__).resolve().parent / "main.py").resolve()

    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found at: {main_path}")
    return main_path


def validate_input_dir(input_dir: str) -> Path:
    p = Path(input_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {p}")
    return p


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

    for child in root_proc.children(recursive=True):
        if is_generate_stage_process(child):
            return True

    return False


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
    start = time.time()
    end = start + seconds
    while True:
        remaining = int(round(end - time.time()))
        if remaining <= 0:
            print(f"\r{label}: done{' ' * 20}")
            break
        print(f"\r{label}: {remaining:>4}s remaining", end="", flush=True)
        time.sleep(1)


def main() -> int:
    args = parse_args()

    try:
        input_dir = validate_input_dir(args.input_dir)
        main_script = resolve_main_script(args.main_script)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[INFO] === main_slowed.py thermal-aware orchestrator ===")
    print(f"[INFO] Input directory:  {input_dir}")
    print(f"[INFO] Main script:      {main_script}")
    print(f"[INFO] Python binary:    {args.python_bin}")

    if args.plugin_throttle:
        return run_with_plugin_throttle(main_script, input_dir, args)
    else:
        return run_with_pause_generate(main_script, input_dir, args)


def run_with_plugin_throttle(main_script: Path, input_dir: Path, args: argparse.Namespace) -> int:
    cmd = [args.python_bin, str(main_script), str(input_dir), "--throttle"]

    print("\n[MODE] Plugin-level throttling (RECOMMENDED)")
    print("[INFO] Passing --throttle to main.py")
    print(f"[INFO] Command: {' '.join(cmd)}")

    if args.dry_run:
        return 0

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as exc:
        print(f"[ERROR] Failed to run main.py: {exc}", file=sys.stderr)
        return 1


def run_with_pause_generate(main_script: Path, input_dir: Path, args: argparse.Namespace) -> int:
    cmd = [args.python_bin, str(main_script), str(input_dir)]
    if not args.disable_main_throttle:
        cmd.append("--throttle")

    print("\n[MODE] Process-level pausing (FALLBACK)")
    print(f"[INFO] Run slice:        {args.run_seconds}s")
    print(f"[INFO] Cooldown slice:   {args.cooldown_seconds}s")
    print(f"[INFO] Poll interval:    {args.poll_seconds}s")
    print(f"[INFO] CPU trigger:      {not args.disable_load_trigger}")
    print(f"[INFO] CPU threshold:    {args.cpu_threshold:.1f}%")
    print(f"[INFO] Stage-aware pause:{not args.pause_all_stages}")
    print(f"[INFO] Main throttle:    {not args.disable_main_throttle}")
    print(f"[INFO] Section cooldown: {not args.disable_section_cooldown} ({args.inter_section_cooldown_seconds}s)")
    print(f"[INFO] Command:          {' '.join(cmd)}")

    if args.dry_run:
        return 0

    try:
        popen = subprocess.Popen(cmd)
    except Exception as exc:
        print(f"[ERROR] Failed to start main.py: {exc}", file=sys.stderr)
        return 1

    try:
        proc = psutil.Process(popen.pid)
    except psutil.Error as exc:
        print(f"[ERROR] Failed to attach to child process: {exc}", file=sys.stderr)
        popen.terminate()
        return 1

    print(f"[INFO] Started main.py with PID {popen.pid}")

    try:
        proc.cpu_percent(interval=None)
        for child in proc.children(recursive=True):
            child.cpu_percent(interval=None)
    except psutil.Error:
        pass

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
                print(f"\n[INFO] main.py exited with code {ret}")
                return ret

            time.sleep(args.poll_seconds)

            total_cpu = 0.0
            try:
                total_cpu += proc.cpu_percent(interval=None)
                children = proc.children(recursive=True)
                for child in children:
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

                if (
                    seen_generate_sessions > 1
                    and not args.disable_section_cooldown
                    and args.inter_section_cooldown_seconds > 0
                ):
                    print("\n[INFO] Cooling between heavy generate sections.")
                    stop_process_tree(proc)
                    was_paused = True

                    sleep_with_progress(args.inter_section_cooldown_seconds, "[SECTION-COOLDOWN]")

                    print("[INFO] Resuming after section cooldown.")
                    continue_process_tree(proc)
                    was_paused = False

                    run_start = time.time()
                    cpu_samples.clear()

                    try:
                        proc.cpu_percent(interval=None)
                        for child in proc.children(recursive=True):
                            child.cpu_percent(interval=None)
                    except psutil.Error:
                        pass

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

            if fixed_cycle_due or load_cycle_due:
                reason = "fixed pacing"
                cooldown_time = args.cooldown_seconds

                if load_cycle_due and not fixed_cycle_due:
                    reason = "high CPU load"
                    cooldown_time = args.extra_cooldown_seconds

                cooldown_time = adaptive_cooldown_seconds(args, avg_cpu, load_cycle_due)

                print(f"\n[INFO] Pausing main.py due to {reason}.")
                print(f"[INFO] Adaptive cooldown: {cooldown_time}s")
                stop_process_tree(proc)
                was_paused = True
                cycle_count += 1

                sleep_with_progress(cooldown_time, "[COOLDOWN]")

                print("[INFO] Resuming main.py.")
                continue_process_tree(proc)
                was_paused = False

                run_start = time.time()
                cpu_samples.clear()

                try:
                    proc.cpu_percent(interval=None)
                    for child in proc.children(recursive=True):
                        child.cpu_percent(interval=None)
                except psutil.Error:
                    pass

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


if __name__ == "__main__":
    raise SystemExit(main())
