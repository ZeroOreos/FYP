#!/usr/bin/env python3
# python3 main_attack_slowed.py <gallery_dir> [throttle options] -> throttled main_attack.py orchestration

from __future__ import annotations

from pathlib import Path
from typing import Optional

import main_recognition_slowed as slowed_runner


def resolve_main_script_override(path_arg: Optional[str]) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "main_attack.py").resolve()


slowed_runner.resolve_main_script = resolve_main_script_override


if __name__ == "__main__":
    raise SystemExit(slowed_runner.main())
