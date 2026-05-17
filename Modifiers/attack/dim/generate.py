#!/usr/bin/env python3
# Example: python Modifiers/attack/dim/generate.py --help
# Canonical DiM wrapper entrypoint.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modifiers.attack.dim.wrapper import main


if __name__ == "__main__":
    main()
