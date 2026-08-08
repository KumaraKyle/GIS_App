#!/usr/bin/env python3
"""Regenerate the discovery golden file.

Run this only when a payload change is *intended*. The whole value of the golden
file is that an unintended change fails the test, so regenerating to make a red
test go green defeats it -- read the diff first.

    python tests/golden/regen_discovery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_discovery_payloads import GOLDEN, generate  # noqa: E402


def main() -> int:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(generate(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} ({len(generate())} entities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
