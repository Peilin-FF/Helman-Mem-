#!/usr/bin/env python3
"""Validate one of the five exact published Sigma checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feedback_state.published_sigma import validate_published_sigma_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = validate_published_sigma_checkpoint(args.checkpoint)
    print(
        f"valid published Sigma checkpoint ({result.spec.name}): "
        f"state={result.state_sha256} config={result.config_sha256}"
    )


if __name__ == "__main__":
    main()
