#!/usr/bin/env python3
"""Create provenance for an existing checkpoint without modifying its weights/config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feedback_state.checkpoint_manifest import write_checkpoint_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Defaults to offline_data in train_config.json.",
    )
    args = parser.parse_args()

    train_data = args.train_data
    if train_data is None:
        config_path = args.checkpoint / "train_config.json"
        config = json.loads(config_path.read_text())
        configured = config.get("offline_data")
        if not configured:
            raise ValueError("train_config.json has no offline_data; pass --train-data")
        train_data = Path(str(configured))
    manifest = write_checkpoint_manifest(
        args.checkpoint,
        train_data,
        origin="backfill",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
