"""ClassEval, one method at a time (feedback_state.classeval): the pinned data, and a check that the verifier works here.

    python -m pipeline.classeval setup [--data data/classeval]     the benchmark at a pinned revision -> <data>/classes.jsonl,
                                                                    nltk's corpora -> <data>/nltk_data, manifest.json
    python -m pipeline.classeval check [--data data/classeval]     the reference solution against every method's tests and every
                                                                    class's tests -> <data>/reference_check.json; a method whose
                                                                    reference fails here cannot label candidates and is flagged

`setup` needs the network (Hugging Face and nltk's index; https_proxy is honoured, and nltk needs --allow-proxy to use one). `check` executes the benchmark's own reference
code and tests: FEEDBACK_CODE_EXEC_ALLOW=1, and `pip install -r requirements_classeval.txt` first.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _plain(value):
    """numpy arrays and scalars of the parquet file as plain JSON types."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def cmd_setup(args) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from feedback_state.classeval import NLTK_PACKAGES, PIN

    args.data.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(PIN["repo"], PIN["file"], repo_type="dataset", revision=PIN["revision"])
    df = pd.read_parquet(path)
    out = args.data / "classes.jsonl"
    with out.open("w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({c: _plain(row[c]) for c in df.columns}, ensure_ascii=False) + "\n")
    methods = int(df["methods_info"].map(len).sum())
    print(f"[classeval] {len(df)} classes, {methods} methods at {PIN['repo']}@{PIN['revision'][:10]} -> {out}")
    got = []
    if not args.no_nltk:
        target = (args.data / "nltk_data").resolve()
        target.mkdir(parents=True, exist_ok=True)
        os.environ["NLTK_DATA"] = str(target)    # nltk >= 3.9 only writes under a registered data path
        if args.allow_proxy:                     # ... and refuses to fetch through a proxy unless told the proxy is trusted
            os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
        import nltk

        for name in NLTK_PACKAGES:
            ok = nltk.download(name, download_dir=str(target), quiet=True)
            got.append(name) if ok else print(f"[classeval] nltk package {name} could not be downloaded")
        print(f"[classeval] nltk corpora {got} -> {target}")
    (args.data / "manifest.json").write_text(json.dumps({"kind": "classeval", **PIN, "classes": len(df), "methods": methods, "nltk": got,
                                                         "built": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def cmd_check(args) -> None:
    from feedback_state.classeval import class_result, load_classes, use_nltk_data

    use_nltk_data(args.data)
    classes = load_classes(args.data / "classes.jsonl")
    if args.limit:
        classes = classes[: args.limit]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda c: class_result(c, c["solution_code"]), classes))
    report = {"classes": len(classes), "class_pass": 0, "methods": 0, "method_pass": 0, "unusable_methods": [], "failing_classes": []}
    for c, r in zip(classes, results):
        report["class_pass"] += r["class_pass"]
        if not r["class_pass"]:
            report["failing_classes"].append(c["task_id"])
        for name, ok in r["methods"].items():
            report["methods"] += 1; report["method_pass"] += ok
            if not ok:
                report["unusable_methods"].append(f"{c['task_id']}::{name}")
    (args.data / "reference_check.json").write_text(json.dumps(report, indent=1))
    print(f"[classeval] reference solutions: {report['class_pass']}/{report['classes']} classes and {report['method_pass']}/{report['methods']} "
          f"methods pass their tests here ({time.time() - t0:.0f}s); unusable methods: {report['unusable_methods'][:12]}"
          f"{' ...' if len(report['unusable_methods']) > 12 else ''}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--data", type=Path, default=Path("data/classeval"))
    s.add_argument("--no-nltk", action="store_true")
    s.add_argument("--allow-proxy", action="store_true", help="let nltk download through the configured https_proxy (a proxy you trust)")
    c = sub.add_parser("check")
    c.add_argument("--data", type=Path, default=Path("data/classeval"))
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    {"setup": cmd_setup, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
