"""Release the six-peer streams to the public dataset repository (Hugging Face: Sssunset/Sigma-Mem-Data, linked from the
GitHub README).  The three streams behind every result since 2026-09-07:

    data/mixed_train_big6/train.jsonl   17,709 events  GSM8K + SQuAD + APPS, six peers      (the training stream)
    data/indist6/test.jsonl              4,319 events  GSM8K + SQuAD + APPS test sets        (in-distribution stream)
    data/ood6/test.jsonl                17,403 events  PIQA, MMLU, OpenBookQA, SciQ, BBH, SuperGLUE (the OOD stream)

    PYTHONPATH=. python scripts/release_dataset.py --stage outputs/release/Sigma-Mem-Data    # export + card + manifest, no upload
    PYTHONPATH=. python scripts/release_dataset.py --stage ... --upload                       # then push to the Hub (cached login)

Export: the records are copied verbatim except that peer_metadata[*].model, which holds local paths in some streams, is
rewritten to the Hugging Face model id.  The existing dataset card and manifest are downloaded from the Hub and
extended (idempotent: re-running replaces the six-peer section), so the earlier three-peer release stays intact.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

REPO = "Sssunset/Sigma-Mem-Data"
STREAMS = [   # (path in the release = path under data/, config name, split, description)
    ("mixed_train_big6/train.jsonl", "six_peer_train", "train", "Training stream: GSM8K, SQuAD and APPS with six peers"),
    ("indist6/test.jsonl", "six_peer_indist", "test", "In-distribution test stream: GSM8K test, SQuAD dev and APPS test with six peers"),
    ("ood6/test.jsonl", "six_peer_ood", "test", "OOD test stream: PIQA, MMLU, OpenBookQA, SciQ, BBH and SuperGLUE with six peers"),
]
HF_IDS = {"gemma-3-4b-it": "google/gemma-3-4b-it", "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
          "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct", "Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
          "DeepSeek-Coder-V2-Lite-Instruct": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
SECTION = "## Six-Peer Streams (version 3, September 2026)"


def hf_id(model: str) -> str:
    name = model.rstrip("/").split("/")[-1]
    return HF_IDS.get(name, model if ("/" in model and not model.startswith("/")) else name)


def export(src: Path, dst: Path) -> dict:
    """Copy a stream, normalising the peer model ids; returns the manifest entry."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows, sources, peers, tasks = 0, collections.Counter(), {}, collections.Counter()
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            for k, meta in (r.get("peer_metadata") or {}).items():
                if isinstance(meta, dict) and meta.get("model"):
                    meta["model"] = hf_id(str(meta["model"]))
                    peers.setdefault(k, meta["model"])
            rows += 1
            sources[str(r.get("source"))] += 1
            tasks[str(r.get("task_type"))] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256()
    with dst.open("rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(dst.relative_to(dst.parents[1])), "bytes": dst.stat().st_size, "sha256": digest.hexdigest(), "rows": rows,
            "sources": dict(sorted(sources.items())), "task_types": dict(sorted(tasks.items())), "peer_counts": {"6": rows},
            "peers": dict(sorted(peers.items()))}


def card_text(old: str, entries: list[dict]) -> str:
    """The dataset card with the six-peer configs in the front matter and a six-peer section in the body."""
    head, body = old.split("\n---\n", 1) if old.startswith("---") else ("---\npretty_name: Sigma-Mem Data", old)
    head = re.sub(r"\n- config_name: six_peer_\w+\n  data_files:\n  - split: \w+\n    path: [^\n]+", "", head)
    for (path, cfg, split, _), _e in zip(STREAMS, entries):
        head += f"\n- config_name: {cfg}\n  data_files:\n  - split: {split}\n    path: {path}"
    if SECTION in body:   # replace the earlier six-peer section
        body = re.sub(re.escape(SECTION) + r".*?(?=\n## |\Z)", "@@SIX@@", body, flags=re.S)
    peers = entries[0]["peers"]
    rows = "\n".join(f"| `{cfg}` | `{path}` | {e['rows']:,} | {desc} |" for (path, cfg, _s, desc), e in zip(STREAMS, entries))
    mapping = "\n".join(f"| `{k}` | `{v}` |" for k, v in peers.items())
    sec = f"""{SECTION}

Version 3 adds the three streams behind the reliability-memory experiments of September 2026 (the Bayesian record read
before write along the stream, the attention tilt on the central model, the judgement work). Every event carries six
independently generated peer answers and their externally evaluated correctness; the labels are post-decision feedback,
exactly as in the three-peer release.

| Configuration | File | Events | Content |
| --- | --- | ---: | --- |
{rows}

The event order of each file is the release order; the evaluation streams are read once, cold start, with the record
written only after each event's answer. The in-distribution test stream keeps the APPS hidden tests (`test_cases`) so
the code events can be graded locally; the OOD stream has `choices` / `choice_labels` for the multiple-choice tasks.
The six peers of these streams (keys `peer_0` .. `peer_5`; note that `peer_3` and `peer_4` differ from the
counterfactual streams' mapping above):

| Key | Model |
| --- | --- |
{mapping}

Peer answers were generated once with vLLM (temperature 0.2, top-p 0.95; 768 new tokens on the training and
in-distribution streams, 96 on OOD; the reasoning peer's answer is the text after its think block) and graded with the
verifier of the corresponding task (exact match after answer extraction for math, F1 ≥ 0.5 for reading, the hidden
tests for code, option match for OOD). Central-model results on these streams: `docs/memory_judge_design.md` and the
result pages in the GitHub repository.
"""
    body = body.replace("@@SIX@@", sec) if "@@SIX@@" in body else (body.rstrip() + "\n\n" + sec)
    return head + "\n---\n" + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--stage", type=Path, default=Path("outputs/release/Sigma-Mem-Data"))
    ap.add_argument("--upload", action="store_true", help="push the staged files to the Hub (needs a cached login with write access)")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    args.stage.mkdir(parents=True, exist_ok=True)
    entries = []
    for path, *_ in STREAMS:
        src, dst = args.data / path, args.stage / path
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[release] {dst} exists; recomputing its manifest entry", flush=True)
        e = export(src, dst)
        entries.append(e)
        print(f"[release] {path}: {e['rows']:,} rows, {e['bytes'] / 1e6:.0f} MB, sources {e['sources']}", flush=True)
    old_card = Path(hf_hub_download(args.repo, "README.md", repo_type="dataset")).read_text()
    manifest = json.load(open(hf_hub_download(args.repo, "manifest.json", repo_type="dataset")))
    keep = [f for f in manifest.get("files", []) if not any(f["path"] == e["path"] for e in entries)]
    manifest["files"] = keep + entries
    manifest["version"] = max(int(manifest.get("version", 1)), 3)
    manifest.setdefault("versions", {})["3"] = "2026-09: six-peer streams mixed_train_big6, indist6, ood6 added"
    (args.stage / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (args.stage / "README.md").write_text(card_text(old_card, entries))
    print(f"[release] staged card ({(args.stage / 'README.md').stat().st_size} bytes) and manifest (version {manifest['version']}, {len(manifest['files'])} files) in {args.stage}", flush=True)
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"[release] uploading to {args.repo} as {api.whoami()['name']}", flush=True)
        info = api.upload_folder(folder_path=str(args.stage), repo_id=args.repo, repo_type="dataset",
                                 commit_message="Version 3: six-peer streams (mixed_train_big6, indist6, ood6)")
        print(f"[release] uploaded: {info}", flush=True)


if __name__ == "__main__":
    main()
