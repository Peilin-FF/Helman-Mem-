"""Compute label-free evidence lines for every peer answer of a stream (method B, stage 0).

For each record: one line per canonical peer (peer_0 .. peer_5), from feedback_state.evidence:
  code     the peer's program run on the sample I/O pairs printed in the statement (sandboxed, the
           same runner the grader uses, but on the SAMPLE cases only)
  math     arithmetic steps in the solution re-executed
  rag      answer span found in the passage or not
  others   arithmetic steps if any, else 'no check available'
Writes <out>: {id: {"peer_0": line, ...}} plus a diagnostic of how informative each check is against
the verified labels (which the model never sees; this is for us).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from feedback_state.data import JsonlDataset
from feedback_state.evidence import arithmetic_checks, extract_sample_io, line_for, span_in_context
from feedback_state.tasks import code_extract_answer, task_type_of


def _code_evidence(args) -> tuple[str, str, dict]:
    rid, key, program, cases, timeout = args
    from data.builders.common.code_grading import _run_io_case
    if not cases:
        return rid, key, {"kind": "code", "n_cases": 0}
    if not program.strip():
        return rid, key, {"kind": "code", "n_cases": len(cases), "passed": 0, "error": "no program"}
    passed, err = 0, ""
    for c in cases:
        r = _run_io_case(program, c["input"], c["output"], timeout)
        if r.passed:
            passed += 1
        elif r.error and not err and not r.error.startswith("wrong"):
            err = r.error.split("\n")[0][:60]
    return rid, key, {"kind": "code", "n_cases": len(cases), "passed": passed, "error": err if passed == 0 and err else ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    recs = JsonlDataset(args.records).records
    if args.limit:
        recs = recs[: args.limit]
    out, code_jobs, labels = {}, [], {}
    for r in recs:
        rid = str(r.get("id") or r.get("uid"))
        task = task_type_of(r)
        keys = sorted(r.get("peer_responses", {}))
        out[rid] = {}
        lab = r.get("correctness_by_peer") or {k: int(float(v) >= 0.5) for k, v in (r.get("peer_correct") or {}).items()}
        labels[rid] = {k: int(lab.get(k, -1)) for k in keys}
        if task == "code":
            cases = extract_sample_io(str(r.get("problem", ""))) if str(r.get("code_format", "io")) == "io" else []
            for k in keys:
                code_jobs.append((rid, k, code_extract_answer(str(r["peer_responses"][k])), cases, args.timeout))
            continue
        for k in keys:
            text = str(r["peer_responses"][k])
            if task == "rag":
                out[rid][k] = {"kind": "span", "found": span_in_context(text, r.get("context"))}
            else:
                n, bad = arithmetic_checks(text)
                out[rid][k] = {"kind": "arith", "n": n, "bad": bad} if (n or task == "math") else {"kind": "none"}
    print(f"{len(recs)} records; {len(code_jobs)} peer programs to run on sample tests ({time.time()-t0:.0f}s)", flush=True)
    if code_jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (rid, k, res) in enumerate(pool.map(_code_evidence, code_jobs, chunksize=4)):
                out[rid][k] = res
                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{len(code_jobs)} programs ({time.time()-t0:.0f}s)", flush=True)

    # render lines (canonical order; the prompt patchers reorder by peer_order) and a diagnostic
    lines, diag = {}, {}
    for rid, per in out.items():
        lines[rid] = {}
        for j, k in enumerate(sorted(per)):
            res = per[k]
            if res["kind"] == "span" and res.get("found") is None:
                lines[rid][k] = line_for(j + 1, "none")
            else:
                lines[rid][k] = line_for(j + 1, res["kind"], **{kk: v for kk, v in res.items() if kk != "kind"})
            # informativeness: does a 'good' verdict from the check line up with the label?
            good = None
            if res["kind"] == "code" and res.get("n_cases", 0) > 0:
                good = res["passed"] == res["n_cases"]
            elif res["kind"] == "arith" and res.get("n", 0) > 0:
                good = not res["bad"]
            elif res["kind"] == "span" and res.get("found") is not None:
                good = bool(res["found"])
            if good is not None and labels[rid].get(k, -1) >= 0:
                d = diag.setdefault(res["kind"], {"n": 0, "good_and_right": 0, "good": 0, "bad_and_right": 0, "bad": 0})
                d["n"] += 1
                if good:
                    d["good"] += 1; d["good_and_right"] += labels[rid][k]
                else:
                    d["bad"] += 1; d["bad_and_right"] += labels[rid][k]
    for kind, d in diag.items():
        pr_good = d["good_and_right"] / max(d["good"], 1); pr_bad = d["bad_and_right"] / max(d["bad"], 1)
        print(f"  {kind:<6} checks={d['n']:>6}  check passes: {d['good']:>6} of which right {100*pr_good:5.1f}%   check fails: {d['bad']:>6} of which right {100*pr_bad:5.1f}%")
    covered = sum(1 for rid in lines for k in lines[rid] if "no check available" not in lines[rid][k] and "no sample tests" not in lines[rid][k] and "no arithmetic steps" not in lines[rid][k])
    total = sum(len(v) for v in lines.values())
    print(f"  definitive evidence for {covered}/{total} peer answers ({100*covered/max(total,1):.1f}%)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"records": args.records, "n": len(recs), "lines": lines, "diagnostic": diag}, open(args.out, "w"))
    print(f"wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
