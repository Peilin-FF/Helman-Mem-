"""Method B, stage 1: the central model asks for ONE check on the peer answers, a sandbox runs it, the result is put
into the reply, and only then does the model write its Trust line and answer (docs section 22).

The check is relational and label-free: it is run on every peer at once, from the problem statement and the peers'
own answers, never from the verifier.

    Check: input=<text>      code: the text is fed as standard input to every peer's program; each program's output
                             comes back, with which peers agree (a differential test: no expected output is needed)
    Check: python=<expr>     the expression is evaluated; its value comes back next to each peer's final answer
    Check: quote=<text>      whether the text occurs in the passage, and whether each peer's answer does
    </check>

Two passes of generation: pass 1 stops at ``</check>`` (or ends the reply without a check), the sandbox runs the
check, the ``Result:`` block is appended, pass 2 continues to the Trust line and the answer.  Inside the trainer the
result tokens are excluded from the loss (loss_mask); at test time the evaluator does the same two passes.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

SEEK_INSTRUCTION = (
    "Before deciding you may run ONE check on the peer answers. To use it, begin your reply with a single line of the form "
    "'Check: input=<text>' (the text is fed as standard input to every peer's program and each program's output comes back), "
    "'Check: python=<expression>' (the expression is evaluated and its value is shown next to each peer's final answer), or "
    "'Check: quote=<text>' (you learn whether the text occurs in the passage and whether each peer's answer does), then a line "
    "'</check>'. The result is inserted after it; only the first check is run, never write a second one or a result yourself. "
    "After that (or right away if you skip the check) write exactly one line of the form 'Trust: a > b > c > ...' ranking all "
    "the peers from most to least trustworthy for this question by their numbers, and then answer exactly as the instruction "
    "below requires."
)
CHECK_END = "</check>"
RESULT_HEADER = "Result:"
KINDS = ("input", "python", "quote")
_CHECK = re.compile(r"^\s*Check:\s*(input|python|quote)\s*=\s*(.*?)\s*\n\s*</check>", re.S | re.I)
MAX_RESULT_CHARS = 1400
PER_PEER_CHARS = 80


def add_seek_instruction(messages: list[dict]) -> list[dict]:
    """Append the check + Trust-line instruction to the user turn (returns a new list)."""
    out = [dict(m) for m in messages]
    user = out[-1]
    if user.get("role") != "user" or SEEK_INSTRUCTION in user["content"]:
        return out
    c = user["content"]
    k = c.rfind("\n\nInstruction:")   # before the task instruction, which must stay last (a long trailing text displaces it: code answers vanish)
    user["content"] = (c[:k] + "\n\n" + SEEK_INSTRUCTION + c[k:]) if k >= 0 else (c.rstrip() + "\n\n" + SEEK_INSTRUCTION)
    return out


def parse_check(text: str) -> dict | None:
    """{'kind': 'input'|'python'|'quote', 'payload': str} from the start of a reply, or None."""
    m = _CHECK.search(text or "")
    if not m or not m.group(2).strip():
        return None
    return {"kind": m.group(1).lower(), "payload": m.group(2)}


def has_result(text: str) -> bool:
    return f"\n{RESULT_HEADER}\n" in (text or "") or (text or "").startswith(RESULT_HEADER + "\n")


# ---------------------------------------------------------------- the sandbox
def _run_program(program: str, stdin: str, timeout: float) -> tuple[str, str]:
    """(stdout, error) of a program on the given stdin, under the grader's resource limits."""
    if os.environ.get("FEEDBACK_CODE_EXEC_ALLOW") != "1":
        return "", "code execution is disabled (FEEDBACK_CODE_EXEC_ALLOW)"
    from data.builders.common.code_grading import DEFAULT_MEM_MB, _build_runner
    import signal
    source = _build_runner(program, mem_mb=DEFAULT_MEM_MB, cpu_s=int(timeout) + 1)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "prog.py"
        script.write_text(source)
        proc = subprocess.Popen([sys.executable, "-I", str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=tmp, start_new_session=True)   # its own process group: a timeout kills the program and anything it spawned
        try:
            out, err = proc.communicate(stdin.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            return "", "timeout"
    if proc.returncode != 0:
        lines = (err or b"").decode(errors="replace").strip().splitlines()
        return "", (lines[-1] if lines else f"exit {proc.returncode}")[:PER_PEER_CHARS]
    return (out or b"").decode(errors="replace").strip(), ""


def _short(s: str, n: int = PER_PEER_CHARS) -> str:
    s = " | ".join(line.strip() for line in str(s).splitlines() if line.strip())
    return s if len(s) <= n else s[: n - 3] + "..."


def _agreement(values: Sequence[str | None]) -> str:
    groups: dict[str, list[int]] = {}
    for j, v in enumerate(values):
        if v is not None:
            groups.setdefault(v, []).append(j + 1)
    parts = [f"peers {', '.join(map(str, g))}" if len(g) > 1 else f"peer {g[0]} alone" for g in sorted(groups.values(), key=lambda g: (-len(g), g))]
    return "Same output: " + "; ".join(parts) if parts else "No output from any peer"


def run_check(record: dict[str, Any], peer_texts: Sequence[str], check: dict, *, timeout: float = 5.0) -> str:
    """The result block for one check (without the trailing newline)."""
    from feedback_state.tasks import _rag_context_text, code_extract_answer, qa_extract_answer
    from feedback_state.utils import extract_final_answer, math_equal
    kind, payload = check["kind"], check["payload"]
    lines = [RESULT_HEADER]
    if kind == "input":
        programs = [code_extract_answer(str(t)) for t in peer_texts]
        if not any(p.strip() for p in programs):
            lines.append("No peer answer contains a program to run.")
        else:
            with ThreadPoolExecutor(max_workers=min(6, len(programs))) as ex:
                runs = list(ex.map(lambda p: _run_program(p, payload, timeout) if p.strip() else ("", "no program"), programs))
            outs = []
            for j, (out, err) in enumerate(runs):
                if err:
                    lines.append(f"Peer {j + 1}: {'no program' if err == 'no program' else 'error: ' + _short(err)}")
                    outs.append(None)
                else:
                    lines.append(f"Peer {j + 1}: output {_short(out)!r}")
                    outs.append(out.strip())
            lines.append(_agreement(outs))
    elif kind == "python":
        out, err = _run_program("import math\nfrom fractions import Fraction\nprint(" + payload.strip() + ")", "", timeout)
        if err:
            lines.append(f"Expression failed: {_short(err)}")
        else:
            lines.append(f"Value: {_short(out)}")
            for j, t in enumerate(peer_texts):
                a = extract_final_answer(str(t))
                same = bool(a) and len(out) <= 200 and len(a) <= 200 and math_equal(a, out)
                lines.append(f"Peer {j + 1}: final answer {_short(a, 40)!r}" + (" = value" if same else " (different)" if a else ""))
    elif kind == "quote":
        passage = _rag_context_text(record) if record.get("context") else ""
        if not passage.strip():
            lines.append("No passage for this task.")
        else:
            norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()   # noqa: E731
            lines.append("Quote found in the passage." if norm(payload) and norm(payload) in norm(passage) else "Quote NOT found in the passage.")
            for j, t in enumerate(peer_texts):
                a = qa_extract_answer(str(t))
                lines.append(f"Peer {j + 1}: answer {_short(a, 40)!r} " + ("found in the passage" if a and norm(a) in norm(passage) else "not found in the passage"))
    text = "\n".join(lines)
    return text if len(text) <= MAX_RESULT_CHARS else text[: MAX_RESULT_CHARS - 3] + "..."


def check_task(record_json: str, peer_texts: list[str], check: dict, timeout: float) -> str:
    """One check in a fresh process (a Ray task or a spawned pool worker); never raises."""
    import json
    try:
        return run_check(json.loads(record_json), peer_texts, check, timeout=timeout)
    except Exception as e:   # a broken check must not break the rollout
        return f"{RESULT_HEADER}\nThe check could not be run ({type(e).__name__})."


FAILED_RESULT = f"{RESULT_HEADER}\nThe check could not be run (time limit)."
_remote_check = None


def run_checks(record_jsons: Sequence[str | None], peer_texts: Sequence[Sequence[str] | None], checks: Sequence[dict | None], *,
               timeout: float = 5.0, pool=None, hard_timeout: float = 90.0) -> list[str | None]:
    """Result blocks for many samples at once (None where there is no check).

    The checks never run in the calling process: it hosts the inference engine and has a hundred threads, and a fork of
    such a process can deadlock before exec.  Inside the trainer (Ray initialised) each check is a Ray task in a fresh
    worker, like the code grader; the evaluator passes a ProcessPoolExecutor spawned before its engine started.
    """
    results: list[str | None] = [None] * len(checks)
    idx = [i for i, c in enumerate(checks) if c is not None and record_jsons[i] is not None and peer_texts[i] is not None]
    if not idx:
        return results
    args = [(record_jsons[i], list(peer_texts[i]), checks[i], timeout) for i in idx]
    use_ray = False
    if pool is None:
        try:
            import ray
            use_ray = ray.is_initialized()
        except Exception:
            use_ray = False
    if use_ray:
        global _remote_check
        if _remote_check is None:
            _remote_check = ray.remote(num_cpus=0.5)(check_task)
        refs = [_remote_check.remote(*a) for a in args]
        ready, pending = ray.wait(refs, num_returns=len(refs), timeout=hard_timeout)
        done = {r: v for r, v in zip(ready, ray.get(ready))}
        for r in pending:
            ray.cancel(r, force=True)
        for i, r in zip(idx, refs):
            results[i] = done.get(r, FAILED_RESULT)
    elif pool is not None:
        futs = [pool.submit(check_task, *a) for a in args]
        for i, f in zip(idx, futs):
            try:
                results[i] = f.result(timeout=hard_timeout)
            except Exception:
                results[i] = FAILED_RESULT
    else:   # plain process (tests, scripts): threads + subprocesses are fine there
        with ThreadPoolExecutor(max_workers=16) as ex:
            for i, v in zip(idx, ex.map(lambda a: check_task(*a), args)):
                results[i] = v
    return results


# ---------------------------------------------------------------- the two passes with vLLM
def two_pass_generate(engine, vllm_inputs: list[dict], base_params, infos: Sequence[dict | None], tokenizer, *, n: int,
                      response_length: int, check_tokens: int = 160, result_tokens: int = 352, timeout: float = 5.0,
                      lora_request=None, min_answer_tokens: int = 64, pool=None) -> tuple[list[list[int]], list[list[int]], list[list[float]], dict]:
    """Pass 1 (n samples per prompt, stop at </check>), the checks, pass 2 (one continuation per sample).

    Returns per sample (prompt-major order): response token ids, loss mask (0 on the result tokens), the rollout
    log-probs (0 on the result tokens), and statistics.
    """
    import copy
    import json

    from feedback_state.memory_rl import peer_texts_in_prompt_order

    p1 = copy.deepcopy(base_params)
    p1.n, p1.max_tokens, p1.stop, p1.include_stop_str_in_output, p1.detokenize = n, min(check_tokens, response_length), [CHECK_END], True, True
    outs = engine.generate(prompts=vllm_inputs, sampling_params=p1, lora_request=lora_request, use_tqdm=False)
    owner, r1, r1_lp, kinds, checks, recs, peers = [], [], [], [], [], [], []
    for i, out in enumerate(outs):
        info = infos[i] if infos is not None and i < len(infos) else None
        rec_json = info["record"] if isinstance(info, dict) and isinstance(info.get("record"), str) else None
        rec = json.loads(rec_json) if rec_json is not None else None
        texts = peer_texts_in_prompt_order(rec, info["peer_order"]) if (rec is not None and info.get("peer_order") is not None) else None
        for o in out.outputs:
            ids = list(o.token_ids)
            owner.append(i); r1.append(ids)
            r1_lp.append([lp[t].logprob for t, lp in zip(ids, o.logprobs)] if o.logprobs else [0.0] * len(ids))
            stopped = o.finish_reason == "stop" and o.stop_reason == CHECK_END
            chk = parse_check(tokenizer.decode(ids)) if stopped else None
            usable = chk is not None and rec is not None and texts is not None and str(info.get("prompt_source", "peers")) != "solo"
            checks.append(chk if usable else None); kinds.append((chk or {}).get("kind") if usable else None); recs.append(rec_json); peers.append(texts)
            # a reply cut by the pass-1 budget without a check continues in pass 2 with no result block
            kinds[-1] = kinds[-1] or ("length" if o.finish_reason == "length" else "eos")
    results = run_checks(recs, peers, checks, timeout=timeout, pool=pool)
    cont_idx, cont_inputs, cont_params, result_ids = [], [], [], [None] * len(r1)
    for s in range(len(r1)):
        needs = results[s] is not None or kinds[s] == "length"
        if not needs:
            continue
        rid = []
        if results[s] is not None:
            rid = tokenizer.encode("\n" + results[s] + "\n", add_special_tokens=False)[: result_tokens]
            rid = rid[: max(0, response_length - len(r1[s]) - min_answer_tokens)]
        room = response_length - len(r1[s]) - len(rid)
        if room <= 0:
            continue
        result_ids[s] = rid
        p2 = copy.deepcopy(base_params)
        p2.n, p2.max_tokens, p2.stop, p2.include_stop_str_in_output, p2.detokenize = 1, room, None, False, False
        cont_idx.append(s); cont_inputs.append({"prompt_token_ids": list(vllm_inputs[owner[s]]["prompt_token_ids"]) + r1[s] + rid}); cont_params.append(p2)
    r2, r2_lp = {}, {}
    if cont_inputs:
        outs2 = engine.generate(prompts=cont_inputs, sampling_params=cont_params, lora_request=lora_request, use_tqdm=False)
        for s, out in zip(cont_idx, outs2):
            o = out.outputs[0]
            r2[s] = list(o.token_ids); r2_lp[s] = [lp[t].logprob for t, lp in zip(r2[s], o.logprobs)] if o.logprobs else [0.0] * len(r2[s])
    responses, masks, logps = [], [], []
    for s in range(len(r1)):
        rid = result_ids[s] or []
        ids = r1[s] + rid + r2.get(s, [])
        responses.append(ids[:response_length])
        masks.append(([1] * len(r1[s]) + [0] * len(rid) + [1] * len(r2.get(s, [])))[:response_length])
        logps.append((r1_lp[s] + [0.0] * len(rid) + r2_lp.get(s, []))[:response_length])
    stats = {"samples": len(r1), "checks_run": sum(1 for r in results if r is not None),
             "checks_by_kind": {k: sum(1 for x in kinds if x == k) for k in KINDS},
             "continued_without_check": sum(1 for s in cont_idx if results[s] is None)}
    return responses, masks, logps, stats
