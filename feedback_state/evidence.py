"""Label-free evidence about the peers' answers (method B, stage 0 of the judgement work; docs section 22).

Every check here uses only the problem statement and the peer's own answer, never the hidden verifier:

  code      run the peer's program on the SAMPLE input/output pairs printed in the statement
  math      re-execute every arithmetic step written in the peer's solution
  reading   whether the peer's answer text occurs in the passage
  others    arithmetic steps if the answer contains any; otherwise no check

The result for each peer is one plain line, inserted into the prompt as an "Evidence" block after the
peer answers and before the instruction, so that the peer blocks (and the tilt on them) are untouched.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

EVIDENCE_HEADER = "Evidence (automatic checks on the peer answers; no answer key was used):"

# ---------------------------------------------------------------- sample I/O in a statement
_SAMPLE_BLOCK = re.compile(
    r"-{2,}\s*(?:Sample|Example)\s*Input\s*(?:\d+)?\s*:?\s*-{2,}\s*\n(.*?)\n-{2,}\s*(?:Sample|Example)\s*Output\s*(?:\d+)?\s*:?\s*-{2,}\s*\n(.*?)(?=\n-{2,}|\Z)",
    re.S | re.I)
_EXAMPLE_BLOCK = re.compile(
    r"-{2,}\s*Examples?\s*-{2,}\s*\n(.*?)(?=\n-{2,}|\Z)", re.S | re.I)
_IO_PAIR = re.compile(r"Input\s*\n(.*?)\nOutput\s*\n(.*?)(?=\nInput\s*\n|\Z)", re.S)


def extract_sample_io(statement: str) -> list[dict]:
    """[{'input': str, 'output': str}] from the statement's sample blocks; [] if none are printed."""
    cases = []
    for inp, out in _SAMPLE_BLOCK.findall(statement or ""):
        if inp.strip() and out.strip():
            cases.append({"input": inp.strip("\n"), "output": out.strip("\n")})
    if not cases:
        m = _EXAMPLE_BLOCK.search(statement or "")
        if m:
            for inp, out in _IO_PAIR.findall(m.group(1)):
                if inp.strip() and out.strip():
                    cases.append({"input": inp.strip("\n"), "output": out.strip("\n")})
    return cases


# ---------------------------------------------------------------- arithmetic steps in a solution
_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_EQ = re.compile(rf"({_NUM}(?:\s*[-+*/x×÷]\s*{_NUM})+)\s*=\s*({_NUM})(?!\d|\.\d)")


def _to_float(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


def _eval_chain(expr: str) -> float | None:
    toks = re.findall(rf"{_NUM}|[-+*/x×÷]", expr)
    if not toks:
        return None
    try:
        val = _to_float(toks[0])
        i = 1
        while i + 1 < len(toks):
            op, b = toks[i], _to_float(toks[i + 1])
            if op in "x×*":
                val *= b
            elif op in "/÷":
                if b == 0:
                    return None
                val /= b
            elif op == "+":
                val += b
            elif op == "-":
                val -= b
            i += 2
        return val
    except (ValueError, ZeroDivisionError):
        return None


def arithmetic_checks(text: str) -> tuple[int, list[str]]:
    """(number of arithmetic steps found, list of failing steps written as 'a op b = c is wrong (d)')."""
    n, bad = 0, []
    for lhs, rhs in _EQ.findall(text or ""):
        got, want = _eval_chain(lhs), _to_float(rhs)
        if got is None:
            continue
        n += 1
        tol = 1e-6 * max(1.0, abs(want))
        if abs(got - want) > tol:
            g = f"{got:g}"
            bad.append(f"{lhs.strip()} = {rhs.strip()} is wrong ({g})")
    return n, bad


# ---------------------------------------------------------------- answer span in a passage
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def span_in_context(answer: str, context: Any) -> bool | None:
    """True/False whether the (normalised) answer occurs in the passage; None if there is no passage or answer."""
    ctx = " ".join(context) if isinstance(context, (list, tuple)) else str(context or "")
    a = _norm(answer)
    if not a or not ctx.strip():
        return None
    return f" {a} " in f" {_norm(ctx)} "


# ---------------------------------------------------------------- lines and the prompt block
def line_for(peer_no: int, kind: str, **k) -> str:
    """One evidence line for peer `peer_no` (1-based, prompt order)."""
    head = f"Peer {peer_no}: "
    if kind == "code":
        if k.get("n_cases", 0) == 0:
            return head + "no sample tests in the statement"
        if k.get("error"):
            return head + f"runtime error on the sample input ({k['error']})"
        return head + f"matches {k['passed']} of {k['n_cases']} sample outputs"
    if kind == "arith":
        n, bad = k["n"], k["bad"]
        if n == 0:
            return head + "no arithmetic steps to check"
        if not bad:
            return head + f"all {n} arithmetic steps check"
        return head + f"{len(bad)} of {n} arithmetic steps fail; first: {bad[0]}"
    if kind == "span":
        v = k["found"]
        return head + ("answer found in the passage" if v else "answer not found in the passage")
    return head + "no check available"


def add_evidence_block(messages: list[dict], lines: Sequence[str]) -> list[dict]:
    """Insert the evidence block before the 'Instruction:' line of the user turn (returns a new list)."""
    out = [dict(m) for m in messages]
    user = out[-1]
    if user.get("role") != "user" or EVIDENCE_HEADER in user["content"] or not lines:
        return out
    block = EVIDENCE_HEADER + "\n" + "\n".join(lines)
    c = user["content"]
    k = c.rfind("\n\nInstruction:")
    user["content"] = (c[:k] + "\n\n" + block + c[k:]) if k >= 0 else (c.rstrip() + "\n\n" + block)
    return out
