"""ClassEval as a stream of agentic sub-steps: one event per method, answered by a pool of peers, verified by its hidden tests.

ClassEval (Du et al., ICSE 2024; https://github.com/FudanSELab/ClassEval) has 100 Python classes with 410 methods, and every
method has its own hidden unittest class. A class is built one method at a time, in the benchmark's order (its incremental
strategy), so a task is a sequence of sub-steps whose boundaries the record is never told:

  context   the class so far: its imports, docstring and constructor; the methods before this one with their bodies (the
            gold bodies offline -- teacher forcing -- or the ones the central model committed online); the methods after it
            as signature + docstring with a body of `...` (implemented elsewhere, may be called)
  problem   implement method m (its signature and docstring)
  answers   each peer's method from its own agentic loop: it writes the method, the docstring's examples run in a sandbox
            (the class with the rest in place), it reads the failure and revises (visible_check, the task's feedback_fn)
  label     the method's hidden tests with the answer put into the GOLD class: a peer is judged on its own method only,
            never on what was committed before it (credit assignment per sub-step)

Model-generated code runs through data.builders.common.code_grading.run_python: a subprocess with limits, not a sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_URL = ("https://raw.githubusercontent.com/FudanSELab/ClassEval/eaeac44d0d5dcd8a95feec50726d66fedc73a98f/"
              "data/ClassEval_data.json")
SOURCE_SHA256 = "50a1ac0e4d0c238573c10c0ab11feae00b14a80ae9e4c9cc28148707ecf0b6a2"
DEFAULT_SOURCE = "datasets/classeval/ClassEval_data.json"

TIMEOUT = 10.0            # seconds per program (ClassEval's own harness allows 5 per test class)
MEM_MB = 4096             # numpy / pandas / gensim need more address space than the code grader's default
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
FEEDBACK_CHARS = 1500     # of a failure, shown to the agent
DISPLAY_CHARS = 3000      # an answer as the judge and the central model see it (feedback_state.attn_bias.CHAR_LIMIT)
TASK = "classeval"


# --- the benchmark's data ------------------------------------------------------------------------------------------------
def fetch(path: str | Path = DEFAULT_SOURCE) -> Path:
    """The pinned ClassEval_data.json: downloaded once, checked against its sha256."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as r:
            tmp.write_bytes(r.read())
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"{path}: sha256 {digest} is not the pinned ClassEval data ({SOURCE_SHA256})")
    return path


def load_classes(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


# --- source surgery --------------------------------------------------------------------------------------------------------
def _parse(src: str) -> ast.Module | None:
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError):
        return None


def _span(node: ast.AST) -> tuple[int, int]:
    """0-based [start, end) line span of a definition, its decorators included."""
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
    return start, node.end_lineno


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare) or len(node.test.comparators) != 1:
        return False
    sides = [node.test.left, node.test.comparators[0]]
    return any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides) and \
        any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)


def strip_main(src: str) -> str:
    """The module without its top-level `if __name__ == "__main__":` blocks (a test file would run unittest.main() there)."""
    tree = _parse(src)
    if tree is None:
        return src
    lines = src.splitlines()
    for node in reversed(tree.body):
        if _is_main_guard(node):
            a, b = _span(node)
            del lines[a:b]
    return "\n".join(lines) + "\n"


def _class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)


def _methods(cls: ast.ClassDef) -> dict[str, ast.AST]:
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _segment(lines: list[str], node: ast.AST) -> str:
    a, b = _span(node)
    return textwrap.dedent("\n".join(lines[a:b]))


def reindent(src: str, n: int) -> str:
    """The block dedented, then indented by n spaces (blank lines left empty)."""
    body = textwrap.dedent(src.expandtabs(4)).strip("\n")
    return "\n".join((" " * n + line) if line.strip() else "" for line in body.splitlines())


def _def_name(src: str) -> str | None:
    tree = _parse(textwrap.dedent(src))
    node = next((n for n in (tree.body if tree else []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    return node.name if node else None


def _decorated(src: str) -> bool:
    first = next((line.strip() for line in src.splitlines() if line.strip()), "")
    return first.startswith("@")


# --- the events --------------------------------------------------------------------------------------------------------------
def gold_methods(cls: dict) -> dict[str, str]:
    """Every method of the gold class, dedented, decorators included."""
    src = cls["solution_code"].expandtabs(4)
    tree = _parse(src)
    node = _class(tree, cls["class_name"]) if tree else None
    if node is None:
        raise ValueError(f"{cls['task_id']}: no class {cls['class_name']} in the solution")
    lines = src.splitlines()
    return {name: _segment(lines, n) for name, n in _methods(node).items()}


def _description(method: dict) -> str:
    """The method's signature and docstring as a parseable block at column 0. The benchmark stores it with its first line
    stripped of the class indentation (the rest kept at 8 spaces) or fully dedented; three docstrings are malformed (curly
    quotes, no closing quotes) and are repaired."""
    desc = method["method_description"].expandtabs(4).strip("\n")
    fixed = desc.replace("“”“", '"""').replace("”“”", '"""')
    if fixed.count('"""') % 2 == 1:
        lines = fixed.splitlines()
        indent = next((len(l) - len(l.lstrip()) for l in lines[1:] if '"""' in l), 8)
        fixed += "\n" + " " * indent + '"""'
    for text in (desc, fixed):
        lines = text.splitlines()
        for variant in (textwrap.dedent("\n".join(["    " + lines[0]] + lines[1:])), textwrap.dedent(text)):
            tree = _parse(variant)
            node = next((n for n in (tree.body if tree else []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if node is not None and node.name == method["method_name"]:
                return variant.strip("\n")
    raise ValueError(f"method {method['method_name']}: its description does not parse")


def stub(description: str) -> str:
    """A later method as the context shows it: its signature and docstring, and `...` for a body."""
    lines = description.splitlines()
    indent = next((len(l) - len(l.lstrip()) for l in lines[1:] if l.strip()), 4) or 4
    return description.rstrip() + "\n" + " " * indent + "..."


def class_context(cls: dict, k: int, bodies: dict[str, str]) -> str:
    """The class so far for method k: the imports, the class docstring and constructor, methods before k with their bodies,
    methods after k as stubs. Method k itself is the question."""
    cons = cls["class_constructor"].expandtabs(4).rstrip("\n").splitlines()
    desc = cls.get("class_description", "").expandtabs(4).rstrip("\n").splitlines()
    parts = list(cls.get("import_statement") or [])
    parts += ["", cons[0].rstrip()] + desc + ([""] if desc else []) + cons[1:]
    for j, m in enumerate(cls["methods_info"]):
        if j == k:
            continue
        src = bodies[m["method_name"]] if j < k else stub(_description(m))
        parts += ["", reindent(src, 4)]
    return "\n".join(parts).strip("\n") + "\n"


def doctest_source(description: str) -> str:
    """The method's docstring when it has examples (>>>), for the visible check; empty otherwise."""
    tree = _parse(description)
    node = next((n for n in (tree.body if tree else []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    doc = ast.get_docstring(node, clean=True) if node else None
    return doc if doc and ">>>" in doc else ""


def event(cls: dict, k: int, bodies: dict[str, str] | None = None, committed: dict[str, str] | None = None) -> dict:
    """The event of method k. bodies: the earlier methods' bodies the context shows (default: gold, teacher forcing);
    committed: the central model's methods, which then also stand in the sandbox the agents run their examples in."""
    gold = gold_methods(cls)
    m = cls["methods_info"][k]
    name = m["method_name"]
    desc = _description(m)
    shown = {**gold, **(committed or {}), **(bodies or {})}
    info = {"task_id": cls["task_id"], "class_name": cls["class_name"], "method_name": name, "step": k, "steps": len(cls["methods_info"]),
            "methods": [x["method_name"] for x in cls["methods_info"]], "solution": cls["solution_code"], "test": cls["test"],
            "test_class": m["test_class"], "test_classes": list(cls.get("test_classes") or []), "doctest": doctest_source(desc),
            "dependencies": m.get("dependencies") or {}}
    if committed:
        info["committed"] = dict(committed)
    return {"id": f"{cls['task_id']}/{name}", "task_type": TASK, "source": "classeval",
            "problem": f"Implement the method `{name}` of the class `{cls['class_name']}`:\n\n```python\n{desc}\n```",
            "context": class_context(cls, k, shown), "answer": gold[name], "classeval": info,
            "peer_responses": {}, "peer_metadata": {}, "peer_correct": {}, "correctness_by_peer": {}}


def events_of(cls: dict) -> list[dict]:
    """A class's events under teacher forcing: every method sees the gold methods before it."""
    return [event(cls, k) for k in range(len(cls["methods_info"]))]


# --- answers -------------------------------------------------------------------------------------------------------------------
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)


def _strip_thinking(text: str) -> str:
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def _sources(text: str) -> list[str]:
    """Where a method may be in an answer: its fenced blocks (last first), an unclosed last block, the whole text."""
    blocks = list(reversed(_FENCE.findall(text)))
    if text.count("```") % 2 == 1:
        tail = text.rsplit("```", 1)[1]
        blocks.append(tail.split("\n", 1)[1] if "\n" in tail else "")
    return blocks + [text]


def _find(src: str, name: str) -> dict | None:
    for variant in (src.expandtabs(4), textwrap.dedent(src.expandtabs(4))):
        tree = _parse(variant)
        if tree is None:
            continue
        target, container = None, None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                target, container = node, None
            elif isinstance(node, ast.ClassDef):
                hit = _methods(node).get(name)
                if hit is not None:
                    target, container = hit, node
        if target is None:
            continue
        lines = variant.splitlines()
        imports = [ast.get_source_segment(variant, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                   and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
        functions = [_segment(lines, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not target]
        methods = [_segment(lines, n) for n in _methods(container).values() if n is not target] if container is not None else []
        return {"method": _segment(lines, target), "imports": [i for i in imports if i], "functions": functions, "methods": methods}
    return None


def _by_lines(text: str, name: str) -> dict | None:
    """ClassEval's own fallback: from the line defining the method to the first line indented no deeper than it."""
    lines = text.expandtabs(4).splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(rf"\s*(async\s+)?def\s+{re.escape(name)}\s*\(", l)), None)
    if start is None:
        return None
    lead = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > lead):
        end += 1
    found = _find("\n".join(lines[start:end]), name)
    return found


def extract_method(text: str, name: str) -> dict | None:
    """The method `name` of an answer: {method, imports, functions, methods} (the method's source dedented, the answer's
    imports, its top-level helper functions, and the other methods of the class it was written in); None if there is none."""
    raw = _strip_thinking(str(text or ""))
    for src in _sources(raw):
        found = _find(src, name)
        if found is not None:
            return found
    return _by_lines(raw, name)


def assemble(ev: dict, candidate: dict | None, overrides: dict[str, str] | None = None) -> str | None:
    """The gold module with the candidate in place of the event's method, and `overrides` (method -> source) in place of
    other methods; the candidate's imports go first, its helper functions last, its helper methods after it (only names the
    gold class does not have: a peer never replaces a method it was not asked for). None if the result does not parse."""
    c = ev["classeval"]
    src = strip_main(c["solution"]).expandtabs(4)
    tree = _parse(src)
    cls = _class(tree, c["class_name"]) if tree else None
    if cls is None:
        return None
    nodes = _methods(cls)
    lines = src.splitlines()
    repl = {k: v for k, v in (overrides or {}).items() if k in nodes and k != c["method_name"]}
    extra = []
    if candidate is not None:
        repl[c["method_name"]] = candidate["method"]
        extra = [m for m in candidate.get("methods", []) if _def_name(m) not in nodes]
    edits = []
    for name, new in repl.items():
        node = nodes[name]
        a, b = _span(node)
        text = new
        if node.decorator_list and not _decorated(new):
            text = "\n".join(lines[i].strip() for i in range(a, node.lineno - 1)) + "\n" + textwrap.dedent(new.expandtabs(4))
        block = reindent(text, node.col_offset)
        if candidate is not None and name == c["method_name"]:
            block += "".join("\n\n" + reindent(m, node.col_offset) for m in extra)
        edits.append((a, b, block.splitlines()))
    for a, b, new_lines in sorted(edits, reverse=True):
        lines[a:b] = new_lines
    have = {line.strip() for line in lines}
    head = [i for i in (candidate or {}).get("imports", []) if i.strip() not in have]
    tail = [textwrap.dedent(f.expandtabs(4)) for f in (candidate or {}).get("functions", [])]
    out = "\n".join(head + lines) + "\n" + "".join("\n\n" + f + "\n" for f in tail)
    return out if _parse(out) is not None else None


# --- running --------------------------------------------------------------------------------------------------------------------
_RUNNER = """
import unittest as _unittest, sys as _sys
_ok = True
for _name in {names}:
    _result = _unittest.TextTestRunner(verbosity=0, stream=_sys.stderr).run(_unittest.TestLoader().loadTestsFromTestCase(globals()[_name]))
    _ok = _ok and _result.wasSuccessful() and _result.testsRun > 0
_sys.exit(0 if _ok else 1)
"""

# The visible check runs the docstring's examples. ClassEval's examples are often illustrative (the gold method's own output
# differs), so each event keeps the strictest form its gold method passes (validate): "doctest" compares the outputs, "run"
# only requires that no example raises, "load" only that the class loads and defines the method.
VISIBLE = ("doctest", "run", "load")

_DOCTEST = """
import doctest as _doctest, sys as _sys
class _AnyOutput(_doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        return True
_test = _doctest.DocTestParser().get_doctest({doc}, dict(globals()), {name}, "<docstring>", 0)
_runner = _doctest.DocTestRunner(checker=None if {compare} else _AnyOutput(), verbose=False,
                                 optionflags=_doctest.ELLIPSIS | _doctest.NORMALIZE_WHITESPACE)
_out = []
_result = _runner.run(_test, out=_out.append)
if _result.failed:
    _sys.stderr.write("".join(_out))
    _sys.exit(1)
"""

_DEFINED = """
import sys as _sys
_sys.exit(0 if callable(getattr({cls}, {name}, None)) else 1)
"""


# The benchmark was written against NumPy 1.x: its gold solutions call np.mat, which NumPy 2.0 removed. Restored for every
# program alike (gold and answers), only where the program calls it.
_NUMPY_MAT = "try:\n    import numpy as _np\n    if not hasattr(_np, 'mat'):\n        _np.mat = _np.asmatrix\nexcept Exception:\n    pass\n"


def _run(program: str, timeout: float = TIMEOUT):
    from data.builders.common.code_grading import run_python

    if re.search(r"\b(np|numpy)\.mat\(", program):
        program = _NUMPY_MAT + program
    return run_python(program, timeout=timeout, mem_mb=MEM_MB, env=THREADS, tail=4000, private_tmp=True)


def run_tests(ev: dict, module: str, test_classes: list[str], timeout: float = TIMEOUT):
    """The named unittest classes of the event's test module against a module; passes iff all run and succeed."""
    program = "\n\n".join([module, strip_main(ev["classeval"]["test"]), _RUNNER.format(names=json.dumps(list(test_classes)))])
    return _run(program, timeout)


def _failure(stderr: str) -> str:
    text = re.sub(r'File "[^"]*prog\.py"', 'File "<sandbox>"', stderr.strip())
    if "Failed example" in text:
        return text[:FEEDBACK_CHARS]
    return text[-FEEDBACK_CHARS:]


def hidden_test(ev: dict, text: str, overrides: dict[str, str] | None = None, timeout: float = TIMEOUT):
    """The label: the event's hidden test class on the gold class with the answer's method put in (and `overrides`, when the
    question is how the method does in a committed class rather than on its own)."""
    from data.builders.common.code_grading import ExecResult

    name = ev["classeval"]["method_name"]
    candidate = extract_method(text, name)
    if candidate is None:
        return ExecResult(False, f"no method named {name}")
    module = assemble(ev, candidate, overrides)
    if module is None:
        return ExecResult(False, "the method does not fit into the class")
    return run_tests(ev, module, [ev["classeval"]["test_class"]], timeout)


def visible_mode(ev: dict) -> str:
    c = ev["classeval"]
    return c.get("visible") or ("doctest" if c.get("doctest") else "load")


def _visible_program(ev: dict, module: str, mode: str) -> str:
    c = ev["classeval"]
    if mode == "load":
        return module + _DEFINED.format(cls=c["class_name"], name=json.dumps(c["method_name"]))
    return module + _DOCTEST.format(doc=json.dumps(c["doctest"]), name=json.dumps(c["method_name"]), compare=mode == "doctest")


def visible_check(ev: dict, text: str, timeout: float = TIMEOUT) -> tuple[bool, str]:
    """What an agent sees after writing the method: the docstring's examples run on the sandbox class (the gold class, with
    the committed methods in place online, and the answer's method), in the event's visible mode. Returns (passed, the
    message for the agent's next turn: empty when passed)."""
    c = ev["classeval"]
    name, cls = c["method_name"], c["class_name"]
    ask = f"Return the complete corrected method `{name}` (its def line and body) in a single ```python code block."
    candidate = extract_method(text, name)
    if candidate is None:
        return False, (f"I could not find a complete, syntactically valid method named `{name}` in your answer. "
                       f"Return the whole method (its def line and body) in a single ```python code block.")
    module = assemble(ev, candidate, c.get("committed"))
    if module is None:
        return False, f"Your method does not fit into the class `{cls}` (its indentation or syntax is broken). {ask}"
    mode = visible_mode(ev)
    res = _run(_visible_program(ev, module, mode), timeout)
    if res.passed:
        return True, ""
    if mode == "load":
        return False, f"Loading the class `{cls}` with your method failed:\n```\n{_failure(res.error)}\n```\n{ask}"
    what = "They failed" if mode == "doctest" else "They raised an error"
    return False, (f"I ran the examples from the docstring of `{name}` with your method in the class (the rest of the class is "
                   f"in place). {what}:\n```\n{_failure(res.error)}\n```\n{ask}")


def validate(ev: dict, timeout: float = TIMEOUT) -> dict:
    """The gold method in its own harness: its hidden tests must pass here (else the environment lacks something and the
    event is dropped), and the visible check is the strictest mode its docstring's examples pass on the gold method."""
    module = assemble(ev, None)
    hidden = run_tests(ev, module, [ev["classeval"]["test_class"]], timeout) if module else None
    out = {"hidden_ok": bool(hidden and hidden.passed), "hidden_error": "" if hidden and hidden.passed else (hidden.error[-600:] if hidden else "assemble"),
           "visible": "load"}
    if out["hidden_ok"] and ev["classeval"].get("doctest"):
        for mode in ("doctest", "run"):
            if _run(_visible_program(ev, module, mode), timeout).passed:
                out["visible"] = mode
                break
    return out


def class_test(ev: dict, committed: dict[str, str], timeout: float = 3 * TIMEOUT):
    """A committed class as a whole: every method the central model's, all of the class's test classes."""
    module = assemble(ev, None, committed)
    if module is None:
        from data.builders.common.code_grading import ExecResult

        return ExecResult(False, "the committed class does not parse")
    return run_tests(ev, module, ev["classeval"]["test_classes"] or [ev["classeval"]["test_class"]], timeout)


# --- prompts and display -------------------------------------------------------------------------------------------------
def peer_prompt(ev: dict) -> str:
    c = ev["classeval"]
    return (f"You are one of several developers building a Python class together, one method at a time. The class so far is "
            f"below. Methods whose body is `...` are implemented elsewhere in the codebase; you may call them.\n\n"
            f"```python\n{ev['context'].rstrip()}\n```\n\n{ev['problem']}\n\n"
            f"Return the complete method `{c['method_name']}` (its def line and body) in a single ```python code block. "
            f"Do not repeat the rest of the class.")


def display(ev: dict, text: str) -> str:
    """An answer as the judge and the central model read it: the method (with the imports and helpers it brought) in one
    code block; an answer without the method is shown as it is, clipped."""
    candidate = extract_method(text, ev["classeval"]["method_name"])
    if candidate is None:
        raw = _strip_thinking(str(text or "")).strip()
        return raw if len(raw) <= DISPLAY_CHARS else raw[: DISPLAY_CHARS - 40].rstrip() + "\n[... cut off]"
    code = "\n\n".join(["\n".join(candidate["imports"])] * bool(candidate["imports"]) + [candidate["method"]]
                       + candidate["methods"] + candidate["functions"])
    shown = f"```python\n{code}\n```"
    return shown if len(shown) <= DISPLAY_CHARS else shown[: DISPLAY_CHARS - 40].rstrip() + "\n# [... cut off]\n```"


# --- the task (feedback_state.tasks registers these) -------------------------------------------------------------------------
def target(text: str, record: dict[str, Any]) -> float:
    return 1.0 if hidden_test(record, text).passed else 0.0


def correct(text: str, record: dict[str, Any]) -> bool:
    return target(text, record) >= 0.5


def prompt(record: dict[str, Any], with_context: bool) -> str:
    return peer_prompt(record)


def extract(text: str) -> str:
    from feedback_state.tasks import code_extract_answer

    return code_extract_answer(_strip_thinking(str(text or "")))
