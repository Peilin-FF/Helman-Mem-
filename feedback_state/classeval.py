"""ClassEval (Du et al., 2023), one method at a time: the data, a method's code in and out of a class, and the verifier.

The benchmark has 100 Python classes (410 methods). A class comes with its skeleton (imports, the class docstring, the
constructor and every method's signature and docstring), a reference solution, and unit tests: one test class per method
and one for the class as a whole. Built one method at a time, every step has an executable verifier:

    candidates   several models each write method m against the same partial class (the skeleton with the methods already
                 placed in it)
    label        a candidate is right when method m's own tests pass with the candidate swapped into the REFERENCE class, so
                 the label is about this candidate and not about earlier choices (method_label)
    the class    the chosen method goes into the partial class; when every method is placed, the assembled class is scored
                 with all its tests (class_result): the benchmark's class-level and method-level results

Programs run through data.builders.common.code_grading.run_python (a fresh `python -I` subprocess in a temporary directory,
wall-clock timeout, address-space and CPU limits; needs FEEDBACK_CODE_EXEC_ALLOW=1). That is not a security sandbox.
"""
from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from pathlib import Path

PIN = {"repo": "FudanSELab/ClassEval", "revision": "fef204b34e221f207f47904ee660bb920d4c5d1d",
       "file": "data/test-00000-of-00001-5c45fa6e45572491.parquet"}
NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]
TIMEOUT, MEM_MB = 30.0, 4096          # a test class imports pandas, nltk or gensim: the executor's 1 GB address space is too small


def load_classes(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def method_names(cls: dict) -> list[str]:
    return [m["method_name"] for m in cls["methods_info"]]


def method_info(cls: dict, name: str) -> dict:
    return next(m for m in cls["methods_info"] if m["method_name"] == name)


# --- a method's code, out of a model's reply and into a class ------------------------------------------------------------------
def _blocks(text: str) -> list[str]:
    text = str(text or "").rsplit("</think>", 1)[-1]
    fenced = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    return fenced[::-1] + [text]            # the last block first; the whole reply as a last resort


def _find(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def extract_method(text: str, name: str) -> str | None:
    """The definition of method `name` in a model's reply, dedented to column 0 (decorators included), or None.

    The reply may hold the bare method, the method inside its class, or prose around a code block; a think block is dropped.
    """
    for block in _blocks(text):
        for source in (textwrap.dedent(block), block):
            try:
                node = _find(ast.parse(source), name)
            except SyntaxError:
                continue
            if node is None:
                continue
            lines = source.splitlines()
            first = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            return textwrap.dedent("\n".join(lines[first: node.end_lineno])).rstrip() + "\n"
        m = re.search(rf"^([ \t]*)(?:async[ \t]+)?def[ \t]+{re.escape(name)}\b.*", block, flags=re.MULTILINE | re.DOTALL)
        if m:                                # a block cut off mid-way, or with prose after it: keep the longest prefix that parses
            raw, width = m.group(0).splitlines(), len(m.group(1).expandtabs(4))
            body = [raw[0]]
            for line in raw[1:]:             # the method ends at the first line that is not indented deeper than its `def`
                if line.strip() and len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip()) <= width:
                    break
                body.append(line)
            lines = textwrap.dedent("\n".join(body)).splitlines()
            for end in range(len(lines), 0, -1):
                try:
                    ast.parse("\n".join(lines[:end]))
                    return "\n".join(lines[:end]).rstrip() + "\n"
                except SyntaxError:
                    continue
    return None


def replace_method(source: str, class_name: str, name: str, method: str) -> str:
    """`source` with method `name` of class `class_name` replaced by `method` (a dedented definition); appended if absent."""
    tree = ast.parse(source)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        raise ValueError(f"no class {class_name} in the source")
    lines = source.splitlines()
    node = next((n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if node is None:
        indent = " " * (cls.body[0].col_offset if cls.body else 4)
        return "\n".join(lines[: cls.end_lineno] + [""] + textwrap.indent(method.rstrip(), indent).splitlines() + lines[cls.end_lineno:]) + "\n"
    first = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
    new = textwrap.indent(textwrap.dedent(method).rstrip(), " " * node.col_offset).splitlines()
    return "\n".join(lines[:first] + new + lines[node.end_lineno:]) + "\n"


def partial_class(cls: dict, chosen: dict[str, str]) -> str:
    """The class as built so far: the skeleton, with the methods already chosen in place of their signatures."""
    source = cls["skeleton"]
    for name, method in chosen.items():
        source = replace_method(source, cls["class_name"], name, method)
    return source


def reference_with(cls: dict, name: str, method: str) -> str:
    """The reference class with only method `name` replaced: the context a candidate is verified in."""
    return replace_method(cls["solution_code"], cls["class_name"], name, method)


def method_prompt(cls: dict, partial: str, name: str) -> str:
    """What every candidate writer is asked: the same question for the peers and the central model."""
    info = method_info(cls, name)
    return ("Complete one method of a Python class. This is the class so far; a method whose body is only its docstring is not written yet.\n\n"
            f"```python\n{partial.rstrip()}\n```\n\n"
            f"Implement the method `{name}` of class `{cls['class_name']}` exactly as its docstring specifies:\n\n"
            f"```python\n{textwrap.dedent(info['method_description']).rstrip()}\n```\n\n"
            "You may use the class's fields and call its other methods. Return only that method, the full `def` with its body, "
            "inside a single ```python code block.")


# --- the verifier ---------------------------------------------------------------------------------------------------------------
_RUNNER = """

if __name__ == "__main__":
    import sys as _sys, unittest as _unittest
    _suite = _unittest.TestSuite()
    for _name in {names!r}:
        _suite.addTests(_unittest.defaultTestLoader.loadTestsFromTestCase(globals()[_name]))
    _res = _unittest.TextTestRunner(stream=open(__import__("os").devnull, "w"), verbosity=0).run(_suite)
    print("CLASSEVAL ran", _res.testsRun, "failures", len(_res.failures), "errors", len(_res.errors))
    _sys.exit(0 if _res.testsRun > 0 and _res.wasSuccessful() else 1)
"""


def test_program(source: str, cls: dict, test_classes: list[str]) -> str:
    """One runnable program: the class, the benchmark's test module, and a runner for the named test classes only."""
    return source.rstrip() + "\n\n\n" + cls["test"].rstrip() + "\n" + _RUNNER.format(names=list(test_classes))


def run_tests(source: str, cls: dict, test_classes: list[str], *, timeout: float = TIMEOUT, mem_mb: int = MEM_MB) -> dict:
    from data.builders.common.code_grading import run_python

    res = run_python(test_program(source, cls, test_classes), timeout=timeout, mem_mb=mem_mb)
    return {"passed": bool(res.passed), "error": "" if res.passed else str(res.error)[-300:]}


def method_label(cls: dict, name: str, method: str | None) -> dict:
    """Is this candidate right? Method `name`'s own tests, with the candidate swapped into the reference class."""
    if not method:
        return {"passed": False, "error": "no method definition in the reply"}
    try:
        source = reference_with(cls, name, method)
    except (SyntaxError, ValueError) as e:
        return {"passed": False, "error": f"cannot place the method: {type(e).__name__}: {e}"}
    return run_tests(source, cls, [method_info(cls, name)["test_class"]])


def class_result(cls: dict, source: str) -> dict:
    """The benchmark's results for an assembled class: every test class passes (class level), and each method's own tests."""
    methods = {m["method_name"]: run_tests(source, cls, [m["test_class"]])["passed"] for m in cls["methods_info"]}
    return {"class_pass": run_tests(source, cls, list(cls["test_classes"]))["passed"], "methods": methods}


def use_nltk_data(root: Path) -> None:
    """The environment of the test subprocesses (they inherit this process's): nltk's corpora from `pipeline.classeval setup`, and
    one thread per numeric library. On a many-core machine numpy's and gensim's thread pools otherwise burn the executor's CPU-time
    limit within seconds, and a correct class is killed before its tests finish."""
    os.environ["NLTK_DATA"] = str((Path(root) / "nltk_data").resolve())
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
