"""ClassEval with a pool of peers (feedback_state.classeval, feedback_state.classeval_online, pipeline.classeval): the events, the
answer surgery, the hidden and visible checks, the agentic loop, the online loop, the report and the job expansion.

CPU only; a small class in the benchmark's schema stands in for the data. The checks run programs (a subprocess each).
"""
import ast
import json
from pathlib import Path

import pytest
import torch

from feedback_state import classeval as ce
from feedback_state.classeval_online import ClassEvalAdapter, commit_source
from feedback_state.online_swarm import OnlineRecord, Track, metrics, run_online
from feedback_state.peer_generation import agentic_answers
from feedback_state.reading_line import ReadingLine

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"

# the benchmark stores a description with its first line stripped (add, root: the rest at 8 spaces, a decorator on the stripped
# line) or fully dedented (double); root's example is illustrative (the gold prints 3), double's example raises on the gold
CLS = {
    "task_id": "ClassEval_T", "class_name": "Counter", "import_statement": ["import math"],
    "class_description": '    """\n    A counter.\n    """\n',
    "class_constructor": "class Counter: \n    def __init__(self, start=0):\n        self.value = start\n",
    "solution_code": ("import math\n\n\nclass Counter:\n    def __init__(self, start=0):\n        self.value = start\n\n"
                      "    def add(self, n):\n        self.value += n\n        return self.value\n\n"
                      "    @staticmethod\n    def root(x):\n        return math.isqrt(x)\n\n"
                      "    def double(self):\n        return self.add(self.value)\n"),
    "methods_info": [
        {"method_name": "add", "test_class": "CounterTestAdd",
         "method_description": 'def add(self, n):\n        """\n        Add n and return the new value.\n        >>> c = Counter(1)\n        >>> c.add(2)\n        3\n        """',
         "dependencies": {"Standalone": False, "lib_dependencies": [], "field_dependencies": ["self.value"], "method_dependencies": []}},
        {"method_name": "root", "test_class": "CounterTestRoot",
         "method_description": '@staticmethod\n    def root(x):\n        """\n        The integer square root.\n        >>> Counter.root(10)\n        99\n        """',
         "dependencies": {"Standalone": True, "lib_dependencies": ["math"], "field_dependencies": [], "method_dependencies": []}},
        {"method_name": "double", "test_class": "CounterTestDouble",
         "method_description": 'def double(self):\n    """\n    Double the value.\n    >>> Counter().missing()\n    0\n    """',
         "dependencies": {"Standalone": False, "lib_dependencies": [], "field_dependencies": ["self.value"], "method_dependencies": ["add"]}},
    ],
    "test": ("import unittest\n\nclass CounterTestAdd(unittest.TestCase):\n    def test_add(self):\n        c = Counter(1)\n        self.assertEqual(c.add(2), 3)\n        self.assertEqual(c.add(-5), -2)\n\n"
             "class CounterTestRoot(unittest.TestCase):\n    def test_root(self):\n        self.assertEqual(Counter.root(10), 3)\n\n"
             "class CounterTestDouble(unittest.TestCase):\n    def test_double(self):\n        self.assertEqual(Counter(3).double(), 6)\n\n"
             "class CounterTestMain(unittest.TestCase):\n    def test_main(self):\n        c = Counter(2)\n        c.add(2)\n        self.assertEqual(c.double(), 8)\n\n"
             "if __name__ == '__main__':\n    unittest.main()\n"),
    "test_classes": ["CounterTestAdd", "CounterTestRoot", "CounterTestDouble", "CounterTestMain"],
}
GOOD = {"add": "def add(self, n):\n    self.value += n\n    return self.value",
        "root": "@staticmethod\ndef root(x):\n    return math.isqrt(x)",
        "double": "def double(self):\n    return self.add(self.value)"}
BAD = {"add": "def add(self, n):\n    self.value -= n\n    return self.value",
       "root": "@staticmethod\ndef root(x):\n    return x // 2",
       "double": "def double(self):\n    return 0"}


def fenced(code: str, before: str = "Here it is:") -> str:
    return f"{before}\n```python\n{code}\n```\n"


@pytest.fixture
def run_code(monkeypatch):
    monkeypatch.setenv("FEEDBACK_CODE_EXEC_ALLOW", "1")


def test_events_show_the_class_so_far_with_gold_methods_before_and_stubs_after():
    evs = ce.events_of(CLS)

    assert [e["id"] for e in evs] == ["ClassEval_T/add", "ClassEval_T/root", "ClassEval_T/double"]
    assert all(e["task_type"] == "classeval" and e["peer_responses"] == {} for e in evs)
    ctx = evs[1]["context"]
    ast.parse(ctx)                                                            # the context is a valid class
    assert "self.value += n" in ctx                                           # add: before root, its gold body
    assert "def double(self):" in ctx and "Double the value." in ctx and "        ..." in ctx   # double: after root, a stub
    assert "def root" not in ctx                                              # root is the question
    assert "@staticmethod\ndef root(x):" in evs[1]["problem"]                 # the stripped first line repaired
    assert "A counter." in ctx and "self.value = start" in ctx                # the class docstring and constructor
    assert evs[1]["answer"].startswith("@staticmethod\ndef root(x):")         # the gold method, never shown
    assert evs[0]["classeval"]["doctest"].startswith("Add n") and evs[2]["classeval"]["doctest"]


def test_the_method_is_found_in_the_usual_shapes_of_an_answer():
    whole_class = "```python\nimport math\n\nclass Counter:\n    def add(self, n):\n        self.value += n\n        return self.value\n\n    def helper(self):\n        return 1\n```"
    found = ce.extract_method(whole_class, "add")
    assert found["method"].startswith("def add(self, n):") and found["imports"] == ["import math"] and found["methods"][0].startswith("def helper")

    assert ce.extract_method(fenced(GOOD["add"]), "add")["method"] == GOOD["add"]
    assert ce.extract_method("<think>def add(self): pass</think>" + fenced(GOOD["add"]), "add")["method"] == GOOD["add"]
    indented = "```python\n    def add(self, n):\n        return n\n```"
    assert ce.extract_method(indented, "add")["method"] == "def add(self, n):\n    return n"
    usage_after = fenced(GOOD["add"]) + "Usage:\n```python\nc = Counter()\nc.add(1)\n```"
    assert ce.extract_method(usage_after, "add")["method"] == GOOD["add"]        # a later block without the method is skipped
    assert ce.extract_method(GOOD["add"], "add")["method"] == GOOD["add"]        # no fence
    assert ce.extract_method("```python\ndef add(self, n)\n    return n\n```", "add") is None   # does not parse
    assert ce.extract_method(fenced(GOOD["double"]), "add") is None             # another method
    unclosed = "```python\ndef add(self, n):\n    return n\n"
    assert ce.extract_method(unclosed, "add")["method"] == "def add(self, n):\n    return n"


def test_only_the_asked_method_goes_into_the_gold_class():
    ev = ce.events_of(CLS)[1]
    answer = ("```python\nimport functools\n\nclass Counter:\n    def add(self, n):\n        return 'rewritten'\n\n    def root(x):\n"
              "        return _half(x) + Counter._one()\n\n    @staticmethod\n    def _one():\n        return 0\n\ndef _half(x):\n    return x\n```")
    module = ce.assemble(ev, ce.extract_method(answer, "root"))
    tree = ast.parse(module)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "rewritten" not in module                                          # add is the gold one
    assert [d.id for d in methods["root"].decorator_list] == ["staticmethod"]  # the gold decorator carried over
    assert "_one" in methods                                                  # a helper method the gold class lacks
    assert module.startswith("import functools") and "def _half(x):" in module
    committed = ce.assemble(ev, None, {"add": "def add(self, n):\n    return 'committed'"})
    assert "'committed'" in committed and "math.isqrt" in committed           # overrides replace, the rest stays gold


def test_hidden_tests_label_each_method_in_the_gold_class(run_code):
    for ev in ce.events_of(CLS):
        name = ev["classeval"]["method_name"]
        assert ce.hidden_test(ev, fenced(GOOD[name])).passed, name
        assert not ce.hidden_test(ev, fenced(BAD[name])).passed, name
        assert not ce.hidden_test(ev, "I cannot do this.").passed
    ev = ce.events_of(CLS)[2]
    # double judged on its own: a broken committed add does not count against it; inside the committed class it fails
    assert ce.hidden_test(ev, fenced(GOOD["double"])).passed
    assert not ce.hidden_test(ev, fenced(GOOD["double"]), overrides={"add": "def add(self, n):\n    return 0"}).passed
    assert ce.class_test(ev, GOOD).passed and not ce.class_test(ev, dict(GOOD, add=BAD["add"])).passed


def test_the_visible_check_is_the_strictest_the_gold_method_passes(run_code):
    add, root, double = ce.events_of(CLS)
    assert [ce.validate(e)["visible"] for e in (add, root, double)] == ["doctest", "run", "load"]
    assert all(ce.validate(e)["hidden_ok"] for e in (add, root, double))
    add["classeval"]["visible"], root["classeval"]["visible"], double["classeval"]["visible"] = "doctest", "run", "load"

    ok, msg = ce.visible_check(add, fenced(BAD["add"]))
    assert not ok and "Failed example" in msg and "Expected:\n    3" in msg and "```python code block" in msg
    assert ce.visible_check(add, fenced(GOOD["add"])) == (True, "")
    assert ce.visible_check(root, fenced(BAD["root"])) == (True, "")        # run: a wrong value is not visible
    ok, msg = ce.visible_check(root, fenced("@staticmethod\ndef root(x):\n    return undefined_name"))
    assert not ok and "NameError" in msg                                    # but an exception is
    assert ce.visible_check(double, fenced(BAD["double"])) == (True, "")    # load: the class loads, the method exists
    ok, msg = ce.visible_check(add, "no code here")
    assert not ok and "could not find" in msg


def test_a_peer_revises_on_the_visible_check_until_it_passes():
    records = [{"id": "a", "task_type": "classeval"}, {"id": "b", "task_type": "classeval"}]
    script = {"a": ["wrong", "right"], "b": ["right"]}
    seen = []

    def generate(items):
        seen.append([(i, len(conv)) for i, conv in items])
        return [script[records[i]["id"]][len(conv) // 2] for i, conv in items]

    feedback = lambda r, text: (text == "right", "" if text == "right" else "the examples failed")
    out = agentic_answers(records, generate, feedback, turns=3, first_prompt=lambda r: f"write {r['id']}", workers=2)

    assert seen == [[(0, 1), (1, 1)], [(0, 3)]]                              # only the failing event gets a second turn, with its history
    assert [o["response"] for o in out] == ["right", "right"] and [o["turns"] for o in out] == [2, 1]
    assert [v["passed"] for v in out[0]["visible"]] == [False, True]
    capped = agentic_answers(records[:1], lambda items: ["wrong"] * len(items), feedback, turns=2, first_prompt=lambda r: "x", workers=1)
    assert capped[0]["turns"] == 2 and capped[0]["response"] == "wrong"     # the last answer is kept, right or not


def test_an_answer_is_shown_as_its_method():
    ev = ce.events_of(CLS)[0]
    shown = ce.display(ev, "Sure!\n```python\nimport re\n\ndef add(self, n):\n    return n\n```\nThis adds n.")
    assert shown == "```python\nimport re\n\ndef add(self, n):\n    return n\n```"
    assert ce.display(ev, "x" * 5000).endswith("[... cut off]") and len(ce.display(ev, "x" * 5000)) <= ce.DISPLAY_CHARS


class _Projection:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, X):
        return X[:, : self.dim].float()


def test_the_online_loop_commits_reads_before_writing_and_learns_the_reliable_peer(run_code):
    other = json.loads(json.dumps(CLS).replace("ClassEval_T", "ClassEval_U"))
    classes = [CLS, other]
    n_peers = 6
    feats = {}

    def features_fn(ev, texts):   # an address per event: the question part varies with the event, the answer part with the peer
        g = torch.Generator().manual_seed(len(feats))
        feats[ev["id"]] = True
        return {"sem": torch.randn(4, generator=g), "peer_hidden": torch.eye(n_peers + 1, 4 + n_peers + 1)[:, : 4 + n_peers + 1] * 2.0,
                "margins": torch.zeros(n_peers + 1)}

    def peers_fn(events):          # peer 0 is always right, the others always wrong
        return [[{"response": fenced((GOOD if p == 0 else BAD)[ev["classeval"]["method_name"]]), "turns": 1, "visible": [{"passed": True}]}
                 for p in range(n_peers)] for ev in events]

    def central_fn(requests):      # alone it is wrong; reading, it copies the slot the record trusts most (the first without a record)
        out = []
        for q in requests:
            name = next(n for n in GOOD if f"`{n}`" in q["messages"][1]["content"])
            if q["texts"] is None:
                out.append(fenced(BAD[name]))
            else:
                probs = q["probs"] or [1.0] + [0.0] * (len(q["texts"]) - 1)
                out.append(q["texts"][max(range(len(probs)), key=lambda s: probs[s])])
        return out

    proj_q, proj_c = _Projection(4), _Projection(4 + n_peers + 1)
    tracks = [Track("solo", "solo"), Track("peers", "peers", seed=1),
              Track("tilt", "tilt", record=OnlineRecord(proj_q, proj_c, n_peers + 1, lam=1.0), seed=2),
              Track("combination", "combination", record=OnlineRecord(proj_q, proj_c, n_peers + 1, lam=1.0), line=ReadingLine(), seed=3)]
    def adapter(lanes):
        return ClassEvalAdapter(classes, lanes=lanes, peers_fn=peers_fn, central_fn=central_fn,
                                grade_fn=lambda ev, text, overrides: ce.hidden_test(ev, text, overrides).passed,
                                class_test_fn=lambda ev, committed: ce.class_test(ev, committed).passed, num_peers=n_peers)

    two = adapter(2)
    run_online(two, tracks, central_fn=central_fn, features_fn=features_fn, workers=4, log=lambda *_: None)

    solo, peers, tilt, comb = tracks
    assert [len(t.rows) for t in tracks] == [6, 6, 6, 6] and [len(t.units) for t in tracks] == [2, 2, 2, 2]
    first = [r for r in tilt.rows if r["tick"] == 0]
    assert len(first) == 2 and all(p == 0.5 for r in first for p in r["memory_prob"])    # both lanes read the cold record
    last = max(tilt.rows, key=lambda r: r["pos"])
    assert last["memory_prob"][last["peer_order"].index(0)] > 0.5 > max(p for s, p in enumerate(last["memory_prob"]) if last["peer_order"][s] != 0)
    assert sum(r["correct"] for r in tilt.rows) > sum(r["correct"] for r in solo.rows) == 0
    assert all(c["passed"] == 0 for c in solo.units)
    assert two.committed["tilt"]["ClassEval_T"]["add"] == GOOD["add"] or tilt.rows[0]["correct"] == 0
    # what a track committed is the context of its next methods: solo's broken add is in its double's class
    assert "self.value -= n" in ce.event(CLS, 2, committed=two.committed["solo"]["ClassEval_T"])["context"]
    m = metrics(tilt, two)
    assert m["mode"] == "online" and m["num_samples"] == 6 and "record" in m and m["peers"]["accuracy"][0] == 1.0
    assert m["classes"] == 2 and 0 <= m["class_pass"] <= 1 and "in_context_accuracy" in m and all("in_context_correct" in r for r in tilt.rows)
    assert metrics(comb, two)["combination"]["share_peers_memory"] >= 0 and all("choice" in r for r in comb.rows)
    assert commit_source(CLS, ce.events_of(CLS)[0], "nothing").endswith("raise NotImplementedError")

    # one lane: every method's labels are in the record before the next method is read
    strict = Track("tilt", "tilt", record=OnlineRecord(proj_q, proj_c, n_peers + 1, lam=1.0), seed=2)
    run_online(adapter(1), [strict], central_fn=central_fn, features_fn=features_fn, workers=4, log=lambda *_: None)
    assert [r["tick"] for r in strict.rows] == list(range(6)) and strict.record.mem.writes == 6 * (n_peers + 1)
    trusted = [r["memory_prob"][r["peer_order"].index(0)] for r in sorted(strict.rows, key=lambda r: r["pos"])]
    assert trusted[0] == 0.5 and trusted[1] > 0.5                               # the first method's labels already count for the second
    assert sum(r["correct"] for r in strict.rows) >= 5 and sum(c["passed"] for c in strict.units) >= 1


def test_the_report_puts_the_record_next_to_reliability_tables():
    from pipeline.classeval import reliability

    info = {f"C{c}/m{k}": {"task_id": f"C{c}", "dependencies": {"lib_dependencies": ["re"] if c % 2 else []}} for c in range(10) for k in range(3)}
    rows, pos = [], 0
    for c in range(10):
        for k in range(3):
            y = [1, 0] if c % 2 else [0, 1]           # peer 0 is right exactly on the classes that use re
            rows.append({"pos": pos, "id": f"C{c}/m{k}", "peer_order": [0, 1], "peer_correct": y, "memory_prob": [0.5, 0.5]})
            pos += 1
    rel = reliability(rows, info)
    assert rel["per peer and library"]["pick_right_on_mixed"] > rel["per peer"]["pick_right_on_mixed"]
    assert rel["per peer and class"]["pick_right"] > rel["per peer"]["pick_right"]      # within a class the table learns
    assert rel["_reference"]["events"] == 30 and rel["_reference"]["any_peer_right"] == 1.0


def test_the_record_fits_its_addresses_on_the_warmup_then_replays_it():
    torch.manual_seed(0)
    rec = OnlineRecord(None, None, 3, lam=1.0, dim=2, warmup=4)
    # a question direction shared by the events, and an answer part that tells the three answers apart
    events = [{"sem": torch.ones(6) + 0.1 * torch.randn(6), "peer_hidden": 3 * torch.eye(3, 6) + 0.1 * torch.randn(3, 6),
               "margins": torch.zeros(3)} for _ in range(6)]

    for k, f in enumerate(events[:4]):
        assert rec.read(f) == ([0.5] * 3, [0.0] * 3) and not rec.ready        # cold until the warmup is in
        rec.write(f, [1, 0, 0])
    assert rec.ready and rec.fitted_at == 4 and rec.mem.writes == 4 * 3          # the four events replayed, one write per answer
    p, n = rec.read(events[4])
    assert p[0] > 0.5 > p[1] and min(n) > 0                                      # it learned from the warmup
    with pytest.raises(ValueError):
        OnlineRecord(None, None, 3, warmup=0)


def test_the_classeval_experiments_expand_to_the_intended_jobs(tmp_path):
    from pipeline.config import load
    from pipeline.run import Plan

    over = [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", f"paths.logs={tmp_path}/logs"]
    swarm = Plan(load(EXPERIMENTS / "classeval.yaml", over), "classeval.yaml", smoke=False, gpus=list(range(8)))
    [q] = swarm.questions()
    assert "pipeline.classeval build" in q.cmd and "--class-order shuffled0" in q.cmd and q.gpus == 0
    *features, job = swarm.online()
    assert [f.wave for f in features] == [0] * len(features) and all("--stream" in f.cmd and "mixed_train_big6" in f.cmd for f in features)
    assert job.wave == 1 and job.gpus == 7 and "--lanes 1" in job.cmd and "--turns 3" in job.cmd and "--dim 256" in job.cmd
    assert f"--fit-features {tmp_path}/out/features/q3_4b/train6 --fit-peers 6" in job.cmd and "--warmup" not in job.cmd   # the paper's PCA
    assert '"online_tilt": {"kind": "tilt", "gamma": 3.0}' in job.cmd
    assert '"reasoning": true' in job.cmd and '"prefix_caching": false' in job.cmd      # R1-Distill thinks; DeepSeek-Coder-V2 no prefix cache
    [report] = swarm.diagnose()
    assert "--eval online_tilt=" in report.cmd and "--record" not in report.cmd
    smoke = Plan(load(EXPERIMENTS / "classeval.yaml", over), "classeval.yaml", smoke=True, gpus=list(range(8)))
    *_, job = smoke.online()
    assert "--limit-classes 2" in job.cmd and "/smoke/" in job.cmd and "--dim 8" in job.cmd
    warm = Plan(load(EXPERIMENTS / "classeval.yaml", over + ["online.fit=warmup"]), "classeval.yaml", smoke=False, gpus=list(range(8)))
    [job] = warm.online()
    assert "--warmup 64" in job.cmd and "--fit-stream" not in job.cmd

    offline = Plan(load(EXPERIMENTS / "classeval_offline.yaml", over), "classeval_offline.yaml", smoke=False, gpus=list(range(8)))
    peers = offline.peers()
    assert len(peers) == 6 and all("--turns 3" in j.cmd for j in peers)
    assert any("--reasoning" in j.cmd for j in peers) and any("--no-prefix-caching" in j.cmd for j in peers)
    [report] = [j for j in offline.diagnose() if j.name.startswith("classeval_report")]
    assert "--eval tilt=" in report.cmd and "classeval6+q3_4b" in report.cmd and "--record" in report.cmd
    fitted = Plan(load(EXPERIMENTS / "classeval.yaml", over + ["online.fit=classeval6", "datasets=[classeval6]", "own_answer=true", "record.fit=self"]),
                  "classeval.yaml", smoke=False, gpus=list(range(8)))
    *features, job = fitted.online()
    assert "--fit-stream" in job.cmd and "classeval6+own" in job.cmd and "--fit-peers 7" in job.cmd
    assert features and all("classeval6+q3_4b" in f.cmd for f in features)
