"""ClassEval one method at a time: a method's code out of a reply and into a class, the test program, the verifier."""
import os

import pytest

from feedback_state.classeval import (class_result, extract_method, method_label, method_prompt, partial_class, reference_with, replace_method,
                                      test_program as build_test_program)
from feedback_state.tasks import get_task, peer_is_correct

CLS = {
    "task_id": "Toy_0", "class_name": "Counter",
    "skeleton": 'import math\n\nclass Counter:\n    """\n    Counts.\n    """\n\n    def __init__(self):\n        self.n = 0\n\n    def add(self, k):\n        """\n        Add k and return the total.\n        """\n\n    def double(self):\n        """\n        Double the total using add.\n        """\n',
    "solution_code": "import math\n\nclass Counter:\n    def __init__(self):\n        self.n = 0\n\n    def add(self, k):\n        self.n += k\n        return self.n\n\n    def double(self):\n        return self.add(self.n)\n",
    "test": "import unittest\n\nclass CounterTestAdd(unittest.TestCase):\n    def test_add(self):\n        c = Counter()\n        self.assertEqual(c.add(2), 2)\n\nclass CounterTestDouble(unittest.TestCase):\n    def test_double(self):\n        c = Counter(); c.add(3)\n        self.assertEqual(c.double(), 6)\n\nclass CounterTest(unittest.TestCase):\n    def test_all(self):\n        c = Counter(); c.add(1)\n        self.assertEqual(c.double(), 2)\n",
    "test_classes": ["CounterTestAdd", "CounterTestDouble", "CounterTest"],
    "methods_info": [{"method_name": "add", "method_description": 'def add(self, k):\n        """\n        Add k and return the total.\n        """', "test_class": "CounterTestAdd"},
                     {"method_name": "double", "method_description": 'def double(self):\n        """\n        Double the total using add.\n        """', "test_class": "CounterTestDouble"}],
}
GOOD_ADD = "def add(self, k):\n    self.n += k\n    return self.n\n"


def test_a_method_is_read_out_of_a_reply_whatever_surrounds_it():
    assert extract_method("Here it is:\n```python\ndef add(self, k):\n    self.n += k\n    return self.n\n```\nDone.", "add") == GOOD_ADD
    inside = "```python\nclass Counter:\n    def __init__(self):\n        self.n = 0\n\n    @staticmethod\n    def add(self, k):\n        self.n += k\n        return self.n\n```"
    assert extract_method(inside, "add") == "@staticmethod\n" + GOOD_ADD                                   # found inside its class, dedented, decorator kept
    assert extract_method("<think>```python\ndef add(self, k):\n    return 0\n```</think>```python\n" + GOOD_ADD + "```", "add") == GOOD_ADD   # the think block is dropped
    assert extract_method("```python\n    def add(self, k):\n        self.n += k\n        return self.n\nand then some prose", "add") == GOOD_ADD   # an unclosed block: the prefix that parses
    assert extract_method("I would add k to the total.", "add") is None and extract_method("```python\ndef other(self):\n    pass\n```", "add") is None


def test_a_method_goes_into_the_partial_class_and_into_the_reference():
    partial = partial_class(CLS, {"add": GOOD_ADD})
    assert "        self.n += k\n        return self.n" in partial and '"""\n        Double the total using add.' in partial      # add is written, double still a signature
    assert partial.startswith("import math") and "Add k and return the total." not in partial
    wrong = reference_with(CLS, "double", "def double(self):\n    return 0\n")
    assert "return 0" in wrong and "self.n += k" in wrong and "return self.add(self.n)" not in wrong                       # only that method changed
    assert "def extra(self):" in replace_method(CLS["solution_code"], "Counter", "extra", "def extra(self):\n    return 1\n")   # an absent method is appended
    prompt = method_prompt(CLS, partial, "double")
    assert "self.n += k" in prompt and "Implement the method `double` of class `Counter`" in prompt and prompt.rstrip().endswith("code block.")
    program = build_test_program(CLS["solution_code"], CLS, ["CounterTestDouble"])
    assert program.index("class Counter:") < program.index("class CounterTestAdd") < program.index("'CounterTestDouble'")


def test_the_classeval_task_type_keeps_executed_labels():
    record = {"task_type": "classeval", "problem": "PROMPT", "peer_correct": {"peer_0": 1.0, "peer_1": 0.0}}
    assert get_task("classeval").precomputed and get_task("classeval").prompt_fn(record, True) == "PROMPT"
    assert peer_is_correct(record, "peer_0", "anything") and not peer_is_correct(record, "peer_1", "anything")


@pytest.mark.skipif(os.environ.get("FEEDBACK_CODE_EXEC_ALLOW") != "1", reason="executes code: FEEDBACK_CODE_EXEC_ALLOW=1")
def test_the_verifier_labels_a_candidate_by_its_methods_own_tests_in_the_reference_class():
    assert method_label(CLS, "add", GOOD_ADD)["passed"] and not method_label(CLS, "add", "def add(self, k):\n    return k + 1\n")["passed"]
    assert not method_label(CLS, "add", None)["passed"] and not method_label(CLS, "double", "def double(self):\n    return (\n")["passed"]
    whole = class_result(CLS, partial_class(CLS, {"add": GOOD_ADD, "double": "def double(self):\n    return self.add(self.n)\n"}))
    assert whole == {"class_pass": True, "methods": {"add": True, "double": True}}
    half = class_result(CLS, partial_class(CLS, {"add": GOOD_ADD}))
    assert half["methods"] == {"add": True, "double": False} and not half["class_pass"]
