import hashlib
import json

import pytest

from training.sigma_rl.audit_outcome_data import audit


class Tokenizer:
    name_or_path = "fixture"
    chat_template = "fixture"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)

    def __call__(self, prompts, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False, "padding": False}
        assert all("PRIVATE_GOLD" not in p for p in prompts)
        return {"input_ids": [[1] * len(prompt) for prompt in prompts]}


def test_audit_measures_all_rows_without_filtering_or_changing_the_dataset(tmp_path):
    records = [
        {"id": "short", "problem": "q", "task_type": "math", "answer": "PRIVATE_GOLD",
         "peer_responses": {"p0": "reasoning", "p1": "reasoning"}},
        {"id": "long", "problem": "q", "task_type": "math", "answer": "PRIVATE_GOLD",
         "peer_responses": {"p0": "x" * 9000, "p1": "z" * 9000}},
    ]
    path = tmp_path / "records.jsonl"
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    path.write_bytes(payload)
    report = audit(path, Tokenizer(), max_length=8192, chunk_size=1)
    assert report["records"] == 2 and report["lengths"]["count"] == 2
    assert report["lengths"]["over_budget"] == 1
    assert report["over_budget_examples"][0]["id"] == "long"
    assert report["peer_count_histogram"] == {2: 2}
    assert report["records_sha256"] == hashlib.sha256(payload).hexdigest()
    assert path.read_bytes() == payload


def test_missing_reference_is_reported_but_does_not_change_public_prompt_coverage(tmp_path):
    record = {"id": "no_gold", "task_type": "math", "problem": "q", "peer_responses": {"p": "response"}}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record) + "\n")
    report = audit(path, Tokenizer(), max_length=8192)
    assert report["invalid_reference_records"] == 1
    assert report["invalid_public_records"] == 0 and report["lengths"]["count"] == 1


def test_invalid_audit_limits_fail_before_reading_data(tmp_path):
    with pytest.raises(ValueError):
        audit(tmp_path / "not_opened", Tokenizer(), max_length=0)
