import json

from tests.experiments.counterfactual.beta_b1 import (
    DEFAULT_CF_DIR,
    DEFAULT_WARM_DATA,
    _read_jsonl,
    evaluate_split,
)


def test_cf_beta_routing_is_invariant_to_question_and_responses() -> None:
    warm_records = _read_jsonl(DEFAULT_WARM_DATA)
    records = _read_jsonl(DEFAULT_CF_DIR / "cf_70.jsonl")[:20]
    altered = json.loads(json.dumps(records))
    for record in altered:
        record["problem"] = "hidden"
        record["peer_responses"] = {key: "changed" for key in record["peer_responses"]}

    original = evaluate_split(warm_records, records, gamma=0.9)
    changed = evaluate_split(warm_records, altered, gamma=0.9)
    assert original["correct"] == changed["correct"]
    assert original["selected_peers"] == changed["selected_peers"]
