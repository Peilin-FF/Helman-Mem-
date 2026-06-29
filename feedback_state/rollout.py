from __future__ import annotations

from typing import Any

from feedback_state.utils import extract_final_answer, math_equal


def rollout_score(rollout_answers: list[str], gold_answer: str) -> float:
    if not rollout_answers:
        return 0.0
    correct = sum(1 for answer in rollout_answers if math_equal(answer, gold_answer))
    return correct / float(len(rollout_answers))


def score_rollout_texts(rollout_texts: list[str], gold_answer: str) -> tuple[float, list[str]]:
    answers = [extract_final_answer(text) for text in rollout_texts]
    return rollout_score(answers, gold_answer), answers


def build_step_context(problem: str, previous_steps: list[str], current_subgoal: str) -> str:
    previous = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(previous_steps))
    if not previous:
        previous = "(none)"
    return (
        f"Problem:\n{problem}\n\n"
        f"Previous committed steps:\n{previous}\n\n"
        f"Current planned subgoal:\n{current_subgoal}"
    )


def choose_committed_step(step: dict[str, Any]) -> str:
    if step.get("committed_step"):
        return str(step["committed_step"])
    responses = dict(step.get("peer_step_responses", {}))
    scores = dict(step.get("rollout_scores", {}))
    if responses and scores:
        best_key = max(responses, key=lambda key: float(scores.get(key, 0.0)))
        return str(responses[best_key])
    if responses:
        return str(next(iter(responses.values())))
    return ""
