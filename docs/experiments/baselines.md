# Baselines: majority vote and multi-agent debate

Question: on the same misleading streams, with the same central model (Qwen3-4B) and the same six peer answers, how do
the standard multi-agent methods compare with peers + memory, question + peers, question alone and combination?

## Methods

- **Majority vote (peers)** (`vote_peers`): the six peers' answers are grouped by the answer they give
  (`feedback_state.answer_groups`: canonical option / yes-no / short answer, `math_equal` on final answers, normalised
  RAG answers) and the largest group wins. A tie is broken uniformly at random and scored in expectation. Code has no
  measurable agreement, so its vote is a uniformly random program.
- **Majority vote (peers + own)** (`vote_all`): the same with the central model's question-alone answer as a seventh vote.
- **Debate** (`debate1`, `debate2`; Du et al., 2023, *Improving Factuality and Reasoning in Language Models through
  Multiagent Debate*): the central model starts from its question-alone answer and, in each round, reads the other
  agents' answers and updates its own, keeping its conversation. The round prompt is the paper's ("These are the
  solutions to the problem from other agents: … One agent solution: ``` … ``` … Using the reasoning from other agents as
  additional advice, can you give an updated answer? Examine your solution and that other agents step by step."), with
  its answer-format sentence replaced by the task's instruction. The peers' answers are the released ones and do not
  change between rounds, so a misleading answer stays misleading; only the central model updates.
- **Debate + vote** (`debate_vote`): majority vote over the six peers and the central model's answer after two rounds, the
  paper's final aggregation.

Votes need no GPU (`pipeline.vote`); each debate round is one evaluation (`pipeline.evaluate --mode debate`), round 2
after round 1. Everything else is read from the combination experiment.

## Run

```bash
bash run.sh configs/experiments/combination.yaml --set central=[q3_4b]   # first, if its results are not there
bash run.sh configs/experiments/baselines.yaml --smoke                   # 48 events of indist6_misleading_p050
bash run.sh configs/experiments/baselines.yaml                           # every rate; finished work is skipped
```

## Outputs

```
outputs/eval/q3_4b/<dataset>+own/debate1/, debate2/        generations.jsonl (with every answer so far, `history`) + eval_metrics.json
outputs/eval/q3_4b/<dataset>+own/vote_peers/, vote_all/, debate_vote/
outputs/tables/baselines.md                                 every method per misleading rate, in-distribution and OOD
```
