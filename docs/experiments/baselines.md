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

## Results (2026-09-17, job 20260917-150853)

Qwen3-4B, accuracy %; page: "Multi-Agent Baselines" (https://claude.ai/artifact/LRzXMhXACPWjqmAoqFDHcs).

In-distribution (4,319 questions per rate):

| misleading | question alone | question + peers | vote (peers) | vote (peers + own) | debate 1 | debate 2 | debate + vote | peers + memory | combination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 69.2 | 74.6 | 64.3 | 68.0 | 73.4 | 73.5 | 68.7 | 76.7 | 76.2 |
| 25% | 69.2 | 74.2 | 53.0 | 61.8 | 73.1 | 73.2 | 63.0 | 75.5 | 75.4 |
| 50% | 69.2 | 73.7 | 32.5 | 46.9 | 72.9 | 73.0 | 48.5 | 74.3 | 74.1 |
| 75% | 69.2 | 72.4 | 14.6 | 27.3 | 72.3 | 72.3 | 28.6 | 73.1 | 72.7 |
| 100% | 69.2 | 71.3 | 9.4 | 18.8 | 72.0 | 72.1 | 19.9 | 72.4 | 72.0 |

OOD (17,403 questions per rate):

| misleading | question alone | question + peers | vote (peers) | vote (peers + own) | debate 1 | debate 2 | debate + vote | peers + memory | combination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 67.8 | 69.2 | 64.1 | 68.2 | 71.2 | 71.4 | 68.8 | 73.8 | 75.0 |
| 25% | 67.8 | 64.0 | 48.7 | 56.8 | 68.3 | 68.6 | 57.4 | 70.4 | 72.4 |
| 50% | 67.8 | 58.1 | 24.5 | 34.3 | 64.5 | 64.6 | 34.5 | 67.5 | 70.8 |
| 75% | 67.8 | 51.4 | 5.9 | 10.4 | 60.0 | 60.0 | 10.8 | 62.5 | 70.2 |
| 100% | 67.8 | 45.8 | 1.2 | 3.0 | 56.2 | 56.1 | 3.3 | 50.7 | 69.3 |

- OOD: combination is the most accurate method at every rate; its lead over debate grows from 3.6 to 13.2 points.
- Majority vote collapses once misleading answers are the majority; a seventh vote from Qwen3-4B barely helps.
- In-distribution: combination leads debate by 2.7 points with honest peers, and they are level at 100%.
