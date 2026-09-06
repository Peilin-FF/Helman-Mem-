"""Six-peer training pipeline page: peers, streams, history format, training regime, evaluation protocol, status and results.

Reads (when present): outputs/peergen_new/table_five.json (peer accuracies on the training stream), outputs/gen/q3_4b/probe5/record_*.json
(record quality per stream, 3/5/6 peers), outputs/rl/q3_4b_6peer_*/hf/global_step_*/eval_*_vllm/eval_metrics.json (arm results),
outputs/rl/q3_4b_6peer_*/metrics.jsonl (training progress).  Writes artifacts/helman_mem_pipeline6.html.
"""
from __future__ import annotations

import glob
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/helman_mem_pipeline6.html"
PEERS = [("peer_0", "gemma-3-4b-it", "Google, 4B"), ("peer_1", "Phi-4-mini-instruct", "Microsoft, 3.8B"), ("peer_2", "Qwen2.5-Coder-7B-Instruct", "Alibaba, 7B (legacy Qwen peer)"),
         ("peer_3", "Meta-Llama-3.1-8B-Instruct", "Meta, 8B"), ("peer_4", "DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek, 16B MoE (code specialist)"), ("peer_5", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek, 7B (reasoning; answer after its think block)")]
TRAIN_ACC = {"gsm8k": [69, 61, 60, 88, 93, 91], "squad": [13, 79, 66, 63, 12, 23], "apps": [29, 9, 16, 20, 30, 22]}
ARMS = [("q3_4b_6peer_mix75_memory", "history as written by the memory", "the method"), ("q3_4b_6peer_mix75_peers", "history hidden (peers only)", "control"), ("q3_4b_6peer_mix75_swapped", "history swapped by rank", "control")]
EVALS = [("eval_indist6_memory_vllm", "in-dist, peers + history"), ("eval_indist6_solo_vllm", "in-dist, alone"),
         ("eval_ood6_memory_vllm", "OOD, peers + history"), ("eval_ood6_solo_vllm", "OOD, alone")]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f1(x, d=1, suffix=""):
    return "—" if x is None else f"{x:.{d}f}{suffix}"


# ----------------------------------------------------------------------------------------------- SVG pipeline
def svg_pipeline() -> str:
    W, H = 1180, 420
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Six-peer training pipeline" style="width:100%;height:auto">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dhead"/></marker></defs>']
    def box(x, y, w, h, t, sub, cls="dbox"):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/><text x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle" class="dtitle">{html.escape(t)}</text>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle" class="dsub">{html.escape(sub)}</text>')
    def arrow(x1, y1, x2, y2, cls="darrow"):
        return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#ah)"/>'
    s.append('<text x="20" y="22" class="dcol">1 · data</text>')
    s.append(box(20, 36, 190, 60, "six peers answer", "same prompt, T=0.2, per-task budget"))
    s.append(box(250, 36, 190, 60, "graded", "F1 ≥ 0.5 / exact / tests"))
    s.append(box(480, 36, 190, 60, "six-peer streams", "train 17,709 · in-dist 4,319 · OOD 17,403"))
    s.append(box(710, 36, 200, 60, "frozen-judge features", "question + each peer answer"))
    s.append(box(950, 36, 210, 60, "Kalman record", "read before write, along the stream"))
    s.append(arrow(210, 66, 250, 66)); s.append(arrow(440, 66, 480, 66)); s.append(arrow(670, 66, 710, 66)); s.append(arrow(910, 66, 950, 66))
    s.append('<text x="20" y="152" class="dcol">2 · prompt, history before the answer</text>')
    s.append(box(20, 166, 320, 70, "question + all six solutions", "nothing filtered or ranked away"))
    s.append(box(380, 166, 360, 70, "history on every peer", "estimate at the address · evidence · domain record", "dbox strong"))
    s.append(box(780, 166, 380, 70, "central model answers", "group of 8 samples; 25% of events question-only"))
    s.append(arrow(1055, 96, 560, 166, "darrow")); s.append(arrow(340, 201, 380, 201)); s.append(arrow(740, 201, 780, 201))
    s.append('<text x="20" y="292" class="dcol">3 · learning and evaluation</text>')
    s.append(box(20, 306, 260, 70, "labels after the answer", "verifier reward; GRPO; KL 0.001"))
    s.append(box(320, 306, 260, 70, "record updated", "peers' verified labels of the event"))
    s.append(box(620, 306, 540, 70, "evaluation of the trained model", "with peers + history, and alone, on both test streams"))
    s.append(arrow(970, 236, 150, 306, "darrow")); s.append(arrow(280, 341, 320, 341)); s.append(arrow(450, 306, 1055, 96, "darrow thin")); s.append(arrow(580, 341, 620, 341))
    s.append(f'<text x="590" y="405" text-anchor="middle" class="dsub">peers + history is the deployed condition; alone accuracy guards the model\'s own ability; the swapped-history check is run on demand</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--paper:#f6f7f5;--ink:#1b2229;--muted:#5c6670;--rule:#d7dcd8;--code:#eef1ee;--accent:#0d6b6c;--amber:#b26f12;--amber-bg:#fbf3e4;--teal-bg:#e6f2f1;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#12171b;--ink:#e4e8e6;--muted:#98a3ab;--rule:#2b333a;--code:#1a2126;--accent:#45b8b2;--amber:#dfa24d;--amber-bg:#2a2216;--teal-bg:#152a2a;--bad:#ef7d74}}
:root[data-theme="dark"]{--paper:#12171b;--ink:#e4e8e6;--muted:#98a3ab;--rule:#2b333a;--code:#1a2126;--accent:#45b8b2;--amber:#dfa24d;--amber-bg:#2a2216;--teal-bg:#152a2a;--bad:#ef7d74}
body{background:var(--paper);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:15.5px;line-height:1.55}
.page{max-width:1180px;margin:0 auto;padding:1.6rem 1.4rem 4rem}
header{border-bottom:2px solid var(--accent);padding-bottom:1rem;margin-bottom:1.6rem}
.eyebrow{font-family:"JetBrains Mono",monospace;font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font-family:"Bricolage Grotesque",sans-serif;font-size:2.2rem;margin:.3rem 0 .6rem;text-wrap:balance}
h2{font-family:"Bricolage Grotesque",sans-serif;font-size:1.35rem;margin:2rem 0 .6rem;text-wrap:balance}
h3{font-family:"Bricolage Grotesque",sans-serif;font-size:1.05rem;margin:1.3rem 0 .4rem}
p{max-width:95ch} .thesis{font-size:1.05rem;max-width:100ch}
.tbl{overflow-x:auto;margin:.6rem 0 1rem} table{border-collapse:collapse;font-size:.88rem;min-width:600px} th,td{padding:.4rem .6rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top} th{font-family:"JetBrains Mono",monospace;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)} td{font-variant-numeric:tabular-nums}
td.best{color:var(--accent);font-weight:700} .sub{color:var(--muted);font-size:.86rem;max-width:100ch} .pend{color:var(--amber);font-style:italic}
code{font-family:"JetBrains Mono",monospace;font-size:.86em;background:var(--code);padding:.05em .3em;border-radius:4px}
pre{background:var(--code);border:1px solid var(--rule);border-radius:8px;padding:.6rem .8rem;font-size:.8rem;white-space:pre-wrap;max-width:100ch;overflow-x:auto;font-family:"JetBrains Mono",monospace}
.callout{border-left:4px solid var(--accent);background:var(--teal-bg);padding:.7rem .95rem;border-radius:0 8px 8px 0;margin:.9rem 0;max-width:100ch} .callout.warn{border-left-color:var(--amber);background:var(--amber-bg)}
.diagram{background:var(--code);border:1px solid var(--rule);border-radius:10px;padding:.6rem;margin:.5rem 0 .9rem}
.dbox{fill:var(--paper);stroke:var(--rule);stroke-width:1.2} .dbox.strong{stroke:var(--accent);stroke-width:2} .dtitle{fill:var(--ink);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:600} .dsub{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px}
.dcol{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.darrow{stroke:var(--muted);stroke-width:1.6;fill:none} .darrow.thin{stroke-width:1;opacity:.5;stroke-dasharray:4 3} .dhead{fill:var(--muted)}
ol,ul{max-width:100ch} li{margin:.35rem 0}
"""


def main() -> None:
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()
    def row(cells, head=False):
        tag = "th" if head else "td"
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
    # ---- peers
    prow = [row(["slot", "model", "family / size", "GSM8K (7,473)", "SQuAD (8,500)", "APPS (1,736)"], True)]
    for i, (slot, name, fam) in enumerate(PEERS):
        prow.append(row([slot, name, fam, f"{TRAIN_ACC['gsm8k'][i]}%", f"{TRAIN_ACC['squad'][i]}%", f"{TRAIN_ACC['apps'][i]}%"]))
    # ---- record quality 3 / 5 / 6 peers
    rq = {}
    for tag, name in (("prompts_train_fixed", "train, 3 peers"), ("prompts_train5_fixed", "train, 5 peers"), ("prompts_train6_fixed", "train, 6 peers"),
                      ("prompts_indist_shuffled0", "in-dist, 3 peers"), ("prompts_indist5_shuffled0", "in-dist, 5 peers"), ("prompts_indist6_shuffled0", "in-dist, 6 peers"),
                      ("prompts_ood_shuffled0", "OOD, 3 peers"), ("prompts_ood5_shuffled0", "OOD, 5 peers"), ("prompts_ood6_shuffled0", "OOD, 6 peers")):
        d = load(ROOT / f"outputs/gen/q3_4b/probe5/record_{tag}.json")
        if d:
            rq[name] = d
    rrow = [row(["stream, peers", "events where peers split", "record AUC", "favourite right where peers split", "running mean per peer and task", "favourite against the majority label: right", "split, favourite against the majority (events)"], True)]
    for name, d in rq.items():
        part = d.get("partition", {})
        rrow.append(row([name, f"{d['n_mixed']:,} ({100 * d['n_mixed'] / d['n_events']:.0f}%)", f"{d['memory']['auc']:.3f}", f"{d['memory']['favourite_acc_mixed']:.1f}%", f"{d['running_peer_task']['favourite_acc_mixed']:.1f}%", f"{d['minority_favourite_right']:.1f}%", f"{part.get('split, favourite against the majority', 0):,}"]))
    # ---- arms: progress and results
    arow = [row(["arm", "role", "training", ] + [lab for _, lab in EVALS], True)]
    for exp, desc, role in ARMS:
        m = ROOT / f"outputs/rl/{exp}/metrics.jsonl"
        steps = sum(1 for _ in m.open()) if m.exists() else 0
        status = f"{steps} / 277 steps" if steps else "<span class='pend'>queued</span>"
        cells = [desc, role, status]
        ck = sorted(glob.glob(str(ROOT / f"outputs/rl/{exp}/hf/global_step_*")), key=lambda p: int(p.rsplit("_", 1)[-1]))
        for key, _ in EVALS:
            val = None
            for c in reversed(ck):
                d = load(Path(c) / key / "eval_metrics.json")
                if d:
                    val = 100 * d["accuracy"]; break
            cells.append(f1(val, 2) if val is not None else "<span class='pend'>pending</span>")
        arow.append(row(cells))
    example = ("Peer 1 (estimated probability correct: 0.70, based on 10 similar past cases; on reading comprehension: right on 89 of 108 earlier questions):\n"
               "During the mid-Eocene.\n\nPeer 2 (estimated probability correct: 0.55, based on 10 similar past cases; on reading comprehension: right on 84 of 108 earlier questions):\n"
               "The drainage basin of the Amazon was believed to have split in the middle of South America during the mid-Eocene ...\n\n"
               "Peer 3 (estimated probability correct: 0.68, based on 10 similar past cases; on reading comprehension: right on 95 of 108 earlier questions):\n...")
    page = f'''<title>Six-Peer Training Pipeline</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style>
<div class="page">
<header>
<div class="eyebrow">helman-mem · six-peer pipeline · history before the answer · updated {stamp}</div>
<h1>Six-Peer Training Pipeline</h1>
<p class="thesis">The central model (Qwen3-4B) is trained to answer with six peer solutions in front of it and, before it answers, the memory's history on every peer: how reliable that peer has been on questions at this address, on how much evidence, and its verified record on this kind of task. Nothing is filtered or ranked away; the model decides what to rely on. Correctness is revealed only after the answer. This page records the data, the prompt, the training regime, the evaluation protocol, and the results as they arrive.</p>
</header>

<section>
<h2>Pipeline</h2>
<div class="diagram">{svg_pipeline()}</div>
</section>

<section>
<h2>Peers</h2>
<div class="tbl"><table>{''.join(prow)}</table></div>
<p class="sub">Accuracy on the training stream under the stored labels (token-F1 ≥ 0.5 for SQuAD, exact equality for GSM8K, executed tests for APPS). The new peers' answers were generated with the same task prompt, chat template, sampling (temperature 0.2, top-p 0.95) and per-task token budgets as the original three (96 tokens on the OOD tasks); the reasoning peer keeps a 4,096-token budget and its stored answer is the text after its think block. Unformatted or truncated answers count as wrong: the answer format is part of the task, and the central model is graded the same way.</p>
<p>With six peers, some peer is right on 99% of GSM8K, 87% of SQuAD and 59% of APPS training events, against 90 / 84 / 41% with three; the events where the peers split, where a reliability record can act, grew from 64% to 75% of the training stream, from 60% to 70% in-distribution and from 29% to 50% on OOD (five-peer figures; the six-peer record is below).</p>
</section>

<section>
<h2>The record's quality with more peers</h2>
<div class="tbl"><table>{''.join(rrow) if len(rrow) > 1 else row(["<span class='pend'>record quality pending</span>"])}</table></div>
<p class="sub">Read-before-write on the prompt files. AUC: a right peer is printed with a higher estimate than a wrong one. "Favourite right where peers split": the peer with the highest estimate is right on events where the peers are neither all right nor all wrong; the running mean per peer and task is the trivial history it is measured against. "Favourite against the majority label": when the record's favourite disagrees with the majority of the labels, how often it is the right one.</p>
</section>

<section>
<h2>The prompt: history before the answer</h2>
<p>System prompt: the model is the central model of a multi-agent system; a reliability memory has tracked, from verified feedback on earlier questions, how often each peer was correct on similar questions, and its estimate is given with the number of similar past cases and the peer's verified record on this kind of task; treat the peer answers as evidence weighted by their reliability, verify them, and produce your own final answer. Then the question (with options or passage), all six solutions with their history, and the task instruction. A reading example, from the in-distribution stream:</p>
<pre>{html.escape(example)}</pre>
<p class="sub">Every number is computed before the event is written to the memory. The address estimate comes from the Kalman filter over the frozen judge's features of the question and the peer answer; the domain record is the peer's running count on this task type along the stream. Prompt length with six peers stays under the 8,192-token budget (five-peer maximum 4,177 tokens; six-peer prompts add one clipped answer of at most 3,000 characters).</p>
</section>

<section>
<h2>Training regime</h2>
<ul>
<li><b>Group of 8.</b> Eight sampled answers per event form one GRPO group, all under the same prompt, so the advantage compares answers that saw the same peers and the same history.</li>
<li><b>75% with peers and history, 25% question only.</b> For a quarter of the events, drawn at random in stream order, the whole group answers the question alone; this keeps the model's own ability. There is no relabelling across prompt types (the mechanism that leaked peer style into the alone policy in the three-peer runs).</li>
<li><b>Labels only after the answer.</b> The reward is the task verifier on each sample; the memory is updated with the peers' verified labels of the event afterwards.</li>
<li><b>Plain GRPO otherwise</b>: clip 0.2, learning rate 1e-6, one epoch in stream order (277 steps of 64 events), full parameters, 768-token answers, thinking off, plus a KL penalty of 0.001 to the frozen model against the drift and collapse seen without it. Eight GPUs per arm, one arm at a time.</li>
<li><b>Three arms.</b> The method (history as the memory wrote it); the history hidden (peers only); the history swapped by rank so the highest reliability is printed on the least trusted peer. The controls tell whether the model learned to use the history rather than the peers' content.</li>
</ul>
<div class="callout"><b>Why this regime.</b> The three-peer study found that the model reads the history but does not act on it, because one greedy guided answer per event gives GRPO nothing to contrast, and because relabelling guided answers under the question-only prompt taught the alone policy the peers' style. Groups of eight under the same prompt, no cross-relabelling, an enriched history (domain record) and six peers with a wider split region address each of those directly.</div>
</section>

<section>
<h2>Evaluation protocol</h2>
<p>Each arm's final checkpoint is evaluated on the six-peer test streams (in-distribution 4,319 events; OOD 17,403 events, all tasks unseen in training) in two conditions: with the peers and their history in the prompt (the deployed setting), and alone. Greedy decoding with vLLM, the task verifiers, whole streams.</p>
<ul>
<li><b>Deployed accuracy</b> = with peers and history; compared across the three arms, the difference between the method and the hidden-history arm is the value of the history, and against the swapped arm the cost of a wrong history.</li>
<li><b>Own ability</b> = alone accuracy, compared with the frozen model and the three-peer runs.</li>
<li>The swapped-history evaluation of a single checkpoint (the direct test of learned use of the history) is available on demand with the same script.</li>
</ul>
</section>

<section>
<h2>Status and results</h2>
<div class="tbl"><table>{''.join(arow)}</table></div>
<p class="sub">Accuracy in percent on the whole test streams. Training progress from the runs' metrics files; results from <code>outputs/rl/&lt;arm&gt;/hf/global_step_276/eval_*_vllm/</code>. wandb project <code>helman-mem</code>.</p>
</section>

<section>
<h2>Reproduction</h2>
<pre>{html.escape("""# peers' answers (per model, training and test streams)
python scripts/peer_answers.py --model $HF/<model> --records data/mixed_train_big/train.jsonl --output outputs/peergen_new/<model>
# merge into k-peer streams (answer after </think> for reasoning peers)
python scripts/merge_peers.py --records data/mixed_train_big5/train.jsonl --out data/mixed_train_big6/train.jsonl --peer <model>=outputs/peergen_new/<model>/train*.jsonl
# frozen-judge features and the record along the stream (history before the answer)
python scripts/encode_context_features.py --input data/mixed_train_big6/train.jsonl --output outputs/context_features/q3_4b_big6_ph/train/shard0.pt --central-model $HF/Qwen3-4B --include-context --save-peer-hidden --num-peers 6
python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream train6 --order fixed --out outputs/gen/q3_4b/prompts_train6_fixed.jsonl
# RL data: peers + history as the policy prompt, 25% question-only events; controls: --prompt_source peers, --swap_history
python -m training.sigma_rl.build_rl_data --prompts outputs/gen/q3_4b/prompts_train6_fixed.jsonl --records data/mixed_train_big6/train.jsonl --out outputs/rl/data/q3_4b_6peer/train_mix75_memory.parquet --guided none --prompt_source memory --solo_fraction 0.25
# training (one arm, eight GPUs)
GPUS=0,1,2,3,4,5,6,7 EXP=q3_4b_6peer_mix75_memory TRAIN=... VAL=outputs/rl/data/q3_4b_6peer/val_memory.parquet bash training/scripts/train_grpo.sh actor_rollout_ref.rollout.n=8 memory.guided_rollouts=0 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001""")}</pre>
</section>
</div>
'''
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes); record rows {len(rrow) - 1}, arms {len(ARMS)}")


if __name__ == "__main__":
    main()
