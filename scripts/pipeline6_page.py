"""The six-peer pipeline page: the memory steers the central model's attention (no history text), pilot training with thinking on.

  python scripts/pipeline6_page.py      -> artifacts/helman_mem_pipeline6.html
Reads (when present): record quality under outputs/gen/q3_4b/probe5/record_*.json, the kernel check
outputs/gen/q3_4b/tilt_check2/report.json, the frozen model's reference runs outputs/gen/q3_4b/base6_think/<stream>6_<cond>/,
and the pilot arms outputs/rl/pilot_{tilt,peers}/ (metrics.jsonl, hf/global_step_*/eval_<stream>6_<cond>/eval_metrics.json).
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
CONDS = [("tilt", "peers + memory tilt"), ("peers", "peers, no tilt"), ("solo", "alone"), ("tilt_swapped", "peers + swapped tilt")]
VAL_WEIGHTS = {"rag": 231, "math": 162, "code": 119}   # the 512 validation events by task


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f1(x, d=1, suffix=""):
    return "—" if x is None else f"{x:.{d}f}{suffix}"


def eval_acc(d: Path):
    m = load(d / "eval_metrics.json")
    return None if m is None else 100 * m["accuracy"]


def last_ck(exp: str):
    ck = sorted(glob.glob(str(ROOT / f"outputs/rl/{exp}/hf/global_step_*")), key=lambda p: int(p.rsplit("_", 1)[-1]))
    return Path(ck[-1]) if ck else None


def val_curve(exp: str):
    """(training steps logged, [(step, validation accuracy % weighted by the task counts)])."""
    m = ROOT / f"outputs/rl/{exp}/metrics.jsonl"
    if not m.exists():
        return 0, []
    rows = [json.loads(l) for l in m.open() if l.strip()]
    pts = []
    for r in rows:
        per = {t: float(r[f"val-core/{t}/acc/mean@1"]) for t in VAL_WEIGHTS if f"val-core/{t}/acc/mean@1" in r}
        if per:
            w = sum(VAL_WEIGHTS[t] for t in per)
            pts.append((int(r.get("training/global_step", len(pts))), 100 * sum(VAL_WEIGHTS[t] * v for t, v in per.items()) / w))
    dedup = {}
    for st, v in pts:   # a restarted run appends to the same file: keep the last value per step
        dedup[st] = v
    pts = sorted(dedup.items())
    return len(rows), pts


# ----------------------------------------------------------------------------------------------- SVG pipeline
def svg_pipeline() -> str:
    W, H = 1180, 430
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Six-peer training pipeline with the attention tilt" style="width:100%;height:auto">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dhead"/></marker></defs>']

    def box(x, y, w, h, t, sub, cls="dbox"):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/><text x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle" class="dtitle">{html.escape(t)}</text>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle" class="dsub">{html.escape(sub)}</text>')

    def arrow(x1, y1, x2, y2, cls="darrow"):
        return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#ah)"/>'

    s.append('<text x="20" y="22" class="dcol">1 · data and the record</text>')
    s.append(box(20, 36, 190, 60, "six peers answer", "same prompt, T=0.2, per-task budget"))
    s.append(box(250, 36, 190, 60, "graded", "F1 ≥ 0.5 / exact / tests"))
    s.append(box(480, 36, 190, 60, "six-peer streams", "train 17,709 · in-dist 4,319 · OOD 17,403"))
    s.append(box(710, 36, 200, 60, "frozen-judge features", "question + each peer answer"))
    s.append(box(950, 36, 210, 60, "Bayesian record  p_i", "read before write, along the stream"))
    s.append(arrow(210, 66, 250, 66)); s.append(arrow(440, 66, 480, 66)); s.append(arrow(670, 66, 710, 66)); s.append(arrow(910, 66, 950, 66))
    s.append('<text x="20" y="152" class="dcol">2 · the memory steers the attention, not the text</text>')
    s.append(box(20, 166, 300, 70, "question + all six solutions", "plain prompt: no reliability text at all"))
    s.append(box(360, 166, 400, 70, "tilt on peer i's tokens: + γ · log(p_i / max_j p_j)", "every layer and head · vLLM kernels / HF mask", "dbox strong"))
    s.append(box(800, 166, 360, 70, "central model answers, thinking off", "group of 8 samples; 25% of events question-only"))
    s.append(arrow(1055, 96, 600, 166)); s.append(arrow(320, 201, 360, 201)); s.append(arrow(760, 201, 800, 201))
    s.append('<text x="20" y="292" class="dcol">3 · learning and evaluation</text>')
    s.append(box(20, 306, 260, 70, "labels after the answer", "verifier reward; GRPO; KL 0.01"))
    s.append(box(320, 306, 260, 70, "record updated", "peers' verified labels of the event"))
    s.append(box(620, 306, 540, 70, "tests on both streams, thinking off", "peers + tilt · peers · alone · swapped tilt"))
    s.append(arrow(980, 236, 150, 306)); s.append(arrow(280, 341, 320, 341)); s.append(arrow(450, 306, 1055, 96, "darrow thin")); s.append(arrow(580, 341, 620, 341))
    s.append('<text x="590" y="412" text-anchor="middle" class="dsub">softmax(s + b) = Norm(A ⊙ c), c_i = (p_i / max p)^γ: the favourite peer is untouched, the others are damped; γ = 0 is the base model</text>')
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
.eq{font-family:"JetBrains Mono",monospace;font-size:.92rem;background:var(--code);border-radius:6px;padding:.5rem .8rem;margin:.4rem 0;max-width:100ch;overflow-x:auto}
ol,ul{max-width:100ch} li{margin:.35rem 0}
"""


REPRO = """# frozen-judge features and the record along the stream (read before write)
python scripts/encode_context_features.py --input data/mixed_train_big6/train.jsonl --output outputs/context_features/q3_4b_big6_ph/train/shard0.pt --central-model $HF/Qwen3-4B --include-context --save-peer-hidden --num-peers 6
python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream train6 --order fixed --out outputs/gen/q3_4b/prompts_train6_fixed.jsonl
# kernel check: the tilt inside vLLM equals the HF mask hook
python scripts/check_vllm_tilt.py --stage vllm --out outputs/gen/q3_4b/tilt_check2 && python scripts/check_vllm_tilt.py --stage hf --out outputs/gen/q3_4b/tilt_check2
# pilot data: peers prompt, 25% question-only, every 7th event; the peer blocks' spans ride in extra_info
python -m training.sigma_rl.build_rl_data --prompts outputs/gen/q3_4b/prompts_train6_fixed.jsonl --records data/mixed_train_big6/train.jsonl --out outputs/rl/data/q3_4b_6peer/pilot_peers.parquet --guided none --prompt_source peers --solo_fraction 0.25 --every 7
# the tilt arm (control: drop the attn_* overrides)
GPUS=0,1,2,3,4,5,6,7 EXP=pilot_tilt TRAIN=outputs/rl/data/q3_4b_6peer/pilot_peers.parquet VAL=outputs/rl/data/q3_4b_6peer/val_peers.parquet bash training/scripts/train_grpo.sh \
  data.enable_thinking=True data.max_response_length=4096 data.max_prompt_length=4608 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  data.attn_gamma=3.0 actor_rollout_ref.rollout.attn_bias=True actor_rollout_ref.model.attn_bias=True actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa
# tests (thinking on, whole streams): --attn_gamma 3 [--swap_record] with --mode peers, or --mode peers / solo without it
python -m tests.experiments.common.evaluate_memory_generator --central_model $HF/Qwen3-4B --checkpoint outputs/rl/pilot_tilt/hf/global_step_40 --engine vllm --thinking on --max_new_tokens 4096 --prompts outputs/gen/q3_4b/prompts_ood6_probe.jsonl --records data/ood6/test.jsonl --mode peers --attn_gamma 3 --output .../eval_ood6_tilt"""


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
    # ---- kernel check
    chk = load(ROOT / "outputs/gen/q3_4b/tilt_check2/report.json") or load(ROOT / "outputs/gen/q3_4b/tilt_check/report.json")
    if chk:
        crow = [row(["vLLM continuation", "vs HF with the tilt", "vs HF without the tilt", "HF argmax = vLLM token", "greedy tokens equal (same condition)"], True),
                row(["with the tilt", f"{chk['tilt_vs_hf_tilt']:.4f}", f"{chk['tilt_vs_hf_plain']:.4f}", f"{100 * chk['argmax_agree_tilt']:.1f}%", f"{100 * chk['greedy_match_tilt']:.1f}%"]),
                row(["without the tilt", f"{chk['plain_vs_hf_tilt']:.4f}", f"{chk['plain_vs_hf_plain']:.4f}", f"{100 * chk['argmax_agree_plain']:.1f}%", f"{100 * chk['greedy_match_plain']:.1f}%"])]
        check_html = f"<div class='tbl'><table>{''.join(crow)}</table></div><p class='sub'>Mean absolute difference of the log-probability of vLLM's own greedy tokens ({chk['n']} six-peer prompts with a non-flat record, 48 tokens each) when HF eager attention with the mask hook re-scores them with and without the tilt. Equality up to bf16 noise (about 0.01) in the matching condition and a clear gap in the other is the pass criterion; both passes ran in one engine, so the prefix cache was exercised across tilts.</p>"
    else:
        check_html = "<p class='pend'>kernel check pending</p>"
    def cell(v):
        return f1(v, 1) if v is not None else "<span class='pend'>pending</span>"

    def comp_rows(label, base_dir=None, exp=None, conds=(("tilt", "peers + memory"), ("peers", "peers, no memory"), ("solo", "question only"))):
        out = []
        ck = last_ck(exp) if exp else None
        for c, cl in conds:
            vals = []
            for s_ in ("indist", "oodfull"):
                d = (ROOT / f"outputs/gen/q3_4b/{base_dir}/{s_}6_{c}") if base_dir else ((ck / f"eval_{s_}6_{c}") if ck else None)
                vals.append(eval_acc(d) if d else None)
            out.append(row([label if not out else "", cl] + [cell(v) for v in vals]))
        return out
    comp = [row(["model", "input", "in-dist (4,319)", "OOD (all 17,403)"], True)]
    comp += comp_rows("Base central model", base_dir="base6_nothink", conds=(("tilt", "peers + memory"), ("peers", "question + peers"), ("solo", "question only")))
    comp += comp_rows("Ours: trained under the memory tilt", exp="run3b_tilt")
    comp += comp_rows("Control: trained on question + peers, no memory", exp="ctrl3b_peers")
    comp += comp_rows("Question-only arm: trained on the question alone", exp="solo3b_q")
    repro = html.escape(REPRO)
    page = f'''<title>Six-Peer Training Pipeline</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style>
<div class="page">
<header>
<div class="eyebrow">helman-mem · six-peer pipeline · the memory steers the attention · updated {stamp}</div>
<h1>Six-Peer Training Pipeline</h1>
<p class="thesis">The central model (Qwen3-4B) answers with six peer solutions in front of it. The memory never enters the prompt as text: its estimate of each peer's reliability at this question, a Bayesian linear record (the exact posterior of a linear-Gaussian model of correctness) over the frozen judge's features, is added to the attention scores on that peer's tokens in every layer and head, inside vLLM for rollouts and inside the trainer's forward for the update. Nothing is filtered or ranked away; correctness is revealed only after the answer. This page records the data, the mechanism, the engine work that made it fast, the pilot regime, and the results as they arrive.</p>
</header>

<section>
<h2>Pipeline</h2>
<div class="diagram">{svg_pipeline()}</div>
</section>

<section>
<h2>Peers</h2>
<div class="tbl"><table>{''.join(prow)}</table></div>
<p class="sub">Accuracy on the training stream under the stored labels (token-F1 ≥ 0.5 for SQuAD, exact equality for GSM8K, executed tests for APPS). The new peers' answers were generated with the same task prompt, chat template, sampling (temperature 0.2, top-p 0.95) and per-task token budgets as the original three (96 tokens on the OOD tasks); the reasoning peer keeps a 4,096-token budget and its stored answer is the text after its think block. Unformatted or truncated answers count as wrong: the answer format is part of the task, and the central model is graded the same way.</p>
</section>

<section>
<h2>The record's quality with more peers</h2>
<div class="tbl"><table>{''.join(rrow) if len(rrow) > 1 else row(["<span class='pend'>record quality pending</span>"])}</table></div>
<p class="sub">Read-before-write on the prompt files. AUC: a right peer is printed with a higher estimate than a wrong one. "Favourite right where peers split": the peer with the highest estimate is right on events where the peers are neither all right nor all wrong; the running mean per peer and task is the trivial history it is measured against.</p>
</section>

<section>
<h2>The mechanism: the record tilts the attention</h2>
<p>The prompt is the plain peers prompt: system text, the question (with options or passage), the six solutions as <code>Peer 1 … Peer 6</code>, the task instruction. The record gives each peer i an estimate p_i = σ(w_tᵀ φ_i) from the Kalman state w_t over the frozen judge's features φ_i of that peer's answer, computed before the event is written. For every query in every layer and head, the score onto each token of peer i's block receives</p>
<div class="eq">b_i = γ · log( p_i / max_j p_j ),   γ = 3;   b_i = 0 for the favourite peer, and for every peer when the record is flat (spread ≤ 0.1)</div>
<div class="eq">softmax(s + b) = Norm(A ⊙ c),   c_i = (p_i / max_j p_j)^γ</div>
<p>so the un-normalised attention on peer i is multiplied by c_i and nothing else changes: the favourite is untouched, the others are damped by their reliability ratio, and γ = 0 gives back the base model exactly. Training-free this gave 62.3 vs 59.3 (swapped record) in-distribution and 67.5 vs 62.9 on OOD with thinking off. The reference policy of the KL term carries the same tilt, so the penalty compares the same attention.</p>
<h3>Engine: the tilt inside vLLM</h3>
<p>Serving engines take no per-token attention bias, and HF generation with the mask hook runs at 100–200 tokens per second per GPU (151–324 ms per decode step at batch 32), which put thinking-mode training out of reach. vLLM's Triton attention backend is Python source, so <code>feedback_state/vllm_attn_bias.py</code> rewrites its two kernels at install time: one float per KV slot lives next to the KV cache and is added to the scaled score of every query on that key, in the prefill kernel (cached context blocks and the chunk's own tokens) and in the paged decode kernel. Each request gets its bias vector through an in-process registry keyed by its prompt tokens; prefix-cache block hashes include the bias prefix, so KV blocks are shared only between requests with the same tilt (the 8 samples of one prompt) and never across tilts. Measured on one GPU with thinking on: 4,192 tokens/s with the stock FlashAttention kernel, 1,995 tokens/s with the patched Triton kernels, against 100–200 for HF. The trainer's own forward uses SDPA with the additive mask (<code>use_remove_padding=False</code>).</p>
{check_html}
</section>

<section>
<h2>Training regime</h2>
<ul>
<li><b>Data.</b> The whole six-peer training stream (17,701 events after the prompt-length filter), the six peer answers in the prompt for 75% of the events and the question alone for 25%; labels only after the answer; shuffled batches (the record is order-free, so each prompt's tilt still comes from its own read-before-write prefix).</li>
<li><b>Group of 8, thinking off.</b> Eight sampled answers per event under the same prompt and the same tilt, 768-token budget, verifier reward, GRPO with clip 0.2, learning rate 1e-6, KL 0.01 to the reference, full parameters, 64 events per step, eight GPUs, about 100 s per step.</li>
<li><b>Schedule.</b> 40 steps from the frozen model, then the full epoch (277 steps) restarted from that checkpoint with the reference reset to it and a new shuffle seed. Validation every 20 steps on 512 held-out in-distribution events, checkpoints every 40.</li>
<li><b>Three arms.</b> Ours trains with the tilt (γ = 3) in rollouts, log-probs and the update; the control follows the identical schedule with the same prompts and no memory anywhere; the question-only arm follows the identical schedule with the question alone in every prompt (no peers, no memory), which separates plain RL on the tasks from learning to read peers.</li>
</ul>
</section>

<section>
<h2>Evaluation protocol</h2>
<p>The final checkpoint of each run and the frozen model are evaluated on the two test streams, thinking off, greedy, in three input settings: the question with the six peer answers and the memory's tilt (deployed), the same prompt without the tilt, and the question alone. The record used at test time is built along the test stream, read before write, exactly as in training.</p>
</section>

<section>
<h2>Results</h2>
<div class="tbl"><table>{''.join(comp)}</table></div>
<p class="sub">Accuracy in percent, thinking off, 768-token answers, greedy decoding with vLLM. In-distribution: the whole six-peer test stream (4,319 events: GSM8K, SQuAD, APPS). OOD: the whole OOD stream (17,403 events: yes/no, multiple-choice and short-answer questions never seen in training), so the record accumulates the full stream's feedback rather than a quarter of it. Standard error is about 0.7 in-distribution and 0.35 on OOD. "Peers + memory" is the deployed setting: the six peer answers in the prompt and the record's tilt on the attention. The base rows are the frozen Qwen3-4B. The control is trained with exactly the same schedule, data and settings as ours but never sees the memory; the question-only arm never sees a peer either. Swapped record (peers + memory with the highest estimate on the least trusted peer), in-dist / full OOD: frozen 60.2 / 64.2, ours 73.7 / 66.7, control 70.5 / 67.7, question-only arm 69.5 / 67.8.</p>
<p><b>Reading.</b> The memory is a test-time channel every model reads: over the same prompt it adds +2.2 / +4.9 (in-dist / OOD) on the frozen model, +0.9 / +2.8 on ours, +0.5 / +2.2 on the control and +0.4 / +3.9 on the question-only arm, and the swapped record costs every model 2 to 10 points. On the full OOD stream the effect is concentrated in the short-answer tasks: yes/no and multiple-choice barely move between conditions (84-86 throughout), while short-answer runs from 31.3 (frozen, swapped record) to 58.9 (control, deployed). On OOD the memory is worth about four times what the RL training is: the frozen model deployed reaches 74.0 against 75.1-75.5 for the three trained arms, so training adds about 1.3 points and the tilt adds 4.9. Training under the tilt does not beat training without it (75.9 vs 74.9 in-dist, about 1.5 standard errors; 75.0 vs 75.1 OOD). The question-only arm shows that peer prompts in training cost no own ability (71.5 alone, level with the control, below ours at 73.4) and that learning to read peers is worth 2 to 3 points deployed in-dist (73.0 against 75.9 / 74.9); a model that never saw a peer gains only 1 point from six solutions in-dist and loses 1.4 from them on OOD.</p>
</section>

<section>
<h2>Reproduction</h2>
<pre>{repro}</pre>
</section>
</div>
'''
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes); record rows {len(rrow) - 1}, check {'yes' if chk else 'no'}, pending cells {page.count('pending')}")


if __name__ == "__main__":
    main()
