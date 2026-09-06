"""Build the results page (artifacts/helman_mem_results.html) from the local mirrors of metrics.jsonl and eval_metrics.json.

  python scripts/results_page.py            # after `rproj results` (+ `rproj get outputs/rl/<EXP>/metrics.jsonl` for the per-step files)

Every run present under outputs/rl/q3_4b_grpo_* is included; missing evaluations show as pending.  Re-run as results accumulate.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "outputs/rl"
FROZEN = ROOT / "outputs/gen/q3_4b/frozen"
OUT = ROOT / "artifacts/helman_mem_results.html"
VAL_N = {"rag": 231, "math": 162, "code": 119}
RUNS = [  # (dir, label, labels before the answer?, memory?)
    ("q3_4b_grpo_memory", "ours: peers + memory notes", "no", "yes"),
    ("q3_4b_grpo_peers", "no memory: peers only", "no", "no"),
    ("q3_4b_grpo_peers_labeled", "classical: peers with verified labels", "yes", "no"),
    ("q3_4b_grpo_plain", "plain RLVR: question only", "no", "no"),
]
TASKS_IN = ["math", "rag", "code"]
TASKS_OOD = ["boolqa", "mcqa", "shortqa"]


def load_eval(path: Path):
    """vLLM-decoded evaluations only (<dir>_vllm); the HF-generate results were removed."""
    cands = [path.parent.with_name(path.parent.name + "_vllm") / path.name]
    for f in cands:
        if f.exists():
            m = json.loads(f.read_text())
            return {"acc": 100 * m["accuracy"], "by": {k: 100 * v for k, v in m["by_task"].items()}, "n": m["num_samples"], "engine": m.get("engine", "hf"),
                    "halves": (100 * m["generated"]["first_half"], 100 * m["generated"]["second_half"]), "windows": [100 * w for w in m["generated"]["windows"]]}
    return None


def load_metrics(d: Path):
    f = d / "metrics.jsonl"
    if not f.exists():
        return [], []
    rows = [json.loads(l) for l in f.open()]
    val = []
    for r in rows:
        if any(k.startswith("val-core") for k in r):
            v = {k.split("/")[1]: 100 * r[k] for k in r if k.startswith("val-core") and k.endswith("mean@1")}
            tot = sum(v[t] * VAL_N[t] for t in v) / sum(VAL_N[t] for t in v)
            val.append((int(r["training/global_step"]), tot, v))
    train = [r for r in rows if "reward/acc_solo" in r]
    return val, train


def fmt(x, d=1):
    return "—" if x is None else f"{x:.{d}f}"


def acc_cell(ev, d=2):
    return "—" if ev is None else f"<b>{ev['acc']:.{d}f}</b>" + ("" if ev.get("engine") == "vllm" else " <span class='sub'>hf</span>")


def by_line(ev, tasks):
    return "—" if ev is None else " / ".join(fmt(ev["by"].get(t)) for t in tasks)


COLORS = ["#0d6b6c", "#b26f12", "#5b5fc7", "#c23b6b", "#5c6670", "#2e8b57"]


def svg_curves(series: dict[str, list[tuple[int, float]]], ymin: float, ymax: float, title: str, width=1100, height=300) -> str:
    """Inline SVG line chart (legend rendered as HTML below the plot so labels never overlap)."""
    pad_l, pad_r, pad_t, pad_b = 52, 20, 30, 36
    xs = [s for pts in series.values() for s, _ in pts] or [0, 1]
    xmin, xmax = 0, max(xs)
    def X(s): return pad_l + (s - xmin) / max(1, xmax - xmin) * (width - pad_l - pad_r)
    def Y(v): return pad_t + (ymax - v) / (ymax - ymin) * (height - pad_t - pad_b)
    out = [f'<figure class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" style="width:100%;height:auto">']
    out.append(f'<text x="{pad_l}" y="18" class="ctitle">{html.escape(title)}</text>')
    step = 5 if ymax - ymin > 20 else (0.1 if ymax - ymin <= 1 else 1)
    v = ymin
    while v <= ymax + 1e-9:
        y = Y(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/><text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
        v += step
    for sx in range(0, int(xmax) + 1, 40):
        out.append(f'<text x="{X(sx):.1f}" y="{height - 12}" class="tick" text-anchor="middle">{sx}</text>')
    out.append(f'<text x="{(pad_l + width - pad_r) / 2:.0f}" y="{height - 1}" class="tick" text-anchor="middle">training step</text>')
    legend = []
    for i, (label, pts) in enumerate(series.items()):
        c = COLORS[i % len(COLORS)]
        legend.append(f'<span class="lg"><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><rect x="0" y="3" width="22" height="4" rx="2" fill="{c}"/><circle cx="11" cy="5" r="4" fill="{c}"/></svg><span style="color:{c};font-weight:600">{html.escape(label)}</span></span>')
        if not pts:
            continue
        path = " ".join(f"{'M' if j == 0 else 'L'}{X(s):.1f},{Y(min(max(v, ymin), ymax)):.1f}" for j, (s, v) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2.2" stroke-linejoin="round"/>')
        for s_, v_ in pts:
            out.append(f'<circle cx="{X(s_):.1f}" cy="{Y(min(max(v_, ymin), ymax)):.1f}" r="2.6" fill="{c}"/>')
    out.append("</svg>" + '<figcaption class="legend-row">' + "".join(legend) + "</figcaption></figure>")
    return "\n".join(out)


def main() -> None:
    frozen_in = load_eval(FROZEN / "eval_indist_shuffled0_solo/eval_metrics.json")
    frozen_ood = load_eval(FROZEN / "eval_ood_shuffled0_solo/eval_metrics.json")
    frozen_in_mem = load_eval(FROZEN / "eval_indist_shuffled0_memory/eval_metrics.json")
    frozen_in_peers = load_eval(FROZEN / "eval_indist_shuffled0_peers/eval_metrics.json")
    frozen_ood_mem = load_eval(FROZEN / "eval_ood_shuffled0_memory/eval_metrics.json")
    frozen_ood_peers = load_eval(FROZEN / "eval_ood_shuffled0_peers/eval_metrics.json")
    runs = []
    for d, label, before, mem in RUNS:
        rd = RL / d
        val, train = load_metrics(rd)
        ev_in = load_eval(rd / "hf/global_step_276/eval_indist_solo/eval_metrics.json")
        ev_ood = load_eval(rd / "hf/global_step_276/eval_ood_solo/eval_metrics.json")
        inter = {s: load_eval(rd / f"hf/global_step_{s}/eval_indist_solo/eval_metrics.json") for s in (70, 140, 210)}
        runs.append(dict(dir=d, label=label, before=before, mem=mem, val=val, train=train, ev_in=ev_in, ev_ood=ev_ood, inter=inter,
                         status=("trained" if len(train) >= 276 else (f"training, step {len(train)}" if train else "queued"))))
    # ---------------- tables
    def row(cells, head=False):
        tag = "th" if head else "td"
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
    main_rows = [row(["model", "labels before the answer", "memory", "in-distribution (4,319)", "math / reading / code", "OOD (17,403)", "boolqa / mcqa / shortqa"], True)]
    main_rows.append(row(["frozen Qwen3-4B, alone", "—", "—", acc_cell(frozen_in), by_line(frozen_in, TASKS_IN), acc_cell(frozen_ood), by_line(frozen_ood, TASKS_OOD)]))
    for r in runs:
        ein, eood = r["ev_in"], r["ev_ood"]
        cin = acc_cell(ein) if ein else f"<span class='pend'>{'pending' if r['status'] == 'trained' else r['status']}</span>"
        cood = acc_cell(eood) if eood else f"<span class='pend'>{'pending' if r['status'] == 'trained' else r['status']}</span>"
        main_rows.append(row([r["label"], r["before"], r["mem"], cin, by_line(ein, TASKS_IN), cood, by_line(eood, TASKS_OOD)]))
    ref_rows = [row(["frozen Qwen3-4B with the peers in the prompt", "in-distribution", "OOD"], True),
                row(["with peers + memory notes", fmt(frozen_in_mem["acc"], 2) if frozen_in_mem else "—", (fmt(frozen_ood_mem["acc"], 2) + " (" + by_line(frozen_ood_mem, TASKS_OOD) + ")") if frozen_ood_mem else "—"]),
                row(["with peers, no notes", fmt(frozen_in_peers["acc"], 2) if frozen_in_peers else "—", (fmt(frozen_ood_peers["acc"], 2) + " (" + by_line(frozen_ood_peers, TASKS_OOD) + ")") if frozen_ood_peers else "pending"])]
    # validation table
    steps = sorted({s for r in runs for s, _, _ in r["val"]})
    val_rows = [row(["step"] + [r["label"] for r in runs if r["val"]], True)]
    for s in steps:
        cells = [str(s)]
        for r in runs:
            if not r["val"]:
                continue
            hit = next(((tot, v) for st, tot, v in r["val"] if st == s), None)
            cells.append(f"{tot:.1f} <span class='sub'>({v.get('rag', 0):.0f}/{v.get('math', 0):.0f}/{v.get('code', 0):.0f})</span>" if hit and (tot := hit[0]) is not None and (v := hit[1]) is not None else "—")
        val_rows.append(row(cells))
    # intermediate checkpoints
    inter_rows = [row(["checkpoint (step)", "70", "140", "210", "276 (final)"], True)]
    for r in runs:
        if r["status"] != "trained":
            continue
        inter_rows.append(row([r["label"]] + [fmt(r["inter"][s]["acc"], 2) if r["inter"].get(s) else "pending" for s in (70, 140, 210)] + [fmt(r["ev_in"]["acc"], 2) if r["ev_in"] else "pending"]))
    # stream windows
    def windows(train, key, w=46):
        out = []
        for a in range(0, len(train), w):
            seg = [x.get(key) for x in train[a:a + w] if x.get(key) is not None]
            out.append(sum(seg) / len(seg) * 100 if seg else None)
        return out
    stream_rows = [row(["run", "metric"] + [f"steps {a + 1}–{min(a + 46, 276)}" for a in range(0, 276, 46)], True)]
    for r in runs:
        if not r["train"]:
            continue
        for key, name in (("reward/acc_guided", "guided answer (stream-time)"), ("reward/acc_solo", "question-only samples")):
            stream_rows.append(row([r["label"], name] + [fmt(x) for x in windows(r["train"], key)]))
    # ---------------- training dynamics (the same per-step series wandb shows)
    def series(train, key, w=6):
        pts = []
        for a in range(0, len(train), w):
            seg = [x.get(key) for x in train[a:a + w] if x.get(key) is not None and x.get(key) == x.get(key)]
            if seg:
                pts.append((int(train[min(a + w - 1, len(train) - 1)]["training/global_step"]), sum(seg) / len(seg)))
        return pts
    dyn_specs = [("response_length/mean", "Response length (tokens, mean per step)", 60, 520, 1),
                 ("response_length/clip_ratio", "Share of samples hitting the 768-token cap", 0, 0.6, 100),
                 ("actor/entropy_loss", "Policy entropy", 0, 0.35, 1),
                 ("actor/grad_norm", "Gradient norm (before clipping)", 0, 8, 1),
                 ("actor/ppo_kl", "Policy movement per update (ppo_kl)", -0.15, 0.25, 1),
                 ("memory/zero_groups", "All-wrong groups per step (of 64)", 0, 45, 1)]
    dyn_charts = "".join(svg_curves({r["label"]: [(st, v * mult) for st, v in series(r["train"], key)] for r in runs if r["train"]}, lo * mult if mult == 1 else lo, hi * mult if mult == 1 else hi * mult, title + (" (%)" if mult == 100 else ""), height=270) for key, title, lo, hi, mult in dyn_specs)
    task_charts = "".join(svg_curves({r["label"]: series(r["train"], f"reward/acc_solo/{t}", 12) for r in runs if r["train"]}, {"rag": 0.3, "math": 0.8, "code": 0.0}[t], {"rag": 0.9, "math": 1.0, "code": 0.7}[t], f"Question-only samples on the stream, {t} (accuracy, 12-step means)", height=260) for t in ("rag", "math", "code"))
    def phase_table(train):
        phases = [(0, 46), (46, 92), (92, 138), (138, 184), (184, 230), (230, 276)]
        keys = [("reward/acc_solo", "question-only acc"), ("reward/acc_guided", "guided acc"), ("reward/acc_solo/rag", "question-only, reading"), ("response_length/mean", "response length"),
                ("response_length/clip_ratio", "cap hits"), ("actor/entropy_loss", "entropy"), ("actor/grad_norm", "grad norm"), ("actor/ppo_kl", "ppo_kl"), ("memory/zero_groups", "all-wrong groups"), ("memory/guided_success", "guided correct (of 64)")]
        rows_ = [row(["steps"] + [f"{a + 1}–{b}" for a, b in phases], True)]
        for k, name in keys:
            cells = []
            for a, b in phases:
                seg = [x.get(k) for x in train[a:b] if x.get(k) is not None and x.get(k) == x.get(k)]
                cells.append(f"{sum(seg) / len(seg):.3f}" if seg else "—")
            rows_.append(row([name] + cells))
        return "".join(rows_)
    dyn_tables = "".join(f"<h3>{html.escape(r['label'])}</h3><div class='tbl'><table>{phase_table(r['train'])}</table></div>" for r in runs if r["train"])
    dyn_text = """<ul>
<li><b>Both runs learn the same thing in the same order.</b> Question-only accuracy on the stream rises from 0.68 to about 0.75 within the first 50 steps, all of it on reading (0.62 → 0.68 with memory, 0.64 → 0.67 without), while math stays at 0.92–0.95 and code at 0.28–0.38. The guided answer starts 4 points above the question-only samples and the gap closes to about 0.01 by the middle of the stream: what the model does with the peers in front of it, it then does alone. Policy movement per update (ppo_kl) is tiny in both (≤ 0.012 with memory), the PPO clip is almost never active (clip fraction ≤ 0.006), so the updates are effectively on-policy REINFORCE with the group baseline.</li>
<li><b>The memory run drifts slowly; the no-memory run breaks once.</b> With memory: entropy falls steadily from 0.05 to 0.02, the gradient norm rises from 1.9 to 3.6, response length is flat at 105–110 tokens for two thirds of the epoch and then grows to 145 without ever hitting the cap (cap hits ≤ 1%), and validation oscillates between 66 and 70 after its peak. Without memory: from step 93 the response length creeps up (139 vs 104 tokens at the same phase), cap hits rise to 2–3%, and ppo_kl per update is three to four times larger (0.034–0.044 in steps 139–230); at step 229 the length jumps to 450–510 tokens with 40–56% of the samples cut at the cap, entropy collapses to 0.007, reading accuracy goes to zero for 20 steps, and the gradient norm is non-finite at step 231 (verl skips that update). Recovery at steps 250–260 comes with an entropy burst to 0.3, which is still the state of the final checkpoint: it is a more random policy than the pre-collapse one, so its sampled stream accuracy is lower (0.59 in the last phase) while its greedy validation is fine (71.9).</li>
<li><b>Where the memory notes act.</b> The guided answer with notes is correct on 47–48 of 64 prompts in every phase; without notes it is the same until the collapse (44 in the last phase). The reliability notes did not change what was learned in-domain; if they had an effect it is on stability, and one seed per regime cannot show that.</li>
<li><b>What to change for the next runs.</b> The runs use constant lr 1e-6, no KL term and no length control, the plainest GRPO. The collapse is the textbook failure of that setting on a task with long outputs. The basic remedies, in order of preference: a KL penalty to the frozen model (<code>actor.use_kl_loss=True actor.kl_loss_coef=0.001</code>, one more reference forward per step), a cosine-decayed learning rate, and keeping the best validation checkpoint rather than the last.</li>
</ul>"""
    # charts
    chart_total = svg_curves({r["label"]: [(s, tot) for s, tot, _ in r["val"]] for r in runs if r["val"]}, 55, 75, "Validation: 512 fixed in-distribution prompts, question only, greedy (weighted total)", height=320)
    chart_tasks = "".join(svg_curves({r["label"]: [(s, v.get(t, 0)) for s, _, v in r["val"]] for r in runs if r["val"]}, {"rag": 55, "math": 80, "code": 10}[t], {"rag": 90, "math": 100, "code": 35}[t], f"Validation, {t}", height=260) for t in ("rag", "math", "code"))
    trained = [r for r in runs if r["status"] == "trained"]
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()
    # ---------------- analysis (written from the numbers present)
    mem, peers = runs[0], runs[1]
    lab = next((r for r in runs if r["dir"].endswith("peers_labeled")), None)
    plain = next((r for r in runs if r["dir"].endswith("_plain")), None)
    done = [r for r in runs if r["ev_in"] and r["ev_ood"]]
    findings = []
    def gain(ev, ref, key=None):
        a = ev["by"][key] if key else ev["acc"]; b = ref["by"][key] if key else ref["acc"]
        return f"{a - b:+.1f}"
    if frozen_in and frozen_ood and mem["ev_in"] and mem["ev_ood"]:
        # 1. headline
        if plain and plain["ev_in"] and plain["ev_ood"]:
            findings.append(f"<li><b>1. Plain question-only GRPO is the best model alone, on both streams.</b> It reaches {plain['ev_in']['acc']:.2f} in-distribution ({gain(plain['ev_in'], frozen_in)} over the frozen model) and {plain['ev_ood']['acc']:.2f} on the whole OOD stream ({gain(plain['ev_ood'], frozen_ood)}), improves every task including math ({frozen_in['by']['math']:.1f} → {plain['ev_in']['by']['math']:.1f}) and code ({frozen_in['by']['code']:.1f} → {plain['ev_in']['by']['code']:.1f}), and every OOD task including BIG-Bench Hard ({frozen_ood['by']['shortqa']:.1f} → {plain['ev_ood']['by']['shortqa']:.1f}). Its checkpoints rise monotonically over the epoch ({', '.join('%.1f' % plain['inter'][st]['acc'] for st in (70, 140, 210) if plain['inter'].get(st))} → {plain['ev_in']['acc']:.1f}). It is the only run that never sees a peer solution.</li>")
        # 2. peers in the prompt: reading up, the rest down
        peerish = [r for r in done if r is not plain]
        if peerish:
            rag = ", ".join(f"{r['ev_in']['by']['rag']:.1f}" for r in peerish); code = ", ".join(f"{r['ev_in']['by']['code']:.1f}" for r in peerish)
            findings.append(f"<li><b>2. Peer solutions in the prompt buy reading and cost everything else.</b> Every peer-conditioned run learns reading comprehension far beyond the frozen model (reading {rag} vs {frozen_in['by']['rag']:.1f} frozen; plain {plain['ev_in']['by']['rag']:.1f}" + ("" if not plain else "") + "), the task where the peers are stronger than the model, but loses code ({code} vs {frozen_in['by']['code']:.1f}) and gains nothing on math. On OOD, where the peers are weaker than the model, the peer-conditioned runs end between {min(r['ev_ood']['acc'] for r in peerish):.2f} and {max(r['ev_ood']['acc'] for r in peerish):.2f}, all below plain GRPO's {plain['ev_ood']['acc']:.2f}: what they learned from the peers' answers is partly the peers' style.</li>")
        # 3. memory vs no memory
        if peers["ev_in"] and peers["ev_ood"]:
            findings.append(f"<li><b>3. The memory's notes do not separate ours from the no-memory control.</b> Same protocol, same prompts, notes on or off: {mem['ev_in']['acc']:.2f} vs {peers['ev_in']['acc']:.2f} in-distribution, {mem['ev_ood']['acc']:.2f} vs {peers['ev_ood']['acc']:.2f} OOD (memory {by_line(mem['ev_ood'], TASKS_OOD)}; no memory {by_line(peers['ev_ood'], TASKS_OOD)}). On the training stream the two are indistinguishable (guided answer 0.73–0.77, question-only samples 0.70 → 0.75, all-wrong groups 14 per step in both). This repeats the judge result: in-domain the model can read the three solutions' content itself, so a reliability note adds no information. The one place the notes show is OOD BIG-Bench Hard, where the memory run keeps the largest gain of the peer-conditioned runs ({mem['ev_ood']['by']['shortqa']:.1f} vs {peers['ev_ood']['by']['shortqa']:.1f}) while losing the yes/no task ({mem['ev_ood']['by']['boolqa']:.1f} vs {peers['ev_ood']['by']['boolqa']:.1f}).</li>")
        # 4. classical baseline
        if lab and lab["ev_in"] and lab["ev_ood"]:
            findings.append(f"<li><b>4. The classical labels-before baseline overfits to the peers.</b> With every peer solution marked verified correct or incorrect before the answer, the model reaches the best reading of all runs ({lab['ev_in']['by']['rag']:.1f}) and the best in-distribution total among peer-conditioned runs ({lab['ev_in']['acc']:.2f}, level with plain GRPO's {plain['ev_in']['acc']:.2f}), but the worst OOD ({lab['ev_ood']['acc']:.2f}; BIG-Bench Hard {lab['ev_ood']['by']['shortqa']:.1f}, below the frozen {frozen_ood['by']['shortqa']:.1f}). Labels before the answer let it copy the right peer in-domain and leave it dependent on peers it does not have OOD. The labels-after regimes ({mem['ev_ood']['acc']:.2f}, {peers['ev_ood']['acc']:.2f}) generalise better than it, plain GRPO best of all.</li>")
        # 5. checkpoint selection
        pb = max([(st, e) for st, e in peers["inter"].items() if e] + [(276, peers["ev_in"])], key=lambda x: x[1]["acc"]) if peers["ev_in"] else None
        if pb:
            findings.append(f"<li><b>5. Select checkpoints by validation, not by the last step.</b> The no-memory run peaks at step {pb[0]} with {pb[1]['acc']:.2f} alone on the whole in-distribution stream ({by_line(pb[1], TASKS_IN)}) and ends at {peers['ev_in']['acc']:.2f} after its collapse; the memory run is flat from step 70 ({mem['inter'][70]['acc']:.2f}) to the end ({mem['ev_in']['acc']:.2f}). Only plain GRPO keeps improving to the last step. One epoch is more than the reading gain needs; for the peer-conditioned runs the second half of the epoch is drift.</li>")
    else:
        findings.append("<li>Full-stream evaluations are pending.</li>")
    if peers["train"]:
        seg = [r for r in peers["train"] if 229 <= int(r["training/global_step"]) <= 248]
        if seg and all(r.get("reward/acc_solo/rag", 1) == 0 for r in seg[2:8]):
            L = sum(r["response_length/mean"] for r in seg) / len(seg); clip = sum(r.get("response_length/clip_ratio", 0) for r in seg) / len(seg)
            findings.append(f"<li><b>6. Training dynamics: slow drift everywhere, one collapse.</b> All runs use constant lr 1e-6, no KL term and no length control. Entropy falls from about 0.05 to 0.02 and the gradient norm doubles over the epoch in every run; policy movement per update (ppo_kl) stays below 0.02 except in the no-memory run, where it is three to four times larger from step 139 on. That run then collapsed on reading at steps 229–248: response length jumped from about 130 to {L:.0f} tokens with {100 * clip:.0f}% of samples cut at the 768-token cap, entropy fell to 0.007, reading accuracy went to zero, and a non-finite gradient at step 231 was skipped by verl. GRPO cannot correct an all-wrong group (no gradient), so the state persisted until mixed groups pulled it back with an entropy burst to 0.3, which the final checkpoint still carries. The basic remedies are a KL penalty to the frozen model (<code>actor.use_kl_loss=True, kl_loss_coef=0.001</code>), a cosine-decayed learning rate, and validation-based checkpoint selection.</li>")
    findings.append("<li><b>7. Why math and code barely move: the reward has no gradient there, and thinking mode is off.</b> GRPO learns only from groups whose four samples disagree. On GSM8K the question-only samples are right 92–95% of the time, so most groups are all-correct; on APPS they are right about 30% of the time and most groups are all-wrong. Reading, at 60–70%, is where the mixed groups are, and it moves most. All prompts render Qwen3 with thinking switched off (an empty think block, in training and evaluation), so the model never reasons at length; the thinking-mode runs (memory and plain, 70 steps, 4096-token responses, with thinking-mode frozen references) test whether that changes the reasoning tasks.</li>")
    caveats = ("<li>Single seed per regime; differences of one to two points on the 512-prompt validation set are within noise (±2 points at n=512). The full-stream numbers (4,319 and 17,403 events) are the ones to compare.</li>"
               "<li>The OOD stream is a different task mix (yes/no, multiple choice, BIG-Bench Hard), not harder questions; the frozen model is already at or above the best peer there.</li>"
               "<li>All runs: Qwen3-4B, full parameters, one epoch of the 17,709-event training stream in order, 64 prompts × 4 question-only samples + 1 guided answer per step, GRPO with clip 0.2, lr 1e-6, no KL, verifier reward after the answer.</li>")
    pending = [f"{r['label']}: {r['status']}" + ("" if r["ev_in"] and r["ev_ood"] else " → full-stream evaluation pending") for r in runs if not (r["ev_in"] and r["ev_ood"])]
    page = f'''<title>Helman-Mem Results</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{{--paper:#f6f7f5;--ink:#1b2229;--muted:#5c6670;--rule:#d7dcd8;--code:#eef1ee;--accent:#0d6b6c;--amber:#b26f12;--amber-bg:#fbf3e4;--teal-bg:#e6f2f1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#12171b;--ink:#e4e8e6;--muted:#98a3ab;--rule:#2b333a;--code:#1a2126;--accent:#45b8b2;--amber:#dfa24d;--amber-bg:#2a2216;--teal-bg:#152a2a}}}}
:root[data-theme="dark"]{{--paper:#12171b;--ink:#e4e8e6;--muted:#98a3ab;--rule:#2b333a;--code:#1a2126;--accent:#45b8b2;--amber:#dfa24d;--amber-bg:#2a2216;--teal-bg:#152a2a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;font-size:16.5px;line-height:1.55}}
h1,h2,h3{{font-family:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;text-wrap:balance;letter-spacing:-0.01em;margin:0}} h1{{font-size:2.4rem;line-height:1.08}} h2{{font-size:1.5rem;margin:0 0 .5rem}} h3{{font-size:1.1rem;margin:1rem 0 .3rem}}
.page{{max-width:1120px;margin:0 auto;padding:2.2rem 1.4rem 4rem}} header{{border-bottom:1px solid var(--rule);padding-bottom:1.4rem;margin-bottom:1.4rem}}
.eyebrow{{font-family:"JetBrains Mono",monospace;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:.5rem}}
.thesis{{font-size:1.12rem;max-width:70ch;margin:.8rem 0}} p{{max-width:78ch}} section{{padding:1.3rem 0;border-top:1px solid var(--rule)}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0 1rem;font-size:.93rem;font-variant-numeric:tabular-nums}} th,td{{text-align:left;vertical-align:top;padding:.45rem .55rem;border-bottom:1px solid var(--rule)}} th{{font-family:"Bricolage Grotesque",sans-serif;font-size:.86rem;color:var(--muted)}}
.tbl{{overflow-x:auto}} .sub{{color:var(--muted);font-size:.85em}} .pend{{color:var(--amber);font-family:"JetBrains Mono",monospace;font-size:.8em}}
.grid{{stroke:var(--rule);stroke-width:1}} .tick,.legend,.ctitle{{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px}} .ctitle{{fill:var(--ink);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:700}}
.callout{{border-left:4px solid var(--accent);background:var(--teal-bg);padding:.7rem .95rem;border-radius:0 8px 8px 0;margin:.9rem 0;max-width:80ch}} .callout.warn{{border-left-color:var(--amber);background:var(--amber-bg)}}
ul{{max-width:82ch}} li{{margin:.35rem 0}} code{{font-family:"JetBrains Mono",monospace;font-size:.86em;background:var(--code);padding:.05em .3em;border-radius:4px}}
.charts{{display:grid;grid-template-columns:1fr;gap:.9rem}} figure.chart{{margin:0;background:var(--code);border:1px solid var(--rule);border-radius:10px;padding:.5rem .6rem .4rem}} .legend-row{{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;padding:.4rem .2rem 0;font-family:"JetBrains Mono",monospace;font-size:.8rem;color:var(--ink)}} .legend-row .lg{{display:inline-flex;align-items:center;gap:.45rem;white-space:nowrap}} .legend-row .lg svg{{flex:none}} @media(min-width:900px){{.charts.three{{grid-template-columns:1fr 1fr 1fr}}}}
</style>
<div class="page">
<header>
<div class="eyebrow">helman-mem · labels only after the answer · updated {stamp}</div>
<h1>Helman-Mem Results</h1>
<p class="thesis">Qwen3-4B trained on a stream of peer solutions whose correctness is revealed only after the model has answered, with or without the Kalman reliability memory, against the classical labels-before baselines and plain RLVR. Every number below is the central model answering the question alone, graded on the complete test streams.</p>
</header>

<section>
<h2>Test streams, final checkpoints</h2>
<div class="tbl"><table>{''.join(main_rows)}</table></div>
<div class="tbl"><table>{''.join(ref_rows)}</table></div>
<p class="sub">In-distribution: SQuAD reading, GSM8K math, APPS code (4,319 events). OOD: SuperGLUE yes/no, PIQA/MMLU/SciQ multiple choice, BIG-Bench Hard short answers (17,403 events). Greedy decoding, task verifiers (math equality, normalized QA match, sandboxed code tests). All numbers are decoded with vLLM (greedy, in-process engine); the earlier transformers-generate results agreed within a point (98% identical verdicts on a 300-event check) and were removed.</p>
</section>

<section>
<h2>Findings</h2>
<ul>{''.join(findings)}</ul>
<div class="callout warn"><b>Caveats.</b><ul>{caveats}</ul></div>
</section>

<section>
<h2>Learning curves (question only, 512 fixed validation prompts)</h2>
<div class="charts">{chart_total}</div>
<div class="charts">{chart_tasks}</div>
<div class="tbl"><table>{''.join(val_rows)}</table></div>
<p class="sub">Weighted total and (reading / math / code). Every 20 training steps; step 0 is the frozen model on the same prompts.</p>
</section>

<section>
<h2>Training dynamics (per-step series, as on wandb)</h2>
{dyn_text}
<div class="charts">{dyn_charts}</div>
<div class="charts">{task_charts}</div>
{dyn_tables}
<p class="sub">Six-step means (12 for the per-task curves) of the per-step training metrics. ppo_kl = mean log-ratio between the updated and the sampling policy on the rollout tokens; all-wrong groups = prompts whose four question-only samples all scored 0 (no gradient from that group).</p>
</section>

<section>
<h2>Along the training stream</h2>
<div class="tbl"><table>{''.join(stream_rows)}</table></div>
<p class="sub">Per-step accuracies averaged over six windows of the stream (fixed order). Guided answer = one greedy answer with the regime's guided prompt; question-only samples = four samples per prompt at temperature 1.</p>
</section>

<section>
<h2>Checkpoints on the whole in-distribution stream</h2>
<div class="tbl"><table>{''.join(inter_rows)}</table></div>
</section>

<section>
<h2>Status</h2>
<ul>{''.join(f"<li>{html.escape(p)}</li>" for p in pending) or "<li>all runs evaluated</li>"}</ul>
<p class="sub">Runs: <code>outputs/rl/q3_4b_grpo_*</code>; evaluations: <code>hf/global_step_N/eval_{{indist,ood}}_solo/eval_metrics.json</code>; wandb project <code>helman-mem</code>. Page generated by <code>scripts/results_page.py</code>.</p>
</section>
</div>
'''
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes); runs: " + ", ".join(f"{r['dir']}={r['status']}" for r in runs))


if __name__ == "__main__":
    main()
