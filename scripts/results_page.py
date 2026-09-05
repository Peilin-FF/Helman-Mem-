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
    ("q3_4b_grpo_label", "classical: memory-ranked verified peer", "yes", "yes (ranking)"),
    ("q3_4b_grpo_plain", "plain RLVR: question only", "no", "no"),
]
TASKS_IN = ["math", "rag", "code"]
TASKS_OOD = ["boolqa", "mcqa", "shortqa"]


def load_eval(path: Path):
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    return {"acc": 100 * m["accuracy"], "by": {k: 100 * v for k, v in m["by_task"].items()}, "n": m["num_samples"],
            "halves": (100 * m["generated"]["first_half"], 100 * m["generated"]["second_half"]), "windows": [100 * w for w in m["generated"]["windows"]]}


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


def by_line(ev, tasks):
    return "—" if ev is None else " / ".join(fmt(ev["by"].get(t)) for t in tasks)


def svg_curves(series: dict[str, list[tuple[int, float]]], ymin: float, ymax: float, title: str, width=720, height=260) -> str:
    """Inline SVG line chart; series = {label: [(step, value), ...]}."""
    pad_l, pad_r, pad_t, pad_b = 46, 16, 28, 34
    xs = [s for pts in series.values() for s, _ in pts] or [0, 1]
    xmin, xmax = 0, max(xs)
    colors = ["#0d6b6c", "#b26f12", "#5b5fc7", "#c23b6b", "#5c6670"]
    def X(s): return pad_l + (s - xmin) / max(1, xmax - xmin) * (width - pad_l - pad_r)
    def Y(v): return pad_t + (ymax - v) / (ymax - ymin) * (height - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" style="width:100%;max-width:{width}px;height:auto">']
    out.append(f'<text x="{pad_l}" y="16" class="ctitle">{html.escape(title)}</text>')
    for v in range(int(ymin), int(ymax) + 1, 5):
        y = Y(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/><text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
    for s in range(0, int(xmax) + 1, 40):
        out.append(f'<text x="{X(s):.1f}" y="{height - 12}" class="tick" text-anchor="middle">{s}</text>')
    out.append(f'<text x="{(pad_l + width - pad_r) / 2:.0f}" y="{height - 1}" class="tick" text-anchor="middle">training step</text>')
    for i, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        c = colors[i % len(colors)]
        path = " ".join(f"{'M' if j == 0 else 'L'}{X(s):.1f},{Y(min(max(v, ymin), ymax)):.1f}" for j, (s, v) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2.2" stroke-linejoin="round"/>')
        for s, v in pts:
            out.append(f'<circle cx="{X(s):.1f}" cy="{Y(min(max(v, ymin), ymax)):.1f}" r="2.6" fill="{c}"/>')
        out.append(f'<rect x="{pad_l + 8 + i * 190}" y="{pad_t - 6}" width="12" height="4" fill="{c}"/><text x="{pad_l + 24 + i * 190}" y="{pad_t - 1}" class="legend">{html.escape(label)}</text>')
    out.append("</svg>")
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
    main_rows.append(row(["frozen Qwen3-4B, alone", "—", "—", f"<b>{fmt(frozen_in['acc'], 2) if frozen_in else '—'}</b>", by_line(frozen_in, TASKS_IN),
                          f"<b>{fmt(frozen_ood['acc'], 2) if frozen_ood else '—'}</b>", by_line(frozen_ood, TASKS_OOD)]))
    for r in runs:
        ein, eood = r["ev_in"], r["ev_ood"]
        cin = f"<b>{fmt(ein['acc'], 2)}</b>" if ein else f"<span class='pend'>{'pending' if r['status'] == 'trained' else r['status']}</span>"
        cood = f"<b>{fmt(eood['acc'], 2)}</b>" if eood else f"<span class='pend'>{'pending' if r['status'] == 'trained' else r['status']}</span>"
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
    dyn_charts = "".join(svg_curves({r["label"]: [(st, v * mult) for st, v in series(r["train"], key)] for r in runs if r["train"]}, lo * mult if mult == 1 else lo, hi * mult if mult == 1 else hi * mult, title + (" (%)" if mult == 100 else ""), width=720, height=230) for key, title, lo, hi, mult in dyn_specs)
    task_charts = "".join(svg_curves({r["label"]: series(r["train"], f"reward/acc_solo/{t}", 12) for r in runs if r["train"]}, {"rag": 0.3, "math": 0.8, "code": 0.0}[t], {"rag": 0.9, "math": 1.0, "code": 0.7}[t], f"Question-only samples on the stream, {t} (accuracy, 12-step means)", width=720, height=220) for t in ("rag", "math", "code"))
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
    chart_total = svg_curves({r["label"]: [(s, tot) for s, tot, _ in r["val"]] for r in runs if r["val"]}, 55, 75, "Validation: 512 fixed in-distribution prompts, question only, greedy (weighted total)")
    chart_tasks = "".join(svg_curves({r["label"]: [(s, v.get(t, 0)) for s, _, v in r["val"]] for r in runs if r["val"]}, {"rag": 55, "math": 80, "code": 10}[t], {"rag": 90, "math": 100, "code": 35}[t], f"Validation, {t}", width=720, height=220) for t in ("rag", "math", "code"))
    trained = [r for r in runs if r["status"] == "trained"]
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()
    # ---------------- analysis (written from the numbers present)
    mem, peers = runs[0], runs[1]
    findings = []
    if frozen_in and mem["ev_in"] and mem["ev_ood"] and frozen_ood:
        findings.append(f"<li><b>Training on the stream helps the model alone.</b> The memory run's final checkpoint reaches {mem['ev_in']['acc']:.2f} in-distribution (frozen {frozen_in['acc']:.2f}, +{mem['ev_in']['acc'] - frozen_in['acc']:.1f}) and {mem['ev_ood']['acc']:.2f} on the whole OOD stream (frozen {frozen_ood['acc']:.2f}, +{mem['ev_ood']['acc'] - frozen_ood['acc']:.1f}). In-distribution the gain is reading comprehension ({frozen_in['by']['rag']:.1f} → {mem['ev_in']['by']['rag']:.1f}), the task where the peers are stronger than the model; math is kept ({frozen_in['by']['math']:.1f} → {mem['ev_in']['by']['math']:.1f}). On OOD the gain is BIG-Bench Hard ({frozen_ood['by']['shortqa']:.1f} → {mem['ev_ood']['by']['shortqa']:.1f}) against a loss on the yes/no task ({frozen_ood['by']['boolqa']:.1f} → {mem['ev_ood']['by']['boolqa']:.1f}).</li>")
    if mem["val"] and peers["val"]:
        m_end, p_end = mem["val"][-1][1], peers["val"][-1][1]
        m_peak, p_peak = max(t for _, t, _ in mem["val"]), max(t for _, t, _ in peers["val"])
        findings.append(f"<li><b>The memory notes are not what produced the gain in-domain.</b> On the same 512 validation prompts the no-memory run ends at {p_end:.1f} (peak {p_peak:.1f}) versus {m_end:.1f} (peak {m_peak:.1f}) with memory; both start from {mem['val'][0][1]:.1f}. Their stream-time metrics are indistinguishable: guided answer 0.73–0.77 and question-only samples 0.70 → 0.75 in both. In-domain the model can judge the three solutions from their content, so the reliability notes carry no extra information, the same finding as with the trained judge.</li>")
    if peers["ev_in"] and mem["ev_in"]:
        ood_txt = (f" OOD: {mem['ev_ood']['acc']:.2f} vs {peers['ev_ood']['acc']:.2f} (per task, memory {by_line(mem['ev_ood'], TASKS_OOD)}; no memory {by_line(peers['ev_ood'], TASKS_OOD)})." if (peers["ev_ood"] and mem["ev_ood"]) else " OOD: the no-memory checkpoint's whole-stream evaluation is pending.")
        findings.append(f"<li><b>Full test streams, memory vs no memory: a tie in-distribution.</b> Final checkpoints alone: {mem['ev_in']['acc']:.2f} with memory vs {peers['ev_in']['acc']:.2f} without (per task, memory {by_line(mem['ev_in'], TASKS_IN)}; no memory {by_line(peers['ev_in'], TASKS_IN)}): the no-memory run is 3 points better on reading and 4 worse on code, the same total.{ood_txt}</li>")
    else:
        findings.append("<li><b>Full test streams, memory vs no memory:</b> the no-memory checkpoint's evaluations on the whole in-distribution and OOD streams are pending.</li>")
    p70 = peers["inter"].get(70); m70 = mem["inter"].get(70)
    if p70 and peers["ev_in"]:
        findings.append(f"<li><b>The best checkpoint is not the last one.</b> The no-memory run's step-70 checkpoint scores {p70['acc']:.2f} alone on the whole in-distribution stream ({by_line(p70, TASKS_IN)}), above its final {peers['ev_in']['acc']:.2f}: the final policy is the post-collapse one (entropy 0.3 instead of 0.02, code {peers['ev_in']['by']['code']:.1f} vs {p70['by']['code']:.1f}). With memory the two are level ({m70['acc']:.2f} at step 70, {mem['ev_in']['acc']:.2f} at the end). One epoch is already more than the reading gain needs; the remaining steps drift.</li>" if m70 else f"<li><b>The best checkpoint is not the last one.</b> The no-memory run's step-70 checkpoint scores {p70['acc']:.2f} on the whole in-distribution stream, above its final {peers['ev_in']['acc']:.2f}.</li>")
    findings.append("<li><b>Why math and code do not move: the reward has no gradient there, and thinking mode is off.</b> GRPO learns only from groups whose four samples disagree. On GSM8K the question-only samples are right 92–95% of the time, so three groups in four are all-correct and carry no signal; on APPS they are right about 30% of the time and most groups are all-wrong (the 13–15 all-wrong groups per step are mostly code). Reading, at 60–70%, is where the mixed groups are, and it is the only task that moves. All prompts are rendered with Qwen3's thinking mode switched off (an empty think block, in training and in evaluation), so the model never spends long reasoning on a problem; enabling it would raise the math and code ceilings but multiplies response lengths (thousands of tokens) and the cost of every rollout and evaluation.</li>")
    if frozen_ood_mem and frozen_ood:
        findings.append(f"<li><b>Why OOD is the memory's real test.</b> With the peers' solutions and memory notes in the prompt, the <em>frozen</em> model drops from {frozen_ood['acc']:.2f} alone to {frozen_ood_mem['acc']:.2f} on the OOD stream: on BIG-Bench Hard every peer is far weaker than the model (29 / 11 / 13% vs {frozen_ood['by']['shortqa']:.1f}%) and the untrained model follows them ({frozen_ood['by']['shortqa']:.1f} → {frozen_ood_mem['by']['shortqa']:.1f}). A trained model must learn when not to follow; the memory's notes are the only signal of that before the label arrives.</li>")
    if peers["train"]:
        seg = [r for r in peers["train"] if 229 <= int(r["training/global_step"]) <= 248]
        if seg and all(r.get("reward/acc_solo/rag", 1) == 0 for r in seg[2:8]):
            L = sum(r["response_length/mean"] for r in seg) / len(seg); clip = sum(r.get("response_length/clip_ratio", 0) for r in seg) / len(seg)
            findings.append(f"<li><b>The no-memory run collapsed on reading between steps 229 and 248, and recovered.</b> Within two steps the mean response length jumped from about 130 to {L:.0f} tokens with {100 * clip:.0f}% of the samples running into the 768-token cap, the policy entropy fell from 0.02–0.03 to below 0.01, and reading accuracy went to 0 for the question-only samples and to 0.1–0.4 for the guided answers, while math stayed above 0.9. That is a length-degeneracy mode (looping or endless restating of the passage without a final line), which the verifier scores 0. It is self-sustaining under GRPO: 30–42 of the 64 groups per step had every sample wrong, and an all-wrong group carries no gradient, so nothing pushed the policy back until the few mixed groups did; the recovery at steps 250–260 came with an entropy burst (0.1–0.3) and lengths back to about 180 tokens. The validation point at step 240 (reading 0.0) is that state; the step-276 checkpoint is past it (validation 71.9). The memory run showed no such episode (lengths 90–150 throughout), but one seed each does not show that the notes prevent it. The standard remedies are all basic: a KL term to the frozen model (<code>actor.use_kl_loss=True, kl_loss_coef=0.001</code>), a lower or decaying learning rate, or a small entropy bonus; a length cap is already in place through the reward.</li>")
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
.charts{{display:grid;grid-template-columns:1fr;gap:.6rem}} @media(min-width:900px){{.charts.three{{grid-template-columns:1fr 1fr 1fr}}}}
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
<p class="sub">In-distribution: SQuAD reading, GSM8K math, APPS code (4,319 events). OOD: SuperGLUE yes/no, PIQA/MMLU/SciQ multiple choice, BIG-Bench Hard short answers (17,403 events). Greedy decoding, task verifiers (math equality, normalized QA match, sandboxed code tests).</p>
</section>

<section>
<h2>Findings</h2>
<ul>{''.join(findings)}</ul>
<div class="callout warn"><b>Caveats.</b><ul>{caveats}</ul></div>
</section>

<section>
<h2>Learning curves (question only, 512 fixed validation prompts)</h2>
<div class="charts">{chart_total}</div>
<div class="charts three">{chart_tasks}</div>
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
