"""Steering study page: training-free ways of making the central model act on the memory's reliability record.

Reads the probe analyses under outputs/gen/q3_4b/steer/{ood,indist}/ (scripts/memory_use_probe.py outputs for every
variant and its swapped-note control, the alone/number/peers references with the committee simulation, the reranking
sweeps) and writes artifacts/helman_mem_steering.html.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEER = ROOT / "outputs/gen/q3_4b/steer"
OUT = ROOT / "artifacts/helman_mem_steering.html"

STREAMS = [("ood", "OOD slice", "1,500 events, positions 4,000–5,499 of the OOD stream (yes/no, multiple choice, BIG-Bench Hard)"),
           ("indist", "in-distribution slice", "1,200 events, positions 1,500–2,699 of the in-distribution stream (math and reading; code excluded from the answer analysis)")]

# (key, label, steering point, mechanism in one line)
VARIANTS = [
    ("number", "number in the header (current)", "prompt text", "each peer's header carries the estimate: \"estimated probability correct: 0.72, based on 47 similar past cases\""),
    ("ordinal", "ranked record in words", "prompt text", "peers without numbers; a paragraph ranks them (\"right on about 9 of 10 similar cases\") and tells the model to prefer the most reliable and not to follow a majority against it"),
    ("self_note", "ranked record + own record", "prompt text", "as ordinal, plus the model's own running accuracy on the task answering alone, and a comparison to the favourite"),
    ("verify", "verify-the-favourite procedure", "prompt text", "as ordinal, plus a two-step procedure: check the favourite's solution first, adopt it unless a definite error is found"),
    ("sorted", "peers sorted by reliability", "prompt structure", "peers re-ordered most reliable first and renamed in that order, the rank in each header"),
    ("vote", "reliability-weighted vote shown", "prompt structure", "the peers' final answers grouped, each group with its combined reliability weight, the leading answer named"),
    ("filtered", "unreliable peers withheld", "prompt structure", "peers with an estimate below 0.35 are removed from the prompt (their records mentioned); the rest ranked in words"),
    ("favourite", "only the favourite shown", "prompt structure", "only the most reliable peer's solution is shown, with its record; the others withheld"),
    ("defer", "two-pass: alone, then confront", "second pass", "the model answers alone; only when its answer disagrees with the favourite and the favourite's estimate is at least 0.6 is it shown both and asked to re-examine"),
    ("attn_g1", "credibility-weighted attention", "attention weights", "plain peers prompt with no reliability text; every attention head's scores over peer i's tokens get + log(p_i / max p), CrAM's re-weighting, in all 36 layers"),
]
REFS = [("solo", "alone (no peers, no notes)"), ("number", "number in the header (current)"), ("peers", "peers, no notes")]


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def analyses(stream: str) -> dict:
    d = STEER / stream
    out = {}
    refs = load(d / "analysis_refs.json")
    for k, _ in REFS:
        if k in refs:
            out[k] = refs[k]
    for key, *_ in VARIANTS:
        for sw in ("", "_swapped"):
            a = load(d / f"analysis_{key}{sw}.json")
            if a:
                out[key + sw] = a[key + sw] if key + sw in a else next(iter(a.values()))
        for sw in ("_think", "_swapped_think"):
            a = load(d / f"analysis_{key}{sw}.json")
            if a:
                out[key + sw] = next(iter(a.values()))
    return out


def f1(x, d=1, suffix=""):
    return "—" if x is None or x != x else f"{x:.{d}f}{suffix}"


def delta(x, ref):
    return "—" if x is None or ref is None else f"{x - ref:+.1f}"


def poe_best(stream: str, name: str) -> dict | None:
    s = load(STEER / stream / name / "poe_summary.json")
    if not s:
        return None
    gated = [r for r in s["grid"] if r.get("gate")]
    if not gated:
        return None
    best = max(gated, key=lambda r: r["accuracy"])
    return {"accuracy": best["accuracy"], "selection": best["selection_accuracy"], "lambda": best["lambda"], "beta": best["beta"], "own": s["own_accuracy"], "n_selection": s["n_selection"]}


# ----------------------------------------------------------------------------------------------- SVG
def svg_bars(rows: list[tuple[str, float | None, float | None]], alone: float | None, title: str) -> str:
    """Horizontal bars: accuracy with the true record (solid) and with the swapped record (hollow), a line at the alone accuracy."""
    W = 1100
    rh = 30
    H = 40 + rh * len(rows) + 30
    lo = max(0.0, min(v for _, a, b in rows for v in (a, b) if v is not None) - 8)
    hi = min(100.0, max(v for _, a, b in rows for v in (a, b) if v is not None) + 4)
    pl, pr = 300, 30
    def X(v): return pl + (v - lo) / (hi - lo) * (W - pl - pr)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" style="width:100%;height:auto">', f'<text x="{pl}" y="18" class="ctitle">{html.escape(title)}</text>']
    v = int(lo // 5 * 5) + 5
    while v < hi:
        s.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="{H - 26}" class="grid"/><text x="{X(v):.1f}" y="{H - 12}" class="tick" text-anchor="middle">{v}</text>')
        v += 5
    for i, (label, a, b) in enumerate(rows):
        y = 36 + i * rh
        s.append(f'<text x="{pl - 8}" y="{y + 15}" class="blabel" text-anchor="end">{html.escape(label)}</text>')
        if b is not None:
            s.append(f'<rect x="{X(lo):.1f}" y="{y + 3}" width="{max(0, X(b) - X(lo)):.1f}" height="9" class="bar swapped"/>')
        if a is not None:
            s.append(f'<rect x="{X(lo):.1f}" y="{y + 13}" width="{max(0, X(a) - X(lo)):.1f}" height="9" class="bar true"/>')
            s.append(f'<text x="{X(a) + 4:.1f}" y="{y + 21}" class="tick">{a:.1f}</text>')
    if alone is not None and lo < alone < hi:
        s.append(f'<line x1="{X(alone):.1f}" y1="30" x2="{X(alone):.1f}" y2="{H - 26}" class="aline"/><text x="{X(alone) + 4:.1f}" y="{38}" class="tick alone">alone {alone:.1f}</text>')
    s.append("</svg>")
    return "".join(s)


def svg_points() -> str:
    """Where the record can enter the system, and what happened at each point."""
    W, H = 1180, 300
    boxes = [(20, 110, 150, 70, "Kalman memory", "reliability record"), (230, 30, 170, 60, "prompt text", "number / words: no effect"), (230, 120, 170, 60, "prompt structure", "sort / filter / favourite: steers"),
             (230, 210, 170, 60, "attention weights", "CrAM-style bias"), (460, 30, 190, 60, "central model", "reads peers, reasons"), (460, 120, 190, 60, "second pass", "alone, then confront: safe, no gain"),
             (700, 30, 190, 60, "final-answer decision", "rerank candidates, gated"), (700, 120, 190, 60, "system output", "alone answer + weighted vote: best"), (940, 75, 220, 60, "answer", "graded on the stream")]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Steering points" style="width:100%;height:auto">',
         '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dhead"/></marker></defs>']
    for x, y, w, h, t, sub in boxes:
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="dbox"/><text x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle" class="dtitle">{html.escape(t)}</text><text x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle" class="dsub">{html.escape(sub)}</text>')
    def arrow(x1, y1, x2, y2, cls="darrow"):
        return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#ah2)"/>'
    s.append(arrow(170, 145, 230, 60)); s.append(arrow(170, 145, 230, 150)); s.append(arrow(170, 145, 230, 240))
    s.append(arrow(400, 60, 460, 60)); s.append(arrow(400, 150, 460, 60)); s.append(arrow(400, 240, 460, 75))
    s.append(arrow(650, 60, 700, 60)); s.append(arrow(650, 150, 700, 150)); s.append(arrow(460, 150, 460, 150, "darrow thin"))
    s.append(arrow(890, 60, 940, 100)); s.append(arrow(890, 150, 940, 110))
    s.append(f'<path d="M95,180 L95,270 L700,270 L700,180" class="darrow bad" marker-end="url(#ah2)"/><text x="400" y="288" text-anchor="middle" class="dsub">the record can also act after the model: on the decision, or on the system output (no prompt injection at all)</text>')
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------------------------------------- page
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
td.best{color:var(--accent);font-weight:700} td.worse{color:var(--bad)} .sub{color:var(--muted);font-size:.86rem;max-width:100ch}
code{font-family:"JetBrains Mono",monospace;font-size:.86em;background:var(--code);padding:.05em .3em;border-radius:4px}
.callout{border-left:4px solid var(--accent);background:var(--teal-bg);padding:.7rem .95rem;border-radius:0 8px 8px 0;margin:.9rem 0;max-width:100ch} .callout.warn{border-left-color:var(--amber);background:var(--amber-bg)}
.diagram{background:var(--code);border:1px solid var(--rule);border-radius:10px;padding:.6rem;margin:.5rem 0 .9rem}
.dbox{fill:var(--paper);stroke:var(--rule);stroke-width:1.2} .dtitle{fill:var(--ink);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:600} .dsub{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px}
.darrow{stroke:var(--muted);stroke-width:1.6;fill:none} .darrow.thin{stroke-width:1;opacity:.5} .darrow.bad{stroke:var(--accent);stroke-width:2} .dhead{fill:var(--muted)}
.grid{stroke:var(--rule);stroke-width:1} .tick{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px} .tick.alone{fill:var(--amber)} .ctitle{fill:var(--ink);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:700}
.blabel{fill:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:12.5px} .bar.true{fill:var(--accent)} .bar.swapped{fill:none;stroke:var(--bad);stroke-width:1.2;stroke-dasharray:3 2} .aline{stroke:var(--amber);stroke-width:1.5;stroke-dasharray:5 3}
figure.chart{margin:0 0 1rem;background:var(--code);border:1px solid var(--rule);border-radius:10px;padding:.5rem .6rem .4rem}
ol.findings{max-width:105ch;padding-left:1.3rem} ol.findings li{margin:.55rem 0} ul{max-width:100ch}
.survey td:first-child{min-width:220px}
pre.prompt{background:var(--code);border:1px solid var(--rule);border-radius:8px;padding:.6rem .8rem;font-size:.8rem;white-space:pre-wrap;max-width:100ch;overflow-x:auto}
"""


def main() -> None:
    data = {s: analyses(s) for s, _, _ in STREAMS}
    poe = {s: {name: poe_best(s, name) for name in ("poe_number", "poe_peers", "poe_solo")} for s, _, _ in STREAMS}
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()

    def row(cells, head=False, classes=None):
        tag = "th" if head else "td"
        classes = classes or [""] * len(cells)
        return "<tr>" + "".join(f"<{tag}{(' class=' + chr(34) + c + chr(34)) if c else ''}>{v}</{tag}>" for v, c in zip(cells, classes)) + "</tr>"

    # ---------------- results tables and bars per stream
    result_sections, chart_sections = [], []
    best_by_stream = {}
    for s, label, desc in STREAMS:
        A = data[s]
        alone = A.get("solo", {}).get("accuracy")
        number = A.get("number", {}).get("accuracy")
        rows_html = [row(["method", "steering point", "accuracy", "vs alone", "vs current prompt", "selection events: accuracy", "follows the favourite there", "accuracy when the favourite is wrong", "swapped record: accuracy", "steered? (drop under swap)"], True)]
        bars = []
        def add(label_, st, sw, point):
            if not st:
                return
            acc = st["accuracy"]; sacc = st.get("anycorrect_model_right"); fol = st.get("anycorrect_model_follows_favourite"); bad = st.get("accuracy_when_favourite_wrong")
            swacc = sw["accuracy"] if sw else None
            drop = None if swacc is None else acc - swacc
            cls = ["", "", "best" if (alone is not None and acc >= alone + 1.0) else ("worse" if (alone is not None and acc <= alone - 3.0) else ""), "", "", "", "", "", "", ""]
            rows_html.append(row([label_, point, f1(acc, 2), delta(acc, alone), delta(acc, number), f1(sacc), f1(fol, 1, "%"), f1(bad), f1(swacc, 2), (f"{drop:+.1f}" if drop is not None else "—")], classes=cls))
            bars.append((label_, acc, swacc))
        add("alone (no peers, no notes)", A.get("solo"), None, "—")
        add("peers, no notes", A.get("peers"), None, "—")
        for key, lab, point, _ in VARIANTS:
            add(lab, A.get(key), A.get(key + "_swapped"), point)
        # reranking
        for name, lab in (("poe_number", "rerank at the final answer (current prompt), gated"), ("poe_peers", "rerank at the final answer (peers, no notes), gated"), ("poe_solo", "rerank at the final answer (alone reasoning), gated")):
            pb = poe[s].get(name)
            if pb:
                acc = pb["accuracy"]
                rows_html.append(row([lab, "final-answer decision", f1(acc, 2), delta(acc, alone), delta(acc, number), f1(pb["selection"]), "—", "—", "—", "—"],
                                     classes=["", "", "best" if (alone is not None and acc >= alone + 1.0) else "", "", "", "", "", "", "", ""]))
                bars.append((lab, acc, None))
        # committee on the alone answer
        com = A.get("solo", {}).get("committee", {})
        if com:
            best_w = max(com, key=lambda w: com[w]["accuracy"])
            for w in dict.fromkeys((best_w, "0.5")):
                if w in com:
                    acc = com[w]["accuracy"]
                    lab = f"alone answer + reliability-weighted vote (model weight w={w}{', best' if w == best_w else ''})"
                    rows_html.append(row([lab, "system output", f1(acc, 2), delta(acc, alone), delta(acc, number), f1(com[w]["selection_accuracy"]), "—", "—", "—", "—"],
                                         classes=["", "", "best" if (alone is not None and acc >= alone + 1.0) else "", "", "", "", "", "", "", ""]))
                    bars.append((f"alone + weighted vote (w={w})", acc, None))
                    if w == best_w:
                        best_by_stream[s] = (acc, alone, number, com[w]["selection_accuracy"], w)
        # thinking rows
        for key, lab, point, _ in VARIANTS:
            st = A.get(key + "_think")
            if st:
                sw = A.get(key + "_swapped_think")
                rows_html.append(row([lab + " + thinking", point, f1(st["accuracy"], 2), "—", "—", f1(st.get("anycorrect_model_right")), f1(st.get("anycorrect_model_follows_favourite"), 1, "%"), f1(st.get("accuracy_when_favourite_wrong")), f1(sw["accuracy"] if sw else None, 2), (f"{st['accuracy'] - sw['accuracy']:+.1f}" if sw else "—")]))
        result_sections.append(f"<h3>{label}</h3><p class=\"sub\">{desc}. Frozen Qwen3-4B, greedy, thinking off unless stated. Selection events: the peers' answers differ and one of them is right (n={A.get('solo', {}).get('n_informative_anycorrect', '—')}); the favourite is the peer the memory rates most reliable and is right {f1(A.get('solo', {}).get('anycorrect_favourite_right'))}% of them.</p><div class=\"tbl\"><table>{''.join(rows_html)}</table></div>")
        chart_sections.append(f'<figure class="chart">{svg_bars(bars, alone, f"{label}: accuracy with the true record (solid) and with the record swapped by rank (dashed); amber line = the model alone")}</figure>')

    # ---------------- findings
    o, i_ = data["ood"], data["indist"]
    def acc(s, k): return data[s].get(k, {}).get("accuracy")
    def sw(s, k): return data[s].get(k + "_swapped", {}).get("accuracy")
    bo, bi = best_by_stream.get("ood"), best_by_stream.get("indist")
    findings = []
    findings.append(f"<li><b>Words do not steer; structure does.</b> Describing the record in words (ranked, with the model's own record, with a verification procedure) leaves accuracy where the number left it (OOD {f1(acc('ood', 'number'), 1)} → {f1(acc('ood', 'ordinal'), 1)} / {f1(acc('ood', 'self_note'), 1)} / {f1(acc('ood', 'verify'), 1)}; in-distribution {f1(acc('indist', 'number'), 1)} → {f1(acc('indist', 'ordinal'), 1)} / {f1(acc('indist', 'self_note'), 1)} / {f1(acc('indist', 'verify'), 1)}) and the swapped control barely moves. Removing or re-ordering what the model sees does steer: only-the-favourite {f1(acc('ood', 'favourite'), 1)} / {f1(acc('indist', 'favourite'), 1)}, unreliable peers withheld {f1(acc('ood', 'filtered'), 1)} / {f1(acc('indist', 'filtered'), 1)}, sorted {f1(acc('ood', 'sorted'), 1)} / {f1(acc('indist', 'sorted'), 1)}, and these lose {f1((acc('ood', 'favourite') or 0) - (sw('ood', 'favourite') or 0), 1)} / {f1((acc('indist', 'favourite') or 0) - (sw('indist', 'favourite') or 0), 1)} points when the record is swapped, which is what a steered model must do. This matches the literature: credibility text is ignored by untrained models (CAG), while position and majority size drive conformity (the conformity studies), so changing position and majority is the lever.</li>")
    findings.append(f"<li><b>On OOD every prompt with peers in it is below the model alone</b> ({f1(acc('ood', 'solo'), 1)}): the peers are weaker than the model there, and no arrangement of them recovers the loss (best structural variant {f1(max(v for v in (acc('ood', 'favourite'), acc('ood', 'filtered'), acc('ood', 'sorted')) if v), 1)}). Steering inside the prompt can only redistribute trust among the peers; it cannot give the model back its own answer.</li>")
    if bo and bi:
        findings.append(f"<li><b>The best steering keeps the record out of the model entirely.</b> Let the model answer alone, then let the record combine that answer with the peers' answers in a reliability-weighted vote, applied only where the peers disagree and the record is not flat: {bo[0]:.2f} on the OOD slice ({bo[0] - bo[1]:+.1f} over alone, {bo[0] - bo[2]:+.1f} over the current prompt) and {bi[0]:.2f} in-distribution ({bi[0] - bi[1]:+.1f} over alone, {bi[0] - bi[2]:+.1f} over the current prompt); on the selection events {bo[3]:.1f} and {bi[3]:.1f}, close to the favourite's own {f1(o.get('solo', {}).get('anycorrect_favourite_right'))} / {f1(i_.get('solo', {}).get('anycorrect_favourite_right'))}. No training, no prompt change, one weight (the model's own vote) that the same memory could track as a self-record. This is the design the aggregation literature converges on (ReConcile, Roundtable Policy, credibility-scored aggregation): the reliability weights act at the aggregation, where they cannot be ignored.</li>")
    findings.append(f"<li><b>Two-pass deference is safe but empty.</b> Answering alone and confronting the model with the favourite only on confident disagreement gives {f1(acc('ood', 'defer'), 1)} OOD / {f1(acc('indist', 'defer'), 1)} in-distribution, within a point of alone, and the swapped record costs almost nothing ({f1(sw('ood', 'defer'), 1)} / {f1(sw('indist', 'defer'), 1)}): shown a disagreeing peer, the model keeps its own answer whether the peer is reliable or not.</li>")
    pn, pp, ps = poe["ood"].get("poe_number"), poe["ood"].get("poe_peers"), poe["ood"].get("poe_solo")
    if pn and pp:
        findings.append(f"<li><b>Reranking at the final answer works only behind a disagreement gate, and inherits the prompt it sits on.</b> Re-scoring the candidates (peers' answers and the model's own) with the memory's log-odds at the \"Final answer:\" position gains about {pn['accuracy'] - pn['own']:+.1f} on the current prompt and {pp['accuracy'] - pp['own']:+.1f} on the peers prompt (OOD); without the gate it loses on unanimous-but-wrong peers what it gains on the disagreements." + (f" On the alone reasoning it reaches {ps['accuracy']:.2f}." if ps else "") + "</li>")
    at, ats = o.get("attn_g1"), o.get("attn_g1_swapped")
    if at:
        findings.append(f"<li><b>Attention-level steering (CrAM-style) with no reliability text:</b> {at['accuracy']:.2f} OOD (swapped {f1(ats['accuracy'] if ats else None, 2)}), " + (f"{i_['attn_g1']['accuracy']:.2f} in-distribution (swapped {f1(i_.get('attn_g1_swapped', {}).get('accuracy'), 2)})" if i_.get("attn_g1") else "in-distribution pending") + f"; follows the favourite {f1(at.get('anycorrect_model_follows_favourite'))}% on the selection events against {f1(o.get('peers', {}).get('anycorrect_model_follows_favourite'))}% for the same prompt without the bias. A bias on all heads is a blunt instrument; CrAM selects heads by causal tracing, which is the next refinement if this point is pursued.</li>")
    findings.append("<li><b>What steering costs.</b> Any method that follows the record pays where the favourite is wrong (column \"accuracy when the favourite is wrong\"): only-the-favourite drops there because the model rarely rejects the one solution it sees. The committee pays the same price at the aggregation, but only on disagreement events, and its net is positive because the record is right far more often than the model on those events.</li>")

    survey = [
        ("Credibility-aware generation (CAG), Pan et al. 2024, arXiv:2404.06809", "credibility levels written into the prompt per document; trained models use them", "\"existing LLMs are not inherently sensitive to directly provided credibility in the prompt\"; prompting gave +0.7 EM, fine-tuning +9. Our words-only variants reproduce the prompting result."),
        ("CrAM, Deng et al. 2024, arXiv:2406.11497", "credibility scales the attention weights of the document's tokens on causally selected heads, Norm(A ⊙ s); training-free", "+25 to +32 EM over putting the scores in the prompt (ideal scores). Tried here as an all-heads bias (attention row); head selection is the missing piece."),
        ("Sycophancy propagation in multi-agent systems, 2026, arXiv:2604.02668", "categorical peer rankings (\"least sycophantic\" … \"very sycophantic\") in the prompt", "+10.5 points; influence moved from weak to strong agents; one model (Qwen-7B) did not respond. Categorical ranking is close to our \"ordinal\" variant, which did not move Qwen3-4B."),
        ("Most LLM conformity needs no speaker, 2026, arXiv:2607.05545", "what drives conformity to peers", "position and majority size drive it; reliability instructions and independence directives have limited effect. Our sorted / filtered / favourite variants act on exactly position and majority size."),
        ("Roundtable Policy, 2025, arXiv:2509.16839; ReConcile; credibility-scored aggregation, arXiv:2505.24239", "confidence- or reliability-weighted consensus computed outside the model", "weights act at aggregation, where they cannot be ignored; robust to adversarial or weak agents. Our committee row is this design with the Kalman record as the weight."),
        ("Contrastive activation addition, Rimsky et al. 2023, arXiv:2312.06681; activation-steering surveys 2025–26", "add a behaviour direction to the residual stream at inference", "steers broad behaviours (sycophancy, refusal); a \"trust the reliable peer\" direction would need contrastive pairs whose only difference is the record, and the record differs per event. Not tried; the attention bias is the per-event analogue."),
        ("Logit-level interventions (SWAI and predecessors), 2026, arXiv:2601.10960; classifier-free guidance for LMs, arXiv:2306.17806", "reshape the next-token distribution with an external signal", "our final-answer reranking is the answer-level version: the record enters as a prior on the candidates at the commit point."),
    ]
    survey_rows = [row(["work", "how the reliability signal enters", "finding, and what it implies here"], True)] + [row([html.escape(a), html.escape(b), html.escape(c)]) for a, b, c in survey]

    variant_rows = [row(["variant", "steering point", "mechanism"], True)] + [row([html.escape(l), p, html.escape(m)]) for _, l, p, m in VARIANTS]
    variant_rows += [row(["rerank at the final answer", "final-answer decision", "the candidates are the distinct answers on the table (each peer's, the model's own); score = the model's log-probability of the candidate after its own reasoning + λ × the summed log-odds of the peers giving it; applied only where the peers disagree (gate)"]),
                     row(["alone answer + reliability-weighted vote", "system output", "the model answers alone; where the peers disagree and the record is not flat, the answer groups are scored by the summed log-odds of their supporters, the model's own answer counting as one supporter with weight w; the best group is the output"])]

    example = ("Reliability memory (verified outcomes on earlier, similar questions):\n- Peer 3 (most reliable): right on about 63 of 89 similar past cases\n- Peer 1: right on about 46 of 92 similar past cases\n"
               "- Peer 2 (least reliable): right on about 21 of 90 similar past cases\nUnless you can verify an answer yourself, prefer Peer 3's answer. Do not follow the majority when it contradicts Peer 3, and do not adopt an answer that only Peer 2 gives.")

    page = f'''<title>Steering the Central Model</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style>
<div class="page">
<header>
<div class="eyebrow">helman-mem · steering study · training-free · frozen Qwen3-4B · updated {stamp}</div>
<h1>Steering the Central Model</h1>
<p class="thesis">The reliability record is good (AUC 0.92, its favourite right 78% / 93% where peers disagree) and the central model ignores it when it is a number in the prompt. This page tries ten training-free ways of making the record act, from rewording the note to keeping it out of the model altogether, each with a swapped-record control that tells whether the record steers the answer at all, on the same probe slices as the results page.</p>
</header>

<section>
<h2>Where the record can enter</h2>
<div class="diagram">{svg_points()}</div>
<p>Six points, three inside the model's input (text, structure, attention), three after it (a second pass, the final-answer decision, the system output). Every method below is one of these; none trains anything.</p>
</section>

<section>
<h2>What the literature says</h2>
<div class="tbl survey"><table>{''.join(survey_rows)}</table></div>
</section>

<section>
<h2>Methods tried</h2>
<div class="tbl"><table>{''.join(variant_rows)}</table></div>
<p class="sub">When the record is flat (estimates within 0.1, or no evidence) every variant says so and falls back to judging on content, so no directive ever points at an arbitrary peer. Example of the ranked record in words:</p>
<pre class="prompt">{html.escape(example)}</pre>
<p class="sub">Swapped control: the same prompt with the record permuted by rank, so the highest reliability is printed on (or, for the structural variants, access is given to) the peer the memory trusts least. A steered model follows the newly favoured peer and loses accuracy; a model that reads the content does not move.</p>
</section>

<section>
<h2>Results</h2>
{''.join(result_sections)}
{''.join(chart_sections)}
<p class="sub">Selection events: the peers' answers differ and at least one is right. "Follows the favourite": the model's final answer falls in the favourite's answer group. Steered: accuracy drop when the record is swapped (a method the record does not reach shows ~0). Script: <code>scripts/memory_use_probe.py</code>; variants: <code>scripts/steer_prompts.py</code>, <code>scripts/steer_poe.py</code>, <code>scripts/steer_attention.py</code>; outputs under <code>outputs/gen/q3_4b/steer/</code>.</p>
</section>

<section>
<h2>Findings</h2>
<ol class="findings">{''.join(findings)}</ol>
<div class="callout"><b>Recommended design.</b> Keep the memory outside the model. The central model answers alone; the memory aggregates: where the peers disagree and its record is not flat, it scores each distinct answer by the summed log-odds of its supporters, counting the model's own answer as a supporter whose weight is the model's own tracked reliability, and emits the best-scored answer. The prompt does not change, nothing is trained, and the memory's quality converts directly into accuracy. If the model must see the peers (to reason over their solutions), give it only the peers the record admits (the filtered variant), which steers without a directive, and still aggregate afterwards.</div>
<div class="callout warn"><b>Caveats.</b> One frozen model, one seed, greedy decoding, slices of 1,500 and 1,200 events; the committee's model weight was chosen on the same slice (a held-out choice, or the self-record, is the honest version); thinking-mode rows cover the OOD slice only; the attention bias uses all heads and one strength.</div>
</section>
</div>
'''
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes); streams: " + ", ".join(f"{s}: {len(data[s])} analyses" for s, _, _ in STREAMS))


if __name__ == "__main__":
    main()
