"""Steering page: the two retained methods for making the central model act on the memory's reliability record.

Method 1  fusion at the decision  (scripts/fusion_decision.py): the model answers alone; the record combines its answer
          with the peers' answers by the Nitzan-Paroush rule, the model's vote weighted by its self-record.
Method 2  fusion in the attention (scripts/steer_attention.py): the peers are in the prompt without any reliability
          text; every head's attention over peer i's tokens is tilted by gamma * log(p_i / max p).
Reads the probe analyses under outputs/gen/q3_4b/steer/{ood,indist}/ and writes artifacts/helman_mem_steering.html.
The retired attempts (prompt wording and structure, second pass, final-answer reranking) are kept only as a summary
table of their numbers; their code was removed.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEER = ROOT / "outputs/gen/q3_4b/steer"
PROBE = ROOT / "outputs/gen/q3_4b/probe"
OUT = ROOT / "artifacts/helman_mem_steering.html"

STREAMS = [("ood", "OOD slice", "1,500 events, positions 4,000–5,499 of the OOD stream (yes/no, multiple choice, BIG-Bench Hard)"),
           ("indist", "in-distribution slice", "1,200 events, positions 1,500–2,699 of the in-distribution stream (math and reading; code excluded from the answer analysis)")]
RETIRED = [("ordinal", "ranked record in words", "prompt text"), ("self_note", "ranked record + own record", "prompt text"), ("verify", "verify-the-favourite procedure", "prompt text"),
           ("posterior", "fused posterior + own reliability + decision rule", "prompt text"), ("sorted", "peers sorted by reliability", "prompt structure"), ("vote", "reliability-weighted vote shown", "prompt structure"),
           ("filtered", "unreliable peers withheld", "prompt structure"), ("favourite", "only the favourite shown", "prompt structure"), ("defer", "two-pass: alone, then confront", "second pass")]


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def first(d: dict) -> dict:
    return next(iter(d.values())) if d else {}


def analyses(stream: str) -> dict:
    d = STEER / stream
    out = {}
    refs = load(d / "analysis_refs.json")
    for k in ("solo", "number", "peers"):
        if k in refs:
            out[k] = refs[k]
    fu = load(d / "analysis_fusion.json")
    if fu:
        out["solo_think"] = fu.get("solo_think", {})
        out["fusion_sim"] = fu.get("solo", {}).get("fusion", {})
        out["fusion_sim_think"] = fu.get("solo_think", {}).get("fusion", {})
    pt = load(PROBE / f"analysis_{stream}_think.json")
    for k, name in (("frozen_think_notes", "number_think"), ("frozen_think_nonotes", "peers_think"), ("frozen_think_swapped", "number_swapped_think")):
        if k in pt:
            out[name] = pt[k]
    for key in ("fusion_self", "fusion_self_swapped", "fusion_self_think", "fusion_self_swapped_think", "attn_g1", "attn_g1_swapped", "attn_g3", "attn_g3_swapped"):
        a = load(d / f"analysis_{key}.json")
        if a:
            out[key] = first(a)
    for key, *_ in RETIRED:
        for sw in ("", "_swapped", "_think", "_swapped_think"):
            a = load(d / f"analysis_{key}{sw}.json")
            if a:
                out[key + sw] = first(a)
    return out


def f1(x, d=1, suffix=""):
    return "—" if x is None or x != x else f"{x:.{d}f}{suffix}"


def delta(x, ref):
    return "—" if x is None or ref is None else f"{x - ref:+.1f}"


# ----------------------------------------------------------------------------------------------- SVG
def svg_bars(rows: list[tuple[str, float | None, float | None]], alone: float | None, title: str) -> str:
    W, rh = 1100, 30
    H = 40 + rh * len(rows) + 30
    vals = [v for _, a, b in rows for v in (a, b) if v is not None]
    lo, hi = max(0.0, min(vals) - 6), min(100.0, max(vals) + 4)
    pl, pr = 330, 30
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
            s.append(f'<rect x="{X(lo):.1f}" y="{y + 13}" width="{max(0, X(a) - X(lo)):.1f}" height="9" class="bar true"/><text x="{X(a) + 4:.1f}" y="{y + 21}" class="tick">{a:.1f}</text>')
    if alone is not None and lo < alone < hi:
        s.append(f'<line x1="{X(alone):.1f}" y1="30" x2="{X(alone):.1f}" y2="{H - 26}" class="aline"/><text x="{X(alone) + 4:.1f}" y="38" class="tick alone">alone {alone:.1f}</text>')
    s.append("</svg>")
    return "".join(s)


def svg_methods() -> str:
    """The two retained methods as one diagram: the record acts after the model (1) or inside its attention (2)."""
    W, H = 1180, 330
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="The two methods" style="width:100%;height:auto">',
         '<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dhead"/></marker></defs>']
    def box(x, y, w, h, t, sub, cls="dbox"):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/><text x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle" class="dtitle">{html.escape(t)}</text>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle" class="dsub">{html.escape(sub)}</text>')
    def arrow(x1, y1, x2, y2, cls="darrow"):
        return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#ah3)"/>'
    s.append('<text x="20" y="22" class="dcol">Method 1 · fusion at the decision</text>')
    s.append(box(20, 40, 170, 60, "question", "no peers, no notes"))
    s.append(box(240, 40, 190, 60, "central model", "answers alone (thinking on)"))
    s.append(box(480, 40, 200, 60, "Kalman memory", "peers' records + self-record"))
    s.append(box(730, 40, 220, 60, "log-odds vote", "Σ logit(p) + log(K−1), gated", "dbox strong"))
    s.append(box(1000, 40, 160, 60, "answer", "graded"))
    s.append(arrow(190, 70, 240, 70)); s.append(arrow(430, 70, 730, 60)); s.append(arrow(680, 70, 730, 75)); s.append(arrow(950, 70, 1000, 70))
    s.append('<text x="560" y="130" text-anchor="middle" class="dsub">the peers\' answers enter the vote directly; the model\'s answer is one voter with weight logit(q)</text>')
    s.append('<text x="20" y="182" class="dcol">Method 2 · fusion in the attention</text>')
    s.append(box(20, 200, 170, 60, "question + 3 peers", "no reliability text"))
    s.append(box(240, 200, 190, 60, "central model", "attention over peer tokens"))
    s.append(box(480, 200, 200, 60, "Kalman memory", "c_i = p_i / max p"))
    s.append(box(730, 200, 220, 60, "attention tilt", "softmax(scores + γ log c_i)", "dbox strong"))
    s.append(box(1000, 200, 160, 60, "answer", "graded"))
    s.append(arrow(190, 230, 240, 230)); s.append(arrow(680, 230, 730, 230)); s.append(arrow(840, 200, 335, 200, "darrow bad")); s.append(arrow(430, 230, 1000, 230))
    s.append('<text x="560" y="290" text-anchor="middle" class="dsub">the record never appears in the text; it re-weights what the model attends to, on every head of the chosen layers</text>')
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
td.best{color:var(--accent);font-weight:700} .sub{color:var(--muted);font-size:.86rem;max-width:100ch}
code{font-family:"JetBrains Mono",monospace;font-size:.86em;background:var(--code);padding:.05em .3em;border-radius:4px}
.callout{border-left:4px solid var(--accent);background:var(--teal-bg);padding:.7rem .95rem;border-radius:0 8px 8px 0;margin:.9rem 0;max-width:100ch} .callout.warn{border-left-color:var(--amber);background:var(--amber-bg)}
.diagram{background:var(--code);border:1px solid var(--rule);border-radius:10px;padding:.6rem;margin:.5rem 0 .9rem}
.dbox{fill:var(--paper);stroke:var(--rule);stroke-width:1.2} .dbox.strong{stroke:var(--accent);stroke-width:2} .dtitle{fill:var(--ink);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:600} .dsub{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px}
.dcol{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.darrow{stroke:var(--muted);stroke-width:1.6;fill:none} .darrow.bad{stroke:var(--accent);stroke-width:2} .dhead{fill:var(--muted)}
.grid{stroke:var(--rule);stroke-width:1} .tick{fill:var(--muted);font-family:"JetBrains Mono",monospace;font-size:11px} .tick.alone{fill:var(--amber)} .ctitle{fill:var(--ink);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:700}
.blabel{fill:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:12.5px} .bar.true{fill:var(--accent)} .bar.swapped{fill:none;stroke:var(--bad);stroke-width:1.2;stroke-dasharray:3 2} .aline{stroke:var(--amber);stroke-width:1.5;stroke-dasharray:5 3}
figure.chart{margin:0 0 1rem;background:var(--code);border:1px solid var(--rule);border-radius:10px;padding:.5rem .6rem .4rem}
ol.findings{max-width:105ch;padding-left:1.3rem} ol.findings li{margin:.55rem 0}
.math{font-family:"JetBrains Mono",monospace;font-size:.95rem;background:var(--code);border-radius:8px;padding:.5rem .8rem;display:inline-block}
.survey td:first-child{min-width:220px}
"""


def main() -> None:
    data = {s: analyses(s) for s, _, _ in STREAMS}
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()

    def row(cells, head=False, classes=None):
        tag = "th" if head else "td"
        classes = classes or [""] * len(cells)
        return "<tr>" + "".join(f"<{tag}{(' class=' + chr(34) + c + chr(34)) if c else ''}>{v}</{tag}>" for v, c in zip(cells, classes)) + "</tr>"

    # ---------------- results per stream
    sections, charts = [], []
    for s, label, desc in STREAMS:
        A = data[s]
        alone, alone_t = A.get("solo", {}).get("accuracy"), A.get("solo_think", {}).get("accuracy")
        number, number_t = A.get("number", {}).get("accuracy"), A.get("number_think", {}).get("accuracy")
        rows_html = [row(["system", "record enters", "accuracy", "vs alone", "vs current prompt", "selection events: accuracy", "follows the favourite there", "accuracy when the favourite is wrong", "swapped record: accuracy", "steered? (drop under swap)"], True)]
        bars = []
        def add(lab, st, sw, point, ref_alone, ref_number):
            if not st:
                return
            acc = st["accuracy"]; swacc = sw["accuracy"] if sw else None
            cls = ["", "", "best" if (ref_alone is not None and acc >= ref_alone + 1.0) else "", "", "", "", "", "", "", ""]
            rows_html.append(row([lab, point, f1(acc, 2), delta(acc, ref_alone), delta(acc, ref_number), f1(st.get("anycorrect_model_right")), f1(st.get("anycorrect_model_follows_favourite"), 1, "%"), f1(st.get("accuracy_when_favourite_wrong")), f1(swacc, 2), (f"{acc - swacc:+.1f}" if swacc is not None else "—")], classes=cls))
            bars.append((lab, acc, swacc))
        add("model alone (no peers, no notes)", A.get("solo"), None, "—", alone, number)
        add("peers in the prompt, no notes", A.get("peers"), None, "—", alone, number)
        add("current prompt: number in the header", A.get("number"), None, "prompt text", alone, number)
        add("Method 1: alone + fusion at the decision (self-record weight)", A.get("fusion_self"), A.get("fusion_self_swapped"), "after the model", alone, number)
        add("Method 2: attention tilt, γ = 1", A.get("attn_g1"), A.get("attn_g1_swapped"), "attention weights", alone, number)
        add("Method 2: attention tilt, γ = 3", A.get("attn_g3"), A.get("attn_g3_swapped"), "attention weights", alone, number)
        rows_html.append(row(["<i>thinking on (4,096-token budget)</i>", "", "", "vs alone + thinking", "vs current + thinking", "", "", "", "", ""], True))
        add("model alone + thinking", A.get("solo_think"), None, "—", alone_t, number_t)
        add("peers in the prompt, no notes + thinking", A.get("peers_think"), None, "—", alone_t, number_t)
        add("current prompt + thinking", A.get("number_think"), A.get("number_swapped_think"), "prompt text", alone_t, number_t)
        add("Method 1: alone + thinking + fusion (self-record weight)", A.get("fusion_self_think"), A.get("fusion_self_swapped_think"), "after the model", alone_t, number_t)
        sections.append(f"<h3>{label}</h3><p class=\"sub\">{desc}. Frozen Qwen3-4B, greedy. Selection events: the peers' answers differ and one of them is right (n={A.get('solo', {}).get('n_informative_anycorrect', '—')}); the favourite (the peer the record rates most reliable) is right {f1(A.get('solo', {}).get('anycorrect_favourite_right'))}% of them. Swapped record: the estimates permuted by rank, so the highest reliability is assigned to the least trusted peer; a method the record reaches loses accuracy under the swap.</p><div class=\"tbl\"><table>{''.join(rows_html)}</table></div>")
        charts.append(f'<figure class="chart">{svg_bars(bars, alone, f"{label}: accuracy with the true record (solid) and the swapped record (dashed); amber line = the model alone")}</figure>')

    # ---------------- fusion rule variants (simulation over the alone generations)
    frows = [row(["weights", "model's vote", "OOD slice", "vs alone", "OOD, thinking", "vs alone", "in-distribution", "vs alone", "in-distribution, thinking", "vs alone"], True)]
    def sim(stream, base, rule, w): return data[stream].get(base, {}).get(rule, {}).get(w)
    ao, ai = data["ood"].get("solo", {}).get("accuracy"), data["indist"].get("solo", {}).get("accuracy")
    ato, ati = data["ood"].get("solo_think", {}).get("accuracy"), data["indist"].get("solo_think", {}).get("accuracy")
    frows.append(row(["—", "no fusion (alone)", f1(ao, 2), "—", f1(ato, 2), "—", f1(ai, 2), "—", f1(ati, 2), "—"]))
    for rule, rl in (("logit", "logit(p)"), ("logit+K", "logit(p) + log(K−1)"), ("beta", "ψ(α) − ψ(β)"), ("beta+K", "ψ(α) − ψ(β) + log(K−1)")):
        for w, wl in (("w=self", "self-record"), ("w=0.0", "0 (peers decide)"), ("w=1.0", "1.0")):
            if rule != "logit" and w != "w=self":
                continue
            v = [sim("ood", "fusion_sim", rule, w), sim("ood", "fusion_sim_think", rule, w), sim("indist", "fusion_sim", rule, w), sim("indist", "fusion_sim_think", rule, w)]
            if all(x is None for x in v):
                continue
            frows.append(row([rl, wl, f1(v[0], 2), delta(v[0], ao), f1(v[1], 2), delta(v[1], ato), f1(v[2], 2), delta(v[2], ai), f1(v[3], 2), delta(v[3], ati)]))
    fusion_table = "".join(frows)

    # ---------------- history sufficiency
    hrows = [row(["stream, positions", "similar past cases behind a note (mean / median / min)", "events with no evidence", "events under 10 cases", "flat records", "model alone", "alone + fusion (self-record)", "gain"], True)]
    for s, label, _ in STREAMS:
        for e in load(STEER / s / "evidence_stats.json"):
            fa = load(STEER / s / (f"analysis_fusion_pos{e['start']}.json" if e["start"] not in (4000, 1500) else "analysis_fusion.json"))
            solo = fa.get("solo", {}); alone_ = solo.get("accuracy"); fused = solo.get("fusion", {}).get("logit", {}).get("w=self")
            hrows.append(row([f"{label.replace(' slice', '')} {e['start']:,}–{e['start'] + e['count'] - 1:,}" + (" (probe slice)" if e["start"] in (4000, 1500) else ""), f"{e['mean_evidence']:.0f} / {e['median_evidence']:.0f} / {e['min_evidence']:.0f}", f"{e['no_evidence']} ({100 * e['no_evidence'] / e['count']:.1f}%)", f"{e['under_10']} ({100 * e['under_10'] / e['count']:.1f}%)", f"{e['flat']} ({100 * e['flat'] / e['count']:.1f}%)", f1(alone_, 2), f1(fused, 2), delta(fused, alone_)]))
    history_table = "".join(hrows)

    # ---------------- retired attempts (numbers only)
    rrows = [row(["retired attempt", "record enters", "OOD: accuracy / swapped", "in-distribution: accuracy / swapped", "OOD + thinking: accuracy / swapped"], True)]
    o, i_ = data["ood"], data["indist"]
    for key, lab, point in RETIRED:
        def pair(A, k):
            a, b = A.get(k, {}).get("accuracy"), A.get(k + "_swapped", {}).get("accuracy")
            return f"{f1(a, 1)} / {f1(b, 1)}" if a is not None else "—"
        ot = o.get(key + "_think", {}).get("accuracy"); ots = o.get(key + "_swapped_think", {}).get("accuracy")
        rrows.append(row([lab, point, pair(o, key), pair(i_, key), (f"{f1(ot, 1)} / {f1(ots, 1)}" if ot is not None else "—")]))
    rrows.append(row(["rerank the candidates at the final answer with the record (gated)", "final-answer decision", "64.4 (current prompt), 65.7 (peers prompt), 71.8 (alone reasoning)", "71.6 (alone reasoning)", "—"]))
    retired_table = "".join(rrows)

    survey = [
        ("Credibility-aware generation (CAG), Pan et al. 2024, arXiv:2404.06809", "credibility levels written into the prompt per document; trained models use them", "untrained models are \"not inherently sensitive to directly provided credibility in the prompt\" (+0.7 EM from prompting, +9 from fine-tuning). Reproduced here: every wording of the record left the model where the number left it."),
        ("CrAM, Deng et al. 2024, arXiv:2406.11497", "credibility scales the attention weights of the document's tokens on causally selected heads, Norm(A ⊙ s); training-free", "+25 to +32 EM over putting the scores in the prompt. Method 2 is this rule on all heads; head selection by causal tracing is the refinement."),
        ("Roundtable Policy, 2025, arXiv:2509.16839; ReConcile; credibility-scored aggregation, arXiv:2505.24239", "confidence- or reliability-weighted consensus computed outside the model", "weights act at aggregation, where they cannot be ignored. Method 1 is this design with the Kalman record as the weight and the Nitzan-Paroush rule as the aggregator."),
        ("Nitzan and Paroush 1982; Shapley and Grofman 1984", "optimal weighted majority for independent voters of known competence", "the optimal weights are the log-odds of the competences; the K-alternatives term follows from the symmetric-error model."),
        ("Most LLM conformity needs no speaker, 2026, arXiv:2607.05545; sycophancy propagation, arXiv:2604.02668", "what drives conformity to peers; categorical peer rankings in the prompt", "position and majority size drive it, reliability instructions have limited effect; rankings moved influence for some models (+10.5) and not others. Matches the retired structural variants: they steer, but cannot lift the model above itself on OOD."),
    ]
    survey_rows = [row(["work", "how the reliability signal enters", "finding, and what it implies here"], True)] + [row([html.escape(a), html.escape(b), html.escape(c)]) for a, b, c in survey]

    # ---------------- findings
    def acc(st, k): return data[st].get(k, {}).get("accuracy")
    findings = []
    f_o, f_i = o.get("fusion_self"), i_.get("fusion_self"); ft_o, ft_i = o.get("fusion_self_think"), i_.get("fusion_self_think")
    if f_o and f_i:
        findings.append(f"<li><b>Method 1 is the only method that never falls below the model alone, and it converts the record's quality into accuracy wherever the peers know something the model does not.</b> In-distribution the fused system scores {f1(f_i['accuracy'], 2)} against {f1(ai, 2)} for the model alone and {f1(acc('indist', 'number'), 2)} for the current prompt: under the exact-match grader the model reading the peers already matches the record's favourite on the disagreement events ({f1(i_.get('number', {}).get('anycorrect_model_right'))}% against {f1(i_.get('solo', {}).get('anycorrect_favourite_right'))}%), so in-domain the two routes to the peers' knowledge tie, and the fusion's edge appears with thinking on ({f1(ft_i['accuracy'] if ft_i else None, 2)} against {f1(ati, 2)} alone and {f1(acc('indist', 'number_think'), 2)} for the current prompt), where it reaches {f1(ft_i.get('anycorrect_model_right') if ft_i else None)}% on the selection events. On OOD, where the peers are weaker than the model, it adds {delta(f_o['accuracy'], ao)} without thinking and {delta(ft_o['accuracy'] if ft_o else None, ato)} with, while every prompt with peers in it loses 4 to 13 points: the self-record keeps the model's vote decisive where the model is the more reliable source. The swapped record costs {f1(f_i['accuracy'] - i_.get('fusion_self_swapped', {}).get('accuracy', float('nan')), 1)} points in-distribution and {f1(f_o['accuracy'] - o.get('fusion_self_swapped', {}).get('accuracy', float('nan')), 1)} on OOD: the record is what decides, and a wrong record is expensive, which is the price of a method that actually uses it.</li>")
    a3, a3s, ia3 = o.get("attn_g3"), o.get("attn_g3_swapped"), i_.get("attn_g3")
    if a3:
        findings.append(f"<li><b>Method 2 is the strongest way of steering the model itself, and it is the same rule applied to attention.</b> Attention over the prompt is a distribution over source tokens; softmax(scores + γ·log c<sub>i</sub>) tilts it by a prior over sources (CrAM's rule) with no reliability text anywhere. With γ = 3 the model follows the favourite on {f1(a3.get('anycorrect_model_follows_favourite'))}% of the OOD selection events, the favourite's own rate, and scores {f1(a3['accuracy'], 2)} OOD (swapped {f1(a3s['accuracy'] if a3s else None, 2)}) and {f1(ia3['accuracy'] if ia3 else None, 2)} in-distribution (swapped {f1(i_.get('attn_g3_swapped', {}).get('accuracy'), 2)}). It beats every prompt-level variant on OOD and still sits {delta(a3['accuracy'], ao)} from the model alone there, because the peers are in the prompt and cost more on OOD than the record can repay. The bias is on all heads and one strength; CrAM's head selection and a γ set from the record's own uncertainty are the two refinements.</li>")
    findings.append("<li><b>Why the other attempts were retired.</b> Every wording of the record in the prompt, including the fused posterior with the decision rule spelled out, left accuracy where the number left it: the model reads the number and decides from the content. Re-ordering or withholding peers steers strongly (the swapped record costs up to 19 points) but adds at most two points over the current prompt and cannot lift the model above itself on OOD. The second pass keeps the model's answer whether the shown peer is reliable or not. Reranking at the final answer gains under a point, because the model's log-probability of its own answer, conditioned on its own reasoning, outweighs the record by thousands to one. The numbers are in the table below; the code is gone.</li>")
    findings.append("<li><b>Where nothing can add:</b> OOD with thinking on, where the thinking model is right on 82% of the disagreement events against the record's 78%. Method 1 stays within 0.1 of the model there, which is the correct behaviour of the rule; the limit is the peer pool, not the steering.</li>")

    page = f'''<title>Steering the Central Model</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style>
<div class="page">
<header>
<div class="eyebrow">helman-mem · steering · training-free · frozen Qwen3-4B · updated {stamp}</div>
<h1>Steering the Central Model</h1>
<p class="thesis">The reliability record is good (AUC 0.92; where peers disagree and one is right, its favourite is the right one 78% of the time on OOD and 89% in-distribution under the evaluator's grader) and the central model ignores it when it is a number in the prompt. Two training-free methods make the record act, both instances of one rule, evidence fusion in the log-odds domain: Method 1 applies it after the model, to the answers; Method 2 applies it inside the model, to its attention. Each is tested on the same probe slices as the results page with a swapped-record control.</p>
</header>

<section>
<h2>The two methods</h2>
<div class="diagram">{svg_methods()}</div>
<p><b>Method 1, fusion at the decision</b> (<code>scripts/fusion_decision.py</code>). The model answers alone, with nothing about peers in its prompt. Where the peers' answers differ and the record is not flat, the memory scores each distinct answer by the summed log-odds of its supporters and emits the best one; the model's own answer is one supporter, weighted by the log-odds of its self-record, its running accuracy on the task along the stream, tracked read-before-write like the peers' records. Elsewhere the model's answer stands.</p>
<p class="math">score(a) = Σ<sub>i: v<sub>i</sub> = a</sub> [ log p<sub>i</sub> / (1 − p<sub>i</sub>) + log (K − 1) ] ,&nbsp;&nbsp; the model's vote: log q / (1 − q)</p>
<p>For voters that err independently with competences p<sub>i</sub> this is the Bayes-optimal choice (Nitzan and Paroush, 1982); log (K − 1) is the symmetric-error model with K alternatives, under which agreement between weak peers on an open answer is strong evidence. Under a Beta posterior the expected weight is ψ(α) − ψ(β), which shrinks with little evidence. The memory is an information-form filter, so the whole chain (state update, weight, decision) is additive in log-odds, and each step can be undone.</p>
<p><b>Method 2, fusion in the attention</b> (<code>scripts/steer_attention.py</code>). The peers are in the prompt with no reliability text at all. In every attention head of the chosen layers the scores over peer i's tokens receive γ · log(p<sub>i</sub> / max<sub>j</sub> p<sub>j</sub>); softmax(scores + γ log c) equals Norm(A ⊙ c<sup>γ</sup>), the credibility-aware attention of CrAM. Attention is a distribution over what the model reads, and the tilt is a prior over sources with strength γ. Flat records get no tilt.</p>
</section>

<section>
<h2>Results</h2>
{''.join(sections)}
{''.join(charts)}
<p class="sub">Selection events: the peers' answers differ and at least one is right. "Follows the favourite": the final answer falls in the favourite's answer group. Steered: accuracy drop when the record is swapped by rank. Analysis script <code>scripts/memory_use_probe.py</code>; outputs under <code>outputs/gen/q3_4b/steer/</code>.</p>
<h3>Method 1: the rule's variants</h3>
<p>Simulated over the model's alone answers on the slices (the same computation as the script, over the fusion options): the weights, the K-alternatives term, the Beta-posterior weights, and the model's vote weight.</p>
<div class="tbl"><table>{fusion_table}</table></div>
<p class="sub">"self-record": logit of the model's running accuracy on the task, no tuning. The K term and the Beta weights change the result by less than a point here because the evidence counts are large; they matter when the record is young.</p>
</section>

<section>
<h2>Does the history accumulate enough?</h2>
<p>The probe slices sit late in the streams (12,000 verified outcomes absorbed by OOD position 4,000; 4,500 by in-distribution position 1,500). The gain of Method 1 does not depend on position: it is present in the first slice, where a quarter of the in-distribution events still rest on fewer than 10 similar cases, and the share of flat records on OOD does not shrink with more history, because it comes from peers that are equally good at yes/no and multiple choice.</p>
<div class="tbl"><table>{history_table}</table></div>
</section>

<section>
<h2>What the literature says</h2>
<div class="tbl survey"><table>{''.join(survey_rows)}</table></div>
</section>

<section>
<h2>Findings</h2>
<ol class="findings">{''.join(findings)}</ol>
<div class="callout"><b>Recommended design.</b> Keep the memory outside the model and make the whole chain additive in log-odds. The central model answers alone (thinking on); the same Kalman filter that tracks the peers tracks the central model's own reliability at the question's address; where the peers disagree and the record is not flat, the memory scores each distinct answer by the summed log-odds of its supporters, the model's answer included with its self-record weight, and emits the best-scored answer (Method 1). When the model must read the peers, tilt its attention by the record instead of describing it (Method 2), and aggregate afterwards.</div>
<div class="callout warn"><b>Caveats.</b> One frozen model, one seed, greedy decoding, slices of 1,500 and 1,200 events; thinking-mode rows for Method 2 not run; the attention bias uses all heads and two strengths. <b>Label rule.</b> The stored peer labels for reading comprehension are more lenient than the evaluator's exact-match grader: 15.8% of the peers' reading answers are labelled right and graded wrong (never the reverse; the two rules agree on every other task). The record learned reading reliability under the lenient rule, so its favourite is "right" less often under the grader than the record believes. All numbers on this page are graded with the evaluator's rule, peers included; an earlier simulation that trusted the stored labels put Method 1 in-distribution at 80.5 and 84.7 with thinking, and those figures were withdrawn.</div>
</section>

<section>
<h2>Retired attempts</h2>
<p>Tried on the same slices with the same controls and removed from the code; the numbers stay as the reason the two methods above were kept. Accuracy with the true record / with the swapped record.</p>
<div class="tbl"><table>{retired_table}</table></div>
</section>
</div>
'''
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes); analyses: " + ", ".join(f"{s}={len(data[s])}" for s, _, _ in STREAMS))


if __name__ == "__main__":
    main()
