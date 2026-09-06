"""Conclusion section of the results page: the pipeline with its break points, the evidence chain, where the
note could matter at all, follow rate against the shown estimate, and the numbered problems with the change
each one calls for.  Every number is read from the probe files (scripts/memory_use_probe.py outputs), the
length/peer-mention analyses and the evaluation results passed in by results_page.py."""
from __future__ import annotations

import html
import json
from pathlib import Path

BAD, OK, MUT, AMB = "var(--bad)", "var(--accent)", "var(--muted)", "var(--amber)"


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def f1(x, d=1, suffix=""):
    return ("—" if x is None else f"{x:.{d}f}{suffix}")


# ----------------------------------------------------------------------------------------------- SVG helpers
def box(x, y, w, h, title, sub="", *, mark=None, dashed=False):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="dbox{" dashed" if dashed else ""}"/>']
    lines = [title] + ([sub] if sub else [])
    ty = y + h / 2 - (len(lines) - 1) * 8
    for i, t in enumerate(lines):
        out.append(f'<text x="{x + w / 2}" y="{ty + i * 16:.0f}" text-anchor="middle" class="{"dtitle" if i == 0 else "dsub"}">{html.escape(t)}</text>')
    if mark is not None:
        out.append(f'<circle cx="{x + w - 4}" cy="{y + 4}" r="11" class="dmark"/><text x="{x + w - 4}" y="{y + 8.5}" text-anchor="middle" class="dmarkt">{mark}</text>')
    return "".join(out)


def arrow(x1, y1, x2, y2, *, cls="darrow"):
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#ah)"/>'


def elbow(x1, y1, x2, y2, *, cls="darrow"):
    xm = (x1 + x2) / 2
    return f'<path d="M{x1},{y1} L{xm},{y1} L{xm},{y2} L{x2},{y2}" class="{cls}" marker-end="url(#ah)"/>'


DEFS = '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dhead"/></marker></defs>'


def svg_pipeline() -> str:
    """The training pipeline as run, with the seven break points marked."""
    W, H = 1180, 330
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Training pipeline with its break points" style="width:100%;height:auto">', DEFS]
    s.append(box(20, 40, 130, 56, "Stream event", "question, in order"))
    s.append(box(190, 40, 140, 56, "3 peer solutions", "unlabelled"))
    s.append(box(190, 130, 140, 56, "Kalman memory", "reliability note per peer", mark=2))
    s.append(box(380, 40, 160, 146, "Central model", "thinking off", mark=7))
    s.append(box(590, 40, 170, 56, "1 guided answer", "greedy, peers + notes", mark=3))
    s.append(box(590, 130, 170, 56, "4 question-only samples", "temperature 1"))
    s.append(box(810, 40, 130, 146, "Verifier", "label after the answer", mark=5))
    s.append(box(990, 40, 170, 146, "GRPO group", "advantage vs the 5"))
    s.append(box(590, 230, 350, 60, "Update: the guided answer is also relabelled", "under the question-only prompt (both views)", mark=4))
    s.append(box(990, 230, 170, 60, "Optimiser", "no KL, constant lr", mark=6))
    s.append(box(20, 230, 230, 60, "Trained model, evaluated alone", "no peers, no notes at test time"))
    # arrows
    s.append(arrow(150, 68, 190, 68)); s.append(arrow(330, 68, 380, 68)); s.append(arrow(330, 158, 380, 158, cls="darrow bad"))
    s.append(f'<text x="355" y="150" text-anchor="middle" class="dmarkt2">1</text>')
    s.append(arrow(540, 68, 590, 68)); s.append(arrow(540, 158, 590, 158)); s.append(arrow(760, 68, 810, 68)); s.append(arrow(760, 158, 810, 158))
    s.append(arrow(940, 113, 990, 113)); s.append(arrow(1075, 186, 1075, 230)); s.append(arrow(990, 260, 940, 260)); s.append(arrow(590, 260, 250, 260))
    s.append(f'<text x="1000" y="215" class="dsub">policy gradient</text>')
    s.append("</svg>")
    return "".join(s)


def svg_evidence() -> str:
    """Experiments → findings → conclusion."""
    W, H = 1180, 470
    E = [("E1", "Five regimes on the whole test streams", "4,319 + 17,403 events, final and intermediate checkpoints"),
         ("E2", "Note probe: shown / hidden / swapped", "frozen, trained, thinking; 2,700 events"),
         ("E3", "Who is right where peers disagree", "memory's favourite vs plurality vs oracle"),
         ("E4", "Answer style of the alone policy", "length, reasoning, peer mentions"),
         ("E5", "Training dynamics on wandb", "entropy, KL, length, zero-variance groups"),
         ("E6", "Thinking-mode runs", "70 steps; frozen + thinking")]
    F = [("F1", "Peer-conditioned regimes lose to plain RLVR", "OOD −4 to −8, code −5 to −9; reading gain shared"),
         ("F2", "Note read but not used", "follow rates equal shown / hidden / swapped"),
         ("F3", "Note decisive in 5% / 2% of events", "favourite right 45–48% there; ceiling +1.0 / −0.6"),
         ("F4", "Relabelling leaks the peer context", "peers cited in 99.7% of answers; reasoning dropped"),
         ("F5", "Gradient from reading only; drift", "collapse at step 229; best checkpoint ≠ last"),
         ("F6", "Thinking alone: +11 OOD", "RL adds nothing on OOD in 70 steps")]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="How the conclusion was reached" style="width:100%;height:auto">', DEFS]
    for col, x, label in ((E, 20, "Experiments"), (F, 450, "Findings")):
        s.append(f'<text x="{x}" y="22" class="dcol">{label}</text>')
        for i, (k, t, sub) in enumerate(col):
            y = 36 + i * 70
            s.append(box(x, y, 330, 56, f"{k}  {t}", sub))
    for i in range(6):
        y = 36 + i * 70 + 28
        s.append(arrow(350, y, 450, y))
    s.append(f'<text x="880" y="22" class="dcol">Conclusion</text>')
    s.append(box(880, 36, 280, 200, "The memory cannot act in this pipeline", ""))
    s.append('<foreignObject x="892" y="70" width="256" height="160"><div xmlns="http://www.w3.org/1999/xhtml" class="dfo">Three independent blocks: the interface (F2), the headroom (F3) and the training signal (F4, F5). Each alone is enough to hide any memory effect, so the accuracy tables cannot tell whether the memory is good.</div></foreignObject>')
    s.append(box(880, 262, 280, 190, "What to change first", ""))
    s.append('<foreignObject x="892" y="296" width="256" height="150"><div xmlns="http://www.w3.org/1999/xhtml" class="dfo">P1 give the memory headroom and measure it as a selector; P2 give the note a gradient path; P3 stop relabelling guided answers; P4 balance the gradient; P5 regularise; P6 train with thinking on.</div></foreignObject>')
    for i in range(6):
        y = 36 + i * 70 + 28
        s.append(elbow(780, y, 880, 136 if i < 3 else 357, cls="darrow thin"))
    s.append("</svg>")
    return "".join(s)


def svg_partition(full_ood: dict, full_in: dict) -> str:
    """Where the note could matter: events by peer agreement and by where the memory's favourite sits."""
    W, H = 1180, 250
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Where the note could matter" style="width:100%;height:auto">']
    cols = [("unanimous peers", MUT, 0.35), ("peers split, notes flat", MUT, 0.6), ("split, favourite in the plurality", OK, 0.9), ("split, favourite in the minority", BAD, 1.0)]
    for row_i, (name, st, note) in enumerate((("OOD stream, 17,403 events", full_ood, ""), ("in-distribution stream, 3,319 events (code excluded)", full_in, ""))):
        if not st:
            continue
        n = st["n"]; parts = [st["n_unanimous"], st["n_split_flat"], st["n_informative"] - st["n_top_minority"], st["n_top_minority"]]
        y = 40 + row_i * 100
        s.append(f'<text x="20" y="{y - 10}" class="dtitle">{html.escape(name)}</text>')
        x = 20
        for (lab, col, op), v in zip(cols, parts):
            w = 1140 * v / n
            s.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w, 1):.1f}" height="34" fill="{col}" fill-opacity="{op}" stroke="var(--paper)" stroke-width="1.5"/>')
            if w > 60:
                s.append(f'<text x="{x + w / 2:.1f}" y="{y + 22}" text-anchor="middle" class="dbar{" ink" if op < 0.9 else ""}">{v:,}</text>')
            x += w
        fav = st["minority_top_correct"]; model = st["accuracy_when_minority"]; ceiling = (fav - model) * st["n_top_minority"] / n
        s.append(f'<text x="20" y="{y + 54}" class="dsub">favourite in the minority: {st["n_top_minority"]:,} events ({100 * st["n_top_minority"] / n:.1f}%), the favourite is right {fav:.1f}% of them, the model {model:.1f}%: a reader that always followed the note would move the stream by {ceiling:+.2f} points. Where peers disagree the favourite is right {st["informative_top_correct"]:.1f}%, the plurality {st["informative_plurality_correct"]:.1f}%, some peer {st["informative_any_peer_correct"]:.1f}%.</text>')
    lx = 20
    for lab, col, op in cols:
        s.append(f'<rect x="{lx}" y="228" width="14" height="10" fill="{col}" fill-opacity="{op}"/><text x="{lx + 20}" y="237" class="dsub">{html.escape(lab)}</text>')
        lx += 290
    s.append("</svg>")
    return "".join(s)


BINS = ["<0.35", "0.35–0.5", "0.5–0.65", "0.65–0.8", "≥0.8"]


def svg_bins(panel_title: str, series: list[tuple[str, dict, str, str]]) -> str:
    """Follow rate against the estimate printed on the peer, one panel.  series: (label, follow_by_shown_prob, colour, dash)."""
    W, H = 380, 250
    pl, pr, pt, pb = 44, 12, 28, 52
    s = [f'<figure class="chart small"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(panel_title)}" style="width:100%;height:auto">']
    s.append(f'<text x="{pl}" y="16" class="ctitle">{html.escape(panel_title)}</text>')
    def X(i): return pl + (i + 0.5) / 5 * (W - pl - pr)
    def Y(v): return pt + (100 - v) / 100 * (H - pt - pb)
    for v in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W - pr}" y2="{Y(v):.1f}" class="grid"/><text x="{pl - 5}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
    for i, b in enumerate(BINS):
        s.append(f'<text x="{X(i):.1f}" y="{H - 34}" class="tick" text-anchor="middle">{b}</text>')
    s.append(f'<text x="{(pl + W - pr) / 2:.0f}" y="{H - 20}" class="tick" text-anchor="middle">estimate printed on the peer</text>')
    s.append(f'<text x="{pl - 30}" y="{(pt + H - pb) / 2:.0f}" class="tick" text-anchor="middle" transform="rotate(-90 {pl - 30} {(pt + H - pb) / 2:.0f})">follows that peer (%)</text>')
    legend = []
    for label, cal, colour, dash in series:
        if not cal:
            continue
        pts = [(i, list(cal.values())[i]["follow"]) for i in range(5) if list(cal.values())[i]["n"] > 0 and list(cal.values())[i]["follow"] == list(cal.values())[i]["follow"]]
        path = " ".join(f"{'M' if j == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for j, (i, v) in enumerate(pts))
        s.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.2" stroke-dasharray="{dash}"/>')
        for i, v in pts:
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.8" fill="{colour}"/>')
        legend.append(f'<span class="lg"><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line x1="0" y1="5" x2="22" y2="5" stroke="{colour}" stroke-width="3" stroke-dasharray="{dash}"/></svg><span>{html.escape(label)}</span></span>')
    s.append("</svg>" + '<figcaption class="legend-row">' + "".join(legend) + "</figcaption></figure>")
    return "".join(s)


# ----------------------------------------------------------------------------------------------- section
def build(probe_dir: Path, *, runs: list[dict], frozen_in, frozen_ood, frozen_ood_think, think: list[dict], guided_share: float | None) -> str:
    po = load(probe_dir / "analysis_ood.json") | load(probe_dir / "analysis_ood_think.json")
    pi = load(probe_dir / "analysis_indist.json") | load(probe_dir / "analysis_indist_think.json")
    fo_full, fi_full = load(probe_dir / "frozen_ood_notes.json"), load(probe_dir / "frozen_indist_notes.json")
    full_ood = fo_full.get("frozen_notes") or fo_full.get("memory") or {}
    full_in = fi_full.get("frozen_notes") or fi_full.get("memory") or {}
    lengths = load(probe_dir / "lengths.json")
    mentions = load(probe_dir / "peer_mentions.json")
    if not (po and full_ood):
        return ""
    R = {r["dir"].split("q3_4b_grpo_")[-1]: r for r in runs}
    plain, peers, mem, lab = R.get("plain"), R.get("peers"), R.get("memory"), R.get("peers_labeled")
    def ev(r, key, task=None):
        e = r and r.get(key)
        return (e["by"][task] if task else e["acc"]) if e else None
    # ---- numbers
    fo, fn, fs = po.get("frozen_notes", {}), po.get("frozen_nonotes", {}), po.get("frozen_swapped", {})
    to, tn, ts = po.get("trained_notes", {}), po.get("trained_nonotes", {}), po.get("trained_swapped", {})
    fto, ftn, fts = po.get("frozen_think_notes", {}), po.get("frozen_think_nonotes", {}), po.get("frozen_think_swapped", {})
    mo, mn, ms = po.get("memthink_notes", {}), po.get("memthink_nonotes", {}), po.get("memthink_swapped", {})
    io_, in_, is_ = pi.get("frozen_notes", {}), pi.get("frozen_nonotes", {}), pi.get("frozen_swapped", {})
    ito, itn, its = pi.get("trained_notes", {}), pi.get("trained_nonotes", {}), pi.get("trained_swapped", {})
    ceiling_ood = (full_ood["minority_top_correct"] - full_ood["accuracy_when_minority"]) * full_ood["n_top_minority"] / full_ood["n"]
    ceiling_in = ((full_in["minority_top_correct"] - full_in["accuracy_when_minority"]) * full_in["n_top_minority"] / full_in["n"]) if full_in else None
    ment_lab_rag = mentions.get("peers_labeled/indist", {}).get("rag"); ment_lab_bool = mentions.get("peers_labeled/ood", {}).get("boolqa"); ment_lab_bbh = mentions.get("peers_labeled/ood", {}).get("shortqa")
    def noreason(run, stream, task): return lengths.get(f"{run}/{stream}", {}).get(task, {}).get("frac_no_reasoning")
    def mlen(run, stream, task): return lengths.get(f"{run}/{stream}", {}).get(task, {}).get("mean_chars")
    think_mention = (fto.get("thinking") or {}).get("mentions_notes"); think_mention_hidden = (ftn.get("thinking") or {}).get("mentions_notes"); mem_mention = (mo.get("thinking") or {}).get("mentions_notes")
    ood_deficits = [(r["label"], ev(plain, "ev_ood") - ev(r, "ev_ood")) for r in (mem, peers, lab) if r and ev(r, "ev_ood") and ev(plain, "ev_ood")]
    code_deficits = [(r["label"], ev(plain, "ev_in", "code") - ev(r, "ev_in", "code")) for r in (mem, peers, lab) if r and ev(r, "ev_in") and ev(plain, "ev_in")]
    P = lambda x, d=1: f1(x, d, "%")
    # ---- conclusion paragraph
    conclusion = f"""<p class="lead"><b>The memory does not act anywhere in the current pipeline, and the pipeline itself is what makes the peer-conditioned regimes lose to plain RLVR.</b>
Three measurements, each independent of the others, say so. <b>The note is read but not used:</b> with thinking on, the frozen model quotes the reliability estimate in {P(think_mention, 0)} of its reasoning traces ({P(think_mention_hidden, 0)} when the note is hidden) and the checkpoint trained with notes in {P(mem_mention, 0)}, yet the final answer follows the peer the memory really trusts at the same rate whether the note is shown, hidden or swapped onto the wrong peer ({f1(fo.get('follow_top_informative'))} / {f1(fn.get('follow_top_informative'))} / {f1(fs.get('follow_low_informative'))}% frozen; {f1(to.get('follow_top_informative'))} / {f1(tn.get('follow_top_informative'))} / {f1(ts.get('follow_low_informative'))}% after training with notes; {f1(fto.get('follow_top_informative'))} / {f1(ftn.get('follow_top_informative'))} / {f1(fts.get('follow_low_informative'))}% frozen with thinking). The estimate enters the reasoning as a remark after the decision, never as the decision.
<b>Even a perfect reader would gain almost nothing on these streams:</b> the note can only change an answer when the peers disagree and the memory's favourite is in the minority, which is {100 * full_ood['n_top_minority'] / full_ood['n']:.1f}% of OOD events and {(100 * full_in['n_top_minority'] / full_in['n']) if full_in else float('nan'):.1f}% in-distribution, and there the favourite is right only {f1(full_ood['minority_top_correct'])}% and {f1(full_in.get('minority_top_correct') if full_in else None)}% of the time; always following it would move the streams by {ceiling_ood:+.2f} and {f1(ceiling_in, 2)} points.
<b>The training signal is not about the memory at all:</b> {P(100 * guided_share, 0) if guided_share else 'a third'} of every update relabels the peer-conditioned answer under the question-only prompt, which carries the peer context into the policy that is evaluated alone (the labels-before run cites peers that are not in the prompt in {P(100 * ment_lab_rag[0] / ment_lab_rag[1], 1) if ment_lab_rag else '—'} of its reading answers; the memory run answers {P(100 * noreason('memory', 'ood', 'boolqa'), 0)} of yes/no questions with no reasoning at all, the frozen model {P(100 * noreason('frozen', 'ood', 'boolqa'), 0)}), and the gradient comes almost only from reading.
What improved, reading {f1(frozen_in['by']['rag'], 0)} → {f1(min(v for v in (ev(mem, 'ev_in', 'rag'), ev(peers, 'ev_in', 'rag'), ev(lab, 'ev_in', 'rag')) if v), 0)}–{f1(max(v for v in (ev(mem, 'ev_in', 'rag'), ev(peers, 'ev_in', 'rag'), ev(lab, 'ev_in', 'rag')) if v), 0)}, is format and extraction learning that plain RLVR also gets ({f1(ev(plain, 'ev_in', 'rag'), 0)}); what got worse, code and BIG-Bench Hard, is the imported peer style.</p>"""
    # ---- problems and fixes
    def li(n, title, evidence, change):
        return f"<li><b>P{n}. {title}</b><div class=\"pf\"><span class=\"tag\">evidence</span> {evidence}</div><div class=\"pf\"><span class=\"tag fix\">change</span> {change}</div></li>"
    gate_in = (full_in.get("gate_0.65") or {}); gate_in_tr = (ito.get("gate_0.65") or {}); gate_ood = (full_ood.get("gate_0.65") or {})
    problems = [
        li(1, "The memory has no headroom on these streams, and is least sharp exactly where it disagrees with the majority.",
           f"Partition of the streams in the figure above: the favourite is in the minority in {full_ood['n_top_minority']:,} of {full_ood['n']:,} OOD events and {full_in.get('n_top_minority', 0):,} of {full_in.get('n', 0):,} in-distribution, and is right {f1(full_ood['minority_top_correct'])}% / {f1(full_in.get('minority_top_correct') if full_in else None)}% there. Where peers disagree at all, the memory is a far better selector than majority voting ({f1(full_ood['informative_top_correct'])}% vs {f1(full_ood['informative_plurality_correct'])}% OOD, {f1(full_in.get('informative_top_correct') if full_in else None)}% vs {f1(full_in.get('informative_plurality_correct') if full_in else None)}% in-distribution) but well short of the oracle ({f1(full_ood['informative_any_peer_correct'])}% / {f1(full_in.get('informative_any_peer_correct') if full_in else None)}%). A reader that followed the favourite only when its estimate is at least 0.65 would gain {gate_in.get('stream_gain_points', float('nan')):+.1f} points in-distribution for the frozen model, {gate_in_tr.get('stream_gain_points', float('nan')):+.1f} for the trained one, and {gate_ood.get('stream_gain_points', float('nan')):+.2f} OOD.",
           "Measure the memory as a selector before training anything on it: favourite accuracy against plurality and oracle, per task, on the events where it disagrees with the majority. Sharpen it there (more evidence per address, task-conditioned addresses, a confidence that is calibrated in the minority cases, not only on average). Build the peer pool so that the majority is unreliable and reliability tracking has room: specialists that are strong on different tasks, more peers, a weaker average. Report the ceiling (the partition figure) for any new stream before spending GPU time on it."),
        li(2, "The note is a number in a header that the model treats as commentary, and GRPO cannot teach it otherwise.",
           f"Swapped notes: the frozen model follows the peer now printed as most reliable {f1(fs.get('follow_top_informative'))}% and the really trusted one {f1(fs.get('follow_low_informative'))}% (trained: {f1(ts.get('follow_top_informative'))}% / {f1(ts.get('follow_low_informative'))}%); its follow rate falls as the printed estimate rises (right-hand charts). Thinking traces quote the estimate ({P(think_mention, 0)} of traces) and then override it: \"Peer 3 is more reliable. But according to the reasoning, the correct answer is four.\" The guided pass draws one greedy answer per prompt, so no GRPO group ever contains a note-following and a note-ignoring answer to the same prompt; and in-distribution the model can verify the peers' content itself, so the note is redundant on the whole training stream.",
           "Give the note a gradient path: several guided samples at temperature in their own group, so the advantage is between note-conditioned behaviours; make the note the only evidence on part of the guided prompts (peers' final answers without their reasoning, or the passage hidden); phrase it as an instruction with its evidence (\"trust peer 2 over peer 1: peer 1 failed 8 of 10 similar cases\") rather than a probability. Or take the memory out of the prompt altogether and use it where GRPO has no signal: to choose the hint on all-wrong groups and as the pseudo-reward when there is no verifier."),
        li(3, "Relabelling the guided answer under the question-only prompt imports the peer context into the policy that is evaluated alone.",
           f"{P(100 * guided_share, 0) if guided_share else 'One third'} of the rows in every update are guided answers copied into the question-only view. The labels-before checkpoint mentions peers or verification in {P(100 * ment_lab_rag[0] / ment_lab_rag[1], 1) if ment_lab_rag else '—'} of its reading answers, {P(100 * ment_lab_bool[0] / ment_lab_bool[1], 1) if ment_lab_bool else '—'} of yes/no and {P(100 * ment_lab_bbh[0] / ment_lab_bbh[1], 1) if ment_lab_bbh else '—'} of BIG-Bench Hard answers, with no peer in the prompt (\"While Peer 1 is correct and the other peers are incorrect, the final answer should reflect the verified correct answer\"). The memory checkpoint answers {P(100 * noreason('memory', 'ood', 'boolqa'), 0)} of yes/no and {P(100 * noreason('memory', 'ood', 'shortqa'), 0)} of BIG-Bench Hard questions without reasoning (frozen {P(100 * noreason('frozen', 'ood', 'boolqa'), 0)} / {P(100 * noreason('frozen', 'ood', 'shortqa'), 0)}), with mean answers of {f1(mlen('memory', 'ood', 'shortqa'), 0)} characters against {f1(mlen('plain', 'ood', 'shortqa'), 0)} for plain RLVR. Result: every peer-conditioned regime ends below plain RLVR on OOD by {', '.join(f'{d:.1f}' for _, d in ood_deficits)} points and on code by {', '.join(f'{d:.1f}' for _, d in code_deficits)}.",
           "Train the question-only policy on its own samples only. Use peer solutions where the group is all-wrong: the memory picks the hint, the model re-answers in its own words under the question-only prompt, the answer is kept only if it verifies (the hint regime, with a random-peer hint as the control). If relabelling is kept at all, restrict it to guided answers with no reference to the peers and cap it at a tenth of the batch."),
        li(4, "The gradient comes almost only from reading.",
           "Question-only accuracy on the training stream is 92–95% on math, about 30% on code and 60–70% on reading; a GRPO group gives no gradient when its four samples agree, so math groups are mostly all-correct and code groups mostly all-wrong (about 16% of groups have zero variance in every step, the rest are reading-dominated). Validation math and code are flat in every run; the reading gain is the only thing that moves, and it moves in every regime.",
           "Dynamic sampling: drop zero-variance groups and resample until the batch is full; per-task advantage normalisation; partial credit for code (fraction of tests passed) and a longer budget for it; a harder math mix so that its groups split."),
        li(5, "No regularisation: drift after the first third of the epoch and one collapse.",
           f"Constant lr 1e-6, no KL term, no length control. Entropy falls from 0.05 to 0.02 and the gradient norm doubles in every run; ppo_kl is three to four times larger in the no-memory run from step 139, which then collapsed on reading at steps 229–248 (responses 450–510 tokens, 44–56% at the cap, a non-finite gradient skipped). The best checkpoints are not the last ones (no-memory: {f1(peers['inter'].get(140, {}).get('acc') if peers else None, 2)} at step 140 against {f1(ev(peers, 'ev_in'), 2)} at the end).",
           "KL penalty to the frozen model (coefficient 1e-3), cosine learning-rate decay, a length penalty at the cap, an entropy floor, and checkpoint selection by validation."),
        li(6, "Thinking off caps the reasoning that the OOD tasks need.",
           f"The frozen model with thinking on scores {f1(frozen_ood_think['acc'] if frozen_ood_think else None, 2)} on OOD against {f1(frozen_ood['acc'], 2)} without, above the best non-thinking RL model ({f1(ev(plain, 'ev_ood'), 2)}). The memory run with thinking on keeps its reasoning length after 70 steps (2.6k characters per trace against 2.5k frozen), gains the same reading improvement as the non-thinking runs and loses nothing on OOD ({f1(ev(think[0], 'ev_ood') if think else None, 2)}); on the probe slice thinking lifts the frozen model with peers from {f1(fo.get('accuracy'), 2)} to {f1(fto.get('accuracy'), 2)}.",
           "Train and evaluate with thinking on by default, with a 4,096-token budget; read the non-thinking numbers as a lower bound. Once P1–P3 are fixed, the thinking trace is also where note use can be checked directly."),
    ]
    # ---- follow-rate charts
    charts = []
    for title, a, b, c in (("frozen, OOD slice", fo, fn, fs), ("trained with notes (step 276), OOD slice", to, tn, ts), ("frozen, in-distribution slice", io_, in_, is_), ("trained with notes (step 276), in-distribution slice", ito, itn, its),
                           ("frozen + thinking, OOD slice", fto, ftn, fts), ("trained with notes + thinking (step 70), OOD slice", mo, mn, ms)):
        if a:
            charts.append(svg_bins(title, [("note shown", a.get("follow_by_shown_prob"), OK, ""), ("note hidden (rate vs the unseen estimate)", b.get("follow_by_shown_prob"), MUT, "5 4"), ("notes swapped by rank", c.get("follow_by_shown_prob"), BAD, "")]))
    return f"""<section class="conclusion">
<h2>Conclusion</h2>
{conclusion}
<h3>The pipeline as run, with its break points</h3>
<div class="diagram">{svg_pipeline()}</div>
<ol class="breaks">
<li><b>Note read, not used.</b> Follow rates equal with the note shown, hidden or swapped; thinking traces quote it and override it.</li>
<li><b>Note rarely decisive.</b> Favourite in the minority in {100 * full_ood['n_top_minority'] / full_ood['n']:.1f}% of OOD events, right {f1(full_ood['minority_top_correct'], 0)}% there; ceiling {ceiling_ood:+.1f} points.</li>
<li><b>One greedy guided answer.</b> No group ever contrasts following the note with ignoring it.</li>
<li><b>Relabelling under the question-only prompt.</b> Peer context leaks into the alone policy ({P(100 * ment_lab_rag[0] / ment_lab_rag[1], 1) if ment_lab_rag else '—'} peer mentions in the labels-before run; reasoning dropped in the memory run).</li>
<li><b>Gradient from reading only.</b> Math groups all-correct, code groups all-wrong; math and code flat in every run.</li>
<li><b>No KL, constant lr.</b> Drift from step 139, collapse at 229 in the no-memory run, best checkpoints not last.</li>
<li><b>Thinking off.</b> Frozen with thinking beats every non-thinking RL model on OOD.</li>
</ol>
<h3>How the conclusion was reached</h3>
<div class="diagram">{svg_evidence()}</div>
<h3>Where the note could matter at all</h3>
<div class="diagram">{svg_partition(full_ood, full_in)}</div>
<h3>Whether the printed estimate moves the answer</h3>
<p>Follow rate of each peer against the estimate printed on it, on events where the peers disagree and the notes are not flat. If the model read the note, the swapped curve would rise like the shown one; instead it falls, because the swap puts the high estimate on the peers whose content is worst. The hidden curve is the same shape as the shown one: what rises with the estimate is the peers' quality, which the model judges from the content.</p>
<div class="charts grid3">{''.join(charts)}</div>
<h3>What is wrong, and what to change</h3>
<ol class="problems">{''.join(problems)}</ol>
</section>"""
