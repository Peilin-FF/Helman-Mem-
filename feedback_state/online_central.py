"""The models of an online swarm run (feedback_state.online_swarm), shared by every benchmark:

  serve / stop         vLLM OpenAI servers (the peers, and any role the benchmark's own code calls over HTTP), one per model,
                       spread over the given GPUs with the memory left on each
  central_engine       the central model in-process: vLLM with the tilt's kernels for its answers (feedback_state.vllm_attn_bias)
                       and the frozen judge (transformers) whose hidden states address the record
  projections          the record's address maps (PCA, label-free), fit on a registered stream's judge features
  make_tracks          the tracks of the run's conditions
"""
from __future__ import annotations

import collections
import json
from pathlib import Path


def serve(models: list[dict], gpus: list[str], port: int, out_root: Path, max_model_len: int, reserved: dict | None = None) -> list[dict]:
    """Start a vLLM OpenAI server per model ({name, path, env_vars, prefix_caching, trust_remote_code}), model i on
    gpus[i % len(gpus)] at port + i, each with an equal share of what `reserved` (gpu -> fraction) leaves of its GPU; returns
    the servers ({name, proc, log, port, gpu}) once all are up. A share under 15% of a GPU is warned about."""
    from feedback_state.swarm import start_vllm, wait_vllm

    reserved = reserved or {}
    per_gpu = collections.Counter(gpus[i % len(gpus)] for i in range(len(models)))
    room = {g: 0.88 - float(reserved.get(g, 0.0)) for g in per_gpu}
    if any(room[g] / per_gpu[g] < 0.15 for g in per_gpu):
        print(f"[online] WARNING: {len(models)} servers on GPUs {sorted(per_gpu)} leave less than 15% of a GPU to some model; give the "
              f"job more GPUs", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    servers = []
    try:
        for i, m in enumerate(models):
            gpu = gpus[i % len(gpus)]
            log = out_root / f"vllm_{m['name']}.log"
            proc = start_vllm(m["path"], m["name"], port + i, log, gpu_memory_utilization=round(max(0.05, min(0.85, room[gpu] / per_gpu[gpu])), 2),
                              max_model_len=max_model_len, tool_parser=m.get("tool_parser", "hermes"), gpu=gpu, env_vars=m.get("env_vars"),
                              prefix_caching=m.get("prefix_caching", True), trust_remote_code=bool(m.get("trust_remote_code")))
            servers.append({"name": m["name"], "proc": proc, "log": log, "port": port + i, "gpu": gpu})
            print(f"[online] serving {m['name']} on GPU {gpu} port {port + i}", flush=True)
        for s in servers:
            wait_vllm(s["proc"], s["name"], s["port"], s["log"])
    except BaseException:
        stop(servers)
        raise
    return servers


def stop(servers: list[dict]) -> None:
    for s in servers:
        s["proc"].terminate()
    for s in servers:
        try:
            s["proc"].wait(30)
        except Exception:
            s["proc"].kill()


def central_engine(model: str, *, gpu_memory_utilization: float, max_model_len: int, max_new_tokens: int, bias_form: str,
                   judge_max_length: int, num_answers: int):
    """(central_fn, features_fn) on the first visible GPU: the central model's answers (greedy, thinking off, the tilt registered per
    prompt, as pipeline.evaluate), and the judge's features of an event's answers (as pipeline.features)."""
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from feedback_state import vllm_attn_bias
    from feedback_state.attn_bias import prompt_token_bias
    from feedback_state.judge_features import event_features
    from feedback_state.judge_prompt import yes_no_token_ids
    from feedback_state.memory_generator import render_prompt
    from feedback_state.newarch_loader import dtype_from_name, load_central_model

    vllm_attn_bias.install()
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    llm = LLM(model=model, tokenizer=model, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              enable_prefix_caching=True, trust_remote_code=False, seed=0, disable_sliding_window=True)
    device = torch.device("cuda:0")
    judge = load_central_model(model, dtype=dtype_from_name("bfloat16"), local_files_only=True).to(device=device, dtype=torch.bfloat16)
    judge.eval()
    yes_ids, no_ids = yes_no_token_ids(tok)
    layers = [None]

    def central_fn(requests):
        if not requests:
            return []
        ids_list = []
        for q in requests:
            prompt = render_prompt(tok, q["messages"])
            if q.get("probs") is not None:
                ids, bias = prompt_token_bias(tok, prompt, q["texts"], q["probs"], q["gamma"], bias_form)
                vllm_attn_bias.register(ids, bias)
            else:
                ids = tok(prompt, add_special_tokens=False)["input_ids"]
            ids_list.append(ids)
        outs = llm.generate([{"prompt_token_ids": ids} for ids in ids_list], SamplingParams(temperature=0.0, max_tokens=max_new_tokens), use_tqdm=False)
        vllm_attn_bias.clear()
        return [o.outputs[0].text for o in outs]

    def features_fn(ev, texts):
        f = event_features(judge, tok, ev, texts, num_peers=num_answers, yes_id=int(yes_ids[0]), no_id=int(no_ids[0]),
                           max_length=judge_max_length, question_max_length=2048, device=device, layers=layers[0])
        layers[0] = f["layers"]
        return f

    return central_fn, features_fn


def projections(fit_stream: Path | None, fit_features: Path | None, fit_peers: int | None, dim: int):
    """The address maps fit (label-free) on a stream's judge features; (None, None) without one (the records fit on a warmup)."""
    if fit_stream is None:
        return None, None
    import torch

    from feedback_state.addresses import Projection
    from feedback_state.feature_streams import load_stream_from

    fit = load_stream_from(fit_stream, fit_features, num_peers=int(fit_peers))
    return (Projection(fit.sem, dim, torch.device("cpu")),
            Projection(fit.peer_hidden.reshape(-1, fit.peer_hidden.shape[-1]), dim, torch.device("cpu")))


def make_tracks(conditions: dict | str, proj_q, proj_c, num_answers: int, *, design: str, lam: float, dim: int, warmup: int,
                prior: str, line_lam: float, seed: int) -> list:
    """One track per condition ({name: {kind, gamma}}): a record for tilt and combination (on the given projections, else fit on a
    warmup of `warmup` events), a reading line for combination."""
    from feedback_state.online_swarm import OnlineRecord, Track
    from feedback_state.reading_line import ReadingLine

    conditions = json.loads(conditions) if isinstance(conditions, str) else conditions
    tracks = []
    for name, cd in conditions.items():
        kind = cd["kind"]
        rec = OnlineRecord(proj_q, proj_c, num_answers, design=design, lam=lam, dim=dim, warmup=0 if proj_q is not None else warmup) \
            if kind in ("tilt", "combination") else None
        line = ReadingLine(tuple(float(x) for x in str(prior).split(",")), line_lam) if kind == "combination" else None
        tracks.append(Track(name=name, kind=kind, gamma=float(cd.get("gamma", 3.0)), record=rec, line=line, seed=seed))
    return tracks


def save_tracks(tracks: list, out_root: Path, adapter=None, **meta) -> None:
    """Each track as an evaluation: <out_root>/<track>/generations.jsonl (a row per event, in stream order), units.jsonl (the
    finished tasks: classes, diagnoses) and eval_metrics.json (feedback_state.online_swarm.metrics, and `meta`)."""
    from feedback_state.online_swarm import metrics

    for t in tracks:
        out = out_root / t.name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "generations.jsonl").open("w") as f:
            for r in sorted(t.rows, key=lambda r: r["pos"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out / "units.jsonl").open("w") as f:
            for u in t.units:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        m = dict(metrics(t, adapter), **meta)
        (out / "eval_metrics.json").write_text(json.dumps(m, indent=1))
        print(f"[online] {t.name}: events {100 * m['accuracy']:.1f} right; {adapter.progress(t) if adapter else ''} -> {out}", flush=True)
