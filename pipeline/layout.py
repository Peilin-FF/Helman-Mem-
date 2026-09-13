"""Where everything lives: the one mapping from (stage, model, stream, condition) to paths, shared by run and table.

    data/<dataset>/                                               registered datasets (configs/datasets/): released
                                                                  streams, peers' generated answers (<dataset>/<peer>/),
                                                                  misleading streams (+ manifest.json)
    outputs/features/<model>/<stream>/shard<k>.pt                 the judge's features (pipeline.features)
    outputs/record/<model>/<stream>/<order>.fit-<fit>.jsonl       the record along the stream (+ .quality.json)
    outputs/eval/<model or run>/<stream>/<condition>/             an evaluation (pipeline.evaluate)
    outputs/train/<run>/                                          a training run (pipeline.train); data in outputs/train/data/
    outputs/tables/<experiment>.md, outputs/runs/<experiment>/    the result table, the resolved config and commands
    logs/<experiment>/<job>.log

A smoke run (--smoke) uses outputs/smoke/ and logs/smoke/ for everything it writes, built datasets included, so it can
never be mistaken for, or skip, a real run. An experiment that only evaluates released datasets sets
`smoke: {use_released_data: true}`: its smoke run reads them from paths.data and never builds any.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import REPO
from pipeline.registry import load_registry


class Layout:
    def __init__(self, cfg: dict, smoke: bool = False):
        self.cfg, self.smoke = cfg, smoke
        paths = cfg.get("paths", {})
        self.repo = REPO
        self.data = self._abs(paths.get("data", "data"))
        outputs = self._abs(paths.get("outputs", "outputs"))
        logs = self._abs(paths.get("logs", "logs"))
        self.outputs = outputs / "smoke" if smoke else outputs
        self.logs = logs / "smoke" if smoke else logs
        # built datasets: a smoke run builds its own under outputs/smoke/data, unless the experiment only reads released ones
        self.derived = self.outputs / "data" if smoke and not cfg.get("smoke", {}).get("use_released_data") else self.data
        self.models_root = Path(paths.get("models_root", "models"))
        self.registry = load_registry(paths.get("registry"))

    def _abs(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.repo / p

    # --- registered names -> paths --------------------------------------------------------------------------------
    def model(self, tag: str) -> dict:
        spec = dict(self.registry.model(tag))
        path = Path(spec["path"])
        spec["path"] = path if path.is_absolute() else self.models_root / path
        return spec

    def peer_models(self, peer_set: str) -> list[dict]:
        """The models of a peer set in peer_0 ... order; `name` is the model directory, which names its answers."""
        out = []
        for i, tag in enumerate(self.registry.peer_set(peer_set)):
            spec = self.model(tag)
            out.append(dict(spec, tag=tag, index=i, name=spec["path"].name))
        return out

    def stream(self, name: str) -> dict:
        """A registered dataset: {'name', 'kind', 'path', 'peers' (count), 'peer_set', 'base', 'answers', 'regime', ...}.

        A released stream is read from paths.data; what the pipeline builds (answers, misleading streams) goes to
        paths.data too, or to outputs/smoke/data in a smoke run, so a smoke run never writes next to real data.
        """
        d = dict(self.registry.dataset(name))
        base = self.registry.stream_of(name)
        d["peer_set"] = base["peers"]
        d["peers"] = len(self.registry.peer_set(base["peers"]))
        if d["kind"] == "stream":
            d["path"] = self._abs(Path(self.cfg.get("paths", {}).get("data", "data")) / d["path"])
        elif d["kind"] == "answers":
            d["path"] = self.derived / d.get("path", name)
        else:
            d["path"] = self.derived / d.get("path", f"{name}/{Path(base['path']).name}")
        return d

    # --- artifacts --------------------------------------------------------------------------------------------------
    def features_dir(self, model: str, stream: str) -> Path:
        return self.outputs / "features" / model / stream

    def record_file(self, model: str, stream: str) -> Path:
        rec = self.cfg.get("record", {})
        fit = rec.get("fit", "self")
        order = rec.get("order", "shuffled0")
        extra = ""
        if int(rec.get("dim", 256)) != 256 or float(rec.get("lam", 100.0)) != 100.0 or rec.get("design", "qc") != "qc":
            extra = f".{rec.get('design', 'qc')}-d{int(rec.get('dim', 256))}-lam{float(rec.get('lam', 100.0)):g}"
        return self.outputs / "record" / model / stream / f"{order}.fit-{fit}{extra}.jsonl"

    @staticmethod
    def quality_file(record_file: Path) -> Path:
        return record_file.with_name(record_file.name[: -len(".jsonl")] + ".quality.json")

    def eval_dataset(self, dataset: str, condition: dict) -> str:
        """The dataset whose result a condition reads: a question-only condition sees no peer answers, so every dataset built
        on a stream (its misleading variants) shares the stream's result."""
        return self.registry.stream_of(dataset)["name"] if condition.get("mode") == "solo" else dataset

    def eval_dir(self, model: str, stream: str, condition: str) -> Path:
        return self.outputs / "eval" / model / stream / condition

    def train_dir(self, run: str) -> Path:
        return self.outputs / "train" / run

    def table_file(self, experiment: str) -> Path:
        return self.outputs / "tables" / f"{experiment}.md"

    def run_dir(self, experiment: str) -> Path:
        return self.outputs / "runs" / experiment

    def log(self, experiment: str, job: str) -> Path:
        return self.logs / experiment / f"{job}.log"


def feature_stream(name: str, model: str, *, config: str = "configs/base.yaml", limit: int | None = None):
    """A stream joined with a model's features, by registry name (for the analysis scripts)."""
    from feedback_state.feature_streams import load_stream_from
    from pipeline.config import load

    L = Layout(load(config))
    s = L.stream(name)
    return load_stream_from(s["path"], L.features_dir(model, name), name=name, model=model, num_peers=s["peers"], limit=limit)
