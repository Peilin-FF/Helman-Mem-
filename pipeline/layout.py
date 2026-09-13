"""Where everything lives: the one mapping from (stage, model, stream, condition) to paths, shared by run and table.

    data/<stream>/<split>.jsonl                                   released and derived streams
    outputs/peers/<stream>/<honest|misleading>/<peer>/            a peer's answers (pipeline.peers)
    outputs/features/<model>/<stream>/shard<k>.pt                 the judge's features (pipeline.features)
    outputs/record/<model>/<stream>/<order>.fit-<fit>.jsonl       the record along the stream (+ .quality.json)
    outputs/eval/<model or run>/<stream>/<condition>/             an evaluation (pipeline.evaluate)
    outputs/train/<run>/phase<k>/                                 a training run (pipeline.train); data in outputs/train/data/
    outputs/tables/<experiment>.md, outputs/runs/<experiment>/    the result table, the resolved config and commands
    logs/<experiment>/<job>.log

A smoke run (--smoke) uses outputs/smoke/ and logs/smoke/ for everything it writes, derived streams included, so it can
never be mistaken for, or skip, a real run.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import REPO

ADV = "_adv_"


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
        self.derived = self.outputs / "data" if smoke else self.data
        self.models_root = Path(paths.get("models_root", "models"))

    def _abs(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.repo / p

    # --- registries -------------------------------------------------------------------------------------------------
    def model(self, tag: str) -> dict:
        spec = self.cfg.get("models", {}).get(tag)
        if spec is None:
            raise KeyError(f"model {tag!r} is not in the models registry (configs/base.yaml)")
        spec = {"path": spec} if isinstance(spec, str) else dict(spec)
        path = Path(spec["path"])
        spec["path"] = path if path.is_absolute() else self.models_root / path
        return spec

    def peer_models(self) -> list[dict]:
        out = []
        for p in self.cfg.get("peers", []):
            p = dict(p)
            path = Path(p.get("path", p["name"]))
            p["path"] = path if path.is_absolute() else self.models_root / path
            out.append(p)
        return out

    def stream(self, name: str) -> dict:
        """{'path', 'peers', 'base'}; '<base>_adv_<regime>' streams are derived next to their base."""
        reg = self.cfg.get("streams", {})
        if name in reg:
            spec = dict(reg[name])
            return {"path": self._abs(Path(self.cfg.get("paths", {}).get("data", "data")) / spec["path"]), "peers": int(spec.get("peers", 6)), "base": None}
        if ADV in name:
            base, _, _regime = name.partition(ADV)
            if base in reg:
                b = self.stream(base)
                rel = Path(reg[base]["path"])
                return {"path": self.derived / f"{rel.parent}{ADV}{_regime}" / rel.name, "peers": b["peers"], "base": base}
        raise KeyError(f"stream {name!r} is not in the streams registry (configs/base.yaml)")

    # --- artifacts --------------------------------------------------------------------------------------------------
    def peers_dir(self, stream: str, mode: str, peer: str) -> Path:
        return self.outputs / "peers" / stream / mode / peer

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
