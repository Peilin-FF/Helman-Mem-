"""The Kalman Mem pipeline: one entry point per stage, driven by the experiment YAMLs in configs/ through pipeline.run.

    peers     a peer model's answers on a stream (honest, or verified misleading)
    streams   a misleading stream built from a stream and its peers' misleading answers under a regime
    own       the central model's question-only answer joins a stream as one more answer (with own_answer: true)
    features  the frozen judge's hidden states that address the record
    record    the Bayesian record along a stream, read before write, and its quality
    evaluate  the central model answers each event under a condition (tilt, peers, solo, swap)
    decide    per event, the central model's reading or its own answer, by its reading line
    train     GRPO of the central model (arm data, then the phases)
    table     the experiment's result table

Every stage takes explicit input and output paths; pipeline.run derives them from the experiment config and the registered
datasets, models and peers (pipeline.registry, configs/{datasets,models,peers}/).
"""
