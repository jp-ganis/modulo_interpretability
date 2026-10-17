"""Wheels within wheels: a transformer grokking modular addition.

Modules:
    config    - every hyperparameter in one dataclass
    data      - the (a + b) mod p dataset and train/test split
    model     - a hand-written 1-layer transformer (no LayerNorm)
    train     - training loop: `python -m grokking.train`
    fourier   - Fourier basis over the integers mod p
    analyse   - per-checkpoint analysis: `python -m grokking.analyse`
    progress  - Nanda et al.'s progress measures (restricted / excluded loss)
    plots     - static Matplotlib figures for the README
    export    - compact JSON for the web demo
"""
