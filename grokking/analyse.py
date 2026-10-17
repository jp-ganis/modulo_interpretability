"""Analyse every checkpoint of a run.

    python -m grokking.analyse [--run-name p113_seed0] [--frames 300] [--no-plots] [--no-export]

For each checkpoint this computes:
  1. the Fourier spectrum of the embedding W_E (strength per frequency)
  2. "ring" coordinates of every number in each key frequency's cos/sin plane
  3. a 2-D PCA view of W_E
  4. Nanda et al.'s progress measures (restricted / excluded loss)

and then picks which checkpoints become frames of the web animation.
Results go to runs/<run>/analysis.npz; afterwards the static figures
(grokking.plots) and the web JSON (grokking.export) are regenerated.

The key design choice for the animation: rings and PCA use a *fixed* basis
taken from the final checkpoint. If every frame used its own basis, the axes
would rotate and flip from frame to frame and you couldn't tell structure
emerging from the camera moving. With a fixed basis, early frames show what
the eventual circle directions looked like before training found them
(noise), and you watch the circle form.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import make_dataset
from .fourier import cos_sin_rows, fourier_basis, frequency_strengths, gini
from .model import Transformer
from .progress import progress_measures


# --- Loading --------------------------------------------------------------

def checkpoint_paths(run_dir: Path) -> list[tuple[int, Path]]:
    paths = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    return [(int(p.stem.split("_")[1]), p) for p in paths]


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


# --- Key frequencies ------------------------------------------------------

def choose_key_frequencies(strengths: np.ndarray, max_keys: int = 12) -> list[int]:
    """Pick the key frequencies from the final checkpoint's spectrum.

    Sort strengths in descending order and cut at the biggest relative drop
    (largest ratio between neighbours) among the top `max_keys`. A trained
    model typically has a handful of strong frequencies and then a cliff down
    to a noise floor, so the cliff is a natural, threshold-free cut.
    Frequencies are numbered from 1, so index i -> k = i + 1.
    """
    order = np.argsort(strengths)[::-1]
    s = strengths[order[: max_keys + 1]]
    n_keys = int(np.argmax(s[:-1] / s[1:])) + 1
    return sorted((order[:n_keys] + 1).tolist())


# --- Fixed-basis projections ----------------------------------------------

def ring_basis(W_final: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """A (d_model, 2) matrix that maps an embedding to its 2-D ring coordinates.

    For the final checkpoint, u_cos = W^T cos_k and u_sin = W^T sin_k are the
    directions in embedding space that W_E uses to encode cos(w a) and
    sin(w a). They span a plane. We use the pseudo-inverse of U = [u_cos u_sin],
    i.e. an orthogonal projection onto that plane followed by a 2x2 change of
    coordinates. The coordinates are chosen so that, at the final checkpoint,
    number a lands at (cos(w a), sin(w a)) up to small leakage from other
    frequencies: a unit circle with a at angle 2*pi*k*a/p. A plain
    orthonormal (Gram-Schmidt) basis of the same plane would show an
    ellipse instead, whenever u_cos and u_sin differ in length or aren't
    perpendicular.

    The sqrt(p/2) factor undoes the unit-normalisation of the basis rows.
    """
    p = W_final.shape[0]
    cos_k, sin_k = cos_sin_rows(B, k)
    U = np.stack([W_final.T @ cos_k, W_final.T @ sin_k], axis=1)  # (d, 2)
    return np.linalg.pinv(U).T * np.sqrt(p / 2)                     # (d, 2)


def pca_basis(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-2 principal directions (d, 2) of the (already centred) rows of W, plus explained-variance ratios."""
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[:2].T, (S[:2] ** 2) / (S ** 2).sum()


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rotate/reflect 2-D points X to best match Y (orthogonal Procrustes).

    Each checkpoint's own PCA axes are only defined up to sign and, when two
    components have similar variance, up to rotation. Aligning each frame to
    its neighbour removes those arbitrary flips so the per-frame view animates
    smoothly. Solution: with X^T Y = U S V^T, the best orthogonal R is U V^T.
    """
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return X @ (U @ Vt)


# --- Frame selection ------------------------------------------------------

def select_frames(steps: np.ndarray, signals: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    """Choose which checkpoints become animation frames.

    Think of each checkpoint as a point in a space whose axes are the curves
    we plot (log losses, weight norm, Fourier sparsity, ...), each rescaled
    to [0, 1], plus two "time" axes (log step and linear step). The training
    run traces a path through that space. We pick frames evenly spaced along
    the path's *length*, so stretches where nothing changes get few frames
    and the grokking transition, where everything moves at once, gets many.
    The time axes keep quiet periods from being skipped entirely.
    """
    if len(steps) <= n_frames:
        return np.arange(len(steps))
    cols = [np.log10(steps + 1), steps.astype(float)] + list(signals.values())
    X = np.stack([(c - c.min()) / (np.ptp(c) or 1.0) for c in cols], axis=1)
    seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0, arc[-1], n_frames)
    # Nearest checkpoint to each target; duplicates collapse, so top up with
    # the checkpoints that sit furthest (in arc length) from any chosen one.
    chosen = set(np.abs(arc[None, :] - targets[:, None]).argmin(axis=1).tolist())
    while len(chosen) < n_frames:
        idx = np.array(sorted(chosen))
        gap = np.abs(arc[:, None] - arc[idx][None, :]).min(axis=1)
        chosen.add(int(gap.argmax()))
    return np.array(sorted(chosen))


# --- Phases ---------------------------------------------------------------

def phase_boundaries(a: dict[str, np.ndarray], p: int) -> dict[str, int]:
    """Split training into Nanda et al.'s three phases, using simple, explainable rules.

    a: the arrays saved in analysis.npz.

    memorisation      0 -> the step where excluded loss is lowest, i.e. where the
                      non-Fourier part of the network on its own fits the
                      training set best. Memorisation peaks here.
    circuit formation -> the first step where test loss drops below ln(p), the
                      loss of a uniform random guess. Before this the model is
                      worse than guessing on unseen pairs, even though the
                      Fourier circuit has been growing all along.
    cleanup           -> the first step after which the weight norm stays within
                      5% of its final value. Weight decay has finished pruning
                      the leftover memorisation weights.
    """
    steps = a["steps"]
    mem_end = int(steps[np.argmin(a["excluded_loss"])])
    beats_guessing = (a["ckpt_test_loss"] < np.log(p)) & (steps > mem_end)
    cleanup_start = int(steps[np.argmax(beats_guessing)])
    norm = a["ckpt_weight_norm"]
    outside = np.flatnonzero(norm > 1.05 * norm[-1])
    cleanup_end = int(steps[outside[-1] + 1]) if len(outside) else int(steps[-1])
    return {"memorisation_end": mem_end, "cleanup_start": cleanup_start, "cleanup_end": cleanup_end}


def grok_step(test_acc: np.ndarray, threshold: float = 0.95) -> int:
    """First training step at which test accuracy reaches `threshold`: our 'grokking moment'."""
    return int(np.argmax(test_acc >= threshold))


# --- Main -----------------------------------------------------------------

def analyse(cfg: Config, n_frames: int) -> Path:
    run_dir = cfg.run_dir
    p = cfg.p
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = checkpoint_paths(run_dir)
    steps = np.array([s for s, _ in ckpts])
    print(f"{len(ckpts)} checkpoints, steps {steps[0]}..{steps[-1]}")

    # Embedding rows for the p numbers (the '=' row is excluded: it isn't a
    # point on the circle). Centred per checkpoint: the mean embedding is the
    # constant (k = 0) Fourier component and carries no information about
    # *which* number it is.
    W_all = np.stack([load_state(path)["W_E"][:p].double().numpy() for _, path in ckpts])  # (N, p, d)
    W_centred = W_all - W_all.mean(axis=1, keepdims=True)

    # 1. Fourier spectrum per checkpoint
    B, _ = fourier_basis(p)
    spectrum = np.stack([frequency_strengths(W, B)[0] for W in W_all])  # (N, p//2)
    spectrum_gini = np.array([gini(s) for s in spectrum])

    # 2. Key frequencies from the final checkpoint
    key_freqs = choose_key_frequencies(spectrum[-1])
    energy = spectrum[-1] ** 2
    key_share = energy[np.array(key_freqs) - 1].sum() / energy.sum()
    print(f"key frequencies: {key_freqs}  ({key_share:.1%} of non-constant embedding energy)")

    # 3. Ring coordinates in the final checkpoint's fixed cos/sin planes
    W_final = W_centred[-1]
    rings = np.stack([W_centred @ ring_basis(W_final, B, k) for k in key_freqs], axis=1)  # (N, K, p, 2)

    # 4. PCA: fixed final-checkpoint basis, plus per-checkpoint aligned view
    V_final, pca_explained = pca_basis(W_final)
    pca_fixed = W_centred @ V_final                                     # (N, p, 2)
    pca_aligned = np.empty_like(pca_fixed)
    pca_aligned[-1] = pca_fixed[-1]
    for i in range(len(steps) - 2, -1, -1):                              # walk backwards from the end
        own = W_centred[i] @ pca_basis(W_centred[i])[0]
        pca_aligned[i] = procrustes_align(own, pca_aligned[i + 1])
    print(f"final PCA explained variance: {pca_explained.round(3)}")

    # 5. Progress measures (needs a forward pass over all p^2 inputs)
    tokens, labels = make_dataset(p)
    train_idx = torch.load(run_dir / "split.pt")["train_idx"].numpy()
    labels_np = labels.numpy()
    model = Transformer(cfg).to(device).eval()
    restricted, excluded = np.empty(len(steps)), np.empty(len(steps))
    with torch.no_grad():
        x = tokens.to(device)
        for i, (_, path) in enumerate(ckpts):
            model.load_state_dict(load_state(path))
            logits = model(x).double().cpu().numpy()
            restricted[i], excluded[i] = progress_measures(logits, labels_np, train_idx, p, key_freqs)

    # Losses / weight norm at each checkpoint step, from the training log
    log = dict(np.load(run_dir / "log.npz"))
    at = {k: log[k][steps] for k in ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm")}

    # 6. Frames for the animation
    frame_idx = select_frames(steps, {
        "train": np.log10(at["train_loss"]), "test": np.log10(at["test_loss"]),
        "norm": at["weight_norm"], "gini": spectrum_gini,
        "restricted": np.log10(restricted), "excluded": np.log10(excluded),
    }, n_frames)
    print(f"selected {len(frame_idx)} frames")

    out = run_dir / "analysis.npz"
    np.savez_compressed(
        out, steps=steps, key_freqs=np.array(key_freqs), spectrum=spectrum, gini=spectrum_gini,
        rings=rings, pca_fixed=pca_fixed, pca_aligned=pca_aligned, pca_explained=pca_explained,
        restricted_loss=restricted, excluded_loss=excluded, frame_idx=frame_idx,
        **{f"ckpt_{k}": v for k, v in at.items()},
    )
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=Config.run_name)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(Config(run_name=args.run_name).run_dir / "config.json")
    analyse(cfg, args.frames)

    if not args.no_plots:
        from . import plots
        plots.make_all(cfg.run_dir)
    if not args.no_export:
        from . import export
        export.export(cfg.run_dir)


if __name__ == "__main__":
    main()
