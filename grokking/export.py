"""Export compact JSON for the web demo (docs/data/).

    python -m grokking.export [--run-name p113_seed0]

Three files, so the page can draw the curves before the heavier frame data
has arrived:
    meta.json    config, key frequencies, phases, grok step, colours, axis limits
    curves.json  per-step training curves, downsampled; per-checkpoint progress measures
    frames.json  per animation frame: spectrum, ring points, PCA points

Arrays are stored column-wise with flat [x0, y0, x1, y1, ...] point lists,
and floats are rounded to what the eye can see (3 decimals for coordinates
on a ~unit scale, 4 significant figures for losses). This keeps the
total to a few MB.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .analyse import grok_step, phase_boundaries
from .config import REPO_ROOT, Config
from .plots import number_colours

DATA_DIR = REPO_ROOT / "docs" / "data"


def sig(values, n: int = 4) -> list[float]:
    """Round to n significant figures. Via string formatting so the JSON gets
    short numbers like 0.0123, not 0.012300000000000001."""
    return [float(f"{v:.{n}g}") for v in np.asarray(values, dtype=float).ravel().tolist()]


def dec(values, d: int = 3) -> list[float]:
    """Round to d decimal places (Python's round gives the shortest repr)."""
    return [round(v, d) for v in np.asarray(values, dtype=float).ravel().tolist()]


def curve_indices(n_steps: int, n_points: int = 1500) -> np.ndarray:
    """Which per-step log entries to keep.

    The union of log-spaced indices (detail early on, for the log-x axis) and
    linearly spaced ones (detail late, for a linear axis), always including
    the first and last steps.
    """
    log_idx = np.round(np.logspace(0, np.log10(n_steps), n_points)).astype(int) - 1
    lin_idx = np.round(np.linspace(0, n_steps - 1, n_points)).astype(int)
    return np.unique(np.concatenate([[0, n_steps - 1], log_idx, lin_idx]))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1e6:.2f} MB)")


def export(run_dir: Path) -> None:
    cfg = Config.load(run_dir / "config.json")
    log = dict(np.load(run_dir / "log.npz"))
    a = dict(np.load(run_dir / "analysis.npz"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames = a["frame_idx"]
    key_freqs = a["key_freqs"].tolist()
    colours = ["#%02x%02x%02x" % tuple(np.round(c * 255).astype(int)) for c in number_colours(cfg.p)]

    # Fixed axis limits for the whole animation, so motion means motion and
    # not a rescaled axis. Ring limits are per key frequency and robust (97th
    # percentile of radius over all frames, at least 1.3): during memorisation
    # the embeddings inflate and a few points fly far out (k = 8 is noisiest),
    # and fitting those would shrink the finished unit ring to a small blob.
    # The page clips the rare outliers at the panel edge.
    ring_r = np.linalg.norm(a["rings"][frames], axis=-1)  # (frames, K, p)
    ring_limits = [round(max(1.3, float(np.percentile(ring_r[:, r], 97))) * 1.05, 2) for r in range(len(key_freqs))]
    pca_r = np.abs(a["pca_fixed"][frames]).max()
    pca_aligned_r = np.abs(a["pca_aligned"][frames]).max()

    meta = {
        "config": {k: v for k, v in vars(cfg).items()},
        "p": cfg.p,
        "key_freqs": key_freqs,
        "n_frames": len(frames),
        "final_step": int(log["step"][-1]),
        "grok_step": grok_step(log["test_acc"]),
        "phases": phase_boundaries(a, cfg.p),
        "pca_explained": dec(a["pca_explained"], 3),
        "colours": colours,
        "limits": {
            "ring": ring_limits,
            "pca": round(float(pca_r) * 1.05, 2),
            "pca_aligned": round(float(pca_aligned_r) * 1.05, 2),
            "spectrum": round(float(a["spectrum"][frames].max()) * 1.05, 2),
        },
        "final": {k: sig([log[k][-1]], 4)[0] for k in ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm")},
    }
    write_json(DATA_DIR / "meta.json", meta)

    idx = curve_indices(len(log["step"]))
    curves = {
        "step": log["step"][idx].tolist(),
        **{k: sig(log[k][idx], 4) for k in ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm")},
        # Progress measures exist only at checkpoints (they need a full analysis pass)
        "checkpoints": {
            "step": a["steps"].tolist(),
            "restricted_loss": sig(a["restricted_loss"], 4),
            "excluded_loss": sig(a["excluded_loss"], 4),
            "gini": dec(a["gini"], 4),
        },
    }
    write_json(DATA_DIR / "curves.json", curves)

    frame_data = {
        "step": a["steps"][frames].tolist(),
        "spectrum": [dec(a["spectrum"][i], 3) for i in frames],
        "rings": [[dec(a["rings"][i, r], 3) for r in range(len(key_freqs))] for i in frames],
        "pca": [dec(a["pca_fixed"][i], 3) for i in frames],
        "pca_aligned": [dec(a["pca_aligned"][i], 3) for i in frames],
    }
    write_json(DATA_DIR / "frames.json", frame_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=Config.run_name)
    args = parser.parse_args()
    export(Config(run_name=args.run_name).run_dir)


if __name__ == "__main__":
    main()
