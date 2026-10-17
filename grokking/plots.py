"""Static Matplotlib figures for the README.

    python -m grokking.plots [--run-name p113_seed0]

Writes PNGs to figures/. Needs log.npz (from training) and analysis.npz
(from grokking.analyse).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.pyplot as plt
import numpy as np

from .analyse import grok_step, phase_boundaries
from .config import REPO_ROOT, Config

FIG_DIR = REPO_ROOT / "figures"

# Colours by role (validated categorical slots blue / orange / aqua), shared
# with the web demo so the README and the page read as one system.
TRAIN = "#2a78d6"
TEST = "#eb6834"
THIRD = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b9b8b2"
GRID = "#e4e3df"
PHASE_FILL = ["#f3f2ee", "#ffffff", "#f3f2ee"]
PHASE_NAMES = ["memorisation", "circuit formation", "cleanup"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "legend.frameon": False,
})


# --- Colour for numbers 0..p-1 ---------------------------------------------

def number_colours(p: int, lightness: float = 62.0, chroma: float = 42.0) -> np.ndarray:
    """A cyclic colour per number: hue goes once around the wheel as a goes 0 -> p-1.

    The numbers live on a circle (p-1 is next to 0), so the colour map should
    wrap around too. Matplotlib's built-in cyclic maps either fade to near
    white at the ends ('twilight') or are perceptually lumpy ('hsv'). Instead
    we walk the hue angle of CIE LCh at fixed lightness and chroma: every
    number is equally visible on a light or dark background, and equal steps
    in a look like equal steps in colour.

    Conversion chain: LCh -> Lab -> XYZ (D65) -> linear sRGB -> gamma-encoded sRGB.
    Returns (p, 3) RGB in [0, 1].
    """
    h = 2 * np.pi * np.arange(p) / p
    L = np.full(p, lightness)
    A, Bc = chroma * np.cos(h), chroma * np.sin(h)
    # Lab -> XYZ
    fy = (L + 16) / 116
    fx, fz = fy + A / 500, fy - Bc / 200
    eps = 6 / 29
    finv = lambda t: np.where(t > eps, t ** 3, 3 * eps ** 2 * (t - 4 / 29))
    xyz = np.stack([0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)], axis=1)
    # XYZ -> linear sRGB -> sRGB
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    lin = np.clip(xyz @ M.T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


# --- Helpers ---------------------------------------------------------------

def load(run_dir: Path) -> tuple[dict, dict, Config]:
    log = dict(np.load(run_dir / "log.npz"))
    a = dict(np.load(run_dir / "analysis.npz"))
    return log, a, Config.load(run_dir / "config.json")


def shade_phases(ax, phases: dict[str, int], x0: float, x1: float, label: bool = False) -> None:
    """Background bands for the three training phases (drawn behind everything)."""
    edges = [x0, phases["memorisation_end"], phases["cleanup_start"], phases["cleanup_end"]]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(max(lo, x0), hi, color=PHASE_FILL[i], zorder=0, lw=0)
        if label:
            # Centre each label in its band, except the last: cleanup is a
            # narrow band on a log axis, so its label starts at the band's
            # left edge and runs to the right instead of colliding.
            if i == 2:
                x, ha = lo, "left"
            else:
                x, ha = (np.sqrt(max(lo, 1) * hi) if ax.get_xscale() == "log" else (lo + hi) / 2), "center"
            ax.text(x, 1.01, PHASE_NAMES[i], transform=ax.get_xaxis_transform(),
                    ha=ha, va="bottom", fontsize=8.5, color=INK_2)
    for e in edges[1:]:
        ax.axvline(e, color=MUTED, lw=0.8, zorder=0.5)


def snapshot_indices(a: dict, phases: dict[str, int], grok: int) -> list[tuple[str, int]]:
    """Four moments to show side by side: end of memorisation, mid circuit formation,
    the grokking moment, and the final checkpoint (checkpoint indices into a)."""
    steps = a["steps"]
    mid = (phases["memorisation_end"] + phases["cleanup_start"]) // 2
    picks = [("memorised", phases["memorisation_end"]), ("circuit forming", mid),
             ("grokking", grok), ("final", int(steps[-1]))]
    return [(f"{name}\nstep {steps[i]:,}", i)
            for name, s in picks for i in [int(np.abs(steps - s).argmin())]]


# --- Figures ---------------------------------------------------------------

def plot_training(log: dict, phases: dict, out: Path) -> None:
    """Loss, accuracy and weight norm on a shared log-step axis."""
    step = log["step"] + 1  # +1 so step 0 can sit on a log axis
    g = grok_step(log["test_acc"])

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    ax_loss, ax_acc, ax_norm = axes
    ax_norm.set_xscale("log")
    for i, ax in enumerate(axes):
        shade_phases(ax, phases, 1, step[-1], label=(i == 0))

    ax_loss.plot(step, log["train_loss"], color=TRAIN, label="train")
    ax_loss.plot(step, log["test_loss"], color=TEST, label="test")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Train loss collapses early; test loss waits, then drops", pad=18)
    ax_loss.legend(loc="lower left")

    ax_acc.plot(step, log["train_acc"], color=TRAIN)
    ax_acc.plot(step, log["test_acc"], color=TEST)
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.annotate(f"test acc ≥ 95%\nstep {g:,}", xy=(g + 1, 0.95), xytext=(-8, -2),
                    textcoords="offset points", ha="right", va="top", fontsize=9, color=INK_2)
    ax_acc.plot([g + 1], [0.95], "o", color=TEST, ms=5, mec="white", mew=1.5)

    ax_norm.plot(step, log["weight_norm"], color=INK_2)
    ax_norm.set_ylabel("total weight norm")
    ax_norm.set_xlabel("training step (log scale)")
    ax_norm.set_xlim(1, step[-1] * 1.05)

    fig.align_ylabels(axes)
    fig.savefig(out)
    plt.close(fig)


def plot_spectrum(a: dict, snaps: list, key_freqs: list[int], out: Path) -> None:
    """Fourier spectrum of W_E at four moments: noise -> a few sharp spikes."""
    spec = a["spectrum"]
    ks = np.arange(1, spec.shape[1] + 1)
    is_key = np.isin(ks, key_freqs)
    fig, axes = plt.subplots(len(snaps), 1, figsize=(7.5, 6.5), sharex=True, sharey=True)
    for ax, (title, i) in zip(axes, snaps):
        ax.bar(ks, spec[i], width=0.75, color=np.where(is_key, TRAIN, MUTED), lw=0)
        ax.text(0.99, 0.92, title.replace("\n", " · "), transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=INK_2)
        ax.grid(axis="x", visible=False)
    axes[0].set_title(f"Fourier spectrum of the embedding (key frequencies {key_freqs} in blue)")
    axes[-1].set_xlabel("frequency k")
    fig.supylabel("strength  ‖projection of W_E‖", color=INK_2, fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def plot_rings(a: dict, snaps: list, key_freqs: list[int], p: int, out: Path) -> None:
    """Each number's embedding projected onto each key frequency's (fixed, final) cos/sin plane."""
    colours = number_colours(p)
    rings = a["rings"]
    n_r, n_c = len(key_freqs), len(snaps)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.3 * n_c, 2.3 * n_r))
    lim = 1.6
    for r, k in enumerate(key_freqs):
        for c, (title, i) in enumerate(snaps):
            ax = axes[r, c]
            xy = rings[i, r]
            ax.scatter(xy[:, 0], xy[:, 1], s=9, c=colours, lw=0)
            if c == n_c - 1:  # label 0, 1, 2 to show the jump of k positions per +1
                for n in range(3):
                    ax.annotate(str(n), xy[n], xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRID, lw=0.8, zorder=0))
            if r == 0:
                ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK_2)
            if c == 0:
                ax.set_ylabel(f"k = {k}", fontsize=10, color=INK_2)
    fig.suptitle("Rings emerge: numbers projected onto each key frequency's cos/sin plane", y=1.0)
    fig.savefig(out)
    plt.close(fig)


def plot_pca(a: dict, snaps: list, p: int, out: Path) -> None:
    """PCA of W_E in the final checkpoint's (fixed) top-2 basis."""
    colours = number_colours(p)
    pts = a["pca_fixed"]
    lim = np.abs(pts[[i for _, i in snaps]]).max() * 1.05
    fig, axes = plt.subplots(1, len(snaps), figsize=(2.3 * len(snaps), 2.6))
    for ax, (title, i) in zip(axes, snaps):
        ax.scatter(pts[i, :, 0], pts[i, :, 1], s=9, c=colours, lw=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK_2)
    ev = a["pca_explained"]
    fig.suptitle(f"PCA of the embedding (final basis: PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%} of variance)", y=1.04)
    fig.savefig(out)
    plt.close(fig)


def plot_progress(a: dict, phases: dict, p: int, out: Path) -> None:
    """Nanda et al.'s progress measures alongside train/test loss (linear step axis)."""
    s = a["steps"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    shade_phases(ax, phases, 0, s[-1], label=True)
    ax.plot(s, a["ckpt_train_loss"], color=TRAIN, label="train loss")
    ax.plot(s, a["ckpt_test_loss"], color=TEST, label="test loss")
    ax.plot(s, a["restricted_loss"], color=THIRD, label="restricted loss (Fourier circuit only)")
    ax.plot(s, a["excluded_loss"], color=INK_2, ls="--", lw=1.6, label="excluded loss (circuit removed, train)")
    ax.axhline(np.log(p), color=MUTED, lw=0.8)
    ax.text(s[-1], np.log(p), " ln p (guessing)", va="center", ha="left", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlim(0, s[-1])
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Progress measures reveal the circuit forming before test loss drops", pad=18)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(out)
    plt.close(fig)


def make_all(run_dir: Path) -> None:
    log, a, cfg = load(run_dir)
    phases = phase_boundaries(a, cfg.p)
    key_freqs = a["key_freqs"].tolist()
    snaps = snapshot_indices(a, phases, grok_step(log["test_acc"]))
    FIG_DIR.mkdir(exist_ok=True)

    plot_training(log, phases, FIG_DIR / "training_curves.png")
    plot_spectrum(a, snaps, key_freqs, FIG_DIR / "fourier_spectrum.png")
    plot_rings(a, snaps, key_freqs, cfg.p, FIG_DIR / "rings.png")
    plot_pca(a, snaps, cfg.p, FIG_DIR / "pca.png")
    plot_progress(a, phases, cfg.p, FIG_DIR / "progress_measures.png")
    print(f"wrote figures to {FIG_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=Config.run_name)
    args = parser.parse_args()
    make_all(Config(run_name=args.run_name).run_dir)


if __name__ == "__main__":
    main()
