"""Nanda et al.'s progress measures: restricted loss and excluded loss.

The trained model computes (a + b) mod p roughly like this:

    logit(c) ~= sum over key frequencies w_k of  alpha_k * cos(w_k * (a + b - c))

where w_k = 2*pi*k/p. That is largest exactly when c = a + b (mod p), because
every cosine is then cos(0) = 1, so the correct answer wins.

The trig identity
    cos(w(a+b-c)) = cos(w(a+b)) * cos(wc) + sin(w(a+b)) * sin(wc)
says this logit pattern is built from two "input-side" waves, cos(w(a+b)) and
sin(w(a+b)), each a function of the input pair (a, b), multiplied by some
function of the output c.

So for each key frequency we take the logits on all p^2 inputs and split them
into:
  - the part that varies across inputs like cos(w(a+b)) or sin(w(a+b))
    (the "Fourier circuit"), and
  - everything else.

Two measures follow:
  restricted loss: keep only the circuit part and compute the loss. If this
                   is low, the Fourier algorithm alone can solve the task.
  excluded loss:   remove the circuit part, keep the rest, and compute the
                   loss on the *training* data. That isolates how much the
                   model still relies on memorisation.

During circuit formation, restricted loss falls (the algorithm is being built)
and excluded loss rises (memorisation alone stops being enough). Both happen
while the test loss still looks flat, which is the point of the paper: you can
see grokking coming.
"""

import numpy as np


def apb_waves(p: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm cos(w(a+b)) and sin(w(a+b)) as vectors over all p^2 inputs.

    Inputs are ordered row = a*p + b, matching data.make_dataset.

    These vectors are orthogonal across different k (and cos vs sin): each
    sum s = a + b mod p is hit by exactly p pairs, so their inner products
    reduce to the inner products of the 1-D Fourier basis over Z_p, which is
    orthogonal. That means the projections for each key frequency can be done
    independently and simply added up.
    """
    a = np.arange(p).repeat(p)
    b = np.tile(np.arange(p), p)
    angle = 2 * np.pi * k * (a + b) / p
    cos_v, sin_v = np.cos(angle), np.sin(angle)
    return cos_v / np.linalg.norm(cos_v), sin_v / np.linalg.norm(sin_v)


def circuit_component(logits: np.ndarray, p: int, key_freqs: list[int]) -> np.ndarray:
    """The part of (p^2, p) logits that lives in the key frequencies' cos/sin(w(a+b)) directions.

    For each output c, the column logits[:, c] is a vector over all inputs. We
    project it onto each unit wave u: (u . logits[:, c]) * u. Done for all c at
    once, this is u (u^T L), a rank-1 piece of the logit matrix.
    """
    part = np.zeros_like(logits)
    for k in key_freqs:
        for u in apb_waves(p, k):
            part += np.outer(u, u @ logits)
    return part


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy in float64 (with the usual max-subtraction for stability)."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def progress_measures(logits: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                      p: int, key_freqs: list[int]) -> tuple[float, float]:
    """Return (restricted_loss on all data, excluded_loss on train data).

    Softmax doesn't care about adding a constant to every logit of one
    example, so we first subtract each example's mean logit. Otherwise that
    irrelevant offset would leak into the projections.
    """
    L = logits - logits.mean(axis=1, keepdims=True)
    circuit = circuit_component(L, p, key_freqs)
    restricted = cross_entropy(circuit, labels)
    excluded = cross_entropy((L - circuit)[train_idx], labels[train_idx])
    return restricted, excluded
