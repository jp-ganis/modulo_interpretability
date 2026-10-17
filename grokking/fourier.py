"""The discrete Fourier basis over the integers mod p.

A function on Z_p, such as "the value of embedding dimension j for number a",
is just a vector of length p. We can re-express that vector in a basis of
waves around the circle:

    constant:   1                       (k = 0)
    cos_k(a) =  cos(2*pi*k*a / p)       for k = 1 .. p//2
    sin_k(a) =  sin(2*pi*k*a / p)       for k = 1 .. p//2

Because p = 113 is odd, that's 1 + 2*56 = 113 vectors: exactly p, so this is
a complete change of basis (there's no Nyquist frequency k = p/2, which only
exists for even p). Frequencies above p//2 add nothing new, because
cos(2*pi*(p-k)*a/p) = cos(2*pi*k*a/p), so they're aliases of lower ones.

We normalise each vector to unit length. Over a full period, sum_a cos^2 = p/2
for k != 0, so the cos/sin vectors are scaled by sqrt(2/p) and the constant by
sqrt(1/p). The result is an orthonormal p x p matrix B (B @ B.T = I), which
means projecting onto it preserves lengths: "energy" (sum of squares) in the
Fourier basis equals energy in the original basis.
"""

import numpy as np


def fourier_basis(p: int) -> tuple[np.ndarray, list[str]]:
    """Return (B, names). B is (p, p); row i is the i-th basis wave over a = 0..p-1.

    Row order: [const, cos_1, sin_1, cos_2, sin_2, ..., cos_{p//2}, sin_{p//2}]
    so frequency k's cos/sin rows are 2k-1 and 2k.
    """
    a = np.arange(p)
    rows, names = [np.ones(p)], ["const"]
    for k in range(1, p // 2 + 1):
        rows += [np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)]
        names += [f"cos {k}", f"sin {k}"]
    B = np.stack(rows)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B, names


def cos_sin_rows(B: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The unit-norm cos_k and sin_k vectors over a = 0..p-1."""
    return B[2 * k - 1], B[2 * k]


def frequency_strengths(W: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """How strongly a (p, d) matrix, e.g. W_E's number rows, uses each frequency.

    F = B @ W rewrites every column of W (one function of a per embedding
    dimension) in the Fourier basis. Row 2k-1 of F is how much each dimension
    looks like cos_k; row 2k is how much it looks like sin_k. A wave at
    frequency k with an arbitrary phase is a mix of cos_k and sin_k, so we
    combine both into one number:

        strength_k = sqrt(||F[cos_k]||^2 + ||F[sin_k]||^2)

    Returns (strengths for k = 1..p//2, strength of the constant component).
    """
    F = B @ W
    sq = (F ** 2).sum(axis=1)                    # energy per basis vector
    per_freq = sq[1::2] + sq[2::2]               # cos_k + sin_k
    return np.sqrt(per_freq), float(np.sqrt(sq[0]))


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative values: 0 = perfectly even, ->1 = all mass in one entry.

    Used as a single-number summary of how sparse the Fourier spectrum is.
    Formula (values sorted ascending, 1-indexed i):
        G = sum_i (2i - n - 1) * x_i / (n * sum_i x_i)
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * x).sum() / (n * x.sum()))
