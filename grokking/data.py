"""The modular addition dataset.

Every pair (a, b) with 0 <= a, b < p becomes the token sequence [a, b, =],
where '=' is token id p, and the label is (a + b) mod p.

Rows are ordered so that row index i = a * p + b. That makes it easy to
reshape any per-example quantity into a p x p grid later on.
"""

import torch


def make_dataset(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens, labels) for all p^2 pairs.

    tokens: (p*p, 3) long tensor of [a, b, =]
    labels: (p*p,)   long tensor of (a + b) mod p
    """
    a = torch.arange(p).repeat_interleave(p)  # 0,0,...,0,1,1,...
    b = torch.arange(p).repeat(p)             # 0,1,...,p-1,0,1,...
    eq = torch.full_like(a, p)
    tokens = torch.stack([a, b, eq], dim=1)
    labels = (a + b) % p
    return tokens, labels


def train_test_split(n: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random split of row indices 0..n-1, reproducible from `seed`.

    A dedicated generator is used so the split doesn't depend on (or disturb)
    the global RNG that initialises the model.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(round(train_frac * n))
    return perm[:n_train].sort().values, perm[n_train:].sort().values
