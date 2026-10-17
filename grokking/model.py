"""A one-layer, attention + MLP transformer with no LayerNorm.

Written out by hand (rather than with nn.TransformerEncoderLayer or
TransformerLens) so every weight matrix is visible and named the way the
interpretability literature names it:

    W_E   (d_vocab, d_model)          token embedding      <- rows are numbers 0..p-1, then '='
    W_pos (n_ctx, d_model)            learned positional embedding
    W_Q, W_K, W_V (n_heads, d_model, d_head)
    W_O   (n_heads, d_head, d_model)
    W_in  (d_model, d_mlp), b_in      MLP input
    W_out (d_mlp, d_model), b_out     MLP output
    W_U   (d_model, p)                unembedding (untied from W_E)

Data flow for one sequence [a, b, =]:

    resid = W_E[tokens] + W_pos                 # one vector per position
    final = resid[=] + Attention(resid)         # '=' position reads from a, b and itself
    final = final + MLP(final)
    logits = final @ W_U

Why no LayerNorm? It isn't needed for such a shallow network, and leaving it
out keeps the maths linear between the residual stream and the logits, which
makes the Fourier analysis much cleaner (Nanda et al. made the same choice).
"""

import math

import torch
import torch.nn as nn

from .config import Config


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, h, dh, dm = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp

        # Initialisation: plain Gaussians scaled by 1/sqrt(fan_in), as in
        # Nanda et al.'s original code. With fan_in = d_model this keeps each
        # matrix product's output at roughly unit variance at the start of
        # training, so nothing explodes or vanishes in the residual stream.
        def init(*shape, fan_in):
            return nn.Parameter(torch.randn(*shape) / math.sqrt(fan_in))

        self.W_E = init(cfg.d_vocab, d, fan_in=d)
        self.W_pos = init(cfg.n_ctx, d, fan_in=d)

        self.W_Q = init(h, d, dh, fan_in=d)
        self.W_K = init(h, d, dh, fan_in=d)
        self.W_V = init(h, d, dh, fan_in=d)
        self.W_O = init(h, dh, d, fan_in=d)

        self.W_in = init(d, dm, fan_in=d)
        self.b_in = nn.Parameter(torch.zeros(dm))
        self.W_out = init(dm, d, fan_in=d)
        self.b_out = nn.Parameter(torch.zeros(d))

        # The unembedding is scaled by the output size (p) instead, again
        # following the original code; it keeps the initial logits small so
        # the starting loss is close to ln(p), i.e. a uniform guess.
        self.W_U = init(d, cfg.p, fan_in=cfg.p)

    # Shortcut: compute only what the final position needs.
    # In a one-layer model the logits depend only on the residual stream at
    # the final ('=') position. Attention outputs and MLP activations at
    # positions 0 and 1 are never read by anything. So we compute a query only
    # for the final position (it attends to all three positions, which is
    # exactly what a causal mask allows there), and run the MLP on that one
    # vector. The logits and gradients are identical to the full computation,
    # and it's ~3x cheaper, which matters when training on a CPU.

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head attention from the final position.

        x: (batch, pos, d_model) -> (batch, d_model), the attention output
        written into the final position's residual stream.
        """
        q = torch.einsum("bd,hde->bhe", x[:, -1], self.W_Q)       # (b, head, d_head)
        k = torch.einsum("bpd,hde->bhpe", x, self.W_K)            # (b, head, pos, d_head)
        v = torch.einsum("bpd,hde->bhpe", x, self.W_V)

        scores = torch.einsum("bhe,bhpe->bhp", q, k) / math.sqrt(self.cfg.d_head)
        pattern = scores.softmax(dim=-1)                           # (b, head, pos)

        z = torch.einsum("bhp,bhpe->bhe", pattern, v)              # (b, head, d_head)
        return torch.einsum("bhe,hed->bd", z, self.W_O)            # sum over heads

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_in + self.b_in) @ self.W_out + self.b_out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, 3) -> logits over the answer: (batch, p)."""
        resid = self.W_E[tokens] + self.W_pos          # (b, pos, d_model)
        final = resid[:, -1] + self.attention(resid)   # (b, d_model), the '=' position
        final = final + self.mlp(final)
        return final @ self.W_U
