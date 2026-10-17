"""All hyperparameters for a run, in one place.

Defaults follow Nanda et al. 2023, "Progress Measures for Grokking via
Mechanistic Interpretability". The config is saved next to the checkpoints
as config.json so every later stage knows exactly how a run was produced.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class Config:
    # --- Task -------------------------------------------------------------
    p: int = 113                 # modulus (prime); inputs a, b are in 0..p-1
    train_frac: float = 0.3      # fraction of all p^2 pairs used for training

    # --- Model ------------------------------------------------------------
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32             # n_heads * d_head == d_model
    d_mlp: int = 512
    n_ctx: int = 3               # sequence is [a, b, =]

    # --- Optimisation -----------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1.0    # unusually large; this is what drives grokking
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 10
    max_steps: int = 40_000

    # Stop once test accuracy has stayed at or above `perfect_acc` for
    # `stop_after_perfect` consecutive steps. This keeps some of the
    # post-grokking "cleanup" phase in the recording without wasting time.
    perfect_acc: float = 0.999
    stop_after_perfect: int = 5_000

    # --- Bookkeeping ------------------------------------------------------
    seed: int = 0
    run_name: str = "p113_seed0"

    @property
    def d_vocab(self) -> int:
        """Input vocabulary: the p numbers plus one '=' token (id p)."""
        return self.p + 1

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))
