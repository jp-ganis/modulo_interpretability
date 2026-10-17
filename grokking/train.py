"""Train the transformer on (a + b) mod p and record everything.

    python -m grokking.train                    # default run (runs/p113_seed0)
    python -m grokking.train --max-steps 200    # quick smoke test

Outputs in runs/<run_name>/:
    config.json      hyperparameters
    split.pt         train/test row indices (row i = a*p + b)
    log.npz          per-step train/test loss & accuracy, weight norm
    checkpoints/     step_XXXXXX.pt state dicts
"""

import argparse
import time

import numpy as np
import torch

from .config import Config
from .data import make_dataset, train_test_split
from .model import Transformer


def cross_entropy_f64(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy computed in float64.

    Late in training the train loss reaches ~1e-7. In float32 the correct
    class's log-probability, log(1 - tiny), rounds to exactly 0, and the
    gradients become dominated by rounding error. This causes sudden loss
    spikes (the "slingshot" effect). Nanda et al. hit this and fixed it the
    same way: do only the log-softmax in double precision. It's cheap because
    the logits are only (batch, p).
    """
    log_probs = logits.to(torch.float64).log_softmax(dim=-1)
    return -log_probs.gather(1, labels[:, None]).mean()


def checkpoint_schedule(max_steps: int) -> set[int]:
    """Steps at which to save a checkpoint.

    Memorisation happens fast (within the first few hundred steps), so the
    first 1000 steps are sampled log-spaced. After that we save every 25
    steps: the grokking transition itself can take under 1000 steps, and we
    don't know in advance when it will happen. Checkpoints are ~1 MB each, so
    this is cheap. Analysis later picks the frames to show, packing them more
    densely where things are changing.
    """
    early = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    late = np.arange(1000, max_steps + 1, 25)
    return {0, *early.tolist(), *late.tolist()}


@torch.no_grad()
def weight_norm(model: torch.nn.Module) -> float:
    """Total L2 norm of all parameters, sqrt(sum of squares)."""
    return torch.sqrt(sum((p.double() ** 2).sum() for p in model.parameters())).item()


def train(cfg: Config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for old in ckpt_dir.glob("step_*.pt"):  # a fresh run shouldn't mix with stale files
        old.unlink()
    cfg.save(run_dir / "config.json")

    # --- Data -------------------------------------------------------------
    tokens, labels = make_dataset(cfg.p)
    train_idx, test_idx = train_test_split(len(labels), cfg.train_frac, cfg.seed)
    torch.save({"train_idx": train_idx, "test_idx": test_idx}, run_dir / "split.pt")
    x_train, y_train = tokens[train_idx].to(device), labels[train_idx].to(device)
    x_test, y_test = tokens[test_idx].to(device), labels[test_idx].to(device)
    print(f"device={device}  train={len(train_idx)}  test={len(test_idx)}")

    # --- Model & optimiser ------------------------------------------------
    torch.manual_seed(cfg.seed)
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    # AdamW applies decoupled weight decay (theta <- theta - lr*wd*theta every
    # step) to *all* parameters, embeddings included. With wd = 1.0 this is a
    # strong, constant pull towards small weights; the generalising Fourier
    # solution is far "cheaper" in weight norm than a memorising lookup table,
    # so weight decay slowly favours it.
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    # Linear warmup: LR ramps from lr/warmup to lr over the first steps, which
    # avoids huge Adam updates while its moment estimates are still noisy.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / cfg.warmup_steps, 1.0))

    # --- Training loop ----------------------------------------------------
    save_at = checkpoint_schedule(cfg.max_steps)
    log = {k: [] for k in ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm")}
    perfect_streak = 0
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        # Convention: log[step] and step_{step}.pt both describe the weights
        # *after* `step` updates, measured before applying update step+1.
        # Step 0 is the random initialisation.
        logits = model(x_train)
        train_loss = cross_entropy_f64(logits, y_train)

        with torch.no_grad():
            test_logits = model(x_test)
            test_loss = cross_entropy_f64(test_logits, y_test).item()
            train_acc = (logits.argmax(-1) == y_train).float().mean().item()
            test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()

        log["train_loss"].append(train_loss.item())
        log["test_loss"].append(test_loss)
        log["train_acc"].append(train_acc)
        log["test_acc"].append(test_acc)
        log["weight_norm"].append(weight_norm(model))

        perfect_streak = perfect_streak + 1 if test_acc >= cfg.perfect_acc else 0
        done = step == cfg.max_steps or perfect_streak >= cfg.stop_after_perfect

        if step in save_at or done:
            torch.save(model.state_dict(), ckpt_dir / f"step_{step:06d}.pt")

        if step % 1000 == 0 or done:
            print(f"step {step:6d} | train {log['train_loss'][-1]:.2e} ({train_acc:6.1%}) | "
                  f"test {test_loss:.2e} ({test_acc:6.1%}) | |W| {log['weight_norm'][-1]:6.2f} | "
                  f"{time.time() - t0:6.0f}s", flush=True)
        if done:
            break

        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()
        sched.step()

    np.savez(run_dir / "log.npz", step=np.arange(len(log["train_loss"])),
             **{k: np.array(v) for k, v in log.items()})
    n_ckpts = len(list(ckpt_dir.glob("step_*.pt")))
    print(f"finished at step {step} in {time.time() - t0:.0f}s; {n_ckpts} checkpoints in {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--train-frac", type=float)
    args = parser.parse_args()

    cfg = Config()
    for field, value in vars(args).items():
        if value is not None:
            setattr(cfg, field, value)
    train(cfg)


if __name__ == "__main__":
    main()
