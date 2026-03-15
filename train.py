"""
Autoresearch NBA spread prediction script. Single-device, single-file.
Apple Silicon MLX — predicts NBA game point differential and winner.
Usage: uv run train.py
"""

import gc
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from prepare import TIME_BUDGET, DataLoader, evaluate, get_n_features

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIM = 640
NUM_LAYERS = 2
DROPOUT_RATE = 0.2

# Training
BATCH_SIZE = 512
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 1e-3
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.01

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NBAPredictor(nn.Module):
    """
    MLP for predicting NBA game point differential and winner.
    Outputs: (predicted_point_diff, win_logit)
    """

    def __init__(self, n_features, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = n_features
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            in_dim = hidden_dim
        self.layers = layers
        self.dropout = nn.Dropout(dropout_rate)

        # Separate heads for point diff regression and win classification
        self.diff_head = nn.Linear(hidden_dim, 1, bias=True)
        self.win_head = nn.Linear(hidden_dim, 1, bias=True)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.gelu(x)
            x = self.dropout(x)

        pred_diff = self.diff_head(x)
        win_logit = self.win_head(x)

        return pred_diff, win_logit


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def compute_loss(model, X, y_diff, y_win):
    """Combined MSE (point diff) + BCE (win) loss."""
    pred_diff, win_logit = model(X)

    # MSE loss for point differential prediction
    mse_loss = mx.mean((pred_diff.reshape(-1) - y_diff) ** 2)

    # Binary cross-entropy for win/loss (numerically stable)
    logit = win_logit.reshape(-1)
    bce_loss = mx.mean(
        mx.maximum(logit, mx.array(0.0))
        - logit * y_win
        + mx.log1p(mx.exp(-mx.abs(logit)))
    )

    # Combined loss — weight BCE higher since win prediction is primary goal
    total_loss = 0.3 * mse_loss + 0.7 * bce_loss

    return total_loss


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(7777)

# Load data
n_features = get_n_features()
train_loader = DataLoader(BATCH_SIZE, split="train")
t_data = time.time()
print(f"Data loaded in {t_data - t_start:.1f}s")
print(f"Features: {n_features}, Train samples: {train_loader.n_samples}")

# Create model
model = NBAPredictor(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE)
mx.eval(model.parameters())
num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = optim.AdamW(
    learning_rate=LEARNING_RATE,
    betas=list(ADAM_BETAS),
    weight_decay=WEIGHT_DECAY,
)

loss_grad_fn = nn.value_and_grad(model, compute_loss)

print(f"Time budget: {TIME_BUDGET}s")

# Training
total_training_time = 0.0
step = 0
epoch = 0
smooth_loss = 0.0
best_epoch_loss = float("inf")
t_compiled = None

while True:
    epoch += 1
    train_loader.shuffle()
    epoch_loss = 0.0
    epoch_batches = 0

    model.train()
    for X, y_diff, y_win in train_loader:
        t0 = time.time()

        loss, grads = loss_grad_fn(model, X, y_diff, y_win)
        mx.eval(loss, grads)

        if t_compiled is None:
            t_compiled = time.time()
            print(f"Compiled in {t_compiled - t_data:.1f}s")

        # LR schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        optimizer.learning_rate = LEARNING_RATE * lrm

        model.update(optimizer.apply_gradients(grads, model))
        mx.eval(model.parameters())

        dt = time.time() - t0
        total_training_time += dt

        loss_val = float(loss.item())
        epoch_loss += loss_val
        epoch_batches += 1

        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))

        step += 1

        if step % 50 == 0:
            pct = 100 * progress
            remaining = max(0, TIME_BUDGET - total_training_time)
            print(
                f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.6f} | "
                f"lr: {LEARNING_RATE * lrm:.6f} | dt: {dt*1000:.0f}ms | "
                f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
                end="",
                flush=True,
            )

        if total_training_time >= TIME_BUDGET:
            break

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    if total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - (t_compiled or t_data):.1f}s ({epoch} epochs, {step} steps)")

# Final evaluation
print("Starting final eval...")
model.eval()
results = evaluate(model, batch_size=256)

t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

peak_mem = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_score:            {results['val_score']:.6f}")
print(f"win_accuracy:         {results['win_accuracy']:.6f}")
print(f"mae_point_diff:       {results['mae_point_diff']:.6f}")
print(f"mean_loss:            {results['mean_loss']:.6f}")
print(f"training_seconds:     {total_training_time:.1f}")
print(f"total_seconds:        {t_eval - t_start:.1f}")
print(f"peak_memory_mb:       {peak_mem:.1f}")
print(f"num_steps:            {step}")
print(f"num_epochs:           {epoch}")
print(f"num_params:           {num_params:,}")
print(f"n_features:           {n_features}")
print(f"hidden_dim:           {HIDDEN_DIM}")
print(f"num_layers:           {NUM_LAYERS}")
