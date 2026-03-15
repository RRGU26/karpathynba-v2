# autoresearch-nba

Autonomous NBA game spread prediction research — adapted from Karpathy's autoresearch framework (MLX). Predicts point differential and win/loss for NBA games, designed to find value against Vegas spreads.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — this file, repository context.
   - `prepare.py` — fixed constants, data download, feature engineering, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-nba/` contains `features.pkl`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run `uv run train.py` once to establish YOUR baseline on this hardware.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The Problem

Predict NBA game outcomes from the home team's perspective:
- **Point differential**: Home team points minus away team points (regression)
- **Win/loss**: Did the home team win? (classification)

The model's predicted point differential is compared against Vegas spreads at inference time to find value bets. A model that predicts "home team by 8" when Vegas says "home team by 3" suggests betting the home team to cover.

Input features include:
- Per-team rolling stats (5/10/20 game windows): points, efficiency (eFG%, TS%), turnover rate, pace, rebounding, assists, steals, blocks, win rate, plus/minus
- Elo ratings (with margin-of-victory adjustment, home advantage, season regression)
- Rest days, back-to-back detection, rest advantage
- Calendar features (day of week, month, season progress)
- Differential features (home stat - away stat) for key metrics

All features are z-score normalized using training set statistics.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, loss function weights, regularization, etc.
- Try different architectures: MLP, residual networks, attention layers, mixture of experts, ensemble approaches
- Try different loss functions or loss weightings between point diff and win prediction
- Add feature interactions, learned feature selection, or feature gating
- Experiment with regularization: dropout, weight decay, batch norm, layer norm
- Try different optimizers: Adam, AdamW, SGD with momentum, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, feature engineering, and constants.
- Install new packages or add dependencies. You can only use what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_score.**

`val_score = (1 - win_accuracy) + 0.05 * mae_point_diff`

Lower val_score = better. This combines win prediction accuracy (most important) with point differential prediction quality. A model with 60% win accuracy and 9.0 MAE scores: `(1-0.60) + 0.05*9.0 = 0.40 + 0.45 = 0.85`. A model with 65% accuracy and 8.5 MAE scores: `0.35 + 0.425 = 0.775`.

**Important context for NBA prediction:**
- This is a HARD problem. Home teams win ~58% of games, so always-predict-home is a non-trivial baseline.
- Overfitting is the primary enemy. The model will easily memorize training patterns.
- NBA games have high variance — close games (< 5 point diff) are essentially coin flips.
- The signal is in the "easy" games — large predicted margins.
- Blowouts (20+ points) are noisy outliers. Consider robust loss functions (Huber/MAE instead of MSE).
- Regularization, early stopping logic, and model simplicity often beat complexity.
- Elo difference is historically the single strongest feature. The model should learn to weight it heavily.

## Output format

Once the script finishes it prints:

```
---
val_score:            0.850000
win_accuracy:         0.600000
mae_point_diff:       9.000000
mean_loss:            0.680000
training_seconds:     300.0
total_seconds:        305.0
peak_memory_mb:       512.0
num_steps:            5000
num_epochs:           15
num_params:           50,000
n_features:           90
hidden_dim:           128
num_layers:           3
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and 6 columns:

```
commit	val_score	win_acc	mae	status	description
```

1. git commit hash (short, 7 chars)
2. val_score (e.g. 0.850000) — use 9.999999 for crashes
3. win_acc (e.g. 0.6000) — use 0.0000 for crashes
4. mae (e.g. 9.0000) — use 0.0000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description

Example:

```
commit	val_score	win_acc	mae	status	description
a1b2c3d	0.850000	0.6000	9.0000	keep	baseline MLP
e4f5g6h	0.810000	0.6300	8.6000	keep	add residual connections
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea
3. `git add train.py && git commit -m "experiment: <description>"`
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_score:\|^win_accuracy:\|^mae_point_diff:" run.log`
6. If empty, run crashed. `tail -n 50 run.log` for stack trace. Fix or skip.
7. Record in results.tsv
8. If val_score improved (lower), `git add results.tsv && git commit --amend --no-edit`
9. If val_score is equal or worse, record discard, then `git reset --hard <previous kept commit>`

## Research directions to explore

Here are promising directions, roughly ordered by expected impact:

1. **Regularization tuning**: Dropout rates, weight decay, batch size effects — overfitting is enemy #1
2. **Loss function**: Huber loss for point diff (robust to blowouts), focal loss for close games, different BCE/MSE weightings
3. **Architecture search**: Residual connections, skip connections, bottleneck layers
4. **Feature gating**: Learned attention over input features — many features may be noise, the model should learn which matter
5. **Point diff → direction coupling**: Use predicted point diff sign as auxiliary signal for the win head
6. **Ensemble heads**: Multiple prediction pathways with different architectures, averaged
7. **Learning rate schedules**: Cosine annealing, cyclic LR, different warmup strategies
8. **Label smoothing**: Soften win labels for close games (e.g. if actual point diff was 1, label = 0.55 instead of 1.0)
9. **Gradient clipping**: Prevent exploding gradients from blowout games
10. **Activation functions**: SiLU/Swish, Mish, LeakyReLU instead of GELU
11. **Normalization**: LayerNorm, BatchNorm placement and combinations

**NEVER STOP**: Once the loop begins, do NOT pause to ask. The human might be asleep. You are autonomous. If you run out of ideas, re-read the code, try combining previous wins, try more radical changes. Loop until manually stopped.
