"""Generate Karpathy-style autoresearch progress chart for NBA v1 + v2."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Data ---
v1 = [
    (1.031196, "keep",    "baseline MLP 3x128"),
    (1.079540, "discard", ""),
    (1.068565, "discard", ""),
    (1.097760, "discard", ""),
    (1.105675, "discard", ""),
    (0.991950, "keep",    "bigger 4x256"),
    (1.013256, "discard", ""),
    (1.042908, "discard", ""),
    (0.981307, "keep",    "5x512 do=0.4"),
    (0.980993, "keep",    "do=0.3"),
    (0.952708, "keep",    "3 layers"),
    (0.944590, "keep",    "warmdown=0.1"),
    (0.941972, "keep",    "2L do=0.18"),
    (0.937489, "keep",    "2L do=0.2"),
    (0.937072, "keep",    "hidden=544"),
    (0.930188, "keep",    "lr=1.2e-3"),
    (0.927542, "keep",    "hidden=640"),
]

v2 = [
    (0.967932, "keep",    "v1 config, 20 seasons"),
    (0.966266, "keep",    "batch=256"),
    (0.956844, "keep",    "batch=512"),
    (0.948677, "keep",    "wd=5e-4"),
    (0.945871, "keep",    "skip connection"),
    (0.937480, "keep",    "skip+do=0.25"),
    (0.924473, "keep",    "skip+do=0.3"),
    (0.923884, "keep",    "skip+lr=1e-3"),
    (0.935490, "discard", ""),
    (0.929498, "keep",    "player imputation"),
    (0.937000, "discard", ""),
    (0.934000, "keep",    "96f+Vegas spread"),
]

# --- Build arrays ---
all_scores, all_status, all_labels, all_phase = [], [], [], []
for score, status, label in v1:
    all_scores.append(score); all_status.append(status)
    all_labels.append(label); all_phase.append("v1")
for score, status, label in v2:
    all_scores.append(score); all_status.append(status)
    all_labels.append(label); all_phase.append("v2")

n = len(all_scores)
n_v1 = len(v1)
n_kept = sum(1 for s in all_status if s == "keep")

# --- Compute running best ---
kept_xs, kept_scores = [], []
running_best = float('inf')
for i in range(n):
    if all_status[i] == "keep" and all_scores[i] < running_best:
        running_best = all_scores[i]
        kept_xs.append(i)
        kept_scores.append(running_best)
kept_xs.append(n - 1)
kept_scores.append(kept_scores[-1])

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 7))

# Discarded (gray)
disc_x = [i for i in range(n) if all_status[i] == "discard"]
disc_y = [all_scores[i] for i in disc_x]
ax.scatter(disc_x, disc_y, c="#cccccc", s=50, zorder=2)

# Kept points
for i in range(n):
    if all_status[i] == "keep":
        c = "#2ecc71" if all_phase[i] == "v1" else "#3498db"
        ax.scatter(i, all_scores[i], c=c, s=70, zorder=3, edgecolors="white", linewidths=0.5)

# Running best step line — green for v1 portion, blue for v2
v1_kept_xs = [x for x in kept_xs if x < n_v1]
v1_kept_scores = [kept_scores[kept_xs.index(x)] for x in v1_kept_xs]
# Extend v1 line to the boundary
v1_kept_xs.append(n_v1 - 0.5)
v1_kept_scores.append(v1_kept_scores[-1])
ax.step(v1_kept_xs, v1_kept_scores, where="post", color="#2ecc71", linewidth=2, alpha=0.7, zorder=1)

# v2 line starts from v1 best
last_v1_best = v1_kept_scores[-1]
v2_kept_xs_raw = [x for x in kept_xs if x >= n_v1]
v2_kept_scores_raw = [kept_scores[kept_xs.index(x)] for x in v2_kept_xs_raw]
v2_line_xs = [n_v1 - 0.5] + v2_kept_xs_raw
v2_line_scores = [last_v1_best] + v2_kept_scores_raw
ax.step(v2_line_xs, v2_line_scores, where="post", color="#3498db", linewidth=2, alpha=0.7, zorder=1)

# Phase divider
ax.axvline(x=n_v1 - 0.5, color="#888888", linestyle="--", linewidth=1, alpha=0.4)

# Annotations — only for kept experiments with labels, alternate up/down to reduce overlap
label_up = True
for i in range(n):
    if all_status[i] == "keep" and all_labels[i]:
        c = "#2ecc71" if all_phase[i] == "v1" else "#3498db"
        # Alternate offset direction
        if label_up:
            xytext = (6, 10)
            va = "bottom"
        else:
            xytext = (6, -10)
            va = "top"
        label_up = not label_up
        ax.annotate(all_labels[i], (i, all_scores[i]),
                    textcoords="offset points", xytext=xytext,
                    fontsize=7, color=c, alpha=0.85,
                    rotation=30, ha="left", va=va)

# Phase labels at top
ylims = ax.get_ylim()
ax.text(n_v1 / 2, 1.11, "v1\n10 seasons, 128 features", ha="center", va="top",
        fontsize=11, color="#2ecc71", fontweight="bold", alpha=0.8)
ax.text(n_v1 + len(v2) / 2, 1.11, "v2\n20 seasons, 96 features", ha="center", va="top",
        fontsize=11, color="#3498db", fontweight="bold", alpha=0.8)

# Axes
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("val_score (lower is better)", fontsize=12)
ax.set_title(f"NBA Win Prediction: {n} Experiments, {n_kept} Kept Improvements",
             fontsize=15, fontweight="bold", pad=15)

# Best score annotation
best_i = min(range(n), key=lambda i: all_scores[i])
ax.annotate(f"best: {all_scores[best_i]:.4f}",
            (best_i, all_scores[best_i]),
            textcoords="offset points", xytext=(-40, -20),
            fontsize=9, fontweight="bold", color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', markersize=8, label='Discarded'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='v1 Kept'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='v2 Kept'),
    Line2D([0], [0], color='#2ecc71', linewidth=2, alpha=0.7, label='v1 Running best'),
    Line2D([0], [0], color='#3498db', linewidth=2, alpha=0.7, label='v2 Running best'),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

ax.set_xlim(-1, n + 1)
ax.set_ylim(0.91, 1.115)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("/Users/rr/autoresearch-nba-v2/progress.png", dpi=150, bbox_inches="tight")
print(f"Saved progress.png ({n} experiments, {n_kept} kept)")
