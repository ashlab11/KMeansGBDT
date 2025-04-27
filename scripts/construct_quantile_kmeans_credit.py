import matplotlib.pyplot as plt
import openml
import numpy as np
from src import DataBinner
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

class_benchmark_suite = openml.study.get_suite(337)  # 337 for classification

dataset_idx = 0
col_idx = 3
task_id = class_benchmark_suite.tasks[dataset_idx]
task = openml.tasks.get_task(task_id)
dataset = task.get_dataset()
name = dataset.name

X, y = task.get_X_and_y(dataset_format='dataframe')
quantile_binner = DataBinner(method='quantile', n_bins=255)
kmeans_binner = DataBinner(method='kmeans', n_bins=255)

X_q, X_k = quantile_binner.fit_transform(X), kmeans_binner.fit_transform(X)
splits_q, splits_k = quantile_binner._models, kmeans_binner._models

X_0 = X.iloc[:, col_idx]
splits_q = splits_q[col_idx]
splits_k = splits_k[col_idx]
# --- colour-blind–safe palette (from R's "Okabe-Ito") ------------------------
COL_HIST   = "#56B4E9"   # sky-blue
COL_Q      = "#D55E00"   # vermillion
COL_K      = "#009E73"   # bluish-green

fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)
fig.tight_layout(pad=0.5)

# ── histogram ────────────────────────────────────────────────────────────────
ax.hist(
    X_0, bins=100,
    histtype="stepfilled",
    facecolor=COL_HIST, edgecolor=COL_HIST,
    alpha=0.35,
    label="Original sample"
)

# ── vertical bin edges (sample every 10th to avoid clutter) ─────────────────
step = 1
ax.vlines(
    splits_q[::step],
    ymin=0, ymax=ax.get_ylim()[1],
    colors=COL_Q,
    linestyles="--", linewidth=0.8,
    label="Quantile split"
)
ax.vlines(
    splits_k[::step],
    ymin=0, ymax=ax.get_ylim()[1],
    colors=COL_K,
    linestyles="dotted", linewidth=0.8,
    label="K-Means split"
)

# ── axes styling ─────────────────────────────────────────────────────────────
ax.set(
    title=f"Quantile vs K-Means Binning – Credit, Column {col_idx}",
    xlabel="Feature value",
    ylabel="Count",
    yscale="log"
)

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which="both", axis="y", linewidth=0.3, alpha=0.5)

ax.yaxis.set_major_formatter(ScalarFormatter())
ax.tick_params(axis="both", which="major", labelsize=8)

ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

for spine in ("right", "top"):
    ax.spines[spine].set_visible(False)

#Save as png
plt.savefig(f"images/credit_col_{col_idx}_quantile_vs_kmeans.png", bbox_inches="tight")