import pandas as pd
from src import DataBinner
import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, FuncFormatter


sizes = [10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000, 7500000, 10000000]
times = np.zeros((len(sizes), 4))

linspace = DataBinner(method = 'linspace', n_bins = 255)
quantile = DataBinner(method = 'quantile', n_bins = 255)
kmeans_plus = DataBinner(method = 'kmeans', n_bins = 255, kmeans_init='++')
kmeans_quantile = DataBinner(method = 'kmeans', n_bins = 255, kmeans_init='quantile')


for i, size in enumerate(sizes):
    X = np.random.uniform(0, 1, size)
    X = pd.DataFrame(data = X, columns = ['feature'])
    times[i, 0] = timeit.timeit(lambda: linspace.fit_transform(X), number=1)
    times[i, 1] = timeit.timeit(lambda: quantile.fit_transform(X), number=1)
    times[i, 2] = timeit.timeit(lambda: kmeans_plus.fit_transform(X), number=1)
    times[i, 3] = timeit.timeit(lambda: kmeans_quantile.fit_transform(X), number=1)

# ── seaborn style & palette ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.9)          # unobtrusive grid
palette = sns.color_palette("colorblind", 4)              # C-B-safe colours

labels  = [
    "uniform (equal-width)",
    "quantile",
    "k-means (k-means++ init)",
    "k-means (quantile init)",
]
markers = ["o", "s", "D", "^"]                      # distinct point styles

fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=300)
fig.tight_layout(pad=0.4)

# ── one line per method ─────────────────────────────────────────────────────
for i, (lab, col, mk) in enumerate(zip(labels, palette, markers)):
    ax.plot(
        sizes,
        times[:, i],
        label=lab,
        color=col,
        marker=mk,
        markersize=4,
        linewidth=1.4,
    )

# ── axes, legend, & formatting ───────────────────────────────────────────────
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Number of observations",
    ylabel="Execution time (s)",
    title="Execution-time scaling of binning methods",
)

# minor grid on log axes
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
def eng_fmt(x, _):
    if x == 0:         # special-case zero
        return "0"
    for value, suffix in [(1e6, "M"), (1e3, "k")]:
        if x >= value:
            return f"{x/value:.0f}{suffix}"
    return f"{x:.0f}"

ax.xaxis.set_major_formatter(FuncFormatter(eng_fmt))
ax.xaxis.get_offset_text().set_visible(False)
ax.grid(which="minor", linewidth=0.3, alpha=0)
ax.grid(which="major", linewidth=0.6, alpha=0)

# legend outside plot area
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

# remove superfluous spines
for spine in ("right", "top"):
    ax.spines[spine].set_visible(False)

plt.savefig("images/FAKE_execution_time.png", bbox_inches="tight")
