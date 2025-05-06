import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.ensemble import GradientBoostingRegressor
from typing import Callable
from src import DataBinner 

# ------------------------------------------------------------------
# Simple wrapper to train once on (X, y) with fixed basic hyper‑params
# ------------------------------------------------------------------
def train_model(X, y, binner, test_size=0.2, seed=0):
    """Train a model with fixed hyper-parameters."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=seed, verbose=0)
    model.fit(binner.fit_transform(X_tr), y_tr)
    preds = model.predict(binner.transform(X_te))
    return mean_squared_error(y_te, preds)

def make_synthetic(n_obs, n_features, n_modes, mode_size, skew_factor, skew_scale, rng):
    """Return DataFrame X and target y.
    Parameters:
    -----------
    n_obs : int
        Number of observations.
    n_features : int
        Number of features.
    n_modes : int
        Number of modes in the multimodal distribution.
    mode_size : float
        Difference between modes.
    skew_factor : float
        Probability of adding skewness.
    skew_scale : float
        Scale of the skewness.
    rng : np.random.Generator
        Random number generator.
    Returns:
    --------
    X : pd.DataFrame
        DataFrame with n_obs rows and n_features columns.
    """
    data = np.zeros((n_obs, n_features))
    # --- Multimodal component
    means = np.linspace(0, (n_modes - 1) * mode_size, n_modes) # Ensures modes are spaced apart
    for j in range(n_features):
        mode_choice = rng.integers(0, n_modes, size=n_obs)
        data[:, j] = rng.normal(loc=means[mode_choice], scale=1.0)
        data[:, j] = (data[:, j] - data[:, j].mean()) / data[:, j].std() # Standardizing
    
    # --- Skewness: add positive outliers with prob = skew_factor
    if skew_factor > 0:
        mask = rng.random(size=data.shape) < skew_factor
        data[mask] += rng.exponential(scale=skew_scale, size=mask.sum())
    X = pd.DataFrame(data, columns=[f"f{j}" for j in range(n_features)])
    y = X.sum(axis=1) + rng.normal(0, 0.1, size=n_obs)
    return X, y

#------------------------------------------------------------------
#Code for creating graphs

def plot_heatmap(delta, x_arr, y_arr, x_label, y_label, title, file_path, cmap, max):
    print("Heatmap has been created")
    mean = np.mean(delta, axis = 2)
    se   = delta.std(axis=2) / np.sqrt(30)        # std error per cell
    ci95 = 1.96 * se
    sig  = ci95 < np.abs(mean)      # CI excludes zero

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)

    # ----------------------------------------------------------------
    # imshow with origin='lower' draws (0,0) in the bottom-left cell
    # ----------------------------------------------------------------
    im = ax.imshow(mean,
                cmap=cmap,
                origin='lower',
                vmin=-10,
                vmax=max,
                aspect='auto')

    # ----------------------------------------------------------------
    # Put tick labels in the *centre* of each cell
    # ----------------------------------------------------------------
    ax.set_xticks(np.arange(len(x_arr)))
    ax.set_yticks(np.arange(len(y_arr)))

    # Convert numeric vectors to nicely formatted strings
    ax.set_xticklabels(x_arr)
    ax.set_yticklabels(y_arr)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(title)
    # after imshow() ...
    for (i, j), significant in np.ndenumerate(sig):
        if significant:
            ax.plot(j, i, marker='*', color='k', ms=4)   # black star
    # ----------------------------------------------------------------
    # Colour-bar
    # ----------------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Relative error-reduction (%)', rotation=270, labelpad=18)

    #Get ticks, rounded to nearest 5
    ticks = np.linspace(0, max, 4)
    tick_labels = [f"{int(tick)}%" for tick in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)   # ensures identical text on every fig
            
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

#-------------------------------------------------------------------
# Here we run multiple experiments:
# 1. Number of outliers vs outlier size (1 mode)
# 2. Number of modes vs outlier size (1% outliers)
# 3. Number of modes vs number of outliers (outlier scale = 10)
# 4. Number of outliers vs number of bins (outlier scale = 10)

#-------------------------------------------------------------------
# Experiment 1: Number of outliers vs outlier size (1 mode)
#-------------------------------------------------------------------
def experiment_1(runs = 20):
    skew_scale_grid = [5, 10, 15, 20]
    skew_frac_grid  = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]   # 0 % → 5 % outliers
    delta = np.zeros((len(skew_frac_grid), len(skew_scale_grid), runs))
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        for i, skew_frac in enumerate(skew_frac_grid):
            for j, skew_scale in enumerate(skew_scale_grid):
                X, y = make_synthetic(5000, 3, 1, 4, skew_frac, skew_scale, rng)
                mse_q = train_model(X, y, DataBinner('quantile', 255, random_state=42))
                mse_k = train_model(X, y, DataBinner('kmeans',   255, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q
    
    return delta, skew_scale_grid, skew_frac_grid

#-------------------------------------------------------------------
# Experiment 2: Number of modes vs outlier size (1% outliers)
#-------------------------------------------------------------------
def experiment_2(runs = 20):
    modes_grid = [3, 5, 7, 10]
    skew_scale_grid = [5, 10, 15, 20]
    delta = np.zeros((len(modes_grid), len(skew_scale_grid), runs))
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        for i, n_modes in enumerate(modes_grid):
            for j, skew_scale in enumerate(skew_scale_grid):
                X, y = make_synthetic(5000, 3, n_modes, 4, 0.01, skew_scale, rng)
                mse_q = train_model(X, y, DataBinner('quantile', 255, random_state=42))
                mse_k = train_model(X, y, DataBinner('kmeans',   255, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q
    
    return delta, skew_scale_grid, modes_grid
    
#-------------------------------------------------------------------
# Experiment 3: Number of modes vs number of outliers (outlier scale = 20)
#-------------------------------------------------------------------
def experiment_3(runs = 20):
    modes_grid = [3, 5, 7, 10]
    skew_frac_grid  = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]   # 0 % → 5 % outliers
    delta = np.zeros((len(skew_frac_grid), len(modes_grid), runs))
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        for i, skew_frac in enumerate(skew_frac_grid):
            for j, n_modes in enumerate(modes_grid):
                X, y = make_synthetic(5000, 3, n_modes, 4, skew_frac, 5, rng)
                mse_q = train_model(X, y, DataBinner('quantile', 255, random_state=42))
                mse_k = train_model(X, y, DataBinner('kmeans',   255, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q
    
    return delta, modes_grid, skew_frac_grid
#-------------------------------------------------------------------
# Experiment 4: Number of outliers vs observations per bin (outlier scale = 5, bin number = 255)
#-------------------------------------------------------------------
def experiment_4(runs = 20):
    obs_per_bin_grid = [8, 16, 32, 64, 128]
    skew_frac_grid  = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]   # 0 % → 5 % outliers
    delta = np.zeros((len(skew_frac_grid), len(obs_per_bin_grid), runs))
    for seed in range(runs):
        print(seed)
        rng = np.random.default_rng(seed)
        for i, skew_frac in enumerate(skew_frac_grid):
            for j, obs_per_bin in enumerate(obs_per_bin_grid):
                n_obs = 255 * obs_per_bin
                X, y = make_synthetic(n_obs, 3, 1, 1, skew_frac, 5, rng)
                mse_q = train_model(X, y, DataBinner('quantile', 255, random_state=42))
                mse_k = train_model(X, y, DataBinner('kmeans',   255, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q
    
    return delta, obs_per_bin_grid, skew_frac_grid
#-------------------------------------------------------------------
# Experiment 5: Number of outliers vs number of bins (outlier scale = 5, observations = 5k)
def experiment_5(runs = 20):
    bins = [16, 32, 64, 128, 256]
    skew_frac_grid = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 1]   # 0 % → 5 % outliers
    delta = np.zeros((len(skew_frac_grid), len(bins), runs))
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        for i, skew_frac in enumerate(skew_frac_grid):
            for j, n_bins in enumerate(bins):
                X, y = make_synthetic(5000, 3, 1, 1, skew_frac, 5, rng)
                mse_q = train_model(X, y, DataBinner('quantile', n_bins, random_state=42))
                mse_k = train_model(X, y, DataBinner('kmeans',   n_bins, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q
    
    return delta, bins, skew_frac_grid
#-------------------------------------------------------------------
#Experiment 6: Number of modes vs number of bins (no outliers, observations = 5k)
#-------------------------------------------------------
def experiment_6(runs = 20):
    bins = [16, 32, 64, 128, 256]
    modes_grid = [3, 5, 7, 10]
    delta = np.zeros((len(modes_grid), len(bins), runs))
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        for i, n_modes in enumerate(modes_grid):
            for j, n_bins in enumerate(bins):
                X, y = make_synthetic(5000, 3, n_modes, 4, 0.0, 5, rng)
                mse_q = train_model(X, y, DataBinner('quantile', n_bins, random_state=42))
                mse_k = train_model(X, y, DataBinner('kmeans',   n_bins, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q
    
    return delta, bins, modes_grid
#-------------------------------------------------------------------

if __name__ == "__main__":
    runs = 50
    d1, xarr1, yarr1 = experiment_1(runs = runs)
    np.save('d1.npy', d1)
    print("Experiment 1 done")
    
    d2, xarr2, yarr2 = experiment_2(runs = runs)
    np.save('d2.npy', d2)
    print("Experiment 2 done")
    
    d3, xarr3, yarr3 = experiment_3(runs = runs)
    np.save('d3.npy', d3)
    print("Experiment 3 done")
    
    d4, xarr4, yarr4 = experiment_4(runs = runs)
    np.save('d4.npy', d4)
    print("Experiment 4 done")
    
    d5, xarr5, yarr5 = experiment_5(runs = runs)
    np.save('d5.npy', d5)
    print("Experiment 5 done")
    
    d6, xarr6, yarr6 = experiment_6(runs = runs)
    np.save('d6.npy', d6)
    print("Experiment 6 done")
    
    #Load data
    d1, d2, d3 = np.load("d1.npy"), np.load("d2.npy"), np.load("d3.npy")
    d4, d5, d6 = np.load("d4.npy"), np.load("d5.npy"), np.load("d6.npy")
    xarr1, yarr1 = [5, 10, 15, 20], [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    xarr2, yarr2 = [5, 10, 15, 20], [3, 5, 7, 10]
    xarr3, yarr3 = [5, 10, 15, 20], [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    xarr4, yarr4 = [8, 16, 32, 64, 128], [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    xarr5, yarr5 = [16, 32, 64, 128, 256], [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 1]
    xarr6, yarr6 = [3, 5, 7, 10], [16, 32, 64, 128, 256]
    
    common_cmap = 'YlGn'
    #Taking the maximum value of the average deltas across all three experiments, used to set colorbar limits
    vmax = abs(np.concatenate([d1.mean(axis=2).ravel(), d2.mean(axis=2).ravel(), d3.mean(axis=2).ravel(), 
                               d4.mean(axis=2).ravel(), d5.mean(axis=2).ravel()])).max().round(-1)
    
    plot_heatmap(
        delta=d1,           # experiment-1 matrix
        x_arr=xarr1,
        y_arr=yarr1,
        x_label='Outlier scale',
        y_label='Skew fraction\n(outliers / total)',
        title='(1)  Single-mode data: outlier *fraction* vs. outlier *scale*',
        file_path='images/outlier_sensitivity.png',
        cmap=common_cmap, max = vmax
    )

    plot_heatmap(
        delta=d2,           # experiment-2 matrix
        x_arr=xarr2,
        y_arr=yarr2,
        x_label='Outlier scale',
        y_label='Number of density modes',
        title='(2)  1 % outliers: multi-modality vs. outlier size',
        file_path='images/modes_scale_sensitivity.png',
        cmap=common_cmap, max = vmax
    )

    plot_heatmap(
        delta=d3,           # experiment-3 matrix
        x_arr=xarr3,
        y_arr=yarr3,
        x_label='Number of density modes',
        y_label='Skew fraction\n(outliers / total)',
        title='(3)  Fixed outlier size (5σ): multi-modality vs. outlier mass',
        file_path='images/modes_frac_sensitivity.png',
        cmap=common_cmap, max = vmax
    )
    
    plot_heatmap(
        delta=d4,           # experiment-4 matrix
        x_arr=xarr4,
        y_arr=yarr4,
        x_label='Observations per bin',
        y_label='Skew fraction\n(outliers / total)',
        title='(4)  Bin size vs. outlier mass',
        file_path='images/obs_per_bin_sensitivity.png',
        cmap=common_cmap, max = vmax
    )
    
    plot_heatmap(
        delta=d5,           # experiment-5 matrix
        x_arr=xarr5,
        y_arr=yarr5,
        x_label='Number of bins',
        y_label='Skew fraction\n(outliers / total)',
        title='(5)  Bin number vs. outlier mass',
        file_path='images/num_bins_sensitivity.png',
        cmap=common_cmap, max = vmax
    )
    
    plot_heatmap(
        delta=d6,           # experiment-6 matrix
        x_arr=xarr6,
        y_arr=yarr6,
        x_label='Number of modes',
        y_label='Number of bins',
        title='(6)  Bin number vs. multi-modality',
        file_path='images/modes_bins_sensitivity.png',
        cmap=common_cmap, max = vmax
    )