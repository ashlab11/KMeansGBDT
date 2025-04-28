import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from typing import Callable
from src import DataBinner  # your class


# ------------------------------------------------------------------
# Simple wrapper to train once on (X, y) with fixed hyper‑params
# ------------------------------------------------------------------
def train_lgbm(X, y, binner, test_size=0.2, seed=0):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    model = LGBMRegressor(
        n_estimators=100, learning_rate=0.1, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1, n_jobs = -1)
    model.fit(binner.fit_transform(X_tr), y_tr)
    preds = model.predict(binner.transform(X_te))
    return mean_squared_error(y_te, preds)

def make_synthetic(n_obs, n_features, n_modes, mode_size, skew_factor, skew_scale, rng):
    """Return DataFrame X and target y."""
    data = np.zeros((n_obs, n_features))
    # --- Multimodal component
    means = np.linspace(0, n_modes*mode_size, n_modes)
    for j in range(n_features):
        mode_choice = rng.integers(0, n_modes, size=n_obs)
        data[:, j] = rng.normal(loc=means[mode_choice], scale=1.0)
    # --- Skewness: add positive outliers with prob = skew_factor
    if skew_factor > 0:
        mask = rng.random(size=data.shape) < skew_factor
        data[mask] += rng.exponential(scale=skew_scale, size=mask.sum())
    X = pd.DataFrame(data, columns=[f"f{j}" for j in range(n_features)])
    y = X.sum(axis=1) + rng.normal(0, 0.1, size=n_obs)
    return X, y

def sensitivity_grid(runs = 20):
    modes_grid = [1, 2, 5, 10]
    skew_grid  = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]   # 0 % → 5 % outliers
    delta = np.zeros((len(skew_grid), len(modes_grid), runs))
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        for i, skew in enumerate(skew_grid):
            for j, n_modes in enumerate(modes_grid):
                X, y = make_synthetic(5000, 3, n_modes, 3, skew, rng)
                mse_q = train_lgbm(X, y, DataBinner('quantile', 255, random_state=42))
                mse_k = train_lgbm(X, y, DataBinner('kmeans',   255, random_state=42))
                delta[i, j, seed] = 100 * (mse_q - mse_k) / mse_q

    delta = np.mean(delta, axis = 2)
    abs_max = np.max(np.abs(delta))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)

    # ----------------------------------------------------------------
    # imshow with origin='lower' draws (0,0) in the bottom-left cell
    # ----------------------------------------------------------------
    im = ax.imshow(delta,
                cmap='RdYlGn',
                origin='lower',
                vmin=-np.abs(delta).max(),
                vmax= np.abs(delta).max(),
                aspect='auto')

    # ----------------------------------------------------------------
    # Put tick labels in the *centre* of each cell
    # ----------------------------------------------------------------
    ax.set_xticks(np.arange(len(modes_grid)))
    ax.set_yticks(np.arange(len(skew_grid)))

    # Convert numeric vectors to nicely formatted strings
    ax.set_xticklabels([str(m)          for m in modes_grid])
    ax.set_yticklabels([f'{s:.0f}%'      for s in skew_grid])

    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Skew fraction (outliers / total)')

    ax.set_title('Relative gain of K-means over Quantile\n'
                '(lower = worse, higher = better)')

    # ----------------------------------------------------------------
    # Optional:  annotate each square with the value (integer %)
    # ----------------------------------------------------------------
    for j in range(delta.shape[0]):            # row (skew)
        for i in range(delta.shape[1]):        # column (modes)
            text = ax.text(i, j, f'{delta[j, i]:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if abs(delta[j, i]) < 15 else 'white')

    # ----------------------------------------------------------------
    # Colour-bar
    # ----------------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Relative error-reduction (%)', rotation=270, labelpad=18)

    plt.tight_layout()
    plt.savefig('images/outlier_sensitivity.png', dpi=300)
    plt.show()

       
#----FUNCTION FOR HOW ERROR DECREASES WITH BIN NUMBER ----
def bins_ablation(bins_list=[16, 32, 64, 128, 255, 512], num_datasets=10):
    results = []    
    for b in bins_list:
        print('----- BIN', b)
        for method in ['quantile', 'kmeans', 'linspace']:
            errors = np.zeros(num_datasets)
            for seed in range(num_datasets):
                X, y = make_synthetic(n_obs = 1000, n_features=3, n_modes = 10, skew_factor=0.01, rng = np.random.default_rng(seed))
                mse = train_lgbm(X, y, DataBinner(method, b, 42))
                errors[seed] = mse
            results.append({'bins':b, 'method':method, 'mse':np.mean(errors)})
    df = pd.DataFrame(results)
    
    # ---------- plot ----------
    fig, ax = plt.subplots()
    for m, g in df.groupby('method'):
        ax.plot(g['bins'], g['mse'], marker='o', label=m)
    ax.set_xlabel('# bins'); ax.set_ylabel('Test MSE')
    ax.set_title('Ablation on number of bins')
    ax.legend(); plt.show()
    return df

def bootstrap_stability(X, y, binner_factory: Callable[[], object],
                         n_boot=30, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    mse_vals = []
    for i in range(n_boot):
        # bootstrap rows with replacement
        idx = rng.integers(0, len(X), len(X))
        Xb, yb = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        mse = train_lgbm(Xb, yb, binner_factory())
        mse_vals.append(mse)
    mse_vals = np.array(mse_vals)
    return mse_vals.mean(), mse_vals.var(ddof=1)

def run_stability_demo():
    X, y = make_synthetic(n_obs = 1000, n_features=3, n_modes = 10, skew_factor=0.01, rng = np.random.default_rng())
    q_mean, q_var = bootstrap_stability(X, y,
        lambda: DataBinner('quantile', 255, 42))
    k_mean, k_var = bootstrap_stability(X, y,
        lambda: DataBinner('kmeans', 255, 42))
    print(f'Quantile: mean={q_mean:.4f}, var={q_var:.4e}')
    print(f'K‑means : mean={k_mean:.4f}, var={k_var:.4e}')
    print('σ²_between/σ²_within = ', q_var/k_var)

if __name__ == "__main__":
    #bins_ablation()    
    sensitivity_grid()
    #run_stability_demo()
