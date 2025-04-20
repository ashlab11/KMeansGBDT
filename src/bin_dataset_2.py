import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

class DataBinner:
    """
    Histogram-style discretiser for GBDT pipelines.

    Parameters
    ----------
    method : {"linspace", "quantile", "kmeans"}
        Binning scheme.
    n_bins : int
        Desired number of bins (per feature).
    kmeans_init : {"quantile", "++"}, default "quantile"
        Initialisation strategy used only when method=="kmeans".
    random_state : int, default 0
    """
    def __init__(self,
                 method: str,
                 n_bins: int = 255,
                 #kmeans_init: str = "quantile",
                 random_state: int = 0):
        self.method = method
        self.n_bins = n_bins
        #self.kmeans_init = kmeans_init
        self.random_state = random_state
        # will be filled after fit()
        self._models = []
        self._is_pandas = False

    # ------------------------------------------------------------------ #
    #  Scikit‑learn compatibility
    # ------------------------------------------------------------------ #
    def get_params(self, deep=True):
        return {"method": self.method,
                "n_bins": self.n_bins,
                #"kmeans_init": self.kmeans_init,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------ #
    #  Core logic
    # ------------------------------------------------------------------ #
    def _fit_one_column(self, col: np.ndarray):
        """Return representation needed at transform‑time."""
        n_bins = min(self.n_bins, len(np.unique(col)))
        if self.method == "linspace":
            # n_bins‑1 inner cut points
            cuts = np.linspace(col.min(), col.max(), n_bins + 1)[1:-1]
            return cuts

        if self.method == "quantile":
            # include both end‑points to ensure all values fall in a bin
            cuts = np.quantile(col, np.linspace(0, 1, n_bins + 1))[1:-1]
            return cuts         # drop potential duplicates

        if self.method == "kmeans":
            col_std = (col - np.mean(col)) / np.std(col)
            
            # ---- choose initial centres ----------------------------------
            seeds = np.quantile(col_std,
                                np.linspace(0, 1, n_bins + 2)[1:-1])
            #seeds = np.unique(seeds)         # de‑duplicate"""

            init_arg  = seeds.reshape(-1, 1)
            n_clusters = len(seeds)      # may be < n_bins if ties w/ quantile
            if n_clusters < 2:           # pathological all‑equal column
                return []

            km = MiniBatchKMeans(n_clusters=self.n_bins,
                        n_init=1,
                        #max_iter=100,
                        random_state=self.random_state).fit(col_std.reshape(-1,1))

            centres = np.sort(km.cluster_centers_.ravel())
            centres = centres * np.std(col) + np.mean(col)
            cuts = (centres[:-1] + centres[1:]) / 2.0
            return cuts

        raise ValueError("method must be 'linspace', 'quantile', or 'kmeans'")

    def fit(self, X, y=None):
        """Learn cut‑points per feature."""
        if isinstance(X, pd.DataFrame):
            self._is_pandas = True
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        self._models = [self._fit_one_column(X_arr[:, j])
                        for j in range(X_arr.shape[1])]
        return self

    # ------------------------------------------------------------------ #
    #  Transform
    # ------------------------------------------------------------------ #
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            cols = X.columns
            X_arr = X.values
        else:
            cols = None
            X_arr = np.asarray(X)

        out = np.empty_like(X_arr, dtype=int)

        for j, cuts in enumerate(self._models):
            out[:, j] = np.digitize(X_arr[:, j], cuts)

        if cols is not None:
            out = pd.DataFrame(out, columns=cols)
        return out

    # ------------------------------------------------------------------ #
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
