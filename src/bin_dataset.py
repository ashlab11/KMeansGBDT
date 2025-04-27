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
                 kmeans_init: str = "quantile",
                 random_state: int = 0):
        self.method = method
        self.n_bins = n_bins
        self.kmeans_init = kmeans_init
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
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------ #
    #  Core logic
    # ------------------------------------------------------------------ #
    def _fit_one_column(self, col: np.ndarray):
        """Return representation needed at transform-time."""
        unique = np.unique(col)
        if len(unique) <= self.n_bins:
            sorted_unique = np.sort(unique)
            cuts = (sorted_unique[1:] + sorted_unique[:-1]) / 2.0
            return cuts
        
        if self.method == "linspace":
            # n_bins‑1 inner cut points
            cuts = np.linspace(col.min(), col.max(), self.n_bins + 1)[1:-1]
            return cuts

        if self.method == "quantile":
            # include both end‑points to ensure all values fall in a bin
            cuts = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))[1:-1]
            return cuts         

        if self.method == "kmeans":          
            # ---- choose initial centres ----------------------------------
            seeds = np.quantile(col,
                                np.linspace(0, 1, self.n_bins))
            if self.kmeans_init == "++":
                # k-means++ initialisation
                km = KMeans(n_clusters=self.n_bins,
                        n_init=1,
                        random_state=self.random_state).fit(col.reshape(-1,1))
            else:
                km = KMeans(n_clusters=self.n_bins,
                        init=seeds.reshape(-1, 1),
                        n_init=1,
                        random_state=self.random_state).fit(col.reshape(-1,1))

            centres = np.sort(km.cluster_centers_.ravel())
            cuts = (centres[:-1] + centres[1:]) / 2.0
            return cuts
        
        raise ValueError("method must be 'linspace', 'quantile', or 'kmeans'")

    def fit(self, X, y=None):
        self._models = []
        """Learn cut‑points per feature."""
        if isinstance(X, pd.DataFrame):
            self._is_pandas = True
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        for j in range(X_arr.shape[1]):
            cuts = self._fit_one_column(X_arr[:, j])
            self._models.append(cuts)
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

        out = np.zeros_like(X_arr, dtype=int)

        for j, breakpoints in enumerate(self._models):
            out[:, j] = np.digitize(X_arr[:, j], breakpoints)

        if cols is not None:
            out = pd.DataFrame(out, columns=cols)
        return out

    # ------------------------------------------------------------------ #
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
