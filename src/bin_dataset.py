import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans, MiniBatchKMeans
import pandas as pd
from optbinning import MDLP
class DataBinner:
    def __init__(self, method, n_bins, kmean_init = 'quantile', random_state=0):
        self.method = method
        self.n_bins = n_bins
        self.random_state = random_state
        self.kmean_init = kmean_init
        self.models = []  # For each feature, store breakpoints or fitted model
        self.is_pandas = False

    def get_params(self, deep=True):
        """
        Return parameters for caching/cloning.
        Note: This returns the hyperparameters only. The fitted models
        are stored in the object instance and will be cached along with it.
        """
        return {"method": self.method, "n_bins": self.n_bins, "kmean_init": self.kmean_init, "random_state": self.random_state}

    def set_params(self, **params):
        """
        Set the parameters for this estimator.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, y = None):
        if isinstance(X, pd.DataFrame):
            self.is_pandas = True
        X = np.array(X)
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            unique = np.unique(col)
            n_bins = max(2, min(len(set(col)), self.n_bins) - 2)
            if self.method == 'linspace':
                # Compute equally spaced breakpoints
                breakpoints = np.linspace(col.min(), col.max(), n_bins)
                self.models.append(breakpoints)
            elif self.method == 'quantile':
                breakpoints = np.quantile(col, np.linspace(0, 1, n_bins))
                self.models.append(breakpoints)
            elif self.method == 'minibatch_kmeans':
                # Normalize data to improve clustering performance
                model = MiniBatchKMeans(n_clusters=n_bins, random_state=self.random_state, n_init = 1)
                model.fit(col.reshape(-1, 1))
                # Sort cluster centers so the bin labels reflect ascending numeric order
                centers = model.cluster_centers_.flatten()
                sorted_idx = np.argsort(centers)
                # Create a mapping: old_label -> new_label
                label_mapping = { old_label: new_label
                                    for new_label, old_label in enumerate(sorted_idx) }
                # Store both the fitted model and the mapping
                self.models.append((model, label_mapping))
            elif self.method == 'kmeans':
                if self.kmean_init == '++':
                    model = KMeans(n_clusters=n_bins, random_state=self.random_state, n_init = 1)
                else:
                    inits = np.quantile(unique, np.linspace(0, 1, n_bins))
                    model = KMeans(n_clusters=n_bins, random_state=self.random_state, n_init = 1, init = inits.reshape(-1, 1))
                model.fit(col.reshape(-1, 1))
                # Sort cluster centers so the bin labels reflect ascending numeric order
                centers = model.cluster_centers_.flatten()
                sorted_idx = np.argsort(centers)
                # Create a mapping: old_label -> new_label
                label_mapping = { old_label: new_label
                                    for new_label, old_label in enumerate(sorted_idx) }
                # Store both the fitted model and the mapping
                self.models.append((model, label_mapping))
            else:
                raise ValueError("Invalid method. Choose one of ['linspace', 'quantile', 'jenks', 'kmeans', 'bisecting_kmeans', 'agglomerative'].")
        return self
    
    def transform(self, X, y = None):
        columns = X.columns if self.is_pandas else None
        X = np.array(X)
        X_binned = np.zeros(X.shape)
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            if self.method in ['linspace', 'quantile', 'jenks', 'mdlp']:
                breakpoints = self.models[i]
                # np.digitize assigns bins based on the breakpoints computed from training data.
                X_binned[:, i] = np.digitize(col, breakpoints)
            elif self.method in ['minibatch_kmeans', 'kmeans', 'bisecting_kmeans']:
                model, label_mapping = self.models[i]
                raw_labels = model.predict(col.reshape(-1, 1))
                # Re-map the cluster labels so they follow ascending numeric order
                mapped_labels = [label_mapping[lab] for lab in raw_labels]
                X_binned[:, i] = mapped_labels
            elif 'none' in self.method:
                X_binned = X
        
        if self.is_pandas:
            X_binned = pd.DataFrame(X_binned, columns = columns)
        return X_binned

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
