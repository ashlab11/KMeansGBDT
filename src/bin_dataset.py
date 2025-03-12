import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans, MiniBatchKMeans

class DataBinner:
    def __init__(self, method, n_bins, random_state=0):
        self.method = method
        self.n_bins = n_bins
        self.random_state = random_state
        self.models = []  # For each feature, store breakpoints or fitted model
        
    def fit(self, X, y = None):
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            n_bins = max(2, min(len(set(col)), self.n_bins) - 2)
            if self.method == 'linspace':
                # Compute equally spaced breakpoints
                breakpoints = np.linspace(col.min(), col.max(), n_bins)
                self.models.append(breakpoints)
            elif self.method == 'quantile':
                breakpoints = np.quantile(col, np.linspace(0, 1, n_bins))
                self.models.append(breakpoints)
            elif 'none' in self.method:
                pass
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
                model = KMeans(n_clusters=n_bins, random_state=self.random_state, n_init = 10)
                model.fit(col.reshape(-1, 1))
                # Sort cluster centers so the bin labels reflect ascending numeric order
                centers = model.cluster_centers_.flatten()
                sorted_idx = np.argsort(centers)
                # Create a mapping: old_label -> new_label
                label_mapping = { old_label: new_label
                                    for new_label, old_label in enumerate(sorted_idx) }
                # Store both the fitted model and the mapping
                self.models.append((model, label_mapping))
            elif self.method == 'bisecting_kmeans':
                model = BisectingKMeans(n_clusters=n_bins, random_state=self.random_state, init='k-means++', verbose=0, 
                                        bisecting_strategy = 'largest_cluster')
                model.fit(col.reshape(-1, 1))
                # Same idea: reorder cluster labels by ascending center
                centers = model.cluster_centers_.flatten()
                sorted_idx = np.argsort(centers)
                label_mapping = { old_label: new_label
                                    for new_label, old_label in enumerate(sorted_idx) }
                self.models.append((model, label_mapping))
            elif self.method == 'agglomerative':
                # AgglomerativeClustering does not support predicting new data.
                # We fit it on the training data and then compute cluster centers.
                clustering = AgglomerativeClustering(n_clusters=n_bins)
                labels = clustering.fit_predict(col.reshape(-1, 1))
                centers = np.zeros(self.n_bins)
                for label in np.unique(labels):
                    centers[label] = col[labels == label].mean()
                self.models.append(centers)
            else:
                raise ValueError("Invalid method. Choose one of ['linspace', 'quantile', 'jenks', 'kmeans', 'bisecting_kmeans', 'agglomerative'].")
        return self
    
    def transform(self, X, y = None):
        X_binned = np.zeros(X.shape)
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            if self.method in ['linspace', 'quantile', 'jenks']:
                breakpoints = self.models[i]
                # np.digitize assigns bins based on the breakpoints computed from training data.
                X_binned[:, i] = np.digitize(col, breakpoints)
            elif self.method in ['minibatch_kmeans', 'kmeans', 'bisecting_kmeans']:
                model, label_mapping = self.models[i]
                raw_labels = model.predict(col.reshape(-1, 1))
                # Re-map the cluster labels so they follow ascending numeric order
                mapped_labels = [label_mapping[lab] for lab in raw_labels]
                X_binned[:, i] = mapped_labels
            elif self.method == 'agglomerative':
                centers = self.models[i]
                # For each value, assign the bin corresponding to the nearest center.
                # This is a simple heuristic since AgglomerativeClustering does not provide a predict method.
                distances = np.abs(col.reshape(-1, 1) - centers.reshape(1, -1))
                X_binned[:, i] = np.argmin(distances, axis=1)
            elif 'none' in self.method:
                X_binned = X
        return X_binned

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
