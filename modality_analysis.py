import numpy as np

d1, d2, d3 = np.load("d1.npy"), np.load("d2.npy"), np.load("d3.npy")

print(abs(np.concatenate([d1.mean(axis=2).ravel(), d2.mean(axis=2).ravel(), d3.mean(axis=2).ravel()])).max().round(-1))