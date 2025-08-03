import openml
from scipy.stats import skew
import numpy as np

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(337) #337 for classification
for task_id in benchmark_suite.tasks:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    name = dataset.name
    obs = dataset.qualities['NumberOfInstances']
    features = dataset.qualities['NumberOfFeatures']
    X, y = task.get_X_and_y(dataset_format='dataframe')
    X = X.astype(float)
    skews = np.array([skew(X[col]) for col in X.columns])
    
    print(f"===== DATASET {name} with {obs} observations and {features} features, Mean Skew: {np.mean(skews)} =====")
