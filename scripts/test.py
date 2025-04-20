import json
import openml
from scipy.stats import ttest_rel, rankdata
import lightgbm as lgb
from src import DataBinner
import numpy as np

num_datasets_class = 16
num_datasets_reg = 18

num_seeds = 20
model_order = ['LGBM']

reg_benchmark_suite = openml.study.get_suite(336)
#class_benchmark_suite = openml.study.get_suite(337)  # 337 for classification

#We want to be ranking across datasets within each model type
mrr_reg = {model: {
    'quantile': [], 
    'linspace': [], 
    'k-means': []
} for model in model_order}

mrr_class = {model: {
    'quantile': [], 
    'linspace': [], 
    'k-means': []
} for model in model_order}

mrr_total = {model: {
    'quantile': [], 
    'linspace': [], 
    'k-means': []
} for model in model_order}

for idx in range(num_datasets_reg):
    task_id = reg_benchmark_suite.tasks[idx]
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y = task.get_X_and_y(dataset_format='dataframe')
    name = dataset.name
    name = name.replace("_", "\_")
    obs = dataset.qualities['NumberOfInstances']
    features = dataset.qualities['NumberOfFeatures']
    
    unique_vals = []
    for col_idx in range(X.shape[1]):
        unique_vals.append(len(np.unique(X.iloc[:, col_idx])))
    unique_vals = np.array(unique_vals)
    print(f"For dataset {name}, unique vals per column are: {unique_vals}")

    
    