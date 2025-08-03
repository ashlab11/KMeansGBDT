import openml
from openml.tasks import list_tasks, TaskType
import os
import json
import logging
import numpy as np
import warnings
import pandas as pd
from scipy.stats import uniform, randint, loguniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Memory
from xgboost import XGBRegressor
from src import DataBinner 
import argparse
import time


warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sklearn").setLevel(logging.ERROR)

def sample_params():
    return {
        'n_estimators': randint.rvs(20, 300),
        'learning_rate': loguniform.rvs(0.001, 0.5),
        'max_depth': randint.rvs(3, 6),
        'subsample': uniform.rvs(0.5, 0.5),
        'max_features': uniform.rvs(0.5, 0.5)
    }

# List of binning methods to experiment with
binning_methods = [
    #'quantile',
    #'optimal_reg', 
    #'optimal_kmeans',
    #'kmeans' 
    'jenks'
]

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(336)
task_ids = np.array(benchmark_suite.tasks)[[-2, 7]]

for task_id in task_ids:
    task = openml.tasks.get_task(int(task_id))
    dataset = task.get_dataset()
    name = dataset.name
    obs = dataset.qualities['NumberOfInstances']
    features = dataset.qualities['NumberOfFeatures']
    print(f"===== DATASET {name} with {obs} observations and {features} features =====")

    # Get X and y
    X, y = task.get_X_and_y(dataset_format='dataframe')
    X = X.astype(float)
    y = y.astype(float)
    original_feature_names = X.columns
    running_train_time = 0
    
    for bin_method in binning_methods:
        running_bin_time = 0
            
        # Split the data
        X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Ensure we keep column names for consistency
        X_other = pd.DataFrame(X_other, columns=original_feature_names)
        X_test = pd.DataFrame(X_test, columns=original_feature_names)

        # Time the binning process
        binner = DataBinner(method=bin_method, n_bins=255, random_state=0)

        # Manual random search with cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        best_score = float('-inf')
        best_params = None
        
        for _ in range(30):  # n_iter = 30
            params = sample_params()
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_other):
                X_train = X_other.iloc[train_idx]
                y_train = y_other.iloc[train_idx]
                X_val = X_other.iloc[val_idx]
                y_val = y_other.iloc[val_idx]
                
                start_time = time.time()
                X_train_binned = binner.fit_transform(X_train, y_train)
                X_val_binned = binner.transform(X_val)
                binning_time = time.time() - start_time
                running_bin_time += binning_time
                
                
                model = GradientBoostingRegressor(**params, random_state=0)
                start_time = time.time()
                model.fit(X_train_binned, y_train)
                train_time = time.time() - start_time
                running_train_time += train_time
                val_pred = model.predict(X_val_binned)
                cv_scores.append(-mean_squared_error(y_val, val_pred))
            
            mean_cv_score = np.mean(cv_scores)
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = params

        # Train final model with best parameters
        final_model = GradientBoostingRegressor(**best_params, random_state=0)
        X_other_binned = binner.fit_transform(X_other, y_other)
        final_model.fit(X_other_binned, y_other)
        X_test_binned = binner.transform(X_test)  # Only transform for test set, no fitting needed
        y_pred = final_model.predict(X_test_binned)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Bin method: {bin_method}, Mean MSE: {mse}, Mean R2: {r2}, Average binning time: {running_bin_time / (30 * 5)} seconds")
        
    print(f"Average training time: {running_train_time / (30 * 5 * len(binning_methods))} seconds")
        
