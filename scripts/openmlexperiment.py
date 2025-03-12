import openml
from openml.tasks import list_tasks, TaskType
import os
import json
import logging
import numpy as np
import time
import warnings
import pandas as pd
from scipy.stats import uniform, randint, loguniform, ttest_rel
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor



# Import our binning class.
# DataBinner should be defined as in our previous example.
from src import DataBinner  

# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# Parameter distribution for LightGBM
param_dist_lgbm = {
    'lgbmregressor__n_estimators': randint(20, 150),
    'lgbmregressor__learning_rate': loguniform(0.001, 0.5),
    'lgbmregressor__num_leaves': randint(8, 64),
    'lgbmregressor__subsample': uniform(0.5, 0.5),
    'lgbmregressor__colsample_bytree': uniform(0.5, 0.5)
}

param_dist_xgb = {
    'n_estimators': randint(20, 150),
    'max_depth': randint(3, 6),
    'learning_rate': loguniform(0.001, 0.5),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
}

param_dist_cat = {
    'n_estimators': randint(20, 150),
    'learning_rate': loguniform(0.001, 0.5),
    'depth': randint(3, 6),
    'subsample': uniform(0.5, 0.5),
    'colsample_bylevel': uniform(0.5, 0.5)
}

models = [
    (LGBMRegressor(verbosity=-1, n_jobs=1, random_state=42), "LGBM", param_dist_lgbm), 
    (XGBRegressor(random_state=42, max_bin=255, tree_method = 'hist'), "XGB", param_dist_xgb), 
    (CatBoostRegressor(verbose = False, random_state = 42, max_bin=255), "CAT", param_dist_cat)
]

# List of binning methods to experiment with
binning_methods = [
    'minibatch_kmeans', 
    'quantile',
    'linspace', 
    'none'
]

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(336) #337 for classification
benchmark_id = 12
task_id = benchmark_suite.tasks[benchmark_id]

task = openml.tasks.get_task(task_id)
dataset = task.get_dataset()
name = dataset.name
obs = dataset.qualities['NumberOfInstances']
features = dataset.qualities['NumberOfFeatures']
print(f"===== DATASET {name} with {obs} observations and {features} features =====")

# Get X and y
X, y = task.get_X_and_y(dataset_format='dataframe')
#remove final column from X
#X = X.iloc[:, :-1]

original_feature_names = X.columns


num_seeds = 20  # number of random splits

# Create dictionaries to store the results and best parameters per binning method
results = {}
errors = np.zeros((len(binning_methods), len(models), num_seeds))

for j, (model, model_name, param_dist) in enumerate(models):
    for i, bin_method in enumerate(binning_methods):
        print(f"-----MODEL: {model_name}-----")
        print(f"----- method: {bin_method} -----")
        for seed in range(num_seeds):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            # Ensure we keep column names for consistency
            X_train = pd.DataFrame(X_train, columns=original_feature_names)
            X_test = pd.DataFrame(X_test, columns=original_feature_names)

            # Apply binning to features: fit on training data and then transform both training and test data.
            binner = DataBinner(method=bin_method, n_bins=255, random_state=seed)
            
            pipeline = make_pipeline(binner, model, memory = "../cached_transformers/")
            
            cv = RandomizedSearchCV(
                pipeline, param_dist, n_iter=30, cv=5, n_jobs=-1,
                random_state=seed, scoring='neg_mean_squared_error',
                error_score='raise', verbose=0
            )
            cv.fit(X_train, y_train)

            # Predict on the test set and compute error.
            y_pred = cv.predict(X_test)
            error = mean_squared_error(y_test, y_pred)
            errors[i, j, seed] = error
            print(f"Seed {seed} - Error: {error}")

        method_errors = errors[i, j, :]
        mean_error = np.mean(method_errors)
        std_error = np.std(method_errors) / np.sqrt(num_seeds)
        print(f"Mean error for {bin_method}, {model_name}: {mean_error}, std: {std_error}")

        results.setdefault(bin_method, {})
        results[bin_method][model_name] = method_errors.tolist()

    