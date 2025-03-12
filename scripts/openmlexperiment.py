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
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
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
    'n_estimators': randint(20, 150),
    'learning_rate': loguniform(0.001, 0.5),
    'num_leaves': randint(8, 64),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

param_dist_cat = {
    'n_estimators': randint(20, 150),
    'learning_rate': loguniform(0.001, 0.5),
    'depth': randint(4, 10),
    'subsample': uniform(0.5, 0.5),
    'colsample_bylevel': uniform(0.5, 0.5)
}

# List of binning methods to experiment with
binning_methods = ['none', 'quantile']

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(297)
benchmark_id = 0
task_id = benchmark_suite.tasks[benchmark_id]

task = openml.tasks.get_task(task_id)
dataset = task.get_dataset()
name = dataset.name
obs = dataset.qualities['NumberOfInstances']
features = dataset.qualities['NumberOfFeatures']
print(f"===== DATASET {name} with {obs} observations and {features} features =====")

# Get X and y
X, y = task.get_X_and_y(dataset_format='dataframe')
original_feature_names = X.columns

num_seeds = 20  # number of random splits

# Create dictionaries to store the results and best parameters per binning method
results = {}
best_params_results = {}
errors = np.zeros((len(binning_methods), num_seeds))
times = np.zeros((len(binning_methods), num_seeds))

for i, bin_method in enumerate(binning_methods):
    print(f"----- BINNING METHOD: {bin_method} -----")
    best_params_list = []  # best hyperparameters for each seed under this binning method
    for seed in np.arange(1, num_seeds):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        # Ensure we keep column names for consistency
        X_train = pd.DataFrame(X_train, columns=original_feature_names)
        X_test = pd.DataFrame(X_test, columns=original_feature_names)

        # Apply binning to features: fit on training data and then transform both training and test data.
        binner = DataBinner(method=bin_method, n_bins=255, random_state=seed)
        
        start = time.time()
        X_train_binned = pd.DataFrame(binner.fit_transform(X_train.values), columns = original_feature_names)
        X_test_binned = pd.DataFrame(binner.transform(X_test.values), columns = original_feature_names)
        bin_time = time.time() - start

        # Initialize the LGBM model and set up RandomizedSearchCV.
        model = LGBMRegressor(verbosity=-1, n_jobs=1, random_state=42)
        cv = RandomizedSearchCV(
            model, param_dist_lgbm, n_iter=30, cv=5, n_jobs=-1,
            random_state=seed, scoring='neg_mean_squared_error',
            error_score='raise', verbose=0
        )
        cv.fit(X_train_binned, y_train)
        best_params_list.append(cv.best_params_)

        # Predict on the test set and compute error.
        y_pred = cv.predict(X_test_binned)
        error = mean_squared_error(y_test, y_pred)
        errors[i, seed] = error
        times[i, seed] = bin_time
        print(f"Seed {seed} - Error: {error}, Time: {bin_time}")

    best_params_results[bin_method] = best_params_list
    method_errors = errors[i, :]
    method_times = times[i, :]
    mean_error = np.mean(method_errors)
    mean_times = np.mean(method_times)
    std_error = np.std(method_errors) / np.sqrt(num_seeds)
    std_times = np.std(method_times)
    print(f"Mean error for {bin_method}: {mean_error}, std: {std_error}")
    print(f"Mean time for {bin_method}: {mean_times}, std: {std_times}")

    results[bin_method] = method_errors.tolist()

# Compare the methods: compute mean errors and perform a paired t-test between the best and second-best.
means = errors.mean(axis=1)
argsort_idxs = np.argsort(means)
best_method = binning_methods[argsort_idxs[0]]
second_best_method = binning_methods[argsort_idxs[1]]
best_errors = errors[argsort_idxs[0], :]
second_best_errors = errors[argsort_idxs[1], :]
_, p_value = ttest_rel(best_errors, second_best_errors)
print(f"Best binning method: {best_method} with mean error {best_errors.mean()} +/- {best_errors.std()}")
print(f"P-value comparing best vs. second best ({second_best_method}): {p_value}")

# -------------------------
# Save the overall results and hyperparameters to JSON files
with open(f"param_list/overall_results_binning_{benchmark_id}.json", "w") as f:
    json.dump(results, f, indent=4)

with open(f"param_list/overall_params_binning_{benchmark_id}.json", "w") as f:
    json.dump(best_params_results, f, indent=4)
