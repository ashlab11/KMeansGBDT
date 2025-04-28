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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from joblib import Memory


# Import our binning class.
# DataBinner should be defined as in our previous example.
from src import DataBinner  

def get_memory_for_dataset(bin_method, task_id, seed):
    """
    Create (or reuse) a cache folder specific to this dataset/task.
    This way, each dataset version has its own cache files.
    """
    cache_dir = os.path.join("../cached_datasets", bin_method, str(task_id), str(seed))
    os.makedirs(cache_dir, exist_ok=True)
    return Memory(location=cache_dir, verbose=0)

# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# Parameter distribution for sklearn
param_dist_sklearn = {
    'gradientboostingregressor__n_estimators': randint(20, 300),
    'gradientboostingregressor__learning_rate': loguniform(0.001, 0.5),
    'gradientboostingregressor__max_depth': randint(3, 6),
    'gradientboostingregressor__subsample': uniform(0.5, 0.5),
    'gradientboostingregressor__max_features': uniform(0.5, 0.5)}

# Parameter distribution for LightGBM
param_dist_lgbm = {
    'lgbmregressor__n_estimators': randint(20, 300),
    'lgbmregressor__learning_rate': loguniform(0.001, 0.5),
    'lgbmregressor__num_leaves': randint(8, 64),
    'lgbmregressor__subsample': uniform(0.5, 0.5),
    'lgbmregressor__colsample_bytree': uniform(0.5, 0.5)
}

models = [
    (GradientBoostingRegressor(), "SKL", param_dist_sklearn),
    (LGBMRegressor(verbosity=-1, n_jobs=1, random_state=42), "LGBM", param_dist_lgbm), 
]

# List of binning methods to experiment with
binning_methods = [
    #'exact',
    'kmeans',
    'quantile',
    'linspace'
]

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(336) #337 for classification
benchmark_id = 4
task_id = benchmark_suite.tasks[benchmark_id]

task = openml.tasks.get_task(task_id)
dataset = task.get_dataset()
name = dataset.name
print(dataset.format)
obs = dataset.qualities['NumberOfInstances']
features = dataset.qualities['NumberOfFeatures']
print(f"===== DATASET {name} with {obs} observations and {features} features =====")


# Get X and y
X, y = task.get_X_and_y(dataset_format='dataframe')
X = X.astype(float)
y = y.astype(float)
print(X.dtypes)
original_feature_names = X.columns

num_seeds = 20  # number of random splits

# Creating dictionaries to store the results and best parameters per binning method
results = {}
r2_scores = np.zeros((len(binning_methods), len(models), num_seeds))
mses = np.zeros((len(binning_methods), len(models), num_seeds))

for j, (model, model_name, param_dist) in enumerate(models):
    for i, bin_method in enumerate(binning_methods):
        print(f"-----MODEL: {model_name}-----")
        print(f"----- method: {bin_method} -----")
        for seed in range(num_seeds):
            
            memory = get_memory_for_dataset(bin_method, task_id, seed)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            
            # Ensure we keep column names for consistency
            X_train = pd.DataFrame(X_train, columns=original_feature_names)
            X_test = pd.DataFrame(X_test, columns=original_feature_names)

            # Apply binning to features: fit on training data and then transform both training and test data.
            binner = DataBinner(method=bin_method, n_bins=255, random_state=seed)
            
            if bin_method == 'exact':
                #We also want to test against how the model does assuming NO binning -- will take SIGNIFICANTLY longer
                if model_name == 'LGBM':
                    # This is a workaround for LGBMRegressor, which doesn't natively support naive no-binning methods
                    param_dist['lgbmregressor__max_bin'] = [len(X_train)]
                pipeline = make_pipeline(model, memory = memory)
            else:
                pipeline = make_pipeline(binner, model, memory = memory)
            
            cv = RandomizedSearchCV(
                pipeline, param_dist, n_iter=30, cv=5, n_jobs=-1,
                random_state=seed, scoring='neg_mean_squared_error',
                error_score='raise', verbose=0
            )
            cv.fit(X_train, y_train)
            
            # Predict on the test set and compute error.
            y_pred = cv.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mses[i, j, seed] = mse
            r2 = r2_score(y_test, y_pred)
            r2_scores[i, j, seed] = r2
            
            print(f"Seed {seed}: MSE: {mse}, R2: {r2}")
            
            memory.clear(warn=False)

        #Printing out mean and std of MSE/R2 for each binning method and model
        method_mse = mses[i, j, :]
        mean_mse = np.mean(method_mse)
        std_mse = np.std(method_mse) / np.sqrt(num_seeds)
        
        method_r2 = r2_scores[i, j, :]
        mean_r2 = np.mean(method_r2)
        std_r2 = np.std(method_r2) / np.sqrt(num_seeds)
        
        print(f"For {bin_method}, {model_name}:", 
              f"Mean MSE: {mean_mse}, std: {std_mse},",
              f"Mean R2: {mean_r2}, std: {std_r2}")

        # Storing the results in a dictionary
        results.setdefault(bin_method, {})
        results[bin_method].setdefault(model_name, {})
        
        results[bin_method][model_name]['mse'] = method_mse.tolist()
        results[bin_method][model_name]['r2'] = method_r2.tolist()

#Saving the results to JSON file
with open(f"reg_results_{benchmark_id}.json", "w") as f:
    json.dump(results, f, indent=4)