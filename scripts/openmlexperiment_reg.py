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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Memory
from src import DataBinner 
import argparse

#Keywords that will be used to run this experiment
def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Run one regression-binning experiment")
    parser.add_argument("--num_seeds",   type=int, default=20)
    parser.add_argument("--benchmark_id",type=int, default=0)
    parser.add_argument("--n_bins",      type=int, default=255)
    return parser.parse_args()

args = get_cli_args()
num_seeds   = args.num_seeds
benchmark_id= args.benchmark_id
n_bins      = args.n_bins

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

# Parameter distribution for sklearn
param_dist = {
    'gradientboostingregressor__n_estimators': randint(20, 300),
    'gradientboostingregressor__learning_rate': loguniform(0.001, 0.5),
    'gradientboostingregressor__max_depth': randint(3, 6),
    'gradientboostingregressor__subsample': uniform(0.5, 0.5),
    'gradientboostingregressor__max_features': uniform(0.5, 0.5)}

model = GradientBoostingRegressor()

# List of binning methods to experiment with
binning_methods = [
    'kmeans',
    'quantile',
    'linspace',
    'exact'
]

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(336) #337 for classification
task_id = benchmark_suite.tasks[benchmark_id]

task = openml.tasks.get_task(task_id)
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

# Creating dictionaries to store the results and best parameters per binning method
results = {}
r2_scores = np.zeros((len(binning_methods), num_seeds))
mses = np.zeros((len(binning_methods), num_seeds))

for i, bin_method in enumerate(binning_methods):
    print(f"----- method: {bin_method} -----")
    for seed in range(num_seeds):
        
        memory = get_memory_for_dataset(bin_method, task_id, seed)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Ensure we keep column names for consistency
        X_train = pd.DataFrame(X_train, columns=original_feature_names)
        X_test = pd.DataFrame(X_test, columns=original_feature_names)

        # Apply binning to features: fit on training data and then transform both training and test data.
        binner = DataBinner(method=bin_method, n_bins=n_bins, random_state=seed)
        
        if bin_method == 'exact':
            #We also want to test against how the model does assuming NO binning -- will take SIGNIFICANTLY longer
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
        mses[i, seed] = mse
        r2 = r2_score(y_test, y_pred)
        r2_scores[i, seed] = r2
                
        memory.clear(warn=False)

    #Printing out mean and std of MSE/R2 for each binning method and model
    method_mse = mses[i, :]
    mean_mse = np.mean(method_mse)
    std_mse = np.std(method_mse) / np.sqrt(num_seeds)
    
    method_r2 = r2_scores[i, :]
    mean_r2 = np.mean(method_r2)
    std_r2 = np.std(method_r2) / np.sqrt(num_seeds)
    
    print(f"For {bin_method}:", 
            f"Mean MSE: {mean_mse}, std: {std_mse},",
            f"Mean R2: {mean_r2}, std: {std_r2}")

    # Storing the results in a dictionary
    results.setdefault(bin_method, {})
    
    results[bin_method]['mse'] = method_mse.tolist()
    results[bin_method]['r2'] = method_r2.tolist()

#Saving the results to JSON file
with open(f"benchmark_experiments/reg_bench_{benchmark_id}_bins_{n_bins}.json", "w") as f:
    json.dump(results, f, indent=4)