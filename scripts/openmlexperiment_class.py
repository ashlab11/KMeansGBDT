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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import Memory
import re
import argparse
from src import DataBinner 

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

def get_memory_for_dataset(task_id):
    """
    Create (or reuse) a cache folder specific to this dataset/task.
    This way, each dataset version has its own cache files.
    """
    cache_dir = os.path.join("../cached_datasets", str(task_id))
    os.makedirs(cache_dir, exist_ok=True)
    return Memory(location=cache_dir, verbose=0)

# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# Parameter distribution for sklearn
param_dist = {
    'gradientboostingclassifier__n_estimators': randint(20, 300),
    'gradientboostingclassifier__learning_rate': loguniform(0.001, 0.5),
    'gradientboostingclassifier__max_depth': randint(3, 6),
    'gradientboostingclassifier__subsample': uniform(0.5, 0.5),
    'gradientboostingclassifier__max_features': uniform(0.5, 0.5)}

model = GradientBoostingClassifier()

# List of binning methods to experiment with
binning_methods = [
    #'kmeans',
    #'quantile',
    #'linspace',
    'optimal_class'
    #'exact'
]

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(337) #337 for classification
task_id = benchmark_suite.tasks[benchmark_id]

task = openml.tasks.get_task(task_id)
dataset = task.get_dataset()
name = dataset.name
print(dataset.format)
obs = dataset.qualities['NumberOfInstances']
features = dataset.qualities['NumberOfFeatures']
print(f"===== DATASET {name} with {obs} observations and {features} features =====")

memory = get_memory_for_dataset(task_id)

# Get X and y
X, y = task.get_X_and_y(dataset_format='dataframe')
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X = X.astype(float)
y = LabelEncoder().fit_transform(y)  # Encode labels for classification
original_feature_names = X.columns

num_seeds = 20  # number of random splits

# Creating dictionaries to store the results and best parameters per binning method
results = {}
accuracies = np.zeros((len(binning_methods), num_seeds))
roc_aucs = np.zeros((len(binning_methods), num_seeds))

for i, bin_method in enumerate(binning_methods):
    print(f"----- method: {bin_method} -----")
    for seed in range(num_seeds):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Ensure we keep column names for consistency
        X_train = pd.DataFrame(X_train, columns=original_feature_names)
        X_test = pd.DataFrame(X_test, columns=original_feature_names)

        # Apply binning to features: fit on training data and then transform both training and test data.
        binner = DataBinner(method=bin_method, n_bins=n_bins, random_state=seed)
        model.set_params(random_state=seed)
        
        if bin_method == 'exact':
            #We also want to test against how the model does assuming NO binning -- will take SIGNIFICANTLY longer
            pipeline = make_pipeline(model)
        else:
            pipeline = make_pipeline(binner, model)
        
        cv = RandomizedSearchCV(
            pipeline, param_dist, n_iter=30, cv=5, n_jobs=-1,
            random_state=seed, scoring='roc_auc',
            error_score='raise', verbose=0
        )
        cv.fit(X_train, y_train)
        
        # Get accuracy
        y_pred = cv.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[i, seed] = accuracy
        
        # Get ROC AUC
        y_pred_proba = cv.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        roc_aucs[i, seed] = roc_auc
        
        print(f"Seed {seed}: Accuracy: {accuracy}, ROC AUC: {roc_auc}")
        
        #memory.clear(warn=False)

    #Printing out mean and std of MSE/R2 for each binning method and model
    method_accuracies = accuracies[i, :]
    mean_acc = np.mean(method_accuracies)
    std_acc = np.std(method_accuracies) / np.sqrt(num_seeds)
    
    method_roc = roc_aucs[i, :]
    mean_roc = np.mean(method_roc)
    std_roc = np.std(method_roc) / np.sqrt(num_seeds)
    
    print(f"For {bin_method}:", 
            f"Mean Accuracy: {mean_acc}, std: {std_acc},",
            f"Mean ROC_AUC: {mean_roc}, std: {std_roc}")

    # Storing the results in a dictionary
    results.setdefault(bin_method, {})
    
    results[bin_method]['accuracy'] = method_accuracies.tolist()
    results[bin_method]['roc_auc'] = method_roc.tolist()

#Saving the results to JSON file
with open(f"benchmark_experiments/class_bench_{benchmark_id}_bins_{n_bins}.json", "r") as f:
    existing_results = json.load(f)

for bin_method in binning_methods:
    existing_results[bin_method] = {}
    existing_results[bin_method]['accuracy'] = results[bin_method]['accuracy']
    existing_results[bin_method]['roc_auc'] = results[bin_method]['roc_auc']

with open(f"benchmark_experiments/class_bench_{benchmark_id}_bins_{n_bins}.json", "w") as f:
    json.dump(existing_results, f, indent=4)