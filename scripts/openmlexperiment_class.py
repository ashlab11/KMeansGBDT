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
    'gradientboostingclassifier__n_estimators': randint(20, 300),
    'gradientboostingclassifier__learning_rate': loguniform(0.001, 0.5),
    'gradientboostingclassifier__max_depth': randint(3, 6),
    'gradientboostingclassifier__subsample': uniform(0.5, 0.5),
    'gradientboostingclassifier__max_features': uniform(0.5, 0.5)}

# Parameter distribution for LightGBM
param_dist_lgbm = {
    'lgbmclassifier__n_estimators': randint(20, 300),
    'lgbmclassifier__learning_rate': loguniform(0.001, 0.5),
    'lgbmclassifier__num_leaves': randint(8, 64),
    'lgbmclassifier__subsample': uniform(0.5, 0.5),
    'lgbmclassifier__colsample_bytree': uniform(0.5, 0.5)
}

models = [
    (GradientBoostingClassifier(), "SKL", param_dist_sklearn),
    (LGBMClassifier(verbosity=-1, n_jobs=1, random_state=42), "LGBM", param_dist_lgbm), 
]

# List of binning methods to experiment with
binning_methods = [
    'kmeans',
    'quantile',
    'linspace',
    'exact'
]

# Retrieve a benchmark suite from OpenML and select a task
benchmark_suite = openml.study.get_suite(337) #337 for classification
benchmark_id = 0
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
#Columns 0/3 are the highly-skewed columns
X = X.iloc[:, [0]]

print([X[col].skew() for col in X.columns])
print([len(X[col].unique()) for col in X.columns])
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X = X.astype(float)
y = LabelEncoder().fit_transform(y)  # Encode labels for classification
original_feature_names = X.columns

num_seeds = 20  # number of random splits

# Creating dictionaries to store the results and best parameters per binning method
results = {}
accuracies = np.zeros((len(binning_methods), len(models), num_seeds))
roc_aucs = np.zeros((len(binning_methods), len(models), num_seeds))

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
                    param_dist['lgbmclassifier__max_bin'] = [len(X_train)]
                pipeline = make_pipeline(model, memory=memory)
            else:
                pipeline = make_pipeline(binner, model, memory=memory)
            
            cv = RandomizedSearchCV(
                pipeline, param_dist, n_iter=30, cv=5, n_jobs=-1,
                random_state=seed, scoring='roc_auc',
                error_score='raise', verbose=0
            )
            cv.fit(X_train, y_train)
            
            # Get accuracy
            y_pred = cv.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[i, j, seed] = accuracy
            
            # Get ROC AUC
            y_pred_proba = cv.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            roc_aucs[i, j, seed] = roc_auc
            
            print(f"Seed {seed}: Accuracy: {accuracy}, ROC AUC: {roc_auc}")
            
            memory.clear(warn=False)

        #Printing out mean and std of MSE/R2 for each binning method and model
        method_accuracies = accuracies[i, j, :]
        mean_acc = np.mean(method_accuracies)
        std_acc = np.std(method_accuracies) / np.sqrt(num_seeds)
        
        method_roc = roc_aucs[i, j, :]
        mean_roc = np.mean(method_roc)
        std_roc = np.std(method_roc) / np.sqrt(num_seeds)
        
        print(f"For {bin_method}, {model_name}:", 
              f"Mean MSE: {mean_acc}, std: {std_acc},",
              f"Mean R2: {mean_roc}, std: {std_roc}")

        # Storing the results in a dictionary
        results.setdefault(bin_method, {})
        results[bin_method].setdefault(model_name, {})
        
        results[bin_method][model_name]['accuracy'] = method_accuracies.tolist()
        results[bin_method][model_name]['roc_auc'] = method_roc.tolist()

#Saving the results to JSON file
with open(f"class_results_binning_{benchmark_id}.json", "w") as f:
    json.dump(results, f, indent=4)