import json
import openml
from scipy.stats import ttest_rel, rankdata, false_discovery_control, wilcoxon
import lightgbm as lgb
from src import DataBinner
import numpy as np
import argparse

num_datasets_class = 16
num_datasets_reg = 18

#Keywords that will be used to analyze experiment results
def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Analyze experiments")
    parser.add_argument("--num_seeds",   type=int, default=20)
    parser.add_argument("--std_error",   action='store_true', default=False)
    return parser.parse_args()

args = get_cli_args()
num_seeds   = args.num_seeds
std_error   = args.std_error #Whether to include standard error in the results

reg_benchmark_suite = openml.study.get_suite(336)
class_benchmark_suite = openml.study.get_suite(337) 

#We want to be ranking across datasets within each model type
def star_if_sig(best_val, pval_adj_bool):
    formatted = f"{best_val:.3f}"
    if pval_adj_bool:
        return f"\\textbf{{{formatted}}}"
    else:
        return formatted

p_vals = []
latex_results = [] #Results for each idx, which we'll use later on to print!

#Classification first
print("DATA FOR CLASSIFICATION")
mrr_class = {'quantile': [], 'linspace': [], 'kmeans': []}
for idx in range(num_datasets_class):
    if idx != 8: #Skip Higgs due to large computational cost
        task_id = class_benchmark_suite.tasks[idx]
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        name = dataset.name
        name = name.replace("_", "\_")
        obs = dataset.qualities['NumberOfInstances']
        features = dataset.qualities['NumberOfFeatures']

        with open(f"benchmark_experiments/class_bench_{idx}_bins_255.json", "r") as f:
            results = json.load(f)
            linspace_dict = results['linspace']
            quantile_dict = results['quantile'] 
            kmeans_dict = results['kmeans']
            exact_dict = results['exact']

        kmeans_results = kmeans_dict['roc_auc']
        linspace_results = linspace_dict['roc_auc']
        quantile_results = quantile_dict['roc_auc']
        exact_results = exact_dict['roc_auc']

        kmeans_mean = np.mean(kmeans_results)
        linspace_mean = np.mean(linspace_results)
        quantile_mean = np.mean(quantile_results)
        exact_mean = np.mean(exact_results)
        
        if std_error:
            kmeans_std = np.std(kmeans_results) / np.sqrt(num_seeds)
            linspace_std = np.std(linspace_results) / np.sqrt(num_seeds)
            quantile_std = np.std(quantile_results) / np.sqrt(num_seeds)
            exact_std = np.std(exact_results) / np.sqrt(num_seeds)
        
        # Calculating MRR within this dataset -- DO NOT include exact here
        results = {'kmeans': kmeans_results, 'linspace': linspace_results, 'quantile': quantile_results}
        means = {method: np.mean(results[method]) for method in results}
        rank_list = list(means.values())
        ranks = {method: rank for method, rank in zip(means.keys(), rankdata(-1 * np.array(rank_list)))}
        
        # Get inverse ranks
        inv_ranks = {method: 1 / rank for method, rank in ranks.items()}
        
        for method in results.keys():
            mrr_class[method].append(inv_ranks[method])

        # ------------------------------------------------------------
        # Find best / second-best (lower MSE = better) and format cells
        # ------------------------------------------------------------
        sorted_methods = sorted(means, key=means.get, reverse = True)          # best first
        best, second = sorted_methods[:2]
        
        #Appending p-vals
        p_vals.append(ttest_rel(results[best], results[second]).pvalue)
        if std_error:
            latex_results.append({
                'name': name, 
                'dataset_type': 'Classification',
                'bins': 255,
                'model': 'SKL',
                'best': best,
                'quantile': quantile_mean,
                'quantile_std': quantile_std,
                'linspace': linspace_mean,
                'linspace_std': linspace_std,
                'kmeans': kmeans_mean,
                'kmeans_std': kmeans_std,
                'exact': exact_mean,
                'exact_std': exact_std,  
            })
        else:
            latex_results.append({
                'name': name, 
                'dataset_type': 'Classification',
                'bins': 255,
                'model': 'SKL',
                'best': best,
                'quantile': quantile_mean,
                'linspace': linspace_mean,
                'kmeans': kmeans_mean,
                'exact': exact_mean,  
            })
    
#Dictionary that we put our final rankings in
mrr_class_avg = {method: np.mean(mrr_class[method]) for method in ['quantile', 'linspace', 'kmeans']}

# MRR for Classification
print("%% Mean Reciprocal Rank (MRR) Table for Classification")
print("\\begin{table}[htbp]")
print("    \\centering")
print("    \\caption{Mean Reciprocal Rank (MRR) for each binning method computed within each baseline algorithm based on MSE performance for Regression datasets. Lower MSE yields a better (lower) rank, and the inverse rank is averaged across datasets.}")
print("    \\label{tab:mrr_class}")
print("    \\begin{tabular}{lccc}")
print("        \\toprule")
print("        Baseline & Quantile & Uniform & K-Means \\\\")
print("        \\midrule")
print(f"        \\textbf{{Classification}} & {mrr_class_avg['quantile']:.2f} & {mrr_class_avg['linspace']:.2f} & {mrr_class_avg['kmeans']:.2f} & \\\\")
print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")


#Regression
for n_bins in [63, 255]:
    mrr_reg = {'quantile': [], 'linspace': [], 'kmeans': []}
    print("REGRESSION RESULTS, BINS = ", n_bins)
    for idx in range(num_datasets_reg):
        task_id = reg_benchmark_suite.tasks[idx]
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        name = dataset.name
        name = name.replace("_", "\_")
        obs = dataset.qualities['NumberOfInstances']
        features = dataset.qualities['NumberOfFeatures']

        with open(f"benchmark_experiments/reg_bench_{idx}_bins_{n_bins}.json", "r") as f:
            results = json.load(f)
            linspace_dict = results['linspace']
            quantile_dict = results['quantile'] 
            kmeans_dict = results['kmeans']
            exact_dict = results['exact']

        kmeans_results = kmeans_dict['mse']
        linspace_results = linspace_dict['mse']
        quantile_results = quantile_dict['mse']
        exact_results = exact_dict['mse']

        kmeans_mean = np.mean(kmeans_results)
        #Find correct value for exponent for scientific notation
        exponent = int(np.floor(np.log10(kmeans_mean)))
        divisor = 10 ** exponent
        
        linspace_mean = np.mean(linspace_results)
        quantile_mean = np.mean(quantile_results)
        exact_mean = np.mean(exact_results)
        
        if std_error:
            kmeans_std = np.std(kmeans_results) / np.sqrt(num_seeds)
            linspace_std = np.std(linspace_results) / np.sqrt(num_seeds)
            quantile_std = np.std(quantile_results) / np.sqrt(num_seeds)
            exact_std = np.std(exact_results) / np.sqrt(num_seeds)
        
        # Calculating MRR within this dataset -- DO NOT include exact here
        results = {'kmeans': kmeans_results, 'linspace': linspace_results, 'quantile': quantile_results}
        means = {method: np.mean(results[method]) for method in results}
        
        ranks = {method: rank for method, rank in zip(means.keys(), rankdata(list(means.values())))}
        
        # Get inverse ranks
        inv_ranks = {method: 1 / rank for method, rank in ranks.items()}
        
        for method in results.keys():
            mrr_reg[method].append(inv_ranks[method])

        # ------------------------------------------------------------
        # Find best / second-best (lower MSE = better) and format cells
        # ------------------------------------------------------------
        sorted_methods = sorted(means, key=means.get)          # best first
        best, second = sorted_methods[:2]
        
        #Appending p-vals
        p_vals.append(ttest_rel(results[best], results[second]).pvalue)
        if std_error:
            latex_results.append({
                'name': name, 
                'dataset_type': 'Regression',
                'bins': n_bins,
                'model': 'SKL',
                'best': best,
                'quantile': quantile_mean,
                'quantile_std': quantile_std,
                'linspace': linspace_mean,
                'linspace_std': linspace_std,
                'kmeans': kmeans_mean,
                'kmeans_std': kmeans_std,
                'exact': exact_mean,
                'exact_std': exact_std,  
            })
        else:
            latex_results.append({
            'name': name, 
            'dataset_type': 'Regression',
            'bins': n_bins,
            'model': 'SKL',
            'best': best,
            'quantile': quantile_mean,
            'linspace': linspace_mean,
            'kmeans': kmeans_mean,
            'exact': exact_mean,  
            })
    #Dictionary that we put our final rankings in
    mrr_reg_avg = {method: np.mean(mrr_reg[method]) for method in ['quantile', 'linspace', 'kmeans']}


    # MRR for Regression
    print(f"%% Mean Reciprocal Rank (MRR) Table for Regression (Bins = {n_bins})")
    print("\\begin{table}[htbp]")
    print("    \\centering")
    print("    \\caption{Mean Reciprocal Rank (MRR) for each binning method computed within each baseline algorithm based on MSE performance for Regression datasets. Lower MSE yields a better (lower) rank, and the inverse rank is averaged across datasets.}")
    print("    \\label{tab:mrr_reg}")
    print("    \\begin{tabular}{lccc}")
    print("        \\toprule")
    print("        Baseline & Quantile & Uniform & K-Means \\\\")
    print("        \\midrule")
    print(f"        \\textbf{{Regression}} & {mrr_reg_avg['quantile']:.2f} & {mrr_reg_avg['linspace']:.2f} & {mrr_reg_avg['kmeans']:.2f} & \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")

#Results for the three XGBoost runs
#Regression
mrr_reg = {'quantile': [], 'linspace': [], 'kmeans': []}
print("REGRESSION RESULTS, XGB")
for idx in [0, 8, 15]:
    task_id = reg_benchmark_suite.tasks[idx]
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    name = dataset.name
    name = name.replace("_", "\_")
    obs = dataset.qualities['NumberOfInstances']
    features = dataset.qualities['NumberOfFeatures']

    with open(f"benchmark_experiments/reg_bench_{idx}_xgboost.json", "r") as f:
        results = json.load(f)
        linspace_dict = results['linspace']
        quantile_dict = results['quantile'] 
        kmeans_dict = results['kmeans']
        exact_dict = results['exact']

    kmeans_results = kmeans_dict['mse']
    linspace_results = linspace_dict['mse']
    quantile_results = quantile_dict['mse']
    exact_results = exact_dict['mse']

    kmeans_mean = np.mean(kmeans_results)
    #Find correct value for exponent for scientific notation
    exponent = int(np.floor(np.log10(kmeans_mean)))
    divisor = 10 ** exponent
    
    linspace_mean = np.mean(linspace_results)
    quantile_mean = np.mean(quantile_results)
    exact_mean = np.mean(exact_results)
    
    if std_error:
        kmeans_std = np.std(kmeans_results) / np.sqrt(num_seeds)
        linspace_std = np.std(linspace_results) / np.sqrt(num_seeds)
        quantile_std = np.std(quantile_results) / np.sqrt(num_seeds)
        exact_std = np.std(exact_results) / np.sqrt(num_seeds)
    
    results = {'kmeans': kmeans_results, 'linspace': linspace_results, 'quantile': quantile_results}
    means = {method: np.mean(results[method]) for method in results}
    for method in results.keys():
        mrr_reg[method].append(inv_ranks[method])

    # ------------------------------------------------------------
    # Find best / second-best (lower MSE = better) and format cells
    # ------------------------------------------------------------
    sorted_methods = sorted(means, key=means.get)          # best first
    best, second = sorted_methods[:2]
    
    #Appending p-vals
    p_vals.append(ttest_rel(results[best], results[second]).pvalue)
    if std_error:
        latex_results.append({
            'name': name, 
            'dataset_type': 'Regression',
            'bins': 255,
            'model': 'XGB',
            'best': best,
            'quantile': quantile_mean,
            'quantile_std': quantile_std,
            'linspace': linspace_mean,
            'linspace_std': linspace_std,
            'kmeans': kmeans_mean,
            'kmeans_std': kmeans_std,
            'exact': exact_mean,  
            'exact_std': exact_std,  
        })
    else:
        latex_results.append({
            'name': name, 
            'dataset_type': 'Regression',
            'bins': 255,
            'model': 'XGB',
            'best': best,
            'quantile': quantile_mean,
            'linspace': linspace_mean,
            'kmeans': kmeans_mean,
            'exact': exact_mean,  
        })


print("INDIVIDUAL TABULAR RESULTS")

#Working with adjusted p-values
p_vals = [1 if np.isnan(p) else p for p in p_vals]
ps_adjusted = false_discovery_control(p_vals)

for idx, adjusted_p in enumerate(ps_adjusted):
    if adjusted_p < 0.05:
        latex_results[idx]['pval'] = True
    else:
        latex_results[idx]['pval'] = False
        
curr_dataset_type = "Classification"
curr_bins = 255
curr_model = "SKL"
print(f"Results for {curr_dataset_type} with {curr_bins} bins and {curr_model} model")
print("--------------------------------")

for result in latex_results:
    if result['model'] != curr_model or result['bins'] != curr_bins or result['dataset_type'] != curr_dataset_type:
        curr_dataset_type = result['dataset_type']
        curr_bins = result['bins']
        curr_model = result['model']
        print(f"Results for {curr_dataset_type} with {curr_bins} bins and {curr_model} model")
        print("--------------------------------")

    fmt = {} #Holds strings to print
    
    #Formatting assuming classification
    if result['dataset_type'] == 'Classification':
        for m in ['quantile', 'linspace', 'kmeans']:
            if std_error:
                mean_val = result[m]
                se_val = result[f"{m}_std"]
                if m == result['best']:
                    fmt[m] = f"{star_if_sig(mean_val, result['pval'])} ({se_val:.3f})"
                else:
                    fmt[m] = f"{mean_val:.3f} ({se_val:.3f})"
            else:
                if m == result['best']:
                    fmt[m] = star_if_sig(result[m], result['pval'])
                else:
                    fmt[m] = f"{(result[m]):.3f}"
        if std_error:
            fmt['exact'] = f"{result['exact']:.3f} ({result['exact_std']:.3f})"
        else:
            fmt['exact'] = f"{result['exact']:.3f}"
        
        print(f"{result['name']} & {fmt['quantile']} & {fmt['linspace']} & {fmt['kmeans']} & {fmt['exact']} \\\\")
    else:
        #Need to deal with exponent
        exponent = int(np.floor(np.log10(result['quantile'])))
        divisor = 10 ** exponent
        for m in ['quantile', 'linspace', 'kmeans']:
            if std_error:
                mean_scaled = result[m] / divisor
                se_scaled = result[f"{m}_std"] / divisor
                if m == result['best']:
                    fmt[m] = f"{star_if_sig(mean_scaled, result['pval'])} ({se_scaled:.3f})"
                else:
                    fmt[m] = f"{mean_scaled:.3f} ({se_scaled:.3f})"
            else:
                if m == result['best']:
                    fmt[m] = star_if_sig(result[m] / divisor, result['pval'])
                else:
                    fmt[m] = f"{(result[m] / divisor):.3f}"
        if std_error:
            fmt['exact'] = f"{result['exact'] / divisor:.3f} ({result['exact_std'] / divisor:.3f})"
        else:
            fmt['exact'] = f"{result['exact'] / divisor:.3f}"
        
        print(f"{result['name']} $(10^{{{exponent}}})$ & {fmt['quantile']} & {fmt['linspace']} & {fmt['kmeans']} & {fmt['exact']} \\\\")    
        
#Checking statistical significance for mrr
print("Statistical significance for MRR, Reg")
sorted_mrr_reg = sorted(mrr_reg_avg, key = mrr_reg_avg.get)
best, second = sorted_mrr_reg[-2:]
best_results, second_results = mrr_reg[best], mrr_reg[second]
p_val_reg = wilcoxon(best_results, second_results).pvalue
print(f"Best: {best}, Second: {second}, p-value: {p_val_reg}")

#Checking statistical significance for mrr
print("Statistical significance for MRR, Class")
sorted_mrr_class = sorted(mrr_class_avg, key = mrr_class_avg.get)
best, second = sorted_mrr_class[-2:]
best_results, second_results = mrr_class[best], mrr_class[second]
p_val_class = ttest_rel(best_results, second_results).pvalue
print(f"Best: {best}, Second: {second}, p-value: {p_val_class}")