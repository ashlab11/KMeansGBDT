import json
import openml
from scipy.stats import ttest_rel, rankdata
import lightgbm as lgb
from src import DataBinner
import numpy as np
import argparse

num_datasets_class = 16
num_datasets_reg = 18

#Keywords that will be used to analyze experiment results
def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Run one regression-binning experiment")
    parser.add_argument("--num_seeds",   type=int, default=20)
    return parser.parse_args()

args = get_cli_args()
num_seeds   = args.num_seeds

reg_benchmark_suite = openml.study.get_suite(336)
class_benchmark_suite = openml.study.get_suite(337) 

#We want to be ranking across datasets within each model type
    
def bold_if_sig(best_val, best_arr, second_val, second_arr):
    """Return '{best_val:.5f}' or '\\textbf{...}' depending on p-value."""
    p_val = ttest_rel(best_arr, second_arr).pvalue
    formatted = f"{best_val:.3f}"

    if p_val < 0.05:
        return f"{formatted}$\\mathrlap{{^{{**}}}}$"
    elif p_val < 0.1:
        return f"{formatted}$\\mathrlap{{^{{*}}}}$"
    else:
        return formatted


#Classification first
print("DATA FOR CLASSIFICATION")
mrr_class = {'quantile': [], 'linspace': [], 'kmeans': []}
for idx in range(num_datasets_class):
    if idx not in [2, 8]: #Skip Higgs due to large computational cost
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

        fmt = {}                                              # holds strings to print
        for m in ['quantile', 'linspace', 'kmeans']:
            if m == best:                                     # winner – maybe bold
                fmt[m] = bold_if_sig(means[m],
                                    results[m],              # arrays for t-test
                                    means[second],
                                    results[second])
            else:                                             # plain number
                fmt[m] = f"{(means[m]):.3f}"
                
        print(f"{name} & {fmt['quantile']} & {fmt['linspace']} & {fmt['kmeans']} & {exact_mean:.3f} \\\\")

    
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
    print("REGRESSION RESULTS, Bins = ", n_bins)
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

        fmt = {}                                              # holds strings to print
        for m in ['quantile', 'linspace', 'kmeans']:
            if m == best:                                     # winner – maybe bold
                fmt[m] = bold_if_sig(means[m] / divisor,
                                    results[m],              # arrays for t-test
                                    means[second] / divisor,
                                    results[second])
            else:                                             # plain number
                fmt[m] = f"{(means[m] / divisor):.3f}"
                
        print(f"{name} $(10^{{{exponent}}})$ & {fmt['quantile']} & {fmt['linspace']} & {fmt['kmeans']} & {(exact_mean / divisor) :.3f} \\\\")
        
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

