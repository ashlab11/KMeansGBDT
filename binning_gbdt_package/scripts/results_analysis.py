import json
import openml
from scipy.stats import ttest_rel, rankdata
import lightgbm as lgb
from src import DataBinner
import numpy as np

num_datasets_class = 16
num_datasets_reg = 18

num_seeds = 20
model = 'SKL'

reg_benchmark_suite = openml.study.get_suite(336)
class_benchmark_suite = openml.study.get_suite(337)  # 337 for classification

#We want to be ranking across datasets within each model type
mrr_reg = {'quantile': [], 'linspace': [], 'kmeans': []}
mrr_class = {'quantile': [], 'linspace': [], 'kmeans': []}
mrr_total = {'quantile': [], 'linspace': [], 'kmeans': []}
    
def bold_if_sig(best_val, best_arr, second_val, second_arr):
    """Return '{best_val:.5f}' or '\\textbf{...}' depending on p-value."""
    p_val = ttest_rel(best_arr, second_arr).pvalue
    formatted = f"{best_val:.3f}"
    return f"\\textbf{{{formatted}}}" if p_val < 0.05 else formatted

#Classification first
print("DATA FOR CLASSIFICATION")
for idx in range(num_datasets_class):
    if idx in [0, 1, 3, 4, 6]:
        task_id = class_benchmark_suite.tasks[idx]
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        name = dataset.name
        name = name.replace("_", "\_")
        obs = dataset.qualities['NumberOfInstances']
        features = dataset.qualities['NumberOfFeatures']

        with open(f"class_results_binning_{idx}.json", "r") as f:
            results = json.load(f)
            linspace_dict = results['linspace']
            quantile_dict = results['quantile'] 
            kmeans_dict = results['kmeans']
            exact_dict = results['exact']

        kmeans_results = kmeans_dict[model]['accuracy']
        linspace_results = linspace_dict[model]['accuracy']
        quantile_results = quantile_dict[model]['accuracy']
        exact_results = exact_dict[model]['accuracy']

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
            mrr_total[method].append(inv_ranks[method])

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
print("    \\label{tab:mrr_reg}")
print("    \\begin{tabular}{lccc}")
print("        \\toprule")
print("        Baseline & Quantile & Uniform & K-Means \\\\")
print("        \\midrule")
print(f"        \\textbf{{Classification}} & {mrr_class_avg['quantile']:.2f} & {mrr_class_avg['linspace']:.2f} & {mrr_class_avg['kmeans']:.2f} & \\\\")
print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")


#Regression
print("DATA FOR REGRESSION")
for idx in range(num_datasets_reg):
    task_id = reg_benchmark_suite.tasks[idx]
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    name = dataset.name
    name = name.replace("_", "\_")
    obs = dataset.qualities['NumberOfInstances']
    features = dataset.qualities['NumberOfFeatures']

    with open(f"reg_results_binning_{idx}.json", "r") as f:
        results = json.load(f)
        linspace_dict = results['linspace']
        quantile_dict = results['quantile'] 
        kmeans_dict = results['kmeans']
        exact_dict = results['exact']

    kmeans_results = kmeans_dict[model]['mse']
    linspace_results = linspace_dict[model]['mse']
    quantile_results = quantile_dict[model]['mse']
    exact_results = exact_dict[model]['mse']

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
        mrr_class[method].append(inv_ranks[method])
        mrr_total[method].append(inv_ranks[method])

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
mrr_reg_avg = {method: np.mean(mrr_class[method]) for method in ['quantile', 'linspace', 'kmeans']}


# MRR for Regression
print("%% Mean Reciprocal Rank (MRR) Table for Regression")
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

#TOTAL!
#Dictionary that we put our final rankings in
mrr_total_avg = {method: np.mean(mrr_total[method]) for method in ['quantile', 'linspace', 'kmeans']}
print("TOTAL DATA")
# MRR for Regression
print("%% Mean Reciprocal Rank (MRR) Table for All")
print("\\begin{table}[htbp]")
print("    \\centering")
print("    \\caption{Mean Reciprocal Rank (MRR) for each binning method over all datasets}")
print("    \\label{tab:mrr_reg}")
print("    \\begin{tabular}{lccc}")
print("        \\toprule")
print("        Baseline & Quantile & Uniform & K-Means \\\\")
print("        \\midrule")
print(f"        \\textbf{{Total}} & {mrr_total_avg['quantile']:.2f} & {mrr_total_avg['linspace']:.2f} & {mrr_total_avg['kmeans']:.2f} & \\\\")
print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")


