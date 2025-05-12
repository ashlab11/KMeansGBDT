# Data and code for "A Case for Library-Level k-Means Binning in Histogram Gradient-Boosted Trees"

In this paper, we challenge the long-standing assumption that equal-frequency bins provide adequate accuracy for GBDT histogram binning. We find that k-means often outperforms quantile binning in highly-skewed datasets or when using a low bin budget.

---

## Table of Contents
1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
   * [Running the Master Experiment](#running-the-master-experiment)
   * [Running a Single Experiment Script](#running-a-single-experiment-script)
6. [Results & Logs](#results--logs)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## Overview
This repo contains the **exact code and data pipeline** used in our NeurIPS submission on feature-binning for gradient-boosted trees.

* **What it does** – Compares four binning schemes (`kmeans`, `quantile`, `linspace`, `exact`) on OpenML regression tasks, measuring MSE & ROC_AUC across multiple random seeds.
* **How to run** – Either call the master file `run_all.py` for a full sweep or launch one of the experiments in `scripts` with custom CLI flags.
* **Outputs** – Experiments save JSON summaries, numpy arrays, and images, and print out publication-ready LaTeX with a single command.

Licensed under **Apache 2.0** for unrestricted use and extension.


---

## Folder Structure
```text
.

├── scripts/ #Where experiments live
├── benchmark_experiments/ #Where real-world experiments live
├── saved_synthetic_results/ #Where synthetic experiment results live
├── src/ # reusable modules (e.g. DataBinner)
├── run_all.py #Python code to run all experiments
└── README.md
```

## Installation (Needs **Python ≥ 3.11**)
Follow these instructions to install all packages needed for this project. Begin by downloading this repository/supplementary ZIP. Then, run the following:
```bash
cd KMeansGBDT
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # ← everything you need is pinned here
pip install -e . # Installs this project's source code
```

## Quick Start
### Running the Master Experiment
To run the full sweep in one run, just call 

```python3
python3 run_all.py
```

This will put benchmark results in the [benchmark_experiments](benchmark_experiments) folder, put images in the [images](images) folder, and print out the same LaTeX used for our tables.

### Running a Single Experiment Script
You may instead wish to just run one single experiment (especially if you are compute-constrainted). 

**Synthetic Suite**

To re-run the synthetic experiments, call:

```python3
python3 scripts/construct_modality_skew_analysis.py
```

Optional flags:
| Flag           | Default | Description                                                              | Example        |
| -------------- | ------- | ------------------------------------------------------------------------ | -------------- |
| `--seeds`      | `50`    | Number of random train/test splits to execute                            | `--seeds 10`   |
| `--use_cached` | *off*   | If present, bypasses training and generates images from existing results | `--use_cached` |

**Benchmark Experiments**

To re-run real-world benchmarks, call:
```python3
python3 scripts/openml_experiment_reg.py
```

(If you'd like to call the classifcation experiments instead, replace reg with class).

Optional flags:
| Flag           | Default | Description                                                              | Example        |
| -------------- | ------- | ------------------------------------------------------------------------ | -------------- |
| `--num_seeds`      | `20`    | Number of random train/test splits to execute                            | `--seeds 10`   |
| `--benchmark_id` | 0   | Which dataset to run on (reference dataset description table in paper) | `--benchmark_id 4` |
| `--n_bins` | 255   | How many bins to discretize continuous features into | `--n_bins 63` |

## Results and Logs
**Benchmark results** can be found in [benchmark_experiments](benchmark_experiments), in format 

benchmark_experiments/reg_bench_{benchmark_id}\_bins\_{n_bins}.json

Replace reg with class to see classification experiments.

**Synthetic experiments** have their final graphs in [images](images). Saved numpy arrays can be found in [saved_synthetic_results](saved_synthetic_results)


## Contributing
Fork the repo and create your branch:
```bash 
git checkout -b foo
```
Commit changes
```bash
git commit -m "Changes"
```

Push to the branch
```bash
git push origin foo
```

Open a pull request

## License
Released under the Apache License 2.0.  
See [LICENSE](./LICENSE) for full details.

## Contact
Asher Labovich - asher_labovich@brown.edu