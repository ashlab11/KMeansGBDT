# Data and code for "Drop-in k-means binning for gradient-boosted trees"

In this paper, we challenge the long-standing assumption that equal-frequency bins provide adequate accuracy for GBDT histogram binning. We find that k-means often outperforms quantile binning in highly-skewed datasets or when using a low bin budget.

---

## Table of Contents
1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
   * [Using the Master Executable](#using-the-master-executable)
   * [Running a Single Experiment Script](#running-a-single-experiment-script)
6. [Results & Logs](#results--logs)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## Overview
Explain **what** the experiments investigate (e.g. _“Comparing four feature-binning strategies for gradient-boosted regressors on OpenML benchmark suites”_).  
Include a short diagram or bullet list if helpful.

---

## Folder Structure
```text
.

├── scripts/ #Where experiments live
├── benchmark_experiments/ #Where real-world experiments live
├── saved_synthetic_results/ #Where synthetic experiment results live
├── src/ # reusable modules (e.g. DataBinner)
├── run_all.py #Compiled executable
└── README.md
```

## Installation
Follow these instructions to install all packages needed for this project.
```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # ← everything you need is pinned here
```

## Quick Start
### Using the Master Executable
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