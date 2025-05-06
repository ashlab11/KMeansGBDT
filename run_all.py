#!/usr/bin/env python3
#Runs all scripts necessary for this project
import subprocess
import sys
from pathlib import Path

def run_script(script_name, **kwargs):
    """Run a Python script using subprocess."""
    script_path = Path(script_name).resolve()
    print(f"Running {script_path}...")
    #Run with arguments
    result = subprocess.run(
        [sys.executable, str(script_path), 
         *[f"--{k}={v}" for k, v in kwargs.items()]], 
        capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")
    else:
        print(f"Output of {script_name}:\n{result.stdout}")
        
        
if __name__ == "__main__":
    #Running timing script
    run_script("scripts/construct_computation_test.py")
    
    #Running synthetic experiments
    run_script("scripts/construct_modality_skew_analysis.py")
    
    #Running OpenML experiment for regression with 63/255 bins
    for n_bins in [63, 255]:
        for benchmark_id in range(17):
            run_script("scripts/openmlexperiment_reg.py", 
                       n_bins = n_bins,
                       benchmark_id=benchmark_id)
    #Running OpenML experiment for classification
    for benchmark_id in range(15):
        if benchmark_id == 8: #Skip Higgs for computational purposes
            continue
        run_script("scripts/openmlexperiment_class.py", 
                   benchmark_id=benchmark_id)
        
    #Running results analysis
    run_script("scripts/results_analysis.py", num_seeds=20)