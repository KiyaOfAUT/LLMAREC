import json
import os
import datetime
from lightgcn import run_experiment as run_lightgcn
from hybrid_rec.train import main as run_hybrid

def log_results(model_name, metrics, filename="benchmark_results.log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"\n{'='*20} {model_name} ({timestamp}) {'='*20}\n")
        f.write(json.dumps(metrics, indent=4))
        f.write("\n")

def main():
    results_file = "benchmark_results.log"
    if os.path.exists(results_file):
        os.remove(results_file)
        
    print("Starting Benchmark...")
    
    # 1. Run LightGCN
    print("\nRunning LightGCN...")
    try:
        lightgcn_metrics = run_lightgcn()
        log_results("LightGCN", lightgcn_metrics, results_file)
        print("LightGCN completed and results recorded.")
    except Exception as e:
        print(f"Error running LightGCN: {e}")

    # 2. Run Hybrid Model
    print("\nRunning Hybrid Model...")
    try:
        hybrid_metrics = run_hybrid()
        log_results("Hybrid Model", hybrid_metrics, results_file)
        print("Hybrid Model completed and results recorded.")
    except Exception as e:
        print(f"Error running Hybrid Model: {e}")

    print(f"\nBenchmark finished. Results saved to {results_file}")

if __name__ == "__main__":
    main()
