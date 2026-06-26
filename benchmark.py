import argparse
import json
import os
import datetime
import traceback

from lightgcn import run_experiment as run_lightgcn
from hybrid_rec.train import run_experiment as run_rlmrec


def log_results(model_name, metrics, filename="benchmark_results.log", meta=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"\n{'='*20} {model_name} ({timestamp}) {'='*20}\n")
        if meta:
            f.write(json.dumps(meta, indent=4))
            f.write("\n")
        f.write(json.dumps(metrics, indent=4))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LightGCN vs RLMRec")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for both models (default: 30)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="benchmark_results.log",
        help="Output log file path",
    )
    args = parser.parse_args()

    results_file = args.log
    if os.path.exists(results_file):
        os.remove(results_file)

    run_meta = {"epochs": args.epochs}
    print(f"Starting benchmark (epochs={args.epochs})...")

    print("\nRunning LightGCN...")
    try:
        lightgcn_metrics = run_lightgcn(n_epochs=args.epochs)
        log_results("LightGCN", lightgcn_metrics, results_file, run_meta)
        print("LightGCN completed and results recorded.")
    except Exception as e:
        print(f"Error running LightGCN: {e}")
        traceback.print_exc()

    print("\nRunning RLMRec...")
    try:
        rlmrec_metrics = run_rlmrec(n_epochs=args.epochs)
        log_results("RLMRec", rlmrec_metrics, results_file, run_meta)
        print("RLMRec completed and results recorded.")
    except Exception as e:
        print(f"Error running RLMRec: {e}")
        traceback.print_exc()

    print(f"\nBenchmark finished. Results saved to {results_file}")


if __name__ == "__main__":
    main()
