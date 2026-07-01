import sys
sys.path.append("../scripts")

from pathlib import Path
import json
import pandas as pd

from NN_Cox_mRNA_Expression import run_cv

OUT_DIR = Path("../results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CONFIG = {
    "learning_rate": 0.001,       
    "weight_decay": 0.001,          
    "num_nodes": [16],               
    "dropout": 0.6,                 
    "batch_norm": False,
    "patience": 10,
    "batch_size": 32,
}

def summarize_cv(cv, config):
    return pd.DataFrame([{
        **config,
        "num_nodes": json.dumps(config["num_nodes"]),
        "mean_train_c_index": cv["train_c_index"].mean(),
        "sd_train_c_index": cv["train_c_index"].std(),
        "mean_val_c_index": cv["val_c_index"].mean(),
        "sd_val_c_index": cv["val_c_index"].std(),
        "mean_test_c_index": cv["test_c_index"].mean(),
        "sd_test_c_index": cv["test_c_index"].std(),
        "mean_epochs_trained": cv["epochs_trained"].mean(),
    }])

def main():
    print("Running final mRNA Only NN-Cox model with best configuration:")
    print(BEST_CONFIG)

    cv = run_cv(**BEST_CONFIG)

    summary = summarize_cv(cv, BEST_CONFIG)

    cv.to_csv(
        OUT_DIR / "nn_mRNA_only_best_model_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR / "nn_mRNA_only_best_model_summary.csv",
        index=False,
    )

    print("\nFold-level results:")
    print(cv.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()