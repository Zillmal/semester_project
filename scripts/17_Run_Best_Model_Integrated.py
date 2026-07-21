import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "scripts"))


import json
import pandas as pd

from NN_Cox_Integrated import run_cv


OUT_DIR = PROJECT_ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


BEST_CONFIG = {
    "learning_rate": 0.001,       
    "weight_decay": 0.03,          
    "num_nodes": [8],               
    "dropout": 0.5,                 
    "batch_norm": False,
    "meth_variance_threshold": 0.0005,
    "patience": 10,
    "batch_size": 32,
    "evaluate_on_test": True
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
    print("Running final integrated NN-Cox model with best configuration:")
    print(BEST_CONFIG)

    cv = run_cv(**BEST_CONFIG)

    summary = summarize_cv(cv, BEST_CONFIG)

    cv.to_csv(
        OUT_DIR / "nn_integrated_best_model_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR / "nn_integrated_best_model_summary.csv",
        index=False,
    )

    print("\nFold-level results:")
    print(cv.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()