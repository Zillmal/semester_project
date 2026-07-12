import sys
sys.path.append("../scripts") 

from itertools import product
from pathlib import Path
import pandas as pd
import json
from NN_Cox_Integrated import run_cv

OUT_DIR = Path("../results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def summarize_cv(cv, config, config_id, stage):
    return {
        "stage": stage,
        "config_id": config_id,
        **config,
        "num_nodes": json.dumps(config["num_nodes"]),
        "mean_train_c_index": cv["train_c_index"].mean(),
        "sd_train_c_index": cv["train_c_index"].std(),
        "mean_val_c_index": cv["val_c_index"].mean(),
        "sd_val_c_index": cv["val_c_index"].std(),
        "mean_epochs_trained": cv["epochs_trained"].mean(),
    }


def choose_best_config(summary):
    best_row = summary.loc[summary["mean_val_c_index"].idxmax()]
    cutoff = best_row["mean_val_c_index"] - best_row["sd_val_c_index"]

    candidates = summary[summary["mean_val_c_index"] >= cutoff].copy()

    candidates["model_size"] = candidates["num_nodes"].apply(
        lambda x: sum(json.loads(x))
    )

    candidates["overfit_gap"] = (
        candidates["mean_train_c_index"] - candidates["mean_val_c_index"]
    )

    candidates = candidates.sort_values(
        by=[
            "model_size",
            "overfit_gap",
            "sd_val_c_index",
            "mean_val_c_index",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    )

    return candidates.iloc[0]


def run_tuning(grid, param_names):
    summary_rows = []
    fold_rows = []

    fixed_config = {
        "batch_norm": False,
        "batch_size": 32,
        "patience": 10,
        "meth_variance_threshold": 0.0005,
    }

    stage_name = "conservative_joint_tuning"

    for config_id, values in enumerate(grid, start=1):
        config = fixed_config.copy()
        config.update(dict(zip(param_names, values)))

        print(f"\n[{stage_name}] Running config {config_id}/{len(grid)}")
        print(config)

        cv = run_cv(**config, evaluate_on_test=False)

        cv = cv.copy()
        cv["stage"] = stage_name
        cv["config_id"] = config_id

        for key, value in config.items():
            cv[key] = json.dumps(value) if isinstance(value, list) else value

        fold_rows.append(cv)
        summary_rows.append(summarize_cv(cv, config, config_id, stage_name))

    summary = pd.DataFrame(summary_rows)

    summary["overfit_gap"] = (
        summary["mean_train_c_index"] - summary["mean_val_c_index"]
    )

    summary = (
        summary
        .sort_values(by="mean_val_c_index", ascending=False)
        .reset_index(drop=True)
    )

    folds = pd.concat(fold_rows, ignore_index=True)

    best_config = choose_best_config(summary).to_dict()

    return summary, folds, best_config


grid = list(product(
    [1e-4, 3e-4, 1e-3],          # learning_rate
    [1e-3, 1e-2, 3e-2],          # weight_decay
    [[4], [8], [16], [8, 4]],    # num_nodes
    [0.4, 0.5, 0.6],             # dropout
))

summary, folds, best_config = run_tuning(
    grid=grid,
    param_names=[
        "learning_rate",
        "weight_decay",
        "num_nodes",
        "dropout",
    ],
)

summary.to_csv(
    OUT_DIR / "nn_integrated_conservative_joint_tuning_summary.csv",
    index=False,
)

folds.to_csv(
    OUT_DIR / "nn_integrated_conservative_joint_tuning_folds.csv",
    index=False,
)

print("\nTop configs by validation C-index:")
print(summary.head(10).to_string(index=False))

print("\nSelected config using conservative 1-SD rule:")
print(best_config)
