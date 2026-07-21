import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from pycox.models import CoxPH
warnings.simplefilter("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# network architecture


MAX_EPOCHS    = 256      
VAL_FRACTION  = 0.20 
OUTPUT_BIAS = False 

RANDOM_STATE = 42        

def load_data(data_dir=DATA_DIR, random_state=RANDOM_STATE):
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # import data and make sure each patient is existent in each data frame.

    rna   = pd.read_csv(data_dir / "rna_pam50.csv").set_index("patient")
    surv  = pd.read_csv(data_dir / "survival_luminal_clean.csv").set_index("patient")
    folds = pd.read_csv(data_dir / "cv_fold_assignments.csv").set_index("patient")


    # make sure each patient has a valid follow-up survival time

    surv = surv[surv["time"].notna() & (surv["time"] > 0)]
    patients = rna.index.intersection(surv.index).intersection(folds.index)

    rna     = rna.loc[patients] 
    surv    = surv.loc[patients]
    fold_id = folds.loc[patients, "fold"]

    GENES = list(rna.columns)
    print(f"Overview of Input Data: \n\nPatients: {len(patients)} | genes: {len(GENES)} | "
        f"events: {int(surv['event'].sum())} | folds: {sorted(fold_id.unique())}\n")
    
    return rna, surv, fold_id

rna, surv, fold_id = load_data()


# Define helper functions: 

def make_xy(ids, scaler):
    # Scale features and build the pycox label.
    x = scaler.transform(rna.loc[ids]).astype("float32")
    y = (surv.loc[ids, "time"].values.astype("float32"),
         surv.loc[ids, "event"].values.astype("float32"))
    return x, y


def build_net(
        in_features,
        num_nodes,
        dropout,
        batch_norm):
    # Build the MLP risk network
    layers = []
    prev_units = in_features
    for n_units in num_nodes:
        layers.append(nn.Linear(prev_units, n_units))   
        layers.append(nn.ReLU())                         
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units))      
        if dropout > 0:
            layers.append(nn.Dropout(dropout))           
        prev_units = n_units                             
    layers.append(nn.Linear(prev_units, 1, bias=OUTPUT_BIAS))
    return nn.Sequential(*layers)


def build_model(in_features, 
                learning_rate, 
                weight_decay, 
                num_nodes, 
                dropout, 
                batch_norm):
    # DeepSurv = risk network + Cox partial-likelihood loss (CoxPH).
    net = build_net(in_features, 
                    num_nodes=num_nodes, 
                    dropout=dropout, 
                    batch_norm=batch_norm)
    optimizer = tt.optim.Adam(lr=learning_rate, weight_decay=weight_decay)
    model = CoxPH(net, optimizer)
    return model


def c_index(model, x, durations, events):
    # C-index on the predicted log-risk
    risk = model.predict(x).ravel()
    result = concordance_index_censored(events.astype(bool), durations, risk)
    return result[0] 



# cross validated training & evaluation

def run_cv(learning_rate, 
           weight_decay, 
           num_nodes, 
           dropout, 
           batch_norm,
           patience, 
           batch_size,
           evaluate_on_test=False):
    
    rows = []

    # reset risk-score file once per run to avoid duplicate-fold appends
    risk_path = PROJECT_ROOT / "results" / "tables" / "nn_mrna_only_risk_scores.csv"
    if evaluate_on_test:
        risk_path.unlink(missing_ok=True)

    for f in sorted(fold_id.unique()):
        # Outer split: test set.
        train_val_ids = fold_id.index[fold_id != f]
        test_ids      = fold_id.index[fold_id == f]

        # Create internal validation set out of the training patients for early
        # stopping only (stratified on event so both splits keep some deaths).
        tr_ids, val_ids = train_test_split(
            train_val_ids, test_size=VAL_FRACTION, random_state=RANDOM_STATE,
            stratify=surv.loc[train_val_ids, "event"])

        # Fold-safe scaling: fit to training patients only.
        scaler = StandardScaler().fit(rna.loc[tr_ids])
        x_tr,  y_tr  = make_xy(tr_ids,  scaler)
        x_val, y_val = make_xy(val_ids, scaler)
        x_te,  y_te  = make_xy(test_ids, scaler)

        durations_tr, events_tr = y_tr
        durations_val, events_val = y_val
        durations_te, events_te = y_te 

        # Build and train DeepSurv with early stopping on the validation loss.
        torch.manual_seed(RANDOM_STATE)         
        model = build_model(x_tr.shape[1], 
                            learning_rate, 
                            weight_decay, 
                            num_nodes, 
                            dropout, 
                            batch_norm)
        log = model.fit(
            x_tr, y_tr,
            batch_size=batch_size, epochs=MAX_EPOCHS,
            callbacks=[tt.callbacks.EarlyStopping(patience=patience)],
            val_data=(x_val, y_val), val_batch_size=batch_size, verbose=False)

        # Evaluate on the held-out test fold with C-index.
        train_c_index = c_index(model, x_tr, durations_tr, events_tr)
        val_c_index = c_index(model, x_val, durations_val, events_val)
        if evaluate_on_test:
            test_c_index = c_index(model, x_te, durations_te, events_te)
        else:
            test_c_index = np.nan

        epochs_trained = log.epoch + 1          

        # Save risk scores
        if evaluate_on_test:
            risk_scores = model.predict(x_te).ravel()
            risk_path = PROJECT_ROOT / "results" / "tables" / "nn_mrna_only_risk_scores.csv"
            pd.DataFrame([{"patient": pid, "fold": f, "risk_score": float(r)}
                for pid, r in zip(test_ids, risk_scores)]).to_csv(
            risk_path, mode="a", header=not risk_path.exists(), index=False)
        
        rows.append({"fold": f, 
                     "n_test": len(test_ids),
                     "epochs_trained": epochs_trained, 
                     "learning_rate": learning_rate,
                     "weight_decay": weight_decay,
                     "num_nodes": num_nodes,
                     "dropout": dropout,
                     "batch_norm": batch_norm,
                     "batch_size": batch_size,
                     "patience": patience,
                     "train_c_index": train_c_index,
                     "val_c_index": val_c_index,
                     "test_c_index": test_c_index})
        
    return pd.DataFrame(rows)
        


# summarize results 

def summarize_cv_results(cv):
    print(cv.to_string(index=False))

    train_mean, train_sd = cv["train_c_index"].mean(), cv["train_c_index"].std()
    val_mean, val_sd = cv["val_c_index"].mean(), cv["val_c_index"].std()
    test_mean, test_sd = cv["test_c_index"].mean(), cv["test_c_index"].std()

    print(f"\nMean C-index of NN-Cox mRNA Expression only across 5 folds:\n"
          f"Train: {train_mean:.3f} +/- {train_sd:.3f}\n"
          f"Val:   {val_mean:.3f} +/- {val_sd:.3f}")
    if "test_c_index" in cv.columns and cv["test_c_index"].notna().any():
        test_mean_ci = cv["test_c_index"].mean()
        test_sd_ci = cv["test_c_index"].std()
        print(f"Test:  {test_mean_ci:.3f} +/- {test_sd_ci:.3f}")
    else:
        print("Test: not evaluated")
    
def save_cv_results(cv, filename="nn_cox_mrna_cv_results.csv"):
    out_dir = Path("../results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out_dir / filename, index=False)
    print(f"Saved CV results to {out_dir / filename}")

if __name__ == "__main__":
    print("torch", torch.__version__)

    config = {
        "learning_rate": 0.01,
        "weight_decay": 0.10,
        "num_nodes": [32, 16],
        "dropout": 0.4,
        "batch_norm": True,
        "patience": 15,
        "batch_size": 64,
        "evaluate_on_test": True
    }

    cv = run_cv(**config)
    summarize_cv_results(cv)
    save_cv_results(cv)