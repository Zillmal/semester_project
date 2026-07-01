import sys
sys.path.append("../scripts") # For KNN helper function. 

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
from KNN_Imputation_Helper_Function import fit_transform_train_test_methylation

warnings.simplefilter("ignore")


# network architecture    

MAX_EPOCHS    = 256    
VAL_FRACTION  = 0.20   

KNN_NEIGHBORS = 10       
KNN_WEIGHTS   = "distance"

METH_MISSING_THRESHOLD = 0.3
OUTPUT_BIAS = False

RANDOM_STATE = 42    

DATA_DIR = Path("../data/processed")  




# import data and make sure each patient is existent in each data frame.

def load_data(data_dir=DATA_DIR, random_state=RANDOM_STATE):
    np.random.seed(random_state)
    torch.manual_seed(random_state)


    rna   = pd.read_csv(data_dir / "rna_pam50.csv").set_index("patient")
    meth  = pd.read_csv(data_dir / "meth_pam50.csv").set_index("patient")   
    surv  = pd.read_csv(data_dir / "survival_luminal_clean.csv").set_index("patient")
    folds = pd.read_csv(data_dir / "cv_fold_assignments.csv").set_index("patient")

    # make sure each patient has a valid follow-up survival time

    surv = surv[surv["time"].notna() & (surv["time"] > 0)]

    patients = (
        rna.index
        .intersection(meth.index)
        .intersection(surv.index)
        .intersection(folds.index))

    rna     = rna.loc[patients]
    meth    = meth.loc[patients]
    surv    = surv.loc[patients]
    fold_id = folds.loc[patients, "fold"]

    print(f"Overview of Input Data: \n\n"
          f"Patients: {len(patients)} | genes: {rna.shape[1]} | "
          f"raw CpGs: {meth.shape[1]} | events: {int(surv['event'].sum())} | "
          f"folds: {sorted(fold_id.unique())}")

    return rna, meth, surv, fold_id

rna, meth, surv, fold_id = load_data()




# Define helper functions:


def make_y(ids):
    # pycox label: (durations, events) as float32 arrays.
    return (surv.loc[ids, "time"].values.astype("float32"),
            surv.loc[ids, "event"].values.astype("float32"))


def beta_to_m(B):
    # Convert methylation beta values (0-1) to M-values: log2(beta / (1 - beta)).
    # clip from 0/1 so the log stays finite.
    B = B.clip(1e-4, 1 - 1e-4)
    return np.log2(B / (1 - B))


def build_features(train_ids, val_ids, test_ids, meth_variance_threshold):
    meth_train_raw = meth.loc[train_ids]

    missing_rate = meth_train_raw.isna().mean()
    variance = meth_train_raw.var(skipna=True)

    keep_cpgs = (
        (missing_rate <= METH_MISSING_THRESHOLD) &
        (variance >= meth_variance_threshold)
    )
    selected_cpgs = meth_train_raw.columns[keep_cpgs]

    meth_filtered = meth[selected_cpgs]
    # Fold-safe feature matrix (expression + methylation); everything fitted on train only.

    # Methylation: KNN-impute (fit on train), apply to val and test.
    # The helper fits the imputer on its first arguments training ids. calling it twice
    # with the same train_ids reuses that fit and transforms val and test separately.
    Bm_tr, Bm_te,  _ = fit_transform_train_test_methylation(
        meth_filtered, 
        train_ids, 
        test_ids, 
        n_neighbors=KNN_NEIGHBORS, 
        weights=KNN_WEIGHTS, 
        scale=False)
    
    _,     Bm_val, _ = fit_transform_train_test_methylation(
        meth_filtered, 
        train_ids, 
        val_ids,  
        n_neighbors=KNN_NEIGHBORS, 
        weights=KNN_WEIGHTS, 
        scale=False)

    # convert beta to M-values, then standardise the M-values.
    Mm_tr, Mm_val, Mm_te = beta_to_m(Bm_tr), beta_to_m(Bm_val), beta_to_m(Bm_te)
    m_scaler = StandardScaler().fit(Mm_tr)
    Mm_tr  = pd.DataFrame(m_scaler.transform(Mm_tr),  index=Bm_tr.index,  columns=Mm_tr.columns)
    Mm_val = pd.DataFrame(m_scaler.transform(Mm_val), index=Bm_val.index, columns=Mm_val.columns)
    Mm_te  = pd.DataFrame(m_scaler.transform(Mm_te),  index=Bm_te.index,  columns=Mm_te.columns)

    # Expression: standardise
    r_scaler = StandardScaler().fit(rna.loc[train_ids])
    Xr_tr  = pd.DataFrame(r_scaler.transform(rna.loc[train_ids]), index=train_ids, columns=rna.columns)
    Xr_val = pd.DataFrame(r_scaler.transform(rna.loc[val_ids]),   index=val_ids,   columns=rna.columns)
    Xr_te  = pd.DataFrame(r_scaler.transform(rna.loc[test_ids]),  index=test_ids,  columns=rna.columns)

    # Concatenate expression and methylation data into one feature vector
    x_tr  = pd.concat([Xr_tr,  Mm_tr],  axis=1).to_numpy().astype("float32")
    x_val = pd.concat([Xr_val, Mm_val], axis=1).to_numpy().astype("float32")
    x_te  = pd.concat([Xr_te,  Mm_te],  axis=1).to_numpy().astype("float32")

    return x_tr, x_val, x_te


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


def build_model(
        in_features,
        learning_rate,
        weight_decay,
        num_nodes,
        dropout,
        batch_norm):
    # DeepSurv = risk network + Cox partial-likelihood loss (CoxPH).
    net = build_net(
        in_features,
        num_nodes=num_nodes,
        dropout=dropout,
        batch_norm=batch_norm
    )
    optimizer = tt.optim.Adam(lr=learning_rate, weight_decay=weight_decay)
    model = CoxPH(net, optimizer)
    return model


def c_index(model, x, durations, events):
    # C-index on the predicted log-risk (higher risk = earlier event).
    risk = model.predict(x).ravel()
    result = concordance_index_censored(events.astype(bool), durations, risk)
    return result[0]   # result = (c_index, n_concordant, n_discordant, n_tied_risk, n_tied_time)




# cross validated training & evaluation

def run_cv(
        learning_rate,
        weight_decay,
        num_nodes,
        dropout,
        batch_norm,
        meth_variance_threshold,
        patience,
        batch_size,
        max_epochs=MAX_EPOCHS,
        random_state=RANDOM_STATE):
    
    rows = []

    for f in sorted(fold_id.unique()):
        # Outer split: test set.
        train_val_ids = fold_id.index[fold_id != f]
        test_ids      = fold_id.index[fold_id == f]

        # Create internal validation set out of the training patients for early
        # stopping only (stratified on event so both splits keep some deaths).
        tr_ids, val_ids = train_test_split(
            train_val_ids, test_size=VAL_FRACTION, random_state=random_state,
            stratify=surv.loc[train_val_ids, "event"])

        # Fold-safe scaling: fit to training patients only.
        x_tr, x_val, x_te = build_features(tr_ids, val_ids, test_ids, meth_variance_threshold=meth_variance_threshold)
        y_tr, y_val = make_y(tr_ids), make_y(val_ids)

        durations_tr, events_tr = make_y(tr_ids)
        durations_val, events_val = make_y(val_ids)
        durations_te, events_te = make_y(test_ids)

        # Build and train DeepSurv with early stopping on the validation loss.
        torch.manual_seed(random_state)      
        model = build_model(x_tr.shape[1],
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            num_nodes=num_nodes,
                            dropout=dropout,
                            batch_norm=batch_norm)
        log = model.fit(
            x_tr, y_tr,
            batch_size=batch_size, 
            epochs=max_epochs,
            callbacks=[tt.callbacks.EarlyStopping(patience=patience)],
            val_data=(x_val, y_val), 
            val_batch_size=batch_size, 
            verbose=False)

        # Evaluate on the held-out test fold with C-index.
        train_c_index = c_index(model, x_tr, durations_tr, events_tr)
        val_c_index = c_index(model, x_val, durations_val, events_val)
        test_c_index = c_index(model, x_te, durations_te, events_te)

        epochs_trained = log.epoch + 1       
        rows.append({"fold": f, 
                     "n_test": len(test_ids), 
                     "n_features": x_tr.shape[1],
                     "epochs_trained": epochs_trained, 
                     "learning_rate": learning_rate,
                     "weight_decay": weight_decay,
                     "num_nodes": str(num_nodes),
                     "dropout": dropout,
                     "batch_norm": batch_norm,
                     "batch_size": batch_size,
                     "patience": patience,
                     "meth_variance_threshold": meth_variance_threshold,
                     "train_c_index": train_c_index,
                     "val_c_index": val_c_index, 
                     "test_c_index": test_c_index})
        
    return pd.DataFrame(rows)
    
    

# summarize results 

def summarize_cv_results(cv):
    print(cv.to_string(index=False))

    train_mean_ci, train_sd_ci = cv["train_c_index"].mean(), cv["train_c_index"].std()
    val_mean_ci, val_sd_ci = cv["val_c_index"].mean(), cv["val_c_index"].std()
    test_mean_ci, test_sd_ci = cv["test_c_index"].mean(), cv["test_c_index"].std()

    print(f"\nMean C-index of NN-Cox mRNA Expression + Methylation across 5 folds:")
    print(f"Train: {train_mean_ci:.3f} +/- {train_sd_ci:.3f}")
    print(f"Val: {val_mean_ci:.3f} +/- {val_sd_ci:.3f}")
    print(f"Test: {test_mean_ci:.3f} +/- {test_sd_ci:.3f}")

def save_cv_results(cv, filename="nn_cox_integrated_cv_results.csv"):

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
        "meth_variance_threshold": 0.0001,
        "patience": 15,
        "batch_size": 64
    }

    cv = run_cv(**config)
    summarize_cv_results(cv)
    save_cv_results(cv)

