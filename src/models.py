# models.py

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from econml.dml import CausalForestDML


# ----------------------------------------------------------
# helper functions
# ----------------------------------------------------------

def _clip_propensity(p, eps=0.05):
    return np.clip(p, eps, 1 - eps)


def _to_numpy(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def _pick_device(device="auto"):
    if device != "auto":
        return device

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# ----------------------------------------------------------
# OLS
# ----------------------------------------------------------

def estimate_ols(X, W, Y):
    """
    Naive OLS estimator.

    Regresses Y on [X, W] and returns the coefficient on W.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Covariates
    W : array-like, shape (n,)
        Binary treatment indicator
    Y : array-like, shape (n,)
        Outcome

    Returns
    -------
    ate_hat : float
        Coefficient on W from linear regression.
    """
    X = _to_numpy(X)
    W = _to_numpy(W).reshape(-1, 1)
    Y = _to_numpy(Y).ravel()

    XW = np.column_stack([X, W])
    coef = LinearRegression().fit(XW, Y).coef_
    return float(coef[-1])


# ----------------------------------------------------------
# IPW
# ----------------------------------------------------------

def estimate_ipw(X, W, Y, clip=0.05, max_iter=1000):
    """
    IPW with propensity model 

    Parameters
    ----------
    X : array-like, shape (n, p)
        Covariates
    W : array-like, shape (n,)
        Binary treatment indicator
    Y : array-like, shape (n,)
        Outcome
    clip : float, default=0.05
        Lower/upper clipping threshold for estimated propensities.
    max_iter : int, default=1000
        Max iterations for logistic regression.

    Returns
    -------
    ate_hat : float
        IPW estimate of the ATE.
    """
    X = _to_numpy(X)
    W = _to_numpy(W).ravel()
    Y = _to_numpy(Y).ravel()

    ps_model = LogisticRegression(max_iter=max_iter)
    ps_model.fit(X, W)
    ps_hat = ps_model.predict_proba(X)[:, 1]
    ps_hat = _clip_propensity(ps_hat, eps=clip)

    ate_hat = np.mean(W * Y / ps_hat) - np.mean((1 - W) * Y / (1 - ps_hat))
    return float(ate_hat)


# ----------------------------------------------------------
# AIPW with RF
# ----------------------------------------------------------

def estimate_aipw_rf(
    X,
    W,
    Y,
    n_splits=2,
    seed=538,
    clip=0.05,
    max_iter=1000,
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
):
    """
    Cross-fitted AIPW estimator.

    Propensity is estimated with logistic regression.
    Outcome regression is estimated with a random forest on [X, W].

    Parameters
    ----------
    X : array-like, shape (n, p)
        Covariates
    W : array-like, shape (n,)
        Binary treatment indicator
    Y : array-like, shape (n,)
        Outcome
    n_splits : int, default=2
        Number of folds for cross-fitting.
    seed : int, default=538
        Random seed
    clip : float, default=0.05
        Clipping threshold for estimated propensities
    max_iter : int, default=1000
        Max iterations for logistic regression
    n_estimators : int, default=200
        Number of trees in the random forest
    max_depth : int, default=6
        Max depth of each tree
    min_samples_leaf : int, default=5
        Minimum leaf size for the forest

    Returns
    -------
    ate_hat : float
        AIPW estimate of the ATE.
    """
    X = _to_numpy(X)
    W = _to_numpy(W).ravel()
    Y = _to_numpy(Y).ravel()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    mu1_hat = np.zeros(len(Y))
    mu0_hat = np.zeros(len(Y))
    e_hat = np.zeros(len(Y))

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        W_tr, Y_tr = W[tr_idx], Y[tr_idx]

        # propensity model
        ps_model = LogisticRegression(max_iter=max_iter)
        ps_model.fit(X_tr, W_tr)
        e_hat[te_idx] = ps_model.predict_proba(X_te)[:, 1]

        # outcome model on [X, W]
        XW_tr = np.column_stack([X_tr, W_tr])
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        rf.fit(XW_tr, Y_tr)

        mu1_hat[te_idx] = rf.predict(np.column_stack([X_te, np.ones(len(X_te))]))
        mu0_hat[te_idx] = rf.predict(np.column_stack([X_te, np.zeros(len(X_te))]))

    e_hat = _clip_propensity(e_hat, eps=clip)

    pseudo_outcome = (
        mu1_hat + W * (Y - mu1_hat) / e_hat
        - mu0_hat - (1 - W) * (Y - mu0_hat) / (1 - e_hat)
    )

    return float(np.mean(pseudo_outcome))





# ----------------------------------------------------------
# Causal Forest
# ----------------------------------------------------------
def estimate_causal_forest(X, W, Y, seed=538):
    """
    Estimate the ATE with a causal forest.
    """
    X = _to_numpy(X)
    W = _to_numpy(W).ravel()
    Y = _to_numpy(Y).ravel()

    model_y = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=seed,
    )
    model_t = LogisticRegression(max_iter=1000)

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=200,
        min_samples_leaf=5,
        max_depth=None,
        discrete_treatment=True,
        random_state=seed,
    )
    cf.fit(Y, W, X=X)

    cate_hat = cf.effect(X)
    return float(np.mean(cate_hat))


# ----------------------------------------------------------
# TRAINING FUNCTIONS
# ----------------------------------------------------------

def train_engression(
    df,
    df_weak,
    num_epochs=500,
    batch_size=None,
    lr=0.01,
    device="auto",
    verbose=True,
):
    """
    Train engression on the weak-overlap sample using [X, T] as inputs.

    The model is fit on the broken sample df_weak, but predictions are made on
    the full original sample df to recover:
        - mu0(x) = E[Y | X=x, T=0]
        - mu1(x) = E[Y | X=x, T=1]
        - tau(x) = mu1(x) - mu0(x)

    Parameters
    ----------
    df : pandas.DataFrame
        Original full sample
    df_weak : pandas.DataFrame
        Weak-overlap sample
    num_epochs : int, default=500
        Number of training epochs
    batch_size : int or None, default=None
        Batch size. If None, uses min(512, len(df_weak))
    lr : float, default=0.01
        Learning rate
    device : {"auto", "mps", "cuda", "cpu"}, default="auto"
        Training device
    verbose : bool, default=True
        Whether to print training progress

    Returns
    -------
    eng_model : fitted engression model
    df_eval : pandas.DataFrame
        Original df with predicted mu0, mu1, and tau surfaces attached.
    metrics : dict
        Core diagnostics: plug-in ATE, true ATE, RMSE, correlation, and training settings.
    """
    from engression import engression

    X_cols = [c for c in df.columns if c.startswith("X")]
    device = _pick_device(device)

    if batch_size is None:
        batch_size = min(512, len(df_weak))

    X_train = df_weak[X_cols].values.astype(np.float32)
    T_train = df_weak["T"].values.astype(np.float32).reshape(-1, 1)
    Y_train = df_weak["Y"].values.astype(np.float32).reshape(-1, 1)
    XT_train = np.hstack([X_train, T_train])

    X_eval = df[X_cols].values.astype(np.float32)
    T_eval_0 = np.zeros((len(X_eval), 1), dtype=np.float32)
    T_eval_1 = np.ones((len(X_eval), 1), dtype=np.float32)

    XT_eval_0 = np.hstack([X_eval, T_eval_0])
    XT_eval_1 = np.hstack([X_eval, T_eval_1])

    XT_train_t = torch.tensor(XT_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    XT_eval_0_t = torch.tensor(XT_eval_0, dtype=torch.float32)
    XT_eval_1_t = torch.tensor(XT_eval_1, dtype=torch.float32)

    eng_model = engression(
        XT_train_t,
        Y_train_t,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    mu0_pred = (
        eng_model.predict(XT_eval_0_t, target="mean")
        .flatten()
        .detach()
        .cpu()
        .numpy()
    )

    mu1_pred = (
        eng_model.predict(XT_eval_1_t, target="mean")
        .flatten()
        .detach()
        .cpu()
        .numpy()
    )

    tau_pred = mu1_pred - mu0_pred

    df_eval = df[X_cols + ["Y", "mu0", "mu1", "tau", "T", "e_x"]].copy()
    df_eval["mu0_pred"] = mu0_pred
    df_eval["mu1_pred"] = mu1_pred
    df_eval["tau_pred"] = tau_pred

    rmse_mu0 = float(np.sqrt(np.mean((df_eval["mu0_pred"] - df_eval["mu0"]) ** 2)))
    corr_mu0 = float(np.corrcoef(df_eval["mu0_pred"], df_eval["mu0"])[0, 1])
    ate_engression = float(np.mean(df_eval["tau_pred"]))
    true_ate = float(np.mean(df_eval["tau"]))

    metrics = {
        "rmse_mu0": rmse_mu0,
        "corr_mu0": corr_mu0,
        "ate_engression": ate_engression,
        "true_ate": true_ate,
        "device": device,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_train": len(df_weak),
        "n_eval": len(df),
    }

    return eng_model, df_eval, metrics







def estimate_engression_naive(X, eng_model):
    """
    Plug-in ATE from a trained engression model.

    For each x, predict:
        mu1(x) = E[Y | X=x, T=1]
        mu0(x) = E[Y | X=x, T=0]

    Then return mean(mu1(x) - mu0(x)).
    """
    T_ones  = np.ones((len(X), 1), dtype=np.float32)
    T_zeros = np.zeros((len(X), 1), dtype=np.float32)

    XT_1 = torch.tensor(np.hstack([X, T_ones]), dtype=torch.float32)
    XT_0 = torch.tensor(np.hstack([X, T_zeros]), dtype=torch.float32)

    with torch.no_grad():
        mu1 = eng_model.predict(XT_1, target="mean").flatten().cpu().numpy()
        mu0 = eng_model.predict(XT_0, target="mean").flatten().cpu().numpy()

    return float(np.mean(mu1 - mu0))




def augment_with_engression(
    df_weak,
    eng_model,
    threshold_low=0.05,
    threshold_high=0.95,
    seed=538,
):
    """
    Augment the weak-overlap sample with synthetic counterfactual twins
    generated from a trained engression model.

    In low-propensity regions, control units are duplicated as treated units
    with sampled Y(1) outcomes. In high-propensity regions, treated units are
    duplicated as control units with sampled Y(0) outcomes.

    Parameters
    ----------
    df_weak : pandas.DataFrame
        Weak-overlap dataset.
    eng_model :
        Trained engression model.
    threshold_low : float, default=0.05
        Controls which control units are considered in a very low-propensity region.
    threshold_high : float, default=0.95
        Controls which treated units are considered in a very high-propensity region.
    seed : int, default=538
        Random seed.

    Returns
    -------
    df_aug : pandas.DataFrame
        Augmented dataset containing original observations plus synthetic twins.
    """
    rng = np.random.default_rng(seed)

    X_cols = [c for c in df_weak.columns if c.startswith("X")]
    e_weak = df_weak["e_x_weak"].values
    T = df_weak["T"].values

    df_orig = df_weak.copy()
    df_orig["synthetic"] = 0
    twins = []

    def sample_counterfactual(X_arr, t_label):
        n = len(X_arr)
        T_col = np.full((n, 1), t_label, dtype=np.float32)
        XT = np.hstack([X_arr.astype(np.float32), T_col])

        XT_t = torch.tensor(XT, dtype=torch.float32)

        with torch.no_grad():
            y_sample = eng_model.sample(
                XT_t,
                sample_size=1,
                expand_dim=False,
            )

        if torch.is_tensor(y_sample):
            y_sample = y_sample.cpu().numpy().flatten()
        else:
            y_sample = np.asarray(y_sample).flatten()

        return y_sample

    # duplicate controls as treated in low-propensity regions
    mask_1 = (T == 0) & (e_weak < threshold_low)
    if mask_1.sum() > 0:
        X_twin = df_weak.loc[mask_1, X_cols].values
        Y_twin = sample_counterfactual(X_twin, t_label=1)

        df_twin = df_weak.loc[mask_1, X_cols].copy().reset_index(drop=True)
        df_twin["T"] = 1
        df_twin["Y"] = Y_twin
        df_twin["e_x_weak"] = e_weak[mask_1]
        df_twin["synthetic"] = 1

        twins.append(df_twin)

        print(f"Case 1: {mask_1.sum()} control units duplicated as treated")

    # duplicate treated as controls in high-propensity regions
    mask_2 = (T == 1) & (e_weak > threshold_high)
    if mask_2.sum() > 0:
        X_twin = df_weak.loc[mask_2, X_cols].values
        Y_twin = sample_counterfactual(X_twin, t_label=0)

        df_twin = df_weak.loc[mask_2, X_cols].copy().reset_index(drop=True)
        df_twin["T"] = 0
        df_twin["Y"] = Y_twin
        df_twin["e_x_weak"] = e_weak[mask_2]
        df_twin["synthetic"] = 1

        twins.append(df_twin)

        print(f"Case 2: {mask_2.sum()} treated units duplicated as control")

    df_aug = pd.concat([df_orig] + twins, ignore_index=True)

    for col in df_orig.columns:
        if col not in df_aug.columns:
            df_aug[col] = np.nan

    n = len(df_aug)
    n_treated = int(df_aug["T"].sum())
    n_control = n - n_treated

    print("\nAugmentation summary")
    print(f"Original units        : {len(df_weak)}")
    print(f"Synthetic twins added : {n - len(df_weak)}")
    print(f"Augmented total       : {n}")
    print(
        f"Treated / Control     : {n_treated} / {n_control} "
        f"({n_treated/n*100:.1f}% / {n_control/n*100:.1f}%)"
    )

    return df_aug


# ----------------------------------------------------------
# RUN THIS SIMULATION
# ----------------------------------------------------------


def run_simulation(
    sim_df,
    x_cols,
    true_ate,
    eng_model,
    n_sims=100,
    seed=538,
    min_group_size=10,
    verbose=True,
):
    """
    Run the bootstrap comparison for the estimators on one dataset.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Dataset to resample from.

    x_cols : list[str]
        Covariate column names.

    true_ate : float
        Oracle ATE used to compute bias and RMSE.

    eng_model :
        Trained engression model for the naive plug-in estimator.

    n_sims : int, default=100
        Number of bootstrap draws.

    seed : int, default=538
        Random seed.

    min_group_size : int, default=10
        Skip bootstrap samples with too few treated or control units.

    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    estimates_df : pandas.DataFrame
        Raw bootstrap estimates for each estimator.

    summary_df : pandas.DataFrame
        Summary table with mean, bias, sd, rmse, and number of valid draws.
    """
    rng = np.random.default_rng(seed)

    results = {
        "OLS": [],
        "IPW": [],
        "AIPW_RF": [],
        "Causal_Forest": [],
        "Engression_Naive": [],
    }

    if verbose:
        print(f"Running {n_sims} bootstrap simulations...")
        print(f"True ATE = {true_ate:.4f}\n")

    for sim in range(n_sims):
        idx = rng.choice(len(sim_df), size=len(sim_df), replace=True)
        boot_df = sim_df.iloc[idx].copy()

        X = boot_df[x_cols].values.astype(np.float32)
        W = boot_df["T"].values.astype(np.float32)
        Y = boot_df["Y"].values.astype(np.float32)

        if W.sum() < min_group_size or (1 - W).sum() < min_group_size:
            continue

        results["OLS"].append(estimate_ols(X, W, Y))
        results["IPW"].append(estimate_ipw(X, W, Y))
        results["AIPW_RF"].append(estimate_aipw_rf(X, W, Y))
        results["Engression_Naive"].append(estimate_engression_naive(X, eng_model))
        results["Causal_Forest"].append(estimate_causal_forest(X, W, Y, seed=seed))

        if verbose and (sim + 1) % 20 == 0:
            print(f"  Sim {sim+1}/{n_sims} done")

    if verbose:
        print("\nDone.")

    estimates_df = pd.DataFrame(
        {k: pd.Series(v, dtype=float) for k, v in results.items()}
    )

    summary_rows = []
    for est_name in estimates_df.columns:
        vals = estimates_df[est_name].dropna().values

        summary_rows.append({
            "Estimator": est_name,
            "Mean": np.mean(vals),
            "Bias": np.mean(vals) - true_ate,
            "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "RMSE": np.sqrt(np.mean((vals - true_ate) ** 2)),
            "N": len(vals),
        })

    summary_df = pd.DataFrame(summary_rows)

    return estimates_df, summary_df