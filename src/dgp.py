import numpy as np
import pandas as pd

def generate_data(n_samples=2000, p=25, seed=538):
    """
    This function generates the super population of the dataset that we use to run our simulations. 
    Parameters
    ----------
    n_samples : int, default=2000
        Number of observations to generate.

    p : int, default=25
        # of covariates. In our DGP, the irst few covariates are used in the 
        treatment assignment and treatment effect formulas (see appendix)

    seed : int, default=538
        stat 538!

    Returns
    -------
    df : pandas.DataFrame
        pandas df that contains:
        generated covariates, treatment assignment,
        observed outcome, potential outcomes, individual treatment effects,
        propensity scores, conditional means, and selected noisy covariates.

    true_ate : float
        this is the true ate of the population. this is the oracle ate

    """
    rng   = np.random.default_rng(seed)

    # X ~ N(0, Sigma)
    idx   = np.arange(p)
    Sigma = 0.3 ** np.abs(idx[:, None] - idx[None, :])
    X     = rng.multivariate_normal(np.zeros(p), Sigma, size=n_samples)

    # treatment assignment using nonlienar propensity scores
    logit_e = (0.5 * np.tanh(X[:, 0])
               + 0.3  * X[:, 1]
               - 0.2  * X[:, :5].sum(axis=1))
    e_x = 1 / (1 + np.exp(-logit_e))
    T   = rng.binomial(1, e_x)

    # pre additive noise
    eta = rng.normal(0, 0.4, size=(n_samples, p))
    Z   = X + eta

    # nonlinear baseline
    def g(Z):
        return (2 * np.sin(Z[:, 0]) * np.cos(Z[:, 1])
                + 0.5 * np.tanh(Z.sum(1))
                + 0.3 * np.log1p(np.abs((Z**2).sum(1))))

    # potential outcome means + heterogeneous treatment effect
    mu0 = g(Z)
    tau_x = (1.5 + 0.8 * np.sin(2 * X[:, 0])
             - 0.4 * X[:, 1]**2
             + 0.2 * X[:, :p].mean(1))
    mu1 = mu0 + tau_x

    # add outcome noise to get Y0 and Y1
    eps0 = rng.normal(0, 0.5, n_samples)
    eps1 = rng.normal(0, 0.5, n_samples)
    Y0, Y1 = mu0 + eps0, mu1 + eps1
    Y = T*Y1 +(1 -T) * Y0

    df = pd.DataFrame(X, columns=[f'X{j+1}' for j in range(p)])
    df['T']   = T #binary treatment indicator
    df['Y']   = Y #observed outcome
    df['Y0']  = Y0 #noisy potential outcomes
    df['Y1']  = Y1
    df['tau'] = tau_x #individual treatment effect
    df['e_x'] = e_x #treatment propensity score
    df['mu0'] = mu0 #noiseless conditional means
    df['mu1'] = mu1
    df['Z1']  = Z[:, 0] # x1 (for plotting)
    df['Z2']  = Z[:, 1] # x2 (for plotting)

    # finite-sample "true" ATE for this generated draw
    true_ate = float(tau_x.mean())

    # print diagnostics
    print(f"Samples       : {n_samples}")
    print(f"Covariates    : {p}")
    print(f"Treated units : {T.sum()}  ({T.mean()*100:.1f}%)")
    print(f"Control units : {(1-T).sum()}  ({(1-T).mean()*100:.1f}%)")
    print(f"True ATE      : {true_ate:.4f}")
    print(f"E[e(X)]       : {e_x.mean():.4f}")
    print(f"Y mean        : {float(Y.mean()):.4f}")
    print(f"Y std         : {float(Y.std()):.4f}")

    return df, true_ate



def make_weak_overlap(df, true_ate, gamma=8.0, seed=538):
    """
    this function introduces weak overlap by:

    - scaling propensity score by gamma to drive units towards the tails
    - keep treated units more often when that propensity is high
    - keep control units more often when that propensity is low

    Parameters
    ----------
    df : pandas.DataFrame
       output from generate_data()

    true_ate : float
      the true ate we got from the SUPER POPULATION

    gamma : float, default=8.0
        !!! THIS CONTROLS HOW STRONG THE WEAK OVERLAP IS:
        -> Larger gamma => more severe overlap problems.

    seed : int, default=538
        stat 538!

    Returns
    -------
    df_weak : pandas.DataFrame
        A pandas df containing the weak overlap data.

    """
    rng = np.random.default_rng(seed)

    # grab the X columns from the original df
    X_cols = [c for c in df.columns if c.startswith('X')]
    X = df[X_cols].values

    # scale the propensity with gamma
    logit_weak = gamma * (
          0.6 * X[:, 0]
        + 0.4 * X[:, 1]
        - 0.3 * X[:, 0]**2
    )
    # sigmoid
    e_weak = 1 / (1 + np.exp(-logit_weak))

    # original treatment assignment
    T = df['T'].values

    # rule:
    # treated units are kept more often when e_weak is high
    # control units are kept more often when e_weak is low
    retain_prob = np.where(T == 0, 1 - e_weak, e_weak)

    # avoid dropping literally everything in the worst regions
    min_keep = 0.05
    retain_prob = min_keep + (1 - min_keep) * retain_prob

    # the rows that survive
    keep = rng.uniform(size=len(df)) < retain_prob

    # keep surviving units only
    df_weak = df[keep].copy()
    df_weak['e_x_weak'] = e_weak[keep]

    # diagnostics
    n = len(df_weak)
    n_treated = int(df_weak['T'].sum())
    n_control = n - n_treated
    ew = e_weak[keep]

    print(f"Samples       : {n}  (dropped {len(df)-n} of {len(df)})")
    print(f"Gamma         : {gamma}")
    print(f"Covariates    : {len(X_cols)}")
    print(f"Treated units : {n_treated}  ({n_treated/n*100:.1f}%)")
    print(f"Control units : {n_control}  ({n_control/n*100:.1f}%)")
    print(f"True ATE      : {true_ate:.4f}")
    print(f"E[e(X)]       : {ew.mean():.4f}")
    print(f"Y mean        : {df_weak['Y'].mean():.4f}")
    print(f"Y std         : {df_weak['Y'].std():.4f}")
    print(f"e(X) < 0.1    : {(ew < 0.1).mean()*100:.1f}% of surviving units")
    print(f"e(X) > 0.9    : {(ew > 0.9).mean()*100:.1f}% of surviving units")
    print(f"0.1–0.9       : {((ew > 0.1) & (ew < 0.9)).mean()*100:.1f}% of surviving units")

    return df_weak