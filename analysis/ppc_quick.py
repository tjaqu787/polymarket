# analysis/ppc_quick.py
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

DATA_PATH = "analysis/timing_model_analysis_bft.csv"
ADVI_PATH = "analysis/lambda_model_fixed_advi.nc"
NUTS_PATH = "analysis/lambda_model_fixed_nuts_quick.nc"

TOL = 3.0  # same as model
TARGETS = {"30": 30.0, "7": 7.0, "1": 1.0}

rng = np.random.default_rng(0)

def build_long(df):
    rows = []
    for h, target in TARGETS.items():
        lam = df[f"lambda_{h}d"].to_numpy()
        days = df[f"days_{h}d"].to_numpy()
        m = np.isfinite(lam) & (lam > 0) & np.isfinite(days) & (np.abs(days - target) <= TOL)
        rows.append(pd.DataFrame({
            "horizon": h,
            "log_lambda": np.log(lam[m]),
        }))
    return pd.concat(rows, ignore_index=True)

def flat(idata, name):
    x = idata.posterior[name].values
    return x.reshape(-1)

def simulate_from_posterior(idata, n=50000):
    mu = flat(idata, "mu")
    sigma = flat(idata, "sigma")
    sigma_slug = flat(idata, "sigma_slug")
    d7 = flat(idata, "delta_7")
    d1 = flat(idata, "delta_1")

    idx = rng.integers(0, len(mu), size=n)

    # marginalize slug effect as N(0, sigma_slug)
    u = rng.normal(0.0, sigma_slug[idx])
    e = rng.normal(0.0, sigma[idx])

    # sample horizons uniformly just to get a combined check
    horizons = rng.choice(["30", "7", "1"], size=n, replace=True)
    delta = np.zeros(n)
    delta[horizons == "7"] = d7[idx][horizons == "7"]
    delta[horizons == "1"] = d1[idx][horizons == "1"]

    y = mu[idx] + delta + u + e
    return y

def simulate_from_prior(n=50000):
    # priors from run_lambda_model_fixed.py
    mu = rng.normal(-5.0, 2.0, size=n)
    sigma_delta = np.abs(rng.normal(0.0, 1.0, size=n))
    delta_7 = rng.normal(0.0, sigma_delta, size=n)
    delta_1 = rng.normal(0.0, sigma_delta, size=n)
    sigma_slug = np.abs(rng.normal(0.0, 1.0, size=n))
    sigma = np.abs(rng.normal(0.0, 1.0, size=n))

    horizons = rng.choice(["30", "7", "1"], size=n, replace=True)
    delta = np.zeros(n)
    delta[horizons == "7"] = delta_7[horizons == "7"]
    delta[horizons == "1"] = delta_1[horizons == "1"]

    u = rng.normal(0.0, sigma_slug)
    e = rng.normal(0.0, sigma)
    y = mu + delta + u + e
    return y

def quantiles(x):
    return np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])

def main():
    df = pd.read_csv(DATA_PATH)
    long = build_long(df)
    y_obs = long["log_lambda"].to_numpy()

    id_advi = az.from_netcdf(ADVI_PATH)
    id_nuts = az.from_netcdf(NUTS_PATH)

    y_prior = simulate_from_prior()
    y_post_advi = simulate_from_posterior(id_advi)
    y_post_nuts = simulate_from_posterior(id_nuts)

    out = pd.DataFrame({
        "dist": ["observed", "prior_pred", "post_pred_advi", "post_pred_nuts"],
        "q01": [quantiles(y_obs)[0], quantiles(y_prior)[0], quantiles(y_post_advi)[0], quantiles(y_post_nuts)[0]],
        "q05": [quantiles(y_obs)[1], quantiles(y_prior)[1], quantiles(y_post_advi)[1], quantiles(y_post_nuts)[1]],
        "q50": [quantiles(y_obs)[2], quantiles(y_prior)[2], quantiles(y_post_advi)[2], quantiles(y_post_nuts)[2]],
        "q95": [quantiles(y_obs)[3], quantiles(y_prior)[3], quantiles(y_post_advi)[3], quantiles(y_post_nuts)[3]],
        "q99": [quantiles(y_obs)[4], quantiles(y_prior)[4], quantiles(y_post_advi)[4], quantiles(y_post_nuts)[4]],
    })
    out.to_csv("analysis/ppc_quantiles.csv", index=False)
    print(out)

    # Plot: observed vs prior predictive
    plt.figure(figsize=(10,4))
    plt.hist(y_obs, bins=80, density=True, alpha=0.5, label="observed")
    plt.hist(y_prior, bins=80, density=True, alpha=0.5, label="prior predictive")
    plt.title("Prior predictive check (log lambda)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/prior_predictive_check.png", dpi=200, bbox_inches="tight")

    # Plot: observed vs posterior predictive (ADVI + NUTS)
    plt.figure(figsize=(10,4))
    plt.hist(y_obs, bins=80, density=True, alpha=0.4, label="observed")
    plt.hist(y_post_advi, bins=80, density=True, alpha=0.4, label="posterior predictive (ADVI)")
    plt.hist(y_post_nuts, bins=80, density=True, alpha=0.4, label="posterior predictive (NUTS quick)")
    plt.title("Posterior predictive check (log lambda)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/posterior_predictive_check.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()