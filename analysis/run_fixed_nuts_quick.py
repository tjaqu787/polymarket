import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

def main():
    df = pd.read_csv("analysis/timing_model_analysis_bft.csv")
    for c in ["lambda_30d","lambda_7d","lambda_1d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # optional filter: keep snapshots close to their targets
    df = df[
        (df["days_30d"].sub(30).abs() <= 7) &
        (df["days_7d"].sub(7).abs() <= 2) &
        (df["days_1d"].sub(1).abs() <= 2)
    ].copy()

    long = []
    for h, lam_col in [("30d","lambda_30d"), ("7d","lambda_7d"), ("1d","lambda_1d")]:
        tmp = df[["event_slug", lam_col]].copy()
        tmp = tmp.rename(columns={lam_col: "lambda"})
        tmp["horizon"] = h
        long.append(tmp)

    long = pd.concat(long, ignore_index=True).dropna(subset=["lambda"])
    long = long[long["lambda"] > 0].copy()
    long["log_lambda"] = np.log(long["lambda"])

    # subsample for speed (still plenty to confirm sign/magnitude)
    np.random.seed(0)
    if len(long) > 2500:
        long = long.sample(n=2500, random_state=0).reset_index(drop=True)

    h_map = {"30d": 0, "7d": 1, "1d": 2}
    h_idx = long["horizon"].map(h_map).values
    slugs, slug_idx = np.unique(long["event_slug"].astype(str), return_inverse=True)
    y = long["log_lambda"].values.astype(float)

    with pm.Model() as m:
        mu = pm.Normal("mu", -5.0, 2.0)

        sigma_delta = pm.HalfNormal("sigma_delta", 1.0)
        delta_7 = pm.Normal("delta_7", 0.0, sigma_delta)
        delta_1 = pm.Normal("delta_1", 0.0, sigma_delta)
        delta = pm.math.stack([0.0, delta_7, delta_1])

        sigma_slug = pm.HalfNormal("sigma_slug", 1.0)
        u_slug = pm.Normal("u_slug", 0.0, sigma_slug, shape=len(slugs))

        sigma = pm.HalfNormal("sigma", 1.0)
        mu_obs = mu + delta[h_idx] + u_slug[slug_idx]
        pm.Normal("y", mu_obs, sigma, observed=y)

        idata = pm.sample(draws=100, tune=100, chains=1, cores=1, target_accept=0.9, progressbar=True)

    az.to_netcdf(idata, "analysis/lambda_model_fixed_nuts_quick.nc")
    print(az.summary(idata, var_names=["delta_7","delta_1","sigma_slug","sigma_delta","sigma"]))

if __name__ == "__main__":
    main()