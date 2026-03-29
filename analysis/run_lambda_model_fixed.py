import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

def main():
    df = pd.read_csv("analysis/timing_model_analysis_bft.csv")

    # Use the lambdas computed in SQL (Poisson by-deadline mapping).
    # Drop invalid values (can happen when price is 1 or missing).
    for c in ["lambda_30d","lambda_7d","lambda_1d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional but recommended: enforce that the snapshot horizons are actually close
    # to their intended targets so outliers don't dominate.
    df = df[
        (df["days_30d"].sub(30).abs() <= 7) &
        (df["days_7d"].sub(7).abs() <= 2) &
        (df["days_1d"].sub(1).abs() <= 2)
    ].copy()

    # Long format: one row per market × horizon
    long = []
    for h, lam_col in [("30d","lambda_30d"), ("7d","lambda_7d"), ("1d","lambda_1d")]:
        tmp = df[["event_slug", lam_col]].copy()
        tmp = tmp.rename(columns={lam_col: "lambda"})
        tmp["horizon"] = h
        long.append(tmp)

    long = pd.concat(long, ignore_index=True).dropna(subset=["lambda"])
    long = long[long["lambda"] > 0].copy()
    long["log_lambda"] = np.log(long["lambda"])

    # Horizon coding (baseline 30d)
    h_map = {"30d": 0, "7d": 1, "1d": 2}
    h_idx = long["horizon"].map(h_map).values

    # Group by event_slug for pooling
    slugs, slug_idx = np.unique(long["event_slug"].astype(str), return_inverse=True)
    y = long["log_lambda"].values.astype(float)

    with pm.Model() as m:
        # Priors on the log-lambda scale
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

        # ADVI first (fast and always saved)
        approx = pm.fit(15000, method="advi")
        idata_advi = approx.sample(2000)
        az.to_netcdf(idata_advi, "analysis/lambda_model_fixed_advi.nc")

        # Quick NUTS (second method). Keep this short and reliable.
        idata_nuts = pm.sample(
            draws=300, tune=300,
            chains=1, cores=1,
            target_accept=0.9,
            progressbar=True
        )
        az.to_netcdf(idata_nuts, "analysis/lambda_model_fixed_nuts.nc")

    print("\nADVI (fixed) summary:")
    print(az.summary(idata_advi, var_names=["mu","delta_7","delta_1","sigma","sigma_slug","sigma_delta"]))

    print("\nNUTS (fixed, quick) summary:")
    print(az.summary(idata_nuts, var_names=["mu","delta_7","delta_1","sigma","sigma_slug","sigma_delta"]))

if __name__ == "__main__":
    main()