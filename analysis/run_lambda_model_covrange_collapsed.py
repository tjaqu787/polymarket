import argparse
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt

def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if (not np.isfinite(sd)) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd

def build_long(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for horizon, suf in [(30, "30d"), (7, "7d"), (1, "1d")]:
        part = pd.DataFrame({
            "market_id": df["market_id"],
            "event_id": df["event_id"],
            "event_slug": df["event_slug"],
            "token_id": df["token_id"],
            "horizon": horizon,
            "price": df[f"price_{suf}"],
            "days_to_deadline": df[f"days_{suf}"],
            "price_range_24h": df[f"price_range_24h_{suf}"],
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/timing_model_input_bft_cov2.csv")
    ap.add_argument("--out_prefix", default="analysis/lambda_model_covrange")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--advi_steps", type=int, default=30000)
    ap.add_argument("--advi_draws", type=int, default=2000)
    args = ap.parse_args()

    # Read wide snapshot table
    df = pd.read_csv(args.csv)

    # Same horizon purity filter you already described in the report
    keep = (
        (df["days_30d"].sub(30).abs() <= 7) &
        (df["days_7d"].sub(7).abs()   <= 2) &
        (df["days_1d"].sub(1).abs()   <= 2)
    )
    df = df.loc[keep].copy()
    print(f"[info] Markets after horizon filter: {len(df)}")

    # Long format
    long = build_long(df)

    # Hazard transform
    eps = 1e-6
    p = np.clip(long["price"].to_numpy(dtype=float), eps, 1 - eps)
    t = long["days_to_deadline"].to_numpy(dtype=float)
    lam = -np.log(1 - p) / t
    y = np.log(lam)

    # Horizon indicators (30d is baseline)
    h7 = (long["horizon"].to_numpy() == 7).astype(float)
    h1 = (long["horizon"].to_numpy() == 1).astype(float)

    # Main covariate: local 24h price range around the snapshot
    x_range = np.log1p(long["price_range_24h"].to_numpy(dtype=float))
    x_range = zscore(x_range)

    # Grouping variable
    group_idx, group_levels = pd.factorize(long["event_slug"].astype(str), sort=True)
    group_idx = group_idx.astype("int32")
    n_groups = len(group_levels)
    n_per_group = np.bincount(group_idx, minlength=n_groups).astype(float)

    with pm.Model() as model:
        # Data
        y_obs = pm.Data("y_obs", y)
        h7_d = pm.Data("h7", h7)
        h1_d = pm.Data("h1", h1)
        x_range_d = pm.Data("x_range", x_range)
        g_idx = pm.Data("group_idx", group_idx)
        n_g = pm.Data("n_g", n_per_group)

        # Priors
        mu = pm.Normal("mu", mu=-5.0, sigma=2.0)

        sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.0)
        delta_7 = pm.Normal("delta_7", mu=0.0, sigma=sigma_delta)
        delta_1 = pm.Normal("delta_1", mu=0.0, sigma=sigma_delta)

        sigma_slug = pm.HalfNormal("sigma_slug", sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        beta_range = pm.Normal("beta_range", mu=0.0, sigma=1.0)

        # Linear predictor without explicit group effects
        # eta_ih = mu + delta_h + beta_range * x_range + u_g
        eta_fixed = mu + delta_7 * h7_d + delta_1 * h1_d + beta_range * x_range_d

        # Collapse out the Normal random intercept u_g analytically
        resid = y_obs - eta_fixed

        sum_r = pt.bincount(g_idx, weights=resid, minlength=n_groups)
        sum_r2 = pt.bincount(g_idx, weights=resid**2, minlength=n_groups)

        sigma2 = sigma**2
        tau2 = sigma_slug**2
        n = n_g

        # For each group: covariance = sigma^2 I + tau^2 J
        logdet = (n - 1.0) * pt.log(sigma2) + pt.log(sigma2 + n * tau2)
        quad = (sum_r2 / sigma2) - (tau2 * sum_r**2) / (sigma2 * (sigma2 + n * tau2))

        logp_groups = -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)
        pm.Potential("likelihood", pt.sum(logp_groups))

        # --- Inference method 1: ADVI ---
        approx = pm.fit(n=args.advi_steps, method="advi", progressbar=True)
        idata_advi = approx.sample(draws=args.advi_draws)
        if not hasattr(idata_advi, "posterior"):
            idata_advi = pm.to_inference_data(idata_advi)

        advi_nc = f"{args.out_prefix}_advi.nc"
        idata_advi.to_netcdf(advi_nc)
        print(f"[ok] wrote {advi_nc}")

        # --- Inference method 2: full NUTS ---
        # Using ADVI-based initialization is the cheap "batching-like" speedup
        idata_nuts = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            init="advi+adapt_diag",
            random_seed=123,
            return_inferencedata=True,
            progressbar=True,
        )

        nuts_nc = f"{args.out_prefix}_nuts.nc"
        idata_nuts.to_netcdf(nuts_nc)
        print(f"[ok] wrote {nuts_nc}")

        # Summaries
        vars_main = ["delta_7", "delta_1", "beta_range", "sigma", "sigma_slug", "sigma_delta"]

        summ_advi = az.summary(idata_advi, var_names=vars_main, hdi_prob=0.94)
        summ_nuts = az.summary(idata_nuts, var_names=vars_main, hdi_prob=0.94)

        advi_csv = f"{args.out_prefix}_advi_summary.csv"
        nuts_csv = f"{args.out_prefix}_nuts_summary.csv"

        summ_advi.to_csv(advi_csv)
        summ_nuts.to_csv(nuts_csv)

        print(f"[ok] wrote {advi_csv}")
        print(f"[ok] wrote {nuts_csv}")

        print("\n=== ADVI summary ===")
        print(summ_advi)
        print("\n=== NUTS summary ===")
        print(summ_nuts)

if __name__ == "__main__":
    main()