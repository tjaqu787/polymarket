import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

def zscore(x):
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s == 0:
        return np.zeros_like(x)
    return (x - m) / s

# Rebuild the long dataset exactly like the model did
df = pd.read_csv("analysis/timing_model_input_bft_cov2.csv")
keep = (
    (df["days_30d"].sub(30).abs() <= 7) &
    (df["days_7d"].sub(7).abs()   <= 2) &
    (df["days_1d"].sub(1).abs()   <= 2)
)
df = df.loc[keep].copy()

parts = []
for horizon, suf in [(30, "30d"), (7, "7d"), (1, "1d")]:
    parts.append(pd.DataFrame({
        "event_slug": df["event_slug"],
        "horizon": horizon,
        "price": df[f"price_{suf}"],
        "days_to_deadline": df[f"days_{suf}"],
        "price_range_24h": df[f"price_range_24h_{suf}"],
    }))
long = pd.concat(parts, ignore_index=True)

eps = 1e-6
p = np.clip(long["price"].to_numpy(float), eps, 1 - eps)
t = long["days_to_deadline"].to_numpy(float)
lam = -np.log(1 - p) / t
y_obs = np.log(lam)

h7 = (long["horizon"].to_numpy() == 7).astype(float)
h1 = (long["horizon"].to_numpy() == 1).astype(float)
x_range = zscore(np.log1p(long["price_range_24h"].to_numpy(float)))

group_idx, group_levels = pd.factorize(long["event_slug"].astype(str), sort=True)
n_groups = len(group_levels)

# posterior draws
idata = az.from_netcdf("analysis/lambda_model_covrange_nuts.nc")
post = idata.posterior

mu = post["mu"].values.reshape(-1)
delta_7 = post["delta_7"].values.reshape(-1)
delta_1 = post["delta_1"].values.reshape(-1)
beta_range = post["beta_range"].values.reshape(-1)
sigma = post["sigma"].values.reshape(-1)
sigma_slug = post["sigma_slug"].values.reshape(-1)

# simulate posterior predictive draws over observed design
rng = np.random.default_rng(123)
n_draws = min(1000, len(mu))
idx = rng.choice(len(mu), size=n_draws, replace=False)

y_rep = []
for j in idx:
    u_g = rng.normal(0.0, sigma_slug[j], size=n_groups)
    mean_j = mu[j] + delta_7[j] * h7 + delta_1[j] * h1 + beta_range[j] * x_range + u_g[group_idx]
    y_j = rng.normal(mean_j, sigma[j])
    y_rep.append(y_j)
y_rep = np.concatenate(y_rep)

plt.figure(figsize=(10, 5))
plt.hist(y_obs, bins=80, density=True, alpha=0.5, label="observed")
plt.hist(y_rep, bins=80, density=True, alpha=0.5, label="posterior predictive (new model)")
plt.xlabel("Log implied daily hazard, log(λ)")
plt.ylabel("Density")
plt.title("Posterior predictive check (covariate-augmented model)")
plt.legend()
plt.tight_layout()
plt.savefig("analysis/posterior_predictive_covrange.png", dpi=200)