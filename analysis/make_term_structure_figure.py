import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INFILE = "analysis/timing_model_analysis_bft.csv"
OUTFILE = "analysis/term_structure_fixed.png"

df = pd.read_csv(INFILE)

# Match the model’s horizon proximity filter
df = df[(df["days_30d"] - 30).abs() <= 7]
df = df[(df["days_7d"]  - 7 ).abs() <= 2]
df = df[(df["days_1d"]  - 1 ).abs() <= 2]

# log lambdas
df["log_lambda_30d"] = np.log(df["lambda_30d"])
df["log_lambda_7d"]  = np.log(df["lambda_7d"])
df["log_lambda_1d"]  = np.log(df["lambda_1d"])

horizons = ["30d", "7d", "1d"]
cols = ["log_lambda_30d", "log_lambda_7d", "log_lambda_1d"]

# Bootstrap 94% interval for mean(log lambda)
rng = np.random.default_rng(0)
B = 2000

means = []
lo = []
hi = []
for c in cols:
    x = df[c].to_numpy()
    n = len(x)
    boot = np.empty(B)
    for b in range(B):
        boot[b] = rng.choice(x, size=n, replace=True).mean()
    m = x.mean()
    qlo, qhi = np.quantile(boot, [0.03, 0.97])
    means.append(m)
    lo.append(qlo)
    hi.append(qhi)

xpos = np.arange(len(horizons))

plt.figure(figsize=(8,5))
plt.errorbar(xpos, means, yerr=[np.array(means)-np.array(lo), np.array(hi)-np.array(means)], fmt="o", capsize=4)
plt.xticks(xpos, horizons)
plt.xlabel("Horizon before market deadline")
plt.ylabel("Mean log implied daily hazard (mean log λ)")
plt.title("Implied hazard increases as the deadline approaches")
plt.tight_layout()
plt.savefig(OUTFILE, dpi=200)
print(f"Wrote {OUTFILE} with N={len(df)} markets")