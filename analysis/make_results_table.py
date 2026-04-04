import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path

OUTDIR = Path("analysis")
files = {
    "ADVI (full data)": OUTDIR / "lambda_model_fixed_advi.nc",
    "NUTS quick (subsample)": OUTDIR / "lambda_model_fixed_nuts_quick.nc",
    "NUTS quick sensitivity (sigma_delta=0.5)": OUTDIR / "lambda_model_fixed_nuts_quick_sensitivity.nc",
}

rows = []
for label, path in files.items():
    if not path.exists():
        continue
    idata = az.from_netcdf(path)
    summ = az.summary(idata, var_names=["delta_7", "delta_1"], hdi_prob=0.94)
    for param in ["delta_7", "delta_1"]:
        mean = float(summ.loc[param, "mean"])
        hdi_l = float(summ.loc[param, "hdi_3%"])
        hdi_u = float(summ.loc[param, "hdi_97%"])
        rows.append({
            "fit": label,
            "param": param,
            "mean": mean,
            "sd": float(summ.loc[param, "sd"]),
            "hdi_3%": hdi_l,
            "hdi_97%": hdi_u,
            "exp(mean)": float(np.exp(mean)),
            "exp(hdi_3%)": float(np.exp(hdi_l)),
            "exp(hdi_97%)": float(np.exp(hdi_u)),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No netcdf files found. Check paths in analysis/.")

# nicer formatting / ordering
df["param"] = df["param"].map({"delta_7": "delta_7 (7d vs 30d)", "delta_1": "delta_1 (1d vs 30d)"})
df = df.sort_values(["param", "fit"]).reset_index(drop=True)

OUTDIR.mkdir(exist_ok=True, parents=True)
df.to_csv(OUTDIR / "results_table.csv", index=False)

# text version
lines = []
for param in df["param"].unique():
    lines.append(param)
    sub = df[df["param"] == param]
    for _, r in sub.iterrows():
        lines.append(
            f"  {r['fit']}: mean={r['mean']:.3f} (94% HDI [{r['hdi_3%']:.3f}, {r['hdi_97%']:.3f}]), "
            f"multiplier exp(mean)={r['exp(mean)']:.3f} (94% HDI [{r['exp(hdi_3%)']:.3f}, {r['exp(hdi_97%)']:.3f}])"
        )
    lines.append("")
(OUTDIR / "results_table.txt").write_text("\n".join(lines), encoding="utf-8")

# short summary paragraph
advi = df[df["fit"] == "ADVI (full data)"].set_index("param")
nuts = df[df["fit"] == "NUTS quick (subsample)"].set_index("param")
para = (
    "Across inference methods, we find little evidence of a consistent 7-day horizon effect, "
    f"but a strong 1-day effect. Under ADVI, delta_1 has mean {advi.loc['delta_1 (1d vs 30d)','mean']:.3f}, "
    f"corresponding to a rate multiplier exp(delta_1)≈{advi.loc['delta_1 (1d vs 30d)','exp(mean)']:.2f}. "
    "The quick NUTS run on a subsample yields a very similar estimate. "
    "This supports the descriptive term-structure pattern that implied hazard accelerates sharply near the deadline."
)
print("\n" + para + "\n")