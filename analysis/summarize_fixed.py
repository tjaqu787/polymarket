### RUN THIS TO SUMMARIZE THE .NC POSTERIORS FROM ADVI AND NUTS


import arviz as az
import numpy as np
from pathlib import Path

paths = [
    "analysis/lambda_model_fixed_advi.nc",
    "analysis/lambda_model_fixed_nuts.nc",
    "analysis/lambda_model_fixed_nuts_quick.nc",
]

for path in paths:
    if not Path(path).exists():
        continue

    idata = az.from_netcdf(path)
    print("\n===", path, "===")
    s = az.summary(idata, var_names=["delta_7","delta_1"])
    print(s)

    for name in ["delta_7","delta_1"]:
        m = float(s.loc[name, "mean"])
        lo = float(s.loc[name, "hdi_3%"])
        hi = float(s.loc[name, "hdi_97%"])
        print(f"{name}: exp(mean)={np.exp(m):.3f}, exp(HDI)=[{np.exp(lo):.3f}, {np.exp(hi):.3f}]")