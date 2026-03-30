#MODEL CHECK 1: TRACE PLOT OF delta_7 and delta_1

import arviz as az
import matplotlib.pyplot as plt

idata = az.from_netcdf("analysis/lambda_model_fixed_nuts_quick.nc")

ax = az.plot_posterior(idata, var_names=["delta_7","delta_1"])
plt.tight_layout()
plt.savefig("analysis/posterior_deltas.png", dpi=200)
plt.close()

ax = az.plot_trace(idata, var_names=["delta_7","delta_1"])
plt.tight_layout()
plt.savefig("analysis/trace_deltas.png", dpi=200)
plt.close()

print("[OK] wrote analysis/posterior_deltas.png and analysis/trace_deltas.png")