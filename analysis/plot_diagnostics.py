#MODEL CHECK 1: TRACE PLOT OF delta_7 and delta_1

import arviz as az
import matplotlib.pyplot as plt

idata = az.from_netcdf("analysis/lambda_model_fixed_nuts_quick.nc")
az.plot_trace(idata, var_names=["delta_7","delta_1"])
plt.show()

az.plot_posterior(idata, var_names=["delta_7","delta_1"])
plt.show()