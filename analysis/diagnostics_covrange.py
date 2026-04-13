import arviz as az
import matplotlib.pyplot as plt

idata = az.from_netcdf("analysis/lambda_model_covrange_nuts.nc")

az.plot_trace(idata, var_names=["delta_7", "delta_1", "beta_range"])
plt.tight_layout()
plt.savefig("analysis/trace_covrange.png", dpi=200)

az.plot_posterior(idata, var_names=["delta_7", "delta_1", "beta_range"], hdi_prob=0.94)
plt.tight_layout()
plt.savefig("analysis/posterior_covrange.png", dpi=200)