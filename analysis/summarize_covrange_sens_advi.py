import arviz as az

idata = az.from_netcdf("analysis/lambda_model_covrange_sens_advi.nc")

vars_main = ["delta_7", "delta_1", "beta_range", "sigma", "sigma_slug", "sigma_delta"]
summ = az.summary(idata, var_names=vars_main, hdi_prob=0.94)

print(summ)
summ.to_csv("analysis/lambda_model_covrange_sens_advi_summary.csv")
print("\n[ok] wrote analysis/lambda_model_covrange_sens_advi_summary.csv")