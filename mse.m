function MSE = mse(target,estimator)

MSE = norm(target-estimator)^2;