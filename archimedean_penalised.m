function param_est = archimedean_penalised(family,data,X,lambda,method,a_scad,b_mcp)

% Inputs: - family: 'Gumbel' or 'Clayton'
%         - data: n x p matrix of observations
%         - X: n x p matrix of observed factors (p: dimension problem)
%         - lambda: tuning parameter
%         - method: 'scad', 'mcp', 'lasso'
%         - a_scad: value of the scad parameter
%         - b_mcp: value of the mcp parameter

% Output: - param_est: estimator of vech(Sigma), i.e. column vector that
%           stacks the columns of the lower triangular part (excluding the
%           diagonal elements) of the covariance (correlation) matrix Sigma

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 300000;
optimoptions.MaxFunEvals = 300000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 300000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

dim = size(X,2); param_init = 0.1*ones(dim,1);

[param_est,~,~,~,~,~]=fmincon(@(x)mle_archimedean_penalised(x,family,data,X,lambda,method,a_scad,b_mcp),param_init,[],[],[],[],[],[],@(x)constr(x),optimoptions);