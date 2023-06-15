function param_est = gaussian_copula_penalised(data,lambda,loss,method,a_scad,b_mcp)

% Inputs: - data: n x p matrix of observations
%         - lambda: tuning parameter
%         - loss: 'Gaussian' (Gaussian loss) or 'LS' (least squares loss)
%         - method: 'scad', 'mcp', 'lasso'
%         - a_scad: value of the scad parameter
%         - b_mcp: value of the mcp parameter

% Output: - param_est: estimator of vech(Sigma), i.e. column vector that 
%           stacks the columns of the lower triangular part (excluding the 
%           diagonal elements) of the covariance (correlation) matrix Sigma

% The algorithm is a gradient type algorithm, which uses the SCAD/MCP and
% LASSO updates provided in Loh and Wainwright (2015), 'Regularized
% M-estimators with Nonconvexity: Statistical and Algorithmic Theory for 
% Local Optima', Journal of Machine Learning Research 16 (2015) 559-61: see
% their Section 4.2 (note: their side constraint is discarded)

p = size(data,2); dim = p*(p-1)/2;
S = cov(data);
switch method
    case 'scad'
        mu = 1/(a_scad-1);
    case 'mcp'
        mu = 1/b_mcp;
    case 'lasso'
        mu = 0;
end
eta = 1000;
% calibration of the step size
nu = (1/eta)/(1+(mu/eta));

%param = zeros(dim,1);
param_update = vech_on(S,p); 
maxIt = 1e8;
count = 0;
while count < maxIt
    count=count+1;
    param = param_update;
    Sigma = vech_off(param,p);
    switch loss
        case 'Gaussian'
            gradient = vech_on(inv(Sigma)-inv(Sigma)*S*inv(Sigma),p);
        case 'LS'
            gradient = 2*vech_on(Sigma-S,p);
    end
    gradient_modified = gradient - mu*param;
    Z = (1/(1+(mu/eta)))*(param-gradient_modified/eta);
    param_est = zeros(dim,1);
    switch method
        case 'scad'
            for ii = 1:dim
                tmplambda = lambda;
                if (0 <= abs(Z(ii)) && abs(Z(ii)) <= nu*tmplambda)
                    param_est(ii) = 0;
                elseif (nu*tmplambda <= abs(Z(ii)) && abs(Z(ii)) <= (nu+1)*tmplambda)
                    param_est(ii) = Z(ii)-(sign(Z(ii))*nu*tmplambda);
                elseif ((nu+1)*tmplambda <= abs(Z(ii)) && abs(Z(ii)) <= a_scad*tmplambda)
                    param_est(ii) = (Z(ii)-((sign(Z(ii))*a_scad*nu*tmplambda)/(a_scad-1)))/(1-(nu/(a_scad-1)));
                elseif (a_scad*tmplambda <= abs(Z(ii)))
                    param_est(ii) = Z(ii);
                end
            end
            param_update = param_est;
        case 'mcp'
            for ii = 1:dim
                tmplambda = lambda;
                if (0 <= abs(Z(ii)) && abs(Z(ii)) <= nu*tmplambda)
                    param_est(ii) = 0;
                elseif (nu*tmplambda <= abs(Z(ii)) && abs(Z(ii)) <= b_mcp*tmplambda)
                    param_est(ii) = (Z(ii)-(sign(Z(ii))*nu*tmplambda))/(1-nu/b_mcp);
                elseif (b_mcp*tmplambda <= abs(Z(ii)))
                    param_est(ii) = Z(ii);
                end
            end
            param_update = param_est;
        case 'lasso'
            for ii = 1:dim
                tmplambda = lambda;
                param_est(ii) = sign(Z(ii))*subplus(abs(Z(ii))-tmplambda/eta);
            end
            param_update = param_est;
    end
    if ((norm(param_update - param)^2/max([1,norm(param_update),norm(param)]))<=eps)
        break;
    end
end
param_est = param_update;

% Final step: verify that the estimator of Sigma is positive-definite
if min(eig(vech_off(param_est,p)))<0.001
    % If the estimator is not positive-definite, then a projection on the
    % set of positive-definite matrix is applied (the negative-eigenvalues 
    % are set to small positive values)
    C_proj =  nearcorr(vech_off(param_est,p));
    param_est = vech_on(C_proj,p);
end
