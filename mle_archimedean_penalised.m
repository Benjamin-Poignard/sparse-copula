function L = mle_archimedean_penalised(param,family,data,X,lambda,method,a_scad,b_mcp)

% Inputs: - param: parameters of the conditional copula
%         - family: 'Gumbel' or 'Clayton'
%         - data: n x p matrix of observations
%         - X: n x p matrix of observed factors (p: dimension problem)
%         - lambda: tuning parameter
%         - method: 'scad', 'mcp', 'lasso'
%         - a_scad: value of the scad parameter
%         - b_mcp: value of the mcp parameter

% Output: - L: scalar value of the penalized ML loss function

switch method
    case 'scad'
        pen = scad(param,lambda,a_scad);
    case 'mcp'
        pen = mcp(param,lambda,b_mcp);
    case 'lasso'
        pen = lambda*sum(abs(param));
end

switch family
    case 'Gumbel'
        tau = 2*atan(X*param)/pi;
        copula_param = 1./(1-tau);
        L = -sum(log( exp(-((-log(data(:,1))).^copula_param + (-log(data(:,2))).^copula_param).^(1./copula_param)).*(-log(data(:,1))).^(copula_param-1).*(-log(data(:,2))).^(copula_param-1)./data(:,1)./data(:,2).*((-log(data(:,1))).^copula_param+(-log(data(:,2))).^copula_param).^(1./copula_param-2).*( ((-log(data(:,1))).^copula_param+(-log(data(:,2))).^copula_param).^(1./copula_param) + copula_param - 1 ) ));
    case 'Clayton'
        tau = 2*atan(X*param)/pi;
        copula_param = (2*tau)./(1-tau);
        L = -sum(log((copula_param+1).*(data(:,1).^(-copula_param)+data(:,2).^(-copula_param)-1).^(-1./copula_param-2).*data(:,1).^(-copula_param-1).*data(:,2).^(-copula_param-1) ));
end
L = L/length(data)+pen;
