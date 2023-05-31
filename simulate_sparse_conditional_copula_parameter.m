function [alpha,X,beta_true] = simulate_sparse_conditional_copula_parameter(n,p,family,sparsity,threshold)

% Input: - n: sample size
%        - p: dimension
%        - sparsity: sparsity degree, precentage of p set as zero
%        coefficients
%        - threshold: controls for how many zero coefficients are generated
%        in the true vector of coefficients; the higher, the more zero
%        coefficients are generated

% Outputs: - alpha: copula parameter of size n x 1
%          - X: n x p matrix of covariates simulated in uniform([0,1])
%          - beta_true: true sparse parameter vector of size p x 1

cond_sparse = true;
while cond_sparse
    beta_true = (rand(p,1)>threshold).*(0.05+0.95*rand(p,1));
    count = 0;
    for ii = 1:p
        if (beta_true(ii)==0)
            count = count+1;
        else
            count = count+0;
        end
    end
    cond_sparse = (count>sparsity)||(count<sparsity);
end
X = rand(n,p);
switch family
    case 'Gumbel'
        tau = 2*atan(X*beta_true)/pi;
        alpha = 1./(1-tau);
    case 'Clayton'
        tau = 2*atan(X*beta_true)/pi;
        alpha = (2*tau)./(1-tau);
end