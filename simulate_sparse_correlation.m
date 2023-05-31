function C = simulate_sparse_correlation(p,sparsity,threshold,a1,a2)

% Input:
%       - p: dimension
%       - sparsity: sparsity degree, precentage of p*(p-1)/2 set as zero
%       coefficients
%       - threshold: controls for how many zero coefficients are generated
%       in the true vector of coefficients; the higher, the more zero
%       coefficients are generated
%       - a1 and a2: lower and upper bounds of the interval for the uniform
%       distribution when generating the true non-zero coefficients

dim = p*(p-1)/2; cond = true; parameter = zeros(dim,1);
while cond
    
    for kk=1:dim
        cond_dist = true;
        while cond_dist
            param = -a2+(a2-(-a2))*rand(1);
            cond_dist=(param>-a1 & param<a1);
        end
        parameter(kk) = param;
    end
    C = vech_off((rand(dim,1)>threshold).*parameter,p);
    beta_true = vech_on(C,p);
    count = 0;
    for ii = 1:dim
        if (beta_true(ii)==0)
            count = count+1;
        else
            count = count+0;
        end
    end
    % Check that the two following conditions are satisifed: number of zero
    % coefficients (remains unchanged for all batches);
    % positive-definiteness of the true correlation (or
    % variance-covariance) matrix of the Gaussian copula
    cond = (count>sparsity)||(count<sparsity)||(min(eig(C))<0.15);
end