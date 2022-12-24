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

dim = p*(p-1)/2;
cond = true;
while cond
    temp = round(rand(dim,1)); temp(temp==0)=-1;
    C = vech_off((rand(dim,1)>threshold).*temp.*(a1+(a2-a1)*rand(dim,1)),p);
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

