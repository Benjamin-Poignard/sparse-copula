function [param,lambda_opt] = sparse_gaussian_copula(data,lambda,loss,method,a_scad,b_mcp,K)

% Inputs: - data: n x p matrix of observations
%         - lambda: tuning parameter
%         - loss: 'Gaussian' (Gaussian loss) or 'LS' (least squares loss)
%         - method : 'scad', 'mcp', 'lasso'
%         - a_scad: value of the scad parameter
%         - b_mcp: value of the mcp parameter
%         - K: number of folds (should be strictly larger than 2)

% Outputs: - param: vech(Sigma), i.e. column vector that stacks the
%           columns of the lower triangular part (excluding the diagonal
%           elements) of the covariance (correlation) matrix Sigma
%          - lambda_opt: optimal tuning parameter value chosen by
%          cross-validation; if no-cross validation is performed (i.e., the
%          input lambda is a scalar), then lambda_opt = lambda

% if lambda is a vector of tuning parameters, then apply cross-validatio
if length(lambda)>1
    
    % construction of the train sets and validation datasets
    [n,p] = size(data); len = round(n/K);
    U_f = zeros(len,p,K);
    theta_fold = zeros(p*(p-1)/2,length(lambda),K);
    
    U_f(:,:,1) = data(1:len,:); U_temp = data; U_temp(1:len,:)=[];
    parfor jj = 1:length(lambda)
        theta_fold(:,jj,1) = gaussian_copula_penalised(U_temp,lambda(jj),loss,method,a_scad,b_mcp);
    end
    
    for kk = 2:K-1
        U_f(:,:,kk) = data((kk-1)*len+1:kk*len,:); U_temp = data; U_temp((kk-1)*len+1:kk*len,:)=[];
        parfor jj = 1:length(lambda)
            theta_fold(:,jj,kk) = gaussian_copula_penalised(U_temp,lambda(jj),loss,method,a_scad,b_mcp);
        end
    end
    
    U_f(:,:,K) = data(end-len+1:end,:); U_temp = data; U_temp(end-len+1:end,:) = [];
    parfor jj = 1:length(lambda)
        theta_fold(:,jj,K) = gaussian_copula_penalised(U_temp,lambda(jj),loss,method,a_scad,b_mcp);
    end
    
    % evaluate the loss over the test datasets using the estimators
    % obtained over the train datasets (averaged over the number of folds)
    count = zeros(length(lambda),1);
    for ii = 1:length(lambda)
        for kk = 1:K
            Sigma = vech_off(theta_fold(:,ii,kk),size(data,2));
            switch loss
                case 'Gaussian'
                    L = (trace(cov(U_f(:,:,kk))/Sigma)+log(det(Sigma)));
                case 'LS'
                    L = norm(cov(U_f(:,:,kk))-Sigma,'fro')^2;
            end
            count(ii) = count(ii) + L;
        end
	count(ii) = count(ii)/K;
    end
    clear ii kk
    
    ii = count==min(min(count)); lambda_opt = lambda(ii);
    
    if length(lambda_opt)>1
        lambda_opt = lambda(1);
    end
    param = gaussian_copula_penalised(data,lambda_opt,loss,method,a_scad,b_mcp);
else
    lambda_opt = lambda;
    % if lambda is a scalar, no cross-validation is performed
    param = gaussian_copula_penalised(data,lambda_opt,loss,method,a_scad,b_mcp);
end