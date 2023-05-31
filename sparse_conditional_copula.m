function [param,lambda_opt] = sparse_conditional_copula(family,data,X,lambda,method,a_scad,b_mcp,K)

% Inputs: - family: 'Gumbel' or 'Clayton'
%         - data: n x 2 matrix of observations
%         - X: n x p matrix of observed factors (p: dimension problem)
%         - lambda: tuning parameter
%         - loss: 'Gaussian' (Gaussian loss) or 'LS' (least squares loss)
%         - method : 'scad', 'mcp', 'lasso'
%         - a_scad: value of the scad parameter
%         - b_mcp: value of the mcp parameter
%         - K: number of folds (should be strictly larger than 2)

% Outputs: - param: sparse estimator of the conditional copula
%          - lambda_opt: optimal tuning parameter value chosen by
%          cross-validation; if no-cross validation is performed (i.e., the
%          input lambda is a scalar), then lambda_opt = lambda

% if lambda is a vector of tuning parameters, then apply cross-validation
if length(lambda)>1
    
    % construction of the train sets and validation datasets
    dim = size(X,2);
    [n,p] = size(data); len = round(n/K);
    U_f = zeros(len,p,K); X_f = zeros(len,size(X,2),K);
    theta_fold = zeros(dim,length(lambda),K);
    U_f(:,:,1) = data(1:len,:); U_temp = data; U_temp(1:len,:)=[];
    X_f(:,:,1) = X(1:len,:,:); X_temp = X; X_temp(1:len,:)=[];
    parfor jj = 1:length(lambda)
        theta_fold(:,jj,1) = archimedean_penalised(family,U_temp,X_temp,lambda(jj),method,a_scad,b_mcp);
    end
    
    for kk = 2:K-1
        U_f(:,:,kk) = data((kk-1)*len+1:kk*len,:); U_temp = data; U_temp((kk-1)*len+1:kk*len,:)=[];
        X_f(:,:,kk) = X((kk-1)*len+1:kk*len,:); X_temp = X; X_temp((kk-1)*len+1:kk*len,:)=[];
        parfor jj = 1:length(lambda)
            theta_fold(:,jj,kk) = archimedean_penalised(family,U_temp,X_temp,lambda(jj),method,a_scad,b_mcp);
        end
    end
    
    U_f(:,:,K) = data(end-len+1:end,:); U_temp = data; U_temp(end-len+1:end,:) = [];
    X_f(:,:,K) = X(end-len+1:end,:); X_temp = X; X_temp(end-len+1:end,:)=[];
    parfor jj = 1:length(lambda)
        theta_fold(:,jj,K) = archimedean_penalised(family,U_temp,X_temp,lambda(jj),method,a_scad,b_mcp);
    end
    
    % evaluate the loss over the test datasets using the estimators
    % obtained over the train datasets (averaged over the number of folds)
    count = zeros(length(lambda),1);
    for ii = 1:length(lambda)
        for kk = 1:K
            L = mle_archimedean_penalised(theta_fold(:,ii,kk),family,U_f(:,:,kk),X_f(:,:,kk),0,method,a_scad,b_mcp);
            count(ii) = count(ii) + L;
        end
	count(ii) = count(ii)/K;
    end
    clear ii kk
    
    ii = count==min(min(count)); lambda_opt = lambda(ii);
    
    if length(lambda_opt)>1
        lambda_opt = lambda(1);
    end
    param = archimedean_penalised(family,data,X,lambda_opt,method,a_scad,b_mcp);
    % Negligible estimated values numerically obtained by fmincon are shrunk to zero
    param(abs(param)<0.0001)=0;
else
    lambda_opt = lambda;
    % if lambda is a scalar, no cross-validation is performed
    param = archimedean_penalised(family,data,X,lambda_opt,method,a_scad,b_mcp);
    % Negligible estimated values numerically obtained by fmincon are shrunk to zero
    param(abs(param)<0.0001)=0;
end