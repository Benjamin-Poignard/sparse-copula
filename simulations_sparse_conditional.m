%% Sample size n = 500, Gumbel copula
clear; clc;
rng(1, 'twister' );
% Number of batches
Nsim = 200;

% sample size and number of factors (covariates) in the vector Z entering
% the conditioning set of the conditional copula
n = 500; p = 30;

% Sparsity degree, that is the proportion of zero coefficients in beta, the
% vector of coefficients for the linear combination beta * Z
% triangular part of the correlation matrix
sparsity = round(p*0.9); threshold = 0.8;

% select copula family
family = 'Gumbel';

% Number of folds for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

% values for a_scad and b_mcp
a_scad = [3.7 10 20 40 70]; b_mcp = [3.5 10 20 40 70];
len_scad = length(a_scad); len_mcp = length(b_mcp);
check_prop = zeros(Nsim,len_scad+len_mcp+1); check_prop2 = zeros(Nsim,len_scad+len_mcp+1);
MSE = zeros(Nsim,len_scad+len_mcp+1);

for oo = 1:Nsim
    
    [alpha,X,beta_true] = simulate_sparse_conditional_copula_parameter(n,p,family,sparsity,threshold);
    U = zeros(n,2);
    for ii=1:n
        U(ii,:) = copularnd(family,alpha(ii),1);
    end
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(p)/n);
    
    % SCAD penalization
    for kk = 1:len_scad
        
        [beta_est_scad,~] = sparse_conditional_copula(family,U_po,X,lambda,'scad',a_scad(kk),0,K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_scad = check_NZ(beta_true,beta_est_scad);
        % proportion of zero coefficients correctly identified
        check_prop(oo,kk) =check_zero(beta_true,beta_est_scad)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,kk) = NZ_scad/(p-sparsity);
        % MSE
        MSE(oo,kk) = mse(beta_true,beta_est_scad);
        
    end
    clear kk
    
    % MCP penalization
    for kk = 1:len_mcp
        
        [beta_est_mcp,~] = sparse_conditional_copula(family,U_po,X,lambda,'mcp',0,b_mcp(kk),K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_mcp = check_NZ(beta_true,beta_est_mcp);
        % proportion of zero coefficients correctly identified
        check_prop(oo,len_scad+kk) = check_zero(beta_true,beta_est_mcp)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,len_scad+kk) = NZ_mcp/(p-sparsity);
        % MSE
        MSE(oo,len_scad+kk) = mse(beta_true,beta_est_mcp);
        
    end
    clear kk
    
    % LASSO penalization
    [beta_est_lasso,~] = sparse_conditional_copula(family,U_po,X,lambda,'lasso',a_scad,b_mcp,K);
    
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    % proportion of zero coefficients correctly identified
    check_prop(oo,len_scad+len_mcp+1) = check_zero(beta_true,beta_est_lasso)./sparsity;
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,len_scad+len_mcp+1) = NZ_lasso/(p-sparsity);
    % MSE
    MSE(oo,len_scad+len_mcp+1) = mse(beta_true,beta_est_lasso);
    
end

% Column entries in check_prop, check_prop2 and MSE for each line:
% 1st to len_scad: Gumbel-based ML estimation with SCAD, where the column
% len_scad is the last value of a_scad)
% len_scad+1 to len_scad+len_mcp: Gumbel-based ML estimation with MCP,
% where the column len_scad+len_mcp is the last value of b_mcp)
% last column: Gumbel-based ML estimation with LASSO

%% Sample size n = 1000, Gumbel copula
clear; clc;
rng(2, 'twister' );
Nsim = 200;

% sample size and number of factors (covariates) in the vector Z entering
% the conditioning set of the conditional copula
n = 1000; p = 30;

% Sparsity degree, that is the proportion of zero coefficients in beta, the
% vector of coefficients for the linear combination beta * Z
% triangular part of the correlation matrix
sparsity = round(p*0.9); threshold = 0.8;

% select copula family
family = 'Gumbel';

% Number of folds for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

% values for a_scad and b_mcp
a_scad = [3.7 10 20 40 70]; b_mcp = [3.5 10 20 40 70];
len_scad = length(a_scad); len_mcp = length(b_mcp);
check_prop = zeros(Nsim,len_scad+len_mcp+1); check_prop2 = zeros(Nsim,len_scad+len_mcp+1);
MSE = zeros(Nsim,len_scad+len_mcp+1);

for oo = 1:Nsim
    
    [alpha,X,beta_true] = simulate_sparse_conditional_copula_parameter(n,p,family,sparsity,threshold);
    U = zeros(n,2);
    for ii=1:n
        U(ii,:) = copularnd(family,alpha(ii),1);
    end
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(p)/n);
    
    % SCAD penalization
    for kk = 1:len_scad
        
        [beta_est_scad,~] = sparse_conditional_copula(family,U_po,X,lambda,'scad',a_scad(kk),0,K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_scad = check_NZ(beta_true,beta_est_scad);
        % proportion of zero coefficients correctly identified
        check_prop(oo,kk) =check_zero(beta_true,beta_est_scad)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,kk) = NZ_scad/(p-sparsity);
        % MSE
        MSE(oo,kk) = mse(beta_true,beta_est_scad);
        
    end
    clear kk
    
    % MCP penalization
    for kk = 1:len_mcp
        
        [beta_est_mcp,~] = sparse_conditional_copula(family,U_po,X,lambda,'mcp',0,b_mcp(kk),K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_mcp = check_NZ(beta_true,beta_est_mcp);
        % proportion of zero coefficients correctly identified
        check_prop(oo,len_scad+kk) = check_zero(beta_true,beta_est_mcp)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,len_scad+kk) = NZ_mcp/(p-sparsity);
        % MSE
        MSE(oo,len_scad+kk) = mse(beta_true,beta_est_mcp);
        
    end
    clear kk
    
    % LASSO penalization
    [beta_est_lasso,~] = sparse_conditional_copula(family,U_po,X,lambda,'lasso',a_scad,b_mcp,K);
    
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    % proportion of zero coefficients correctly identified
    check_prop(oo,len_scad+len_mcp+1) = check_zero(beta_true,beta_est_lasso)./sparsity;
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,len_scad+len_mcp+1) = NZ_lasso/(p-sparsity);
    % MSE
    MSE(oo,len_scad+len_mcp+1) = mse(beta_true,beta_est_lasso);
    
end

% Column entries in check_prop, check_prop2 and MSE for each line:
% 1st to len_scad: Gumbel-based ML estimation with SCAD, where the column
% len_scad is the last value of a_scad)
% len_scad+1 to len_scad+len_mcp: Gumbel-based ML estimation with MCP,
% where the column len_scad+len_mcp is the last value of b_mcp)
% last column: Gumbel-based ML estimation with LASSO

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Below is the Clayton copula case
%% Sample size n = 500, Clayton copula
clear; clc;
rng(3, 'twister' );
% Number of batches
Nsim = 200;

% sample size and number of factors (covariates) in the vector Z entering
% the conditioning set of the conditional copula
n = 500; p = 30;

% Sparsity degree, that is the proportion of zero coefficients in beta, the
% vector of coefficients for the linear combination beta * Z
% triangular part of the correlation matrix
sparsity = round(p*0.9); threshold = 0.8;

% select copula family
family = 'Clayton';

% Number of folds for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

% values for a_scad and b_mcp
a_scad = [3.7 10 20 40 70]; b_mcp = [3.5 10 20 40 70];
len_scad = length(a_scad); len_mcp = length(b_mcp);
check_prop = zeros(Nsim,len_scad+len_mcp+1); check_prop2 = zeros(Nsim,len_scad+len_mcp+1);
MSE = zeros(Nsim,len_scad+len_mcp+1);

for oo = 1:Nsim
    
    [alpha,X,beta_true] = simulate_sparse_conditional_copula_parameter(n,p,family,sparsity,threshold);
    U = zeros(n,2);
    for ii=1:n
        U(ii,:) = copularnd(family,alpha(ii),1);
    end
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(p)/n);
    
    % SCAD penalization
    for kk = 1:len_scad
        
        [beta_est_scad,~] = sparse_conditional_copula(family,U_po,X,lambda,'scad',a_scad(kk),0,K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_scad = check_NZ(beta_true,beta_est_scad);
        % proportion of zero coefficients correctly identified
        check_prop(oo,kk) =check_zero(beta_true,beta_est_scad)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,kk) = NZ_scad/(p-sparsity);
        % MSE
        MSE(oo,kk) = mse(beta_true,beta_est_scad);
        
    end
    clear kk
    
    % MCP penalization
    for kk = 1:len_mcp
        
        [beta_est_mcp,~] = sparse_conditional_copula(family,U_po,X,lambda,'mcp',0,b_mcp(kk),K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_mcp = check_NZ(beta_true,beta_est_mcp);
        % proportion of zero coefficients correctly identified
        check_prop(oo,len_scad+kk) = check_zero(beta_true,beta_est_mcp)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,len_scad+kk) = NZ_mcp/(p-sparsity);
        % MSE
        MSE(oo,len_scad+kk) = mse(beta_true,beta_est_mcp);
        
    end
    clear kk
    
    % LASSO penalization
    [beta_est_lasso,~] = sparse_conditional_copula(family,U_po,X,lambda,'lasso',a_scad,b_mcp,K);
    
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    % proportion of zero coefficients correctly identified
    check_prop(oo,len_scad+len_mcp+1) = check_zero(beta_true,beta_est_lasso)./sparsity;
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,len_scad+len_mcp+1) = NZ_lasso/(p-sparsity);
    % MSE
    MSE(oo,len_scad+len_mcp+1) = mse(beta_true,beta_est_lasso);
    
end

% Column entries in check_prop, check_prop2 and MSE for each line:
% 1st to len_scad: Gumbel-based ML estimation with SCAD, where the column
% len_scad is the last value of a_scad)
% len_scad+1 to len_scad+len_mcp: Gumbel-based ML estimation with MCP,
% where the column len_scad+len_mcp is the last value of b_mcp)
% last column: Gumbel-based ML estimation with LASSO

%% Sample size n = 1000, Clayton copula
clear; clc;
rng(4, 'twister' );
% Number of batches
Nsim = 200;

% sample size and number of factors (covariates) in the vector Z entering
% the conditioning set of the conditional copula
n = 1000; p = 30;

% Sparsity degree, that is the proportion of zero coefficients in beta, the
% vector of coefficients for the linear combination beta * Z
% triangular part of the correlation matrix
sparsity = round(p*0.9); threshold = 0.8;

% select copula family
family = 'Clayton';

% Number of folds for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

% values for a_scad and b_mcp
a_scad = [3.7 10 20 40 70]; b_mcp = [3.5 10 20 40 70];
len_scad = length(a_scad); len_mcp = length(b_mcp);
check_prop = zeros(Nsim,len_scad+len_mcp+1); check_prop2 = zeros(Nsim,len_scad+len_mcp+1);
MSE = zeros(Nsim,len_scad+len_mcp+1);

for oo = 1:Nsim
    
    [alpha,X,beta_true] = simulate_sparse_conditional_copula_parameter(n,p,family,sparsity,threshold);
    U = zeros(n,2);
    for ii=1:n
        U(ii,:) = copularnd(family,alpha(ii),1);
    end
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(p)/n);
    
    % SCAD penalization
    for kk = 1:len_scad
        
        [beta_est_scad,~] = sparse_conditional_copula(family,U_po,X,lambda,'scad',a_scad(kk),0,K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_scad = check_NZ(beta_true,beta_est_scad);
        % proportion of zero coefficients correctly identified
        check_prop(oo,kk) =check_zero(beta_true,beta_est_scad)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,kk) = NZ_scad/(p-sparsity);
        % MSE
        MSE(oo,kk) = mse(beta_true,beta_est_scad);
        
    end
    clear kk
    
    % MCP penalization
    for kk = 1:len_mcp
        
        [beta_est_mcp,~] = sparse_conditional_copula(family,U_po,X,lambda,'mcp',0,b_mcp(kk),K);
        
        % compute the number of non-zero coefficients correctly identified
        NZ_mcp = check_NZ(beta_true,beta_est_mcp);
        % proportion of zero coefficients correctly identified
        check_prop(oo,len_scad+kk) = check_zero(beta_true,beta_est_mcp)./sparsity;
        % proportion of non-zero coefficients correctly identified
        check_prop2(oo,len_scad+kk) = NZ_mcp/(p-sparsity);
        % MSE
        MSE(oo,len_scad+kk) = mse(beta_true,beta_est_mcp);
        
    end
    clear kk
    
    % LASSO penalization
    [beta_est_lasso,~] = sparse_conditional_copula(family,U_po,X,lambda,'lasso',a_scad,b_mcp,K);
    
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    % proportion of zero coefficients correctly identified
    check_prop(oo,len_scad+len_mcp+1) = check_zero(beta_true,beta_est_lasso)./sparsity;
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,len_scad+len_mcp+1) = NZ_lasso/(p-sparsity);
    % MSE
    MSE(oo,len_scad+len_mcp+1) = mse(beta_true,beta_est_lasso);
    
end

% Column entries in check_prop, check_prop2 and MSE for each line:
% 1st to len_scad: Gumbel-based ML estimation with SCAD, where the column
% len_scad is the last value of a_scad)
% len_scad+1 to len_scad+len_mcp: Gumbel-based ML estimation with MCP,
% where the column len_scad+len_mcp is the last value of b_mcp)
% last column: Gumbel-based ML estimation with LASSO
