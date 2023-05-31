%% Sample size n = 500, dimension p = 10
clear; clc;
rng(1, 'twister' );
% Number of batches
Nsim = 200;
check = zeros(Nsim,10); check_prop = zeros(Nsim,10); check_prop2 = zeros(Nsim,10);
check2 = zeros(Nsim,20); MSE = zeros(Nsim,10);

% Sample size and dimension
n = 500; p = 10;

% Total number of distinct parameters (size of theta) located in the lower
% tringular part of the correlation matrix
dim = p*(p-1)/2;

% Sparsity degree, that is the proportion of zero coefficients in the lower
% triangular part of the correlation matrix
sparsity = round(dim*0.85);

% Threshold: controls for the number of zero coefficients generated in the
% lower (and so upper) triangular part of the correlation matrix
% a1 and a2: lower and upper bounds when generating the true non-zero
% coefficients of the lower (and so upper) triangular part of the
% correlation matrix in the uniform distribution
threshold = 0.8; a1 = 0.05; a2 = 0.7;

% Number of folds and specification of the grid for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

for oo = 1:Nsim
    
    % simulate the sparse correlation matrix Gaussian copula parameter
    % for each simulation, the size of the true support is unchanged
    Sigma = simulate_sparse_correlation(p,sparsity,threshold,a1,a2);
    beta_true = vech_on(Sigma,p);
    
    % generate the true uniform observations on (0,1)^p
    U = copularnd('Gaussian',Sigma,n);
    % apply a rank-based transformation to get the pseudo-observations
    % hat{U}
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(dim)/n);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    % number of estimated zero coefficients whereas the true coefficient is
    % non-zero and number of estimated non-zero coefficients whereas the
    % true is zero
    [N1_scad,N2_scad] = check_IC_2(beta_true,beta_est_scad);
    % number of non-zero coefficients correctly identified
    NZ_scad = check_NZ(beta_true,beta_est_scad);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l,N2_scad_l] = check_IC_2(beta_true,beta_est_scad_l);
    NZ_scad_l = check_NZ(beta_true,beta_est_scad_l);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp,N2_mcp] = check_IC_2(beta_true,beta_est_mcp);
    NZ_mcp = check_NZ(beta_true,beta_est_mcp);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l,N2_mcp_l] = check_IC_2(beta_true,beta_est_mcp_l);
    NZ_mcp_l = check_NZ(beta_true,beta_est_mcp_l);
    
    % lasso penalization
    beta_est_lasso = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso,N2_lasso] = check_IC_2(beta_true,beta_est_lasso);
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_ls,N2_scad_ls] = check_IC_2(beta_true,beta_est_scad_ls);
    NZ_scad_ls = check_NZ(beta_true,beta_est_scad_ls);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l_ls,N2_scad_l_ls] = check_IC_2(beta_true,beta_est_scad_l_ls);
    NZ_scad_l_ls = check_NZ(beta_true,beta_est_scad_l_ls);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_ls,N2_mcp_ls] = check_IC_2(beta_true,beta_est_mcp_ls);
    NZ_mcp_ls = check_NZ(beta_true,beta_est_mcp_ls);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l_ls,N2_mcp_l_ls] = check_IC_2(beta_true,beta_est_mcp_l_ls);
    NZ_mcp_l_ls = check_NZ(beta_true,beta_est_mcp_l_ls);
    
    % lasso penalization
    beta_est_lasso_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso_ls,N2_lasso_ls] = check_IC_2(beta_true,beta_est_lasso_ls);
    NZ_lasso_ls = check_NZ(beta_true,beta_est_lasso_ls);
    
    % number of zero coefficients correctly identified
    check(oo,:) = [check_zero(beta_true,beta_est_scad) check_zero(beta_true,beta_est_scad_l)...
        check_zero(beta_true,beta_est_scad_ls) check_zero(beta_true,beta_est_scad_l_ls)...
        check_zero(beta_true,beta_est_mcp) check_zero(beta_true,beta_est_mcp_l)...
        check_zero(beta_true,beta_est_mcp_ls) check_zero(beta_true,beta_est_mcp_l_ls)...
        check_zero(beta_true,beta_est_lasso) check_zero(beta_true,beta_est_lasso_ls)];
    
    % proportion of zero coefficients correctly identified
    check_prop(oo,:) = check(oo,:)./sparsity;
    
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,:) = [NZ_scad NZ_scad_l NZ_scad_ls NZ_scad_l_ls NZ_mcp NZ_mcp_l NZ_mcp_ls NZ_mcp_l_ls NZ_lasso NZ_lasso_ls]./(dim-sparsity);
    
    % N1: number of true non-zero coefficients estimated as zero
    % coefficients
    % N2: number of true zero coefficients estimated as non-zero
    % coefficients
    check2(oo,:) = [ N1_scad N2_scad N1_scad_l N2_scad_l N1_scad_ls N2_scad_ls N1_scad_l_ls N2_scad_l_ls...
        N1_mcp N2_mcp N1_mcp_l N2_mcp_l N1_mcp_ls N2_mcp_ls N1_mcp_l_ls N2_mcp_l_ls...
        N1_lasso N2_lasso N1_lasso_ls N2_lasso_ls ];
    
    % Mean squared error
    MSE(oo,:) = [ mse(beta_true,beta_est_scad) mse(beta_true,beta_est_scad_l)...
        mse(beta_true,beta_est_scad_ls) mse(beta_true,beta_est_scad_l_ls)...
        mse(beta_true,beta_est_mcp) mse(beta_true,beta_est_mcp_l)...
        mse(beta_true,beta_est_mcp_ls) mse(beta_true,beta_est_mcp_l_ls)...
        mse(beta_true,beta_est_lasso) mse(beta_true,beta_est_lasso_ls) ];
    
end

% Definition of the variables:
% beta_est_scad: Gaussian loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_l: Gaussian loss with SCAD penalization, a_scad = 40
% beta_est_scad_ls: least squares loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_ls_l: least squares loss with SCAD penalization, a_scad = 40

% beta_est_mcp: Gaussian loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_l: Gaussian loss with MCP penalization, b_mcp = 40
% beta_est_mcp_ls: least squares loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_ls_l: least squares loss with MCP penalization, b_mcp = 40

% beta_est_lasso: Gaussian loss with LASSO penalization
% beta_est_lasso_ls: least squares loss with LASSO penalization

% Results_1 provides the proportion of true zeros correctly identified, the proportion of true non-zeros correctly identified and MSE
Results_1 = [mean(check_prop);mean(check_prop2);mean(MSE)];

% Column entries in check_prop, check_prop2 and MSE and so in Results_1:
% 1. Gaussian loss with SCAD (a_scad = 3.7)
% 2. Gaussian loss with SCAD (a_scad = 40)
% 3. Least squares loss with SCAD (a_scad = 3.7)
% 4. Least squares loss with SCAD (a_scad = 40)
% 5. Gaussian loss with MCP (b_mcp = 3.5)
% 6. Gaussian loss with MCP (b_mcp = 40)
% 7. Least squares loss with MCP (b_mcp = 3.5)
% 8. Least squares loss with MCP (b_mcp = 40)
% 9. Gaussian loss with LASSO
% 10. Least squares loss with LASSO

%% Sample size n = 1000, dimension p = 10
clear; clc;
rng(2, 'twister' );
Nsim = 200;
check = zeros(Nsim,10); check_prop = zeros(Nsim,10); check_prop2 = zeros(Nsim,10);
check2 = zeros(Nsim,20); MSE = zeros(Nsim,10);

% Sample size and dimension
n = 1000; p = 10;

% Total number of distinct parameters (size of theta) located in the lower
% tringular part of the correlation matrix
dim = p*(p-1)/2;

% Sparsity degree, that is the proportion of zero coefficients in the lower
% triangular part of the correlation matrix
sparsity = round(dim*0.85);

% Threshold: controls for the number of zero coefficients generated in the
% lower (and so upper) triangular part of the correlation matrix
% a1 and a2: lower and upper bounds when generating the true non-zero
% coefficients of the lower (and so upper) triangular part of the
% correlation matrix in the uniform distribution
threshold = 0.8; a1 = 0.05; a2 = 0.7;

% Number of folds and specification of the grid for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

for oo = 1:Nsim
    
    % simulate the sparse correlation matrix Gaussian copula parameter
    % for each simulation, the size of the true support is unchanged
    Sigma = simulate_sparse_correlation(p,sparsity,threshold,a1,a2);
    beta_true = vech_on(Sigma,p);
    
    % generate the true uniform observations on (0,1)^p
    U = copularnd('Gaussian',Sigma,n);
    % apply a rank-based transformation to get the pseudo-observations
    % hat{U}
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(dim)/n);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad,N2_scad] = check_IC_2(beta_true,beta_est_scad);
    NZ_scad = check_NZ(beta_true,beta_est_scad);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l,N2_scad_l] = check_IC_2(beta_true,beta_est_scad_l);
    NZ_scad_l = check_NZ(beta_true,beta_est_scad_l);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp,N2_mcp] = check_IC_2(beta_true,beta_est_mcp);
    NZ_mcp = check_NZ(beta_true,beta_est_mcp);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l,N2_mcp_l] = check_IC_2(beta_true,beta_est_mcp_l);
    NZ_mcp_l = check_NZ(beta_true,beta_est_mcp_l);
    
    % lasso penalization
    beta_est_lasso = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso,N2_lasso] = check_IC_2(beta_true,beta_est_lasso);
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_ls,N2_scad_ls] = check_IC_2(beta_true,beta_est_scad_ls);
    NZ_scad_ls = check_NZ(beta_true,beta_est_scad_ls);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l_ls,N2_scad_l_ls] = check_IC_2(beta_true,beta_est_scad_l_ls);
    NZ_scad_l_ls = check_NZ(beta_true,beta_est_scad_l_ls);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_ls,N2_mcp_ls] = check_IC_2(beta_true,beta_est_mcp_ls);
    NZ_mcp_ls = check_NZ(beta_true,beta_est_mcp_ls);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l_ls,N2_mcp_l_ls] = check_IC_2(beta_true,beta_est_mcp_l_ls);
    NZ_mcp_l_ls = check_NZ(beta_true,beta_est_mcp_l_ls);
    
    % lasso penalization
    beta_est_lasso_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso_ls,N2_lasso_ls] = check_IC_2(beta_true,beta_est_lasso_ls);
    NZ_lasso_ls = check_NZ(beta_true,beta_est_lasso_ls);
    
    % number of zero coefficients correctly identified
    check(oo,:) = [check_zero(beta_true,beta_est_scad) check_zero(beta_true,beta_est_scad_l)...
        check_zero(beta_true,beta_est_scad_ls) check_zero(beta_true,beta_est_scad_l_ls)...
        check_zero(beta_true,beta_est_mcp) check_zero(beta_true,beta_est_mcp_l)...
        check_zero(beta_true,beta_est_mcp_ls) check_zero(beta_true,beta_est_mcp_l_ls)...
        check_zero(beta_true,beta_est_lasso) check_zero(beta_true,beta_est_lasso_ls)];
    
    % proportion of zero coefficients correctly identified
    check_prop(oo,:) = check(oo,:)./sparsity;
    
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,:) = [NZ_scad NZ_scad_l NZ_scad_ls NZ_scad_l_ls NZ_mcp NZ_mcp_l NZ_mcp_ls NZ_mcp_l_ls NZ_lasso NZ_lasso_ls]./(dim-sparsity);
    
    % N1: number of true non-zero coefficients estimated as zero
    % coefficients
    % N2: number of true zero coefficients estimated as non-zero
    % coefficients
    check2(oo,:) = [ N1_scad N2_scad N1_scad_l N2_scad_l N1_scad_ls N2_scad_ls N1_scad_l_ls N2_scad_l_ls...
        N1_mcp N2_mcp N1_mcp_l N2_mcp_l N1_mcp_ls N2_mcp_ls N1_mcp_l_ls N2_mcp_l_ls...
        N1_lasso N2_lasso N1_lasso_ls N2_lasso_ls ];
    
    % Mean squared error
    MSE(oo,:) = [ mse(beta_true,beta_est_scad) mse(beta_true,beta_est_scad_l)...
        mse(beta_true,beta_est_scad_ls) mse(beta_true,beta_est_scad_l_ls)...
        mse(beta_true,beta_est_mcp) mse(beta_true,beta_est_mcp_l)...
        mse(beta_true,beta_est_mcp_ls) mse(beta_true,beta_est_mcp_l_ls)...
        mse(beta_true,beta_est_lasso) mse(beta_true,beta_est_lasso_ls) ];
    
end

% Definition of the variables:
% beta_est_scad: Gaussian loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_l: Gaussian loss with SCAD penalization, a_scad = 40
% beta_est_scad_ls: least squares loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_ls_l: least squares loss with SCAD penalization, a_scad = 40

% beta_est_mcp: Gaussian loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_l: Gaussian loss with MCP penalization, b_mcp = 40
% beta_est_mcp_ls: least squares loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_ls_l: least squares loss with MCP penalization, b_mcp = 40

% beta_est_lasso: Gaussian loss with LASSO penalization
% beta_est_lasso_ls: least squares loss with LASSO penalization

% Results_2 provides the proportion of true zeros correctly identified, the proportion of true non-zeros correctly identified and MSE
Results_2 = [mean(check_prop);mean(check_prop2);mean(MSE)];

% Column entries in check_prop, check_prop2 and MSE and so in Results_2:
% 1. Gaussian loss with SCAD (a_scad = 3.7)
% 2. Gaussian loss with SCAD (a_scad = 40)
% 3. Least squares loss with SCAD (a_scad = 3.7)
% 4. Least squares loss with SCAD (a_scad = 40)
% 5. Gaussian loss with MCP (b_mcp = 3.5)
% 6. Gaussian loss with MCP (b_mcp = 40)
% 7. Least squares loss with MCP (b_mcp = 3.5)
% 8. Least squares loss with MCP (b_mcp = 40)
% 9. Gaussian loss with LASSO
% 10. Least squares loss with LASSO

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sample size n = 500, dimension p = 20
clear; clc;
rng(3, 'twister' );
Nsim = 200;
check = zeros(Nsim,10); check_prop = zeros(Nsim,10); check_prop2 = zeros(Nsim,10);
check2 = zeros(Nsim,20); MSE = zeros(Nsim,10);

% Sample size and dimension
n = 500; p = 20;

% Total number of distinct parameters (size of theta) located in the lower
% tringular part of the correlation matrix
dim = p*(p-1)/2;

% Sparsity degree, that is the proportion of zero coefficients in the lower
% triangular part of the correlation matrix
sparsity = round(dim*0.90);

% Threshold: controls for the number of zero coefficients generated in the
% lower (and so upper) triangular part of the correlation matrix
% a1 and a2: lower and upper bounds when generating the true non-zero
% coefficients of the lower (and so upper) triangular part of the
% correlation matrix in the uniform distribution
threshold = 0.85; a1 = 0.05; a2 = 0.7;

% Number of folds and specification of the grid for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

for oo = 1:Nsim
    
    % simulate the sparse correlation matrix Gaussian copula parameter
    % for each simulation, the size of the true support is unchanged
    Sigma = simulate_sparse_correlation(p,sparsity,threshold,a1,a2);
    beta_true = vech_on(Sigma,p);
    
    % generate the true uniform observations on (0,1)^p
    U = copularnd('Gaussian',Sigma,n);
    % apply a rank-based transformation to get the pseudo-observations
    % hat{U}
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(dim)/n);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad,N2_scad] = check_IC_2(beta_true,beta_est_scad);
    NZ_scad = check_NZ(beta_true,beta_est_scad);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l,N2_scad_l] = check_IC_2(beta_true,beta_est_scad_l);
    NZ_scad_l = check_NZ(beta_true,beta_est_scad_l);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp,N2_mcp] = check_IC_2(beta_true,beta_est_mcp);
    NZ_mcp = check_NZ(beta_true,beta_est_mcp);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l,N2_mcp_l] = check_IC_2(beta_true,beta_est_mcp_l);
    NZ_mcp_l = check_NZ(beta_true,beta_est_mcp_l);
    
    % lasso penalization
    beta_est_lasso = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso,N2_lasso] = check_IC_2(beta_true,beta_est_lasso);
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_ls,N2_scad_ls] = check_IC_2(beta_true,beta_est_scad_ls);
    NZ_scad_ls = check_NZ(beta_true,beta_est_scad_ls);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l_ls,N2_scad_l_ls] = check_IC_2(beta_true,beta_est_scad_l_ls);
    NZ_scad_l_ls = check_NZ(beta_true,beta_est_scad_l_ls);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_ls,N2_mcp_ls] = check_IC_2(beta_true,beta_est_mcp_ls);
    NZ_mcp_ls = check_NZ(beta_true,beta_est_mcp_ls);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l_ls,N2_mcp_l_ls] = check_IC_2(beta_true,beta_est_mcp_l_ls);
    NZ_mcp_l_ls = check_NZ(beta_true,beta_est_mcp_l_ls);
    
    % lasso penalization
    beta_est_lasso_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso_ls,N2_lasso_ls] = check_IC_2(beta_true,beta_est_lasso_ls);
    NZ_lasso_ls = check_NZ(beta_true,beta_est_lasso_ls);
    
    % number of zero coefficients correctly identified
    check(oo,:) = [check_zero(beta_true,beta_est_scad) check_zero(beta_true,beta_est_scad_l)...
        check_zero(beta_true,beta_est_scad_ls) check_zero(beta_true,beta_est_scad_l_ls)...
        check_zero(beta_true,beta_est_mcp) check_zero(beta_true,beta_est_mcp_l)...
        check_zero(beta_true,beta_est_mcp_ls) check_zero(beta_true,beta_est_mcp_l_ls)...
        check_zero(beta_true,beta_est_lasso) check_zero(beta_true,beta_est_lasso_ls)];
    
    % proportion of zero coefficients correctly identified
    check_prop(oo,:) = check(oo,:)./sparsity;
    
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,:) = [NZ_scad NZ_scad_l NZ_scad_ls NZ_scad_l_ls NZ_mcp NZ_mcp_l NZ_mcp_ls NZ_mcp_l_ls NZ_lasso NZ_lasso_ls]./(dim-sparsity);
    
    % N1: number of true non-zero coefficients estimated as zero
    % coefficients
    % N2: number of true zero coefficients estimated as non-zero
    % coefficients
    check2(oo,:) = [ N1_scad N2_scad N1_scad_l N2_scad_l N1_scad_ls N2_scad_ls N1_scad_l_ls N2_scad_l_ls...
        N1_mcp N2_mcp N1_mcp_l N2_mcp_l N1_mcp_ls N2_mcp_ls N1_mcp_l_ls N2_mcp_l_ls...
        N1_lasso N2_lasso N1_lasso_ls N2_lasso_ls ];
    
    % Mean squared error
    MSE(oo,:) = [ mse(beta_true,beta_est_scad) mse(beta_true,beta_est_scad_l)...
        mse(beta_true,beta_est_scad_ls) mse(beta_true,beta_est_scad_l_ls)...
        mse(beta_true,beta_est_mcp) mse(beta_true,beta_est_mcp_l)...
        mse(beta_true,beta_est_mcp_ls) mse(beta_true,beta_est_mcp_l_ls)...
        mse(beta_true,beta_est_lasso) mse(beta_true,beta_est_lasso_ls) ];
    
end

% Definition of the variables:
% beta_est_scad: Gaussian loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_l: Gaussian loss with SCAD penalization, a_scad = 40
% beta_est_scad_ls: least squares loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_ls_l: least squares loss with SCAD penalization, a_scad = 40

% beta_est_mcp: Gaussian loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_l: Gaussian loss with MCP penalization, b_mcp = 40
% beta_est_mcp_ls: least squares loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_ls_l: least squares loss with MCP penalization, b_mcp = 40

% beta_est_lasso: Gaussian loss with LASSO penalization
% beta_est_lasso_ls: least squares loss with LASSO penalization

% Results_3 provides the proportion of true zeros correctly identified, the proportion of true non-zeros correctly identified and MSE
Results_3 = [mean(check_prop);mean(check_prop2);mean(MSE)];

% Column entries in check_prop, check_prop2 and MSE and so in Results_3:
% 1. Gaussian loss with SCAD (a_scad = 3.7)
% 2. Gaussian loss with SCAD (a_scad = 40)
% 3. Least squares loss with SCAD (a_scad = 3.7)
% 4. Least squares loss with SCAD (a_scad = 40)
% 5. Gaussian loss with MCP (b_mcp = 3.5)
% 6. Gaussian loss with MCP (b_mcp = 40)
% 7. Least squares loss with MCP (b_mcp = 3.5)
% 8. Least squares loss with MCP (b_mcp = 40)
% 9. Gaussian loss with LASSO
% 10. Least squares loss with LASSO

%% Sample size n = 1000, dimension p = 20
clear; clc;
rng(4, 'twister' );
Nsim = 200;
check = zeros(Nsim,10); check_prop = zeros(Nsim,10); check_prop2 = zeros(Nsim,10);
check2 = zeros(Nsim,20); MSE = zeros(Nsim,10);

% Sample size and dimension
n = 1000; p = 20;

% Total number of distinct parameters (size of theta) located in the lower
% tringular part of the correlation matrix
dim = p*(p-1)/2;

% Sparsity degree, that is the proportion of zero coefficients in the lower
% triangular part of the correlation matrix
sparsity = round(dim*0.90);

% Threshold: controls for the number of zero coefficients generated in the
% lower (and so upper) triangular part of the correlation matrix
% a1 and a2: lower and upper bounds when generating the true non-zero
% coefficients of the lower (and so upper) triangular part of the
% correlation matrix in the uniform distribution
threshold = 0.85; a1 = 0.05; a2 = 0.7;

% Number of folds and specification of the grid for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

for oo = 1:Nsim
    
    % simulate the sparse correlation matrix Gaussian copula parameter
    % for each simulation, the size of the true support is unchanged
    Sigma = simulate_sparse_correlation(p,sparsity,threshold,a1,a2);
    beta_true = vech_on(Sigma,p);
    
    % generate the true uniform observations on (0,1)^p
    U = copularnd('Gaussian',Sigma,n);
    % apply a rank-based transformation to get the pseudo-observations
    % hat{U}
    U_po = pseudoObservations(U);
    
    lambda = grid*sqrt(log(dim)/n);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad,N2_scad] = check_IC_2(beta_true,beta_est_scad);
    NZ_scad = check_NZ(beta_true,beta_est_scad);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l,N2_scad_l] = check_IC_2(beta_true,beta_est_scad_l);
    NZ_scad_l = check_NZ(beta_true,beta_est_scad_l);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp,N2_mcp] = check_IC_2(beta_true,beta_est_mcp);
    NZ_mcp = check_NZ(beta_true,beta_est_mcp);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l,N2_mcp_l] = check_IC_2(beta_true,beta_est_mcp_l);
    NZ_mcp_l = check_NZ(beta_true,beta_est_mcp_l);
    
    % lasso penalization
    beta_est_lasso = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso,N2_lasso] = check_IC_2(beta_true,beta_est_lasso);
    NZ_lasso = check_NZ(beta_true,beta_est_lasso);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    % scad penalization 3.7
    a_scad = 3.7;
    beta_est_scad_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_ls,N2_scad_ls] = check_IC_2(beta_true,beta_est_scad_ls);
    NZ_scad_ls = check_NZ(beta_true,beta_est_scad_ls);
    
    % scad penalization 40
    a_scad = 40;
    beta_est_scad_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad,0,K);
    [N1_scad_l_ls,N2_scad_l_ls] = check_IC_2(beta_true,beta_est_scad_l_ls);
    NZ_scad_l_ls = check_NZ(beta_true,beta_est_scad_l_ls);
    
    % mcp penalization 3.5
    b_mcp = 3.5;
    beta_est_mcp_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_ls,N2_mcp_ls] = check_IC_2(beta_true,beta_est_mcp_ls);
    NZ_mcp_ls = check_NZ(beta_true,beta_est_mcp_ls);
    
    % mcp penalization 40
    b_mcp = 40;
    beta_est_mcp_l_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp,K);
    [N1_mcp_l_ls,N2_mcp_l_ls] = check_IC_2(beta_true,beta_est_mcp_l_ls);
    NZ_mcp_l_ls = check_NZ(beta_true,beta_est_mcp_l_ls);
    
    % lasso penalization
    beta_est_lasso_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'lasso',0,0,K);
    [N1_lasso_ls,N2_lasso_ls] = check_IC_2(beta_true,beta_est_lasso_ls);
    NZ_lasso_ls = check_NZ(beta_true,beta_est_lasso_ls);
    
    % number of zero coefficients correctly identified
    check(oo,:) = [check_zero(beta_true,beta_est_scad) check_zero(beta_true,beta_est_scad_l)...
        check_zero(beta_true,beta_est_scad_ls) check_zero(beta_true,beta_est_scad_l_ls)...
        check_zero(beta_true,beta_est_mcp) check_zero(beta_true,beta_est_mcp_l)...
        check_zero(beta_true,beta_est_mcp_ls) check_zero(beta_true,beta_est_mcp_l_ls)...
        check_zero(beta_true,beta_est_lasso) check_zero(beta_true,beta_est_lasso_ls)];
    
    % proportion of zero coefficients correctly identified
    check_prop(oo,:) = check(oo,:)./sparsity;
    
    % proportion of non-zero coefficients correctly identified
    check_prop2(oo,:) = [NZ_scad NZ_scad_l NZ_scad_ls NZ_scad_l_ls NZ_mcp NZ_mcp_l NZ_mcp_ls NZ_mcp_l_ls NZ_lasso NZ_lasso_ls]./(dim-sparsity);
    
    % N1: number of true non-zero coefficients estimated as zero
    % coefficients
    % N2: number of true zero coefficients estimated as non-zero
    % coefficients
    check2(oo,:) = [ N1_scad N2_scad N1_scad_l N2_scad_l N1_scad_ls N2_scad_ls N1_scad_l_ls N2_scad_l_ls...
        N1_mcp N2_mcp N1_mcp_l N2_mcp_l N1_mcp_ls N2_mcp_ls N1_mcp_l_ls N2_mcp_l_ls...
        N1_lasso N2_lasso N1_lasso_ls N2_lasso_ls ];
    
    % Mean squared error
    MSE(oo,:) = [ mse(beta_true,beta_est_scad) mse(beta_true,beta_est_scad_l)...
        mse(beta_true,beta_est_scad_ls) mse(beta_true,beta_est_scad_l_ls)...
        mse(beta_true,beta_est_mcp) mse(beta_true,beta_est_mcp_l)...
        mse(beta_true,beta_est_mcp_ls) mse(beta_true,beta_est_mcp_l_ls)...
        mse(beta_true,beta_est_lasso) mse(beta_true,beta_est_lasso_ls) ];
    
end

% Definition of the variables:
% beta_est_scad: Gaussian loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_l: Gaussian loss with SCAD penalization, a_scad = 40
% beta_est_scad_ls: least squares loss with SCAD penalization, a_scad = 3.7
% beta_est_scad_ls_l: least squares loss with SCAD penalization, a_scad = 40

% beta_est_mcp: Gaussian loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_l: Gaussian loss with MCP penalization, b_mcp = 40
% beta_est_mcp_ls: least squares loss with MCP penalization, b_mcp = 3.5
% beta_est_mcp_ls_l: least squares loss with MCP penalization, b_mcp = 40

% beta_est_lasso: Gaussian loss with LASSO penalization
% beta_est_lasso_ls: least squares loss with LASSO penalization

% Results_4 provides the proportion of true zeros correctly identified, the proportion of true non-zeros correctly identified and MSE
Results_4 = [mean(check_prop);mean(check_prop2);mean(MSE)];

% Column entries in check_prop, check_prop2 and MSE and so in Results_4:
% 1. Gaussian loss with SCAD (a_scad = 3.7)
% 2. Gaussian loss with SCAD (a_scad = 40)
% 3. Least squares loss with SCAD (a_scad = 3.7)
% 4. Least squares loss with SCAD (a_scad = 40)
% 5. Gaussian loss with MCP (b_mcp = 3.5)
% 6. Gaussian loss with MCP (b_mcp = 40)
% 7. Least squares loss with MCP (b_mcp = 3.5)
% 8. Least squares loss with MCP (b_mcp = 40)
% 9. Gaussian loss with LASSO
% 10. Least squares loss with LASSO