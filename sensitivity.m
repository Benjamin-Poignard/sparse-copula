clear; clc;
rng(8, 'twister' )
% Number of batches
Nsim = 100;
check = zeros(Nsim,10); check_prop = zeros(Nsim,10); check_prop2 = zeros(Nsim,10);
check2 = zeros(Nsim,20); MSE = zeros(Nsim,10);

% Sample size and dimension
n = 500; p = 10; 

% Total number of distinct parameters (size of theta) located in the lower
% tringular part of the correlation matrix
dim = p*(p-1)/2;

% Sparsity degree, that is the proportion of zero coefficients in the lower
% triangular part of the correlation matrix
sparsity = round(dim*0.85); threshold = 0.8;

% True covariance (correlation) matrix of the Gaussian copula: fixed once
Sigma = [
    1.0000         0         0         0    0.3856         0         0         0         0    0.3873
         0    1.0000         0    0.0888         0         0         0         0    0.0964         0
         0         0    1.0000         0         0         0   -0.0631   -0.2680         0         0
         0    0.0888         0    1.0000         0         0         0         0    0.4417         0
    0.3856         0         0         0    1.0000         0         0         0         0         0
         0         0         0         0         0    1.0000         0         0         0         0
         0         0   -0.0631         0         0         0    1.0000         0         0         0
         0         0   -0.2680         0         0         0         0    1.0000         0         0
         0    0.0964         0    0.4417         0         0         0         0    1.0000         0
    0.3873         0         0         0         0         0         0         0         0    1.0000    
    ];
% vectorization of the true parameter (lower triangular part of Sigma,
% excluding the diagonal terms)
beta_true = vech_on(Sigma,p);

% Number of folds and specification of the grid for cross-validation
K = 5; grid = [0.01,0.05:0.05:4.5];

a_scad = [2.1 2.5:0.5:42]; b_mcp = [0.1 0.5:0.5:40]; len = length(a_scad);

mm1_scad_g = []; mm2_scad_g = []; mm3_scad_g = [];
mm1_mcp_g = []; mm2_mcp_g = []; mm3_mcp_g = [];

mm1_scad_ls = []; mm2_scad_ls = []; mm3_scad_ls = [];
mm1_mcp_ls = []; mm2_mcp_ls = []; mm3_mcp_ls = [];

for oo = 1:Nsim

    % generate the true uniform observations on (0,1)^p
    U = copularnd('Gaussian',Sigma,n);
    % apply a rank-based transformation to get the pseudo-observations
    % \hat{U}
    U_po = pseudoObservations(U);
    
    check_prop_scad_g = zeros(1,len); check_prop2_scad_g = zeros(1,len);
    loss1_scad_g = zeros(1,len);
    check_prop_scad_ls = zeros(1,len); check_prop2_scad_ls = zeros(1,len);
    loss1_scad_ls = zeros(1,len);
    
    check_prop_mcp_g = zeros(1,len); check_prop2_mcp_g = zeros(1,len);
    loss1_mcp_g = zeros(1,len);
    check_prop_mcp_ls = zeros(1,len); check_prop2_mcp_ls = zeros(1,len);
    loss1_mcp_ls = zeros(1,len);
    
    for ii = 1:length(a_scad)
        
        lambda = grid*sqrt(log(dim)/n);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        loss = 'Gaussian';
        
        beta_est_scad = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad(ii),0,K);
        [N1_scad,N2_scad] = check_IC_2(beta_true,beta_est_scad);
        NZ_scad = check_NZ(beta_true,beta_est_scad);
        
        [beta_est_mcp,opt] = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp(ii),K);
        [N1_mcp,N2_mcp] = check_IC_2(beta_true,beta_est_mcp);
        NZ_mcp = check_NZ(beta_true,beta_est_mcp);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        loss = 'LS';
        
        beta_est_scad_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'scad',a_scad(ii),0,K);
        [N1_scad_ls,N2_scad_ls] = check_IC_2(beta_true,beta_est_scad_ls);
        NZ_scad_ls = check_NZ(beta_true,beta_est_scad_ls);
        
        beta_est_mcp_ls = sparse_gaussian_copula(norminv(U_po),lambda,loss,'mcp',0,b_mcp(ii),K);
        [N1_mcp_ls,N2_mcp_ls] = check_IC_2(beta_true,beta_est_mcp_ls);
        NZ_mcp_ls = check_NZ(beta_true,beta_est_mcp_ls);
        
        % Proportion of zero coefficients correctly identified,
        % Gaussian-based estimator
        check_prop_scad_g(ii) = check_zero(beta_true,beta_est_scad)/sparsity;
        
        check_prop2_scad_g(ii) =  NZ_scad/(dim-sparsity);
        
        loss1_scad_g(ii) = vec(beta_true-beta_est_scad)'*vec((beta_true)-(beta_est_scad));
        
        check_prop_mcp_g(ii) = check_zero(beta_true,beta_est_mcp)/sparsity;
        
        check_prop2_mcp_g(ii) =  NZ_mcp/(dim-sparsity);
        
        loss1_mcp_g(ii) = vec(beta_true-beta_est_mcp)'*vec((beta_true)-(beta_est_mcp));
        
        % Proportion of zero coefficients correctly identified,
        % Least squares-based estimator
        check_prop_scad_ls(ii) = check_zero(beta_true,beta_est_scad_ls)/sparsity;
        
        check_prop2_scad_ls(ii) =  NZ_scad_ls/(dim-sparsity);
        
        loss1_scad_ls(ii) =vec(beta_true-beta_est_scad_ls)'*vec((beta_true)-(beta_est_scad_ls));
        
        check_prop_mcp_ls(ii) = check_zero(beta_true,beta_est_mcp_ls)/sparsity;
        
        check_prop2_mcp_ls(ii) =  NZ_mcp_ls/(dim-sparsity);
        
        loss1_mcp_ls(ii) = vec(beta_true-beta_est_mcp_ls)'*vec((beta_true)-(beta_est_mcp_ls));
        
    end
    
    mm1_scad_g = [ mm1_scad_g ; check_prop_scad_g];
    mm2_scad_g = [ mm2_scad_g ; check_prop2_scad_g];
    mm3_scad_g = [ mm3_scad_g ; loss1_scad_g];
    
    mm1_mcp_g = [ mm1_mcp_g ; check_prop_mcp_g];
    mm2_mcp_g = [ mm2_mcp_g ; check_prop2_mcp_g];
    mm3_mcp_g = [ mm3_mcp_g ; loss1_mcp_g];
    
    mm1_scad_ls = [ mm1_scad_ls ; check_prop_scad_ls];
    mm2_scad_ls = [ mm2_scad_ls ; check_prop2_scad_ls];
    mm3_scad_ls = [ mm3_scad_ls ; loss1_scad_ls];
    
    mm1_mcp_ls = [ mm1_mcp_ls ; check_prop_mcp_ls];
    mm2_mcp_ls = [ mm2_mcp_ls ; check_prop2_mcp_ls];
    mm3_mcp_ls = [ mm3_mcp_ls ; loss1_mcp_ls];
    
end
% mm1_scad_g mm2_scad_g mm3_scad_g mm1_mcp_g mm2_mcp_g mm3_mcp_g 
% mm1_scad_ls mm2_scad_ls mm3_scad_ls mm1_mcp_ls mm2_mcp_ls mm3_mcp_ls are
% 100 x length(a_scad) or 100 x length(b_mcp) matrices

save('Metrics.mat','mm1_scad_g','mm2_scad_g','mm3_scad_g','mm1_mcp_g','mm2_mcp_g','mm3_mcp_g','mm1_scad_ls','mm2_scad_ls','mm3_scad_ls','mm1_mcp_ls','mm2_mcp_ls','mm3_mcp_ls')

%% Replicate Figure 1: SCAD penalization
load Metrics.mat
%title('SCAD - zero recovery (C1) and non-zero recovery (C2)')
hold on; plot(mean(mm1_scad_g(:,1:77)),'r','LineWidth',1.5); plot(mean(mm2_scad_g(:,1:77)),'r','LineStyle','-.','LineWidth',1.5); ...
    plot(mean(mm1_scad_ls(:,1:77)),'b','LineWidth',1.5); plot(mean(mm2_scad_ls(:,1:77)),'b','LineStyle','-.','LineWidth',1.5)
ylim([0 1])
yticks(0:0.1:1);
yticklabels({'0','10','20','30','40','50','60','70','80','90','100'})


% xticks(5:5:80); 
% xticklabels({'4','6.5','9','11.5','14','16.5','19','21.5','24','26.5','29','31.5','34','36.5','39','41.5'})
xticks(5:5:75); 
xticklabels({'4','6.5','9','11.5','14','16.5','19','21.5','24','26.5','29','31.5','34','36.5','39'})

legend({'C1 - Gaussian loss','C2 - Gaussian loss','C1 - Least squares loss','C2 - Least squares loss'},'Location','southwest')
ylabel('Proportion in %') 
xlabel('a_{scad}') 

figure
%title('SCAD - mean squared error')
hold on; plot(mean(mm3_scad_g(:,1:77)),'r','LineWidth',1.5); plot(mean(mm3_scad_ls(:,1:77)),'b','LineWidth',1.5);
ylim([0 0.05])
yticks(0:0.005:0.05);

% xticks(5:5:80); 
xticks(5:5:75); 
xticklabels({'4','6.5','9','11.5','14','16.5','19','21.5','24','26.5','29','31.5','34','36.5','39'})

legend({'Gaussian loss','Least squares loss'},'Location','southwest')
ylabel('MSE') 
xlabel('a_{scad}') 

%% Replicate Figure 1: MCP penalization
%title('MCP - zero recovery (C1) and non-zero recovery (C2)')

hold on; plot(mean(mm1_mcp_g),'r','LineWidth',1.5); plot(mean(mm2_mcp_g),'r','LineStyle','-.','LineWidth',1.5);...
    plot(mean(mm1_mcp_ls),'b','LineWidth',1.5); plot(mean(mm2_mcp_ls),'b','LineStyle','-.','LineWidth',1.5)
ylim([0 1])
yticks(0:0.1:1);
yticklabels({'0','10','20','30','40','50','60','70','80','90','100'})

xticks(5:5:80); 
xticklabels({'2','4.5','7','9.5','12','14.5','17','19.5','22','24.5','27','29.5','32','34.5','37','39.5'})

legend({'C1 - Gaussian loss','C2 - Gaussian loss','C1 - Least squares loss','C2 - Least squares loss'},'Location','southwest')
ylabel('Proportion in %') 
xlabel('b_{mcp}') 

figure
%title('MCP - mean squared error')
hold on; plot(mean(mm3_mcp_g),'r','LineWidth',1.5); plot(mean(mm3_mcp_ls),'b','LineWidth',1.5);
ylim([0 0.08])
yticks(0:0.005:0.08);

xticks(5:5:80); 
xticklabels({'2','4.5','7','9.5','12','14.5','17','19.5','22','24.5','27','29.5','32','34.5','37','39.5'})

legend({'Gaussian loss','Least squares loss'},'Location','southwest')
ylabel('MSE') 
xlabel('b_{mcp}') 