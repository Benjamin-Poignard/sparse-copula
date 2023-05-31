function H = proj_defpos(M)

[P,K] = eig(M); K = diag(K);
K = subplus(K)+0.01; K = diag(K);
H = P*K*P';