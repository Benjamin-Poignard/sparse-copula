function L = mle_gaussian_penalised(param,data,lambda,loss,method,a_scad,b_mcp)


Sigma = vech_off(param,size(data,2));

switch method
    case 'scad'
        pen = scad(param,lambda,a_scad);
    case 'mcp'
        pen = mcp(param,lambda,b_mcp);
    case 'lasso'
        pen = lambda*sum(abs(param));
end
switch loss
    case 'Gaussian'
        L = (trace(cov(data)*inv(Sigma))-log(det(inv(Sigma))))/length(data)+pen;
    case 'LS'
        L = norm(cov(data)-Sigma,'fro')^2+length(data)*pen;
end

% function L = mle_gaussian_penalised(param,data,lambda,loss,method,a_scad,b_mcp)
% 
% 
% Sigma = vech_off(param,size(data,2));
% 
% switch method
%     case 'scad'
%         pen = scad(param,lambda,a_scad);
%     case 'mcp'
%         pen = mcp(param,lambda,b_mcp);
%     case 'lasso'
%         pen = lambda*sum(abs(param));
% end
% switch loss
%     case 'Gaussian'
%         L = 0;
%         for ii = 1:length(data)
%             L = L+log(exp(-0.5*data(ii,:)*(inv(Sigma))*data(ii,:)')/sqrt(det(Sigma)));
%         end
%         L = -L/length(data)+pen;
%     case 'LS'
%         L = norm(cov(data)-Sigma,'fro')^2+length(data)*pen;
% end
