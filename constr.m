function [c,ceq] = constr(param)

c = abs(param)-3;
ceq = [];