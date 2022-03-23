function D = x2dist(A, B)
% x2dist compute the x2 distance??? \sum (2*(A_i-B_i)^2/(A_i+B_i)). 
% INPUTE: 
%               - A: a feature matrix of size numSamp x nFea;
%               - B: another feature matrix of size numSamp x nFea or 1 x nFea

Dif = bsxfun(@minus, A, B);
S = bsxfun(@plus, A, B);
D = 0.5*sum(Dif.^2./(S+eps), 1);