function [y] = getLabel1(mask, objMask)
mask = repmat(double(mask), 1,1, size(objMask, 3));
objMask = double(objMask);

intersactionMask = mask.*objMask;

unionMask = double((mask+objMask)>0);
intersaction = sum(sum(intersactionMask));
intersaction = intersaction(:);
union = sum(sum(unionMask));
union = union(:);
objArea = sum(sum(double(objMask)));
objArea = objArea(:);
% GTArea = sum(sum(double(mask)));
y(:, 1) = intersaction./objArea;
y(:, 2) = intersaction./(union+eps);
% confObjId = (y>=0.7 | y<=0.3);
