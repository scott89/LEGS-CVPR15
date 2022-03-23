function [y] = getLabel(mask, objMask)
mask = mask(:,:,1);
mask = mat2gray(mask)>0.5;
objMask = objMask>0;

intersactionMask = double(bsxfun(@and, objMask, mask));
unionMask = double(bsxfun(@or, objMask, mask));
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
