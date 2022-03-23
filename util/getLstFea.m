function  [lstFeature] = getLstFea(lstMap, objMask)

[~, ~, pNum] = size(objMask);
lstMap = lstMap>0;
objMask = objMask>0;
intersactionMap = double(bsxfun(@and, objMask, lstMap));
unionMap = double(bsxfun(@or, objMask, lstMap));
intersaction = sum(sum(intersactionMap));
intersaction = reshape(intersaction, pNum, 1);
union = sum(sum(unionMap));
union = union(:);


objArea = sum(sum(double(objMask)));
objArea = reshape(objArea, pNum, 1);
salArea = sum(double(lstMap(:)));
Pre = intersaction./objArea;
Rc = intersaction/salArea;
% PRs = (1+0.3)*Pre.*Rc./(0.3*Pre+Rc);
PRs = Pre.*Rc;
overlap = intersaction./union;
lstFeature = [Pre, Rc, PRs, overlap];
