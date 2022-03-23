function ctrFeature = getCtrFeaTest(im, objMask, histParam, lsm, rlsm)
% Compute the color feature for each objce proposal
% INPUT:
%          - im: the color image (dobule [0,1])
%          - objMask: objcet proposal mask
%  OUTPUT:
%         - colFeature: color feature of each objce proposal, 
%         propNum x  feadim, including rgb and lab color histgram
tic;
RGB_bins = histParam.RGB_bins;
Lab_bins = histParam.Lab_bins;
HSV_bins = histParam.HSV_bins;
nRGBHist = histParam.nRGBHist;
nLabHist = histParam.nLabHist;
nHSVHist = histParam.nHSVHist;
% %% remove frame
%  [im, frameRecord] = removeframe(im, 'sobel');
% objMask = objMask(frameRecord(3):frameRecord(4), frameRecord(5):frameRecord(6), :);
%% 
[h, w, propNum] = size(objMask);
imLab = colorspace('->Lab', im);
imLab(:,:,1) = imLab(:,:,1) / 100;
imLab(:,:,2:3) = imLab(:,:,2:3) / 220 + 0.5;
imHSV = rgb2hsv(im);
imR = im(:,:, 1); imG = im(:,:,2); imB = im(:,:,3); imL = imLab(:,:,1); ima = imLab(:,:,2); imb = imLab(:,:,3);
imH = imHSV(:,:,1); imS = imHSV(:,:,2); imV = imHSV(:,:,3);
%% RGB
QR = min( floor(im(:,:,1)*RGB_bins(1)) + 1, RGB_bins(1) );
QG = min( floor(im(:,:,2)*RGB_bins(2)) + 1, RGB_bins(2) );
QB = min( floor(im(:,:,3)*RGB_bins(3)) + 1, RGB_bins(3) );
Q_RGB = (QR-1) * RGB_bins(2) * RGB_bins(3) + ...
    (QG-1) * RGB_bins(3) + ...
    QB-1 ;

% objQ_RGB = bsxfun(@times, objMask, Q_RGB);
%% Lab
QL = min(floor(imLab(:,:,1)*Lab_bins(1))+1, Lab_bins(1));
Qa = min(floor(imLab(:,:,2)*Lab_bins(2))+1, Lab_bins(2));
Qb = min(floor(imLab(:,:,3)*Lab_bins(3))+1, Lab_bins(3));
Q_Lab = (QL-1)*Lab_bins(2)*Lab_bins(3)+...
    (Qa-1)*Lab_bins(3)+...
    Qb-1;

% objQ_Lab = bsxfun(@times, objMask, Q_Lab);
%% HSV
QH = min(floor(imHSV(:,:,1)*HSV_bins(1))+1, HSV_bins(1));
QS = min(floor(imHSV(:,:,2)*HSV_bins(2))+1, HSV_bins(2));
QV = min(floor(imHSV(:,:,2)*HSV_bins(3))+1, HSV_bins(3));
Q_HSV = (QH-1)*HSV_bins(2)*HSV_bins(3)+...
    (QS-1)*HSV_bins(3)+...
    QV-1;
% objQ_S = bsxfun(@times, objMask, QS);
%% Background histogram
%% %%
imdata = struct('imR', imR, 'imG', imG, 'imB', imB, 'imL', imL, 'ima', ima, 'imb', imb, 'imH', imH, 'imS', imS, 'imV', imV,...
    'Q_RGB', Q_RGB, 'Q_Lab', Q_Lab, 'Q_HSV', Q_HSV, 'h', h, 'w', w);
bgFea = getBgFea(imdata, histParam);
%% Global histogram
gQ_RGBHist = hist(Q_RGB(:), 0:nRGBHist-1);
gQ_RGBHist = gQ_RGBHist/sum(gQ_RGBHist);
gQ_LabHist = hist(Q_Lab(:), 0:nLabHist-1);
gQ_LabHist = gQ_LabHist/sum(gQ_LabHist);
gQ_HSVHist = hist(Q_HSV(:), 0:nHSVHist-1);
gQ_HSVHist = gQ_HSVHist/sum(gQ_HSVHist);
%%
objRGBHist = zeros(propNum, nRGBHist);
objLabHist = zeros(propNum, nLabHist);
objHSVHist = zeros(propNum, nHSVHist);
objRGBVar = zeros(propNum, 3);
objLabVar = zeros(propNum, 3);
objHSVVar = zeros(propNum, 3);
objRGBMean = zeros(propNum, 3);
objLabMean = zeros(propNum, 3);
objHSVMean = zeros(propNum, 3);
fea(:,:,:,1) = im;
fea(:,:,:,2) = imLab;
fea(:,:,:,3) = imHSV;
qfea(:,:,1) = Q_RGB;
qfea(:,:,2) = Q_Lab;
qfea(:,:,3) = Q_HSV;
lsm = lsm>0;
rlsm = rlsm>0;
s_lsm = sum(single(lsm(:)));
s_rlsm = sum(single(rlsm(:)));
[fea_hist, fea_mean, fea_var, lf,rlf, geo_fea] = mexComputeFeature(int32(qfea), single(fea), lsm, rlsm, objMask>0, int32(nRGBHist), s_lsm, s_rlsm);
toc;
lstFeature = getLstFea(lsm, objMask);
lstRegFeature = getLstFea(rlsm, objMask);
parfor pId = 1:propNum
    curObjMask = objMask(:,:,pId);
    objRGBHist(pId, :) = hist(Q_RGB(curObjMask>0), 0:nRGBHist-1);
    objLabHist(pId, :) = hist(Q_Lab(curObjMask>0), 0:nLabHist-1);
    objHSVHist(pId, :) = hist(Q_HSV(curObjMask>0), 0:nHSVHist-1);
    % RGB Var
%     objRGBVar(pId, 1) = var(imR(curObjMask>0));
%     objRGBVar(pId, 2) = var(imG(curObjMask>0));
%     objRGBVar(pId, 3) = var(imB(curObjMask>0));
    objRGBVar(pId, :) = [var(imR(curObjMask>0)), var(imG(curObjMask>0)), var(imB(curObjMask>0))];
    %RGB Mean
%     objRGBMean(pId, 1) = mean(imR(curObjMask>0));
%     objRGBMean(pId, 2) = mean(imG(curObjMask>0));
%     objRGBMean(pId, 3) = mean(imB(curObjMask>0));
    objRGBMean(pId, :) = [mean(imR(curObjMask>0)), mean(imG(curObjMask>0)), mean(imB(curObjMask>0))];
    % Lab Var
%     objLabVar(pId, 1) = var(imL(curObjMask>0));
%     objLabVar(pId, 2) = var(ima(curObjMask>0));
%     objLabVar(pId, 3) = var(imb(curObjMask>0));
    objLabVar(pId, :) = [var(imL(curObjMask>0)), var(ima(curObjMask>0)), var(imb(curObjMask>0))];
    %Lab Mean
%     objLabMean(pId, 1) = mean(imL(curObjMask>0));
%     objLabMean(pId, 2) = mean(ima(curObjMask>0));
%     objLabMean(pId, 3) = mean(imb(curObjMask>0));
    objLabMean(pId, :) = [mean(imL(curObjMask>0)), mean(ima(curObjMask>0)), mean(imb(curObjMask>0))];
    % HSV Var
%     objHSVVar(pId, 1) = var(imH(curObjMask>0));
%     objHSVVar(pId, 2) = var(imS(curObjMask>0));
%     objHSVVar(pId, 3) = var(imV(curObjMask>0));
    objHSVVar(pId, :) = [var(imH(curObjMask>0)), var(imS(curObjMask>0)), var(imV(curObjMask>0))];
    %HSV Mean
%     objHSVMean(pId, 1) = mean(imH(curObjMask>0));
%     objHSVMean(pId, 2) = mean(imS(curObjMask>0));
%     objHSVMean(pId, 3) = mean(imV(curObjMask>0));
    objHSVMean(pId, :) = [mean(imH(curObjMask>0)), mean(imS(curObjMask>0)), mean(imV(curObjMask>0))];
    % 
end
objRGBHist = objRGBHist./repmat(sum(objRGBHist, 2), 1, nRGBHist);
objLabHist = objLabHist./repmat(sum(objLabHist, 2), 1, nLabHist);
objHSVHist = objHSVHist./repmat(sum(objHSVHist, 2), 1, nHSVHist);

 hist_test = find(double(fea_hist)-double([objRGBHist'; objLabHist'; objHSVHist']) > 0.00001);
 if(~isempty(hist_test)) 
     fprintf('Warning test for hist computation is not passed\n'); 
 else
     fprintf('Pass Histogram\n');
 end
 lst_test = find(abs(double(lf)-lstFeature') > 1e-5 );
 if(~isempty(lst_test))
     fprintf('Warning test for local saliency computation is not passed\n');
 else
     fprintf('Pass LST\n');
 end
 rlst_test = find(abs(double(rlf)-lstRegFeature') > 1e-5);
 if(~isempty(rlst_test))
     fprintf('Warning test for local saliency computation is not passed\n');
 else
     fprintf('Pass Fine LST\n');
 end
 mean_test = find(abs(double(fea_mean')-[objRGBMean, objLabMean, objHSVMean]) > 1e-5);
 if(~isempty(mean_test))
     fprintf('Warning test for feature mean computation is not passed\n');
 else
     fprintf('Pass Mean\n');
 end
  var_test = find(abs(double(fea_var')-[objRGBVar, objLabVar, objHSVVar]) > 1e-5);
 if(~isempty(var_test))
     fprintf('Warning test for feature var computation is not passed\n');
 else
     fprintf('Pass Var\n');
 end
  
  


% distance of RGB histgram with doundary 
bTRGBHDist = x2dist(objRGBHist, bgFea.bT_RGBHist);
bBRGBHDist = x2dist(objRGBHist, bgFea.bB_RGBHist);
bLRGBHDist = x2dist(objRGBHist, bgFea.bL_RGBHist);
bRRGBHDist = x2dist(objRGBHist, bgFea.bR_RGBHist);
% distance of Lab histgram with doundary 
bTLabHDist = x2dist(objLabHist, bgFea.bT_LabHist);
bBLabHDist = x2dist(objLabHist, bgFea.bB_LabHist);
bLLabHDist = x2dist(objLabHist, bgFea.bL_LabHist);
bRLabHDist = x2dist(objLabHist, bgFea.bR_LabHist);
% distance of HSV histgram with doundary 
bTHSVHDist = x2dist(objHSVHist, bgFea.bT_HSVHist);
bBHSVHDist = x2dist(objHSVHist, bgFea.bB_HSVHist);
bLHSVHDist = x2dist(objHSVHist, bgFea.bL_HSVHist);
bRHSVHDist = x2dist(objHSVHist, bgFea.bR_HSVHist);
% distance of mean RGB with doundary 
bTRGBMDist = abs(bsxfun(@minus, objRGBMean, bgFea.bT_RGB));
bBRGBMDist = abs(bsxfun(@minus, objRGBMean, bgFea.bB_RGB));
bLRGBMDist = abs(bsxfun(@minus, objRGBMean, bgFea.bL_RGB));
bRRGBMDist = abs(bsxfun(@minus, objRGBMean, bgFea.bR_RGB));
% distance of mean Lab with doundary 
bTLabMDist = abs(bsxfun(@minus, objLabMean, bgFea.bT_Lab));
bBLabMDist = abs(bsxfun(@minus, objLabMean, bgFea.bB_Lab));
bLLabMDist = abs(bsxfun(@minus, objLabMean, bgFea.bL_Lab));
bRLabMDist = abs(bsxfun(@minus, objLabMean, bgFea.bR_Lab));
% distance of mean HSV with doundary 
bTHSVMDist = abs(bsxfun(@minus, objHSVMean, bgFea.bT_HSV));
bBHSVMDist = abs(bsxfun(@minus, objHSVMean, bgFea.bB_HSV));
bLHSVMDist = abs(bsxfun(@minus, objHSVMean, bgFea.bL_HSV));
bRHSVMDist = abs(bsxfun(@minus, objHSVMean, bgFea.bR_HSV));

% distance of histograms with global image
gRGBHDist = x2dist(objRGBHist, gQ_RGBHist);
gLabHDist = x2dist(objLabHist, gQ_LabHist);
gHSVHDist = x2dist(objHSVHist, gQ_HSVHist);


ctrFeature = [bTRGBHDist, bBRGBHDist, bLRGBHDist, bRRGBHDist,...
    bTLabHDist, bBLabHDist, bLLabHDist, bRLabHDist,...
    bTHSVHDist, bBHSVHDist, bLHSVHDist, bRHSVHDist,...
    bTRGBMDist, bBRGBMDist, bLRGBMDist, bRRGBMDist,...
    bTLabMDist, bBLabMDist, bLLabMDist, bRLabMDist,...
    bTHSVMDist, bBHSVMDist, bLHSVMDist, bRHSVMDist,...
    gRGBHDist, gLabHDist, gHSVHDist,...
    objRGBVar, objLabVar, objHSVVar];

