function ctrFeature = getCtrFea(im, objMask, histParam)
% Compute the color feature for each objce proposal
% INPUT:
%          - im: the color image (dobule [0,1])
%          - objMask: objcet proposal mask
%  OUTPUT:
%         - colFeature: color feature of each objce proposal, 
%         propNum x  feadim, including rgb and lab color histgram
RGB_bins = histParam.RGB_bins;
Lab_bins = histParam.Lab_bins;
H_bin = histParam.H_bin;
S_bin = histParam.S_bin;
nRGBHist = histParam.nRGBHist;
nLabHist = histParam.nLabHist;
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
    QB ;

% objQ_RGB = bsxfun(@times, objMask, Q_RGB);
%% Lab
QL = min(floor(imLab(:,:,1)*Lab_bins(1))+1, Lab_bins(1));
Qa = min(floor(imLab(:,:,2)*Lab_bins(2))+1, Lab_bins(2));
Qb = min(floor(imLab(:,:,3)*Lab_bins(3))+1, Lab_bins(3));
Q_Lab = (QL-1)*Lab_bins(2)*Lab_bins(3)+...
    (Qa-1)*Lab_bins(3)+...
    Qb;

% objQ_Lab = bsxfun(@times, objMask, Q_Lab);
%% H
Q_H = min(floor(imHSV(:,:,1)*H_bin)+1, H_bin);
% objQ_H = bsxfun(@times, objMask, QH);
%% S
Q_S = min(floor(imHSV(:,:,2)*S_bin)+1, S_bin);
% objQ_S = bsxfun(@times, objMask, QS);
%% Background histogram
bH = ceil(h*15/400);
bW = ceil(w*15/400);
%% %%
bMask = true(h, w);
bMask(bH+1:end-bH, bW+1:end-bW) = false;
bQ_RGB = hist(Q_RGB(bMask), 1:nRGBHist);
bQ_RGB = bQ_RGB/sum(bQ_RGB);
bQ_Lab = hist(Q_Lab(bMask), 1:nLabHist );
bQ_Lab = bQ_Lab/sum(bQ_Lab);
bQ_H = hist(Q_H(bMask), 1:H_bin);
bQ_H = bQ_H/sum(bQ_H);
bQ_S = hist(Q_S(bMask), 1:S_bin);
bQ_S = bQ_S/sum(bQ_S);
%% Global histogram
gQ_RGB = hist(Q_RGB(:), 1:nRGBHist);
gQ_RGB = gQ_RGB/sum(gQ_RGB);
gQ_Lab = hist(Q_Lab(:), 1:nLabHist);
gQ_Lab = gQ_Lab/sum(gQ_Lab);
gQ_H = hist(Q_H(:), 1:H_bin);
gQ_H = gQ_H/sum(gQ_H);
gQ_S = hist(Q_S(:), 1:S_bin);
gQ_S = gQ_S/sum(gQ_S);
%%
objRGBHist = zeros(propNum, nRGBHist);
objLabHist = zeros(propNum, nLabHist);
objHHist = zeros(propNum, H_bin);
objSHist = zeros(propNum, S_bin);
objRGBVar = zeros(propNum, 3);
objLabVar = zeros(propNum, 3);
objHSVVar = zeros(propNum, 3);
for pId = 1:propNum
    curObjMask = objMask(:,:,pId);
    objRGBHist(pId, :) = hist(Q_RGB(curObjMask>0), 1:nRGBHist);
    objLabHist(pId, :) = hist(Q_Lab(curObjMask>0), 1:nLabHist);
    objHHist(pId, :) = hist(Q_H(curObjMask>0), 1:H_bin);
    objSHist(pId, :) = hist(Q_S(curObjMask>0), 1:S_bin);
    % RGB Var
    objRGBVar(pId, 1) = var(imR(curObjMask>0));
    objRGBVar(pId, 2) = var(imG(curObjMask>0));
    objRGBVar(pId, 3) = var(imB(curObjMask>0));
    % Lab Var
    objLabVar(pId, 1) = var(imL(curObjMask>0));
    objLabVar(pId, 2) = var(ima(curObjMask>0));
    objLabVar(pId, 3) = var(imb(curObjMask>0));
    % HSV Var
    objHSVVar(pId, 1) = var(imH(curObjMask>0));
    objHSVVar(pId, 2) = var(imS(curObjMask>0));
    objHSVVar(pId, 3) = var(imV(curObjMask>0));
end
objRGBHist = objRGBHist./repmat(sum(objRGBHist, 2), 1, nRGBHist);
objLabHist = objLabHist./repmat(sum(objLabHist, 2), 1, nLabHist);
objHHist = objHHist./repmat(sum(objHHist, 2), 1, H_bin);
objSHist = objSHist./repmat(sum(objSHist, 2), 1, S_bin);

bRGBEDist = (sum((bsxfun(@minus, objRGBHist, bQ_RGB)).^2, 2)).^0.5;

bRGBHDist = x2dist(objRGBHist, bQ_RGB);
bLabHDist = x2dist(objLabHist, bQ_Lab);
bHHDist = x2dist(objHHist, bQ_H);
bSHDist = x2dist(objSHist, bQ_S);

gRGBHDist = x2dist(objRGBHist, gQ_RGB);
gLabHDist = x2dist(objLabHist, gQ_Lab);
gHHDist = x2dist(objHHist, gQ_H);
gSHDist = x2dist(objSHist, gQ_S);

ctrFeature = [bRGBEDist, bRGBHDist, bLabHDist, bHHDist, bSHDist, gRGBHDist, gLabHDist, gHHDist, gSHDist, ...
    objRGBVar, objLabVar, objHSVVar];
