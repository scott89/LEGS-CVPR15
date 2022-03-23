function bgFea = getBgFea(imdata, histParam)
%% histogram param

nRGBHist = histParam.nRGBHist;
nLabHist = histParam.nLabHist;
nHSVHist = histParam.nHSVHist;
h = imdata.h; w = imdata.w;
%% bg mask
bgWidth = ceil(max(imdata.h, imdata.w)*15/400);

bTMask = false(h, w);
bBMask = false(h, w);
bLMask = false(h, w);
bRMask = false(h, w);

bTMask(1:bgWidth, :) = true;
bBMask(end-bgWidth+1:end, :) = true;
bLMask(:, 1:bgWidth) = true;
bRMask(:, end-bgWidth+1:end) = true;
%% RGB hist
bT_RGBHist = hist(imdata.Q_RGB(bTMask), 0:nRGBHist-1);
bB_RGBHist = hist(imdata.Q_RGB(bBMask), 0:nRGBHist-1);
bL_RGBHist= hist(imdata.Q_RGB(bLMask), 0:nRGBHist-1);
bR_RGBHist= hist(imdata.Q_RGB(bRMask), 0:nRGBHist-1);

 bgFea.bT_RGBHist = bT_RGBHist/sum(bT_RGBHist);
bgFea.bB_RGBHist = bB_RGBHist/sum(bB_RGBHist);
bgFea.bL_RGBHist = bL_RGBHist/sum(bL_RGBHist);
bgFea.bR_RGBHist = bR_RGBHist/sum(bR_RGBHist);
%% Lab Hist
bT_LabHist = hist(imdata.Q_Lab(bTMask), 0:nLabHist-1);
bB_LabHist = hist(imdata.Q_Lab(bBMask), 0:nLabHist-1);
bL_LabHist = hist(imdata.Q_Lab(bLMask), 0:nLabHist-1);
bR_LabHist = hist(imdata.Q_Lab(bRMask), 0:nLabHist-1);

bgFea.bT_LabHist = bT_LabHist/sum(bT_LabHist);
bgFea.bB_LabHist = bB_LabHist/sum(bB_LabHist);
bgFea.bL_LabHist = bL_LabHist/sum(bL_LabHist);
bgFea.bR_LabHist= bR_LabHist/sum(bR_LabHist);
%% HSV  hist
bT_HSVHist = hist(imdata.Q_HSV(bTMask), 0:nHSVHist-1);
bB_HSVHist = hist(imdata.Q_HSV(bBMask), 0:nHSVHist-1);
bL_HSVHist = hist(imdata.Q_HSV(bLMask), 0:nHSVHist-1);
bR_HSVHist = hist(imdata.Q_HSV(bRMask), 0:nHSVHist-1);

bgFea.bT_HSVHist = bT_HSVHist/sum(bT_HSVHist);
bgFea.bB_HSVHist = bB_HSVHist/sum(bB_HSVHist);
bgFea.bL_HSVHist = bL_HSVHist/sum(bL_HSVHist);
bgFea.bR_HSVHist= bR_HSVHist/sum(bR_HSVHist);

%% mean R
bgFea.bT_RGB(1,1) = mean(imdata.imR(bTMask));
bgFea.bB_RGB(1,1) = mean(imdata.imR(bBMask));
bgFea.bL_RGB(1,1) = mean(imdata.imR(bLMask));
bgFea.bR_RGB(1,1) = mean(imdata.imR(bRMask));
%% G
bgFea.bT_RGB(1,2) = mean(imdata.imG(bTMask));
bgFea.bB_RGB(1,2) = mean(imdata.imG(bBMask));
bgFea.bL_RGB(1,2)= mean(imdata.imG(bLMask));
bgFea.bR_RGB(1,2) = mean(imdata.imG(bRMask));
%% B
bgFea.bT_RGB(1,3) = mean(imdata.imB(bTMask));
bgFea.bB_RGB(1,3) = mean(imdata.imB(bBMask));
bgFea.bL_RGB(1,3) = mean(imdata.imB(bLMask));
bgFea.bR_RGB(1,3) = mean(imdata.imB(bRMask));
%% L
bgFea.bT_Lab(1,1) = mean(imdata.imL(bTMask));
bgFea.bB_Lab(1,1) = mean(imdata.imL(bBMask));
bgFea.bL_Lab(1,1) = mean(imdata.imL(bLMask));
bgFea.bR_Lab(1,1) = mean(imdata.imL(bRMask));
%% a
bgFea.bT_Lab(1,2) = mean(imdata.ima(bTMask));
bgFea.bB_Lab(1,2) = mean(imdata.ima(bBMask));
bgFea.bL_Lab(1,2) = mean(imdata.ima(bLMask));
bgFea.bR_Lab(1,2) = mean(imdata.ima(bRMask));
%% b
bgFea.bT_Lab(1,3) = mean(imdata.imb(bTMask));
bgFea.bB_Lab(1,3) = mean(imdata.imb(bBMask));
bgFea.bL_Lab(1,3) = mean(imdata.imb(bLMask));
bgFea.bR_Lab(1,3) = mean(imdata.imb(bRMask));
%% H
bgFea.bT_HSV(1,1) = mean(imdata.imH(bTMask));
bgFea.bB_HSV(1,1) = mean(imdata.imH(bBMask));
bgFea.bL_HSV(1,1) = mean(imdata.imH(bLMask));
bgFea.bR_HSV(1,1) = mean(imdata.imH(bRMask));
%% S
bgFea.bT_HSV(1,2) = mean(imdata.imS(bTMask));
bgFea.bB_HSV(1,2) = mean(imdata.imS(bBMask));
bgFea.bL_HSV(1,2) = mean(imdata.imS(bLMask));
bgFea.bR_HSV(1,2) = mean(imdata.imS(bRMask));
%% V
bgFea.bT_HSV(1,3) = mean(imdata.imV(bTMask));
bgFea.bB_HSV(1,3) = mean(imdata.imV(bBMask));
bgFea.bL_HSV(1,3) = mean(imdata.imV(bLMask));
bgFea.bR_HSV(1,3) = mean(imdata.imV(bRMask));

