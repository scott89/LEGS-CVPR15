function Feature = matComputeFeature(im, objMask, histParam, lsm, rlsm)
% Compute the color feature for each objce proposal
% INPUT:
%          - im: the color image (dobule [0,1])
%          - objMask: objcet proposal mask
%  OUTPUT:
%         - colFeature: color feature of each objce proposal, 
%         propNum x  feadim, including rgb and lab color histgram
% tic;
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
% imdata = struct('imR', imR, 'imG', imG, 'imB', imB, 'imL', imL, 'ima', ima, 'imb', imb, 'imH', imH, 'imS', imS, 'imV', imV,...
%     'Q_RGB', Q_RGB, 'Q_Lab', Q_Lab, 'Q_HSV', Q_HSV, 'h', h, 'w', w);
% bgFea = getBgFea(imdata, histParam);

bgMask = getBkgGlobalMask(w, h);
objMask = cat(3, bgMask, objMask>0);

%% Global histogram
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
%% global feature
global_hist = fea_hist(:, 1);
global_rgb_hist = global_hist(1:4096);
global_lab_hist = global_hist(4097:8192);
global_hsv_hist = global_hist(8193:end);

% global_mean = fea_mean(:, 1);
% global_var = fea_var(:, 1);
%% background feature
% histogram
background_top_hist = fea_hist(:, 2);
background_top_rgb_hist = background_top_hist(1:4096);
background_top_lab_hist = background_top_hist(4097:8192);
background_top_hsv_hist = background_top_hist(8193:end);

background_bottom_hist = fea_hist(:, 3);
background_bottom_rgb_hist = background_bottom_hist(1:4096);
background_bottom_lab_hist = background_bottom_hist(4097:8192);
background_bottom_hsv_hist = background_bottom_hist(8193:end);

background_left_hist = fea_hist(:, 4);
background_left_rgb_hist = background_left_hist(1:4096);
background_left_lab_hist = background_left_hist(4097:8192);
background_left_hsv_hist = background_left_hist(8193:end);

background_right_hist = fea_hist(:, 5);
background_right_rgb_hist = background_right_hist(1:4096);
background_right_lab_hist = background_right_hist(4097:8192);
background_right_hsv_hist = background_right_hist(8193:end);

% mean
background_top_mean = fea_mean(:, 2);
background_top_rgb_mean = background_top_mean(1:3);
background_top_lab_mean = background_top_mean(4:6);
background_top_hsv_mean = background_top_mean(7:9);

background_bottom_mean = fea_mean(:, 3);
background_bottom_rgb_mean = background_bottom_mean(1:3);
background_bottom_lab_mean = background_bottom_mean(4:6);
background_bottom_hsv_mean = background_bottom_mean(7:9);

background_left_mean = fea_mean(:, 4);
background_left_rgb_mean = background_left_mean(1:3);
background_left_lab_mean = background_left_mean(4:6);
background_left_hsv_mean = background_left_mean(7:9);

background_right_mean = fea_mean(:, 5);
background_right_rgb_mean = background_right_mean(1:3);
background_right_lab_mean = background_right_mean(4:6);
background_right_hsv_mean = background_right_mean(7:9);
% var
% background_top_var = fea_var(:, 2);
% background_bottom_var = fea_var(:, 3);
% background_left_var = fea_var(:, 4);
% background_right_var = fea_var(:, 5);
%% object feature
object_hist = fea_hist(:, 6:end);
object_rgb_hist = object_hist(1:4096, :);
object_lab_hist = object_hist(4097:8192, :);
object_hsv_hist = object_hist(8193:end, :);

object_mean = fea_mean(:, 6:end);
object_rgb_mean = object_mean(1:3, :);
object_lab_mean = object_mean(4:6, :);
object_hsv_mean = object_mean(7:9, :);

object_var = fea_var(:, 6:end);
object_local_saliency = lf(:, 6:end);
object_refiend_local_saliency = rlf(:, 6:end);
object_geo_fea = geo_fea(:, 6:end);
%% distance of histgram 
top_rgb_hdist = x2dist(object_rgb_hist, background_top_rgb_hist);
top_lab_hdist = x2dist(object_lab_hist, background_top_lab_hist);
top_hsv_hdist = x2dist(object_hsv_hist, background_top_hsv_hist);

bottom_rgb_hdist = x2dist(object_rgb_hist, background_bottom_rgb_hist);
bottom_hsv_hdist = x2dist(object_lab_hist, background_bottom_lab_hist);
bottom_lab_hdist = x2dist(object_hsv_hist, background_bottom_hsv_hist);

left_rgb_hdist = x2dist(object_rgb_hist, background_left_rgb_hist);
left_lab_hdist = x2dist(object_lab_hist, background_left_lab_hist);
left_hsv_hdist = x2dist(object_hsv_hist, background_left_hsv_hist);

right_rgb_hdist = x2dist(object_rgb_hist, background_right_rgb_hist);
right_lab_hdist = x2dist(object_lab_hist, background_right_lab_hist);
right_hsv_hdist = x2dist(object_hsv_hist, background_right_hsv_hist);

global_rgb_hdist = x2dist(object_rgb_hist, global_rgb_hist);
global_lab_hdist = x2dist(object_lab_hist, global_lab_hist);
global_hsv_hdist = x2dist(object_hsv_hist, global_hsv_hist);
%% distance of mean 
top_rgb_mdist = abs(bsxfun(@minus, object_rgb_mean, background_top_rgb_mean));
top_lab_mdist = abs(bsxfun(@minus, object_lab_mean, background_top_lab_mean));
top_hsv_mdist = abs(bsxfun(@minus, object_hsv_mean, background_top_hsv_mean));

bottom_rgb_mdist = abs(bsxfun(@minus, object_rgb_mean, background_bottom_rgb_mean));
bottom_lab_mdist = abs(bsxfun(@minus, object_lab_mean, background_bottom_lab_mean));
bottom_hsv_mdist = abs(bsxfun(@minus, object_hsv_mean, background_bottom_hsv_mean));

left_rgb_mdist = abs(bsxfun(@minus, object_rgb_mean, background_left_rgb_mean));
left_lab_mdist = abs(bsxfun(@minus, object_lab_mean, background_left_lab_mean));
left_hsv_mdist = abs(bsxfun(@minus, object_hsv_mean, background_left_hsv_mean));

right_rgb_mdist = abs(bsxfun(@minus, object_rgb_mean, background_right_rgb_mean));
right_lab_mdist = abs(bsxfun(@minus, object_lab_mean, background_right_lab_mean));
right_hsv_mdist = abs(bsxfun(@minus, object_hsv_mean, background_right_hsv_mean));

% 
Feature = [top_rgb_hdist; bottom_rgb_hdist; left_rgb_hdist ; right_rgb_hdist ; global_rgb_hdist;
    top_lab_hdist; bottom_lab_hdist; left_lab_hdist ; right_lab_hdist ; global_lab_hdist;
    top_hsv_hdist; bottom_hsv_hdist; left_hsv_hdist ; right_hsv_hdist ; global_hsv_hdist;
    top_rgb_mdist; bottom_rgb_mdist ; left_rgb_mdist; right_rgb_mdist;
    top_lab_mdist; bottom_lab_mdist ; left_lab_mdist; right_lab_mdist;
    top_hsv_mdist; bottom_hsv_mdist ; left_hsv_mdist; right_hsv_mdist;
    object_var;
    object_local_saliency; object_refiend_local_saliency;
    object_geo_fea];
% ctrFeature = [bTRGBHDist, bBRGBHDist, bLRGBHDist, bRRGBHDist,...
%     bTLabHDist, bBLabHDist, bLLabHDist, bRLabHDist,...
%     bTHSVHDist, bBHSVHDist, bLHSVHDist, bRHSVHDist,...
%     bTRGBMDist, bBRGBMDist, bLRGBMDist, bRRGBMDist,...
%     bTLabMDist, bBLabMDist, bLLabMDist, bRLabMDist,...
%     bTHSVMDist, bBHSVMDist, bLHSVMDist, bRHSVMDist,...
%     gRGBHDist, gLabHDist, gHSVHDist,...
%     objRGBVar, objLabVar, objHSVVar];

