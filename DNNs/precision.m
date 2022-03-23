close all
clear
clc
call_caffe
% matcaffe_demo
% dataSet = 'DUT-OMRON';
% dataSet = 'MSRA-5000';
dataSet = 'MSRA10K';
% dataSet = 'ECCSD';
% dataSet = 'me';
% dataSet = 'THUS';
% imPath = ['/media/iiau/linux_file/AlexnetFloor3MSRA5000/data/' dataSet '/img/'];
imPath = ['../data/' dataSet '/img/'];

resPath = ['./res/' dataSet '/'];
if ~isdir(resPath)
    mkdir(resPath);
end


imSet = dir([imPath '*.jpg']);

% patchMean = h5read('mean_file_net3.h5', '/meanfile');
% patchMean = single(patchMean);

step = [2,2];
rfSz = 51;
blobCount = 512;
for imInd =8000:5:length(imSet)
    fprintf('Image: %d\n', imInd);
    imName = imSet(imInd).name(1:end-4);
    im = imread([imPath imSet(imInd).name]);
%          im = imresize(im, 51/82);

    %% Todo: remove image frames
    [noFrameImg, frameRecord] = removeframe(im, 'sobel');
    imSz = size(noFrameImg);

    %% Pass the image forword the DNN
    im = imresize(double(im), [200,300]);
    figure(1)
    subplot(1,2,1)
    imshow(mat2gray(im));
    im = impreprocess(double(im));
    im = permute(im, [2,1,3]);
    data = {single(im)};
    scores = caffe('forward', data);
    a = scores{1}(:,:,1);
    b = scores{1}(:,:,2);
    x = exp(b)./(exp(a)+exp(b));
    figure(1); subplot(1,2,2);imshow(mat2gray(permute(x,[2,1,3])))
    
%         count = 0;
%         data = nan(rfSz, rfSz, 3 ,blobCount);
%         data = permute(data, [2,1,3, 4]);
%         pred_label = [];
%         noFrameImg = impreprocess(double(noFrameImg));
%         for r = 1:step(1):imSz(1)-rfSz+1
%             fprintf('Row: %d/%d\n', r, imSz(1)-rfSz+1);
%             for c = 1:step(2):imSz(2)-rfSz+1
%                 count = count+1;
%                 patch = single(noFrameImg(r:r+rfSz-1,c:c+rfSz-1,:));
%                 %RGB -> GBR
%                 %             patch = patch(:, :, [3 2 1]);
%                 patch = permute(patch, [2, 1, 3]);
%                 %             patch = (patch-patchMean);
%                 data(:,:,:, count) = patch;
%                 if (count == blobCount)
%                     scores = caffe('forward', {single(data)});
%                     scores = reshape(scores{1}, 2, []);
%                     %                 pred = double(scores(2,:)>scores(1,:));
%                     pred = double(scores(2,:));
%                     pred_label = [pred_label, pred];
%                     count = 0;
%                 end
%             end
%         end
    
    
%     if (count~=0)
%         scores = caffe('forward', {single(data)});
%         scores = reshape(scores{1}, 2, []);
% %         pred = double(scores(2,:)>scores(1,:));
%         pred = double(scores(2,:));
%         pred = pred(1,1:count);
%         pred_label = [pred_label, pred];
%     end
%     pxlMap = reshape(pred_label, floor((imSz(2)-rfSz)/step(2)) + 1, []);
%     pxlMap =pxlMap';
%     
%     figure(2)
%     subplot(1,2,1)
%     imshow(pxlMap);
%     
%     %% convert from pixel level to superpixel level
%     pxlMapUp = zeros(imSz(1) , imSz(2));
%     
%     pxlMapUp(ceil(rfSz/2):step(1):end-floor(rfSz/2), ceil(rfSz/2):step(2):end-floor(rfSz/2)) = pxlMap;
%     
%     
%    
%     
%     [hReal, wReal, ~] = size(im);
%     pxlMapAddFrm = zeros(hReal, wReal);
%     pxlMapAddFrm(frameRecord(3):frameRecord(4), frameRecord(5):frameRecord(6)) = pxlMapUp;
%     imwrite(pxlMapAddFrm, [resPath imName '_pxl.jpg']);
%     imwrite(im, [resPath imName '_im.png'])
end

