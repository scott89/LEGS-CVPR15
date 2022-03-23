function scores = matcaffe_demo_vgg_mean_pix(im, use_gpu, model_def_file, model_file)
% scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file)
%
% Demo of the matlab wrapper based on the networks used for the "VGG" entry
% in the ILSVRC-2014 competition and described in the tech. report 
% "Very Deep Convolutional Networks for Large-Scale Image Recognition"
% http://arxiv.org/abs/1409.1556/
%
% INPUT
%   im - color image as uint8 HxWx3
%   use_gpu - 1 to use the GPU, 0 to use the CPU
%   model_def_file - network configuration (.prototxt file)
%   model_file - network weights (.caffemodel file)
%
% OUTPUT
%   scores   1000-dimensional ILSVRC score vector
%
% EXAMPLE USAGE
%  model_def_file = 'zoo/deploy.prototxt';
%  model_file = 'zoo/model.caffemodel';
%  use_gpu = true;
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file);
% 
% NOTES
%  mean pixel subtraction is used instead of the mean image subtraction
%
% PREREQUISITES
%  You may need to do the following before you start matlab:
%   $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%   $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
%  Or the equivalent based on where things are installed on your system

% init caffe network (spews logging info)
matcaffe_init(use_gpu, model_def_file, model_file);

% mean BGR pixel
mean_pix = [103.939, 116.779, 123.68];

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image(im, mean_pix)};
toc;

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;

% scores = scores{1};
% size(scores)
% scores = squeeze(scores);
% scores = mean(scores,2);
% scores = 0;
% [~,maxlabel] = max(scores);

% ------------------------------------------------------------------------
function im = prepare_image(im, mean_pix)
% ------------------------------------------------------------------------


% resize to fixed input size
im = single(im);

% RGB -> BGR
im = im(:, :, [3 2 1]);


for c = 1:3
    im(:, :, c, :) = im(:, :, c, :) - mean_pix(c);
end