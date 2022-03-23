
%% path setting
addpath('./util/');
addpath('./DNNs');
addpath('./gop_1.3/matlab/');


im_path = './images/';
res_path = './maps/';


if ~isdir(res_path)
    mkdir(res_path);
end


im_suffix = '.jpg';
map_suffix = '.png';

files = dir([im_path '*' im_suffix]);
%% init caffe
call_caffe;

%% init gop
init_gop;
gop_mex( 'setDetector', 'MultiScaleStructuredForest("gop_1.3/data/sf.dat")' );
% p = Proposal('max_iou', 0.9,...
%     'unary', 130, 5, 'seedUnary()', 'backgroundUnary({0,15})',...
%     'unary', 0, 5, 'zeroUnary()', 'backgroundUnary({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})' ...
%     );
p = Proposal('max_iou', 0.9,...
             'seed', 'gop_1.3/data/seed_final.dat',...
             'unary', 140, 4, 'binaryLearnedUnary("gop_1.3/data/masks_final_0_fg.dat")', 'binaryLearnedUnary("gop_1.3/data/masks_final_0_bg.dat"',...
             'unary', 140, 4, 'binaryLearnedUnary("gop_1.3/data/masks_final_1_fg.dat")', 'binaryLearnedUnary("gop_1.3/data/masks_final_1_bg.dat"',...
             'unary', 140, 4, 'binaryLearnedUnary("gop_1.3/data/masks_final_2_fg.dat")', 'binaryLearnedUnary("gop_1.3/data/masks_final_2_bg.dat"',...
             'unary', 0, 5, 'zeroUnary()', 'backgroundUnary({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})' ...
             );
%% init feature extraction module

hist_param.RGB_bins = [16,16,16];
hist_param.Lab_bins = [16,16,16];
hist_param.HSV_bins = [16,16,16];
hist_param.nRGBHist = prod(hist_param.RGB_bins);
hist_param.nLabHist = prod(hist_param.Lab_bins);
hist_param.nHSVHist = prod(hist_param.HSV_bins);
load('DNNs/feature_static.mat');
feature_stv = feature_var.^0.5;
if ~exist('util/mexComputeFeature.mexa64')
    cd util
    system('make');
    cd ../
end
