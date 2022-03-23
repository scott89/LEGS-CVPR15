function test
init;
cleanupObj = onCleanup(@cleanupFun);
test_start = 1;
test_end = length(files);
test_num = test_end - test_start + 1;
if test_num < 1
    return
end

for imInd = test_start:test_end
    fprintf('Processing Img:  %d/%d\n', imInd, test_num);
    tic;
    name = files(imInd).name(1:end-length(im_suffix));
    im = imread([im_path name im_suffix]);
    if size(im, 3) ~= 3
        im(:,:,2) = im(:,:,1);
        im(:,:,3) = im(:,:,1);
    end
    im_ori = im;
    cur_im_sz = size(im);

    if cur_im_sz(1)*cur_im_sz(2)>80000
        im = imresize(im, round(cur_im_sz(1:2).*80000/(cur_im_sz(1)*cur_im_sz(2))) );
    end
    imdouble = im2double(im);
    re_im_sz = size(im);
    caffe('set_input_dim', 'DNNL', [0, 1, 3, re_im_sz(1), re_im_sz(2)]);
    [refine_lsm, lsm, obj_mask] = ComputeLsm(im, p); 
     
    [feature] = matComputeFeature(imdouble, obj_mask, hist_param, lsm, refine_lsm);

    data = bsxfun(@rdivide, bsxfun(@minus, feature, feature_mean), feature_stv);
    data = {single(data)};
    caffe('set_input_dim', 'DNNG', [0, size(data{1}, 2), 1, 1, size(data{1}, 1)]);
    PRscores = caffe('forward', 'DNNG', data);
    PRscores = reshape(PRscores{1}, 2, []);
    PRscores = PRscores';
    scores = PRscores(:,1).*PRscores(:,2);
    [~, order] = sort(scores, 'descend');
    salMap_20 = sum(obj_mask(:,:,order(1:12)), 3)/12;
    salMap_20 = salMap_20/max(salMap_20(:));
    salMap_20 = imresize(salMap_20, cur_im_sz(1:2));
    figure(1); subplot(1,2,1); imshow(im_ori); subplot(1,2,2); imshow(salMap_20);
    imwrite(salMap_20, [res_path name map_suffix]);
    fprintf('Time cost: %.2f s\n', toc);
end
end

    
    
    
    
    
    
    
    
    
