function cleanupFun
fprintf('Running clean up function\n');
caffe('reset','DNNL');
caffe('reset','DNNG');