
addpath('./caffe/matlab/caffe/')

DNNL_model_def_file = './DNNs/test_DNNL.prototxt';
DNNL_model_file = './DNNs/DNNL.caffemodel';
matcaffe_init('DNNL', 1, DNNL_model_def_file, DNNL_model_file);

DNNG_model_def_file = './DNNs/test_DNNG.prototxt';
DNNG_model_file = './DNNs/DNNG.caffemodel';

matcaffe_init('DNNG', 1, DNNG_model_def_file, DNNG_model_file);