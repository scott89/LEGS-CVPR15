#include "mex.h"
#include "matrix.h"
//#include <cuda_runtime.height>
#include "ComputeFeature.hpp"
#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

//mexComputeHist(qualified_fea, fea, local_map, refined_local_map, reg, num_bin, local_map_sum, refined_local_map_sum)
//                      0        1       2                3         4      5            6                    7
void mexFunction(MEX_ARGS) {
  //mexLock();  // Avoid clearing the mex file.
  if (nrhs != 8) {
    mexErrMsgTxt("Invalid input number: mexComputeHist(qualified_fea, fea, local_map, local_map_refined,  reg, num_bin)");
    return;
  }
  if (!mxIsInt32(prhs[0])) {
    mexErrMsgTxt("qualified_fea requires int32 data.");
    return;
  }
  if (!mxIsSingle(prhs[1])) {
    mexErrMsgTxt("fea requires single data.");
    return;
  }
  if (!mxIsLogical(prhs[2])) {
    mexErrMsgTxt("local_map requires logical data.");
    return;
  }
  if (!mxIsLogical(prhs[3])) {
    mexErrMsgTxt("refined_local_map requires logical data.");
    return;
  }

  if (!mxIsLogical(prhs[4])) {
    mexErrMsgTxt("Region mask requires logical data.");
    return;
  }


  int width, height, num_reg, num_bin, num_qualified_fea, num_fea;
  // qualified_fea 
  const int *qualified_fea = reinterpret_cast<const int*>(mxGetPr(prhs[0]));
  const int *qualified_fea_dim = static_cast<const int*>(mxGetDimensions(prhs[0]));
  unsigned int qualified_fea_dim_num = static_cast<unsigned int>(mxGetNumberOfDimensions(prhs[0]));
  num_qualified_fea = qualified_fea_dim_num==2?1:qualified_fea_dim[2];
  // fea
  const float *fea = reinterpret_cast<const float*>(mxGetPr(prhs[1]));
  const int *fea_dim = static_cast<const int*>(mxGetDimensions(prhs[1]));
  unsigned int fea_dim_num = static_cast<unsigned int>(mxGetNumberOfDimensions(prhs[1]));
  if (fea_dim[2] != COLOR_CHANNEL) {
    mexErrMsgTxt("The channel number of color features should be 3.");
    return;
  }
  num_fea = fea_dim_num==3?1:fea_dim[3]; 
  if (num_fea != num_qualified_fea) {
    mexErrMsgTxt("Different number of fea and qualified_fea.");
    return;
  }
  // local_map
  const bool *local_map = static_cast<const bool*>(mxGetLogicals(prhs[2]));
  // refined_local_map
  const bool *refined_local_map = static_cast<const bool*>(mxGetLogicals(prhs[3]));
  // reg
  const bool *reg = static_cast<const bool*>(mxGetLogicals(prhs[4]));
  const int *reg_dim = static_cast<const int*>(mxGetDimensions(prhs[4]));
  unsigned int reg_dim_num = static_cast<unsigned int>(mxGetNumberOfDimensions(prhs[4]));
  
  height = reg_dim[0]; width = reg_dim[1]; num_reg = reg_dim_num==2?1:reg_dim[2];  
  num_bin = static_cast<int>(mxGetScalar(prhs[5]));
  float local_map_sum = static_cast<float>(mxGetScalar(prhs[6]));
  float refined_local_map_sum = static_cast<float>(mxGetScalar(prhs[7]));
  /* printf("width = %d, height = %d, num_reg = %d, num_bin = %d", width, height, num_reg, num_bin);
  for(int i = 0; i < 4; i++) {
    //printf("\num_reg%d", reg[i]);
    printf("\num_reg%d", qualified_fea[i]);
  }
  if(reg[static_cast<int>(qualified_fea[0])]) 
  printf("True");*/

 // plhs[0] = mxCreateNumericMatrix(num_bin, num_reg, mxUINT32_CLASS, mxREAL);
  // hist
  mwSize hist_dim_num = 2;
 // mwSize hist_dim[hist_dim_num] = {num_bin, num_reg, num_fea};
  mwSize hist_dim[hist_dim_num] = {num_bin * num_fea, num_reg};
 // plhs[0] = mxCreateNumericArray(hist_dim_num, hist_dim, mxUINT32_CLASS, mxREAL);
  plhs[0] = mxCreateNumericMatrix(num_bin * num_fea, num_reg, mxSINGLE_CLASS, mxREAL);
//  printf("matlab hist size: %d * %d * %d\n", num_bin,  num_fea, num_reg);
  float *hist = reinterpret_cast<float*>(mxGetData(plhs[0]));
  /// fea_mean
  plhs[1] = mxCreateNumericMatrix(fea_dim[2] * num_fea, num_reg, mxSINGLE_CLASS, mxREAL); 
  float *fea_mean = reinterpret_cast<float*>(mxGetData(plhs[1]));
  // fea_var
  plhs[2] = mxCreateNumericMatrix(fea_dim[2] * num_fea, num_reg, mxSINGLE_CLASS, mxREAL); 
  float *fea_var = reinterpret_cast<float*>(mxGetData(plhs[2]));

  // local fea
  int num_local_fea = 4;
  plhs[3] = mxCreateNumericMatrix(num_local_fea, num_reg, mxSINGLE_CLASS, mxREAL); // [Pre; Rc; PR; overlap]
  float *local_fea = reinterpret_cast<float*>(mxGetData(plhs[3]));
  // local_refined_fea
  plhs[4] = mxCreateNumericMatrix(num_local_fea, num_reg, mxSINGLE_CLASS, mxREAL);
  float *refined_local_fea = reinterpret_cast<float*>(mxGetData(plhs[4]));
  // geo_fea
  plhs[5] = mxCreateNumericMatrix(NUM_GEO_FEA, num_reg, mxSINGLE_CLASS, mxREAL);
  float *geo_fea = reinterpret_cast<float*>(mxGetData(plhs[5])); 
  ComputeFeature(qualified_fea, fea, local_map, refined_local_map, reg, width, height, num_reg, num_bin, num_fea, num_local_fea, local_map_sum, refined_local_map_sum, hist, fea_mean, fea_var, local_fea, refined_local_fea, geo_fea);
  return;
}
