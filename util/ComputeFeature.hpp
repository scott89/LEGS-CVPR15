#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define COLOR_CHANNEL 3
#define NUM_GEO_FEA 9 // central x, y, bounding box width, height, aspect ratio, major_length, minor_length, area, area/box, 
///////////////////////////////////////////////
//#define BLOCK_SIZE 32
///////////////////////////////////////////////
#define CUDA_CHECK(err) \
  err = cudaGetLastError(); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "Failed. to launch vectorAdd kernel (error code %s)!", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }

#define AllocateDevice(var) \
{ \
  err = cudaMalloc((void **)&device_##var, size_##var); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "Failed to allocate device_%s (error code %s)!", #var, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

#define SetDevice(var, val) \
{ \
  err = cudaMemset((void*)device_##var, val, size_##var); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "Failed to reset device_%s (error code %s)!",#var,  cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}
#define CopyHosttoDevice(var) \
{ \
  err = cudaMemcpy(device_##var, host_##var, size_##var, cudaMemcpyHostToDevice); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "Failed to copy %s from host to device (error code %s)!", #var, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} 
#define CopyDevicetoHost(var) \
{ \
  err = cudaMemcpy(host_##var, device_##var, size_##var, cudaMemcpyDeviceToHost); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "Failed to copy %s from device to host (error code %s)!", #var, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} 

#define FreeDevice(var) \
{ \
  err = cudaFree(device_##var); \
  if (err != cudaSuccess) \
  { \
    fprintf(stderr, "Failed to free device_%s  (error code %s)!", #var, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

#define GRID_DIM_X 2147483647
#define GRID_DIM_Y 65535
#define GRID_DIM_Z 65535
//void compute_hist(const int *host_qualified_fea, const float *host_fea, const bool *host_local_map, const bool *host_refined_local_map, const bool *host_reg, const int width, const int height, const int num_reg, const int num_bin, const int num_fea, const int num_local_fea, unsigned int *host_hist, float* host_fea_mean, float* host_fea_var, float* host_local_fea, float* host_refined_local_fea );

void ComputeFeature(const int *host_qualified_fea, const float *host_fea, const bool *host_local_map, const bool *host_refined_local_map, const bool *host_reg, const int width, const int height, const int num_reg, const int num_bin, const int num_fea, const int num_local_fea, const float local_map_sum, const float refined_local_map_sum, float *host_hist, float *host_fea_mean, float *host_fea_var,  float* host_local_fea, float* host_refined_local_fea, float* host_geo_fea );
