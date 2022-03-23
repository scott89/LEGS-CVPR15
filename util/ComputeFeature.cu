#include "ComputeFeature.hpp"
#include <stdio.h>
#include <limits.h>
//dim3 GridDim(int num_blocks) {
//  dim3 grid_dim;
//  if (num_blocks <= GRID_DIM_X) {
//    grid_dim.x = num_blocks; 
//  } else if(num_blocks <= GRID_DIM_X * GRID_DIM_Y) {
//    grid_dim.x = GRID_DIM_X;
//    grid_dim.y = num_blocks / GRID_DIM_X + 1;
//  } else if(num_blocks <= GRID_DIM_X * GRID_DIM_Y * GRID_DIM_Z) {
//    grid_dim.x = GRID_DIM_X;
//    grid_dim.y = GRID_DIM_Y;
//    grid_dim.z = num_blocks / (GRID_DIM_X * GRID_DIM_Y) + 1;
//  }
//  return grid_dim;
//}
__device__ unsigned int float_to_uint(float f)
{
  unsigned int *p = reinterpret_cast<unsigned int*>(&f);
  unsigned int input = *p;
  unsigned int mask = -int(input  >> 31) | 0x80000000;
  return input ^ mask;
}
__device__ float uint_to_float(const unsigned int f)
{
  unsigned int mask = ((f >> 31) - 1) | 0x80000000;
  unsigned int output = f ^ mask;
  float *p = reinterpret_cast<float*>(&output);
  return *p;
}
__global__ void compute_pr(float* local_fea, float* reg_area,  float local_map_sum, const int num_reg, const int num_local_fea) {
  int reg_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (reg_id < num_reg) {
    float *cur_local_fea = local_fea + reg_id * num_local_fea;
    float intersection_local = cur_local_fea[0];
    float union_area_local = cur_local_fea[1];
    float reg_area_local = reg_area[reg_id]; 

    cur_local_fea[0] = intersection_local / reg_area_local;
    cur_local_fea[1] = intersection_local / local_map_sum; 
    cur_local_fea[2] = cur_local_fea[0] * cur_local_fea[1]; 
    cur_local_fea[3] = intersection_local / union_area_local;
  }
} 


__global__ void compute_local_saliency(const bool *local_map, const bool *reg, const bool compute_reg_area, const int width, const int height, const int num_reg, const int num_local_fea, const float local_map_sum, const int blocks_per_reg, float *local_fea, float *reg_area) {
  // Step 1: compute the index of current threads and determine whether it is on a region proposal
  int reg_id = blockIdx.x / blocks_per_reg;
  int px_id = (blockIdx.x % blocks_per_reg) * blockDim.x + threadIdx.x;
  int map_offset = width * height;
  const bool *cur_reg = reg + reg_id * map_offset;
  float *cur_local_fea = local_fea + num_local_fea * reg_id;

  // Step 2: init the shared memories: intersection, union_area, reg_area 
  //__shared__ float intersection_tmp;
  //__shared__ float union_area_tmp;
  //__shared__ float reg_area_tmp;
  extern __shared__ float shared_local_saliency[];
  float reg_area_tmp;
  // Compute reg_are_tmp
  if (compute_reg_area) {
    if (px_id < map_offset && cur_reg[px_id]) {
      shared_local_saliency[threadIdx.x] = 1; 
    } else {
      shared_local_saliency[threadIdx.x] = 0;
    }
    __syncthreads();
      for (int s = (blockDim.x + 1) / 2, e = blockDim.x / 2; s > 0; s = (s + 1) / 2, e /= 2) {
	if (threadIdx.x < e || threadIdx.x == 0) {
	  atomicAdd(&shared_local_saliency[threadIdx.x], shared_local_saliency[threadIdx.x + s]); 
	}
	__syncthreads();
	if (s == 1) break;
      }
    reg_area_tmp = shared_local_saliency[0];
  }
  // Compute intersection_tmp 
  if (px_id < map_offset && cur_reg[px_id] && local_map[px_id]) {
    shared_local_saliency[threadIdx.x] = 1; 
  } else {
    shared_local_saliency[threadIdx.x] = 0;
  }
  __syncthreads();
    for (int s = (blockDim.x + 1) / 2, e = blockDim.x / 2; s > 0; s = (s + 1) / 2, e /= 2) {
      if (threadIdx.x < e || threadIdx.x == 0) {
	atomicAdd(&shared_local_saliency[threadIdx.x], shared_local_saliency[threadIdx.x + s]); 
      }
      if (s == 1) break;
      __syncthreads();
    }
  float intersection_tmp = shared_local_saliency[0];
  // Compute union_area_tmp 
  if (px_id < map_offset && (cur_reg[px_id] || local_map[px_id])) {
    shared_local_saliency[threadIdx.x] = 1; 
  } else {
    shared_local_saliency[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int s = (blockDim.x + 1) / 2, e = blockDim.x / 2; s > 0; s = (s + 1) / 2, e /= 2) {
    if (threadIdx.x < e || threadIdx.x == 0) {
      atomicAdd(&shared_local_saliency[threadIdx.x], shared_local_saliency[threadIdx.x + s]); 
    }
    __syncthreads();
    if (s ==1) break;
  }
  float union_area_tmp = shared_local_saliency[0];



  // Step 4: sum the block shared memories to global momories
  if (threadIdx.x == 0) {
    atomicAdd(cur_local_fea, intersection_tmp); // 0: intersetcion
    atomicAdd(&cur_local_fea[1], union_area_tmp); // 1: union_area_tmp
    if(compute_reg_area) { atomicAdd(&reg_area[reg_id], reg_area_tmp); } // 2: reg_area
  }
  __syncthreads();

  //if (px_id < map_offset) {
  //  if(px_id == 0) {
  //    //float intersection_local = cur_local_fea[0];
  //    //float union_area_local = cur_local_fea_space[1];
  //    //float reg_area_local = compute_reg_area ? cur_local_fea[2] : reg_area[reg_id]; 
  //    //float intersection_local, union_area_local, reg_area_local;
  //    //intersection_local = atomicExch(cur_local_fea, (float)0);
  //    //union_area_local = atomicExch(&cur_local_fea[1],(float)0);
  //    //reg_area_local = atomicExch(&cur_local_fea[2], (float)0);
  //    //__syncthreads();

  //    //cur_local_fea[0] = intersection_local / reg_area_local;
  //    //cur_local_fea[1] = intersection_local / local_map_sum; 
  //    //cur_local_fea[2] = cur_local_fea[0] * cur_local_fea[1]; 
  //    //cur_local_fea[3] = intersection_local / union_area_local;
  //    //cur_local_fea[0] = intersection_local;//intersection_local;
  //    //cur_local_fea[1] = union_area_local;
  //    //cur_local_fea[2] = intersection_local;
  //    //cur_local_fea[3] = intersection_local / union_area_local;
  //    //if(compute_reg_area) reg_area[reg_id] = reg_area_local;
  //  }
  //}
  //__syncthreads();
}

__global__ void  compute_color_hist(const int* qualified_fea, const bool* reg, const float* reg_area, const int width, const int height, const int num_reg, const int num_fea, const int num_bin,  const int blocks_per_reg, float* hist) {
  // Step 1: parameter initialziation and compute the index of current threads and determine whether it is on a region proposal
  int shared_hist_size = num_bin;
  extern __shared__ float shared_hist[];
  //  __shared__ unsigned int shared_hist[12288];
  for (int i = threadIdx.x; i < shared_hist_size; i += blockDim.x) 
    shared_hist[i] = 0;
  __syncthreads();

  int reg_id = blockIdx.x / blocks_per_reg;
  int px_id = (blockIdx.x % blocks_per_reg) * blockDim.x + threadIdx.x;
  int fea_id = blockIdx.y;
  int im_offset = width * height;
  int hist_offset = num_bin;
  const bool * cur_reg = reg + reg_id * im_offset;
  if (px_id < im_offset && cur_reg[px_id]) {
    // Step 2:  compute hist in each block 
    //for (int fea_id = 0; fea_id < num_fea; fea_id++) {
    const int* cur_fea = qualified_fea + im_offset * fea_id;
    // unsigned int * cur_shared_hist = shared_hist + hist_offset * fea_id;
    //   if (cur_reg[px_id]) { 
    atomicAdd(&shared_hist[cur_fea[px_id]], 1);
    //   }
    // }
  }
  __syncthreads();
  // Step 3: collect the final hist from each block
  for (int i = threadIdx.x; i < shared_hist_size; i += blockDim.x) {
    float *cur_hist = hist + (reg_id * num_fea + fea_id) * hist_offset + i;
    float cur_shared_hist_value = shared_hist[i] / reg_area[reg_id];
    atomicAdd(cur_hist, cur_shared_hist_value);
  }
  __syncthreads();
}

__global__ void compute_color_mean(const float* fea, const bool* reg, const float* reg_area, const int width, const int height, const int num_reg, const int num_fea, const int blocks_per_reg, float* fea_mean) {
  // Step 1: parameter initialziation and compute the index of current threads and determine whether it is on a region proposal
  int shared_mean_size = COLOR_CHANNEL * num_fea;
  extern __shared__ float shared_mean[];
  for (int i = threadIdx.x; i < shared_mean_size; i += blockDim.x) {
    shared_mean[i] = 0;
  }
  __syncthreads();

  int reg_id = blockIdx.x / blocks_per_reg;
  int px_id = (blockIdx.x % blocks_per_reg) * blockDim.x + threadIdx.x;
  int im_offset = width * height;
  const bool * cur_reg = reg + reg_id * im_offset;
  // Step 2: compute mean values within blocks
  if (px_id < im_offset && cur_reg[px_id]) {
    for (int fea_id = 0; fea_id < shared_mean_size; fea_id++) {
      const float *cur_fea = fea + fea_id * im_offset;
      float *cur_shared_mean = shared_mean + fea_id;
      atomicAdd(cur_shared_mean, cur_fea[px_id]);
    }
  }
  __syncthreads();
  // Step 3: Average within block
  for (int i = threadIdx.x; i < shared_mean_size; i += blockDim.x) {
    shared_mean[i] /= reg_area[reg_id];
  }
  __syncthreads();
  // Step 4: collect mean values from shared memories
  float *cur_fea_mean = fea_mean + reg_id * shared_mean_size;
  for (int fea_id = threadIdx.x; fea_id < shared_mean_size; fea_id += blockDim.x) {
    atomicAdd(&cur_fea_mean[fea_id], shared_mean[fea_id]);
  }
  __syncthreads();
}

__global__ void compute_color_var(const float* fea, const float* fea_mean, const bool* reg, const float* reg_area,  const int width, const int height, const int num_reg, const int num_fea, const int blocks_per_reg, float* fea_var) {
  // Step 1: parameter initialziation and compute the index of current threads and determine whether it is on a region proposal
  int shared_var_size = COLOR_CHANNEL * num_fea;
  extern __shared__ float shared_var[];
  for (int i = threadIdx.x; i < shared_var_size; i += blockDim.x) {
    shared_var[i] = 0;
  }
  __syncthreads();

  int reg_id = blockIdx.x / blocks_per_reg;
  int px_id = (blockIdx.x % blocks_per_reg) * blockDim.x + threadIdx.x;
  int im_offset = width * height;
  const bool * cur_reg = reg + reg_id * im_offset;
  // Step 2: compute mean values within blocks
  if (px_id < im_offset && cur_reg[px_id]) {
    for (int fea_id = 0; fea_id < shared_var_size; fea_id++) {
      const float *cur_fea = fea + fea_id * im_offset;
      float *cur_shared_var = shared_var + fea_id;
      const float *cur_fea_mean = fea_mean + reg_id * shared_var_size;
      float deviation = cur_fea[px_id] - cur_fea_mean[fea_id];
      atomicAdd(cur_shared_var, deviation * deviation);
    }
  }
  __syncthreads();
  // Step 3: average within block
  for (int i = threadIdx.x; i < shared_var_size; i += blockDim.x) {
    shared_var[i] /= (reg_area[reg_id] - 1);
  }
  // Step 4: collect var from shared memories to device memories
  float *cur_fea_var = fea_var + reg_id * shared_var_size;
  for (int fea_id = threadIdx.x ; fea_id < shared_var_size; fea_id += blockDim.x) {
    atomicAdd(&cur_fea_var[fea_id], shared_var[fea_id]);
  }
  __syncthreads();
}

__global__ void compute_geo_fea(const bool* reg, const float* reg_area, const int width, const int height, const int num_reg, const int blocks_per_reg, float* geo_fea) {
  // Step 1: compute pixel location and region id, and initialize shared memory 
  int reg_id = blockIdx.x / blocks_per_reg;
  int px_id = (blockIdx.x % blocks_per_reg) * blockDim.x + threadIdx.x;
  int im_offset = width * height;
  const bool * cur_reg = reg + reg_id * im_offset;
  __shared__ float center_x, center_y;
  __shared__ int max_x, max_y, min_x, min_y;
  center_x = 0; center_y = 0; max_x = 0; min_x = width; max_y = 0; min_y = height;
  // Step 2: accumulate coordinates and record extrema within each block
  if (px_id < im_offset && cur_reg[px_id]) {
    int x = px_id / height + 1;
    int y = px_id % height + 1;
    float normalized_x = x / (1.0 * reg_area[reg_id]);
    float normalized_y = y / (1.0 * reg_area[reg_id]);
    atomicAdd(&center_x, normalized_x);
    atomicAdd(&center_y, normalized_y);
    atomicMax(&max_x, x);
    atomicMax(&max_y, y);
    atomicMin(&min_x, x);
    atomicMin(&min_y, y);
  }
  __syncthreads();
  // Step 3: collect results from blocks to device memories
  int *cur_geo_fea_int = reinterpret_cast<int*>(geo_fea + reg_id * NUM_GEO_FEA);
  float *cur_geo_fea_float = geo_fea + reg_id * NUM_GEO_FEA;
  // init global values:
  if (px_id == 0)
  {
    cur_geo_fea_int[2] = INT_MAX;
    cur_geo_fea_int[3] = INT_MAX;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(&cur_geo_fea_float[0], center_x);
    // cur_geo_fea_float[0] = 100;
  }
  if (threadIdx.x == 1) {
    atomicAdd(&cur_geo_fea_float[1], center_y);
  }
  if (threadIdx.x == 2) {
    atomicMin(&cur_geo_fea_int[2], min_x);
  }
  if (threadIdx.x == 3) {
    atomicMin(&cur_geo_fea_int[3],min_y);
  }
  if (threadIdx.x == 4) {
    atomicMax(&cur_geo_fea_int[4], max_x);
  }
  if (threadIdx.x == 5) {
    atomicMax(&cur_geo_fea_int[5], max_y);
  }
  if(threadIdx.x == 7) {
    cur_geo_fea_float[7] = reg_area[reg_id];
  }
  __syncthreads();
  // Convert unsinged int to float
}
// center_x, center_y, min_x, min_y, max_x, max_y, reg_area

__global__ void process_geo_fea(float* geo_fea, const bool* reg, const int num_reg, const int width, const int height) {
  int shared_size = max(width, height);
  //int half_shared_size = shared_size / 2;
  int thread_id = threadIdx.x;
  int reg_id = blockIdx.x;
  extern __shared__ int shared_space[]; 

  float width_float = static_cast<float>(width);
  float height_float = static_cast<float>(height);
  int *cur_geo_fea_int = reinterpret_cast<int*>(geo_fea + reg_id * NUM_GEO_FEA);
  float *cur_geo_fea_float = geo_fea + reg_id * NUM_GEO_FEA;
  const bool *cur_reg = reg + width * height * reg_id;
  float center_x = cur_geo_fea_float[0];
  float center_y = cur_geo_fea_float[1];
  int center_x_int = static_cast<int>(center_x)-1;
  int center_y_int = static_cast<int>(center_y)-1;
  float normalized_center_x = center_x / width_float;
  float normalized_center_y = center_y / height_float;
  float normalized_min_x = cur_geo_fea_int[2] / width_float;
  float normalized_min_y = cur_geo_fea_int[3] / height_float;
  float normalized_max_x = cur_geo_fea_int[4] / width_float;
  float normalized_max_y = cur_geo_fea_int[5] / height_float;
  int id_left = center_y_int; // the index of the left most pixel on the same raw of center
  int id_top = center_x_int * height; // the index of the top most pixel on the same col of center
  __syncthreads();

  //shared_space to compute min_center_x
  for (int i = thread_id; i < width; i += blockDim.x) { // shared_space == blockDim.x
    if (cur_reg[id_left + i * height]) {
      shared_space[i] = i + 1; // store x cooridnates
    } else {
      shared_space[i] = width; // if the pixel not on the region, set the maximum coordinate
    }
  }
  __syncthreads();
  // Continue after syncthreads
  // peform reduction to compute minimum x coordinates
  for (int s = (width + 1) / 2, e = width / 2; s > 0; s = (s + 1) / 2, e /= 2) {
    if (thread_id < e || thread_id == 0) {
      atomicMin(&shared_space[thread_id], shared_space[thread_id + s]);
    }
    __syncthreads();
    if (s == 1) break;
  }

  float min_center_x;
  if (thread_id == 0) {
    min_center_x = shared_space[0] / width_float;
  }

  //////////////////////////////////////////////////////////////////
  // reload shared_space to compute max_center_x
  for (int i = thread_id; i < width; i += blockDim.x) {
    if (cur_reg[id_left + i * height]) {
      shared_space[i] = i + 1; // store x cooridnates
    } else {
      shared_space[i] = 0; // if the pixel not on the region, set 0
    }
  }
  __syncthreads();
  // Continue after syncthreads
  // peform reduction to compute maximun x coordinates
  for (int s = (width + 1) / 2, e = width / 2; s > 0; s = (s + 1) / 2, e /= 2) {
    if (thread_id < e || thread_id ==0) {
      atomicMax(&shared_space[thread_id], shared_space[thread_id + s] );
    }
    __syncthreads();
    if (s == 1) break;
  }
  float max_center_x;
  if (thread_id == 0) {
    max_center_x = shared_space[0] / width_float;
  }
  //////////////////////////////////////////////////////////////////
  // reload shared_space to compute min_center_y
  for (int i = thread_id; i < height; i += blockDim.x) {
    if (cur_reg[id_top + i]) {
      shared_space[i] = i + 1; // store y cooridnates
    } else {
      shared_space[i] = height; // if the pixel not on the region, set maximun y
    }
  }
  __syncthreads();
  // Continue after syncthreads
  // peform reduction to compute minimun y coordinates
  for (int s = (height + 1) / 2, e = height / 2; s > 0; s = (s + 1) / 2, e /= 2) {
    if (thread_id < e || thread_id == 0) {
      atomicMin(&shared_space[thread_id], shared_space[thread_id + s] );
    }
       __syncthreads();
    if (s == 1) break;
  }
  float min_center_y;
  if (thread_id == 0) {
    min_center_y = shared_space[0] / height_float;
  }

  //////////////////////////////////////////////////////////////////
  // reload shared_space to compute max_center_y
  for (int i = thread_id; i < height; i += blockDim.x) {
    if (cur_reg[id_top + i]) {
      shared_space[i] = i + 1; // store y cooridnates
    } else {
      shared_space[i] = 0; // if the pixel not on the region, set to 0
    }
  }
  __syncthreads();
  // Continue after syncthreads
  // peform reduction to compute maximum y coordinates
  for (int s = (height + 1) / 2, e = height / 2; s > 0; s = (s + 1) / 2, e /= 2) {
    if (thread_id < e || thread_id == 0) {
      atomicMax(&shared_space[thread_id], shared_space[thread_id + s] );
    }
    __syncthreads();
    if (s == 1) break;
  }
  float max_center_y;
  if (thread_id == 0) {
    max_center_y = shared_space[0] / height_float;
    //}
    /////////////////////////////////////////////////////////////////////
    //if (thread_id == 0) {
  cur_geo_fea_float[0] = normalized_center_x; 
  cur_geo_fea_float[1] = normalized_center_y;
  cur_geo_fea_float[2] = normalized_max_x -normalized_min_x;
  cur_geo_fea_float[3] = normalized_max_y - normalized_min_y;
  cur_geo_fea_float[4] = (normalized_max_x - normalized_min_x) / (normalized_max_y - normalized_min_y);

  cur_geo_fea_float[5] = max_center_x - min_center_x;
  cur_geo_fea_float[6] = max_center_y - min_center_y;
  //cur_geo_fea_float[5] = min_center_y;
  //cur_geo_fea_float[6] = max_center_y; 
  float reg_area = cur_geo_fea_float[7];
  float normalized_reg_area = reg_area / (height_float * width_float);
  cur_geo_fea_float[7] = normalized_reg_area;
  cur_geo_fea_float[8] = normalized_reg_area / (normalized_max_x - normalized_min_x) / (normalized_max_y - normalized_min_y) ;
}
}
// central x, y, bounding box width, height, aspect ratio, major_length, minor_length, area, area/box, 


/**
 * Host routine
 */

//void compute_hist(const int *h_Qfea,const float *h_fea, const bool *h_lsm, const bool *h_Rlsm, const bool *h_reg, const int width, const int height, const int num_reg, const int num_bin, const int num_fea, const int num_ls_fea, unsigned int *h_hist, float *Lfea, float *LRfea ) {
void ComputeFeature(const int *host_qualified_fea, const float *host_fea, const bool *host_local_map, const bool *host_refined_local_map, const bool *host_reg, const int width, const int height, const int num_reg, const int num_bin, const int num_fea, const int num_local_fea, const float local_map_sum, const float refined_local_map_sum, float *host_hist, float *host_fea_mean, float *host_fea_var,  float* host_local_fea, float* host_refined_local_fea, float* host_geo_fea) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  size_t size_qualified_fea = width * height * num_fea * sizeof(int);
  size_t size_fea = width * height * COLOR_CHANNEL * num_fea * sizeof(float);
  size_t size_local_map = width * height * sizeof(bool);
  size_t size_refined_local_map = width * height * sizeof(bool);
  size_t size_reg = width * height * num_reg * sizeof(bool);
  size_t size_hist = num_bin * num_reg * num_fea * sizeof(float);
  size_t size_fea_mean = num_fea * COLOR_CHANNEL * num_reg * sizeof(float);
  size_t size_fea_var = num_fea * COLOR_CHANNEL * num_reg * sizeof(float);
  size_t size_local_fea = num_local_fea * num_reg * sizeof(float);
  size_t size_refined_local_fea = num_local_fea * num_reg * sizeof(float);
  size_t size_reg_area = num_reg * sizeof(float);
  //size_t size_local_fea_space = num_local_fea * num_reg * sizeof(float);
  size_t size_shared_hist = num_bin * sizeof(float);
  size_t size_shared_mean = num_fea * COLOR_CHANNEL * sizeof(float);
  size_t size_shared_var = num_fea * COLOR_CHANNEL * sizeof(float);
  size_t size_geo_fea = num_reg * NUM_GEO_FEA * sizeof(float);

  //  printf("cuda hist size: %d * %d * %d", num_bin, num_fea, num_reg);
  // Allocate the device_qualified_fea
  int *device_qualified_fea = NULL;
  AllocateDevice(qualified_fea);
  float *device_fea;
  AllocateDevice(fea);
  bool *device_local_map = NULL;
  AllocateDevice(local_map);
  bool *device_refined_local_map = NULL;
  AllocateDevice(refined_local_map);
  bool *device_reg = NULL;
  AllocateDevice(reg);
  float *device_hist = NULL;
  AllocateDevice(hist);
  float *device_fea_mean = NULL;
  AllocateDevice(fea_mean);
  float *device_fea_var = NULL;
  AllocateDevice(fea_var);
  float *device_local_fea = NULL;
  AllocateDevice(local_fea);
  float *device_refined_local_fea = NULL;
  AllocateDevice(refined_local_fea);
  float *device_reg_area = NULL;
  AllocateDevice(reg_area);
 // float *device_local_fea_space = NULL;
 // AllocateDevice(local_fea_space);
  float *device_geo_fea = NULL;
  AllocateDevice(geo_fea);

  CopyHosttoDevice(qualified_fea);
  CopyHosttoDevice(fea);
  CopyHosttoDevice(local_map);
  CopyHosttoDevice(refined_local_map);
  CopyHosttoDevice(reg);
  //CopyHosttoDevice(hist);
  //CopyHosttoDevice(fea_mean);
  //CopyHosttoDevice(fea_var);
  // CopyHosttoDevice(local_fea);
  //CopyHosttoDevice(refined_local_fea);

  SetDevice(hist, 0);
  SetDevice(fea_mean, 0);
  SetDevice(fea_var, 0);
  SetDevice(local_fea, 0);
  SetDevice(refined_local_fea, 0);
 // SetDevice(local_fea_space, 0);
  SetDevice(reg_area, 0);
  SetDevice(geo_fea, 0);


  // Launch the compute_local_saliency CUDA Kernel for local saliency map
  int threads_per_block = 256;
  int blocks_per_reg = (width * height + threads_per_block - 1) / threads_per_block;
  int blocks_per_grid = num_reg * blocks_per_reg;
  compute_local_saliency<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(device_local_map, device_reg, true, width, height, num_reg, num_local_fea, local_map_sum, blocks_per_reg, device_local_fea, device_reg_area);
  CUDA_CHECK(err);
  blocks_per_grid = (num_reg + threads_per_block - 1) / threads_per_block;
  compute_pr<<<blocks_per_grid, threads_per_block >>>(device_local_fea, device_reg_area, local_map_sum, num_reg, num_local_fea);
  CUDA_CHECK(err);
  // Launch the compute_local_saliency CUDA Kernel for refined local saliency map
  blocks_per_grid = num_reg * blocks_per_reg;
  compute_local_saliency<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(device_refined_local_map, device_reg, false, width, height, num_reg, num_local_fea, refined_local_map_sum, blocks_per_reg, device_refined_local_fea, device_reg_area);
  CUDA_CHECK(err);
  blocks_per_grid = (num_reg + threads_per_block - 1) / threads_per_block;
  compute_pr<<<blocks_per_grid, threads_per_block >>>(device_refined_local_fea, device_reg_area, refined_local_map_sum, num_reg, num_local_fea);
  CUDA_CHECK(err);
  // Launch the compute_color_hist CUDA Kernel
  blocks_per_grid = num_reg * blocks_per_reg;
  dim3 grid_dim;
  grid_dim.x = blocks_per_grid;
  grid_dim.y = num_fea; 
  compute_color_hist<<<grid_dim, threads_per_block, size_shared_hist>>>(device_qualified_fea, device_reg, device_reg_area,  width, height, num_reg, num_fea, num_bin, blocks_per_reg, device_hist);
  // Launch the compute_color_mean CUDA Kernel
  compute_color_mean<<<blocks_per_grid, threads_per_block, size_shared_mean>>>(device_fea, device_reg, device_reg_area, width, height, num_reg, num_fea, blocks_per_reg, device_fea_mean);
  // Launch the compute_var CUDA Kernel 
  compute_color_var<<<blocks_per_grid, threads_per_block, size_shared_var>>>(device_fea, device_fea_mean, device_reg, device_reg_area,  width, height, num_reg, num_fea, blocks_per_reg, device_fea_var);
  // Launch the compute_geo_fea CUDA Kernel
  compute_geo_fea<<<blocks_per_grid, threads_per_block>>>(device_reg, device_reg_area, width, height, num_reg, blocks_per_reg, device_geo_fea);
  threads_per_block = max(height, width);
  size_t size_shared_space = max(height, width) * sizeof(int);
  process_geo_fea<<<num_reg, threads_per_block, size_shared_space>>>(device_geo_fea, device_reg, num_reg, width, height);
  // central x, y, bounding box width, height, aspect ratio, major_length, minor_length, area, area/box, 
  CopyDevicetoHost(hist);
  CopyDevicetoHost(fea_mean);
  CopyDevicetoHost(fea_var);
  CopyDevicetoHost(local_fea);
  CopyDevicetoHost(refined_local_fea);
  CopyDevicetoHost(geo_fea);
  // Free device global memory
  FreeDevice(qualified_fea);
  FreeDevice(fea);
  FreeDevice(local_map);
  FreeDevice(refined_local_map);
  FreeDevice(reg);
  FreeDevice(hist);
  FreeDevice(fea_mean);
  FreeDevice(fea_var);
  FreeDevice(local_fea);
  FreeDevice(refined_local_fea);
  FreeDevice(reg_area);
 // FreeDevice(local_fea_space);
  FreeDevice(geo_fea);
  // Free host memory
  // free(h_img);
  // free(h_reg);
  // free(h_hist);

  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  //  err = cudaDeviceReset();
  //
  //  if (err != cudaSuccess)
  //  {
  //    fprintf(stderr, "Failed to deinitialize the device! error=%s\num_reg", cudaGetErrorString(err));
  //    exit(EXIT_FAILURE);
  //  }

  return;
}

