#ifndef GPU_KERNELS
#define GPU_KERNELS
#include <time.h>
#include <iostream>

#include "GPUMemoryPool.cuh"
#include "GPU_Graph_Storage.cuh"
#include "GPU_Node_Storage.cuh"
#include "GPUCache.cuh"

#define SAMPLING_THREAD_NUM 1024
typedef void* stream_handle;
typedef void* event_handle;

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
extern "C"
void* d_alloc_space(int64_t num_bytes);

extern "C"
void* d_alloc_space_managed(unsigned int num_bytes);

extern "C"
void d_copy_2_h(void* h_ptr, void* d_ptr, unsigned int num_bytes);

extern "C"
void d_free_space(void* d_ptr);

extern "C"
void SetGPUDevice(int32_t shard_id);

extern "C"
int32_t GetGPUDevice();

extern "C"
void* host_alloc_space(unsigned int num_bytes);

extern "C"
void batch_generator_kernel(
	stream_handle strm_hdl, 
	GPUNodeStorage* noder,
	GPUCache* cache,
	GPUMemoryPool* memorypool,
	int32_t batch_size, 
	int32_t counter, 
  int32_t part_id,
  int32_t dev_id,
  int32_t mode);

extern "C"											
void GPU_Random_Sampling(
  stream_handle strm_hdl, 
  GPUGraphStorage* graph,
  GPUCache* cache,
  GPUMemoryPool* memorypool,
  int32_t count,
  int32_t op_id);

extern "C"
void get_feature_kernel(
  stream_handle strm_hdl,
  GPUCache* cache, 
  GPUNodeStorage* noder,
  GPUMemoryPool* memorypool,
  int32_t dev_id,
  int32_t op_id);

extern "C"
void make_update_plan(
	stream_handle strm_hdl, 
	GPUGraphStorage* graph, 
	GPUCache* cache,
  GPUMemoryPool* memorypool,
	int32_t dev_id,
  int32_t mode);

extern "C"
void update_cache(
  stream_handle strm_hdl, 
  GPUCache* cache, 
  GPUNodeStorage* noder,
  GPUMemoryPool* memorypool,
  int32_t dev_id,
  int32_t mode);

#endif