#include <vector>
#include "GPU_Graph_Storage.cuh"
#include "GPU_Node_Storage.cuh"
#include "GPUCache.cuh"
#include "CUDA_IPC_Service.h"

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

class GPUGraphStore {
public:
  
  void Initialze(int32_t shard_count);

  GPUGraphStorage* GetGraph();

  GPUNodeStorage* GetNode();

  GPUCache* GetCache(); 

  IPCEnv* GetIPCEnv();

  int32_t Shard_To_Device(int32_t part_id);

  int32_t Shard_To_Partition(int32_t part_id);

  int32_t Central_Device();

private:
  void EnableP2PAccess();

  void ConfigPartition(BuildInfo* info, int32_t shard_count);

  void ReadMetaFIle(BuildInfo* info);

  void Load_Graph(BuildInfo* info);

  void Load_Feature(BuildInfo* info);
  
  int32_t central_device_;
  std::vector<int> shard_to_device_;
  std::vector<int> shard_to_partition_;

  int32_t edge_num_;
  int32_t node_num_;

  int32_t training_set_num_;
  int32_t validation_set_num_;
  int32_t testing_set_num_;

  int32_t float_attr_len_;

  int32_t cache_cap_;
  int32_t cache_way_;
  int32_t future_batch_;

  std::string dataset_path_;
  int32_t raw_batch_size_;
  int32_t epoch_;

  GPUGraphStorage* graph_;
  GPUNodeStorage* node_;
  GPUCache* cache_;
  IPCEnv* env_;
};


