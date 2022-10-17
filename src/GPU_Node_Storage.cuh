#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_NODE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_NODE_STORAGE_H_

#include <cstdint>
#include <string>
#include <vector>
#include "BuildInfo.h"

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
  

class GPUNodeStorage {
public: 
    virtual ~GPUNodeStorage() = default;

    virtual void Build(BuildInfo* info) = 0;
    virtual void Finalize() = 0;

    virtual int32_t* GetTrainingSetIds(int32_t part_id) const = 0;
    virtual int32_t* GetValidationSetIds(int32_t part_id) const = 0;
    virtual int32_t* GetTestingSetIds(int32_t part_id) const = 0;

    virtual int32_t* GetTrainingLabels(int32_t part_id) const = 0;
    virtual int32_t* GetValidationLabels(int32_t part_id) const = 0;
    virtual int32_t* GetTestingLabels(int32_t part_id) const = 0;

    virtual int32_t TrainingSetSize(int32_t part_id) const = 0;
    virtual int32_t ValidationSetSize(int32_t part_id) const = 0;
    virtual int32_t TestingSetSize(int32_t part_id) const = 0;

    virtual int32_t TotalNodeNum() const = 0;
    virtual int64_t* GetAllIntAttr() const = 0;
    virtual int32_t GetIntAttrLen() const = 0;
    virtual float* GetAllFloatAttr() const = 0;
    virtual int32_t GetFloatAttrLen() const = 0;
  
};

extern "C" 
GPUNodeStorage* NewGPUMemoryNodeStorage();

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_NODE_STORAGE_H_