#include <iostream>
#include <vector>
#include "helper_multiprocess.h"
#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "ipc_service.h"

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#define MAX_DEVICE 8
#define PIPELINE_DEPTH 2
#define MEMORY_USAGE 7

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

  typedef struct shmStruct_st {
    int32_t steps[3];
    cudaIpcMemHandle_t memHandle[MAX_DEVICE][PIPELINE_DEPTH][MEMORY_USAGE];
  } shmStruct;

class GPUIPCEnv : public IPCEnv {
public: 
  int Initialize() override {
    volatile shmStruct *shm = NULL;
    int central_device = -1;
    cudaGetDevice(&central_device);
    cudaCheckError();
    sharedMemoryInfo info;
    const char shmName[] = "simpleIPCshm";
    if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
      printf("Failed to create shared memory slab\n");
      exit(EXIT_FAILURE);
    }

    shm = (volatile shmStruct *)info.addr;
    train_step_ = shm->steps[0];
    valid_step_ = shm->steps[1];
    test_step_ = shm->steps[2];
    ids_.resize(PIPELINE_DEPTH);
    float_features_.resize(PIPELINE_DEPTH);
    labels_.resize(PIPELINE_DEPTH);
    agg_src_.resize(PIPELINE_DEPTH);
    agg_dst_.resize(PIPELINE_DEPTH);
    node_counter_.resize(PIPELINE_DEPTH);
    edge_counter_.resize(PIPELINE_DEPTH);

    for(int i = 0; i < PIPELINE_DEPTH; i++){
      cudaIpcOpenMemHandle(&ids_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][0], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&float_features_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][1], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&labels_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][2], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&agg_src_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][3], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&agg_dst_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][4], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&node_counter_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][5], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&edge_counter_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][6], cudaIpcMemLazyEnablePeerAccess);
      cudaCheckError();
    }
    std::cout<<"CUDA: "<<central_device<<" IPC shared memory opened\n";

    semr_.resize(PIPELINE_DEPTH);
    semw_.resize(PIPELINE_DEPTH);
    for(int i = 0; i < PIPELINE_DEPTH; i++){
      std::string ssr = "sem_r_";
      std::string ssw = "sem_w_";
      std::string ssri = ssr + std::to_string(central_device) + "_" + std::to_string(i);
      std::string sswi = ssw + std::to_string(central_device) + "_" + std::to_string(i);
      semr_[i] = sem_open(ssri.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semr_[i] == SEM_FAILED ){
        printf("errno = %d\n", errno );
        return -1;
      }
      semw_[i] = sem_open(sswi.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semw_[i] == SEM_FAILED){
        printf("errno = %d\n", errno );
        return -1;
      }
      sem_post(semr_[i]);
    }

    current_pipe_ = 0;
    return central_device;
  }

  void Wait() override {
    sem_t* sem = semw_[current_pipe_];
    sem_wait(sem);
  }

  void Post() override {
    sem_t* sem = semr_[current_pipe_];
    sem_post(sem);
    current_pipe_ = (current_pipe_ + 1)%PIPELINE_DEPTH;
  }

  int32_t* GetIds() override {
    return (int32_t*)ids_[current_pipe_];
  }
  float* GetFloatFeatures() override {
    return (float*)float_features_[current_pipe_];
  }
  int32_t* GetLabels() override {
    return (int32_t*)labels_[current_pipe_];
  }
  int32_t* GetAggSrc() override {
    return (int32_t*)agg_src_[current_pipe_];
  }
  int32_t* GetAggDst() override {
    return (int32_t*)agg_dst_[current_pipe_];
  }
  int32_t* GetNodeCounter() override {
    return (int32_t*)(node_counter_[current_pipe_]);
  }

  int32_t* GetEdgeCounter() override {
    return (int32_t*)(edge_counter_[current_pipe_]);
  }
  int32_t GetTrainStep() override {
    return train_step_;
  }
  int32_t GetValidStep() override {
    return valid_step_;
  }
  int32_t GetTestStep() override {
    return test_step_;
  }

  void Finalize() override {
    for(int i = 0; i < PIPELINE_DEPTH; i++){
      cudaIpcCloseMemHandle(ids_[i]);
      cudaIpcCloseMemHandle(float_features_[i]);
      cudaIpcCloseMemHandle(labels_[i]);
      cudaIpcCloseMemHandle(agg_src_[i]);
      cudaIpcCloseMemHandle(agg_dst_[i]);
      cudaIpcCloseMemHandle(node_counter_[i]);
      cudaIpcCloseMemHandle(edge_counter_[i]);
      sem_t* sem = semw_[i];
      if(sem_close(sem) == -1){
        std::cout<<"close sem "<<i<<" failed\n";
      }
    }
  }
private:
  std::vector<void*> ids_;
  std::vector<void*> float_features_;
  std::vector<void*> labels_;
  std::vector<void*> agg_src_;
  std::vector<void*> agg_dst_;
  std::vector<void*> node_counter_;
  std::vector<void*> edge_counter_;
  std::vector<sem_t*> semw_;
  std::vector<sem_t*> semr_;

  int32_t train_step_;
  int32_t valid_step_;
  int32_t test_step_;
  int current_pipe_;
};
IPCEnv* NewIPCEnv(){
  return new GPUIPCEnv();
}

// Define the GPU implementation that launches the CUDA kernel.

std::vector<torch::Tensor> cuda_get_next(
    int32_t* ids,
    float* float_features, 
    int32_t* labels,
    int feature_dim,
    int32_t* agg_src,
    int32_t* agg_dst,
    int32_t* node_counter,
    int32_t* edge_counter,
    int32_t* h_node_counter,
    int32_t* h_edge_counter
    ){
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);
    cudaCheckError();

    cudaMemcpy(h_node_counter, node_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_counter, edge_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    torch::Tensor block1_agg_src_tensor = torch::from_blob(
      agg_src,
        {(long long)h_edge_counter[4]},
        torch::TensorOptions().dtype(torch::kI32).device(device));
    torch::Tensor block1_agg_dst_tensor = torch::from_blob(
      agg_dst,
        {(long long)h_edge_counter[4]},
        torch::TensorOptions().dtype(torch::kI32).device(device));
    torch::Tensor block2_agg_src_tensor = torch::from_blob(
      agg_src,
        {(long long)h_edge_counter[3]},
        torch::TensorOptions().dtype(torch::kI32).device(device));
    torch::Tensor block2_agg_dst_tensor = torch::from_blob(
      agg_dst,
        {(long long)h_edge_counter[3]},
        torch::TensorOptions().dtype(torch::kI32).device(device));
    torch::Tensor ids_tensor = torch::from_blob(
      ids,
      {(long long)h_node_counter[9]},
      torch::TensorOptions().dtype(torch::kI32).device(device));

    torch::Tensor feature_tensor = torch::from_blob(
      float_features,
      {(long long)(h_node_counter[9]), (long long)(feature_dim)},
      torch::TensorOptions().dtype(torch::kF32).device(device));

    torch::Tensor labels_tensor = torch::from_blob(
      labels,
      {(long long)h_node_counter[5]},
      torch::TensorOptions().dtype(torch::kI32).device(device));      

    return {ids_tensor, feature_tensor, labels_tensor, block1_agg_src_tensor, block1_agg_dst_tensor, block2_agg_src_tensor, block2_agg_dst_tensor};
}
