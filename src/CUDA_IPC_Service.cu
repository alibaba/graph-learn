#include "CUDA_IPC_Service.h"

#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>

#include <chrono>
#include <vector>
#include <semaphore.h>

#include "helper_multiprocess.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_DEVICE 8
#define PIPELINE_DEPTH 2
#define MEMORY_USAGE 7
#define TRAINMODE 0
#define VALIDMODE 1
#define TESTMODE  2

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

class CUDAIPCEnv : public IPCEnv {
public:
  CUDAIPCEnv(int32_t device_count){
    const char shmName[] = "simpleIPCshm";
    if (sharedMemoryCreate(shmName, sizeof(*shm_), &info_) != 0) {
      printf("Failed to create shared memory slab\n");
      exit(EXIT_FAILURE);
    }
    shm_ = (volatile shmStruct *)info_.addr;
    memset((void *)shm_, 0, sizeof(*shm_));
    ids_.resize(device_count);
    float_features_.resize(device_count);
    labels_.resize(device_count);
    agg_src_.resize(device_count);
    agg_dst_.resize(device_count);
    node_counter_.resize(device_count);
    edge_counter_.resize(device_count);
    device_count_ = device_count;

    semr_.resize(device_count);
    semw_.resize(device_count);
  }
  
  void Coordinate(BuildInfo* info) override {
    int32_t partition_count = info->partition_count;
    epoch_ = info->epoch;

    int32_t max_train_size = 0;
    for(int32_t i = 0; i < partition_count; i++){
      int32_t ts_num = info->training_set_num[i];
      if(ts_num > max_train_size){
        max_train_size = ts_num;
      }
    }
    train_step_ = (max_train_size - 1) / raw_batch_size_ + 1;
    for(int32_t i = 0; i < partition_count; i++){
      train_batch_size_.push_back(((info->training_set_num[i] - 1)/train_step_ + 1));
    }

    int32_t max_valid_size = 0;
    for(int32_t i = 0; i < partition_count; i++){
      int32_t ts_num = info->validation_set_num[i];
      if(ts_num > max_valid_size){
        max_valid_size = ts_num;
      }
    }
    valid_step_ = (max_valid_size - 1) / raw_batch_size_ + 1;
    for(int32_t i = 0; i < partition_count; i++){
      valid_batch_size_.push_back(((info->validation_set_num[i] - 1)/valid_step_ + 1));
    }

    int32_t max_test_size = 0;
    for(int32_t i = 0; i < partition_count; i++){
      int32_t ts_num = info->testing_set_num[i];
      if(ts_num > max_test_size){
        max_test_size = ts_num;
      }
    }
    test_step_ = (max_test_size - 1) / raw_batch_size_ + 1;
    for(int32_t i = 0; i < partition_count; i++){
      test_batch_size_.push_back(((info->testing_set_num[i] - 1)/test_step_ + 1));
    }

    shm_->steps[0] = train_step_;
    shm_->steps[1] = valid_step_;
    shm_->steps[2] = test_step_;
  }

  int32_t GetMaxStep() override {
    return (((train_step_ + valid_step_) * epoch_) + test_step_);
  }

  void InitializeBuffer(int32_t batch_size, int32_t num_ids, int32_t feature_dim, int32_t device_id, int32_t pipeline_depth) override {
    cudaSetDevice(device_id);
    std::string ssr = "sem_r_";
    std::string ssw = "sem_w_";
    (semr_[device_id]).resize(pipeline_depth);
    (semw_[device_id]).resize(pipeline_depth);
    for(int32_t i = 0; i < PIPELINE_DEPTH; i++){
      void* new_ids;
      cudaMalloc(&new_ids, num_ids * sizeof(int32_t));
      cudaCheckError();
      void* new_features;
      cudaMalloc(&new_features, int64_t(int64_t(int64_t(num_ids) * feature_dim) * sizeof(float)));
      cudaCheckError();
      void* new_labels;
      cudaMalloc(&new_labels, batch_size * sizeof(int32_t));
      cudaCheckError();
      void* new_agg_src;
      cudaMalloc(&new_agg_src, num_ids * sizeof(int32_t));
      cudaCheckError();
      void* new_agg_dst;
      cudaMalloc(&new_agg_dst, num_ids * sizeof(int32_t));
      cudaCheckError();
      void* new_node_counter;
      cudaMalloc(&new_node_counter, 16 * sizeof(int32_t));
      cudaCheckError();
      void* new_edge_counter;
      cudaMalloc(&new_edge_counter, 16 * sizeof(int32_t));
      cudaCheckError();

      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][0], new_ids);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][1], new_features);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][2], new_labels);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][3], new_agg_src);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][4], new_agg_dst);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][5], new_node_counter);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][6], new_edge_counter);
      cudaCheckError();

      ids_[device_id].push_back(new_ids);
      float_features_[device_id].push_back(new_features);
      labels_[device_id].push_back(new_labels);
      agg_src_[device_id].push_back(new_agg_src);
      agg_dst_[device_id].push_back(new_agg_dst);
      node_counter_[device_id].push_back(new_node_counter);
      edge_counter_[device_id].push_back(new_edge_counter);

      //memory lock
      std::string ssri = ssr + std::to_string(device_id) + "_" + std::to_string(i);
      std::string sswi = ssw + std::to_string(device_id) + "_" + std::to_string(i);
      semr_[device_id][i] = sem_open(ssri.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semr_[device_id][i] == SEM_FAILED ){
        printf("errno = %d\n", errno );
        return;
      }
      semw_[device_id][i] = sem_open(sswi.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semw_[device_id][i] == SEM_FAILED){
        printf("errno = %d\n", errno );
        return;
      }
    }
    pipeline_depth_ = PIPELINE_DEPTH;
  }

  int32_t GetRawBatchsize() override {
    return raw_batch_size_;
  }

  int32_t GetLocalBatchId(int32_t global_batch_id) override {
    int32_t local_batch_id = -1;
    if((global_batch_id) < ((train_step_ + valid_step_) * epoch_)){//train & valid
      int32_t epoch_batch_id = global_batch_id % (train_step_ + valid_step_);
      if(epoch_batch_id < train_step_){
        local_batch_id = epoch_batch_id;
      }else{
        local_batch_id = epoch_batch_id - train_step_;
      }
    }else{//test
      int32_t epoch_batch_id = (global_batch_id - ((train_step_ + valid_step_) * epoch_))%test_step_;
      local_batch_id = epoch_batch_id;
    }
    return local_batch_id;
  }

  int32_t GetCurrentBatchsize(int32_t dev_id, int32_t current_mode) override {
    if(current_mode == TRAINMODE){//train
      return train_batch_size_[dev_id];
    }else if(current_mode == VALIDMODE){//valid
      return valid_batch_size_[dev_id];
    }else{//test
      return test_batch_size_[dev_id];
    }
  }

  //tape_id / shard_count = batch_id
  int32_t GetCurrentMode(int32_t global_batch_id) override {
    int32_t current_mode;
    if((global_batch_id) < ((train_step_ + valid_step_) * epoch_)){//train & valid
      int32_t epoch_batch_id = global_batch_id % (train_step_ + valid_step_);
      if(epoch_batch_id < train_step_){
        current_mode = TRAINMODE;
      }else{
        current_mode = VALIDMODE;
      }
    }else{//test
      current_mode = TESTMODE;
    }
    return current_mode;
  }

  int32_t* GetIds(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(ids_[dev_id][current_pipe%pipeline_depth_]);
  }

  float* GetFloatFeatures(int32_t dev_id, int32_t current_pipe) override {
    return (float*)(float_features_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetLabels(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(labels_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetAggSrc(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(agg_src_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetAggDst(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(agg_dst_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetNodeCounter(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(node_counter_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetEdgeCounter(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(edge_counter_[dev_id][current_pipe%pipeline_depth_]);
  }

  void IPCPost(int32_t dev_id, int32_t current_pipe) override {
    sem_t* sem = semw_[dev_id][current_pipe];
    sem_post(sem);
  }

  void IPCWait(int32_t dev_id, int32_t current_pipe) override {
    sem_t* sem = semr_[dev_id][current_pipe];
    sem_wait(sem);
  }

  void Finalize() override {
    for(int32_t i = 0; i < device_count_; i++){
      cudaSetDevice(i);
      for(int32_t j = 0; j < PIPELINE_DEPTH; j++){
        cudaFree(ids_[i][j]);
        cudaFree(float_features_[i][j]);
        cudaFree(labels_[i][j]);
        cudaFree(agg_src_[i][j]);
        cudaFree(agg_dst_[i][j]);
        cudaFree(node_counter_[i][j]);
        cudaFree(edge_counter_[i][j]);
        cudaCheckError();
        sem_t* sem = semw_[i][j];
        if(sem_close(sem) == -1){
          std::cout<<"close sem "<<i<<" "<<j<<" failed\n";
        }
      }
    }

    sharedMemoryClose(&info_);
  }
  
private:
  volatile shmStruct *shm_;
  sharedMemoryInfo info_;
  std::vector<std::vector<void*>> ids_;
  std::vector<std::vector<void*>> float_features_;
  std::vector<std::vector<void*>> labels_;
  std::vector<std::vector<void*>> agg_src_;
  std::vector<std::vector<void*>> agg_dst_;
  std::vector<std::vector<void*>> node_counter_;
  std::vector<std::vector<void*>> edge_counter_;
  std::vector<std::vector<sem_t*>> semr_;
  std::vector<std::vector<sem_t*>> semw_; 

  int32_t raw_batch_size_;
  std::vector<int32_t> train_batch_size_;
  std::vector<int32_t> valid_batch_size_;
  std::vector<int32_t> test_batch_size_;

  int32_t device_count_;

  int32_t train_step_;
  int32_t valid_step_;
  int32_t test_step_;

  int32_t epoch_;
  int32_t pipeline_depth_;
};

IPCEnv* NewIPCEnv(int32_t device_count){
  return new CUDAIPCEnv(device_count);
}