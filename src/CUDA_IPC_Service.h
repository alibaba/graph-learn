#ifndef IPC_ENV_H
#define IPC_ENV_H
#include <semaphore.h>
#include "BuildInfo.h"

class IPCEnv {
public:
  virtual void Coordinate(BuildInfo* info) = 0;
  virtual int32_t GetMaxStep() = 0;

  virtual void InitializeBuffer(int32_t batch_size, int32_t num_ids, int32_t feature_dim, int32_t device_id, int32_t pipeline_depth) = 0;

  virtual int32_t GetRawBatchsize() = 0;
  virtual int32_t GetLocalBatchId(int32_t global_batch_id) = 0;
  virtual int32_t GetCurrentBatchsize(int32_t dev_id, int32_t current_mode) = 0;
  virtual int32_t GetCurrentMode(int32_t global_batch_id) = 0;
  
  virtual int32_t* GetIds(int32_t dev_id, int32_t current_pipe) = 0;
  virtual float* GetFloatFeatures(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetLabels(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetAggSrc(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetAggDst(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetNodeCounter(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetEdgeCounter(int32_t dev_id, int32_t current_pipe) = 0;

  virtual void IPCPost(int32_t dev_id, int32_t current_pipe) = 0;
  virtual void IPCWait(int32_t dev_id, int32_t current_pipe) = 0;

  virtual void Finalize() = 0;
};
 
IPCEnv* NewIPCEnv(int32_t device_count);

#endif