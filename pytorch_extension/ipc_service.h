#include <semaphore.h>

class IPCEnv {
public:
  virtual int Initialize() = 0;
  virtual int32_t* GetIds() = 0;
  virtual float* GetFloatFeatures() = 0;
  virtual int32_t* GetLabels() = 0;
  virtual int32_t* GetAggSrc() = 0;
  virtual int32_t* GetAggDst() = 0;
  virtual int32_t* GetNodeCounter() = 0;
  virtual int32_t* GetEdgeCounter() = 0;
  virtual int32_t GetTrainStep() = 0;
  virtual int32_t GetValidStep() = 0;
  virtual int32_t GetTestStep() = 0;
  virtual void Finalize() = 0;
  virtual void Wait() = 0;
  virtual void Post() = 0;
};
IPCEnv* NewIPCEnv();