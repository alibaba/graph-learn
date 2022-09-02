#ifndef SERVER_H
#define SERVER_H
#include <vector>

struct RunnerParams {
    int device_id;
    std::vector<int> fanout;
    void* cache;
    void* graph;
    void* noder;
    void* env;
    int global_batch_id;
};

class Server {
    virtual void Initialize(int global_shard_count) = 0;
    virtual void Run() = 0;
    virtual void Finalize() = 0;
};
Server* NewGPUServer();

class Runner {
    virtual void RunOnce(RunnerParams* params) = 0;
};
Runner* NewGPURunner(RunnerParams* params);

#endif