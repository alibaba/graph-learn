#include "Operator.h"
#include "GPUGraphStore.cuh"
#include "GPU_Graph_Storage.cuh"
#include "GPU_Node_Storage.cuh"
#include "CUDA_IPC_Service.h"
#include "GPUCache.cuh"
#include "Kernels.cuh"
#include "GPUMemoryPool.cuh"

class Batch_Generator : public Operator {
public:
    Batch_Generator(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        GPUNodeStorage* noder = (GPUNodeStorage*)(params->noder);
        GPUCache* cache = (GPUCache*)(params->cache);
        GPUMemoryPool* memorypool = (GPUMemoryPool*)(params->memorypool);
        int32_t mode = memorypool->GetCurrentMode();
        int32_t iter = memorypool->GetIter();
        IPCEnv* env = (IPCEnv*)(params->env);
        int32_t device_id = params->device_id;
        int32_t batch_size = env->GetCurrentBatchsize(device_id, mode);

        batch_generator_kernel(params->stream, noder, cache, memorypool, batch_size, iter, device_id, device_id, mode);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewBatchGenerator(int op_id){
    return new Batch_Generator(op_id);
}

class Random_Sampler : public Operator {
public:
    Random_Sampler(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        GPUMemoryPool* memorypool = (GPUMemoryPool*)(params->memorypool);
        int32_t count = params->neighbor_count;
        GPUGraphStorage* graph = (GPUGraphStorage*)(params->graph);
        GPUCache* cache = (GPUCache*)(params->cache);

        GPU_Random_Sampling(params->stream, graph, cache, memorypool, count, op_id_);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewRandomSampler(int op_id){
    return new Random_Sampler(op_id);
}

class Feature_Extractor : public Operator {
public:
    Feature_Extractor(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        GPUNodeStorage* noder = (GPUNodeStorage*)(params->noder);
        GPUCache* cache = (GPUCache*)(params->cache);
        GPUMemoryPool* memorypool = (GPUMemoryPool*)(params->memorypool);
        
        get_feature_kernel(params->stream, cache, noder, memorypool, params->device_id, op_id_);
    }
private:
    int op_id_;
};

Operator* NewFeatureExtractor(int op_id){
    return new Feature_Extractor(op_id);
}

class Cache_Planner : public Operator {
public:
    Cache_Planner(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        GPUGraphStorage* graph = (GPUGraphStorage*)(params->graph);
        GPUCache* cache = (GPUCache*)(params->cache);
        GPUMemoryPool* memorypool = (GPUMemoryPool*)(params->memorypool);
        int mode = memorypool->GetCurrentMode();

        make_update_plan(params->stream, graph, cache, memorypool, params->device_id, mode);
        cudaEventRecord(((params->event)), ((params->stream)));
    }
private:
    int op_id_;
};

Operator* NewCachePlanner(int op_id){
    return new Cache_Planner(op_id);
}

class Cache_Updater : public Operator {
public:
    Cache_Updater(int op_id) {
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        GPUCache* cache = (GPUCache*)(params->cache);
        GPUNodeStorage* noder = (GPUNodeStorage*)(params->noder);
        GPUMemoryPool* memorypool = (GPUMemoryPool*)(params->memorypool);
        int mode = memorypool->GetCurrentMode();

        update_cache(params->stream, cache, noder, memorypool, params->device_id, mode);
        cudaEventRecord(((params->event)), ((params->stream)));
    }
private:
    int op_id_;
};

Operator* NewCacheUpdater(int op_id){
    return new Cache_Updater(op_id);
}

