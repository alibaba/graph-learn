#include "GPUGraphStore.cuh"
#include "GPU_Graph_Storage.cuh"
#include "GPU_Node_Storage.cuh"
#include "CUDA_IPC_Service.h"
#include "GPUCache.cuh"
#include "Kernels.cuh"
#include "GPUMemoryPool.cuh"
#include "Operator.h"
#include "Server.h"
#include <thread>
#include <functional>

#define PIPELINE_DEPTH 2

void RunnerLoop(int max_step, Runner* runner, RunnerParams* params){
    for(int i = 0; i < max_step; i++){
        params->global_batch_id = i;
        runner->RunOnce(params);
    }
}

class GPUServer : public Server {
public:
    void Initialize(int global_shard_count) override {
        shard_count_ = global_shard_count;
        std::cout<<"CUDA Device Count: "<<shard_count_<<"\n";
        
        GPUGraphStore* gpu_graph_store = new GPUGraphStore();
        gpu_graph_store->Initialze(shard_count_);
        gpu_graph_storage_ptr_ = gpu_graph_store->GetGraph();
        gpu_node_storage_ptr_ = gpu_graph_store->GetNode();
        gpu_cache_ptr_ = gpu_graph_store->GetCache();
        gpu_ipc_env_ = gpu_graph_store->GetIPCEnv();

        max_step_ = gpu_ipc_env_->GetMaxStep();
        
        for(int i = 0; i < shard_count_; i++){
            RunnerParams* new_params = new RunnerParams();
            new_params->device_id = i;
            (new_params->fanout).push_back(25);
            (new_params->fanout).push_back(10);
            new_params->cache = (void*)gpu_cache_ptr_;
            new_params->graph = (void*)gpu_graph_storage_ptr_;
            new_params->noder = (void*)gpu_node_storage_ptr_;
            new_params->env = (void*)gpu_ipc_env_;
            new_params->global_batch_id = 0;
            params_.push_back(new_params);
            Runner* new_runner = NewGPURunner(new_params);
            runners_.push_back(std::move(new_runner));
        }
    }

    void Run() override {
        for(int i = 0; i < shard_count_; i++){
            Runner* runner = runners_[i];
            RunnerParams* params = params_[i];
            std::thread th(&RunnerLoop, max_step_, runner, params);
            thread_pool_.push_back(std::move(th));
        }
        for(auto &th : thread_pool_){
            th.join();
        }
    }

    void Finalize() override {
        gpu_graph_storage_ptr_->Finalize();
        gpu_node_storage_ptr_->Finalize();
        gpu_cache_ptr_->Finalize();
        gpu_ipc_env_->Finalize();
    }
private:

    GPUGraphStorage* gpu_graph_storage_ptr_;
    GPUNodeStorage* gpu_node_storage_ptr_;
    GPUCache* gpu_cache_ptr_;
    IPCEnv* gpu_ipc_env_;
    int shard_count_;
    int max_step_;

    std::vector<std::thread> thread_pool_;
    std::vector<Runner*> runners_;
    std::vector<RunnerParams*> params_;
};

Server* NewGPUServer(){
    return new GPUServer();
}

class GPURunner : public Runner {
public:
    GPURunner(RunnerParams* params){
        cudaSetDevice(params->device_id);
        local_dev_id_ = params->device_id;

        GPUCache* cache = (GPUCache*)(params->cache);
        GPUGraphStorage* graph = (GPUGraphStorage*)(params->graph);
        GPUNodeStorage* noder = (GPUNodeStorage*)(params->noder);
        IPCEnv* env = (IPCEnv*)(params->env);

        /*initialize GPU environment*/
        streams_.resize(2);
        cudaStreamCreate(&streams_[0]);
        cudaStreamCreate(&streams_[1]);
        events_.resize(2);
        cudaEventCreate(&events_[0]);
        cudaEventCreate(&events_[1]);

        /*dag params analysis*/
        int batch_size = env->GetRawBatchsize();
        int max_ids_num = batch_size;
        std::vector<int32_t> max_num_per_hop;
        int hop_num = (params->fanout).size();
        max_num_per_hop.resize(hop_num);
        max_num_per_hop[0] = batch_size * (params->fanout)[0];
        for(int i = 1; i < hop_num; i++){
            max_num_per_hop[i] = max_num_per_hop[i - 1] * (params->fanout)[i];
        }
        for(int i = 0; i < hop_num; i++){
            max_ids_num += max_num_per_hop[i];
        }
        num_ids_ = max_ids_num;

        op_num_ = (hop_num + 1) * 2 + 2;
        op_factory_.resize(op_num_);
        op_factory_[0] = NewBatchGenerator(0);
        op_factory_[1] = NewFeatureExtractor(1);
        for(int i = 0; i < hop_num; i++){
            op_factory_[2 * i + 2] = NewRandomSampler(2 * i + 2);
            op_factory_[2 * i + 3] = NewFeatureExtractor(2 * i + 3);
        };
        op_factory_[op_num_ - 2] = NewCachePlanner(op_num_ - 2);
        op_factory_[op_num_ - 1] = NewCacheUpdater(op_num_ - 1);

        /*buffer allocation*/
        cache_capacity_ = cache->Capacity();
        int pipeline_depth = PIPELINE_DEPTH;
        pipeline_depth_ = pipeline_depth;

        int total_num_nodes = noder->TotalNodeNum();
        cache->InitializeCacheController(local_dev_id_, cache_capacity_, num_ids_, total_num_nodes, batch_size);/*control cache memory by current actor*/

        memorypool_ = new GPUMemoryPool(pipeline_depth);
        int32_t* cache_search_buffer = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetCacheSearchBuffer(cache_search_buffer);
        int32_t* accessed_map = (int32_t*)d_alloc_space(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
        memorypool_->SetAccessedMap(accessed_map);
        int32_t* position_map = (int32_t*)d_alloc_space(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
        memorypool_->SetPositionMap(position_map);
        int32_t* agg_src_ids = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetAggSrcId(agg_src_ids);
        int32_t* agg_dst_ids = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetAggDstId(agg_dst_ids);

        int32_t float_attr_len = noder->GetFloatAttrLen();
        float_attr_len_ = float_attr_len;
        env->InitializeBuffer(batch_size, num_ids_, float_attr_len_, local_dev_id_, pipeline_depth);
        current_pipe_ = 0;
        for(int i = 0; i < PIPELINE_DEPTH; i++){
          memorypool_->SetSampledIds(env->GetIds(local_dev_id_, i), i);
          memorypool_->SetFloatFeatures(env->GetFloatFeatures(local_dev_id_, i), i);
          memorypool_->SetLabels(env->GetLabels(local_dev_id_, i), i);
          memorypool_->SetAggSrcOf(env->GetAggSrc(local_dev_id_, i), i);
          memorypool_->SetAggDstOf(env->GetAggDst(local_dev_id_, i), i);
          memorypool_->SetNodeCounter(env->GetNodeCounter(local_dev_id_, i), i);
          memorypool_->SetEdgeCounter(env->GetEdgeCounter(local_dev_id_, i), i);
        }
        
        op_params_.resize(op_num_);
        for(int i = 0; i < op_num_; i++){
            op_params_[i] = new OpParams();
            op_params_[i]->device_id = local_dev_id_;
            op_params_[i]->stream = (void*)(&streams_[i%2]);
            op_params_[i]->memorypool = memorypool_;
            op_params_[i]->cache = cache;
            op_params_[i]->graph = graph;
            op_params_[i]->noder = noder;
            op_params_[i]->env = env;
        }
        op_params_[op_num_- 2]->event = (void*)(&events_[0]);
        op_params_[op_num_- 1]->event = (void*)(&events_[1]);

        for(int i = 0; i < hop_num; i++){
            op_params_[2 * i + 2]->neighbor_count = (params->fanout)[i];
        }
    }
    

    void RunOnce(RunnerParams* params) override {
        cudaSetDevice(local_dev_id_);
        IPCEnv* env = (IPCEnv*)(params->env);
        int32_t batch_id = params->global_batch_id;
        mode_ = env->GetCurrentMode(batch_id);
        memorypool_->SetCurrentMode(mode_);
        memorypool_->SetIter(env->GetLocalBatchId(batch_id));
        env->IPCWait(local_dev_id_, current_pipe_);
        
        bool is_ready = false;
        for(int i = 0; i < op_num_; i++){
            op_factory_[i]->run(op_params_[i]);
            if(i % 2 == 0){
                is_ready = false;
                while(!is_ready){
                    if(!(cudaEventQuery(static_cast<cudaEvent_t>(op_params_[i]->event)) == cudaErrorNotReady)){
                        is_ready = true;
                    }
                }
            }
        }
        
        is_ready = false;
        while(!is_ready){
            if(!(cudaEventQuery(static_cast<cudaEvent_t>(op_params_[op_num_-1]->event)) == cudaErrorNotReady)){
                is_ready = true;
            }
        }

        env->IPCPost(local_dev_id_, current_pipe_);
        current_pipe_ = (current_pipe_ + 1) % pipeline_depth_;
        memorypool_ -> SetCurrentPipe(current_pipe_);
    }

private:

    /*vertex id & feature buffer*/
    int32_t num_ids_;
    int32_t float_attr_len_;

    /*buffers for multi gpu task*/
    GPUMemoryPool* memorypool_;

    /*dynamic cache config*/
    int32_t cache_capacity_;

    /*pipeline*/
    int current_pipe_;
    int pipeline_depth_;

    /*map to physical device*/
    int local_dev_id_;

    /*mode, training(0), validation(1), testing(2)*/
    int mode_;
    int op_num_;
    
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    std::vector<Operator*> op_factory_;
    std::vector<OpParams*> op_params_;
};

Runner* NewGPURunner(RunnerParams* params){
    return new GPURunner(params);
}

