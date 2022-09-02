#include "GPUCache.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

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

#define SHMEMSIZE 10240

#define THRESHOLD1 1
#define THRESHOLD2 0

#define TOTAL_DEV_NUM 8
#define TOL 16
#define NCOUNT_1 4
#define NCOUNT_2 10

__global__ void Find_Kernel(
    int32_t* sampled_ids, 
    int32_t* cache_offset, 
    int32_t* node_counter, 
    int32_t total_num_nodes,
    int32_t op_id,
    int32_t* cache_map)
{
    int32_t batch_size = 0;
	int32_t node_off = 0;
	if(op_id == 5){
		node_off = node_counter[3];
		batch_size = node_counter[4];
	}else if(op_id == 6){
		node_off = node_counter[5];
		batch_size = node_counter[6];
	}else if(op_id == 7){
		node_off = node_counter[7];
		batch_size = node_counter[8];
	}
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < batch_size; thread_idx += gridDim.x * blockDim.x){
        int32_t id = sampled_ids[node_off + thread_idx];
        // offset[thread_idx] = -1;
        if(id < 0){
            cache_offset[thread_idx] = -1;
        }else{
            cache_offset[thread_idx] = cache_map[id%total_num_nodes];
        }
    }
}

__global__ void Init_Int64(
    int64_t* array, 
    int32_t length, 
    int32_t value)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < length; thread_idx += gridDim.x * blockDim.x){
        array[thread_idx] = value;
    }
}

__global__ void Init_Int32(
    int32_t* array, 
    int32_t length, 
    int32_t value)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < length; thread_idx += gridDim.x * blockDim.x){
        array[thread_idx] = value;
    }
}

__global__ void get_current_key(
    int32_t* d_keys, 
    int32_t* current_key, 
    float* d_values, 
    int32_t num_keys, 
    int32_t mode,
    int32_t* is_recent)
{
    if(mode == 0){
        for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_keys; thread_idx += gridDim.x * blockDim.x){
            if(d_values[thread_idx] > THRESHOLD1){
                current_key[thread_idx] = d_keys[thread_idx];
            }else{
                current_key[thread_idx] = -1;
            }
        } 
    }else if(mode == 1){
        for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_keys; thread_idx += gridDim.x * blockDim.x){
            if((d_values[thread_idx] <= THRESHOLD1) && (is_recent[thread_idx] > 0)){
                current_key[thread_idx] = d_keys[thread_idx];
            }else{
                current_key[thread_idx] = -1;
            }
        } 
    }else if(mode == 2){
        for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_keys; thread_idx += gridDim.x * blockDim.x){
            if((d_values[thread_idx] <= THRESHOLD2) && (is_recent[thread_idx] == 0)){
                current_key[thread_idx] = d_keys[thread_idx];
            }else{
                current_key[thread_idx] = -1;
            }
        } 
    }else if(mode == 3){
        for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_keys; thread_idx += gridDim.x * blockDim.x){
            current_key[thread_idx] = d_keys[thread_idx];
        } 
    }
}
/*remove duplicate, and construct candidates map*/
__global__ void filtering(
    int32_t* candidate_fifo, 
    int32_t* candidates, 
    int32_t* candidates_map, 
    int32_t* global_counter, 
    int32_t num_candidates)
{
    __shared__ int32_t local_fifo[1024];
    __shared__ int32_t offset[2];//0 : local offset, 1 : global offset
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        if(threadIdx.x == 0){
            offset[0] = 0;
            offset[1] = 0;
        }
        local_fifo[threadIdx.x] = 0;
        __syncthreads();
        int32_t cand_id = candidates[thread_idx];
        if(cand_id >= 0){
            int32_t map_count = atomicAdd(candidates_map + cand_id, 1);
            if(map_count == 0){
                int32_t local_offset = atomicAdd(offset, 1);
                local_fifo[local_offset] = cand_id;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            offset[1] = atomicAdd(global_counter, offset[0]);
        }
        __syncthreads();
        if(threadIdx.x < offset[0]){
            int32_t global_offset = offset[1];
            candidate_fifo[(global_offset + threadIdx.x)%num_candidates] = local_fifo[threadIdx.x];
        }
    }
}

__global__ void remove_out_cache_candidates(
    int32_t capacity, 
    int32_t num_candidates, 
    int32_t* candidates_map, 
    int32_t* candidate_fifo)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (num_candidates - capacity); thread_idx += gridDim.x * blockDim.x){
        int32_t cand_id = candidate_fifo[capacity + thread_idx];
        if(cand_id < 0){
            continue;
        }
        atomicAnd(candidates_map + cand_id, 0);
    }
}

__global__ void get_empty_position(
    int32_t* cache_ids, 
    int32_t* cache_fifo, 
    int32_t* candidates_map, 
    int32_t* cache_map, 
    int32_t capacity,
    int32_t* global_counter)
{
    __shared__ int32_t local_fifo[1024];
    __shared__ int32_t offset[2];//0 : local offset, 1 : global offset
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        if(threadIdx.x == 0){
            offset[0] = 0;
            offset[1] = 0;
        }
        local_fifo[threadIdx.x] = 0;
        __syncthreads();
        int32_t cache_id = cache_ids[thread_idx];
        if(cache_id < 0){
            int32_t local_offset = atomicAdd(offset, 1);
            local_fifo[local_offset] = thread_idx;//cache offset
        }else{
            if(candidates_map[cache_id] <= 0){
                cache_map[cache_id] = -1;//clear old map
                int32_t local_offset = atomicAdd(offset, 1);
                local_fifo[local_offset] = thread_idx;//cache offset
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            offset[1] = atomicAdd(global_counter + 1, offset[0]);
        }
        __syncthreads();
        if(threadIdx.x < offset[0]){
            int32_t global_offset = offset[1];
            cache_fifo[(global_offset + threadIdx.x)%capacity] = local_fifo[threadIdx.x];
        }
    }
}

__global__ void final_filtering(
    int32_t* candidate_fifo, 
    int32_t* global_counter, 
    int32_t* cache_map, 
    int32_t* final_candidates,
    int32_t capacity)
{
    __shared__ int32_t local_fifo[1024];
    __shared__ int32_t offset[2];//0 : local offset, 1 : global offset
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        if(threadIdx.x == 0){
            offset[0] = 0;
            offset[1] = 0;
        }
        local_fifo[threadIdx.x] = 0;
        __syncthreads();
        int32_t cand_id = candidate_fifo[thread_idx];
        if(cand_id >= 0){
            if(cache_map[cand_id] < 0){
                int32_t local_offset = atomicAdd(offset, 1);
                local_fifo[local_offset] = cand_id;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            offset[1] = atomicAdd(global_counter + 2, offset[0]);
        }
        __syncthreads();
        if(threadIdx.x < offset[0]){
            int32_t global_offset = offset[1];
            final_candidates[(global_offset + threadIdx.x)%capacity] = local_fifo[threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void replace(
    int32_t* global_counter, 
    int32_t* cache_fifo, 
    int32_t* final_candidates, 
    int32_t* cache_map, 
    int32_t* cache_ids,
    int32_t current_offset)
{
    int32_t cache_empty_count = global_counter[1];
    int32_t candidate_count = global_counter[2];
    if(cache_empty_count < candidate_count){
        return;
    }
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < candidate_count; thread_idx += gridDim.x * blockDim.x){
        int32_t cache_offset = cache_fifo[thread_idx];
        int32_t cand_id = final_candidates[thread_idx];
        cache_ids[cache_offset] = cand_id;
        cache_map[cand_id] = cache_offset + current_offset;
    }
}

__global__ void access_count(int32_t* ids, int32_t* ac, int32_t num, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num; thread_idx += gridDim.x * blockDim.x){
        int32_t id = ids[thread_idx];
        if(id >= 0){
            atomicAdd(ac + (id%total_num_nodes), 1);
        }
    }
}  

class DirectMapCacheController : public CacheController{
public:
    DirectMapCacheController(){}
    
    virtual ~DirectMapCacheController(){}
    
    void Initialize(
        int32_t dev_id, 
        int32_t capacity, 
        int32_t sampled_num, 
        int32_t total_num_nodes,
        int32_t batch_size) override 
    {
        // device_idx_ = dev_id;
        // capacity_ = capacity;
        // candidates_num_ = sampled_num + capacity;
        // total_num_nodes_ = total_num_nodes;
        // cudaSetDevice(dev_id);
        // cudaCheckError();
        // cudaMalloc(&cache_map_, total_num_nodes * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&candidates_map_, total_num_nodes * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&candidate_fifo_, (candidates_num_) * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&cache_fifo_, capacity * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&final_candidates_, capacity * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&current_key_, (candidates_num_) * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&is_recent_, (candidates_num_) * sizeof(int32_t));
        // cudaMemset(is_recent_, 0, candidates_num_ * sizeof(int32_t));
        // cudaCheckError();
        // cudaMalloc(&global_counter_, 3 * sizeof(int32_t));
        // cudaCheckError();
        // dim3 block_num(80, 1);
        // dim3 thread_num(1024, 1);
        // Init_Int32<<<block_num, thread_num>>>(cache_map_, total_num_nodes, -1);
        // cudaCheckError();
        // Init_Int32<<<block_num, thread_num>>>(candidate_fifo_, (candidates_num_), -1);
        // cudaCheckError();
        // all_cached_ids_ = all_ids;
        // Init_Int32<<<block_num, thread_num>>>(all_cached_ids_, capacity, -1);
        // cudaCheckError();
        // cudaMalloc(&access_time_, total_num_nodes * sizeof(int32_t));
        // cudaMemset(access_time_, 0, total_num_nodes * sizeof(int32_t));

        // std::cout<<"candidates "<<candidates_num_<<"\n";
        // std::cout<<"capacity "<<capacity_<<"\n";
        // std::cout<<"id cache initialize on "<<dev_id<<"\n";

        // iter_ = 0;
    }

    void Finalize(int32_t dev_id) override {}

    void Find(
        int32_t* sampled_ids, 
        int32_t* cache_offset, 
        int32_t* node_counter, 
        int32_t op_id,
        void* stream) override 
    {
        // dim3 block_num(64, 1);
        // dim3 thread_num(1024, 1);
        // Find_Kernel<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(d_query, d_result, num_queries, total_num_nodes_, cache_map_);
        // cudaCheckError();
    }
    
    void MakePlan(        
        int32_t* candidates, 
        float* importance, 
        int32_t* node_counter, 
        int64_t** csr_node_index,
        int32_t** csr_dst_node_ids, 
        char* partition_index,
        int32_t* parition_offset,
        void* stream) override 
    {
        // dim3 block_num(64, 1);
        // dim3 thread_num(1024, 1);
        // Init_Int32<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(candidate_fifo_, num_candidates, -1);
        // cudaCheckError();
        // cudaMemsetAsync(candidates_map_, 0, total_num_nodes_ * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        // cudaCheckError();
        // cudaMemsetAsync(global_counter_, 0, 3 * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        // cudaCheckError();
        // for(int32_t mode_id = 0; mode_id < 3; mode_id++){
        //     get_current_key<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(candidates, current_key_, importance, num_candidates, mode_id, is_recent_);
        //     cudaCheckError();
        //     filtering<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(candidate_fifo_, candidates, candidates_map_, global_counter_, num_candidates);
        //     cudaCheckError();
        // }
        // remove_out_cache_candidates<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(capacity_, num_candidates, candidates_map_, candidate_fifo_);
        // cudaCheckError();
        // get_empty_position<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(all_cached_ids_, cache_fifo_, candidates_map_, cache_map_, capacity_, global_counter_);
        // cudaCheckError();
        // final_filtering<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(candidate_fifo_, global_counter_, cache_map_, final_candidates_, capacity_);
        // cudaCheckError();
        // replace<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(global_counter_, cache_fifo_, final_candidates_, cache_map_, all_cached_ids_, 0);
        // cudaCheckError();
        // // if(device_idx_ == 1){
        // //     int32_t* h_counter = (int32_t*)malloc(3 * sizeof(int32_t));
        // //     cudaMemcpy(h_counter, global_counter_, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        // //     for(int32_t i = 0; i < 3; i++){
        // //         std::cout<<h_counter[i]<<" ";
        // //     }
        // //     std::cout<<"\n";
        // // }
        // iter_++;
    }

    void Update(
        int32_t* candidates_ids, 
        float* candidates_float_feature, 
        float* cache_float_feature,
        int32_t float_attr_len, 
        void* stream) override 
    {  

    }
    
    void AccessCount(
        int32_t* d_key,
        int32_t num_keys, 
        void* stream) override {
        dim3 block_num(64, 1);
        dim3 thread_num(1024, 1);
        access_count<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(d_key, access_time_, num_keys, total_num_nodes_);
    }

    int32_t* FutureBatch() override 
    {
        return future_ids_;
    }

    int32_t Capacity() override 
    {
        return capacity_;
    }

    int32_t* AllCachedIds() override 
    {
        return all_cached_ids_;
    }

    int32_t* RecentMark() override {
        return is_recent_;
    }

private:
    int32_t device_idx_;
    int32_t capacity_;
    int32_t candidates_num_;
    int32_t total_num_nodes_;

    int32_t* all_cached_ids_;
    int32_t* candidate_fifo_;
    int32_t* cache_fifo_;
    int32_t* final_candidates_;

    int32_t* global_counter_;
    int32_t* current_key_;
    int32_t* cache_map_;
    int32_t* candidates_map_;

    int32_t* is_recent_;

    int32_t iter_;
    int32_t* future_ids_;

    int32_t* access_time_;
};

CacheController* NewDirectMapCacheController()
{
    return new DirectMapCacheController();
} 

__device__ int32_t SetHash(int32_t id, int32_t set_num){
    return id % set_num;
}

__global__ void CacheHitTimes(int32_t* raw_candidates, int32_t* node_counter, int32_t* cache_map){
    __shared__ int32_t count[1];
    if(threadIdx.x == 0){
        count[0] = 0;
    }
    __syncthreads();
    int32_t num_candidates = node_counter[9];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = raw_candidates[thread_idx];
        if(cid >= 0){
            int32_t cache_offset = cache_map[cid];
            if(cache_offset >= 0){
                atomicAdd(count, 1);
            }
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(node_counter + 10, count[0]);
    }
}

__global__ void CacheHitRate(int32_t* node_counter, int32_t dev_id){
    printf("%d: %f\n", dev_id, (node_counter[10]*1.0/node_counter[9]));
}

__global__ void FilterCandidate(int32_t* raw_candidates, int32_t* filtered_candidates,
                                int32_t* filtered_index, int32_t* node_counter, 
                                int32_t* candidates_map, int32_t* cache_map, int32_t* global_counter)
{
    __shared__ int32_t local_fifo[1024];
    __shared__ int32_t local_index[1024];
    __shared__ int32_t offset[2];//0 : local offset, 1 : global offset
    int32_t num_candidates = node_counter[9];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        if(threadIdx.x == 0){
            offset[0] = 0;
            offset[1] = 0;
        }
        local_fifo[threadIdx.x] = 0;
        __syncthreads();
        int32_t cid = raw_candidates[thread_idx];
        if(cid >= 0){
            int32_t cache_hit = cache_map[cid];
            if(cache_hit < 0){
                int32_t repeat = atomicAdd(candidates_map + cid, 1);
                if(repeat == 0){
                    int32_t local_offset = atomicAdd(offset, 1);
                    local_fifo[local_offset] = cid;
                    local_index[local_offset] = thread_idx;
                }
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            offset[1] = atomicAdd(global_counter, offset[0]);
        }
        __syncthreads();
        if(threadIdx.x < offset[0]){
            int32_t global_offset = offset[1];
            filtered_candidates[(global_offset + threadIdx.x)%num_candidates] = local_fifo[threadIdx.x];
            filtered_index[(global_offset + threadIdx.x)%num_candidates] = local_index[threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void SelectSet(int32_t* candidate, int32_t* global_counter,
                          int32_t* set_fifo, int32_t set_num,
                          int32_t* cache_map,
                          int32_t* set_conflict_map, int32_t* set_pos_map)
{
    __shared__ int32_t local_fifo[1024];
    __shared__ int32_t offset[2];//0 : local offset, 1 : global offset
    int32_t num_candidates = global_counter[0];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        if(threadIdx.x == 0){
            offset[0] = 0;
            offset[1] = 0;
        }
        local_fifo[threadIdx.x] = 0;
        __syncthreads();
        int32_t cid = candidate[thread_idx];
        if(cid >= 0){
            int32_t set_id = SetHash(cid, set_num);
            int32_t set_conflict = atomicAdd(set_conflict_map + set_id, 1);
            if(set_conflict == 0){
                int32_t local_offset = atomicAdd(offset, 1);
                local_fifo[local_offset] = set_id;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            offset[1] = atomicAdd(global_counter + 1, offset[0]);
        }
        __syncthreads();
        if(threadIdx.x < offset[0]){
            int32_t global_offset = offset[1];
            int32_t set_id = local_fifo[threadIdx.x];
            int32_t set_pos = (global_offset + threadIdx.x)%num_candidates;
            set_fifo[set_pos] = set_id;
            set_pos_map[set_id] = set_pos;
        }
        __syncthreads();
    }
}

__global__ void GatherCandidate(int32_t* filtered_candidate, int32_t* filtered_index,
                                int32_t* global_counter, int32_t* set_conflict_map,
                                int32_t* final_candidates, int32_t* final_index, int32_t set_num, int32_t* set_pos_map)
{
    int32_t candidate_count = global_counter[0];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < candidate_count; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = filtered_candidate[thread_idx];
        int32_t iid = filtered_index[thread_idx];
        if(cid >= 0){
            int32_t set_id = SetHash(cid, set_num);
            int32_t set_pos = set_pos_map[set_id];
            int32_t set_conflict = atomicAdd(set_conflict_map + set_id, 1);
            final_candidates[(set_pos * TOL) + (set_conflict%TOL)] = cid;
            final_index[(set_pos * TOL) + (set_conflict%TOL)] = iid;
        }
    }
}

__global__ void GetCachedIds(int32_t* set_fifo, int32_t* all_cached_ids, int32_t* current_cache_ids, int32_t* global_counter, int32_t way_num){
    int32_t set_num = global_counter[1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < set_num * way_num; thread_idx += gridDim.x * blockDim.x){
        int32_t set_id  = set_fifo[thread_idx/way_num];
        int32_t cached_id = all_cached_ids[(set_id * way_num) + (thread_idx % way_num)];
        current_cache_ids[thread_idx] = cached_id;
    }
}

__global__ void FutureBatchAccess1Hop(
    int32_t* future_ids,
    int32_t* future_access_map,
    int32_t batch_size,
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
    char* partition_index,
	int32_t* parition_offset)
{
    for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * NCOUNT_1; idx += gridDim.x * blockDim.x){
		int32_t batch_idx = idx / NCOUNT_1;
		int32_t src_id = future_ids[batch_idx];
		if(src_id < 0){
			continue;
		}
        int32_t neighbor_offset = idx % NCOUNT_1;
        thrust::minstd_rand engine;
		engine.discard(idx);
		int32_t part_id = partition_index[src_id];
		int32_t part_offset = parition_offset[src_id];
		int64_t start_index = csr_node_index[part_id][part_offset];
		int32_t col_size = csr_node_index[part_id][(part_offset + 1)] - start_index;
        if(neighbor_offset < col_size){
            thrust::uniform_int_distribution<> dist(0, col_size - 1);
			int32_t dst_index = dist(engine);
            int32_t dst_id = csr_dst_node_ids[part_id][(int64_t(start_index + int64_t(dst_index)))];
            atomicAdd(future_access_map + dst_id, 1);
        }
    }
}

__global__ void FutureBatchAccess2Hop(
    int32_t* future_ids,
    int32_t* future_access_map,
    int32_t batch_size,
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
    char* partition_index,
	int32_t* parition_offset)
{
    for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * NCOUNT_1; idx += gridDim.x * blockDim.x){
		int32_t batch_idx = idx / NCOUNT_1;
		int32_t src_id = future_ids[batch_idx];
		if(src_id < 0){
			continue;
		}
        int32_t neighbor_offset = idx % NCOUNT_1;
        thrust::minstd_rand engine_hop1;
		engine_hop1.discard(idx);
		int32_t part_id_hop1 = partition_index[src_id];
		int32_t part_offset_hop1 = parition_offset[src_id];
		int64_t start_index_hop1 = csr_node_index[part_id_hop1][part_offset_hop1];
		int32_t col_size_hop1 = csr_node_index[part_id_hop1][(part_offset_hop1 + 1)] - start_index_hop1;
        if(neighbor_offset < col_size_hop1){
            thrust::uniform_int_distribution<> dist_hop1(0, col_size_hop1 - 1);
			int32_t dst_index_hop1 = dist_hop1(engine_hop1);
            int32_t dst_id_hop1 = csr_dst_node_ids[part_id_hop1][(int64_t(start_index_hop1 + int64_t(dst_index_hop1)))];
            if(dst_id_hop1 >= 0){
                atomicAdd(future_access_map + dst_id_hop1, 1);
                int32_t part_id_hop2 = partition_index[dst_id_hop1];
                int32_t part_offset_hop2 = parition_offset[dst_id_hop1];
                int64_t start_index_hop2 = csr_node_index[part_id_hop2][part_offset_hop2];
                int32_t col_size_hop2 = csr_node_index[part_id_hop2][(part_offset_hop2 + 1)] - start_index_hop2;
                thrust::uniform_int_distribution<> dist_hop2(0, col_size_hop2 - 1);
                thrust::minstd_rand engine_hop2;
                engine_hop2.discard(idx);

                for(int32_t i = 0; (i < NCOUNT_2) && (i < col_size_hop2); i++){
                    int32_t dst_index_hop2 = dist_hop2(engine_hop2);
                    int32_t dst_id_hop2 = csr_dst_node_ids[part_id_hop2][(int64_t(start_index_hop2 + int64_t(dst_index_hop2)))];
                    if(dst_id_hop2 >= 0){
                        atomicAdd(future_access_map + dst_id_hop2, 1);
                    }
                }
            }
        }
    }
}

__global__ void ComputeImportance(
    int32_t* vertex_ids, 
	float* vertex_importance, 
    int32_t* global_counter,
    int32_t expand,
    int32_t* future_access_map,
    int32_t is_candidate)
{
    int32_t num_vertex = global_counter[1] * expand;
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_vertex; thread_idx += gridDim.x * blockDim.x){
        int32_t vid = vertex_ids[thread_idx];
        if(vid < 0){
            float imp = -1;
            vertex_importance[thread_idx] = imp;
        }else{
            // float imp = future_access_map[vid] * 1.0;
            // vertex_importance[thread_idx] = imp;
            if(is_candidate){
                vertex_importance[thread_idx] = 1.0;
            }else{
                vertex_importance[thread_idx] = 0;
            }
        }
    }

}

/*one warp deal with one set*/
__global__ void MakePlanKernel(float* cachedids_importance, float* candidates_importance, 
                            int32_t* set_fifo, int32_t* set_conflict_map, int32_t way_num,
                            int32_t* global_counter,
                            int32_t* update_plan)
{
    __shared__ float sh_cached_importance[1024];
    __shared__ float sh_cand_importance[1024];
    __shared__ float sh_min_importance[1024];
    __shared__ int32_t sh_min_index[1024];
    __shared__ int32_t sh_way_updated[1024];

    int32_t lane_id = threadIdx.x & 31;
    int32_t global_warp_id;
    int32_t warp_size = 32;
    int32_t local_warp_id = threadIdx.x >> 5;
    int32_t set_per_block = blockDim.x / warp_size;
    int32_t set_num = global_counter[1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < set_num * warp_size; thread_idx += gridDim.x * blockDim.x){
        global_warp_id = thread_idx >> 5;
        int32_t block_start_set = blockIdx.x * set_per_block;
        if(threadIdx.x < set_per_block * way_num){
            sh_cached_importance[threadIdx.x] = cachedids_importance[block_start_set * way_num + threadIdx.x];
        } 
        if(threadIdx.x < set_per_block * TOL){
            sh_cand_importance[threadIdx.x] = candidates_importance[(block_start_set * TOL + threadIdx.x)%(set_num*TOL)];
        }
        __syncthreads();
        int32_t set_id = set_fifo[global_warp_id];
        int32_t set_conflict = set_conflict_map[set_id];
        sh_way_updated[threadIdx.x] = -1;
        for(int32_t i = 0; i < set_conflict && i < TOL; i++){
            sh_min_importance[threadIdx.x] = sh_cached_importance[local_warp_id * way_num + lane_id];
            sh_min_index[threadIdx.x] = lane_id;
            float former = 0;
            float latter = 0;
            for(int32_t lane_mask = way_num / 2; lane_mask >= 1; lane_mask = lane_mask / 2){
                if(lane_id < lane_mask){
                    former = sh_min_importance[threadIdx.x];
                    latter = sh_min_importance[threadIdx.x + lane_mask];
                    sh_min_importance[threadIdx.x] = (former < latter) ? former : latter;
                    sh_min_index[threadIdx.x] = (former < latter) ? sh_min_index[threadIdx.x] : sh_min_index[threadIdx.x + lane_mask];
                }
            }
            if(lane_id == 0){
                float current_min = sh_min_importance[threadIdx.x];
                int32_t target_way = -1;
                if(current_min <= sh_cand_importance[local_warp_id * TOL + i]){
                    target_way = sh_min_index[threadIdx.x];
                    int32_t is_updated = sh_way_updated[threadIdx.x + target_way];
                    if(is_updated >= 0){
                        update_plan[global_warp_id * TOL + is_updated] = -1;
                    }
                    sh_way_updated[threadIdx.x + target_way] = i;
                    sh_cached_importance[local_warp_id * way_num + target_way] = sh_cand_importance[local_warp_id * TOL + i];
                    update_plan[global_warp_id * TOL + i] = target_way;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void UpdateIdKernel(int32_t* final_candidates,
                                int32_t* final_index, 
                                int32_t* set_fifo,
                                int32_t* update_plan,
                                int32_t way_num,
                                int32_t* candidates_ids, 
                                int32_t* all_cache_ids,
                                int32_t* cache_map,
                                int32_t* global_counter)
{
    int32_t set_num = global_counter[1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < set_num * TOL; thread_idx += gridDim.x * blockDim.x){
        int32_t target_way = update_plan[thread_idx];
        if(target_way >= 0){
            int32_t index = final_index[thread_idx];
            int32_t cid = candidates_ids[index];
            int32_t set_id = set_fifo[thread_idx / TOL];
            int32_t oid = all_cache_ids[(set_id * way_num + target_way)];
            all_cache_ids[(set_id * way_num + target_way)] = cid;
            cache_map[cid] = set_id * way_num + target_way;
            if(oid >= 0){
                cache_map[oid] = -1;
            }
        }
    }
}

__global__ void UpdateFeatureKernel(int32_t* final_index, 
                                    int32_t* set_fifo,
                                    int32_t* update_plan,
                                    int32_t way_num,
                                    float* candidates_float_feature, 
                                    float* cache_float_feature,
                                    int32_t float_attr_len,
                                    int32_t* global_counter)
{
    int32_t set_num = global_counter[1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < set_num * TOL * float_attr_len; thread_idx += gridDim.x * blockDim.x){
        int32_t target_way = update_plan[thread_idx/float_attr_len];
        if(target_way >= 0){
            int32_t index = final_index[thread_idx / float_attr_len];
            int32_t offset = thread_idx % float_attr_len;
            int32_t set_id = set_fifo[(thread_idx / float_attr_len) / TOL];
            cache_float_feature[int64_t(int64_t(int64_t(int64_t(int64_t(set_id) * way_num) + target_way) * float_attr_len) + offset)] = candidates_float_feature[int64_t(int64_t(int64_t(index) * float_attr_len) + offset)];
        }
    }
}

class SetAssociateCacheController : public CacheController {
public:
    SetAssociateCacheController(int32_t way_num, int32_t K_Batch){
        way_num_ = way_num;
        k_batch_ = K_Batch;
    }
    virtual ~SetAssociateCacheController(){}

    void Initialize(
        int32_t dev_id, 
        int32_t capacity, 
        int32_t sampled_num, 
        int32_t total_num_nodes,
        int32_t batch_size) override
    {
        device_idx_ = dev_id;
        capacity_ = capacity;
        total_num_nodes_ = total_num_nodes;
        set_num_ = capacity / way_num_;//ensure being divisible
        sampled_num_ = sampled_num;
        batch_size_ = batch_size;
        dim3 block_num(80, 1);
        dim3 thread_num(1024, 1);
        cudaSetDevice(dev_id);
        cudaMalloc(&cache_map_, total_num_nodes * sizeof(int32_t));//may overflow when meet 1B nodes
        cudaMalloc(&candidates_map_, total_num_nodes * sizeof(int32_t));
        cudaMalloc(&future_access_map_, total_num_nodes * sizeof(int32_t));
        Init_Int32<<<block_num, thread_num>>>(cache_map_, total_num_nodes, -1);

        cudaMalloc(&filtered_candidates_, sampled_num * sizeof(int32_t));
        cudaMalloc(&final_candidates_, sampled_num * TOL * sizeof(int32_t));
        cudaMalloc(&candidates_importance_, sampled_num * TOL * sizeof(float));
        cudaMalloc(&filtered_index_, sampled_num * sizeof(int32_t));
        cudaMalloc(&final_index_, sampled_num * TOL * sizeof(int32_t));

        cudaMalloc(&global_counter_, 4 * sizeof(int32_t));

        cudaMalloc(&set_fifo_, sampled_num * sizeof(int32_t));
        cudaMalloc(&set_conflict_map_, set_num_ * sizeof(int32_t));
        cudaMalloc(&set_pos_map_, set_num_ * sizeof(int32_t));

        cudaMalloc(&current_cache_ids_, sampled_num * way_num_ * sizeof(int32_t));
        cudaMalloc(&cachedids_importance_, sampled_num * way_num_ * sizeof(float));

        cudaMalloc(&update_plan_, sampled_num * TOL * sizeof(int32_t));// candidate sets

        cudaMalloc(&all_cached_ids_, capacity * sizeof(int32_t));
        Init_Int32<<<block_num, thread_num>>>(all_cached_ids_, capacity, -1);

        cudaMalloc(&future_ids_, batch_size_ * k_batch_ * sizeof(int32_t));
        
        // cudaMalloc(&access_time_, total_num_nodes * sizeof(int32_t));
        // cudaMemset(access_time_, 0, total_num_nodes * sizeof(int32_t));
        iter_ = 0;

        // std::ifstream infile;
        // infile.open("/home/sunjie/dataset/paper100M/paper_presc1_cache.txt");
        // if(!infile.is_open()){
        //     std::cout<<"Error: can not open in file\n";
        //     return;
        // }
        // std::istringstream iss;
        // std::string buff;
        // int32_t count = 0;
        // int32_t cache_id;
        // std::vector<int> h_cache_ids;
        // std::vector<int> h_cache_map;
        // for(int32_t i = 0; i < total_num_nodes; i++){
        //     h_cache_map.push_back(-1);
        // }
        // while(getline(infile, buff)){
        //     if(count >= capacity){
        //         break;
        //     }
        //     iss.clear();
        //     iss.str(buff);
        //     iss >> cache_id;
        //     h_cache_ids.push_back(cache_id);
        //     h_cache_map[cache_id] = count;
        //     count++;
        // }

        // cudaMemcpy(cache_map_, &h_cache_map[0], total_num_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
        // cudaMemcpy(all_cached_ids_, &h_cache_ids[0], capacity * sizeof(int32_t), cudaMemcpyHostToDevice);
        // std::cout<<"cache controller initialized on "<<device_idx_<<"\n";
        

    }  

    void Finalize(int32_t dev_id) override {
        cudaSetDevice(dev_id);
        cudaFree(cache_map_);
        cudaFree(candidates_map_);
        cudaFree(future_access_map_);
        cudaFree(filtered_candidates_);
        cudaFree(final_candidates_);
        cudaFree(candidates_importance_);
        cudaFree(filtered_index_);
        cudaFree(final_index_);
        cudaFree(global_counter_);
        cudaFree(set_fifo_);
        cudaFree(set_conflict_map_);
        cudaFree(set_pos_map_);
        cudaFree(current_cache_ids_);
        cudaFree(cachedids_importance_);
        cudaFree(update_plan_);
        cudaFree(all_cached_ids_);
        cudaFree(future_ids_);
        cudaCheckError();
    }

    void Find(
        int32_t* sampled_ids, 
        int32_t* cache_offset, 
        int32_t* node_counter, 
        int32_t op_id,
        void* stream) override 
    {
        dim3 block_num(64, 1);
        dim3 thread_num(1024, 1);
        Find_Kernel<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(sampled_ids, cache_offset, node_counter, total_num_nodes_, op_id, cache_map_);
        cudaCheckError();
    }

    void MakePlan(int32_t* candidates, 
                 float* importance, 
                 int32_t* node_counter, 
                 int64_t** csr_node_index,
                 int32_t** csr_dst_node_ids, 
                 char* partition_index,
                 int32_t* parition_offset,
                 void* stream) override
    {
        dim3 block_num(64, 1);
        dim3 thread_num(1024, 1);
        
        CacheHitTimes<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(candidates, node_counter, cache_map_);
        int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
        cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
        std::cout<<device_idx_<<" "<<h_node_counter[10]*1.0/h_node_counter[9]<<"\n";

        cudaMemsetAsync(global_counter_, 0, 4 * sizeof(int32_t) , static_cast<cudaStream_t>(stream));
        cudaCheckError();
        cudaMemsetAsync(candidates_map_, 0, total_num_nodes_ * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        cudaCheckError();
        FilterCandidate<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(candidates, filtered_candidates_, filtered_index_, node_counter, candidates_map_, cache_map_, global_counter_);
        cudaCheckError();
        cudaMemsetAsync(set_conflict_map_, 0, set_num_ * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        cudaCheckError();
        cudaMemsetAsync(set_pos_map_, 0, set_num_ * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        cudaCheckError();
        SelectSet<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(filtered_candidates_, global_counter_, set_fifo_, set_num_, cache_map_, set_conflict_map_, set_pos_map_);
        cudaCheckError();
        cudaMemsetAsync(set_conflict_map_, 0, set_num_ * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        cudaCheckError();
        Init_Int32<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(final_candidates_, sampled_num_*TOL, -1);
        cudaCheckError();
        GatherCandidate<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(filtered_candidates_, filtered_index_, global_counter_, set_conflict_map_, final_candidates_, final_index_, set_num_, set_pos_map_);
        cudaCheckError();
        GetCachedIds<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(set_fifo_, all_cached_ids_, current_cache_ids_, global_counter_, way_num_);
        cudaCheckError();
        // cudaMemsetAsync(future_access_map_, 0, total_num_nodes_ * sizeof(int32_t), static_cast<cudaStream_t>(stream));
        // cudaCheckError();
        // FutureBatchAccess2Hop<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(future_ids_, future_access_map_, batch_size_ * k_batch_, csr_node_index, csr_dst_node_ids, partition_index, parition_offset);
        // cudaCheckError();
        ComputeImportance<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(final_candidates_, candidates_importance_, global_counter_, TOL, future_access_map_, 1);
        cudaCheckError();
        ComputeImportance<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(current_cache_ids_, cachedids_importance_, global_counter_, way_num_, future_access_map_, 0);
        cudaCheckError();
        Init_Int32<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(update_plan_, sampled_num_ * TOL, -1);
        cudaCheckError();
        MakePlanKernel<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(cachedids_importance_, candidates_importance_, set_fifo_, set_conflict_map_, way_num_, global_counter_, update_plan_);
        cudaCheckError();
    }
    /*num candidates = sampled num*/
    void Update(
        int32_t* candidates_ids, 
        float* candidates_float_feature, 
        float* cache_float_feature,
        int32_t float_attr_len, 
        void* stream) override 
    {  
        dim3 block_num(64, 1);
        dim3 thread_num(1024, 1);
        UpdateIdKernel<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(final_candidates_, final_index_, set_fifo_, update_plan_, way_num_, candidates_ids, all_cached_ids_, cache_map_, global_counter_);
        cudaCheckError();
        UpdateFeatureKernel<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(final_index_, set_fifo_, update_plan_, way_num_, candidates_float_feature, cache_float_feature, float_attr_len, global_counter_);
        cudaCheckError();
    }

    void AccessCount(
        int32_t* d_key, 
        int32_t num_keys, 
        void* stream) override 
    {
        dim3 block_num(64, 1);
        dim3 thread_num(1024, 1);
        access_count<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(d_key, access_time_, num_keys, total_num_nodes_);
        // access_count<<<block_num, thread_num>>>(d_key, access_time_, num_keys, total_num_nodes_);
        iter_++; 
        if(device_idx_ == 0){
            if(iter_ == (78124*3)){
                std::cout<<"start out put access count\n";
                std::ofstream ofs;
                ofs.open("paper_access_time_epoch1.txt");
                int* h_access_count = (int*)malloc(total_num_nodes_*sizeof(int));
                cudaMemcpy(h_access_count, access_time_, total_num_nodes_ * sizeof(int), cudaMemcpyDeviceToHost);
                for(int32_t i = 0; i < total_num_nodes_; i++){
                    ofs<<h_access_count[i]<<"\n";
                }
            }
        }
    }

    int32_t* FutureBatch() override 
    {
        return future_ids_;
    }

    int32_t Capacity() override 
    {
        return capacity_;
    }

    int32_t* AllCachedIds() override 
    {
        return all_cached_ids_;
    }

    int32_t* RecentMark() override {
        return is_recent_;
    }

private:
    int32_t device_idx_;
    int32_t capacity_;
    int32_t total_num_nodes_;
    int32_t set_num_;
    int32_t way_num_;
    int32_t sampled_num_;
    int32_t k_batch_;
    int32_t batch_size_;

    int32_t* all_cached_ids_;
    int32_t* current_cache_ids_;
    int32_t* candidate_fifo_;
    int32_t* set_fifo_;
    int32_t* filtered_candidates_;
    int32_t* final_candidates_;
    int32_t* filtered_index_;
    int32_t* final_index_;

    int32_t* global_counter_;
    int32_t* cache_map_;
    int32_t* candidates_map_;
    int32_t* set_pos_map_;
    int32_t* set_conflict_map_;

    int32_t* is_recent_;

    int32_t* future_ids_;
    int32_t* future_access_map_;

    int32_t* update_plan_;

    float* cachedids_importance_;
    float* candidates_importance_;

    int32_t* access_time_;
    int32_t iter_;
};

CacheController* NewSetAssociateCacheController(int32_t way_num, int32_t K_Batch)
{
    return new SetAssociateCacheController(way_num, K_Batch);
} 

void GPUCache::Initialize(
    std::vector<int> device, 
    int32_t capacity, 
    int32_t int_attr_len, 
    int32_t float_attr_len, 
    int32_t K_batch, 
    int32_t way_num)
{
    dev_ids_.resize(TOTAL_DEV_NUM);
    cache_controller_.resize(TOTAL_DEV_NUM);

    cache_hit_.resize(TOTAL_DEV_NUM);

    k_batch_ = K_batch;
    way_num_ = way_num;

    iter_.resize(8);
    for(int32_t i = 0 ; i < TOTAL_DEV_NUM; i++){
        iter_[i] = 0;
    }

    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        dev_ids_[i] = false;
        cache_hit_[i] = 0;
    }

    if(int_attr_len > 0){
        int_feature_cache_.resize(TOTAL_DEV_NUM);
    }
    if(float_attr_len > 0){
        float_feature_cache_.resize(TOTAL_DEV_NUM);
    }
    cudaCheckError();

    d_accessed.resize(8);
    global_count.resize(8);

    for(int32_t i = 0; i < device.size(); i++){
        int32_t dev_id = device[i];
        dev_ids_[dev_id] = true;

        CacheController* cctl = NewSetAssociateCacheController(way_num_, k_batch_);
        cache_controller_[dev_id] = cctl;

        if(float_attr_len > 0){
            cudaSetDevice(dev_id);
            float* new_float_feature_cache;
            cudaMalloc(&new_float_feature_cache, int64_t(int64_t(int64_t(capacity) * float_attr_len) * sizeof(float)));
            float_feature_cache_[dev_id] = new_float_feature_cache;
        }
    }
    capacity_ = capacity;
    int_attr_len_ = int_attr_len;
    float_attr_len_ = float_attr_len;
}

void GPUCache::InitializeCacheController(
    int32_t dev_id, 
    int32_t capacity, 
    int32_t sampled_num, 
    int32_t total_num_nodes,
    int32_t batch_size)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->Initialize(dev_id, capacity, sampled_num, total_num_nodes, batch_size);
    }else{
        std::cout<<"invalid device for cache\n"; 
    }
}

void GPUCache::Finalize(){
    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        if(dev_ids_[i] == true){
            cudaSetDevice(i);
            cache_controller_[i]->Finalize(i);
            cudaFree(float_feature_cache_[i]);
        }
    }
}

int32_t GPUCache::Capacity(){
    return capacity_;
}

void GPUCache::Find(
    int32_t* sampled_ids, 
    int32_t* cache_offset, 
    int32_t* node_counter, 
    int32_t op_id,
    void* stream,
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->Find(sampled_ids, cache_offset, node_counter, op_id, stream);
    }else{
        std::cout<<"invalid device for cache\n"; 
    }
}

void GPUCache::MakePlan(
    int32_t* candidates, 
    float* importance, 
    int32_t* node_counter, 
    int64_t** csr_node_index,
    int32_t** csr_dst_node_ids, 
    char* partition_index,
    int32_t* parition_offset,
    void* stream,
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->MakePlan(candidates, importance, node_counter, csr_node_index, csr_dst_node_ids, partition_index, parition_offset, stream);
    }else{
        std::cout<<"invalid device for cache\n"; 
    }
}

void GPUCache::Update(
    int32_t* candidates_ids, 
    float* candidates_float_feature, 
    float* cache_float_feature,
    int32_t float_attr_len, 
    void* stream,
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->Update(candidates_ids, candidates_float_feature, cache_float_feature, float_attr_len, stream);
    }else{
        std::cout<<"invalid device for cache\n"; 
    }
}

void GPUCache::AccessCount(
    int32_t* d_key, 
    int32_t num_keys, 
    void* stream, 
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->AccessCount(d_key, num_keys, stream);
    }else{
        std::cout<<"invalid device for cache\n"; 
    }
}

int32_t* GPUCache::FutureBatch(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->FutureBatch();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int32_t* GPUCache::AllCachedIds(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->AllCachedIds();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int32_t* GPUCache::RecentMark(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->RecentMark();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

float* GPUCache::Float_Feature_Cache(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return float_feature_cache_[dev_id];
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int64_t* GPUCache::Int_Feature_Cache(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return int_feature_cache_[dev_id];
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int32_t GPUCache::K_Batch()
{
    return k_batch_;
}

int64_t GPUCache::GetCacheHit(int32_t dev_id)
{
    return cache_hit_[dev_id];
}

void GPUCache::SetCacheHit(int32_t dev_id, int32_t hit_times)
{
    cache_hit_[dev_id] = int64_t(cache_hit_[dev_id] + int64_t(hit_times));
}

void GPUCache::ResetCacheHit(int32_t dev_id)
{
    cache_hit_[dev_id] = 0;
}
