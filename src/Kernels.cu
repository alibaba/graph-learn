#include "Kernels.cuh"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include "GPUCache.cuh"
#include <random>

#define TRAINMODE 0
#define VALIDMODE 1
#define TESTMODE  2

extern "C"
void* d_alloc_space(int64_t num_bytes) {
    void *ret;
	cudaMalloc(&ret, num_bytes);
	cudaCheckError();
    return ret;
}

extern "C"
void* d_alloc_space_managed(unsigned int num_bytes) {
    void *ret;
	cudaMallocManaged(&ret, num_bytes);
	cudaCheckError();
    return ret;
}

extern "C"
void d_copy_2_h(void* h_ptr, void* d_ptr, unsigned int num_bytes){
	cudaMemcpy(h_ptr, d_ptr, num_bytes, cudaMemcpyDeviceToHost);
	cudaCheckError();
}


extern "C"
void SetGPUDevice(int32_t shard_id){
	cudaSetDevice(shard_id);
	cudaCheckError();
}

extern "C"
int32_t GetGPUDevice(){
	int32_t dev_id;
	cudaGetDevice(&dev_id);
	return dev_id;
}

extern "C"
void d_free_space(void* d_ptr){
	cudaFree(d_ptr);
}


extern "C"
void* host_alloc_space(unsigned int num_bytes) {
    void* host_ptr;
	void* ret;
	cudaHostAlloc(&host_ptr, num_bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&ret, host_ptr, 0);
	cudaCheckError();
    return ret;
}

//assume no duplicate
__global__ void batch_generator(
	int32_t* batch_ids, 
	int32_t* labels, 
	int32_t batch_size, 
	int32_t counter, 
	int32_t* all_ids, 
	int32_t* all_labels, 
	int32_t total_cap,
	int32_t* accessed_map,
	int32_t* position_map)
{
	int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < batch_size){
		if((batch_size * counter + idx) >= total_cap){
			batch_ids[idx] = -1;
			labels[idx] = -1;
		}else{
			int32_t src_id = all_ids[(batch_size * counter + idx)%(total_cap)];
			batch_ids[idx] = src_id;
			accessed_map[src_id] = 1;
			position_map[src_id] = idx;
			labels[idx] = all_labels[(batch_size * counter + idx)%(total_cap)];
		}	
	}
}

__global__ void future_batch_generator(
	int32_t* future_batch_ids, 
	int32_t batch_size, 
	int32_t k_batch, 
	int32_t counter, 
	int32_t total_cap, 
	int32_t* all_ids)
{
	int32_t idx;
	for(idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * k_batch; idx += blockDim.x * gridDim.x){
		future_batch_ids[idx] = all_ids[(batch_size * (counter + 1) + idx)%(total_cap)];
	}
}

__global__ void update_counter(
	int32_t* node_counter, 
	int32_t* edge_counter, 
	int32_t op_id, 
	int32_t size)
{
	if(op_id == 1){
		node_counter[0] = size;//global offset for output
		node_counter[1] = 0;//for next op count
		node_counter[2] = size;//for next op input size
		node_counter[3] = 0;//global offset for feature extracting
		node_counter[4] = size;//feature extracting size
		edge_counter[0] = 0;//global offset for output
		edge_counter[1] = 0;//for next op count
		edge_counter[2] = 0;
		edge_counter[3] = 0;//agg edge count
	}else if(op_id == 2){
		node_counter[0] += node_counter[1];//global offset for output
		node_counter[5] = node_counter[3] + node_counter[4];//global offset for feature extracting
		node_counter[6] = node_counter[1];//feature extracting size
		node_counter[1] = 0;//for next op count
		node_counter[2] = edge_counter[1];
		edge_counter[3] += edge_counter[1];//agg edge count
		edge_counter[2] = edge_counter[0];
		edge_counter[0] += edge_counter[1];//global offset for output
		edge_counter[1] = 0;//for next op count
	}else if(op_id == 3){
		node_counter[0] += node_counter[1];//global offset for output
		node_counter[7] = node_counter[5] + node_counter[6];//global offset for feature extracting
		node_counter[8] = node_counter[1];//feature extracting size
		node_counter[9] = node_counter[7] + node_counter[8];
		node_counter[1] = 0;//for next op count
		node_counter[2] = edge_counter[1];
		edge_counter[4] = edge_counter[3] + edge_counter[1];//agg edge count
		edge_counter[2] = edge_counter[0];
		edge_counter[0] += edge_counter[1];//global offset for output
		edge_counter[1] = 0;//for next op count
	}
}

__global__ void Init_Array_Int32(
    int32_t* array, 
    int32_t length, 
    int32_t value)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < length; thread_idx += gridDim.x * blockDim.x){
        array[thread_idx] = value;
    }
}

extern "C"
void batch_generator_kernel(
	stream_handle strm_hdl, 
	GPUNodeStorage* noder,
	GPUCache* cache,
	GPUMemoryPool* memorypool,
	int32_t batch_size, 
	int32_t counter, 
	int32_t part_id,
	int32_t dev_id,
	int32_t mode)
{
	int32_t* all_ids = nullptr;
	int32_t* all_labels = nullptr;
	int32_t total_cap = 0;
	if(mode == TRAINMODE){
		all_ids = noder->GetTrainingSetIds(dev_id);
		all_labels = noder->GetTrainingLabels(dev_id);
		total_cap = noder->TrainingSetSize(dev_id);
	}else if(mode == VALIDMODE){
		all_ids = noder->GetValidationSetIds(dev_id);
		all_labels = noder->GetValidationLabels(dev_id);
		total_cap = noder->ValidationSetSize(dev_id);
	}else if(mode == TESTMODE){
		all_ids = noder->GetTestingSetIds(dev_id);
		all_labels = noder->GetTestingLabels(dev_id);
		total_cap = noder->TestingSetSize(dev_id);
	}

	int32_t total_node_num = noder->TotalNodeNum();

	int32_t k_batch = cache->K_Batch();
	int32_t* all_future_ids = cache->FutureBatch(dev_id);

	int32_t* batch_ids = memorypool->GetSampledIds();
	int32_t* labels = memorypool->GetLabels();
	int32_t* accessed_map = memorypool->GetAccessedMap();
	int32_t* position_map = memorypool->GetPositionMap();
	int32_t* node_counter = memorypool->GetNodeCounter();
	int32_t* edge_counter = memorypool->GetEdgeCounter();
	int32_t* agg_src_ids = memorypool->GetAggSrcId();
	int32_t* agg_dst_ids = memorypool->GetAggDstId();
	int32_t* agg_src_off = memorypool->GetAggSrcOf();
	int32_t* agg_dst_off = memorypool->GetAggDstOf();

	if(all_ids == nullptr){
		std::cout<<"invalid src id ptr\n";
		return;
	}
	if(all_labels == nullptr){
		std::cout<<"invalid label ptr\n";
		return;
	}

	cudaMemsetAsync(accessed_map, 0, int64_t(int64_t(total_node_num) * int64_t(sizeof(int32_t))), static_cast<cudaStream_t>(strm_hdl));
	cudaMemsetAsync(position_map, 0, int64_t(int64_t(total_node_num) * int64_t(sizeof(int32_t))), static_cast<cudaStream_t>(strm_hdl));

	cudaMemsetAsync(node_counter, 0, 16 * sizeof(int32_t), static_cast<cudaStream_t>(strm_hdl));
	cudaMemsetAsync(edge_counter, 0, 16 * sizeof(int32_t), static_cast<cudaStream_t>(strm_hdl));

	int32_t size = ((batch_size*(counter+1)) >= total_cap) ? (total_cap - batch_size * counter) : batch_size;
	dim3 bg_block((size - 1)/1024 + 1, 1);
	dim3 bg_thread(1024, 1);
	batch_generator<<<bg_block, bg_thread, 0, static_cast<cudaStream_t>(strm_hdl)>>>(batch_ids, labels, size, counter, all_ids, all_labels, total_cap, accessed_map, position_map);
	cudaCheckError();
	if(mode == TRAINMODE){
		future_batch_generator<<<bg_block, bg_thread, 0, static_cast<cudaStream_t>(strm_hdl)>>>(all_future_ids, batch_size, k_batch, counter, total_cap, all_ids);
		cudaCheckError();
	}
	update_counter<<<1, 1, 0, static_cast<cudaStream_t>(strm_hdl)>>>(node_counter, edge_counter, 1, size);
	cudaCheckError();
}


/////////random sampler//////////
__global__ void kernel_random_sampler(
	int32_t* d_src_ids, 
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
	char* partition_index,
	int32_t* parition_offset,
	int32_t batch_size, 
	int32_t count, 
	int32_t partition_count, 
	int32_t* GPU_dst_nodes_ids
	)
{
	int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < batch_size * count){
		thrust::minstd_rand engine;
		engine.discard(idx);
		int32_t batch_idx = idx / count;
		int32_t src_id = d_src_ids[batch_idx];
		if(src_id < 0){
			GPU_dst_nodes_ids[idx] = -1;
			return;
		}
		int32_t part_id = partition_index[src_id];
		int32_t part_offset = parition_offset[src_id];
		int64_t start_index = csr_node_index[part_id][part_offset];
		int32_t col_size = csr_node_index[part_id][(part_offset + 1)] - start_index;
		
		if(col_size > 0){
			thrust::uniform_int_distribution<> dist(0, col_size - 1);
			int32_t dst_index = dist(engine);
			GPU_dst_nodes_ids[idx] = csr_dst_node_ids[part_id][(int64_t(start_index + dst_index))];
		}else{
			GPU_dst_nodes_ids[idx] = -1;
		}
	}
}

/////////random sampler//////////
__global__ void kernel_random_sampler_bounded(
	int32_t* d_src_ids, 
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
	char* partition_index,
	int32_t* parition_offset,
	int32_t batch_size, 
	int32_t count, 
	int32_t partition_count, 
	int32_t* GPU_dst_nodes_ids
	)
{
	int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < batch_size * count){
		int32_t src_id = d_src_ids[idx/count];
		if(src_id >= 0){
			int32_t neighbor_offset = idx%count;
			int32_t part_id = partition_index[src_id];
			int32_t part_offset = parition_offset[src_id];
			int64_t start_index = csr_node_index[part_id][part_offset];
			int32_t col_size = csr_node_index[part_id][(part_offset + 1)] - start_index;
			if(neighbor_offset >= col_size){
				GPU_dst_nodes_ids[idx] = -1;
			}else{
				thrust::minstd_rand engine;
				engine.discard(idx);
				thrust::uniform_int_distribution<> dist(0, col_size - 1);
				int32_t dst_index = dist(engine);
				GPU_dst_nodes_ids[idx] = csr_dst_node_ids[part_id][(int64_t(start_index + dst_index))];
			}
		}else{
			GPU_dst_nodes_ids[idx] = -1;
		}
	}
}


/////////random sampler//////////
__global__ void kernel_random_sampler_optimized(
	int32_t*  sampled_ids,
	int32_t   op_id,
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
	char*     partition_index,
	int32_t*  parition_offset,
	int32_t   count, 
	int32_t       partition_count, 
	int32_t*  agg_src_ids,
	int32_t*  agg_dst_ids,
	int32_t*  accessed_map,
	int32_t*  position_map,
	int32_t*  node_counter,
	int32_t*  edge_counter
	)
{	
	/*the direction for agg is reversed*/
	__shared__ int32_t sh_agg_src_ids[1024];
	__shared__ int32_t sh_agg_dst_ids[1024];
	__shared__ int32_t sh_sampled_id[1024];
	/*local offset: 0, local node offset; 1, global node offset; 2, local edge offset; 3, global edge offset */
	__shared__ int32_t local_offset[4];
	int32_t* input_ids = nullptr;
	int32_t batch_size = 0;
	if(op_id == 2){
		input_ids = sampled_ids;
		batch_size = node_counter[2];
	}else if(op_id > 2){
		input_ids = agg_src_ids + edge_counter[2];
		batch_size = node_counter[2];
	}
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * count; idx += gridDim.x * blockDim.x){
		if(threadIdx.x == 0){
            local_offset[0] = 0;
			local_offset[1] = 0;
            local_offset[2] = 0;
            local_offset[3] = 0;
        }
        __syncthreads();
		int32_t sample_src_id = input_ids[idx/count];
		int32_t sample_dst_id;
		if(sample_src_id >= 0){
			int32_t neighbor_offset = idx%count;
			int32_t part_id = partition_index[sample_src_id];
			int32_t part_offset = parition_offset[sample_src_id];
			int64_t start_index = csr_node_index[part_id][part_offset];
			int32_t col_size = csr_node_index[part_id][(part_offset + 1)] - start_index;
			if(neighbor_offset >= col_size){
				sample_dst_id = -1;
			}else{
				thrust::minstd_rand engine;
				engine.discard(idx);
				thrust::uniform_int_distribution<> dist(0, col_size - 1);
				int32_t dst_index = dist(engine);
				sample_dst_id = csr_dst_node_ids[part_id][(int64_t(start_index + int64_t(dst_index)))];
				if(sample_dst_id >= 0){
					int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					if(acc_count == 0){//first time node is sampled
						int32_t node_off = atomicAdd(local_offset, 1);
						sh_sampled_id[node_off] = sample_dst_id;
					}
					int32_t edge_off = atomicAdd(local_offset + 2, 1);
					sh_agg_src_ids[edge_off] = sample_dst_id;
					sh_agg_dst_ids[edge_off] = sample_src_id;
				}
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
            local_offset[1] = atomicAdd(node_counter + 1, local_offset[0]);//global node count current hop
			local_offset[3] = atomicAdd(edge_counter + 1, local_offset[2]);//global edge count current hop
		}
        __syncthreads();
		if(threadIdx.x < local_offset[0]){
			int32_t node_base = local_offset[1] + node_counter[0];
			int32_t dst_id = sh_sampled_id[threadIdx.x]; 
			sampled_ids[node_base + threadIdx.x] = dst_id;
			position_map[dst_id] = node_base + threadIdx.x;
		}
		__syncthreads();
		if(threadIdx.x < local_offset[2]){
			int32_t edge_base = local_offset[3] + edge_counter[0];
			agg_src_ids[edge_base + threadIdx.x] = sh_agg_src_ids[threadIdx.x];
			agg_dst_ids[edge_base + threadIdx.x] = sh_agg_dst_ids[threadIdx.x]; 
		}
		__syncthreads();
	}
}

__global__ void construct_graph(int32_t* agg_src_ids, int32_t* agg_dst_ids, 
								int32_t* agg_src_off, int32_t* agg_dst_off,
								int32_t* position_map, int32_t* edge_counter, int32_t* node_counter, int32_t op_id, int32_t dev_id){
	int32_t edge_num = edge_counter[1];
	int32_t edge_off = edge_counter[0];
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < edge_num; idx += gridDim.x * blockDim.x){
		int32_t src_id = agg_src_ids[edge_off + idx];
		int32_t dst_id = agg_dst_ids[edge_off + idx];
		int32_t src_of = position_map[src_id];
		int32_t dst_of = position_map[dst_id];
		agg_src_off[edge_off + idx] = src_of;
		agg_dst_off[edge_off + idx] = dst_of;
	}
}

extern "C" 
void GPU_Random_Sampling(
	stream_handle strm_hdl, 
	GPUGraphStorage* graph,
	GPUCache* cache,
	GPUMemoryPool* memorypool,
	int32_t count,
	int32_t op_id) 
{		
	int32_t dev_id;
	cudaGetDevice(&dev_id);
	if(graph == nullptr){
		std::cout<<"invalid storage ptr\n";
		return;
	}

	int32_t** csr_dst_node_ids = graph -> GetCSRNodeMatrix(dev_id);
	int64_t** csr_node_index  = graph -> GetCSRNodeIndex(dev_id);
	int32_t partition_count = graph -> GetPartitionCount();
	char* partition_index = graph -> PartitionIndex(dev_id);
	int* parition_offset = graph -> PartitionOffset(dev_id);

	int32_t* accessed_map = memorypool->GetAccessedMap();
	int32_t* position_map = memorypool->GetPositionMap();
	int32_t* node_counter = memorypool->GetNodeCounter();
	int32_t* edge_counter = memorypool->GetEdgeCounter();
	int32_t* sampled_ids = memorypool->GetSampledIds();
	int32_t* agg_src_ids = memorypool->GetAggSrcId();
	int32_t* agg_dst_ids = memorypool->GetAggDstId();
	int32_t* agg_src_off = memorypool->GetAggSrcOf();
	int32_t* agg_dst_off = memorypool->GetAggDstOf();

    dim3 block_num(40, 1);
    dim3 thread_num(1024, 1);
	kernel_random_sampler_optimized<<<block_num, thread_num, 0, static_cast<cudaStream_t>(strm_hdl)>>>(sampled_ids, op_id,
																										csr_node_index, csr_dst_node_ids, 
																										partition_index, parition_offset,
																										count, partition_count,
																										agg_src_ids, agg_dst_ids,
																										accessed_map,
																										position_map,
																										node_counter,
																										edge_counter);		
	cudaCheckError();
	construct_graph<<<block_num, thread_num, 0, static_cast<cudaStream_t>(strm_hdl)>>>(agg_src_ids, agg_dst_ids,
																						agg_src_off, agg_dst_off,
																						position_map, edge_counter, node_counter, op_id, dev_id);
	cudaCheckError();	
	update_counter<<<1, 1, 0, static_cast<cudaStream_t>(strm_hdl)>>>(node_counter, edge_counter, op_id, 0);		
	cudaCheckError();																																																																									
}

__global__ void zero_copy_with_cache(
	float* cpu_float_attrs, float* cache_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t total_num_nodes,
	int32_t dev_id,
	int32_t op_id)
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
	int32_t fidx; 
	int32_t foffset;
	if(float_attr_len > 0){
		for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (int64_t(batch_size) * float_attr_len); thread_idx += blockDim.x * gridDim.x){
			fidx = cache_index[thread_idx / float_attr_len];
			foffset = thread_idx % float_attr_len;
			if(fidx < 0){/*cache miss*/
				fidx = sampled_ids[node_off + (thread_idx / float_attr_len)];
				if(fidx >= 0){
					dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_attr_len)) + thread_idx)] = cpu_float_attrs[int64_t(int64_t(int64_t(fidx%total_num_nodes) * float_attr_len) + foffset)];
				}
			}else{/*cache hit, find gpu*/
				int32_t local_id = fidx % cache_capacity;
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_attr_len)) + thread_idx)] = cache_float_attrs[int64_t(int64_t(int64_t(local_id) * float_attr_len) + foffset)];
			}
		}
	}
}

extern "C"
void get_feature_kernel(
	stream_handle strm_hdl, 
	GPUCache* cache, 
	GPUNodeStorage* noder,
	GPUMemoryPool* memorypool,
	int32_t dev_id,
	int32_t op_id)
{	
	float* float_attrs = noder->GetAllFloatAttr();
	int32_t float_attr_len = noder->GetFloatAttrLen();
	int32_t total_num_nodes = noder->TotalNodeNum();

	int32_t cache_capacity = cache->Capacity(); 
	int32_t* cache_index = memorypool->GetCacheSearchBuffer();
	int32_t* sampled_ids = memorypool->GetSampledIds();
	int32_t* node_counter = memorypool->GetNodeCounter();
	float* dst_float_buffer = memorypool->GetFloatFeatures();

	cache->Find(sampled_ids, cache_index, node_counter, op_id, strm_hdl, dev_id);
	cudaCheckError();

	float* cpu_float_attrs = float_attrs;
	float* cache_float_attrs = nullptr;

	if(float_attr_len > 0){
		cache_float_attrs = cache->Float_Feature_Cache(dev_id);
	}
	dim3 block_num(40, 1);
	dim3 thread_num(1024, 1);
	zero_copy_with_cache<<<block_num, thread_num, 0, static_cast<cudaStream_t>(strm_hdl)>>>(cpu_float_attrs, cache_float_attrs, float_attr_len,
																							sampled_ids, cache_index, cache_capacity,
																							node_counter, dst_float_buffer,
																							total_num_nodes,
																							dev_id, op_id);
	cudaCheckError();
}	


extern "C"
void make_update_plan(
	stream_handle strm_hdl, 
	GPUGraphStorage* graph, 
	GPUCache* cache,
	GPUMemoryPool* memorypool,
	int32_t dev_id,
	int32_t mode)
{
	if(mode == TRAINMODE){
		int32_t** csr_dst_node_ids = graph -> GetCSRNodeMatrix(dev_id);
		int64_t** csr_node_index = graph -> GetCSRNodeIndex(dev_id);
		char* partition_index = graph -> PartitionIndex(dev_id);
		int* parition_offset = graph -> PartitionOffset(dev_id);
		int32_t* node_counter = memorypool->GetNodeCounter();
		int32_t* sampled_ids = memorypool->GetSampledIds();
		cache -> MakePlan(sampled_ids, nullptr, node_counter, csr_node_index, csr_dst_node_ids, partition_index, parition_offset, strm_hdl, dev_id);
	}
}

/*every shard has different piece of candidates*/
extern "C"
void update_cache(
	stream_handle strm_hdl, 
	GPUCache* cache, 
	GPUNodeStorage* noder,
	GPUMemoryPool* memorypool,
	int32_t dev_id,
	int32_t mode)
{
	if(mode == TRAINMODE){
		int32_t float_attr_len = noder->GetFloatAttrLen();
		float* cache_float_feature = nullptr;
		int32_t* candidates_ids = memorypool->GetSampledIds();
		float* candidates_float_feature = memorypool->GetFloatFeatures();
		if(float_attr_len > 0){
			cache_float_feature = cache -> Float_Feature_Cache(dev_id);
		}
		cache->Update(candidates_ids, candidates_float_feature, cache_float_feature, float_attr_len, strm_hdl, dev_id);
	}
}
