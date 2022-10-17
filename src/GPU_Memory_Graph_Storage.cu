#include "GPU_Graph_Storage.cuh"
#include <iostream>


__global__ void assign_memory(int32_t** int32_pptr, int32_t* int32_ptr, int64_t** int64_pptr, int64_t* int64_ptr, int32_t device_id){
    int32_pptr[device_id] = int32_ptr;
    int64_pptr[device_id] = int64_ptr;
}


/*in this version, partition id = shard id = device id*/
class GPUMemoryGraphStorage : public GPUGraphStorage {
public:
    GPUMemoryGraphStorage() {
    }

    virtual ~GPUMemoryGraphStorage() {
    }

    void Build(BuildInfo* info) override {
        int32_t partition_count = info->partition_count;
        partition_count_ = partition_count;
        
        // shard count == partition count now
        csr_node_index_.resize(partition_count_);
        csr_dst_node_ids_.resize(partition_count_);
        partition_index_.resize(partition_count_);
        partition_offset_.resize(partition_count_);

        for(int32_t i = 0; i < partition_count; i++){
            cudaSetDevice(i);
            cudaMalloc(&csr_node_index_[i], partition_count * sizeof(int64_t*));
            cudaMalloc(&csr_dst_node_ids_[i], partition_count * sizeof(int32_t*));
            cudaMalloc(&partition_index_[i], int64_t(int64_t(info->total_num_nodes) * sizeof(char)));
            cudaMemcpy(partition_index_[i], &(info->partition_index[0]), int64_t(int64_t(info->total_num_nodes) * sizeof(char)), cudaMemcpyHostToDevice);
            cudaMalloc(&partition_offset_[i], int64_t(int64_t(info->total_num_nodes) * sizeof(int32_t)));
            cudaMemcpy(partition_offset_[i], &(info->partition_offset[0]), int64_t(int64_t(info->total_num_nodes) * sizeof(int32_t)), cudaMemcpyHostToDevice);
        }

        src_size_.resize(partition_count);
        dst_size_.resize(partition_count);
        cudaCheckError();

        for(int32_t i = 0; i < info->shard_to_partition.size(); i++){
            int32_t part_id = info->shard_to_partition[i];
            int32_t device_id = info->shard_to_device[i];
            cudaSetDevice(device_id);

            src_size_[part_id] = info->csr_node_index[part_id].size();
            std::cout<<"part "<<part_id<<" src size "<<src_size_[part_id]<<"\n";
            int64_t* d_csr_node_index;
            if(src_size_[part_id] > 0){
                cudaMalloc(&d_csr_node_index, src_size_[part_id] * sizeof(int64_t));
                cudaCheckError();
                cudaMemcpy(d_csr_node_index, info->csr_node_index[part_id].data(), src_size_[part_id] * sizeof(int64_t), cudaMemcpyHostToDevice);    
                cudaCheckError();
            }
            // csr_node_index_[part_id] = d_csr_node_index;
            cudaCheckError();

            dst_size_[part_id] = info->csr_dst_node_ids[part_id].size();
            std::cout<<"part "<<part_id<<" dst size "<<dst_size_[part_id]<<"\n";
            int32_t* d_csr_dst_node_ids;
            if(dst_size_[part_id] > 0){
                cudaMalloc(&d_csr_dst_node_ids, int64_t(dst_size_[part_id] * sizeof(int32_t)));
                cudaCheckError();
                cudaMemcpy(d_csr_dst_node_ids, info->csr_dst_node_ids[part_id].data(), int64_t(dst_size_[part_id] * sizeof(int32_t)), cudaMemcpyHostToDevice); 
                cudaCheckError();   
            }
            // csr_dst_node_ids_[part_id] = d_csr_dst_node_ids;

            cudaSetDevice(0);
            assign_memory<<<1,1>>>(csr_dst_node_ids_[0], d_csr_dst_node_ids, csr_node_index_[0], d_csr_node_index, device_id);
            cudaCheckError();
        }
        cudaSetDevice(0);
        for(int32_t i = 1; i < partition_count; i++){
            cudaMemcpy(csr_node_index_[i], csr_node_index_[0], partition_count * sizeof(int64_t*), cudaMemcpyDeviceToDevice);
            cudaCheckError();
            cudaMemcpy(csr_dst_node_ids_[i], csr_dst_node_ids_[0], partition_count * sizeof(int32_t*), cudaMemcpyDeviceToDevice);
            cudaCheckError();
        }
    }
    
    void Finalize() override {
        for(int32_t i = 0; i < partition_count_; i++){
            cudaFree(partition_index_[i]);
            cudaFree(partition_offset_[i]);
        }
    }

    //CSR
    int32_t GetPartitionCount() const override {
        return partition_count_;
    }
	int64_t** GetCSRNodeIndex(int32_t dev_id) const override {
		return csr_node_index_[dev_id];
	}
	int32_t** GetCSRNodeMatrix(int32_t dev_id) const override {
        return csr_dst_node_ids_[dev_id];
    }
    
    int64_t** GetCSRNodeIndexOnPart(int32_t part_id) const override {
        return csr_node_index_[part_id];
    }

    int32_t** GetCSRNodeMatrixOnPart(int32_t part_id) const override {
        return csr_dst_node_ids_[part_id];
    }

    int64_t Src_Size(int32_t part_id) const override {
        return src_size_[part_id];
    }
    int64_t Dst_Size(int32_t part_id) const override {
        return dst_size_[part_id];
    }
    char* PartitionIndex(int32_t dev_id) const override {
        return partition_index_[dev_id];
    }
    int32_t* PartitionOffset(int32_t dev_id) const override {
        return partition_offset_[dev_id];
    }
private:
    std::vector<int64_t> src_size_;	
	std::vector<int64_t> dst_size_;

	//CSR graph, every partition has a ptr copy
    int32_t partition_count_;
	std::vector<int64_t**> csr_node_index_;
	std::vector<int32_t**> csr_dst_node_ids_;	
    std::vector<char*> partition_index_;
    std::vector<int32_t*> partition_offset_;

};

extern "C" 
GPUGraphStorage* NewGPUMemoryGraphStorage(){
    GPUMemoryGraphStorage* ret = new GPUMemoryGraphStorage();
    return ret;
}
