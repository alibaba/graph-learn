#include "GPUGraphStore.cuh"
#include <algorithm>
#include <functional>
#include <iostream>
#include <cuda_runtime.h>

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <math.h>
#include <thread>
#include <numeric>
#include <chrono>
#include <random>

#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

void mmap_trainingset_read(std::string &training_file, std::vector<int32_t>& training_set_ids){
    int64_t t_idx = 0;
    int32_t fd = open(training_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        training_set_ids[t_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_partition_read(std::string &partition_file, std::vector<char>& partition_index){
    int64_t part_idx = 0;
    int32_t fd = open(partition_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        partition_index[part_idx++] = (char)temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_indptr_read(std::string &indptr_file, std::vector<int64_t>& indptr){
    int64_t indptr_index = 0;
    int32_t fd = open(indptr_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int64_t *buf = (int64_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int64_t* buf_end = buf + buf_len/sizeof(int64_t);
    int64_t temp;
    while(buf < buf_end){
        temp = *buf;
        indptr[indptr_index++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_indices_read(std::string &indices_file, std::vector<int32_t>& indices){
    int64_t indices_index = 0;
    int32_t fd = open(indices_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t *buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        indices[indices_index++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_features_read(std::string &features_file, float* features){
    int64_t n_idx = 0;
    int32_t fd = open(features_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const float *buf = (float *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const float* buf_end = buf + buf_len/sizeof(float);
    float temp;
    while(buf < buf_end){
        temp = *buf;
        features[n_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_labels_read(std::string &labels_file, std::vector<int32_t>& labels){
    int64_t n_idx = 0;
    int32_t fd = open(labels_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t *buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        labels[n_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void GPUGraphStore::EnableP2PAccess(){
    int32_t central_device = -1;
    cudaGetDevice(&central_device);

    int32_t device_count = -1;
    cudaGetDeviceCount(&device_count);
    for(int32_t i = 0; i < device_count; i++){
        cudaSetDevice(i);
        cudaCheckError();
        for(int32_t j = 0; j < device_count; j++){
          if(j != i){
            int32_t accessible = 0;
            cudaDeviceCanAccessPeer(&accessible, i, j);
            cudaCheckError();
            if(accessible){
              cudaDeviceEnablePeerAccess(j, 0);
              cudaCheckError();
            }
          }
        }
      }
    cudaSetDevice(central_device);
    central_device_ = central_device;
}

void GPUGraphStore::ConfigPartition(BuildInfo* info, int32_t shard_count){

    shard_to_device_.resize(shard_count);
    for(int32_t i = 0; i < shard_count; i++){
        shard_to_device_[i] = i;
    }
    shard_to_partition_.resize(shard_count);
    for(int32_t i = 0; i < shard_count; i++){
        shard_to_partition_[i] = i;
    }
    for(int32_t i = 0; i < shard_count; i++){
        info->shard_to_partition.push_back(shard_to_partition_[i]);
    }
    for(int32_t i = 0; i < shard_count; i++){
        info->shard_to_device.push_back(shard_to_device_[i]);
    }

    info->partition_count = shard_count;
}

void GPUGraphStore::ReadMetaFIle(BuildInfo* info){
    std::istringstream iss;
    std::string buff;
    std::ifstream Metafile("/home/sunjie/gl-716/meta_config");
    if(!Metafile.is_open()){
     std::cout<<"unable to open meta config file"<<"\n";
    }
    getline(Metafile, buff);
    iss.clear();
    iss.str(buff);
    iss >> dataset_path_;
    std::cout<<"Dataset path:       "<<dataset_path_<<"\n";
    iss >> raw_batch_size_;
    std::cout<<"Raw Batchsize:      "<<raw_batch_size_<<"\n";
    iss >> node_num_;
    std::cout<<"Graph nodes num:    "<<node_num_<<"\n";
    iss >> edge_num_;
    std::cout<<"Graph edges num:    "<<edge_num_<<"\n";
    iss >> float_attr_len_;
    std::cout<<"Feature dim:        "<<float_attr_len_<<"\n";
    iss >> training_set_num_;
    std::cout<<"Training set num:   "<<training_set_num_<<"\n";
    iss >> validation_set_num_;
    std::cout<<"Validation set num: "<<validation_set_num_<<"\n";
    iss >> testing_set_num_;
    std::cout<<"Testing set num:    "<<testing_set_num_<<"\n";
    iss >> cache_cap_;
    std::cout<<"Cache capacity:     "<<cache_cap_<<"\n";
    iss >> cache_way_;
    std::cout<<"Cache way num:      "<<cache_way_<<"\n"; 
    iss >> future_batch_;
    std::cout<<"Predict by K batch: "<<future_batch_<<"\n"; 
    iss >> epoch_;
    std::cout<<"Train epoch:        "<<epoch_<<"\n";
    info->epoch = epoch_;
}

void GPUGraphStore::Load_Graph(BuildInfo* info){
    std::cout<<"Start load graph\n";

    int32_t partition_count = info->partition_count;
    int32_t node_num = node_num_;
    int64_t edge_num = edge_num_;

    std::string partition_path = dataset_path_ + "partition_" + std::to_string(partition_count);

    std::vector<int> partition_id_num;
    for(int32_t i = 0; i < partition_count; i++){
        partition_id_num.push_back(0);
    }
    (info->partition_offset).resize(node_num);
    (info->partition_index).resize(node_num);
    mmap_partition_read(partition_path, info->partition_index);

    int32_t partition_id;
    for(int32_t i = 0; i < node_num; i++){
        partition_id = info->partition_index[i];
        info->partition_offset[i] = partition_id_num[partition_id];
        partition_id_num[partition_id] += 1;
    }

    std::string edge_src_path = dataset_path_ + "edge_src";
    std::string edge_dst_path = dataset_path_ + "edge_dst";

    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;

    indptr.resize(node_num+1);
    indices.resize(edge_num);

    mmap_indptr_read(edge_src_path, indptr);
    mmap_indices_read(edge_dst_path, indices);

    info->csr_node_index.resize(partition_count);
    info->csr_dst_node_ids.resize(partition_count);

    //partition edges
    for(int32_t i = 0; i < node_num; i++){
        int32_t part_id = info->partition_index[i];
        info->csr_node_index[part_id].push_back(info->csr_dst_node_ids[part_id].size());
        int32_t neighbors_num = indptr[i + 1] - indptr[i];
        for(int32_t j = 0; j < neighbors_num; j++){
            info->csr_dst_node_ids[part_id].push_back(indices[indptr[i] + j]);
        }
    }

    for(int32_t i = 0; i < partition_count; i++){
        info->csr_node_index[i].push_back(info->csr_dst_node_ids[i].size());
    }

    indptr.clear();
    indices.clear();
}


void GPUGraphStore::Load_Feature(BuildInfo* info){
    std::cout<<"start load node\n";

    int32_t partition_count = info->partition_count;

    int32_t node_num = node_num_;
    int32_t nf = float_attr_len_;

    (info->training_set_ids).resize(partition_count);
    (info->training_labels).resize(partition_count);
    (info->validation_set_ids).resize(partition_count);
    (info->validation_labels).resize(partition_count);
    (info->testing_set_ids).resize(partition_count);
    (info->testing_labels).resize(partition_count);

    std::string training_path = dataset_path_  + "trainingset";
    std::string validation_path = dataset_path_  + "validationset";
    std::string testing_path = dataset_path_  + "testingset";
    std::string features_path = dataset_path_ + "features";
    std::string labels_path = dataset_path_ + "labels";
    std::string partition_path = dataset_path_ + "partition_" + std::to_string(partition_count);

    std::vector<int32_t> training_ids;
    training_ids.resize(training_set_num_);
    std::vector<int32_t> validation_ids;
    validation_ids.resize(validation_set_num_);
    std::vector<int32_t> testing_ids;
    testing_ids.resize(testing_set_num_);
    std::vector<int32_t> all_labels;
    all_labels.resize(node_num);
    std::vector<char> partition_index;
    partition_index.resize(node_num);
    float* host_float_attrs;
    cudaHostAlloc(&host_float_attrs, int64_t(int64_t(int64_t(node_num) * nf) * sizeof(float)), cudaHostAllocMapped);
    cudaCheckError();

    mmap_trainingset_read(training_path, training_ids);
    mmap_trainingset_read(training_path, validation_ids);
    mmap_trainingset_read(training_path, testing_ids);
    mmap_features_read(features_path, host_float_attrs);
    mmap_labels_read(labels_path, all_labels);
    mmap_partition_read(partition_path, partition_index);

    //partition nodes
    for(int32_t i = 0; i < training_set_num_; i++){
        int32_t tid = training_ids[i];
        // int32_t part_id = tid % partition_count;
        int32_t part_id = partition_index[tid];
        if(part_id < partition_count){
            (info->training_set_ids[part_id]).push_back(tid);
        }
    }

    for(int32_t i = 0; i < validation_set_num_; i++){
        int32_t tid = validation_ids[i];
        // int32_t part_id = tid % partition_count;
        int32_t part_id = partition_index[tid];
        if(part_id < partition_count){
            (info->validation_set_ids[part_id]).push_back(tid);
        }
    }

    for(int32_t i = 0; i < testing_set_num_; i++){
        int32_t tid = testing_ids[i];
        // int32_t part_id = tid % partition_count;
        int32_t part_id = partition_index[tid];
        if(part_id < partition_count){
            (info->testing_set_ids[part_id]).push_back(tid);
        }
    }

    //partition labels
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < info->training_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->training_set_ids[part_id][i]];
            info->training_labels[part_id].push_back(ts_label);
        }
        info->training_set_num.push_back(info->training_set_ids[part_id].size());
    }

    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < info->validation_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->validation_set_ids[part_id][i]];
            info->validation_labels[part_id].push_back(ts_label);
        }
        info->validation_set_num.push_back(info->validation_set_ids[part_id].size());
    }

    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < info->testing_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->testing_set_ids[part_id][i]];
            info->testing_labels[part_id].push_back(ts_label);
        }
        info->testing_set_num.push_back(info->testing_set_ids[part_id].size());
    }

    info->host_float_attrs = host_float_attrs;
    info->float_attr_len = float_attr_len_;
    info->host_int_attrs = nullptr;
    info->int_attr_len = 0;
    info->total_num_nodes = node_num_;

    training_ids.clear();
    validation_ids.clear();
    testing_ids.clear();
    all_labels.clear();
}

void GPUGraphStore::Initialze(int32_t shard_count){

    BuildInfo* info = new BuildInfo();

    EnableP2PAccess();
    
    ConfigPartition(info, shard_count);

    ReadMetaFIle(info);

    Load_Graph(info);

    Load_Feature(info);

    env_ = NewIPCEnv(shard_count);
    env_ -> Coordinate(info);

    node_ = NewGPUMemoryNodeStorage();
    node_ -> Build(info);  

    graph_ = NewGPUMemoryGraphStorage();
    graph_ -> Build(info);

    cudaCheckError();

    cache_ = new GPUCache();
    std::vector<int> device;

    for(int32_t i = 0; i < shard_to_device_.size(); i++){
        if(shard_to_device_[i] >= 0){
            device.push_back(shard_to_device_[i]);
        }
    }

    cudaSetDevice(0);
    cache_ -> Initialize(device, cache_cap_, 0, float_attr_len_, future_batch_, cache_way_);
    cudaSetDevice(0);
}

GPUGraphStorage* GPUGraphStore::GetGraph(){
    return graph_;
}

GPUNodeStorage* GPUGraphStore::GetNode(){
    return node_;
}

GPUCache* GPUGraphStore::GetCache(){
    return cache_;
}

IPCEnv* GPUGraphStore::GetIPCEnv(){
    return env_;
}

int32_t GPUGraphStore::Shard_To_Device(int32_t shard_id){
    return shard_to_device_[shard_id];
}

int32_t GPUGraphStore::Shard_To_Partition(int32_t shard_id){
    return shard_to_partition_[shard_id];
}

int32_t GPUGraphStore::Central_Device(){
    return central_device_;
}