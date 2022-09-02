#ifndef GRPAHLEARN_CORE_GRAPH_GPUSTORAGE_TYPES_H
#define GRPAHLEARN_CORE_GRAPH_GPUSTORAGE_TYPES_H
#include <cstdint>
#include <vector>
struct BuildInfo{
    //device                                                                                                                        ``
    std::vector<int32_t> shard_to_partition;
    std::vector<int32_t> shard_to_device;
    int32_t partition_count;
    //training set
    std::vector<int32_t> training_set_num;
    std::vector<std::vector<int32_t>> training_set_ids;
    std::vector<std::vector<int32_t>> training_labels;
    //validation set
    std::vector<int32_t> validation_set_num;
    std::vector<std::vector<int32_t>> validation_set_ids;
    std::vector<std::vector<int32_t>> validation_labels;
    //testing set
    std::vector<int32_t> testing_set_num;
    std::vector<std::vector<int32_t>> testing_set_ids;
    std::vector<std::vector<int32_t>> testing_labels;
    //features
    int32_t total_num_nodes;
    int32_t int_attr_len;
    int32_t float_attr_len;
    int64_t* host_int_attrs;//allocated by cudaHostAlloc
    float* host_float_attrs;//allocated by cudaHostAlloc
    //csr
    std::vector<std::vector<int64_t>> csr_node_index;
    std::vector<std::vector<int32_t>> csr_dst_node_ids;
    std::vector<char> partition_index;
    std::vector<int32_t> partition_offset;
    //train
    int32_t epoch;
};

#endif