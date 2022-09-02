#include <fcntl.h>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <torch/extension.h>
#include "ipc_service.h"

IPCEnv* env;
int32_t h_node_counter[16];
int32_t h_edge_counter[16];

void InitializeIPC(){
    env = NewIPCEnv();
    env->Initialize();
}

void FinalizeIPC(){
    env->Finalize();
}

std::vector<torch::Tensor> cuda_get_next(
    int32_t* ids,
    float* float_features, 
    int32_t* labels,
    int feature_dim,
    int32_t* agg_src,
    int32_t* agg_dst,
    int32_t* node_counter,
    int32_t* edge_counter,
    int32_t* h_node_counter,
    int32_t* h_edge_counter
    );

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> get_next(int feature_dim) {
    env->Wait();
    int32_t* ids = env->GetIds();
    float* float_features = env->GetFloatFeatures();
    int32_t* labels = env->GetLabels();
    int32_t* agg_src = env->GetAggSrc();
    int32_t* agg_dst = env->GetAggDst();
    int32_t* node_counter = env->GetNodeCounter();
    int32_t* edge_counter = env->GetEdgeCounter();
    auto result = cuda_get_next(ids, float_features, labels,
                                feature_dim,
                                agg_src, agg_dst, 
                                node_counter, edge_counter, 
                                h_node_counter, h_edge_counter);
    return result;
}

std::vector<int> get_block_size() {
    std::vector<int> ret;
    int block1_src_node = h_node_counter[9];
    int block1_dst_node = h_node_counter[7];
    int block2_src_node = h_node_counter[7];
    int block2_dst_node = h_node_counter[5];

    ret.push_back(block1_src_node);
    ret.push_back(block1_dst_node);
    ret.push_back(block2_src_node);
    ret.push_back(block2_dst_node);
    return ret;
}

std::vector<int32_t> get_steps(){
    std::vector<int32_t> ret;
    ret.push_back(env->GetTrainStep());
    ret.push_back(env->GetValidStep());
    ret.push_back(env->GetTestStep());
    return ret;
} 

void Synchronize(){
    env->Post();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_next", &get_next, "dataset get next (CUDA)");
  m.def("get_block_size", &get_block_size, "get dgl block size(CUDA)");
  m.def("get_steps", &get_steps, "get steps(CUDA)");
  m.def("initialize", &InitializeIPC, "InitializeIPC (CUDA)");
  m.def("finalize", &FinalizeIPC, "FinalizeIPC (CUDA)");
  m.def("synchronize", &Synchronize, "synchronize (CUDA)");
}