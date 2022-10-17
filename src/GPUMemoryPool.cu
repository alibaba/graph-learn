#include <math.h>
#include <iostream>
#include "GPUMemoryPool.cuh"

GPUMemoryPool::GPUMemoryPool(int32_t pipeline_depth){
    pipeline_depth_ = pipeline_depth;
    current_pipe_ = 0;
    sampled_ids_.resize(pipeline_depth_);
    labels_.resize(pipeline_depth_);
    float_features_.resize(pipeline_depth_);
    agg_dst_off_.resize(pipeline_depth_);
    agg_src_off_.resize(pipeline_depth_);
    node_counter_.resize(pipeline_depth_);
    edge_counter_.resize(pipeline_depth_);
}
