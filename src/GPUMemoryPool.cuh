#ifndef GPU_MEMORY_POOL
#define GPU_MEMORY_POOL
#include <math.h>
#include <iostream>
#include <vector>

class GPUMemoryPool {
public:
    GPUMemoryPool(int32_t pipeline_depth);

    int32_t GetOpId(){//only used by sampler
        return op_id_;
    }

    int32_t GetIter(){
        return iter_;
    }

    int32_t GetCurrentMode(){
        return mode_;
    }

    float* GetFloatFeatures(){
        return float_features_[current_pipe_];
    }

    int32_t* GetCacheSearchBuffer(){
        return cache_search_buffer_;
    }

    int32_t* GetLabels(){
        return labels_[current_pipe_];
    }

    int32_t* GetAccessedMap(){
        return accessed_map_;
    }

    int32_t* GetPositionMap(){
        return position_map_;
    }

    int32_t* GetNodeCounter(){
        return node_counter_[current_pipe_];
    }

    int32_t* GetEdgeCounter(){
        return edge_counter_[current_pipe_];
    }

    int32_t* GetSampledIds(){
        return sampled_ids_[current_pipe_];
    }

    int32_t* GetAggSrcId(){
        return agg_src_ids_;
    }

    int32_t* GetAggDstId(){
        return agg_dst_ids_;
    }

    int32_t* GetAggSrcOf(){
        return agg_src_off_[current_pipe_];
    }

    int32_t* GetAggDstOf(){
        return agg_dst_off_[current_pipe_];
    }

    void SetFloatFeatures(float* float_features, int32_t current_pipe){
        float_features_[current_pipe] = float_features;
    }

    void SetCacheSearchBuffer(int32_t* cache_search_buffer){
        cache_search_buffer_ = cache_search_buffer;
    }

    void SetLabels(int32_t* labels, int32_t current_pipe){
        labels_[current_pipe] = labels;
    }

    void SetAccessedMap(int32_t* accessed_map){
        accessed_map_ = accessed_map;
    }

    void SetPositionMap(int32_t* position_map){
        position_map_ = position_map;
    }

    void SetNodeCounter(int32_t* node_counter, int32_t current_pipe){
        node_counter_[current_pipe] = node_counter;
    }

    void SetEdgeCounter(int32_t* edge_counter, int32_t current_pipe){
        edge_counter_[current_pipe] = edge_counter;
    }

    void SetSampledIds(int32_t* sampled_ids, int32_t current_pipe){
        sampled_ids_[current_pipe] = sampled_ids;
    }

    void SetAggSrcId(int32_t* agg_src_ids){
        agg_src_ids_ = agg_src_ids;
    }

    void SetAggDstId(int32_t* agg_dst_ids){
        agg_dst_ids_ = agg_dst_ids;
    }

    void SetAggSrcOf(int32_t* agg_src_off, int32_t current_pipe){
        agg_src_off_[current_pipe] = agg_src_off;
    }

    void SetAggDstOf(int32_t* agg_dst_off, int32_t current_pipe){
        agg_dst_off_[current_pipe] = agg_dst_off;
    }

    void SetOpId(int32_t op_id){
        op_id_ = op_id;
    }

    void SetCurrentPipe(int32_t current_pipe){
        current_pipe_ = current_pipe;
    }

    void SetCurrentMode(int32_t mode){
        mode_ = mode;
    }

    void SetIter(int32_t iter){
        iter_ = iter;
    }

private:
    int32_t iter_;
    int32_t mode_;
    int32_t op_id_;
    int32_t* cache_search_buffer_;
    int32_t* accessed_map_;
	int32_t* position_map_;
	int32_t* agg_src_ids_;
	int32_t* agg_dst_ids_;

    int32_t pipeline_depth_;
    int32_t current_pipe_;
    std::vector<float*> float_features_;
    std::vector<int32_t*> labels_;
	std::vector<int32_t*> node_counter_;
	std::vector<int32_t*> edge_counter_;
	std::vector<int32_t*> sampled_ids_;
	std::vector<int32_t*> agg_src_off_;
	std::vector<int32_t*> agg_dst_off_;
};

#endif