#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <iostream>
#include <vector>

class CacheController{
public:
    virtual ~CacheController() = default;
    
    virtual void Initialize(
        int32_t dev_id, 
        int32_t capacity, 
        int32_t sampled_num, 
        int32_t total_num_nodes,
        int32_t batch_size) = 0;

    virtual void Finalize(int32_t dev_id) = 0;

    virtual void Find(
        int32_t* sampled_ids, 
        int32_t* cache_offset, 
        int32_t* node_counter, 
        int32_t op_id,
        void* stream) = 0;

    virtual void MakePlan(
        int32_t* candidates, 
        float* importance, 
        int32_t* node_counter, 
        int64_t** csr_node_index,
        int32_t** csr_dst_node_ids, 
        char* partition_index,
        int32_t* parition_offset,
        void* stream) = 0;

    virtual void Update(
        int32_t* candidates_ids, 
        float* candidates_float_feature, 
        float* cache_float_feature,
        int32_t float_attr_len, 
        void* stream) = 0;

    virtual void AccessCount(
        int32_t* d_key, 
        int32_t num_keys, 
        void* stream) = 0;
   
    virtual int32_t* FutureBatch() = 0;
    
    virtual int32_t Capacity() = 0;

    virtual int32_t* AllCachedIds() = 0;

    virtual int32_t* RecentMark() = 0;
};

CacheController* NewDirectMapCacheController();
CacheController* NewSetAssociateCacheController(int32_t way_num, int32_t K_Batch);

class GPUCache{
public:
    void Initialize(
        std::vector<int> device, 
        int32_t capacity, 
        int32_t int_attr_len, 
        int32_t float_attr_len, 
        int32_t K_batch, 
        int32_t way_num);
    
    void InitializeCacheController(
        int32_t dev_id, 
        int32_t capacity, 
        int32_t sampled_num, 
        int32_t total_num_nodes,
        int32_t batch_size);

    void Finalize();

    int32_t Capacity();

    //these api will change, find, update, clear
    void Find(
        int32_t* sampled_ids, 
        int32_t* cache_offset, 
        int32_t* node_counter, 
        int32_t op_id,
        void* stream,
        int32_t dev_id);

    void MakePlan(
        int32_t* candidates, 
        float* importance, 
        int32_t* node_counter, 
        int64_t** csr_node_index,
        int32_t** csr_dst_node_ids, 
        char* partition_index,
        int32_t* parition_offset,
        void* stream,
        int32_t dev_id);
    
    void Update(
        int32_t* candidates_ids, 
        float* candidates_float_feature, 
        float* cache_float_feature,
        int32_t float_attr_len, 
        void* stream,
        int32_t dev_id);

    void AccessCount(
        int32_t* d_key, 
        int32_t num_keys, 
        void* stream, 
        int32_t dev_id);
    
    int32_t* FutureBatch(int32_t dev_id);

    int32_t* AllCachedIds(int32_t dev_id);//get cache ids on gpu dev_id
    
    int32_t* RecentMark(int32_t dev_id);

    float* Float_Feature_Cache(int32_t dev_id);//return all features

    int64_t* Int_Feature_Cache(int32_t dev_id);

    int32_t K_Batch();

    int64_t GetCacheHit(int32_t dev_id);

    void SetCacheHit(int32_t dev_id, int32_t hit_times);

    void ResetCacheHit(int32_t dev_id);

private:    
    std::vector<bool> dev_ids_;/*valid device, indexed by device id, False means invalid, True means valid*/
    std::vector<CacheController*> cache_controller_;
    int32_t capacity_;

    std::vector<int64_t*> int_feature_cache_;
    std::vector<float*> float_feature_cache_; 

    int32_t int_attr_len_;
    int32_t float_attr_len_;
    int32_t k_batch_;
    int32_t way_num_;
    std::vector<int64_t> cache_hit_;

public:
    std::vector<int> iter_;
    std::vector<int*> d_accessed;
    std::vector<int*> global_count;
};

#endif