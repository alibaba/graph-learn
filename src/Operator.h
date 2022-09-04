#ifndef OPERATOR_H
#define OPERATOR_H

struct OpParams {
    int device_id;
    cudaStream_t stream;
    cudaEvent_t event;
    void* memorypool;
    void* cache;
    void* graph;
    void* noder;
    void* env;
    int neighbor_count;
};

class Operator {
public:
    virtual void run(OpParams* params) = 0;
};

Operator* NewBatchGenerator(int op_id);
Operator* NewRandomSampler(int op_id);
Operator* NewFeatureExtractor(int op_id);
Operator* NewCachePlanner(int op_id);
Operator* NewCacheUpdater(int op_id);

#endif