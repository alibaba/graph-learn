/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <random>
#include <unordered_set>
#include "graphlearn/core/operator/sampler/sampler.h"

namespace graphlearn {
namespace op {

class RandomNegativeSampler : public Sampler {
public:
  virtual ~RandomNegativeSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t count = req->NeighborCount();
    int32_t batch_size = req->BatchSize();

    res->SetBatchSize(batch_size);
    res->SetNeighborCount(count);
    res->InitEdgeIds(batch_size * count);
    res->InitNeighborIds(batch_size * count);

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());

    auto dst_ids = storage->GetAllDstIds();
    std::uniform_int_distribution<> dist(0, dst_ids.Size() - 1);
    for (int32_t i = 0; i < batch_size; ++i) {
      for (int32_t j = 0; j < count; ++j) {
        int32_t idx = dist(engine);
        res->AppendNeighborId(dst_ids[idx]);
      }
    }

    return Status::OK();
  }
};

REGISTER_OPERATOR("RandomNegativeSampler", RandomNegativeSampler);

}  // namespace op
}  // namespace graphlearn
