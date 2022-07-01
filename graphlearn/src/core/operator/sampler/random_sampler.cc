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

#include <cmath>
#include <random>
#include "graphlearn/core/operator/sampler/sampler.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace op {

class RandomSampler : public Sampler {
public:
  virtual ~RandomSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t count = req->NeighborCount();
    int32_t batch_size = req->BatchSize();

    res->SetBatchSize(batch_size);
    res->SetNeighborCount(count);
    res->InitNeighborIds(batch_size * count);
    res->InitEdgeIds(batch_size * count);

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());

    const int64_t* src_ids = req->GetSrcIds();
    const int64_t* filters = req->GetFilters();

    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      if (!neighbor_ids || (filters && neighbor_ids.Size() == 1
          && neighbor_ids[0] == filters[i])) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } else {
        auto edge_ids = storage->GetOutEdges(src_id);
        std::uniform_int_distribution<> dist(0, neighbor_ids.Size() - 1);
        for (int32_t j = 0; j < count;) {
          int32_t idx = dist(engine);
          if (!filters || filters[i] != neighbor_ids[idx]) {
            res->AppendNeighborId(neighbor_ids[idx]);
            res->AppendEdgeId(edge_ids[idx]);
            ++j;
          }
        }
      }
    }
    return Status::OK();
  }
};

REGISTER_OPERATOR("RandomSampler", RandomSampler);

}  // namespace op
}  // namespace graphlearn
