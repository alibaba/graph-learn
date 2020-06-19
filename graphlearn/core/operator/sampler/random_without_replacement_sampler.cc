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
#include <numeric>
#include <algorithm>
#include "graphlearn/core/operator/sampler/padder/padder.h"
#include "graphlearn/core/operator/sampler/sampler.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace op {

class RandomWithoutReplacementSampler : public Sampler {
public:
  virtual ~RandomWithoutReplacementSampler() {}

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

    const int64_t* src_ids = req->GetSrcIds();
    Status s;
    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      if (neighbor_ids == nullptr) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } else {
        thread_local static std::random_device rd;
        thread_local static std::mt19937 engine(rd());

        int32_t neighbor_size = neighbor_ids->size();
        auto edge_ids = storage->GetOutEdges(src_id);

        std::vector<int32_t> indices(neighbor_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), engine);

        auto padder = GetPadder(*neighbor_ids, *edge_ids, indices);
        s = padder->Pad(res, count, neighbor_size);
        if (!s.ok()) {
          return s;
        }
      }
    }
    return s;
  }
};

REGISTER_OPERATOR(
    "RandomWithoutReplacementSampler", RandomWithoutReplacementSampler);

}  // namespace op
}  // namespace graphlearn
