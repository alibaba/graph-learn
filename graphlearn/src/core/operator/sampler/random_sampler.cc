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
#include "core/operator/sampler/sampler.h"
#include "include/config.h"
#include "include/constants.h"

namespace graphlearn {
namespace op {

class RandomSampler : public Sampler {
public:
  virtual ~RandomSampler() {}

  /// Two types of filters are supported in RandomSampler.
  /// Including: (1) filtering for the dst_id is equal to the given one,
  /// and (2) filtering for the neighbor edge with timestamp exceeding the
  ///  time constraint.
  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t count = req->NeighborCount();
    int32_t batch_size = req->BatchSize();

    res->SetShape(batch_size, count);
    res->InitNeighborIds();
    res->InitEdgeIds();

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());

    const int64_t* src_ids = req->GetSrcIds();
    auto filter = req->GetFilter();
    int32_t retry_times = GLOBAL_FLAG(SamplingRetryTimes);

    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      auto edge_ids = storage->GetOutEdges(src_id);

      if (!neighbor_ids || filter->HitAll(i, neighbor_ids, edge_ids, storage)) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } else {

        std::uniform_int_distribution<> dist(0, neighbor_ids.Size() - 1);
        for (int32_t j = 0; j < count;) {
          int32_t idx = dist(engine);
          if (!*filter || !filter->Hit(i, neighbor_ids, edge_ids, idx, storage) ||
              --retry_times < 0) {
            res->AppendNeighborId(neighbor_ids[idx]);
            res->AppendEdgeId(edge_ids[idx]);
            ++j;
            retry_times = GLOBAL_FLAG(SamplingRetryTimes);
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
