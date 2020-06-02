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

#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/operator/sampler/padder/padder.h"
#include "graphlearn/core/operator/sampler/sampler.h"

namespace graphlearn {
namespace op {

class FullSampler : public Sampler {
public:
  virtual ~FullSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t batch_size = req->BatchSize();

    res->SetSparseFlag();
    res->SetBatchSize(batch_size);
    res->SetNeighborCount(0);
    res->InitDegrees(batch_size);

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    int32_t sum_degree = 0;
    const int64_t* src_ids = req->GetSrcIds();
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      auto edge_ids = storage->GetOutEdges(src_id);
      if (neighbor_ids && edge_ids) {
        if (neighbor_ids->size() != edge_ids->size()) {
          LOG(FATAL) << "Inconsistent size of neighbors and edges.";
          return ::graphlearn::error::Internal("Storage error");
        } else {
          res->AppendDegree(neighbor_ids->size());
          sum_degree += neighbor_ids->size();
        }
      } else {
        res->AppendDegree(0);
      }
    }

    res->InitNeighborIds(sum_degree);
    res->InitEdgeIds(sum_degree);

    Status s;
    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      auto edge_ids = storage->GetOutEdges(src_id);
      if (neighbor_ids) {
        int32_t neighbor_size = neighbor_ids->size();
        auto padder = GetPadder(*neighbor_ids, *edge_ids);
        s = padder->Pad(res, neighbor_size, neighbor_size);
        if (!s.ok()) {
          return s;
        }
      }
    }
    return s;
  }
};

REGISTER_OPERATOR("FullSampler", FullSampler);

}  // namespace op
}  // namespace graphlearn
