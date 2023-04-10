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

#include <numeric>
#include <vector>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/operator/sampler/padder/padder.h"
#include "core/operator/sampler/sampler.h"

namespace graphlearn {
namespace op {

class FullSampler : public Sampler {
public:
  virtual ~FullSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t batch_size = req->BatchSize();
    int32_t max_limit_size = req->NeighborCount();

    std::vector<int32_t> degrees(batch_size, max_limit_size);

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    int32_t sum_degree = 0;
    const int64_t* src_ids = req->GetSrcIds();
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      auto edge_ids = storage->GetOutEdges(src_id);
      if (neighbor_ids) {
        if (neighbor_ids.Size() != edge_ids.Size()) {
          LOG(FATAL) << "Inconsistent size of neighbors and edges.";
          return ::graphlearn::error::Internal("Storage error");
        } else {
          int32_t truncated_size =
            GetTruncatedSize(max_limit_size, neighbor_ids.Size());
          degrees[i] = truncated_size;
          sum_degree += truncated_size;
        }
      } else {
        degrees[i] = 0;
      }
    }

    res->SetShape(batch_size, max_limit_size, std::move(degrees));
    res->InitNeighborIds();
    res->InitEdgeIds();

    Status s;
    auto filter = req->GetFilter();
    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      if (neighbor_ids) {
        auto edge_ids = storage->GetOutEdges(src_id);
        std::vector<int32_t> indices(neighbor_ids.Size());
        std::iota(indices.begin(), indices.end(), 0);

        if (*filter) {
          filter->ActOn(i, neighbor_ids, edge_ids, storage, &indices);
        }
        auto padder = GetPadder(neighbor_ids, edge_ids);
        int32_t truncated_size =
          GetTruncatedSize(max_limit_size, neighbor_ids.Size());
        padder->SetIndex(indices);
        s = padder->Pad(res, truncated_size);
        if (!s.ok()) {
          return s;
        }
      }
    }
    return s;
  }

private:
  int32_t GetTruncatedSize(int32_t max_limit_size, int32_t actual_size) {
    if (max_limit_size > 0 && max_limit_size < actual_size) {
      return max_limit_size;
    } else {
      return actual_size;
    }
  }

};

REGISTER_OPERATOR("FullSampler", FullSampler);

}  // namespace op
}  // namespace graphlearn
