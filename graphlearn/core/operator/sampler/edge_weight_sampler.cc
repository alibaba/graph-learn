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
#include <vector>
#include "graphlearn/core/operator/sampler/alias_method.h"
#include "graphlearn/core/operator/sampler/padder/padder.h"
#include "graphlearn/core/operator/sampler/sampler.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace op {

class EdgeWeightSampler : public Sampler {
public:
  virtual ~EdgeWeightSampler() {}

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

    std::vector<int32_t> indices(count);

    const int64_t* src_ids = req->GetSrcIds();
    Status s;
    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      if (!neighbor_ids) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } else {
        auto edge_ids = storage->GetOutEdges(src_id);
        SampleFrom(edge_ids, storage, count, indices.data());
        auto padder = GetPadder(neighbor_ids, edge_ids, indices);
        s = padder->Pad(res, count, count);
        if (!s.ok()) {
          return s;
        }
      }
    }
    return s;
  }

private:
  void SampleFrom(const ::graphlearn::io::IdArray& edge_ids,
                  ::graphlearn::io::GraphStorage* storage,
                  int32_t n, int32_t* indices) {
    std::vector<float> edge_weights;
    edge_weights.reserve(edge_ids.Size());
    for (size_t i = 0; i < edge_ids.Size(); ++i) {
      ::graphlearn::io::IdType edge_id = edge_ids[i];
      edge_weights.push_back(storage->GetEdgeWeight(edge_id));
    }

    AliasMethod am(&edge_weights);
    am.Sample(n, indices);
  }
};

REGISTER_OPERATOR("EdgeWeightSampler", EdgeWeightSampler);

}  // namespace op
}  // namespace graphlearn
