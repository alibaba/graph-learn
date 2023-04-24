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
#include <numeric>
#include <vector>
#include "core/operator/sampler/alias_method.h"
#include "core/operator/sampler/padder/padder.h"
#include "core/operator/sampler/sampler.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

class InDegreeSampler : public Sampler {
public:
  virtual ~InDegreeSampler() {}

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

    const int64_t* src_ids = req->GetSrcIds();
    auto filter = req->GetFilter();
    std::vector<int32_t> indices(count);
    Status s;
    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      if (!neighbor_ids) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } else {
        auto neighbor_size = neighbor_ids.Size();
        auto edge_ids = storage->GetOutEdges(src_id);

        if (*filter) {
          indices.resize(neighbor_size);
          std::iota(indices.begin(), indices.end(), 0);
          filter->ActOn(i, neighbor_ids, edge_ids, storage, &indices);
          SampleFromIndices(neighbor_ids, storage, count, &indices);
        } else {
          SampleFrom(neighbor_ids, storage, count, indices.data());
        }
        auto padder = GetPadder(neighbor_ids, edge_ids);
        padder->SetIndex(indices);
        s = padder->Pad(res, count);
        if (!s.ok()) {
          return s;
        }
      }
    }
    return s;
  }

private:
  void SampleFrom(const ::graphlearn::io::IdArray& neighbor_ids,
                  ::graphlearn::io::GraphStorage* storage,
                  int32_t n, int32_t* indices) {
    std::vector<float> in_degrees;
    in_degrees.reserve(neighbor_ids.Size());
    for (size_t i = 0; i < neighbor_ids.Size(); ++i) {
      ::graphlearn::io::IdType neighbor_id = neighbor_ids[i];
      in_degrees.push_back(
        static_cast<float>(storage->GetInDegree(neighbor_id)));
    }
    AliasMethod am(&in_degrees);
    am.Sample(n, indices);
  }

  void SampleFromIndices(const ::graphlearn::io::IdArray& neighbor_ids,
                         ::graphlearn::io::GraphStorage* storage,
                         int32_t n, std::vector<int32_t>* indices) {
    if (indices->size() > 0) {
      std::vector<float> in_degrees;
      in_degrees.reserve(indices->size());

      for (size_t i = 0; i < indices->size(); ++i) {
        ::graphlearn::io::IdType neighbor_id = neighbor_ids[indices->at(i)];
        in_degrees.push_back(
          static_cast<float>(storage->GetInDegree(neighbor_id)));
      }
      AliasMethod am(&in_degrees);
      std::vector<int32_t> ret(n);
      am.Sample(n, ret.data());
      for (size_t i = 0; i < n; ++i) {
        ret[i] = indices->at(ret[i]);
      }
      indices->swap(ret);
    }
  }
};

REGISTER_OPERATOR("InDegreeSampler", InDegreeSampler);

}  // namespace op
}  // namespace graphlearn
