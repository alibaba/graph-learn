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
#include <unordered_set>
#include "common/base/log.h"
#include "core/operator/sampler/alias_method.h"
#include "core/operator/sampler/sampler.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

namespace {
const int32_t kRetryTimes = 3;
}  // anonymous space

class InDegreeNegativeSampler : public Sampler {
public:
  virtual ~InDegreeNegativeSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t count = req->NeighborCount();
    int32_t batch_size = req->BatchSize();

    res->SetShape(batch_size, count);
    res->InitEdgeIds();
    res->InitNeighborIds();

    const int64_t* src_ids = req->GetSrcIds();
    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    AliasMethod* am = CreateAM(edge_type, storage);
    SampleAndFill(storage, src_ids, batch_size, count, am, res);

    return Status::OK();
  }

protected:
  virtual void SampleAndFill(::graphlearn::io::GraphStorage* storage,
                             const int64_t* src_ids,
                             int32_t batch_size,
                             int32_t n,
                             AliasMethod* am,
                             SamplingResponse* res) {
    std::unique_ptr<int32_t[]> indices(new int32_t[n]);
    auto dst_ids = storage->GetAllDstIds();
    if (!dst_ids) {
      LOG(ERROR) << "Sample negatively on not existed edge_type.";
      res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      return;
    }
    for (int32_t i = 0; i < batch_size; ++i) {
      auto nbr_ids = storage->GetNeighbors(src_ids[i]);

      std::unordered_set<int64_t> sets;
      for (int32_t k = 0; k < nbr_ids.Size(); ++k) {
        sets.insert(nbr_ids[k]);
      }

      int32_t count = 0;
      int32_t cursor = 0;
      int32_t retry_times = kRetryTimes + 1;
      while (count < n && retry_times >= 0) {
        cursor %= n;
        if (cursor == 0) {
          am->Sample(n, indices.get());
          if (--retry_times <= 0) {
            // After trying RetryTimes, the nbr_ids' size is still
            // less than nbr_count, we should fill nbr_ids with random dst
            // node ids and no longer strictly guarantee the negative ids
            // are true negative.
            sets.clear();
          }
        }

        int64_t item = dst_ids[indices[cursor++]];
        if (sets.find(item) == sets.end()) {
          res->AppendNeighborId(item);
          ++count;
        }
      }
    }
  }

private:
  AliasMethod* CreateAM(const std::string& type,
                        ::graphlearn::io::GraphStorage* storage) {
    AliasMethodFactory* factory = AliasMethodFactory::GetInstance();
    auto in_degrees = storage->GetAllInDegrees();
    return factory->LookupOrCreate(type, in_degrees);
  }
};

class SoftInDegreeNegativeSampler : public InDegreeNegativeSampler {
public:
  virtual ~SoftInDegreeNegativeSampler() = default;

protected:
  void SampleAndFill(::graphlearn::io::GraphStorage* storage,
                     const int64_t* src_ids,
                     int32_t batch_size,
                     int32_t count,
                     AliasMethod* am,
                     SamplingResponse* res) override {
    std::unique_ptr<int32_t[]> indices(new int32_t[count]);
    auto dst_ids = storage->GetAllDstIds();
    for (int32_t i = 0; i < batch_size; ++i) {
      am->Sample(count, indices.get());
      for (int32_t j = 0; j < count; ++j) {
        int32_t idx = indices[j];
        res->AppendNeighborId(dst_ids[idx]);
      }
    }
  }
};

REGISTER_OPERATOR("InDegreeNegativeSampler", InDegreeNegativeSampler);
REGISTER_OPERATOR("SoftInDegreeNegativeSampler", SoftInDegreeNegativeSampler);

}  // namespace op
}  // namespace graphlearn
