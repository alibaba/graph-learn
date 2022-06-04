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
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/operator/sampler/alias_method.h"
#include "graphlearn/core/operator/sampler/sampler.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace op {

namespace {
const int32_t kRetryTimes = 3;
}  // anonymous space

class NodeWeightNegativeSampler : public Sampler {
public:
  virtual ~NodeWeightNegativeSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t count = req->NeighborCount();
    int32_t batch_size = req->BatchSize();

    res->SetBatchSize(batch_size);
    res->SetNeighborCount(count);
    res->InitEdgeIds(batch_size * count);
    res->InitNeighborIds(batch_size * count);

    const int64_t* src_ids = req->GetSrcIds();
    const std::string& node_type = req->Type();
    Noder* noder = graph_store_->GetNoder(node_type);
    auto storage = noder->GetLocalStorage();

    AliasMethod* am = CreateAM(node_type, storage);
    SampleAndFill(storage, src_ids, batch_size, count, am, res);

    return Status::OK();
  }

protected:
  virtual void SampleAndFill(::graphlearn::io::NodeStorage* storage,
                             const int64_t* src_ids,
                             int32_t batch_size,
                             int32_t n,
                             AliasMethod* am,
                             SamplingResponse* res) {
    std::unique_ptr<int32_t[]> indices(new int32_t[n]);
    auto ids = storage->GetIds();
    if (!ids) {
      LOG(ERROR) << "Sample negatively on not existed node_type.";
      res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      return;
    }
    std::unordered_set<int64_t> sets(src_ids, src_ids + batch_size);
    for (int32_t i = 0; i < batch_size; ++i) {
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

        int64_t item = ids[indices[cursor++]];
        if (sets.find(item) == sets.end()) {
          res->AppendNeighborId(item);
          ++count;
        }
      }
    }
  }

private:
  AliasMethod* CreateAM(const std::string& type,
                        ::graphlearn::io::NodeStorage* storage) {
    AliasMethodFactory* factory = AliasMethodFactory::GetInstance();
    auto weights = storage->GetWeights();
    return factory->LookupOrCreate(type, weights);
  }
};

REGISTER_OPERATOR("NodeWeightNegativeSampler", NodeWeightNegativeSampler);

}  // namespace op
}  // namespace graphlearn
