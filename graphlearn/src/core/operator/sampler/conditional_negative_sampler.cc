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

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/base/macros.h"
#include "core/operator/sampler/alias_method.h"
#include "core/operator/sampler/condition_table.h"
#include "core/operator/sampler/sampler.h"
#include "core/operator/utils/storage_wrapper.h"
#include "core/operator/utils/get_node_attributes_wrapper.h"

namespace graphlearn {
namespace op {

class ConditionalNegativeSampler : public Sampler {
public:
  virtual ~ConditionalNegativeSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    const ConditionalSamplingRequest* request =
      static_cast<const ConditionalSamplingRequest*>(req);
    const int64_t* src_ids = request->GetSrcIds();
    const int64_t* dst_ids = request->GetDstIds();
    int32_t batch_size = request->BatchSize();
    int32_t count = request->NeighborCount();
    const std::string& type = request->Type();
    const std::string& dst_node_type = request->DstNodeType();
    const std::string& strategy = request->Strategy();
    res->SetBatchSize(batch_size);
    res->SetNeighborCount(count);
    res->InitEdgeIds(batch_size * count);
    res->InitNeighborIds(batch_size * count);

    SelectedColumns selected_cols(
        request->IntCols(), request->IntProps(),
        request->FloatCols(), request->FloatProps(),
        request->StrCols(), request->StrProps());
    NodeFrom node_from =
        (strategy == "node_weight") ? NodeFrom::kNode : NodeFrom::kEdgeDst;
    StorageWrapper storage = StorageWrapper(node_from, type, graph_store_);
    ConditionTable* ct = nullptr;
    AliasMethod* am = nullptr;  // default method.
    CreateConditionTable(type, dst_node_type, selected_cols,
        strategy, &storage, &ct, &am);
    RETURN_IF_NOT_OK(ct->GetStatus())
    // Get attributes of input dst ids as sampling condition.
    GetNodeAttributesWrapper attr_wrapper(dst_node_type, dst_ids, batch_size);
    RETURN_IF_NOT_OK(attr_wrapper.GetStatus())
    SampleAndFill(request, &storage, &attr_wrapper, ct, am, res);
    return Status::OK();
  }

private:
  void CreateConditionTable(const std::string& type,
      const std::string& id_type,
      const SelectedColumns& selected_cols,
      const std::string& strategy,
      StorageWrapper* storage,
      ConditionTable** ct,
      AliasMethod** am) {
    auto ids = storage->GetIds();
    ConditionTableFactory* ct_factory = ConditionTableFactory::GetInstance();
    AliasMethodFactory* am_factory = AliasMethodFactory::GetInstance();
    if (strategy == "in_degree") {
      auto weights = storage->GetAllInDegrees();
      *ct = ct_factory->LookupOrCreate(type, id_type, selected_cols,
                                       *ids, *weights);
      *am = am_factory->LookupOrCreate(type, weights);
    } else if (strategy == "node_weight"){
      auto weights = storage->GetNodeWeights();
      *ct = ct_factory->LookupOrCreate(type, id_type, selected_cols,
                                       *ids, *weights);
      *am = am_factory->LookupOrCreate(type, weights);
    } else {  // random as default.
      *ct = ct_factory->LookupOrCreate(type, id_type, selected_cols, *ids);
      *am = am_factory->LookupOrCreate(type, ids->size());
    }
  }

  void SampleAndFill(const ConditionalSamplingRequest* req,
                     StorageWrapper* storage,
                     GetNodeAttributesWrapper* attr_wrapper,
                     ConditionTable* ct,
                     AliasMethod* am,
                     SamplingResponse* res) {
    const int64_t* src_ids = req->GetSrcIds();
    const int64_t* dst_ids = req->GetDstIds();
    int32_t batch_size = req->BatchSize();
    int32_t num = req->NeighborCount();
    bool batch_share = req->BatchShare();
    bool unique = req->Unique();

    std::unordered_set<int64_t> nbr_set;
    if (batch_share) {
      for (int32_t i = 0; i < batch_size; ++i) {
        nbr_set.insert(dst_ids[i]);
      }
    }

    std::unique_ptr<int32_t[]> indices(new int32_t[num]);
    auto ids = storage->GetIds();
    for (int32_t idx = 0; idx < batch_size; ++idx) {
      auto nbr_ids = storage->GetNeighbors(src_ids[idx]);
      if (!batch_share){
        for (int32_t k = 0; k < nbr_ids.Size(); ++k) {
          nbr_set.insert(nbr_ids[k]);
        }
        nbr_set.insert(dst_ids[idx]);
      }
      ct->Sample(attr_wrapper, &nbr_set, num, unique, res);

      int32_t count = res->TotalNeighborCount() - idx * num;
      int32_t cursor = 0;
      int32_t retry_times = GLOBAL_FLAG(NegativeSamplingRetryTimes) + 1;
      while (count < num && retry_times >= 0) {   // default sampling.
        cursor %= num;
        if (cursor == 0) {
          am->Sample(num, indices.get());
          if (--retry_times <= 0) {
            // After trying RetryTimes, the nbr_ids' size is still
            // less than nbr_count, we should fill nbr_ids with random dst
            // node ids and no longer strictly guarantee the negative ids
            // are true negative.
            nbr_set.clear();
          }
        }

        int64_t item = ids->at(indices[cursor++]);
        if (nbr_set.find(item) == nbr_set.end()) {
          res->AppendNeighborId(item);
          if (unique) {
            nbr_set.insert(item);
          }
          ++count;
        }
      }
    }
  }

};


REGISTER_OPERATOR("ConditionalNegativeSampler", ConditionalNegativeSampler);

}  // namespace op
}  // namespace graphlearn
