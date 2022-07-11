/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "core/execution/op_registry.h"

namespace dgs {
namespace execution {

class LookupEdgeOp : public Op {
public:
  LookupEdgeOp() : Op() {}
  ~LookupEdgeOp() override = default;

  seastar::future<Op::Records>
  Process(const storage::SampleStore* store,
          std::vector<Tensor>&& inputs,
          QueryResponseBuilder* builder) override {
    // Multiple upstreams are not supported for LookupEdgeOp now.
    auto& input = inputs[0].ids;
    Op::Records res;
    for (size_t idx = 0; idx < input.size(); ++idx) {
      auto vid = input[idx];
      if (unique_keys_.find(vid) == unique_keys_.end()) {
        unique_keys_.emplace(vid);
        std::vector<storage::KVPair> records;
        Op::Records nbrs;
        storage::Key::Prefix prefix(vtype_, vid, id_);
        store->GetEdgesByPrefix(prefix, &records);
        for (auto& rec : records) {
          res.emplace_back(std::move(rec.value.Share()));
          nbrs.emplace_back(std::move(rec.value.Share()));
        }
        builder->Put(id_, vid, nbrs);
      }
    }
    return seastar::make_ready_future<Op::Records>(std::move(res));
  }
};

class LookupVertexOp : public Op {
public:
  LookupVertexOp() : Op() {}
  ~LookupVertexOp() override = default;

  seastar::future<Op::Records>
  Process(const storage::SampleStore* store,
          std::vector<Tensor>&& inputs,
          QueryResponseBuilder* builder) override {
    // Multiple upstreams are supported for LookupVertexOp.
    Op::Records res;
    for (auto& input : inputs) {
      for (size_t idx = 0; idx < input.ids.size(); ++idx) {
        auto vid = input.ids[idx];
        if (unique_keys_.find(vid) == unique_keys_.end()) {
          unique_keys_.emplace(vid);
          std::vector<storage::KVPair> records;
          Op::Records vertices;
          storage::Key::Prefix prefix(vtype_, vid, id_);
          store->GetVerticesByPrefix(prefix, &records);
          for (auto& rec : records) {
            res.emplace_back(std::move(rec.value.Share()));
            vertices.emplace_back(std::move(rec.value.Share()));
          }
          builder->Put(id_, vid, vertices);
        }
      }
    }
    return seastar::make_ready_future<Op::Records>(std::move(res));
  }
};

namespace registration {

OpRegistration<LookupEdgeOp> LookupEdgeOpRegistration("LookupEdgeOp");
OpRegistration<LookupVertexOp> LookupVertexOpRegistration("LookupVertexOp");

}  // namespace registration

}  // namespace execution
}  // namespace dgs
