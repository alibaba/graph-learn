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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_REPLICATE_PADDER_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_REPLICATE_PADDER_H_

#include <algorithm>
#include <vector>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/operator/sampler/padder/padder.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

class ReplicatePadder : public BasePadder {
public:
  ReplicatePadder(const IdArray& neighbors, const IdArray& edges)
    : BasePadder(neighbors, edges) {
  }

  ~ReplicatePadder() = default;

  Status Pad(SamplingResponse* res, int32_t target_size) override {
    int32_t size = 0;
    if (indices_) {
      size = std::min(target_size, static_cast<int32_t>(indices_->size()));
    } else {
      size = std::min(target_size, neighbors_.Size());
    }
    for (int32_t idx = 0; idx < size; idx++) {
      int32_t cursor = idx;
      if (indices_ == nullptr) {
        return error::InvalidArgument("Invalid sampler implementation.");
      }
      res->AppendNeighborId(neighbors_[cursor]);
      res->AppendEdgeId(edges_[cursor]);
    }
    for (int32_t idx = size; idx < target_size; idx++) {
      res->AppendNeighborId(GLOBAL_FLAG(DefaultNeighborId));
      res->AppendEdgeId(-1);
    }
    return Status::OK();
  }
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_REPLICATE_PADDER_H_
