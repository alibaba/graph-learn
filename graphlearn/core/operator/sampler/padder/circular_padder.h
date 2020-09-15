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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_CIRCULAR_PADDER_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_CIRCULAR_PADDER_H_

#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/operator/sampler/padder/padder.h"

namespace graphlearn {
namespace op {

class CircularPadder : public BasePadder {
public:
  CircularPadder(const ::graphlearn::io::IdArray& neighbor_ids,
                 const ::graphlearn::io::IdArray& edge_ids,
                 const std::vector<int32_t>& indices)
      : BasePadder(neighbor_ids, edge_ids, indices) {
  }

  ~CircularPadder() = default;

  Status Pad(SamplingResponse* res,
             int32_t target_size,
             int32_t actual_size) override {
    int32_t cursor = 0;
    int32_t idx = 0;
    int32_t value_size = indices_.size();
    while (idx < target_size) {
      cursor %= actual_size;
      if (value_size == 0) {
        res->AppendNeighborId(neighbor_ids_[cursor]);
        if (edge_ids_.Size() > 0) {
          res->AppendEdgeId(edge_ids_[cursor]);
        }
      } else if (value_size >= actual_size) {
        res->AppendNeighborId(neighbor_ids_[indices_[cursor]]);
        if (edge_ids_.Size() > 0) {
          res->AppendEdgeId(edge_ids_[indices_[cursor]]);
        }
      } else {
        LOG(ERROR) << "Padding value size of indices "
                   << value_size
                   << " < actual_size " << actual_size;
        return error::InvalidArgument("Padding value size too small.");
      }
      ++idx;
      ++cursor;
    }
    return Status::OK();
  }
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_CIRCULAR_PADDER_H_
