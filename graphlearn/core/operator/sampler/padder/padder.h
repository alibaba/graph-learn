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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_PADDER_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_PADDER_H_

#include "graphlearn/core/graph/storage/types.h"
#include "graphlearn/include/sampling_request.h"

namespace graphlearn {
namespace op {

class BasePadder {
public:
  BasePadder(const ::graphlearn::io::IdArray& neighbor_ids,
             const ::graphlearn::io::IdArray& edge_ids,
             const std::vector<int32_t>& indices)
      : neighbor_ids_(neighbor_ids),
        edge_ids_(edge_ids),
        indices_(indices) {
  }

  virtual ~BasePadder() = default;

  virtual Status Pad(SamplingResponse* res,
                     int32_t target_size,
                     int32_t actual_size) = 0;

protected:
  const ::graphlearn::io::IdArray& neighbor_ids_;
  const ::graphlearn::io::IdArray& edge_ids_;  
  const std::vector<int32_t>& indices_;
};

typedef std::unique_ptr<BasePadder> PadderPtr;

PadderPtr GetPadder(
  const ::graphlearn::io::IdArray& node_ids,
  const ::graphlearn::io::IdArray& edge_ids,
  const std::vector<int32_t>& indices = {});

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_PADDER_H_
