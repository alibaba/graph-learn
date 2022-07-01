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

#include <memory>
#include <vector>
#include "graphlearn/core/graph/storage/types.h"
#include "graphlearn/include/sampling_request.h"

namespace graphlearn {
namespace op {

typedef ::graphlearn::io::IdArray IdArray;

class BasePadder {
public:
  BasePadder(const IdArray& neighbors, const IdArray& edges)
    : neighbors_(neighbors),
      edges_(edges),
      filter_(-1),
      indices_(nullptr) {
  }

  virtual ~BasePadder() = default;

  void SetFilter(int64_t filter);
  void SetIndex(const std::vector<int32_t>& indices);

  virtual Status Pad(SamplingResponse* res, int32_t target_size) = 0;

protected:
  bool HitFilter(int64_t value);

protected:
  const IdArray& neighbors_;
  const IdArray& edges_;
  int64_t filter_;
  const std::vector<int32_t>* indices_;
};

typedef std::unique_ptr<BasePadder> PadderPtr;
PadderPtr GetPadder(const IdArray& neighbors, const IdArray& edges);

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_PADDER_PADDER_H_
