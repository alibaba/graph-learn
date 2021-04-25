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

#include "graphlearn/core/operator/sampler/padder/padder.h"

#include "graphlearn/core/operator/sampler/padder/circular_padder.h"
#include "graphlearn/core/operator/sampler/padder/replicate_padder.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace op {

class PadderCreator {
public:
  PadderCreator() {}
  ~PadderCreator() = default;

  PadderPtr operator() (const IdArray& neighbors, const IdArray& edges) {
    if (GLOBAL_FLAG(PaddingMode == kCircular)) {
      return PadderPtr(new CircularPadder(neighbors, edges));
    } else {
      return PadderPtr(new ReplicatePadder(neighbors, edges));
    }
  }
};

void BasePadder::SetFilter(int64_t filter) {
  filter_ = filter;
}

void BasePadder::SetIndex(const std::vector<int32_t>& indices) {
  indices_ = &indices;
}

bool BasePadder::HitFilter(int64_t value) {
  if (filter_ == -1) {
    return false;
  }
  return filter_ == value;
}

PadderPtr GetPadder(const IdArray& neighbors, const IdArray& edges) {
  static PadderCreator creator;
  return creator(neighbors, edges);
}

}  // namespace op
}  // namespace graphlearn

