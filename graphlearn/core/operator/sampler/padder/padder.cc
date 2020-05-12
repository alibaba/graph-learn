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

  PadderPtr operator() (const std::vector<int64_t>& neighbor_ids,
                        const std::vector<int64_t>& edge_ids,
                        const std::vector<int32_t>& indices) {
    
    if (GLOBAL_FLAG(PaddingMode == kCircular)) {
      PadderPtr ret(new CircularPadder(neighbor_ids, edge_ids, indices));
      return ret;
    } else {
      PadderPtr ret(new ReplicatePadder(neighbor_ids, edge_ids, indices));
      return ret;
    }
  }
};

PadderPtr GetPadder(const std::vector<int64_t>& node_ids,
                    const std::vector<int64_t>& edge_ids,
                    const std::vector<int32_t>& indices) {
  static PadderCreator creator;
  return creator(node_ids, edge_ids, indices);
}

}  // namespace op
}  // namespace graphlearn

