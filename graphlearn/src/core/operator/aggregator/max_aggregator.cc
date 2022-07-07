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

#include <float.h>
#include "core/operator/aggregator/aggregator.h"

namespace graphlearn {
namespace op {

class MaxAggregator : public Aggregator {
public:
  virtual ~MaxAggregator() {}

  void InitFunc(float* value, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
      value[i] = FLT_MIN_10_EXP;
    }
  }

  void AggFunc(float* left,
               const float* right,
               int32_t size,
               const int32_t* segments,
               int32_t num_segments) override {
    for (int32_t i = 0; i < size; ++i) {
      left[i] = std::max(left[i], right[i]);
    }
  }
};

REGISTER_OPERATOR("MaxAggregator", MaxAggregator);

}  // namespace op
}  // namespace graphlearn
