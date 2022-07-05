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

#include "core/operator/aggregator/aggregator.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

class MeanAggregator : public Aggregator {
public:
  virtual ~MeanAggregator() {}

  void AggFunc(float* left,
               const float* right,
               int32_t size,
               const int32_t* segments,
               int32_t num_segments) override {
    if (segments != nullptr) {
      int32_t dim = size / num_segments;
      for (int32_t i = 0; i < num_segments; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
          left[i * dim + j] += right[i * dim + j] * segments[i];
        }
      }
    } else {
      for (int32_t i = 0; i < size; ++i) {
        left[i] = left[i] + right[i];
      }
    }
  }

  void FinalFunc(float* values,
                 int32_t size,
                 const int32_t* segments,
                 int32_t num_segments) override {
    int32_t dim = size / num_segments;
    for (int32_t idx = 0; idx < num_segments; ++idx) {
      if (segments[idx] == 0) {
        for (int32_t i = 0; i < dim; ++i) {
          values[idx * dim + i] = GLOBAL_FLAG(DefaultFloatAttribute);
        }
      } else {
        for (int32_t i = 0; i < dim; ++i) {
          values[idx * dim + i] = values[idx * dim + i] / segments[idx];
        }
      }
    }
  }
};

REGISTER_OPERATOR("MeanAggregator", MeanAggregator);

}  // namespace op
}  // namespace graphlearn
