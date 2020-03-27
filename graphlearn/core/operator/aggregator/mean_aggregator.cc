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

#include "graphlearn/core/operator/aggregator/aggregator.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace op {

class MeanAggregator : public Aggregator {
public:
  virtual ~MeanAggregator() {}

  void AggFunc(std::vector<float>* left,
               const std::vector<float>& right) override {
    for (int32_t i = 0; i < left->size(); ++i) {
      left->at(i) += right[i];
    }
  }

  void FinalFunc(std::vector<float>* values,
                 int32_t total) override {
    int32_t size = values->size();
    if (total == 0) {
      values->assign(values->size(),
                     GLOBAL_FLAG(DefaultFloatAttribute));
    } else {
      for (int32_t i = 0; i < size; ++i) {
        values->at(i) = values->at(i) / total;
      }
    }
  }
};

REGISTER_OPERATOR("MeanAggregator", MeanAggregator);

}  // namespace op
}  // namespace graphlearn
