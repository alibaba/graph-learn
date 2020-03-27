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

#include "graphlearn/core/operator/operator_factory.h"

#include <unordered_map>
#include "graphlearn/common/base/log.h"

namespace graphlearn {
namespace op {

OperatorFactory::~OperatorFactory() {
  for (auto it : map_) {
    delete it.second;
  }
}

void OperatorFactory::Set(GraphStore* graph_store) {
  for (auto& op : map_) {
    op.second -> Set(graph_store);
  }
}

void OperatorFactory::Register(const std::string& name, Operator* op) {
  std::unique_lock<std::mutex> _(mtx_);
  if (map_.find(name) != map_.end()) {
    LOG(WARNING) << "Repeated register operator:" << name;
    return;
  }
  map_[name] = op;
}

Operator* OperatorFactory::Lookup(const std::string& name) {
  auto it = map_.find(name);
  if (it != map_.end()) {
    return it->second;
  }
  return nullptr;
}

}  // namespace op
}  // namespace graphlearn
