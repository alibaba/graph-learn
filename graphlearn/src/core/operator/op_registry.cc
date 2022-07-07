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

#include "core/operator/op_registry.h"

#include <unordered_map>
#include "common/base/log.h"

namespace graphlearn {
namespace op {

void OpRegistry::Register(const std::string& name, OpCreator creator) {
  std::unique_lock<std::mutex> _(mu_);
  if (creator_map_.find(name) != creator_map_.end()) {
    LOG(WARNING) << "Repeated register operator:" << name;
    return;
  }
  creator_map_[name] = creator;
}

OpRegistry::OpCreator* OpRegistry::Lookup(const std::string& name) {
  // Lock free, because the map_ is fixed when lookup.
  auto it = creator_map_.find(name);
  if (it == creator_map_.end()) {
    LOG(ERROR) << "No operator creator named " << name;
    return nullptr;
  }
  return &(it->second);
}

}  // namespace op
}  // namespace graphlearn
