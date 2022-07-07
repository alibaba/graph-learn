/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_CORE_EXECUTION_OP_REGISTRY_H_
#define DGS_CORE_EXECUTION_OP_REGISTRY_H_

#include <mutex>
#include <unordered_map>

#include "core/execution/op.h"

namespace dgs {
namespace execution {

class OpRegistry {
public:
  typedef Op* (*OpCreator)();

public:
  static OpRegistry* GetInstance() {
    static OpRegistry registry;
    return &registry;
  }

  OpRegistry(const OpRegistry&) = delete;
  OpRegistry& operator=(const OpRegistry&) = delete;

  void Register(const std::string& name, OpCreator creator) {
    std::unique_lock<std::mutex> _(mu_);
    if (creator_map_.find(name) != creator_map_.end()) {
      LOG(WARNING) << "Repeated register operator:" << name;
      return;
    }
    creator_map_[name] = creator;
  }

  OpCreator* Lookup(const std::string& name) {
    // Lock free, because the map_ is fixed when lookup.
    auto it = creator_map_.find(name);
    if (it == creator_map_.end()) {
      return nullptr;
    }
    return &(it->second);
  }

private:
  OpRegistry() = default;
  ~OpRegistry() = default;

private:
  std::mutex mu_;
  std::unordered_map<std::string, OpCreator> creator_map_;
};

template <typename T>
class OpRegistration {
public:
  explicit OpRegistration(const std::string& name) {
    static_assert(std::is_base_of<Op, T>::value,
        "T must be a derived class of operator base.");
    OpRegistry::GetInstance()->Register(name, ConstructOp);
  }

private:
  static Op* ConstructOp() {
    return new T{};
  }
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_OP_REGISTRY_H_
