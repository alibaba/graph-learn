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

#include "graphlearn/core/operator/op_factory.h"

#include <unordered_map>
#include "graphlearn/common/base/log.h"

namespace graphlearn {
namespace op {

/// Maintain an instances map to ensure that instance for one operator
/// only be created once, and manage life cycle of instances in the map.
class CreateOnceOpFactory : public OpFactory {
public:
  CreateOnceOpFactory() : OpFactory() {}

  ~CreateOnceOpFactory() {
    for (auto& it : map_) {
      delete it.second;
    }
  }

  void Set(GraphStore* graph_store) override {
    graph_store_ = graph_store;

    std::unique_lock<std::mutex> _(mtx_);
    for (auto it = map_.begin(); it != map_.end(); ++it) {
      it->second->Set(graph_store_);
    }
  }

  Operator* Create(const std::string& name) override {
    std::unique_lock<std::mutex> _(mtx_);
    auto it = map_.find(name);
    if (it == map_.end()) {
      auto creator = op_registry_->Lookup(name);
      if (!creator) {
        LOG(ERROR) << "No Operator named " << name;
        return nullptr;
      }
      Operator* op = (*creator)();
      if (graph_store_) {
        op->Set(graph_store_);
      }
      map_[name] = op;
    }
    return map_[name];
  }

private:
  std::mutex mtx_;
  std::unordered_map<std::string, Operator*> map_;
};

/// Create operator instance no matter if it has been created,
/// and not manage life cycle of the instance.
class CreateAlwaysOpFactory : public OpFactory {
public:
  CreateAlwaysOpFactory() : OpFactory() {}

  ~CreateAlwaysOpFactory() {}

  Operator* Create(const std::string& name) override {
    auto creator = op_registry_->Lookup(name);
    if (!creator) {
      LOG(ERROR) << "No Operator named " << name;
      return nullptr;
    }
    Operator* op = (*creator)();
    if (graph_store_) {
      op->Set(graph_store_);
    }
    return op;
  }
};

OpFactory::OpFactory()
  : op_registry_(OpRegistry::GetInstance()),
    graph_store_(nullptr) {
}

void OpFactory::Set(GraphStore* graph_store) {
  graph_store_ = graph_store;
}

OpFactory* OpFactory::GetInstance() {
  if (GLOBAL_FLAG(EnableActor) < 1) {
    static CreateOnceOpFactory factory;
    return &factory;
  } else {
    static CreateAlwaysOpFactory factory;
    return &factory;
  }
}

}  // namespace op
}  // namespace graphlearn
