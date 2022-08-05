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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_OP_REF_FACTORY_H_
#define GRAPHLEARN_ACTOR_OPERATOR_OP_REF_FACTORY_H_

#include <functional>
#include <unordered_map>

#include "common/base/log.h"

#include "actor/generated/operator/base_op_ref.act.autogen.h"

namespace graphlearn {
namespace act {

class BaseOperatorActor_ref;

using OpRefBuilder = std::function<
    BaseOperatorActor_ref*(int64_t, hiactor::scope_builder&)>;

class OpRefFactory {
public:
  static OpRefFactory& Get() {
    static OpRefFactory inst;
    return inst;
  }

  BaseOperatorActor_ref* Create(const std::string& op_name,
                                int64_t guid,
                                hiactor::scope_builder* builder) {
    auto iter = map_.find(op_name);
    if (iter != map_.end()) {
      return (iter->second)(guid, *builder);
    } else {
      LOG(ERROR) << op_name + "is not registered.";
      USER_LOG(op_name + "is not registered.");
      return nullptr;
    }
  }

  void Register(const std::string& op_name, OpRefBuilder func) {
    map_[op_name] = std::move(func);
  }

private:
  OpRefFactory() = default;
  OpRefFactory(const OpRefFactory&) = default;

private:
  std::unordered_map<std::string, OpRefBuilder> map_;
};

template <typename T>
class OpRefRegistration {
public:
  explicit OpRefRegistration(const std::string& op_name) noexcept {
    static_assert(std::is_base_of<BaseOperatorActor_ref, T>::value,
        "T must be a derived class of BaseOperatorActor_ref.");
    OpRefFactory::Get().Register(
      op_name,
      [] (int64_t guid, hiactor::scope_builder& builder) {
        return builder.new_ref<T>(guid);
      });
  }
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_OP_REF_FACTORY_H_
