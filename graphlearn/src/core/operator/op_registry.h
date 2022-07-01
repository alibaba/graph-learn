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

#ifndef GRAPHLEARN_CORE_OPERATOR_OP_REGISTRY_H_
#define GRAPHLEARN_CORE_OPERATOR_OP_REGISTRY_H_

#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include "graphlearn/core/operator/operator.h"

namespace graphlearn {

namespace op {

class OpRegistry {
public:
  typedef Operator* (*OpCreator)();

public:
  static OpRegistry* GetInstance() {
    static OpRegistry registry;
    return &registry;
  }

  OpRegistry(const OpRegistry&) = delete;
  OpRegistry& operator=(const OpRegistry&) = delete;

  void Register(const std::string& name, OpCreator creator);

  OpCreator* Lookup(const std::string& name);

private:
  OpRegistry() = default;
  ~OpRegistry() = default;

private:
  std::mutex mu_;
  std::unordered_map<std::string, OpCreator> creator_map_;
};

}  // namespace op
}  // namespace graphlearn

#define REGISTER_OPERATOR(OpName, OpClass)                              \
  inline ::graphlearn::op::Operator* Create##OpClass() {                \
    return new OpClass();                                               \
  }                                                                     \
  class Register##OpClass {                                             \
  public:                                                               \
    Register##OpClass() {                                               \
      auto registry = ::graphlearn::op::OpRegistry::GetInstance();      \
      registry->Register(OpName, Create##OpClass);                      \
    }                                                                   \
  };                                                                    \
  static Register##OpClass register_##OpClass;

#endif  // GRAPHLEARN_CORE_OPERATOR_OP_REGISTRY_H_
