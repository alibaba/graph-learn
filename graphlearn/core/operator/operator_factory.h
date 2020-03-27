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

#ifndef GRAPHLEARN_CORE_OPERATOR_OPERATOR_FACTORY_H_
#define GRAPHLEARN_CORE_OPERATOR_OPERATOR_FACTORY_H_

#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include "graphlearn/core/operator/operator.h"

namespace graphlearn {

class GraphStore;

namespace op {

class OperatorFactory {
public:
  static OperatorFactory& GetInstance() {
    static OperatorFactory factory;
    return factory;
  }

  OperatorFactory(const OperatorFactory&) = delete;
  OperatorFactory& operator=(const OperatorFactory&) = delete;

  ~OperatorFactory();

  // set graph store of registed operators.
  void Set(GraphStore* graph_store);

  void Register(const std::string& name, Operator* op);

  Operator* Lookup(const std::string& name);

private:
  OperatorFactory() = default;

private:
  std::unordered_map<std::string, Operator*> map_;
  std::mutex mtx_;
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
      auto& factory = ::graphlearn::op::OperatorFactory::GetInstance(); \
      factory.Register(OpName, Create##OpClass());                      \
    }                                                                   \
  };                                                                    \
  static Register##OpClass register_##OpClass;

#endif  // GRAPHLEARN_CORE_OPERATOR_OPERATOR_FACTORY_H_
