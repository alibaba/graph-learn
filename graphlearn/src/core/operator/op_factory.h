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

#ifndef GRAPHLEARN_CORE_OPERATOR_OP_FACTORY_H_
#define GRAPHLEARN_CORE_OPERATOR_OP_FACTORY_H_

#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/op_registry.h"

namespace graphlearn {

class GraphStore;

namespace op {

class OpFactory {
public:
  static OpFactory* GetInstance();

  virtual void Set(GraphStore* graph_store);

  virtual Operator* Create(const std::string& name) = 0;

protected:
  OpFactory();
  virtual ~OpFactory() = default;

protected:
  OpRegistry* op_registry_;
  GraphStore* graph_store_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_OP_FACTORY_H_
