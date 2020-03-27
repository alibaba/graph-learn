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

#ifndef GRAPHLEARN_CORE_OPERATOR_OPERATOR_H_
#define GRAPHLEARN_CORE_OPERATOR_OPERATOR_H_

#include <string>
#include "graphlearn/include/op_request.h"
#include "graphlearn/include/status.h"
#include "graphlearn/core/graph/graph_store.h"

namespace graphlearn {
namespace op {

class Operator {
public:
  Operator() : graph_store_(nullptr) {}
  virtual ~Operator() = default;

  void Set(GraphStore* graph_store) {
    graph_store_ = graph_store;
  }

  virtual Status Process(const OpRequest* req,
                         OpResponse* res) = 0;

protected:
  GraphStore* graph_store_;
};

class RemoteOperator : public Operator {
public:
  virtual ~RemoteOperator() = default;

  virtual Status Call(int32_t remote_id,
                      const OpRequest* req,
                      OpResponse* res) = 0;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_OPERATOR_H_
