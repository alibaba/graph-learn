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

#ifndef GRAPHLEARN_SERVICE_EXECUTOR_H_
#define GRAPHLEARN_SERVICE_EXECUTOR_H_

#include <memory>
#include "include/dag_request.h"
#include "include/op_request.h"
#include "include/status.h"

namespace graphlearn {

class Env;
class GraphStore;

namespace op {
class OpFactory;
}  // namespace op

class Executor {
public:
  Executor(Env* env, GraphStore* graph_store);
  ~Executor() = default;

  Status RunOp(const OpRequest* request, OpResponse* response);
  Status RunDag(const DagDef& def);
  Status GetDagValues(const GetDagValuesRequest* request,
                      GetDagValuesResponse* response);

private:
  Env*           env_;
  GraphStore*    graph_store_;
  op::OpFactory* factory_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_EXECUTOR_H_
