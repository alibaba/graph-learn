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

#include "graphlearn/service/executor.h"

#include <memory>
#include <string>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/core/runner/distribute_runner.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

Executor::Executor(Env* env, GraphStore* graph_store)
    : env_(env), graph_store_(graph_store) {
  factory_ = &(op::OperatorFactory::GetInstance());
  factory_->Set(graph_store);
}

Status Executor::RunOp(const OpRequest* request, OpResponse* response) {
  std::string op_name = request->Name();
  op::Operator* op = factory_->Lookup(op_name);
  if (op == nullptr) {
    LOG(ERROR) << "No supported op: " << op_name << ", size:" << op_name.size();
    return error::InvalidArgument("No supported op: %s", op_name.c_str());
  }

  std::unique_ptr<OpRunner> runner = GetOpRunner(env_, op);
  return runner->Run(request, response);
}

}  // namespace graphlearn
