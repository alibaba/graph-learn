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

#include "service/executor.h"

#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/graph/graph_store.h"
#include "core/operator/op_factory.h"
#include "core/runner/dag_scheduler.h"
#include "core/runner/op_runner.h"
#include "platform/env.h"

namespace graphlearn {

Executor::Executor(Env* env, GraphStore* graph_store)
    : env_(env), graph_store_(graph_store) {
  factory_ = op::OpFactory::GetInstance();
  factory_->Set(graph_store);
}

Status Executor::RunOp(const OpRequest* request, OpResponse* response) {
  std::string op_name = request->Name();
  op::Operator* op = factory_->Create(op_name);
  if (op == nullptr) {
    LOG(ERROR) << "No supported op: " << op_name << ", size:" << op_name.size();
    return error::InvalidArgument("No supported op: %s", op_name.c_str());
  }

  std::unique_ptr<OpRunner> runner = GetOpRunner(env_, op);
  return runner->Run(request, response);
}

Status Executor::RunDag(const DagDef& def) {
  Dag* dag = nullptr;
  Status s = DagFactory::GetInstance()->Create(def, &dag);

  if (s.ok()) {
    LOG(INFO) << dag->DebugString();
    DagScheduler::Take(env_, dag);
  } else if (error::IsAlreadyExists(s)) {
    LOG(WARNING) << "Dag " << def.id() << " has already existed.";
    return Status::OK();
  }

  return s;
}

Status Executor::GetDagValues(const GetDagValuesRequest* request,
                              GetDagValuesResponse* response) {
  TapeStorePtr store = GetTapeStore(request->Id());
  Tape* tape = store->WaitAndPop(request->ClientId());
  response->SetIndex(tape->Id());
  response->SetEpoch(tape->Epoch());
  if (tape->IsReady()) {
    response->MoveFrom(tape);
  }
  delete tape;
  return Status::OK();
}

}  // namespace graphlearn
