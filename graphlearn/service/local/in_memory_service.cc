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

#include "graphlearn/service/local/in_memory_service.h"

#include "graphlearn/common/base/log.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/call.h"
#include "graphlearn/service/executor.h"
#include "graphlearn/service/local/event_queue.h"

namespace graphlearn {

InMemoryService::InMemoryService(Env* env, Executor* executor)
    : env_(env), executor_(executor), thread_(nullptr) {
}

InMemoryService::~InMemoryService() {
  delete thread_;
}

void InMemoryService::Start() {
  thread_ = new std::thread(&InMemoryService::Monitor, this);
}

void InMemoryService::Init() {
}

void InMemoryService::Stop() {
  GetInMemoryEventQueue<Call*>()->Cancel();
  thread_->join();
}

void InMemoryService::Monitor() {
  EventQueue<Call*>* queue = GetInMemoryEventQueue<Call*>();
  Call* call = NULL;
  while (queue->Pop(&call)) {
    Closure<void>* task = NewClosure(this, &InMemoryService::Handler, call);
    env_->InterThreadPool()->AddTask(task);
  }
}

void InMemoryService::Handler(Call* call) {
  Status s;
  switch (call->method_) {
  case kUserDefinedOp:
    s = executor_->RunOp(
      static_cast<const OpRequest*>(call->req_),
      static_cast<OpResponse*>(call->res_));
    break;
  case kOtherToExtend:
    // some request may not need executor_, just like some system status.
  default:
    LOG(ERROR) << "Unsupported method: " << call->method_;
    s = error::Unimplemented("Unsupported method: %d", call->method_);
    break;
  }
  call->status_->Signal(s);
}

}  // namespace graphlearn
