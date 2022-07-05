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

#include "service/client_impl.h"

#include "include/config.h"
#include "service/call.h"
#include "service/local/event_queue.h"
#include "service/local/in_memory_channel.h"

namespace graphlearn {

class InMemoryClientImpl : public ClientImpl {
public:
  InMemoryClientImpl() {
    EventQueue<Call*>* queue = GetInMemoryEventQueue<Call*>();
    channel_ = new InMemoryChannel(queue);
  }

  virtual ~InMemoryClientImpl() {
    delete channel_;
  }

  Status RunOp(const OpRequest* request,
               OpResponse* response) override {
    StatusWrapper status;
    channel_->CallMethod(kUserDefinedOp, request, response, &status);
    return status.s_;
  }

  Status RunDag(const DagRequest* request) override {
    StatusWrapper status;
    channel_->CallMethod(kRunDag, request, nullptr, &status);
    return status.s_;
  }

  Status GetDagValues(const GetDagValuesRequest* request,
                      GetDagValuesResponse* response) override {
    StatusWrapper status;
    channel_->CallMethod(kGetDagValues, request, response, &status);
    return status.s_;
  }

  Status Stop() override {
    if (GLOBAL_FLAG(DeployMode) == kWorker) {
      StatusWrapper status;
      channel_->CallMethod(kStop, nullptr, nullptr, &status);
      return status.s_;
    }
    return Status::OK();
  }

private:
  InMemoryChannel* channel_;
};

ClientImpl* NewInMemoryClientImpl() {
  return new InMemoryClientImpl();
}

}  // namespace graphlearn
