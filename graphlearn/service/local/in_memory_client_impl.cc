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

#include "graphlearn/service/client_impl.h"

#include "graphlearn/service/call.h"
#include "graphlearn/service/local/event_queue.h"
#include "graphlearn/service/local/in_memory_channel.h"

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

  Status Stop() override {
    return Status::OK();
  }

private:
  InMemoryChannel* channel_;
};

ClientImpl* NewInMemoryClientImpl() {
  return new InMemoryClientImpl();
}

}  // namespace graphlearn
